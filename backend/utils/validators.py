"""
Validation utilities for file uploads, data processing, and security.
"""

import os
import mimetypes
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, validator
from utils.config import settings


class FileUploadValidator:
    """Validates file uploads for security and format compliance."""
    
    @staticmethod
    def validate_file_type(filename: str, content_type: Optional[str] = None) -> bool:
        """
        Validate file type based on extension and content type.
        
        Args:
            filename: Name of the uploaded file
            content_type: MIME type of the file content
            
        Returns:
            bool: True if file type is allowed
        """
        # Check file extension
        file_ext = Path(filename).suffix.lower().lstrip('.')
        if file_ext not in settings.allowed_file_types:
            return False
        
        # Validate content type if provided
        if content_type:
            expected_mime_types = {
                'pdf': 'application/pdf',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'doc': 'application/msword'
            }
            
            if file_ext in expected_mime_types:
                expected_mime = expected_mime_types[file_ext]
                if not content_type.startswith(expected_mime.split('/')[0]):
                    return False
        
        return True
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """
        Validate file size against maximum allowed size.
        
        Args:
            file_size: Size of the file in bytes
            
        Returns:
            bool: True if file size is within limits
        """
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal and other security issues.
        
        Args:
            filename: Original filename
            
        Returns:
            str: Sanitized filename
        """
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove potentially dangerous characters
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename


class ContentValidator:
    """Validates content for processing and security."""
    
    @staticmethod
    def validate_placeholder_text(text: str) -> bool:
        """
        Validate placeholder text for processing.
        
        Args:
            text: Placeholder text to validate
            
        Returns:
            bool: True if placeholder text is valid
        """
        if not text or not text.strip():
            return False
        
        # Check for minimum length
        if len(text.strip()) < 3:
            return False
        
        # Check for maximum length
        if len(text) > 10000:
            return False
        
        return True
    
    @staticmethod
    def validate_user_context(context: str) -> bool:
        """
        Validate user-provided context text.
        
        Args:
            context: User context text
            
        Returns:
            bool: True if context is valid
        """
        if not context:
            return True  # Context is optional
        
        # Check for maximum length
        if len(context) > 50000:
            return False
        
        return True
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """
        Sanitize text content to remove potentially harmful content.
        
        Args:
            text: Text to sanitize
            
        Returns:
            str: Sanitized text
        """
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()


class ProcessFlowValidator:
    """Validates process flow images and descriptions."""
    
    @staticmethod
    def validate_base64_image(base64_data: str) -> bool:
        """
        Validate base64 encoded image data.
        
        Args:
            base64_data: Base64 encoded image
            
        Returns:
            bool: True if image data is valid
        """
        if not base64_data:
            return True  # Process flow image is optional
        
        # Check for valid base64 format
        try:
            import base64
            base64.b64decode(base64_data, validate=True)
        except Exception:
            return False
        
        # Check for reasonable size (max 10MB when decoded)
        if len(base64_data) > 13333333:  # ~10MB in base64
            return False
        
        return True
    
    @staticmethod
    def validate_image_format(base64_data: str) -> bool:
        """
        Validate that base64 data represents a supported image format.
        
        Args:
            base64_data: Base64 encoded image
            
        Returns:
            bool: True if image format is supported
        """
        if not base64_data:
            return True
        
        try:
            import base64
            from PIL import Image
            import io
            
            # Decode base64
            image_data = base64.b64decode(base64_data)
            
            # Try to open with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Check if format is supported
            supported_formats = ['JPEG', 'PNG', 'GIF', 'BMP', 'TIFF']
            return image.format in supported_formats
            
        except Exception:
            return False


class ResponseValidator:
    """Validates LLM responses and generated content."""
    
    @staticmethod
    def validate_llm_response(response: str) -> bool:
        """
        Validate LLM-generated response.
        
        Args:
            response: Generated response text
            
        Returns:
            bool: True if response is valid
        """
        if not response or not response.strip():
            return False
        
        # Check for minimum length
        if len(response.strip()) < 10:
            return False
        
        # Check for maximum length
        if len(response) > 50000:
            return False
        
        # Check for suspicious content patterns
        suspicious_patterns = [
            'error occurred',
            'cannot complete',
            'insufficient data',
            'missing information',
            '[ERROR]',
            '[FAILED]'
        ]
        
        response_lower = response.lower()
        for pattern in suspicious_patterns:
            if pattern in response_lower:
                return False
        
        return True
    
    @staticmethod
    def validate_table_content(content: str) -> bool:
        """
        Validate content for table placeholders.
        
        Args:
            content: Generated table content
            
        Returns:
            bool: True if content is appropriate for tables
        """
        if not ResponseValidator.validate_llm_response(content):
            return False
        
        # Check if content is too long for table cells
        if len(content) > 2000:
            return False
        
        return True
    
    @staticmethod
    def validate_section_content(content: str) -> bool:
        """
        Validate content for section placeholders.
        
        Args:
            content: Generated section content
            
        Returns:
            bool: True if content is appropriate for sections
        """
        return ResponseValidator.validate_llm_response(content)


# Pydantic models for request validation
class PDFUploadRequest(BaseModel):
    """Request model for PDF upload validation."""
    filename: str
    content_type: Optional[str] = None
    file_size: int
    
    @validator('filename')
    def validate_filename(cls, v):
        if not FileUploadValidator.validate_file_type(v):
            raise ValueError('Invalid file type')
        return FileUploadValidator.sanitize_filename(v)
    
    @validator('file_size')
    def validate_file_size(cls, v):
        if not FileUploadValidator.validate_file_size(v):
            raise ValueError('File size exceeds maximum allowed')
        return v


class TemplateUploadRequest(BaseModel):
    """Request model for template upload validation."""
    filename: str
    content_type: Optional[str] = None
    file_size: int
    user_context: Optional[str] = None
    process_flow_image: Optional[str] = None
    
    @validator('filename')
    def validate_filename(cls, v):
        if not FileUploadValidator.validate_file_type(v):
            raise ValueError('Invalid file type')
        return FileUploadValidator.sanitize_filename(v)
    
    @validator('file_size')
    def validate_file_size(cls, v):
        if not FileUploadValidator.validate_file_size(v):
            raise ValueError('File size exceeds maximum allowed')
        return v
    
    @validator('user_context')
    def validate_user_context(cls, v):
        if v and not ContentValidator.validate_user_context(v):
            raise ValueError('Invalid user context')
        return ContentValidator.sanitize_text(v) if v else v
    
    @validator('process_flow_image')
    def validate_process_flow_image(cls, v):
        if v and not ProcessFlowValidator.validate_base64_image(v):
            raise ValueError('Invalid process flow image')
        return v
