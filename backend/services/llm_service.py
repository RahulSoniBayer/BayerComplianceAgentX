"""
LLM service for interfacing with MyGenAssist API and other language models.
Handles text generation, image processing, and response validation.
"""

import json
import logging
import base64
import io
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import httpx
from PIL import Image

from utils.config import settings
from utils.validators import ResponseValidator

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Request structure for LLM calls."""
    prompt: str
    context_type: str  # 'table' or 'section'
    retrieved_chunks: List[Dict[str, Any]]
    user_context: Optional[str] = None
    process_flow_description: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded image


@dataclass
class LLMResponse:
    """Response structure from LLM calls."""
    content: str
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MyGenAssistClient:
    """Client for MyGenAssist API integration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = settings.mygenassist_api_key
        self.base_url = settings.mygenassist_base_url
        self._client = None
    
    async def _get_client(self):
        """Get HTTP client instance."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._client
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate completion using MyGenAssist API.
        
        Args:
            request: LLM request with prompt and context
            
        Returns:
            LLMResponse: Generated response
        """
        try:
            client = await self._get_client()
            
            # Prepare messages for chat completion
            messages = self._prepare_messages(request)
            
            # Prepare request payload
            payload = {
                "model": "gpt-4",  # or appropriate model name
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000 if request.context_type == "section" else 500,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
            
            # Make API call
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            
            if response.status_code != 200:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                return LLMResponse(
                    content="",
                    success=False,
                    error_message=error_msg
                )
            
            # Parse response
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            # Validate response
            if not self._validate_response(content, request.context_type):
                return LLMResponse(
                    content="",
                    success=False,
                    error_message="Generated content failed validation"
                )
            
            return LLMResponse(
                content=content,
                success=True,
                metadata={
                    "model": response_data.get("model"),
                    "usage": response_data.get("usage", {}),
                    "finish_reason": response_data["choices"][0].get("finish_reason")
                }
            )
            
        except Exception as e:
            error_msg = f"Error calling MyGenAssist API: {str(e)}"
            self.logger.error(error_msg)
            return LLMResponse(
                content="",
                success=False,
                error_message=error_msg
            )
    
    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """
        Prepare messages for the chat completion API.
        
        Args:
            request: LLM request
            
        Returns:
            List[Dict]: Formatted messages for the API
        """
        messages = []
        
        # System message with instructions
        system_prompt = self._build_system_prompt(request.context_type)
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # User message with context
        user_prompt = self._build_user_prompt(request)
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        # Add image if present
        if request.image_data:
            # Note: MyGenAssist API might need different format for images
            # This is a placeholder for the actual implementation
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Process flow image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{request.image_data}"
                        }
                    }
                ]
            })
        
        return messages
    
    def _build_system_prompt(self, context_type: str) -> str:
        """
        Build system prompt based on context type.
        
        Args:
            context_type: Type of context ('table' or 'section')
            
        Returns:
            str: System prompt
        """
        if context_type == "table":
            return """You are an advanced AI assistant specialized in filling table placeholders in document templates.

Your task is to generate concise, structured content for table cells based on provided context.

Rules for table content:
- Generate 1-3 sentences or short bullet points
- Keep content concise and factual
- Use clear, professional language
- Avoid lengthy explanations
- Focus on key information only
- If information is missing, use reasonable domain knowledge to fill gaps

Output plain text only. No markdown, no special formatting, no symbols."""
        
        else:  # section
            return """You are an advanced AI assistant specialized in filling section placeholders in document templates.

Your task is to generate detailed, structured content for document sections based on provided context.

Rules for section content:
- Generate comprehensive, well-structured text
- Use proper paragraphs and logical flow
- Include relevant details and explanations
- Maintain professional tone and clarity
- If information is missing, use reasonable domain knowledge to fill gaps
- Ensure content is contextually appropriate

Output plain text only. No markdown, no special formatting, no symbols."""
    
    def _build_user_prompt(self, request: LLMRequest) -> str:
        """
        Build user prompt with all context information.
        
        Args:
            request: LLM request
            
        Returns:
            str: User prompt
        """
        prompt_parts = []
        
        # Placeholder information
        prompt_parts.append(f"Placeholder Text: {request.prompt}")
        
        # Retrieved chunks
        if request.retrieved_chunks:
            prompt_parts.append("\nRetrieved Context:")
            for i, chunk in enumerate(request.retrieved_chunks, 1):
                content = chunk.get("metadata", {}).get("content", "")
                if content:
                    prompt_parts.append(f"{i}. {content[:500]}...")  # Truncate long chunks
        
        # User context
        if request.user_context:
            prompt_parts.append(f"\nUser Context: {request.user_context}")
        
        # Process flow description
        if request.process_flow_description:
            prompt_parts.append(f"\nProcess Flow Summary: {request.process_flow_description}")
        
        return "\n".join(prompt_parts)
    
    def _validate_response(self, content: str, context_type: str) -> bool:
        """
        Validate LLM response content.
        
        Args:
            content: Generated content
            context_type: Type of context
            
        Returns:
            bool: True if response is valid
        """
        if context_type == "table":
            return ResponseValidator.validate_table_content(content)
        else:
            return ResponseValidator.validate_section_content(content)
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


class ProcessFlowAnalyzer:
    """Analyzer for processing flow images and generating descriptions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._client = None
    
    async def _get_client(self):
        """Get HTTP client for MyGenAssist API."""
        if self._client is None:
            self._client = MyGenAssistClient()
        return self._client
    
    async def analyze_process_flow(self, image_data: str) -> str:
        """
        Analyze a process flow image and generate a description.
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            str: Description of the process flow
        """
        try:
            # Validate image format
            if not self._validate_image_format(image_data):
                return "Invalid image format for process flow analysis."
            
            # Prepare request for image analysis
            request = LLMRequest(
                prompt="Describe the following process flow image in detail. Return plain text only.",
                context_type="section",
                retrieved_chunks=[],
                image_data=image_data
            )
            
            client = await self._get_client()
            response = await client.generate_completion(request)
            
            if response.success:
                return response.content
            else:
                self.logger.error(f"Error analyzing process flow: {response.error_message}")
                return "Unable to analyze process flow image."
            
        except Exception as e:
            self.logger.error(f"Error in process flow analysis: {str(e)}")
            return "Error analyzing process flow image."
    
    def _validate_image_format(self, image_data: str) -> bool:
        """
        Validate that the image data is in a supported format.
        
        Args:
            image_data: Base64 encoded image
            
        Returns:
            bool: True if image format is valid
        """
        try:
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Try to open with PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Check format
            supported_formats = ['JPEG', 'PNG', 'GIF', 'BMP', 'TIFF']
            return image.format in supported_formats
            
        except Exception as e:
            self.logger.warning(f"Invalid image format: {str(e)}")
            return False


class LLMService:
    """Main LLM service for handling all language model operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._client = MyGenAssistClient()
        self._process_analyzer = ProcessFlowAnalyzer()
    
    async def fill_placeholder(self, request: LLMRequest) -> LLMResponse:
        """
        Fill a placeholder using LLM with provided context.
        
        Args:
            request: LLM request with all context
            
        Returns:
            LLMResponse: Generated content for the placeholder
        """
        try:
            self.logger.info(f"Generating content for placeholder: {request.prompt[:50]}...")
            
            # Generate completion
            response = await self._client.generate_completion(request)
            
            if response.success:
                self.logger.info(f"Successfully generated content for placeholder")
            else:
                self.logger.error(f"Failed to generate content: {response.error_message}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in placeholder filling: {str(e)}")
            return LLMResponse(
                content="",
                success=False,
                error_message=f"Error generating content: {str(e)}"
            )
    
    async def analyze_process_flow_image(self, image_data: str) -> str:
        """
        Analyze a process flow image and return description.
        
        Args:
            image_data: Base64 encoded image
            
        Returns:
            str: Description of the process flow
        """
        try:
            self.logger.info("Analyzing process flow image")
            description = await self._process_analyzer.analyze_process_flow(image_data)
            self.logger.info("Successfully analyzed process flow image")
            return description
            
        except Exception as e:
            self.logger.error(f"Error analyzing process flow image: {str(e)}")
            return "Error analyzing process flow image."
    
    async def batch_fill_placeholders(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """
        Fill multiple placeholders in batch.
        
        Args:
            requests: List of LLM requests
            
        Returns:
            List[LLMResponse]: List of generated responses
        """
        responses = []
        
        for request in requests:
            try:
                response = await self.fill_placeholder(request)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Error in batch processing: {str(e)}")
                responses.append(LLMResponse(
                    content="",
                    success=False,
                    error_message=f"Batch processing error: {str(e)}"
                ))
        
        return responses
    
    async def close(self):
        """Close all clients."""
        await self._client.close()


# Convenience functions
async def fill_placeholder_with_llm(
    placeholder_text: str,
    context_type: str,
    retrieved_chunks: List[Dict[str, Any]],
    user_context: Optional[str] = None,
    process_flow_description: Optional[str] = None,
    image_data: Optional[str] = None
) -> LLMResponse:
    """
    Convenience function to fill a placeholder using LLM.
    
    Args:
        placeholder_text: Text of the placeholder
        context_type: Type of context ('table' or 'section')
        retrieved_chunks: List of retrieved context chunks
        user_context: Additional user context
        process_flow_description: Description of process flow
        image_data: Base64 encoded image data
        
    Returns:
        LLMResponse: Generated response
    """
    service = LLMService()
    request = LLMRequest(
        prompt=placeholder_text,
        context_type=context_type,
        retrieved_chunks=retrieved_chunks,
        user_context=user_context,
        process_flow_description=process_flow_description,
        image_data=image_data
    )
    
    response = await service.fill_placeholder(request)
    await service.close()
    return response


async def analyze_process_flow(image_data: str) -> str:
    """
    Convenience function to analyze a process flow image.
    
    Args:
        image_data: Base64 encoded image
        
    Returns:
        str: Description of the process flow
    """
    service = LLMService()
    description = await service.analyze_process_flow_image(image_data)
    await service.close()
    return description
