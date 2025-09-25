"""
PDF parsing module for extracting text, tables, and images from PDF documents.
Uses PyMuPDF for high-quality text and layout extraction.
"""

import fitz  # PyMuPDF
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from utils.validators import ContentValidator

logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """Represents a chunk of content extracted from a PDF."""
    content: str
    content_type: str  # 'text', 'table', 'image'
    page_number: int
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PDFParser:
    """Main PDF parsing class for extracting structured content."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_pdf(self, file_path: str) -> List[PDFChunk]:
        """
        Parse a PDF file and extract chunks of content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List[PDFChunk]: List of extracted content chunks
        """
        try:
            doc = fitz.open(file_path)
            chunks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_chunks = self._parse_page(page, page_num + 1)
                chunks.extend(page_chunks)
            
            doc.close()
            self.logger.info(f"Successfully parsed PDF: {file_path}, extracted {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise
    
    def _parse_page(self, page: fitz.Page, page_number: int) -> List[PDFChunk]:
        """
        Parse a single page and extract different types of content.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number (1-indexed)
            
        Returns:
            List[PDFChunk]: List of chunks from this page
        """
        chunks = []
        
        # Extract text blocks with layout information
        text_blocks = page.get_text("dict")
        chunks.extend(self._extract_text_chunks(text_blocks, page_number))
        
        # Extract tables
        tables = page.find_tables()
        chunks.extend(self._extract_table_chunks(tables, page_number))
        
        # Extract images (metadata only, not actual image data)
        images = page.get_images()
        chunks.extend(self._extract_image_chunks(images, page_number))
        
        return chunks
    
    def _extract_text_chunks(self, text_blocks: Dict, page_number: int) -> List[PDFChunk]:
        """
        Extract text chunks from page text blocks.
        
        Args:
            text_blocks: Text blocks dictionary from PyMuPDF
            page_number: Page number
            
        Returns:
            List[PDFChunk]: List of text chunks
        """
        chunks = []
        current_section = None
        
        for block in text_blocks.get("blocks", []):
            if "lines" not in block:
                continue
            
            # Extract text from lines
            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        block_text += span["text"] + " "
            
            block_text = block_text.strip()
            if not block_text:
                continue
            
            # Clean and validate text
            block_text = ContentValidator.sanitize_text(block_text)
            if not ContentValidator.validate_placeholder_text(block_text):
                continue
            
            # Detect section headers (heuristic: short lines with specific formatting)
            if self._is_section_header(block_text, block):
                current_section = block_text
                continue
            
            # Create chunk
            chunk = PDFChunk(
                content=block_text,
                content_type="text",
                page_number=page_number,
                section_title=current_section,
                metadata={
                    "bbox": block.get("bbox"),
                    "font_info": self._extract_font_info(block),
                    "block_type": "text"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_table_chunks(self, tables: List, page_number: int) -> List[PDFChunk]:
        """
        Extract table chunks from page tables.
        
        Args:
            tables: List of table objects from PyMuPDF
            page_number: Page number
            
        Returns:
            List[PDFChunk]: List of table chunks
        """
        chunks = []
        
        for i, table in enumerate(tables):
            try:
                # Extract table data
                table_data = table.extract()
                
                if not table_data or len(table_data) < 2:
                    continue
                
                # Convert table to readable text format
                table_text = self._format_table_text(table_data)
                
                if not table_text:
                    continue
                
                # Clean and validate text
                table_text = ContentValidator.sanitize_text(table_text)
                if not ContentValidator.validate_placeholder_text(table_text):
                    continue
                
                chunk = PDFChunk(
                    content=table_text,
                    content_type="table",
                    page_number=page_number,
                    metadata={
                        "table_index": i,
                        "rows": len(table_data),
                        "columns": len(table_data[0]) if table_data else 0,
                        "bbox": table.bbox,
                        "block_type": "table"
                    }
                )
                chunks.append(chunk)
                
            except Exception as e:
                self.logger.warning(f"Error extracting table {i} from page {page_number}: {str(e)}")
                continue
        
        return chunks
    
    def _extract_image_chunks(self, images: List, page_number: int) -> List[PDFChunk]:
        """
        Extract image metadata chunks.
        
        Args:
            images: List of image objects from PyMuPDF
            page_number: Page number
            
        Returns:
            List[PDFChunk]: List of image metadata chunks
        """
        chunks = []
        
        for i, image in enumerate(images):
            try:
                # Get image metadata
                xref = image[0]
                image_info = {
                    "image_index": i,
                    "xref": xref,
                    "page_number": page_number,
                    "block_type": "image"
                }
                
                # Create a descriptive chunk for the image
                content = f"[Image {i+1} on page {page_number}]"
                
                chunk = PDFChunk(
                    content=content,
                    content_type="image",
                    page_number=page_number,
                    metadata=image_info
                )
                chunks.append(chunk)
                
            except Exception as e:
                self.logger.warning(f"Error processing image {i} from page {page_number}: {str(e)}")
                continue
        
        return chunks
    
    def _is_section_header(self, text: str, block: Dict) -> bool:
        """
        Heuristic to detect if a text block is a section header.
        
        Args:
            text: Text content
            block: Block metadata from PyMuPDF
            
        Returns:
            bool: True if likely a section header
        """
        # Check text length (headers are usually short)
        if len(text) > 100:
            return False
        
        # Check if text is in all caps or title case
        if text.isupper() or text.istitle():
            return True
        
        # Check for common header patterns
        header_patterns = [
            "section", "chapter", "part", "appendix",
            "introduction", "conclusion", "summary",
            "overview", "background", "methodology"
        ]
        
        text_lower = text.lower()
        for pattern in header_patterns:
            if pattern in text_lower:
                return True
        
        return False
    
    def _extract_font_info(self, block: Dict) -> Dict[str, Any]:
        """
        Extract font information from a text block.
        
        Args:
            block: Block metadata from PyMuPDF
            
        Returns:
            Dict: Font information
        """
        font_info = {
            "fonts": set(),
            "sizes": set(),
            "styles": set()
        }
        
        for line in block.get("lines", []):
            for span in line["spans"]:
                font_info["fonts"].add(span.get("font", "unknown"))
                font_info["sizes"].add(span.get("size", 0))
                font_info["styles"].add(span.get("flags", 0))
        
        # Convert sets to lists for JSON serialization
        font_info["fonts"] = list(font_info["fonts"])
        font_info["sizes"] = list(font_info["sizes"])
        font_info["styles"] = list(font_info["styles"])
        
        return font_info
    
    def _format_table_text(self, table_data: List[List[str]]) -> str:
        """
        Format table data as readable text.
        
        Args:
            table_data: Raw table data from PyMuPDF
            
        Returns:
            str: Formatted table text
        """
        if not table_data:
            return ""
        
        formatted_rows = []
        
        for row in table_data:
            # Clean and join cells
            clean_row = []
            for cell in row:
                if cell:
                    clean_cell = ContentValidator.sanitize_text(str(cell))
                    clean_row.append(clean_cell)
                else:
                    clean_row.append("")
            
            # Join cells with tab separator
            formatted_rows.append("\t".join(clean_row))
        
        return "\n".join(formatted_rows)
    
    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict: Document metadata
        """
        try:
            doc = fitz.open(file_path)
            metadata = doc.metadata
            
            # Add additional metadata
            additional_info = {
                "page_count": len(doc),
                "file_size": Path(file_path).stat().st_size,
                "creation_date": metadata.get("creationDate"),
                "modification_date": metadata.get("modDate"),
                "creator": metadata.get("creator"),
                "producer": metadata.get("producer"),
                "title": metadata.get("title"),
                "author": metadata.get("author"),
                "subject": metadata.get("subject"),
                "keywords": metadata.get("keywords")
            }
            
            doc.close()
            return additional_info
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return {}
    
    def validate_pdf(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate that a file is a valid PDF.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            doc = fitz.open(file_path)
            
            # Basic validation
            if len(doc) == 0:
                doc.close()
                return False, "PDF has no pages"
            
            # Try to read first page
            first_page = doc[0]
            first_page.get_text()
            
            doc.close()
            return True, ""
            
        except Exception as e:
            return False, f"Invalid PDF file: {str(e)}"


# Utility functions
def parse_pdf_file(file_path: str) -> List[PDFChunk]:
    """
    Convenience function to parse a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List[PDFChunk]: List of extracted chunks
    """
    parser = PDFParser()
    return parser.parse_pdf(file_path)


def validate_pdf_file(file_path: str) -> Tuple[bool, str]:
    """
    Convenience function to validate a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    parser = PDFParser()
    return parser.validate_pdf(file_path)
