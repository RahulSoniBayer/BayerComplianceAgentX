"""
Unit tests for PDF parser functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from parsers.pdf_parser import PDFParser, PDFChunk, validate_pdf_file


class TestPDFParser:
    """Test cases for PDFParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PDFParser()
    
    def test_pdf_chunk_creation(self):
        """Test PDFChunk dataclass creation."""
        chunk = PDFChunk(
            content="Test content",
            content_type="text",
            page_number=1,
            section_title="Test Section",
            metadata={"test": "data"}
        )
        
        assert chunk.content == "Test content"
        assert chunk.content_type == "text"
        assert chunk.page_number == 1
        assert chunk.section_title == "Test Section"
        assert chunk.metadata == {"test": "data"}
    
    def test_is_section_header(self):
        """Test section header detection."""
        # Test short text (should be header)
        assert self.parser._is_section_header("Introduction", {})
        
        # Test long text (should not be header)
        assert not self.parser._is_section_header("This is a very long text that should not be considered a header", {})
        
        # Test title case
        assert self.parser._is_section_header("Chapter One", {})
        
        # Test all caps
        assert self.parser._is_section_header("OVERVIEW", {})
    
    def test_extract_font_info(self):
        """Test font information extraction."""
        mock_block = {
            "lines": [
                {
                    "spans": [
                        {
                            "font": "Arial",
                            "size": 12,
                            "flags": 0
                        },
                        {
                            "font": "Times",
                            "size": 14,
                            "flags": 1
                        }
                    ]
                }
            ]
        }
        
        font_info = self.parser._extract_font_info(mock_block)
        
        assert "Arial" in font_info["fonts"]
        assert "Times" in font_info["fonts"]
        assert 12 in font_info["sizes"]
        assert 14 in font_info["sizes"]
        assert 0 in font_info["styles"]
        assert 1 in font_info["styles"]
    
    def test_format_table_text(self):
        """Test table text formatting."""
        table_data = [
            ["Header 1", "Header 2"],
            ["Row 1 Col 1", "Row 1 Col 2"],
            ["Row 2 Col 1", "Row 2 Col 2"]
        ]
        
        formatted = self.parser._format_table_text(table_data)
        
        assert "Header 1" in formatted
        assert "Header 2" in formatted
        assert "Row 1 Col 1" in formatted
        assert "\t" in formatted  # Tab separator
        assert "\n" in formatted  # Newline separator
    
    def test_validate_pdf_invalid_file(self):
        """Test PDF validation with invalid file."""
        # Create a temporary text file (not a PDF)
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b"This is not a PDF file")
            temp_file.flush()
            
            is_valid, error_msg = validate_pdf_file(temp_file.name)
            
            assert not is_valid
            assert "Invalid PDF file" in error_msg
            
            os.unlink(temp_file.name)
    
    @patch('parsers.pdf_parser.fitz')
    def test_parse_pdf_mock(self, mock_fitz):
        """Test PDF parsing with mocked PyMuPDF."""
        # Mock document and pages
        mock_doc = Mock()
        mock_page = Mock()
        
        # Mock text blocks
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Sample text content",
                                    "font": "Arial",
                                    "size": 12,
                                    "flags": 0
                                }
                            ]
                        }
                    ],
                    "bbox": [0, 0, 100, 20]
                }
            ]
        }
        
        # Mock table and image methods
        mock_page.find_tables.return_value = []
        mock_page.get_images.return_value = []
        
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        
        mock_fitz.open.return_value = mock_doc
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"Mock PDF content")
            temp_file.flush()
            
            chunks = self.parser.parse_pdf(temp_file.name)
            
            assert len(chunks) > 0
            assert chunks[0].content == "Sample text content"
            assert chunks[0].content_type == "text"
            assert chunks[0].page_number == 1
            
            os.unlink(temp_file.name)
    
    @patch('parsers.pdf_parser.fitz')
    def test_get_document_metadata_mock(self, mock_fitz):
        """Test document metadata extraction with mocked PyMuPDF."""
        mock_doc = Mock()
        mock_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "subject": "Test Subject",
            "creator": "Test Creator",
            "producer": "Test Producer",
            "creationDate": "D:20231201120000",
            "modDate": "D:20231201120000"
        }
        
        mock_doc.__len__.return_value = 5
        
        mock_fitz.open.return_value = mock_doc
        
        # Mock file stats
        with patch('parsers.pdf_parser.Path') as mock_path:
            mock_path.return_value.stat.return_value.st_size = 1024
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(b"Mock PDF content")
                temp_file.flush()
                
                metadata = self.parser.get_document_metadata(temp_file.name)
                
                assert metadata["page_count"] == 5
                assert metadata["file_size"] == 1024
                assert metadata["title"] == "Test Document"
                assert metadata["author"] == "Test Author"
                
                os.unlink(temp_file.name)


class TestPDFValidation:
    """Test cases for PDF validation utilities."""
    
    def test_validate_pdf_file_not_found(self):
        """Test validation with non-existent file."""
        is_valid, error_msg = validate_pdf_file("nonexistent.pdf")
        
        assert not is_valid
        assert "Invalid PDF file" in error_msg
    
    def test_validate_pdf_empty_file(self):
        """Test validation with empty file."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"")
            temp_file.flush()
            
            is_valid, error_msg = validate_pdf_file(temp_file.name)
            
            assert not is_valid
            assert "Invalid PDF file" in error_msg
            
            os.unlink(temp_file.name)


if __name__ == "__main__":
    pytest.main([__file__])
