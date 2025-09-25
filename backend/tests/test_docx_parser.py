"""
Unit tests for DOCX parser functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from parsers.docx_parser import DOCXParser, Placeholder, validate_docx_template


class TestDOCXParser:
    """Test cases for DOCXParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DOCXParser()
    
    def test_placeholder_creation(self):
        """Test Placeholder dataclass creation."""
        placeholder = Placeholder(
            text="test placeholder",
            placeholder_type="section",
            context_type="section",
            position_in_document=1,
            paragraph_index=0,
            run_index=0,
            start_pos=0,
            end_pos=20,
            metadata={"pattern": "{{.*?}}"}
        )
        
        assert placeholder.text == "test placeholder"
        assert placeholder.placeholder_type == "section"
        assert placeholder.context_type == "section"
        assert placeholder.position_in_document == 1
        assert placeholder.metadata == {"pattern": "{{.*?}}"}
    
    def test_placeholder_patterns(self):
        """Test placeholder pattern matching."""
        test_text = "This is {{placeholder1}} and [[placeholder2]] and <placeholder3> and %placeholder4%"
        
        # Test each pattern
        patterns = [
            r'{{([^}]+)}}',  # {{placeholder}}
            r'\[\[([^\]]+)\]\]',  # [[placeholder]]
            r'<([^>]+)>',  # <placeholder>
            r'%([^%]+)%',  # %placeholder%
        ]
        
        import re
        found_placeholders = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, test_text)
            for match in matches:
                found_placeholders.append(match.group(1).strip())
        
        assert "placeholder1" in found_placeholders
        assert "placeholder2" in found_placeholders
        assert "placeholder3" in found_placeholders
        assert "placeholder4" in found_placeholders
    
    def test_classify_placeholder(self):
        """Test placeholder classification."""
        # Mock paragraph
        mock_paragraph = Mock()
        mock_paragraph.style = Mock()
        mock_paragraph.style.name = "Normal"
        
        # Test table keywords
        table_type, table_context = self.parser._classify_placeholder("table data", mock_paragraph, 0)
        assert table_type == "table"
        assert table_context == "table"
        
        # Test section keywords
        section_type, section_context = self.parser._classify_placeholder("section description", mock_paragraph, 0)
        assert section_type == "section"
        assert section_context == "section"
        
        # Test long text (should be section)
        long_type, long_context = self.parser._classify_placeholder(
            "This is a very long placeholder text that should be classified as section", 
            mock_paragraph, 0
        )
        assert long_type == "section"
        assert long_context == "section"
        
        # Test short text (should be inline)
        short_type, short_context = self.parser._classify_placeholder("short", mock_paragraph, 0)
        assert short_type == "inline"
        assert short_context == "section"
    
    def test_extract_placeholders_from_paragraph(self):
        """Test placeholder extraction from paragraph."""
        # Mock paragraph with runs
        mock_run = Mock()
        mock_run.text = "This is a {{test placeholder}} in the text"
        
        mock_paragraph = Mock()
        mock_paragraph.runs = [mock_run]
        
        placeholders = self.parser._extract_placeholders_from_paragraph(mock_paragraph, 0, 0)
        
        assert len(placeholders) == 1
        assert placeholders[0].text == "test placeholder"
        assert placeholders[0].paragraph_index == 0
        assert placeholders[0].run_index == 0
    
    def test_extract_placeholders_from_table(self):
        """Test placeholder extraction from table."""
        # Mock table structure
        mock_run = Mock()
        mock_run.text = "Table cell with {{table placeholder}}"
        
        mock_paragraph = Mock()
        mock_paragraph.runs = [mock_run]
        
        mock_cell = Mock()
        mock_cell.paragraphs = [mock_paragraph]
        
        mock_row = Mock()
        mock_row.cells = [mock_cell]
        
        mock_table = Mock()
        mock_table.rows = [mock_row]
        
        placeholders = self.parser._extract_placeholders_from_table(mock_table, 0, 0)
        
        assert len(placeholders) == 1
        assert placeholders[0].text == "table placeholder"
        assert placeholders[0].placeholder_type == "table"
        assert placeholders[0].context_type == "table"
        assert placeholders[0].metadata["is_in_table"] is True
    
    @patch('parsers.docx_parser.Document')
    def test_extract_placeholders_mock(self, mock_document_class):
        """Test placeholder extraction with mocked python-docx."""
        # Mock document
        mock_doc = Mock()
        mock_document_class.return_value = mock_doc
        
        # Mock paragraph
        mock_run = Mock()
        mock_run.text = "This is {{test placeholder}}"
        
        mock_paragraph = Mock()
        mock_paragraph.runs = [mock_run]
        
        mock_doc.paragraphs = [mock_paragraph]
        mock_doc.tables = []
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            temp_file.write(b"Mock DOCX content")
            temp_file.flush()
            
            placeholders = self.parser.extract_placeholders(temp_file.name)
            
            assert len(placeholders) == 1
            assert placeholders[0].text == "test placeholder"
            
            os.unlink(temp_file.name)
    
    @patch('parsers.docx_parser.Document')
    def test_fill_placeholders_mock(self, mock_document_class):
        """Test placeholder filling with mocked python-docx."""
        # Mock document
        mock_doc = Mock()
        mock_document_class.return_value = mock_doc
        
        # Mock paragraph with placeholder
        mock_run = Mock()
        mock_run.text = "This is {{test placeholder}}"
        
        mock_paragraph = Mock()
        mock_paragraph.runs = [mock_run]
        
        mock_doc.paragraphs = [mock_paragraph]
        mock_doc.tables = []
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as input_file:
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as output_file:
                input_file.write(b"Mock DOCX content")
                input_file.flush()
                
                placeholder_fills = {
                    "test placeholder": "This is the filled content"
                }
                
                success = self.parser.fill_placeholders(
                    input_file.name,
                    placeholder_fills,
                    output_file.name
                )
                
                assert success is True
                mock_doc.save.assert_called_once_with(output_file.name)
                
                os.unlink(input_file.name)
                os.unlink(output_file.name)
    
    def test_update_paragraph_runs(self):
        """Test paragraph run updating."""
        # Mock paragraph with runs
        mock_run1 = Mock()
        mock_run1.text = "Original text"
        
        mock_run2 = Mock()
        mock_run2.text = "More text"
        
        mock_paragraph = Mock()
        mock_paragraph.runs = [mock_run1, mock_run2]
        
        run_mapping = [
            {"run": mock_run1, "start": 0, "end": 13, "original_text": "Original text"},
            {"run": mock_run2, "start": 13, "end": 22, "original_text": "More text"}
        ]
        
        original_text = "Original textMore text"
        modified_text = "Original textModified text"
        
        self.parser._update_paragraph_runs(mock_paragraph, run_mapping, original_text, modified_text)
        
        # First run should be updated with modified text
        assert mock_run1.text == modified_text
        # Second run should be cleared
        mock_run2.clear.assert_called_once()
    
    @patch('parsers.docx_parser.Document')
    def test_validate_docx_mock(self, mock_document_class):
        """Test DOCX validation with mocked python-docx."""
        # Mock valid document
        mock_doc = Mock()
        mock_doc.paragraphs = [Mock(), Mock()]  # Non-empty paragraphs
        mock_doc.tables = [Mock()]
        mock_document_class.return_value = mock_doc
        
        # Mock placeholder extraction
        with patch.object(self.parser, 'extract_placeholders') as mock_extract:
            mock_extract.return_value = [Mock()]  # Non-empty placeholders
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                temp_file.write(b"Mock DOCX content")
                temp_file.flush()
                
                is_valid, error_msg = self.parser.validate_docx(temp_file.name)
                
                assert is_valid is True
                assert error_msg == ""
                
                os.unlink(temp_file.name)
    
    @patch('parsers.docx_parser.Document')
    def test_get_document_metadata_mock(self, mock_document_class):
        """Test document metadata extraction with mocked python-docx."""
        # Mock document with metadata
        mock_doc = Mock()
        mock_doc.paragraphs = [Mock(), Mock()]
        mock_doc.tables = [Mock()]
        mock_doc.core_properties.created = "2023-12-01T12:00:00"
        mock_doc.core_properties.modified = "2023-12-01T12:00:00"
        mock_doc.core_properties.title = "Test Document"
        mock_doc.core_properties.author = "Test Author"
        mock_doc.core_properties.subject = "Test Subject"
        mock_doc.core_properties.keywords = "test, document"
        
        mock_document_class.return_value = mock_doc
        
        # Mock file stats
        with patch('parsers.docx_parser.Path') as mock_path:
            mock_path.return_value.stat.return_value.st_size = 2048
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                temp_file.write(b"Mock DOCX content")
                temp_file.flush()
                
                metadata = self.parser.get_document_metadata(temp_file.name)
                
                assert metadata["paragraph_count"] == 2
                assert metadata["table_count"] == 1
                assert metadata["file_size"] == 2048
                assert metadata["title"] == "Test Document"
                assert metadata["author"] == "Test Author"
                
                os.unlink(temp_file.name)


class TestDOCXValidation:
    """Test cases for DOCX validation utilities."""
    
    def test_validate_docx_template_not_found(self):
        """Test validation with non-existent file."""
        parser = DOCXParser()
        is_valid, error_msg = parser.validate_docx("nonexistent.docx")
        
        assert not is_valid
        assert "Invalid DOCX file" in error_msg
    
    @patch('parsers.docx_parser.Document')
    def test_validate_docx_empty_document(self, mock_document_class):
        """Test validation with empty document."""
        mock_doc = Mock()
        mock_doc.paragraphs = []
        mock_doc.tables = []
        mock_document_class.return_value = mock_doc
        
        parser = DOCXParser()
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            temp_file.write(b"Mock DOCX content")
            temp_file.flush()
            
            is_valid, error_msg = parser.validate_docx(temp_file.name)
            
            assert not is_valid
            assert "DOCX file appears to be empty" in error_msg
            
            os.unlink(temp_file.name)


if __name__ == "__main__":
    pytest.main([__file__])
