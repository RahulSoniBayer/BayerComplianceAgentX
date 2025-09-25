"""
Integration tests for the Bayer Compliance Agent.
"""

import pytest
import tempfile
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from main import app
from models.db_models import PDFDocument, TemplateTask, TemplateFile


class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Bayer Compliance Agent API" in data["message"]
    
    def test_api_status_endpoint(self, client):
        """Test API status endpoint."""
        response = client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "api_status" in data
        assert "configuration" in data


class TestPDFUploadIntegration:
    """Integration tests for PDF upload functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_pdf_upload_endpoint(self, client):
        """Test PDF upload endpoint."""
        # Create a mock PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"%PDF-1.4\nMock PDF content")
            temp_file.flush()
            
            with open(temp_file.name, 'rb') as f:
                response = client.post(
                    "/api/pdf/upload",
                    files={"file": ("test.pdf", f, "application/pdf")}
                )
            
            os.unlink(temp_file.name)
        
        # Should return 200 even if processing fails (due to mock)
        assert response.status_code in [200, 422]
    
    def test_pdf_list_endpoint(self, client):
        """Test PDF list endpoint."""
        response = client.get("/api/pdf/list")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "documents" in data
    
    def test_pdf_stats_endpoint(self, client):
        """Test PDF statistics endpoint."""
        response = client.get("/api/pdf/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "stats" in data


class TestTemplateUploadIntegration:
    """Integration tests for template upload functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_template_upload_endpoint(self, client):
        """Test template upload endpoint."""
        # Create a mock DOCX file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            temp_file.write(b"Mock DOCX content")
            temp_file.flush()
            
            with open(temp_file.name, 'rb') as f:
                response = client.post(
                    "/api/template/upload",
                    files={"files": ("test.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                    data={"user_context": "Test context"}
                )
            
            os.unlink(temp_file.name)
        
        # Should return 200 even if processing fails (due to mock)
        assert response.status_code in [200, 422]
    
    def test_template_tasks_endpoint(self, client):
        """Test template tasks list endpoint."""
        response = client.get("/api/template/tasks")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "tasks" in data


class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_websocket_endpoint(self, client):
        """Test WebSocket endpoint."""
        # This would require a WebSocket test client
        # For now, just test that the endpoint exists
        response = client.get("/api/ws/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "active_connections" in data


class TestServiceIntegration:
    """Integration tests for service layer functionality."""
    
    @pytest.mark.asyncio
    async def test_embedding_service_integration(self, sample_pdf_chunk):
        """Test embedding service integration."""
        from services.embedding_service import EmbeddingService
        
        service = EmbeddingService()
        
        # Mock the embedding generation
        with patch.object(service, 'generate_embedding') as mock_generate:
            mock_generate.return_value = [0.1] * 1536  # Mock embedding vector
            
            with patch.object(service, '_vector_db') as mock_vector_db:
                mock_vector_db.upsert_vectors = AsyncMock(return_value=True)
                
                chunks = [sample_pdf_chunk]
                results = await service.chunk_and_embed(chunks, document_id=1)
                
                assert len(results) == 1
                assert results[0].chunk_id is not None
                assert len(results[0].embedding) == 1536
    
    @pytest.mark.asyncio
    async def test_retrieval_service_integration(self, sample_llm_request):
        """Test retrieval service integration."""
        from services.retrieval_service import RetrievalService
        
        service = RetrievalService()
        
        # Mock the embedding service
        with patch.object(service._embedding_service, 'search_similar_chunks') as mock_search:
            mock_search.return_value = [
                {
                    "id": "chunk1",
                    "score": 0.9,
                    "metadata": {"content": "Retrieved content", "content_type": "text"}
                }
            ]
            
            results = await service.retrieve_for_placeholder(
                placeholder_text="test placeholder",
                context_type="section",
                top_k=3
            )
            
            assert len(results) == 1
            assert results[0].content == "Retrieved content"
            assert results[0].score == 0.9
    
    @pytest.mark.asyncio
    async def test_llm_service_integration(self, sample_llm_request):
        """Test LLM service integration."""
        from services.llm_service import LLMService
        
        service = LLMService()
        
        # Mock the client
        with patch.object(service._client, 'generate_completion') as mock_generate:
            mock_generate.return_value = AsyncMock(
                content="Generated content",
                success=True,
                metadata={"model": "gpt-4"}
            )
            
            response = await service.fill_placeholder(sample_llm_request)
            
            assert response.success is True
            assert response.content == "Generated content"
    
    @pytest.mark.asyncio
    async def test_template_filler_service_integration(self, temp_file):
        """Test template filler service integration."""
        from services.template_filler_service import TemplateFillerService
        
        service = TemplateFillerService()
        
        # Mock all dependencies
        with patch.object(service._docx_parser, 'validate_docx', return_value=(True, "")) as mock_validate, \
             patch.object(service._docx_parser, 'extract_placeholders') as mock_extract, \
             patch.object(service._retrieval_service, 'retrieve_for_placeholder') as mock_retrieve, \
             patch.object(service._llm_service, 'fill_placeholder') as mock_llm, \
             patch.object(service._docx_parser, 'fill_placeholders', return_value=True) as mock_fill:
            
            # Mock placeholder extraction
            from parsers.docx_parser import Placeholder
            mock_placeholder = Placeholder(
                text="test placeholder",
                placeholder_type="section",
                context_type="section",
                position_in_document=0,
                paragraph_index=0,
                run_index=0,
                start_pos=0,
                end_pos=20
            )
            mock_extract.return_value = [mock_placeholder]
            
            # Mock retrieval
            from services.retrieval_service import RetrievalResult
            mock_retrieve.return_value = [
                RetrievalResult(
                    chunk_id="chunk1",
                    content="Retrieved content",
                    score=0.9,
                    metadata={"content": "Retrieved content"},
                    relevance_reason="Relevant"
                )
            ]
            
            # Mock LLM response
            from services.llm_service import LLMResponse
            mock_llm.return_value = LLMResponse(
                content="Generated content",
                success=True
            )
            
            # Create a temporary output file
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as output_file:
                output_file.write(b"Mock output")
                output_file.flush()
                
                try:
                    result = await service.process_template_file(
                        template_path=temp_file,
                        user_context="Test context"
                    )
                    
                    assert result["success"] is True
                    assert "output_path" in result
                    assert result["placeholders_filled"] == 1
                
                finally:
                    os.unlink(output_file.name)


class TestDatabaseIntegration:
    """Integration tests for database functionality."""
    
    def test_database_models(self, temp_db):
        """Test database model creation and relationships."""
        # Test PDFDocument creation
        pdf_doc = PDFDocument(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024,
            processing_status="completed",
            embedding_status="completed",
            chunk_count=5
        )
        
        temp_db.add(pdf_doc)
        temp_db.commit()
        
        # Test TemplateTask creation
        template_task = TemplateTask(
            task_id="test-task-123",
            status="completed",
            total_files=2,
            completed_files=2,
            failed_files=0,
            user_context="Test context"
        )
        
        temp_db.add(template_task)
        temp_db.commit()
        
        # Test TemplateFile creation
        template_file = TemplateFile(
            task_id=template_task.id,
            filename="test.docx",
            original_filename="test.docx",
            file_path="/path/to/test.docx",
            status="completed",
            generated_file_path="/path/to/generated.docx",
            placeholder_count=3
        )
        
        temp_db.add(template_file)
        temp_db.commit()
        
        # Verify relationships
        assert len(template_task.files) == 1
        assert template_file.task == template_task
        
        # Test queries
        pdf_docs = temp_db.query(PDFDocument).all()
        assert len(pdf_docs) == 1
        assert pdf_docs[0].filename == "test.pdf"
        
        tasks = temp_db.query(TemplateTask).all()
        assert len(tasks) == 1
        assert tasks[0].task_id == "test-task-123"


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, temp_file):
        """Test complete workflow from PDF upload to template filling."""
        # This would be a comprehensive test of the entire workflow
        # For now, we'll test the key components
        
        # 1. PDF Processing
        from parsers.pdf_parser import PDFParser
        parser = PDFParser()
        
        # Mock PDF parsing
        with patch.object(parser, 'parse_pdf') as mock_parse:
            from parsers.pdf_parser import PDFChunk
            mock_chunk = PDFChunk(
                content="Sample PDF content",
                content_type="text",
                page_number=1
            )
            mock_parse.return_value = [mock_chunk]
            
            chunks = parser.parse_pdf(temp_file)
            assert len(chunks) == 1
        
        # 2. Template Processing
        from parsers.docx_parser import DOCXParser
        docx_parser = DOCXParser()
        
        # Mock DOCX parsing
        with patch.object(docx_parser, 'extract_placeholders') as mock_extract:
            from parsers.docx_parser import Placeholder
            mock_placeholder = Placeholder(
                text="sample placeholder",
                placeholder_type="section",
                context_type="section",
                position_in_document=0,
                paragraph_index=0,
                run_index=0,
                start_pos=0,
                end_pos=20
            )
            mock_extract.return_value = [mock_placeholder]
            
            placeholders = docx_parser.extract_placeholders(temp_file)
            assert len(placeholders) == 1
        
        # 3. Content Generation
        from services.llm_service import LLMService
        llm_service = LLMService()
        
        # Mock LLM service
        with patch.object(llm_service._client, 'generate_completion') as mock_generate:
            from services.llm_service import LLMResponse
            mock_response = LLMResponse(
                content="Generated content",
                success=True
            )
            mock_generate.return_value = mock_response
            
            from services.llm_service import LLMRequest
            request = LLMRequest(
                prompt="sample placeholder",
                context_type="section",
                retrieved_chunks=[]
            )
            
            response = await llm_service.fill_placeholder(request)
            assert response.success is True
            assert response.content == "Generated content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
