"""
Unit tests for LLM service functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from services.llm_service import (
    LLMRequest, LLMResponse, MyGenAssistClient, 
    ProcessFlowAnalyzer, LLMService
)


class TestLLMRequest:
    """Test cases for LLMRequest dataclass."""
    
    def test_llm_request_creation(self):
        """Test LLMRequest creation with all fields."""
        request = LLMRequest(
            prompt="Test prompt",
            context_type="section",
            retrieved_chunks=[{"content": "test chunk"}],
            user_context="Additional context",
            process_flow_description="Process description",
            image_data="base64_image_data"
        )
        
        assert request.prompt == "Test prompt"
        assert request.context_type == "section"
        assert len(request.retrieved_chunks) == 1
        assert request.user_context == "Additional context"
        assert request.process_flow_description == "Process description"
        assert request.image_data == "base64_image_data"
    
    def test_llm_request_minimal(self):
        """Test LLMRequest creation with minimal fields."""
        request = LLMRequest(
            prompt="Test prompt",
            context_type="table",
            retrieved_chunks=[]
        )
        
        assert request.prompt == "Test prompt"
        assert request.context_type == "table"
        assert len(request.retrieved_chunks) == 0
        assert request.user_context is None
        assert request.process_flow_description is None
        assert request.image_data is None


class TestLLMResponse:
    """Test cases for LLMResponse dataclass."""
    
    def test_llm_response_success(self):
        """Test successful LLMResponse creation."""
        response = LLMResponse(
            content="Generated content",
            success=True,
            metadata={"model": "gpt-4", "usage": {"tokens": 100}}
        )
        
        assert response.content == "Generated content"
        assert response.success is True
        assert response.error_message is None
        assert response.metadata == {"model": "gpt-4", "usage": {"tokens": 100}}
    
    def test_llm_response_error(self):
        """Test error LLMResponse creation."""
        response = LLMResponse(
            content="",
            success=False,
            error_message="API error occurred"
        )
        
        assert response.content == ""
        assert response.success is False
        assert response.error_message == "API error occurred"
        assert response.metadata == {}


class TestMyGenAssistClient:
    """Test cases for MyGenAssistClient class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = MyGenAssistClient()
    
    @patch('services.llm_service.settings')
    def test_init(self, mock_settings):
        """Test client initialization."""
        mock_settings.mygenassist_api_key = "test_key"
        mock_settings.mygenassist_base_url = "https://api.test.com/v1"
        
        client = MyGenAssistClient()
        
        assert client.api_key == "test_key"
        assert client.base_url == "https://api.test.com/v1"
    
    def test_build_system_prompt_table(self):
        """Test system prompt building for table context."""
        prompt = self.client._build_system_prompt("table")
        
        assert "table placeholders" in prompt.lower()
        assert "concise" in prompt.lower()
        assert "1-3 sentences" in prompt.lower()
    
    def test_build_system_prompt_section(self):
        """Test system prompt building for section context."""
        prompt = self.client._build_system_prompt("section")
        
        assert "section placeholders" in prompt.lower()
        assert "comprehensive" in prompt.lower()
        assert "well-structured" in prompt.lower()
    
    def test_build_user_prompt(self):
        """Test user prompt building."""
        request = LLMRequest(
            prompt="Fill this placeholder",
            context_type="section",
            retrieved_chunks=[
                {"metadata": {"content": "Retrieved content 1"}},
                {"metadata": {"content": "Retrieved content 2"}}
            ],
            user_context="User provided context",
            process_flow_description="Process flow description"
        )
        
        prompt = self.client._build_user_prompt(request)
        
        assert "Fill this placeholder" in prompt
        assert "Retrieved content 1" in prompt
        assert "Retrieved content 2" in prompt
        assert "User provided context" in prompt
        assert "Process flow description" in prompt
    
    def test_prepare_messages(self):
        """Test message preparation for API."""
        request = LLMRequest(
            prompt="Test prompt",
            context_type="section",
            retrieved_chunks=[],
            user_context="Test context"
        )
        
        messages = self.client._prepare_messages(request)
        
        assert len(messages) == 2  # System and user message
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Test prompt" in messages[1]["content"]
        assert "Test context" in messages[1]["content"]
    
    def test_prepare_messages_with_image(self):
        """Test message preparation with image."""
        request = LLMRequest(
            prompt="Test prompt",
            context_type="section",
            retrieved_chunks=[],
            image_data="base64_image_data"
        )
        
        messages = self.client._prepare_messages(request)
        
        assert len(messages) == 3  # System, user, and image message
        assert messages[2]["role"] == "user"
        assert "image_url" in messages[2]["content"][1]
    
    @patch('services.llm_service.ResponseValidator')
    def test_validate_response_table(self, mock_validator):
        """Test response validation for table content."""
        mock_validator.validate_table_content.return_value = True
        
        result = self.client._validate_response("Short content", "table")
        
        assert result is True
        mock_validator.validate_table_content.assert_called_once_with("Short content")
    
    @patch('services.llm_service.ResponseValidator')
    def test_validate_response_section(self, mock_validator):
        """Test response validation for section content."""
        mock_validator.validate_section_content.return_value = True
        
        result = self.client._validate_response("Long detailed content", "section")
        
        assert result is True
        mock_validator.validate_section_content.assert_called_once_with("Long detailed content")
    
    @patch('services.llm_service.httpx')
    async def test_generate_completion_success(self, mock_httpx):
        """Test successful completion generation."""
        # Mock HTTP client
        mock_http_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated content"}}],
            "model": "gpt-4",
            "usage": {"total_tokens": 100}
        }
        mock_http_client.post.return_value = mock_response
        mock_httpx.AsyncClient.return_value = mock_http_client
        
        # Mock validation
        with patch.object(self.client, '_validate_response', return_value=True):
            request = LLMRequest(
                prompt="Test prompt",
                context_type="section",
                retrieved_chunks=[]
            )
            
            response = await self.client.generate_completion(request)
            
            assert response.success is True
            assert response.content == "Generated content"
            assert response.metadata["model"] == "gpt-4"
    
    @patch('services.llm_service.httpx')
    async def test_generate_completion_api_error(self, mock_httpx):
        """Test completion generation with API error."""
        # Mock HTTP client with error response
        mock_http_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_http_client.post.return_value = mock_response
        mock_httpx.AsyncClient.return_value = mock_http_client
        
        request = LLMRequest(
            prompt="Test prompt",
            context_type="section",
            retrieved_chunks=[]
        )
        
        response = await self.client.generate_completion(request)
        
        assert response.success is False
        assert "API request failed" in response.error_message
        assert response.content == ""


class TestProcessFlowAnalyzer:
    """Test cases for ProcessFlowAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ProcessFlowAnalyzer()
    
    @patch('services.llm_service.Image')
    @patch('services.llm_service.io')
    def test_validate_image_format_valid(self, mock_io, mock_image):
        """Test valid image format validation."""
        mock_image.open.return_value.format = "JPEG"
        
        result = self.analyzer._validate_image_format("valid_base64_data")
        
        assert result is True
    
    @patch('services.llm_service.Image')
    @patch('services.llm_service.io')
    def test_validate_image_format_invalid(self, mock_io, mock_image):
        """Test invalid image format validation."""
        mock_image.open.return_value.format = "INVALID"
        
        result = self.analyzer._validate_image_format("invalid_base64_data")
        
        assert result is False
    
    @patch('services.llm_service.base64')
    def test_validate_image_format_base64_error(self, mock_base64):
        """Test image format validation with base64 decode error."""
        mock_base64.b64decode.side_effect = Exception("Invalid base64")
        
        result = self.analyzer._validate_image_format("invalid_base64")
        
        assert result is False
    
    @patch.object(ProcessFlowAnalyzer, '_validate_image_format', return_value=True)
    @patch.object(ProcessFlowAnalyzer, '_get_client')
    async def test_analyze_process_flow_success(self, mock_get_client, mock_validate):
        """Test successful process flow analysis."""
        # Mock client and response
        mock_client = AsyncMock()
        mock_response = LLMResponse(
            content="This is a process flow diagram showing...",
            success=True
        )
        mock_client.generate_completion.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = await self.analyzer.analyze_process_flow("base64_image_data")
        
        assert result == "This is a process flow diagram showing..."
        mock_validate.assert_called_once_with("base64_image_data")
    
    @patch.object(ProcessFlowAnalyzer, '_validate_image_format', return_value=False)
    async def test_analyze_process_flow_invalid_image(self, mock_validate):
        """Test process flow analysis with invalid image."""
        result = await self.analyzer.analyze_process_flow("invalid_base64_data")
        
        assert result == "Invalid image format for process flow analysis."


class TestLLMService:
    """Test cases for LLMService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = LLMService()
    
    @patch.object(LLMService, '_client')
    async def test_fill_placeholder_success(self, mock_client):
        """Test successful placeholder filling."""
        mock_response = LLMResponse(
            content="Generated content",
            success=True
        )
        mock_client.generate_completion.return_value = mock_response
        
        request = LLMRequest(
            prompt="Test prompt",
            context_type="section",
            retrieved_chunks=[]
        )
        
        response = await self.service.fill_placeholder(request)
        
        assert response.success is True
        assert response.content == "Generated content"
        mock_client.generate_completion.assert_called_once_with(request)
    
    @patch.object(LLMService, '_client')
    async def test_fill_placeholder_error(self, mock_client):
        """Test placeholder filling with error."""
        mock_client.generate_completion.side_effect = Exception("API error")
        
        request = LLMRequest(
            prompt="Test prompt",
            context_type="section",
            retrieved_chunks=[]
        )
        
        response = await self.service.fill_placeholder(request)
        
        assert response.success is False
        assert "Error generating content" in response.error_message
    
    @patch.object(LLMService, '_process_analyzer')
    async def test_analyze_process_flow_image(self, mock_analyzer):
        """Test process flow image analysis."""
        mock_analyzer.analyze_process_flow.return_value = "Process description"
        
        result = await self.service.analyze_process_flow_image("base64_image")
        
        assert result == "Process description"
        mock_analyzer.analyze_process_flow.assert_called_once_with("base64_image")
    
    @patch.object(LLMService, 'fill_placeholder')
    async def test_batch_fill_placeholders(self, mock_fill):
        """Test batch placeholder filling."""
        mock_fill.side_effect = [
            LLMResponse(content="Content 1", success=True),
            LLMResponse(content="Content 2", success=True),
            LLMResponse(content="", success=False, error_message="Error")
        ]
        
        requests = [
            LLMRequest(prompt="Prompt 1", context_type="section", retrieved_chunks=[]),
            LLMRequest(prompt="Prompt 2", context_type="table", retrieved_chunks=[]),
            LLMRequest(prompt="Prompt 3", context_type="section", retrieved_chunks=[])
        ]
        
        responses = await self.service.batch_fill_placeholders(requests)
        
        assert len(responses) == 3
        assert responses[0].success is True
        assert responses[1].success is True
        assert responses[2].success is False
        assert mock_fill.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__])
