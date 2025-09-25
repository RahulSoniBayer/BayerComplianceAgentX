"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.db_models import Base


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"Test content")
        temp_file.flush()
        
        yield temp_file.name
        
        os.unlink(temp_file.name)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with pytest.MonkeyPatch().context() as m:
        m.setattr("utils.config.settings", Mock())
        yield m


@pytest.fixture
def sample_pdf_chunk():
    """Sample PDF chunk for testing."""
    from parsers.pdf_parser import PDFChunk
    
    return PDFChunk(
        content="This is a sample PDF chunk content",
        content_type="text",
        page_number=1,
        section_title="Sample Section",
        metadata={"font": "Arial", "size": 12}
    )


@pytest.fixture
def sample_placeholder():
    """Sample placeholder for testing."""
    from parsers.docx_parser import Placeholder
    
    return Placeholder(
        text="sample placeholder",
        placeholder_type="section",
        context_type="section",
        position_in_document=1,
        paragraph_index=0,
        run_index=0,
        start_pos=10,
        end_pos=30,
        metadata={"pattern": "{{.*?}}"}
    )


@pytest.fixture
def sample_llm_request():
    """Sample LLM request for testing."""
    from services.llm_service import LLMRequest
    
    return LLMRequest(
        prompt="Test prompt",
        context_type="section",
        retrieved_chunks=[
            {"content": "Retrieved chunk 1", "score": 0.9},
            {"content": "Retrieved chunk 2", "score": 0.8}
        ],
        user_context="Additional context",
        process_flow_description="Process description"
    )


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for testing."""
    from services.llm_service import LLMResponse
    
    return LLMResponse(
        content="Generated content for the placeholder",
        success=True,
        metadata={"model": "gpt-4", "usage": {"total_tokens": 150}}
    )


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing."""
    from unittest.mock import AsyncMock
    
    client = AsyncMock()
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    
    return client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    from unittest.mock import AsyncMock
    
    client = AsyncMock()
    client.embeddings.create = AsyncMock()
    
    return client


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing."""
    from unittest.mock import AsyncMock
    
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    ws.accept = AsyncMock()
    
    return ws


@pytest.fixture
def sample_file_data():
    """Sample file data for testing."""
    return {
        "filename": "test_document.pdf",
        "content_type": "application/pdf",
        "size": 1024,
        "content": b"Mock PDF content"
    }


@pytest.fixture
def sample_upload_response():
    """Sample upload response for testing."""
    return {
        "success": True,
        "message": "File uploaded successfully",
        "document_id": 1,
        "filename": "test_document.pdf",
        "status": "processing"
    }


@pytest.fixture
def sample_task_response():
    """Sample task response for testing."""
    return {
        "success": True,
        "message": "Templates uploaded successfully",
        "task_id": "test-task-id-123",
        "total_files": 2,
        "status": "processing"
    }


@pytest.fixture
def sample_websocket_message():
    """Sample WebSocket message for testing."""
    return {
        "type": "progress_update",
        "task_id": "test-task-id-123",
        "status": "processing",
        "message": "Processing file 1 of 2",
        "progress": 50,
        "timestamp": 1234567890
    }


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "MYGENASSIST_API_KEY": "test_api_key",
        "OPENAI_API_KEY": "test_openai_key",
        "SECRET_KEY": "test_secret_key",
        "ENCRYPTION_KEY": "test_encryption_key_32_bytes_long",
        "DATABASE_URL": "sqlite:///:memory:",
        "VECTOR_DB_TYPE": "faiss",
        "MAX_FILE_SIZE_MB": "50",
        "UPLOAD_DIR": "./test_uploads",
        "GENERATED_DIR": "./test_generated"
    }
    
    with pytest.MonkeyPatch().context() as m:
        for key, value in env_vars.items():
            m.setenv(key, value)
        yield env_vars


# Async fixtures
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_temp_db():
    """Create an async temporary database for testing."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    # Create async in-memory SQLite database
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as session:
        yield session
    
    await engine.dispose()
