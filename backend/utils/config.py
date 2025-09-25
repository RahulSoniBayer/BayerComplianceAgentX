"""
Configuration management for the Bayer Compliance Agent.
Handles environment variables, settings, and configuration validation.
"""

import os
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation and environment variable support."""
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./app.db", env="DATABASE_URL")
    
    # Vector Database Configuration
    vector_db_type: str = Field(default="faiss", env="VECTOR_DB_TYPE")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="bayer-compliance-docs", env="PINECONE_INDEX_NAME")
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    
    # AI/LLM Configuration
    mygenassist_api_key: str = Field(default="dummy_api_key_for_testing", env="MYGENASSIST_API_KEY")
    mygenassist_base_url: str = Field(default="https://api.mygenassist.com/v1", env="MYGENASSIST_BASE_URL")
    openai_api_key: str = Field(default="dummy_openai_key_for_testing", env="OPENAI_API_KEY")
    
    # Security
    secret_key: str = Field(default="test-secret-key-change-in-production", env="SECRET_KEY")
    encryption_key: str = Field(default="test-32-byte-encryption-key-here", env="ENCRYPTION_KEY")
    
    # Application Settings
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    allowed_file_types: List[str] = Field(default=["pdf", "docx"], env="ALLOWED_FILE_TYPES")
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    generated_dir: str = Field(default="./generated", env="GENERATED_DIR")
    
    # WebSocket Configuration
    ws_heartbeat_interval: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", env="LOG_FILE")
    
    @validator('vector_db_type')
    def validate_vector_db_type(cls, v):
        """Validate vector database type."""
        allowed_types = ['pinecone', 'weaviate', 'faiss']
        if v not in allowed_types:
            raise ValueError(f'vector_db_type must be one of {allowed_types}')
        return v
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        """Validate encryption key length."""
        if len(v.encode()) != 32:
            raise ValueError('encryption_key must be exactly 32 bytes')
        return v
    
    @validator('allowed_file_types')
    def validate_file_types(cls, v):
        """Validate allowed file types."""
        allowed_types = ['pdf', 'docx', 'doc']
        for file_type in v:
            if file_type not in allowed_types:
                raise ValueError(f'File type {file_type} not allowed. Allowed: {allowed_types}')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        settings.upload_dir,
        settings.generated_dir,
        os.path.dirname(settings.log_file)
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Initialize directories on import
ensure_directories()
