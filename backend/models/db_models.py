"""
Database models for the Bayer Compliance Agent.
Defines SQLAlchemy models for storing document metadata, embeddings, and processing tasks.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class PDFDocument(Base):
    """Model for storing PDF document metadata."""
    
    __tablename__ = "pdf_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    upload_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    embedding_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    chunk_count = Column(Integer, default=0)
    doc_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Model for storing document chunks and their embeddings."""
    
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("pdf_documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=False)  # text, table, image
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(255), nullable=True)
    doc_metadata = Column(JSON, default=dict)
    embedding_vector = Column(Text, nullable=True)  # JSON string of embedding
    vector_id = Column(String(255), nullable=True, index=True)  # ID in vector database
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("PDFDocument", back_populates="chunks")


class TemplateTask(Base):
    """Model for tracking template processing tasks."""
    
    __tablename__ = "template_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed, cancelled
    total_files = Column(Integer, default=0)
    completed_files = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)
    user_context = Column(Text, nullable=True)
    process_flow_description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    files = relationship("TemplateFile", back_populates="task", cascade="all, delete-orphan")


class TemplateFile(Base):
    """Model for tracking individual template files in a processing task."""
    
    __tablename__ = "template_files"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("template_tasks.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    generated_file_path = Column(String(500), nullable=True)
    placeholder_count = Column(Integer, default=0)
    processing_started = Column(DateTime(timezone=True), nullable=True)
    processing_completed = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    task = relationship("TemplateTask", back_populates="files")
    placeholders = relationship("TemplatePlaceholder", back_populates="file", cascade="all, delete-orphan")


class TemplatePlaceholder(Base):
    """Model for tracking placeholders found in template files."""
    
    __tablename__ = "template_placeholders"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("template_files.id"), nullable=False, index=True)
    placeholder_text = Column(Text, nullable=False)
    placeholder_type = Column(String(50), nullable=False)  # table, section, inline
    context_type = Column(String(50), nullable=False)  # table, section
    position_in_document = Column(Integer, nullable=False)
    retrieved_chunk_ids = Column(Text, nullable=True)  # JSON string of chunk IDs
    generated_content = Column(Text, nullable=True)
    generation_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    file = relationship("TemplateFile", back_populates="placeholders")


class ProcessingLog(Base):
    """Model for storing processing logs and audit trails."""
    
    __tablename__ = "processing_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), nullable=True, index=True)
    file_id = Column(Integer, nullable=True, index=True)
    log_level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserSession(Base):
    """Model for tracking user sessions and WebSocket connections."""
    
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    task_id = Column(String(255), nullable=True, index=True)
    is_active = Column(Boolean, default=True)
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class VectorIndex(Base):
    """Model for tracking vector database index information."""
    
    __tablename__ = "vector_indexes"
    
    id = Column(Integer, primary_key=True, index=True)
    index_name = Column(String(255), unique=True, nullable=False, index=True)
    vector_db_type = Column(String(50), nullable=False)
    total_vectors = Column(Integer, default=0)
    last_updated = Column(DateTime(timezone=True), server_default=func.now())
    doc_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Database utility functions
def get_database_url() -> str:
    """Get database URL from configuration."""
    from utils.config import settings
    return settings.database_url


def create_tables(engine):
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def drop_tables(engine):
    """Drop all database tables."""
    Base.metadata.drop_all(bind=engine)
