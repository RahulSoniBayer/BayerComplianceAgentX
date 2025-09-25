"""
PDF upload and management routes.
Handles PDF file uploads, parsing, embedding generation, and document management.
"""

import os
import logging
import asyncio
from typing import List, Optional
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


from models.db_models import PDFDocument, DocumentChunk, create_tables
from parsers.pdf_parser import PDFParser, validate_pdf_file
from services.embedding_service import EmbeddingService
from utils.config import settings
from utils.validators import FileUploadValidator, PDFUploadRequest

logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they don't exist
create_tables(engine)

router = APIRouter()


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF file for processing and embedding generation.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded PDF file
        db: Database session
        
    Returns:
        JSONResponse: Upload result with document ID
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Validate file type and size
        file_validator = FileUploadValidator()
        if not file_validator.validate_file_type(file.filename, file.content_type):
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
        
        # Read file content
        content = await file.read()
        if not file_validator.validate_file_size(len(content)):
            raise HTTPException(status_code=400, detail="File size exceeds maximum allowed size.")
        
        # Sanitize filename
        sanitized_filename = file_validator.sanitize_filename(file.filename)
        
        # Validate PDF format - use simple header validation first
        if not content.startswith(b'%PDF-'):
            raise HTTPException(status_code=400, detail="Invalid PDF file: File does not appear to be a PDF")
        
        # Additional validation with PyMuPDF for more thorough checking
        is_valid, error_msg = validate_pdf_file_from_content(content)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {error_msg}")
        
        # Save file
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Create database record
        db_document = PDFDocument(
            filename=sanitized_filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size=len(content),
            processing_status="pending",
            embedding_status="pending"
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Start background processing
        background_tasks.add_task(
            process_pdf_background,
            db_document.id,
            str(file_path),
            sanitized_filename
        )
        
        logger.info(f"PDF uploaded successfully: {file.filename} (ID: {db_document.id})")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "PDF uploaded successfully",
                "document_id": db_document.id,
                "filename": sanitized_filename,
                "status": "processing"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


async def process_pdf_background(document_id: int, file_path: str, filename: str):
    """
    Background task to process PDF and generate embeddings.
    
    Args:
        document_id: Database document ID
        file_path: Path to the PDF file
        filename: Filename
    """
    db = SessionLocal()
    try:
        logger.info(f"Starting background processing for document {document_id}")
        
        # Update status to processing
        db_document = db.query(PDFDocument).filter(PDFDocument.id == document_id).first()
        if not db_document:
            logger.error(f"Document {document_id} not found")
            return
        
        db_document.processing_status = "processing"
        db.commit()
        
        # Parse PDF
        parser = PDFParser()
        chunks = parser.parse_pdf(file_path)
        
        if not chunks:
            db_document.processing_status = "failed"
            db_document.embedding_status = "failed"
            db.commit()
            logger.error(f"No chunks extracted from PDF {document_id}")
            return
        
        # Generate embeddings
        embedding_service = EmbeddingService()
        embedding_results = await embedding_service.chunk_and_embed(chunks, document_id)
        
        if not embedding_results:
            db_document.processing_status = "failed"
            db_document.embedding_status = "failed"
            db.commit()
            logger.error(f"No embeddings generated for PDF {document_id}")
            return
        
        # Save chunks to database
        for i, chunk in enumerate(chunks):
            db_chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=i,
                content=chunk.content,
                content_type=chunk.content_type,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                metadata=chunk.metadata,
                vector_id=f"{document_id}_{i}"
            )
            db.add(db_chunk)
        
        # Update document status
        db_document.processing_status = "completed"
        db_document.embedding_status = "completed"
        db_document.chunk_count = len(chunks)
        
        # Get document metadata
        doc_metadata = parser.get_document_metadata(file_path)
        db_document.metadata = doc_metadata
        
        db.commit()
        
        logger.info(f"Successfully processed PDF {document_id}: {len(chunks)} chunks, {len(embedding_results)} embeddings")
        
    except Exception as e:
        logger.error(f"Error in background processing for document {document_id}: {str(e)}")
        
        # Update status to failed
        try:
            db_document = db.query(PDFDocument).filter(PDFDocument.id == document_id).first()
            if db_document:
                db_document.processing_status = "failed"
                db_document.embedding_status = "failed"
                db.commit()
        except Exception as db_error:
            logger.error(f"Error updating document status: {str(db_error)}")
    
    finally:
        db.close()


def validate_pdf_file_from_content(content: bytes) -> tuple[bool, str]:
    """
    Validate PDF file from content bytes.
    
    Args:
        content: PDF file content as bytes
        
    Returns:
        tuple: (is_valid, error_message)
    """
    temp_file_path = None
    doc = None
    
    try:
        import tempfile
        import fitz
        
        # Write content to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name
            
        # Validate using PyMuPDF
        doc = fitz.open(temp_file_path)
        
        if len(doc) == 0:
            return False, "PDF has no pages"
        
        # Try to read first page to validate it's readable
        first_page = doc[0]
        text_content = first_page.get_text()
        
        # Basic validation - PDF should have some content or be readable
        if not text_content.strip() and len(doc) == 1:
            # Single page with no text - might be image-only PDF, which is valid
            pass
            
        return True, ""
        
    except Exception as e:
        return False, f"Invalid PDF file: {str(e)}"
        
    finally:
        # Clean up resources
        try:
            if doc:
                doc.close()
        except:
            pass
            
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception as cleanup_error:
            # Log cleanup error but don't fail validation
            logger.warning(f"Could not delete temp file {temp_file_path}: {str(cleanup_error)}")


@router.get("/list")
async def list_pdfs(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List uploaded PDF documents.
    
    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        status: Filter by processing status
        db: Database session
        
    Returns:
        JSONResponse: List of PDF documents
    """
    try:
        query = db.query(PDFDocument)
        
        if status:
            query = query.filter(PDFDocument.processing_status == status)
        
        documents = query.offset(skip).limit(limit).all()
        
        result = []
        for doc in documents:
            result.append({
                "id": doc.id,
                "filename": doc.filename,
                "original_filename": doc.original_filename,
                "file_size": doc.file_size,
                "upload_date": doc.upload_date.isoformat(),
                "processing_status": doc.processing_status,
                "embedding_status": doc.embedding_status,
                "chunk_count": doc.chunk_count,
                "metadata": doc.metadata
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "documents": result,
                "total": len(result)
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing PDFs: {str(e)}")


@router.get("/{document_id}/status")
async def get_pdf_status(document_id: int, db: Session = Depends(get_db)):
    """
    Get processing status for a PDF document.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        JSONResponse: Document status
    """
    try:
        document = db.query(PDFDocument).filter(PDFDocument.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "document_id": document.id,
                "filename": document.filename,
                "processing_status": document.processing_status,
                "embedding_status": document.embedding_status,
                "chunk_count": document.chunk_count,
                "upload_date": document.upload_date.isoformat(),
                "updated_at": document.updated_at.isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting PDF status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting PDF status: {str(e)}")


@router.get("/{document_id}/chunks")
async def get_pdf_chunks(
    document_id: int,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get chunks for a PDF document.
    
    Args:
        document_id: Document ID
        skip: Number of chunks to skip
        limit: Maximum number of chunks to return
        db: Database session
        
    Returns:
        JSONResponse: List of document chunks
    """
    try:
        # Check if document exists
        document = db.query(PDFDocument).filter(PDFDocument.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).offset(skip).limit(limit).all()
        
        result = []
        for chunk in chunks:
            result.append({
                "id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "content_type": chunk.content_type,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "metadata": chunk.metadata,
                "created_at": chunk.created_at.isoformat()
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "document_id": document_id,
                "chunks": result,
                "total": len(result)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting PDF chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting PDF chunks: {str(e)}")


@router.delete("/{document_id}")
async def delete_pdf(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a PDF document and its associated data.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        JSONResponse: Deletion result
    """
    try:
        document = db.query(PDFDocument).filter(PDFDocument.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file
        try:
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
        except Exception as e:
            logger.warning(f"Could not delete file {document.file_path}: {str(e)}")
        
        # Delete chunks (cascade should handle this)
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        
        # Delete document
        db.delete(document)
        db.commit()
        
        logger.info(f"Successfully deleted PDF document {document_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Document {document_id} deleted successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")


@router.get("/stats")
async def get_pdf_stats(db: Session = Depends(get_db)):
    """
    Get statistics about uploaded PDFs.
    
    Args:
        db: Database session
        
    Returns:
        JSONResponse: PDF statistics
    """
    try:
        total_documents = db.query(PDFDocument).count()
        completed_documents = db.query(PDFDocument).filter(
            PDFDocument.processing_status == "completed"
        ).count()
        failed_documents = db.query(PDFDocument).filter(
            PDFDocument.processing_status == "failed"
        ).count()
        total_chunks = db.query(DocumentChunk).count()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": {
                    "total_documents": total_documents,
                    "completed_documents": completed_documents,
                    "failed_documents": failed_documents,
                    "processing_documents": total_documents - completed_documents - failed_documents,
                    "total_chunks": total_chunks
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting PDF stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting PDF stats: {str(e)}")
