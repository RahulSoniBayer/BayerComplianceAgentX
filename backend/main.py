"""
Main FastAPI application for the Bayer Compliance Agent.
Integrates all API routes, WebSocket connections, and middleware.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.pdf_routes import router as pdf_router
from api.template_routes import router as template_router
from api.websocket_routes import router as websocket_router, cleanup_websocket_connections
from utils.config import settings
from models.db_models import create_tables, get_database_url
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting Bayer Compliance Agent API")
    
    try:
        # Initialize database
        engine = create_engine(get_database_url())
        create_tables(engine)
        logger.info("Database initialized successfully")
        
        # Ensure directories exist
        import os
        os.makedirs(settings.upload_dir, exist_ok=True)
        os.makedirs(settings.generated_dir, exist_ok=True)
        os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
        logger.info("Directories created successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    
    # Shutdown
    logger.info("Shutting down Bayer Compliance Agent API")
    
    try:
        # Clean up WebSocket connections
        await cleanup_websocket_connections()
        logger.info("WebSocket connections cleaned up")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title="Bayer Compliance Agent API",
    description="AI-assisted document automation platform for filling templates with retrieved content",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(
    pdf_router,
    prefix="/api/pdf",
    tags=["PDF Management"]
)

app.include_router(
    template_router,
    prefix="/api/template",
    tags=["Template Processing"]
)

app.include_router(
    websocket_router,
    prefix="/api",
    tags=["WebSocket"]
)


# Global exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler for HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": f"HTTP {exc.status_code}",
            "message": exc.detail
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API status.
    
    Returns:
        JSONResponse: Health status
    """
    try:
        # Check database connection
        engine = create_engine(get_database_url())
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "status": "healthy",
                "message": "API is running normally",
                "version": "1.0.0"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "status": "unhealthy",
                "message": f"API is experiencing issues: {str(e)}",
                "version": "1.0.0"
            }
        )


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        JSONResponse: API information
    """
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Bayer Compliance Agent API",
            "version": "1.0.0",
            "description": "AI-assisted document automation platform",
            "endpoints": {
                "health": "/health",
                "pdf_upload": "/api/pdf/upload",
                "pdf_list": "/api/pdf/list",
                "template_upload": "/api/template/upload",
                "websocket": "/api/ws/progress/{task_id}",
                "documentation": "/docs"
            }
        }
    )


# API status endpoint
@app.get("/api/status")
async def api_status():
    """
    Get detailed API status and configuration.
    
    Returns:
        JSONResponse: API status and configuration
    """
    try:
        # Get vector database stats
        from services.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        db_stats = await embedding_service.get_database_stats()
        
        # Get file system status
        import os
        upload_dir_exists = os.path.exists(settings.upload_dir)
        generated_dir_exists = os.path.exists(settings.generated_dir)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "api_status": "active",
                "version": "1.0.0",
                "configuration": {
                    "vector_db_type": settings.vector_db_type,
                    "max_file_size_mb": settings.max_file_size_mb,
                    "allowed_file_types": settings.allowed_file_types,
                    "upload_dir": settings.upload_dir,
                    "generated_dir": settings.generated_dir
                },
                "vector_database": db_stats,
                "file_system": {
                    "upload_dir_exists": upload_dir_exists,
                    "generated_dir_exists": generated_dir_exists
                },
                "features": {
                    "pdf_processing": True,
                    "template_filling": True,
                    "real_time_progress": True,
                    "websocket_support": True,
                    "batch_processing": True
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting API status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to get API status",
                "message": str(e)
            }
        )


# Development server runner
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting development server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )
