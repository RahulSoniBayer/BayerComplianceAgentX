"""
WebSocket routes for real-time progress updates and communication.
Handles WebSocket connections for template processing progress.
"""

import json
import logging
import asyncio
from typing import Dict, Any, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect, Depends
from fastapi.routing import APIRouter

from models.db_models import UserSession
from utils.config import settings

logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_tasks: Dict[WebSocket, asyncio.Task] = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket, task_id: str):
        """Accept a WebSocket connection and add it to the task group."""
        await websocket.accept()
        
        if task_id not in self.active_connections:
            self.active_connections[task_id] = set()
        
        self.active_connections[task_id].add(websocket)
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat(websocket))
        self.connection_tasks[websocket] = heartbeat_task
        
        self.logger.info(f"WebSocket connected for task {task_id}")
    
    async def disconnect(self, websocket: WebSocket, task_id: str):
        """Remove a WebSocket connection and clean up."""
        if task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)
            
            # Clean up empty task groups
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        
        # Cancel heartbeat task
        if websocket in self.connection_tasks:
            task = self.connection_tasks[websocket]
            task.cancel()
            del self.connection_tasks[websocket]
        
        self.logger.info(f"WebSocket disconnected for task {task_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            self.logger.error(f"Error sending personal message: {str(e)}")
    
    async def broadcast_to_task(self, task_id: str, message: Dict[str, Any]):
        """Broadcast a message to all connections for a specific task."""
        if task_id not in self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for websocket in self.active_connections[task_id]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                self.logger.error(f"Error broadcasting to task {task_id}: {str(e)}")
                disconnected.add(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected:
            await self.disconnect(websocket, task_id)
    
    async def _heartbeat(self, websocket: WebSocket):
        """Send periodic heartbeat messages to keep connection alive."""
        try:
            while True:
                await asyncio.sleep(settings.ws_heartbeat_interval)
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": asyncio.get_event_loop().time()
                }))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Heartbeat error: {str(e)}")


# Global connection manager
manager = ConnectionManager()

# WebSocket router
router = APIRouter()


@router.websocket("/ws/progress/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time progress updates.
    
    Args:
        websocket: WebSocket connection
        task_id: Task ID to track progress for
    """
    await manager.connect(websocket, task_id)
    
    try:
        while True:
            # Wait for client messages (ping, status requests, etc.)
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": asyncio.get_event_loop().time()}),
                    websocket
                )
            elif message.get("type") == "status_request":
                # Send current status (could be retrieved from database)
                status_message = {
                    "type": "status_update",
                    "task_id": task_id,
                    "status": "active",
                    "timestamp": asyncio.get_event_loop().time()
                }
                await manager.send_personal_message(json.dumps(status_message), websocket)
    
    except WebSocketDisconnect:
        await manager.disconnect(websocket, task_id)
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {str(e)}")
        await manager.disconnect(websocket, task_id)


# Utility functions for sending progress updates
async def send_progress_update(
    task_id: str,
    status: str,
    message: str,
    progress: int = 0,
    **kwargs
):
    """
    Send a progress update to all connected clients for a task.
    
    Args:
        task_id: Task ID
        status: Current status
        message: Status message
        progress: Progress percentage (0-100)
        **kwargs: Additional data to include
    """
    update_message = {
        "type": "progress_update",
        "task_id": task_id,
        "status": status,
        "message": message,
        "progress": progress,
        "timestamp": asyncio.get_event_loop().time(),
        **kwargs
    }
    
    await manager.broadcast_to_task(task_id, update_message)
    logger.info(f"Progress update sent for task {task_id}: {status} - {message}")


async def send_file_completion(
    task_id: str,
    file_index: int,
    filename: str,
    completed_files: int,
    total_files: int,
    download_url: Optional[str] = None
):
    """
    Send a file completion notification.
    
    Args:
        task_id: Task ID
        file_index: Index of the completed file
        filename: Name of the completed file
        completed_files: Number of files completed so far
        total_files: Total number of files
        download_url: URL to download the file
    """
    completion_message = {
        "type": "file_completed",
        "task_id": task_id,
        "file_index": file_index,
        "filename": filename,
        "completed_files": completed_files,
        "total_files": total_files,
        "download_url": download_url,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    await manager.broadcast_to_task(task_id, completion_message)
    logger.info(f"File completion sent for task {task_id}: {filename}")


async def send_task_completion(task_id: str, success: bool, message: str = "", **kwargs):
    """
    Send a task completion notification.
    
    Args:
        task_id: Task ID
        success: Whether the task completed successfully
        message: Completion message
        **kwargs: Additional data to include
    """
    completion_message = {
        "type": "task_completed",
        "task_id": task_id,
        "success": success,
        "message": message,
        "timestamp": asyncio.get_event_loop().time(),
        **kwargs
    }
    
    await manager.broadcast_to_task(task_id, completion_message)
    logger.info(f"Task completion sent for task {task_id}: {'success' if success else 'failed'}")


async def send_error_notification(task_id: str, error_message: str, error_details: Dict[str, Any] = None):
    """
    Send an error notification.
    
    Args:
        task_id: Task ID
        error_message: Error message
        error_details: Additional error details
    """
    error_message_data = {
        "type": "error",
        "task_id": task_id,
        "error": error_message,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    if error_details:
        error_message_data["details"] = error_details
    
    await manager.broadcast_to_task(task_id, error_message_data)
    logger.error(f"Error notification sent for task {task_id}: {error_message}")


# WebSocket status and management endpoints
@router.get("/ws/status")
async def get_websocket_status():
    """Get WebSocket connection status."""
    return {
        "active_connections": sum(len(connections) for connections in manager.active_connections.values()),
        "active_tasks": len(manager.active_connections),
        "connection_tasks": len(manager.connection_tasks)
    }


@router.get("/ws/tasks/{task_id}/connections")
async def get_task_connections(task_id: str):
    """Get connection count for a specific task."""
    connection_count = len(manager.active_connections.get(task_id, set()))
    return {
        "task_id": task_id,
        "connection_count": connection_count,
        "has_connections": connection_count > 0
    }


@router.post("/ws/tasks/{task_id}/broadcast")
async def broadcast_message_to_task(task_id: str, message: Dict[str, Any]):
    """Broadcast a custom message to all connections for a task."""
    await manager.broadcast_to_task(task_id, message)
    return {"message": "Broadcast sent successfully"}


# WebSocket connection cleanup
async def cleanup_websocket_connections():
    """Clean up WebSocket connections and tasks."""
    for websocket in list(manager.connection_tasks.keys()):
        task = manager.connection_tasks[websocket]
        task.cancel()
    
    manager.connection_tasks.clear()
    manager.active_connections.clear()
    logger.info("WebSocket connections cleaned up")
