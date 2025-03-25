import os
import json
import logging
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState
from jose import jwt, JWTError

from app.auth.config import get_auth_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
auth_settings = get_auth_settings()


class ProcessingStatus:
    """Status tracking for processing requests"""

    def __init__(self):
        # Map of request_id -> status updates
        self.statuses: Dict[str, List[Dict]] = {}
        
    def add_status(self, request_id: str, status_update: Dict):
        """Add a status update for a request"""
        if request_id not in self.statuses:
            self.statuses[request_id] = []
        
        self.statuses[request_id].append(status_update)
        
    def get_status_updates(self, request_id: str) -> List[Dict]:
        """Get all status updates for a request"""
        return self.statuses.get(request_id, [])
    
    def clear_status(self, request_id: str):
        """Clear status updates for a request"""
        if request_id in self.statuses:
            del self.statuses[request_id]


class ConnectionManager:
    """WebSocket connection manager for user-specific connections"""

    def __init__(self):
        # Map of user_id -> list of WebSocket connections
        self.user_connections: Dict[str, List[WebSocket]] = {}
        # Map of WebSocket -> user_id for quick lookups
        self.connection_users: Dict[WebSocket, str] = {}
        # Processing status tracker
        self.status_tracker = ProcessingStatus()

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept and store a new WebSocket connection for a specific user"""
        await websocket.accept()
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
            
        self.user_connections[user_id].append(websocket)
        self.connection_users[websocket] = user_id
        
        await websocket.send_json({
            "type": "connection_established",
            "user_id": user_id,
            "message": "Connection established. Listening for status updates."
        })

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.connection_users:
            user_id = self.connection_users[websocket]
            
            if user_id in self.user_connections:
                if websocket in self.user_connections[user_id]:
                    self.user_connections[user_id].remove(websocket)
                
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            del self.connection_users[websocket]

    async def send_status_update(self, user_id: str, request_id: str, status_update: Dict):
        """Send a status update to all connections for a specific user"""
        # Add status to tracker
        self.status_tracker.add_status(request_id, status_update)
        
        # Prepare the message
        message = {
            "type": "status_update",
            "request_id": request_id,
            "timestamp": status_update.get("timestamp"),
            "status": status_update.get("status"),
            "message": status_update.get("message"),
            "data": status_update.get("data")
        }
        
        # Send to all user's connections
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                if connection.client_state != WebSocketState.DISCONNECTED:
                    try:
                        await connection.send_json(message)
                    except Exception as e:
                        logger.error(f"Error sending message: {e}")
                        await self.disconnect(connection)

    async def send_initial_status(self, websocket: WebSocket, request_id: str):
        """Send all existing status updates for a request to a new connection"""
        status_updates = self.status_tracker.get_status_updates(request_id)
        
        if status_updates:
            for update in status_updates:
                message = {
                    "type": "status_update",
                    "request_id": request_id,
                    "timestamp": update.get("timestamp"),
                    "status": update.get("status"),
                    "message": update.get("message"),
                    "data": update.get("data")
                }
                
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending initial status: {e}")
                    return


# Create a singleton instance
manager = ConnectionManager()


async def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and extract user_id"""
    try:
        payload = jwt.decode(
            token, auth_settings.SECRET_KEY, algorithms=[auth_settings.ALGORITHM]
        )
        email = payload.get("sub")
        if email is None:
            return None
        
        # In a real implementation, you would look up the user_id by email
        # For now, we'll use the email as the user_id
        return email
    except JWTError:
        return None


router = FastAPI()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Legacy WebSocket endpoint - redirects to authenticated endpoint"""
    await websocket.accept()
    await websocket.send_text("Please use the authenticated WebSocket endpoint: /ws/status/{request_id}")
    await websocket.close()


@router.websocket("/ws/status/{request_id}")
async def status_websocket_endpoint(
    websocket: WebSocket, 
    request_id: str,
    token: str = Query(None)
):
    """
    WebSocket endpoint for tracking processing status of a specific request.
    
    Args:
        websocket: The WebSocket connection
        request_id: ID of the processing request to track
        token: JWT authentication token
    """
    if not token:
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "message": "Authentication required. Please provide a valid token."
        })
        await websocket.close()
        return
    
    user_id = await verify_token(token)
    if not user_id:
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "message": "Invalid authentication token."
        })
        await websocket.close()
        return
    
    await manager.connect(websocket, user_id)
    
    try:
        # Send initial status if available
        await manager.send_initial_status(websocket, request_id)
        
        # Keep the connection alive
        while True:
            # Just waiting for client messages or disconnection
            data = await websocket.receive_text()
            # We're not expecting client messages, but could handle them here
            await websocket.send_json({
                "type": "acknowledgment",
                "message": "Message received"
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    websocket_port = int(os.environ.get("WS_PORT", 9090))
    host = os.environ.get("WS_HOST", "127.0.0.1")

    uvicorn.run(router, host=host, port=websocket_port)
