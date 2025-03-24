import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConnectionManager:
    """WebSocket connection manager"""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and store a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        await websocket.send_text("Connection established. Listen for messages.")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific client"""
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients"""
        for connection in self.active_connections:
            if connection.client_state != WebSocketState.DISCONNECTED:
                await connection.send_text(message)


# Create a connection manager instance
manager = ConnectionManager()


# This will be used by the main FastAPI app
router = FastAPI()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to handle client connections"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive and echo back the message
            message = await websocket.receive_text()
            await manager.send_personal_message(message, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    # For testing the WebSocket server independently
    import uvicorn
    
    websocket_port = int(os.environ.get("WEBSOCKET_PORT", 9090))
    host = os.environ.get("WEBSOCKET_HOST", "127.0.0.1")
    
    uvicorn.run(router, host=host, port=websocket_port)
