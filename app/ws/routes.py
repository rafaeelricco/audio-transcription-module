import asyncio
import json
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect, HTTPException, APIRouter
from app.db.database import get_db
from app.model.request import ProcessingRequest
from app.utils.auth import decode_token


class ConnectionManager:
    def __init__(self):
        # connections[request_id] = {"user_id": user_id, "websocket": websocket}
        self.connections: Dict[str, Dict[str, Any]] = {}
        # tasks for background polling
        self.background_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, request_id: str, user_id: str):
        await websocket.accept()
        if request_id in self.connections:
            # Close the existing connection if there is one
            try:
                await self.connections[request_id]["websocket"].close()
            except:
                pass
        
        self.connections[request_id] = {"user_id": user_id, "websocket": websocket}
        
        # Create a new background task for this connection
        task = asyncio.create_task(self.poll_status(request_id, user_id))
        self.background_tasks[request_id] = task
    
    def disconnect(self, request_id: str):
        if request_id in self.connections:
            # Cancel the background task
            if request_id in self.background_tasks:
                self.background_tasks[request_id].cancel()
                del self.background_tasks[request_id]
            
            del self.connections[request_id]
    
    async def send_status(self, request_id: str, status_data: Dict[str, Any]):
        if request_id in self.connections:
            try:
                await self.connections[request_id]["websocket"].send_json(status_data)
            except Exception as e:
                print(f"Error sending status: {e}")
                self.disconnect(request_id)
    
    async def poll_status(self, request_id: str, user_id: str):
        """Poll the database for status updates and send them to the client"""
        db = next(get_db())
        try:
            # Initial status fetch
            request = db.query(ProcessingRequest).filter(
                ProcessingRequest.id == request_id,
                ProcessingRequest.user_id == user_id
            ).first()
            
            if not request:
                await self.send_status(request_id, {
                    "status": "error",
                    "message": "Request not found"
                })
                self.disconnect(request_id)
                return
            
            # Send initial status
            last_status = request.status
            await self.send_status(request_id, {
                "status": request.status,
                "logs": request.logs,
                "created_at": request.created_at.isoformat() if request.created_at else None,
                "updated_at": request.updated_at.isoformat() if request.updated_at else None,
                "result": json.loads(request.result) if request.result else None
            })
            
            # If terminal state, don't poll
            terminal_states = ["completed", "failed", "error"]
            if request.status in terminal_states:
                self.disconnect(request_id)
                return
            
            # Poll for updates every 2 seconds
            while True:
                await asyncio.sleep(2)  # Poll every 2 seconds
                
                # Make sure connection still exists
                if request_id not in self.connections:
                    return
                
                # Refresh db session to avoid stale data
                db.close()
                db = next(get_db())
                
                # Check for updates
                request = db.query(ProcessingRequest).filter(
                    ProcessingRequest.id == request_id,
                    ProcessingRequest.user_id == user_id
                ).first()
                
                if not request:
                    await self.send_status(request_id, {
                        "status": "error",
                        "message": "Request not found"
                    })
                    self.disconnect(request_id)
                    return
                
                # Only send update if status changed
                if request.status != last_status:
                    last_status = request.status
                    await self.send_status(request_id, {
                        "status": request.status,
                        "logs": request.logs,
                        "created_at": request.created_at.isoformat() if request.created_at else None,
                        "updated_at": request.updated_at.isoformat() if request.updated_at else None,
                        "result": json.loads(request.result) if request.result else None
                    })
                
                # If terminal state, stop polling
                if request.status in terminal_states:
                    self.disconnect(request_id)
                    return
        
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            pass
        except Exception as e:
            print(f"Error in poll_status: {e}")
        finally:
            db.close()


async def get_current_user_from_ws(websocket: WebSocket):
    """Extract and validate JWT token from WebSocket query parameters"""
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        raise HTTPException(status_code=401, detail="Missing authentication token")
    
    try:
        payload = decode_token(token)
        user_id = payload.get("user_id")
        if not user_id:
            await websocket.close(code=1008, reason="Invalid token payload")
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return user_id
    except Exception as e:
        await websocket.close(code=1008, reason="Invalid authentication token")
        raise HTTPException(status_code=401, detail=f"Invalid authentication token: {str(e)}")


# Initialize the connection manager
manager = ConnectionManager()

# Create router for WebSocket endpoints
router = APIRouter()

@router.websocket("/status/{request_id}")
async def websocket_status(websocket: WebSocket, request_id: str):
    """WebSocket endpoint for getting real-time status updates on processing requests
    
    The client should connect to this endpoint with a JWT token as a query parameter:
    ws://domain/ws/status/{request_id}?token=<JWT_TOKEN>
    
    The connection will stay open and send status updates whenever the status changes.
    """
    try:
        # Authenticate the user from the token in query parameters
        user_id = await get_current_user_from_ws(websocket)
        
        # Connect the WebSocket client
        await manager.connect(websocket, request_id, user_id)
        
        # Keep the connection open until closed by client or server
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(request_id)
    except HTTPException as e:
        # Authentication failed - connection should already be closed
        print(f"WebSocket connection failed: {e.detail}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason="Server error")
        except:
            pass
        manager.disconnect(request_id)
