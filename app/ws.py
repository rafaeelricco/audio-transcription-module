import os
import json
import logging
import asyncio
import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends
from starlette.websockets import WebSocketState
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.database import get_db
from app.model.request import ProcessingRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
auth_settings = get_settings()


class ProcessingStatus:
    """Status tracking for processing requests"""

    def __init__(self):
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
        self.user_connections: Dict[str, List[WebSocket]] = {}
        self.connection_users: Dict[WebSocket, str] = {}
        self.status_tracker = ProcessingStatus()

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept and store a new WebSocket connection for a specific user"""
        await websocket.accept()

        if user_id not in self.user_connections:
            self.user_connections[user_id] = []

        self.user_connections[user_id].append(websocket)
        self.connection_users[websocket] = user_id

        await websocket.send_json(
            {
                "type": "connection_established",
                "user_id": user_id,
                "message": "Connection established. Listening for status updates.",
            }
        )

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

    async def send_status_update(
        self, user_id: str, request_id: str, status_update: Dict, db: Session = None
    ):
        """Send a status update to all connections for a specific user and save to DB"""
        self.status_tracker.add_status(request_id, status_update)

        message = {
            "type": "status_update",
            "request_id": request_id,
            "timestamp": status_update.get("timestamp"),
            "status": status_update.get("status"),
            "message": status_update.get("message"),
            "data": status_update.get("data"),
        }
        
        # Save log to database if db session provided
        if db:
            try:
                request = (
                    db.query(ProcessingRequest)
                    .filter(
                        ProcessingRequest.id == request_id,
                        ProcessingRequest.user_id == user_id,
                    )
                    .first()
                )
                
                if request:
                    if not request.logs:
                        request.logs = []
                    
                    log_entry = {
                        "timestamp": status_update.get("timestamp"),
                        "status": status_update.get("status"),
                        "message": status_update.get("message"),
                    }
                    
                    request.logs.append(log_entry)
                    db.commit()
            except Exception as e:
                logger.error(f"Error saving log to database: {e}")

        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                if connection.client_state != WebSocketState.DISCONNECTED:
                    try:
                        await connection.send_json(message)
                    except Exception as e:
                        logger.error(f"Error sending message: {e}")
                        self.disconnect(connection)

    async def send_initial_status(
        self, websocket: WebSocket, request_id: str, user_id: str, db: Session
    ):
        """Send initial status from database and any cached status updates for a request"""
        request = (
            db.query(ProcessingRequest)
            .filter(
                ProcessingRequest.id == request_id,
                ProcessingRequest.user_id == user_id,
            )
            .first()
        )

        if request:
            initial_status = {
                "type": "initial_status",
                "request_id": request_id,
                "status": request.status,
                "result": request.result,
                "url": request.url,
                "created_at": (
                    request.created_at.isoformat() if request.created_at else None
                ),
            }

            if request.logs:
                initial_status["logs"] = request.logs

            try:
                await websocket.send_json(initial_status)
            except Exception as e:
                logger.error(f"Error sending initial database status: {e}")
                return

        status_updates = self.status_tracker.get_status_updates(request_id)

        if status_updates:
            for update in status_updates:
                message = {
                    "type": "status_update",
                    "request_id": request_id,
                    "timestamp": update.get("timestamp"),
                    "status": update.get("status"),
                    "message": update.get("message"),
                    "data": update.get("data"),
                }

                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending cached status: {e}")
                    return


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

        return email
    except JWTError as e:
        logger.error(f"JWT Error: {str(e)}")
        return None


router = FastAPI()

app = router


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Legacy WebSocket endpoint - redirects to authenticated endpoint"""
    await websocket.accept()
    await websocket.send_text(
        "Please use the authenticated WebSocket endpoint: /ws/status/{request_id}"
    )
    await websocket.close()


@router.websocket("/ws/status/{request_id}")
async def status_websocket_endpoint(
    websocket: WebSocket,
    request_id: str,
    token: str = Query(None),
    db: Session = Depends(get_db),
):
    """
    WebSocket endpoint for tracking processing status of a specific request.

    Args:
        websocket: The WebSocket connection
        request_id: ID of the processing request to track
        token: JWT authentication token
        db: Database session
    """
    await websocket.accept()
    
    if not token:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Authentication required. Please provide a valid token.",
            }
        )
        await websocket.close()
        return

    user_id = await verify_token(token)
    if not user_id:
        await websocket.send_json(
            {"type": "error", "message": "Invalid authentication token."}
        )
        await websocket.close()
        return

    # Find user by email (user_id is actually the email from JWT)
    from app.model.user import User
    user = db.query(User).filter(User.email == user_id).first()
    
    if not user:
        await websocket.send_json(
            {"type": "error", "message": "User not found."}
        )
        await websocket.close()
        return
        
    # Now use the actual user.id
    user_id = user.id
    
    request = (
        db.query(ProcessingRequest)
        .filter(
            ProcessingRequest.id == request_id,
            ProcessingRequest.user_id == user_id,
        )
        .first()
    )

    if not request:
        # Just return logs if any, even if request not found
        status_updates = manager.status_tracker.get_status_updates(request_id)
        if status_updates:
            for update in status_updates:
                message = {
                    "type": "status_update",
                    "request_id": request_id,
                    "timestamp": update.get("timestamp"),
                    "status": update.get("status"),
                    "message": update.get("message"),
                    "data": update.get("data"),
                }
                await websocket.send_json(message)
        else:
            await websocket.send_json(
                {"type": "error", "message": "Request not found or access denied."}
            )
            await websocket.close()
            return

    await manager.connect(websocket, user_id)

    try:
        await manager.send_initial_status(websocket, request_id, user_id, db)

        polling_task = asyncio.create_task(
            poll_database_for_updates(websocket, request_id, user_id, db)
        )

        while True:
            data = await websocket.receive_text()
            await websocket.send_json(
                {"type": "acknowledgment", "message": "Message received"}
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
    finally:
        if "polling_task" in locals() and not polling_task.done():
            polling_task.cancel()
            try:
                await polling_task
            except asyncio.CancelledError:
                pass


async def poll_database_for_updates(
    websocket: WebSocket, request_id: str, user_id: str, db: Session
):
    """
    Periodically poll the database for updates to the request status
    and send them to the connected client.

    Args:
        websocket: The WebSocket connection
        request_id: ID of the processing request to track
        user_id: ID of the user who owns the request
        db: Database session
    """
    last_status = None
    last_result = None
    last_logs_count = 0

    while True:
        try:
            request = (
                db.query(ProcessingRequest)
                .filter(
                    ProcessingRequest.id == request_id,
                    ProcessingRequest.user_id == user_id,
                )
                .first()
            )

            if request:
                logs_count = len(request.logs) if request.logs else 0
                
                if (request.status != last_status or 
                    request.result != last_result or 
                    logs_count != last_logs_count):
                    
                    last_status = request.status
                    last_result = request.result
                    last_logs_count = logs_count

                    update = {
                        "type": "db_status_update",
                        "request_id": request_id,
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "status": request.status,
                        "result": request.result,
                        "url": request.url,
                        "created_at": (
                            request.created_at.isoformat()
                            if request.created_at
                            else None
                        ),
                    }

                    if request.logs:
                        update["logs"] = request.logs

                    await websocket.send_json(update)

                    if request.status in ["completed", "failed", "error"]:
                        logger.info(
                            f"Request {request_id} reached terminal state: {request.status}. Stopping polling."
                        )
                        break
            else:
                logger.warning(
                    f"Request {request_id} not found in database during polling."
                )
                break

        except Exception as e:
            logger.error(f"Error polling database: {e}")
            break

        await asyncio.sleep(2)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.ws:app",
        host="0.0.0.0",
        port=8081,
        log_level="info",
    )

    websocket_port = int(os.environ.get("WS_PORT", 9090))
    host = os.environ.get("WS_HOST", "127.0.0.1")

    uvicorn.run(router, host=host, port=websocket_port)
