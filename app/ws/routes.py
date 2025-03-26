"""WebSocket routes for the Audio-to-Text application using picows library.

This module implements WebSocket endpoints for real-time communication, including:
1. Status updates for processing requests
2. Echo functionality (example)
3. Chat functionality (example)
"""

import asyncio
import json
from typing import Dict, Callable, Optional, List, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime

from picows import (
    ws_create_server,
    WSFrame,
    WSTransport,
    WSListener,
    WSMsgType,
    WSUpgradeRequest,
    WSCloseCode,
)

from app.db.database import get_db
from app.model.user import User
from app.model.request import ProcessingRequest
from app.utils.auth import verify_token

# Create a router to include in the FastAPI app
router = APIRouter()


# Active connections manager
class ConnectionManager:
    def __init__(self):
        # Map of user_id -> {request_id -> transport}
        self.active_connections: Dict[str, Dict[str, WSTransport]] = {}
        # Map of transport -> {user_id, request_id}
        self.connection_info: Dict[WSTransport, Dict[str, str]] = {}

    def register_connection(
        self, user_id: str, request_id: str, transport: WSTransport
    ):
        """Register a new WebSocket connection for a user and request."""
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}

        self.active_connections[user_id][request_id] = transport
        self.connection_info[transport] = {"user_id": user_id, "request_id": request_id}

    def remove_connection(self, transport: WSTransport):
        """Remove a WebSocket connection."""
        if transport in self.connection_info:
            info = self.connection_info[transport]
            user_id = info["user_id"]
            request_id = info["request_id"]

            if user_id in self.active_connections:
                if request_id in self.active_connections[user_id]:
                    del self.active_connections[user_id][request_id]

                # Clean up user entry if no more connections
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]

            del self.connection_info[transport]

    def get_connection(self, user_id: str, request_id: str) -> Optional[WSTransport]:
        """Get a WebSocket connection for a specific user and request."""
        if (
            user_id in self.active_connections
            and request_id in self.active_connections[user_id]
        ):
            return self.active_connections[user_id][request_id]
        return None

    async def send_status_update(
        self, user_id: str, request_id: str, data: Dict[str, Any], db: Session
    ):
        """Send a status update to a connected client."""
        transport = self.get_connection(user_id, request_id)
        if transport:
            try:
                message = json.dumps(data)
                transport.send(WSMsgType.TEXT, message.encode())
            except Exception as e:
                print(f"Error sending status update: {str(e)}")
        else:
            # Update the database with the status even if no active connection
            if data.get("status") in ["completed", "failed", "error"]:
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
                        request.status = data["status"]
                        request.updated_at = datetime.utcnow()
                        db.commit()
                except Exception as e:
                    print(f"Error updating database: {str(e)}")


# Create a global connection manager instance
manager = ConnectionManager()


# Status WebSocket listener for processing request updates
class StatusListener(WSListener):
    def __init__(self, user_id: str, request_id: str, db: Session):
        self.user_id = user_id
        self.request_id = request_id
        self.db = db
        self.transport = None
        self.polling_task = None

    def on_ws_connected(self, transport: WSTransport):
        """Handle new WebSocket connection."""
        self.transport = transport
        manager.register_connection(self.user_id, self.request_id, transport)

        # Send initial status from database
        asyncio.create_task(self.send_initial_status())

        # Start polling for updates
        self.polling_task = asyncio.create_task(self.poll_status_updates())

    async def send_initial_status(self):
        """Send the initial status from the database."""
        try:
            request = (
                self.db.query(ProcessingRequest)
                .filter(
                    ProcessingRequest.id == self.request_id,
                    ProcessingRequest.user_id == self.user_id,
                )
                .first()
            )

            if request:
                status_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": request.status,
                    "message": f"Connected to status updates for request {self.request_id}",
                    "data": {
                        "url": request.url,
                        "result": request.result,
                        "logs": request.logs,
                    },
                }

                message = json.dumps(status_data)
                self.transport.send(WSMsgType.TEXT, message.encode())
            else:
                error_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "error",
                    "message": f"Request {self.request_id} not found",
                }
                message = json.dumps(error_data)
                self.transport.send(WSMsgType.TEXT, message.encode())
                # Close connection as the request doesn't exist
                self.transport.send_close(WSCloseCode.POLICY_VIOLATION)
                self.transport.disconnect()

        except Exception as e:
            error_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "error",
                "message": f"Error fetching initial status: {str(e)}",
            }
            message = json.dumps(error_data)
            self.transport.send(WSMsgType.TEXT, message.encode())

    async def poll_status_updates(self):
        """Poll for status updates from the database every 2 seconds."""
        try:
            while True:
                await asyncio.sleep(2)  # Poll every 2 seconds

                # Check if connection is still active
                if not self.transport or self.transport.is_closed():
                    break

                # Get the latest status from the database
                request = (
                    self.db.query(ProcessingRequest)
                    .filter(
                        ProcessingRequest.id == self.request_id,
                        ProcessingRequest.user_id == self.user_id,
                    )
                    .first()
                )

                if request:
                    # Only send update if status is terminal
                    if request.status in ["completed", "failed", "error"]:
                        status_data = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "status": request.status,
                            "message": f"Processing {request.status}",
                            "data": {
                                "url": request.url,
                                "result": request.result,
                                "logs": request.logs,
                            },
                        }

                        message = json.dumps(status_data)
                        self.transport.send(WSMsgType.TEXT, message.encode())

                        # If status is terminal, close connection
                        self.transport.send_close(WSCloseCode.OK)
                        self.transport.disconnect()
                        break
                else:
                    # Request no longer exists
                    error_data = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "error",
                        "message": f"Request {self.request_id} no longer exists",
                    }
                    message = json.dumps(error_data)
                    self.transport.send(WSMsgType.TEXT, message.encode())
                    self.transport.send_close(WSCloseCode.POLICY_VIOLATION)
                    self.transport.disconnect()
                    break

        except Exception as e:
            if self.transport and not self.transport.is_closed():
                error_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "error",
                    "message": f"Error polling status: {str(e)}",
                }
                message = json.dumps(error_data)
                self.transport.send(WSMsgType.TEXT, message.encode())
                self.transport.send_close(WSCloseCode.INTERNAL_ERROR)
                self.transport.disconnect()
        finally:
            # Clean up when finished
            if self.transport:
                manager.remove_connection(self.transport)

    def on_ws_frame(self, transport: WSTransport, frame: WSFrame):
        """Handle incoming WebSocket frames."""
        if frame.msg_type == WSMsgType.CLOSE:
            transport.send_close(frame.get_close_code(), frame.get_close_message())
            transport.disconnect()
        elif frame.msg_type == WSMsgType.TEXT:
            # Echo back any received messages for now
            message = frame.get_payload_as_ascii_text()
            try:
                # Parse command if any
                data = json.loads(message)
                if data.get("command") == "refresh":
                    asyncio.create_task(self.send_initial_status())
                else:
                    # Echo back the message
                    transport.send(WSMsgType.TEXT, frame.get_payload_as_bytes())
            except:
                # Just echo if not valid JSON
                transport.send(WSMsgType.TEXT, frame.get_payload_as_bytes())


# Echo WebSocket listener (example)
class EchoListener(WSListener):
    def on_ws_connected(self, transport: WSTransport):
        """Handle new WebSocket connection."""
        welcome_message = {"message": "Connected to echo server"}
        transport.send(WSMsgType.TEXT, json.dumps(welcome_message).encode())

    def on_ws_frame(self, transport: WSTransport, frame: WSFrame):
        """Handle incoming WebSocket frames."""
        if frame.msg_type == WSMsgType.CLOSE:
            transport.send_close(frame.get_close_code(), frame.get_close_message())
            transport.disconnect()
        else:
            # Echo back all messages with a timestamp
            echo_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "message": (
                    frame.get_payload_as_ascii_text()
                    if frame.msg_type == WSMsgType.TEXT
                    else "Binary data received"
                ),
            }
            transport.send(WSMsgType.TEXT, json.dumps(echo_data).encode())


# Chat WebSocket listener (example)
class ChatListener(WSListener):
    # Static registry of chat clients
    clients: List[WSTransport] = []

    def on_ws_connected(self, transport: WSTransport):
        """Handle new WebSocket connection."""
        ChatListener.clients.append(transport)
        welcome_message = {
            "type": "system",
            "message": f"Welcome! {len(ChatListener.clients)} users connected",
        }
        transport.send(WSMsgType.TEXT, json.dumps(welcome_message).encode())

        # Broadcast join message to all other clients
        join_message = {"type": "system", "message": "A new user has joined the chat"}
        self.broadcast(json.dumps(join_message).encode(), exclude=transport)

    def on_ws_disconnected(self, transport: WSTransport):
        """Handle WebSocket disconnection."""
        if transport in ChatListener.clients:
            ChatListener.clients.remove(transport)

        # Broadcast leave message to all remaining clients
        leave_message = {"type": "system", "message": "A user has left the chat"}
        self.broadcast(json.dumps(leave_message).encode())

    def on_ws_frame(self, transport: WSTransport, frame: WSFrame):
        """Handle incoming WebSocket frames."""
        if frame.msg_type == WSMsgType.CLOSE:
            transport.send_close(frame.get_close_code(), frame.get_close_message())
            transport.disconnect()
        elif frame.msg_type == WSMsgType.TEXT:
            # Broadcast the message to all clients
            try:
                message = frame.get_payload_as_ascii_text()
                data = json.loads(message)
                chat_message = {
                    "type": "chat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "sender": data.get("sender", "Anonymous"),
                    "message": data.get("message", ""),
                }
                self.broadcast(json.dumps(chat_message).encode())
            except:
                # Send error for invalid messages
                error_message = {"type": "error", "message": "Invalid message format"}
                transport.send(WSMsgType.TEXT, json.dumps(error_message).encode())

    def broadcast(self, message: bytes, exclude: WSTransport = None):
        """Broadcast a message to all connected clients."""
        for client in ChatListener.clients:
            if client != exclude and not client.is_closed():
                try:
                    client.send(WSMsgType.TEXT, message)
                except:
                    # Ignore errors, client will be removed on next interaction
                    pass


# Factory function for WebSocket endpoints
async def ws_listener_factory(
    request: WSUpgradeRequest, path_params: dict, query_params: dict, db: Session
):
    """Factory for WebSocket listeners based on the requested path."""
    path = request.path

    if path.startswith("/ws/status/"):
        # Extract request_id from path
        request_id = path_params.get("request_id")
        token = query_params.get("token")

        if not token:
            return None  # No token provided

        try:
            # Verify token and get user info
            payload = verify_token(token)
            user_id = payload.get("sub")

            if not user_id:
                return None  # Invalid token

            # Create a status listener for this user and request
            return StatusListener(user_id, request_id, db)
        except:
            return None

    elif path == "/ws/echo":
        # Echo example endpoint - no authentication required
        return EchoListener()

    elif path == "/ws/chat":
        # Chat example endpoint - no authentication required
        return ChatListener()

    # Unknown path
    return None


# Function to start the WebSocket server
async def start_ws_server(
    host: str = "127.0.0.1", port: int = 8001, db: Session = None
):
    """Start the WebSocket server on the specified host and port."""

    # Middleware for db sessions and path params
    def listener_factory(request: WSUpgradeRequest):
        # Extract path parameters
        path = request.path
        path_params = {}

        # Extract request_id from status path
        if path.startswith("/ws/status/"):
            parts = path.split("/")
            if len(parts) >= 4:
                path_params["request_id"] = parts[3]

        # Extract query parameters
        query_params = {}
        if request.query_string:
            query_string = request.query_string.decode()
            pairs = query_string.split("&")
            for pair in pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    query_params[key] = value

        # Create a database session
        db_session = next(get_db())

        return ws_listener_factory(request, path_params, query_params, db_session)

    # Create and start the WebSocket server
    server = await ws_create_server(listener_factory, host, port)
    for s in server.sockets:
        print(f"WebSocket server started on {s.getsockname()}")

    return server


# Example implementation of a WebSocket route using FastAPI's websocket route
@router.websocket("/ws/status/{request_id}")
async def websocket_status_endpoint(request_id: str, token: str = Query(None)):
    """
    This is a placeholder to document the WebSocket endpoint.

    The actual WebSocket handling is done by the picows server.
    """
    pass


@router.websocket("/ws/echo")
async def websocket_echo_endpoint():
    """
    Echo WebSocket endpoint example.

    The actual WebSocket handling is done by the picows server.
    """
    pass


@router.websocket("/ws/chat")
async def websocket_chat_endpoint():
    """
    Chat WebSocket endpoint example.

    The actual WebSocket handling is done by the picows server.
    """
    pass


# Startup event to initialize the WebSocket server
async def init_ws_server():
    """Initialize the WebSocket server when the application starts."""
    # This should be called from the app's startup event
    db = next(get_db())
    server = await start_ws_server(db=db)
    return server
