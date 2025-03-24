"""Server initialization and WebSocket integration for the audio-to-text application."""

import threading
import asyncio
import logging
import sys
import uvicorn
import os
from app.ws import router as ws_router
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Get application settings
settings = get_settings()


def start_websocket_server():
    """Start the WebSocket server in a separate thread"""
    try:
        uvicorn.run(
            ws_router, 
            host=settings.WEBSOCKET_HOST, 
            port=settings.WEBSOCKET_PORT,
            log_level="error"
        )
    except Exception as e:
        logging.error(f"WebSocket server error: {e}")
        # Try alternative port if the default one is in use
        try:
            uvicorn.run(
                ws_router, 
                host=settings.WEBSOCKET_HOST, 
                port=settings.WEBSOCKET_PORT + 1,
                log_level="error"
            )
        except Exception as e:
            logging.error(f"WebSocket server failed on alternate port too: {e}")


def start_ws_server_thread():
    """Start the WebSocket server in a background thread"""
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    return ws_thread
