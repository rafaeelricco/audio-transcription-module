"""Server initialization with API endpoints and WebSocket integration for the audio-to-text application."""

import threading
import asyncio
import logging
import sys
import uvicorn
import os
import platform
import datetime
from fastapi import FastAPI, APIRouter
from app.ws import router as ws_router
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

settings = get_settings()

app = FastAPI(
    title="Audio-to-Text API",
    description="An API for processing YouTube videos and converting audio to text",
    version="1.0.0",
)


def start_websocket_server():
    """Start the WebSocket server in a separate thread"""
    try:
        uvicorn.run(
            ws_router,
            host=settings.WS_HOST,
            port=settings.WS_PORT,
            log_level="error",
        )
    except Exception as e:
        logging.error(f"WebSocket server error: {e}")
        try:
            uvicorn.run(
                ws_router,
                host=settings.WS_HOST,
                port=settings.WS_PORT + 1,
                log_level="error",
            )
        except Exception as e:
            logging.error(f"WebSocket server failed on alternate port too: {e}")


def start_ws_server_thread():
    """Start the WebSocket server in a background thread"""
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    return ws_thread


@app.get("/")
def index():
    """Root endpoint that returns information about the API.

    Returns:
        Dictionary with API information and status
    """
    api_info = {
        "name": "Audio-to-Text API",
        "version": "1.0.0",
        "description": "An API for processing YouTube videos and converting audio to text",
        "endpoints": {
            "/": "This information",
            "/process": "Process YouTube URLs and convert audio to text (POST)",
        },
        "status": "online",
        "server_time": datetime.datetime.now().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "node": platform.node(),
        },
    }

    return api_info


def start_api_server():
    """Start the API server"""
    try:
        uvicorn.run(
            app,
            host=settings.APP_HOST if hasattr(settings, "APP_HOST") else "0.0.0.0",
            port=settings.APP_PORT if hasattr(settings, "APP_PORT") else 8000,
            log_level="info",
        )
    except Exception as e:
        logging.error(f"API server error: {e}")
