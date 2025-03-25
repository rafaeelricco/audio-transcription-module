"""Server initialization with API endpoints and WebSocket integration for the audio-to-text application."""

import logging
import sys
import uvicorn
import platform
import datetime

from fastapi import FastAPI
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

# Mount the WebSocket router directly to the main app
# This eliminates the need for a separate WebSocket server
app.mount("/ws", ws_router)


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
    """Start the API server with integrated WebSocket support"""
    try:
        # Log that we're starting with WebSocket support
        logging.info(
            f"Starting API server with WebSocket support on port {settings.APP_PORT}"
        )
        logging.info(
            f"WebSocket endpoints available at ws://{settings.APP_HOST}:{settings.APP_PORT}/ws/..."
        )

        uvicorn.run(
            app,
            host=settings.APP_HOST if hasattr(settings, "APP_HOST") else "0.0.0.0",
            port=settings.APP_PORT if hasattr(settings, "APP_PORT") else 8000,
            log_level="info",
        )
    except Exception as e:
        logging.error(f"API server error: {e}")
