import os
import sys
import logging
import uvicorn
from dotenv import load_dotenv
from pathlib import Path
from app.server import start_ws_server_thread
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Load environment variables
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


def start_servers():
    """Start both the WebSocket server and FastAPI application"""
    # Start the FastAPI application using Uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "error",
    )


if __name__ == "__main__":
    start_servers()
