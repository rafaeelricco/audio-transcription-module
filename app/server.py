"""Server initialization and WebSocket integration for the audio-to-text application."""

import threading
import asyncio
import logging
import sys

from app.ws import main as ws_main

# Only configure logging for werkzeug and flask
logger = logging.getLogger()
# Remove any existing handlers to avoid duplicates
for handler in logger.handlers:
    logger.removeHandler(handler)

# Add specific handler for Flask/Werkzeug
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set higher log level for our application modules to suppress their logs
logging.getLogger('app').setLevel(logging.ERROR)


def start_websocket_server():
    """Start the WebSocket server in a separate thread"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ws_main())
    except Exception:
        # Silently handle exceptions
        pass


def start_ws_server_thread():
    """Start the WebSocket server in a background thread"""
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    return ws_thread
