import os
import threading
import time
import sys
import logging
from dotenv import load_dotenv
from pathlib import Path
from app.factory import create_app
from app.server import start_ws_server_thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

flask_env = os.environ.get("FLASK_ENV", "development")
flask_port = int(os.environ.get("FLASK_PORT", 8080))
debug_mode = os.environ.get("DEBUG", "True").lower() in ("true", "1", "t")

app = create_app(flask_env)


def start_servers():
    """Start both the WebSocket server and Flask API"""
    # Start WebSocket server in a background thread
    print("Starting WebSocket server in background thread")
    ws_thread = start_ws_server_thread()
    print(f"WebSocket server thread started with ID: {ws_thread.ident}")
    
    # Give the WebSocket server a moment to initialize
    time.sleep(1)
    
    # Start Flask app in the main thread
    print(f"Starting Flask server on port {flask_port}")
    app.run(host="0.0.0.0", port=flask_port, debug=debug_mode, threaded=True)


if __name__ == "__main__":
    start_servers()
