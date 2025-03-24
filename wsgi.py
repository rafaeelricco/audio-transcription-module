import os
import threading
import asyncio
from dotenv import load_dotenv
from app.factory import create_app
from app.server import main as websocket_server

# Load environment variables
env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env):
    load_dotenv(env)

# Get environment configuration
FLASK_ENV = os.getenv('FLASK_ENV', 'production')
DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
FLASK_PORT = int(os.getenv('FLASK_PORT', 8080))

# Create Flask app with the appropriate environment
app = create_app(FLASK_ENV)

# Function to run the websocket server
def run_websocket_server():
    asyncio.run(websocket_server())

if __name__ == "__main__":
    # Start websocket server in a separate thread
    websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
    websocket_thread.start()
    print("WebSocket server started in background thread")
    
    # Start Flask app with debug mode and port from environment variables
    # Avoid using port 5000 (AirPlay Receiver on macOS) 
    app.run(debug=DEBUG, host='0.0.0.0', port=FLASK_PORT)
