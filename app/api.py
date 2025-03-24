"""API routes for the audio-to-text application."""
from flask import Blueprint, jsonify
import platform
import datetime

api = Blueprint('api', __name__)


@api.route('/', methods=['GET'])
def index():
    """
    Root endpoint that returns information about the API.
    
    Returns:
        JSON response with API information and status
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
            "node": platform.node()
        }
    }
    
    return jsonify(api_info)
