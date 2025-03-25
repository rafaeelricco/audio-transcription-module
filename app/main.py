"""
FastAPI application entry point for the audio-to-text service.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
from dotenv import load_dotenv
from app.config import get_settings
from app.database import init_db

# Load environment variables
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

# Get application settings
settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title="Audio-to-Text API",
    description="An API for processing YouTube videos and converting audio to text",
    version="1.0.0",
)

# Add Session middleware (required for OAuth)
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# Import routers after app creation to avoid circular imports
from app.api import router as api_router
from app.auth.router import router as auth_router

# Include routers
app.include_router(api_router)
app.include_router(auth_router)
