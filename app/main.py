"""
FastAPI application entry point for the audio-to-text service.
"""
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.api import router as api_router
from app.config import get_settings

# Load environment variables
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

# Database setup
Base = declarative_base()
settings = get_settings()

# Create database engine based on settings
engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency for database session management.
    Creates a new session for each request and closes it when done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create FastAPI application
app = FastAPI(
    title="Audio-to-Text API",
    description="An API for processing YouTube videos and converting audio to text",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include API router
app.include_router(api_router)

# Create all database tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
