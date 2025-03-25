"""
FastAPI application entry point for the audio-to-text service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
from app.config import get_settings
from dotenv import load_dotenv
from app.db.database import init_db

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

settings = get_settings()

app = FastAPI(
    title="Audio-to-Text API",
    description="An API for processing YouTube videos and converting audio to text",
    version="1.0.0",
)

app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    init_db()


from app.api import router as api_router
from app.auth.router import router as auth_router

app.include_router(api_router)
app.include_router(auth_router)
