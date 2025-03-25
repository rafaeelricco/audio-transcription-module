"""API module for the audio-to-text application."""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["api"])

from app.api.routes import *  # noqa
