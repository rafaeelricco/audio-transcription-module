import os
import asyncio
import datetime
import uuid

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Security, status, Body, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from app.db.database import get_db
from app.model.user import User
from app.model.request import ProcessingRequest
from app.auth.utils import get_current_user
from app.video.yt_downloader import YouTubeDownloader
from app.utils.logger import Logger
from app.utils.functions import ensure_dir
from app.utils.command import ProcessRunner
from app.ai.transcription import process_text
from app.ws import manager as ws_manager

router = APIRouter()


class ProcessRequest(BaseModel):
    urls: List[str] = Field(..., description="List of YouTube URLs to process")


@router.post("/api/process", tags=["api"])
async def process_video(
    url: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Process a YouTube URL by running the transcription command.
    
    This endpoint will execute the command equivalent to:
    python3 run.py --youtube "<URL>"
    
    The logs and status can be tracked via the WebSocket endpoint at /api/status/{request_id}
    """
    request_id = str(uuid.uuid4())
    db_request = ProcessingRequest(
        id=request_id, url=url, user_id=user.id, status="pending", logs=[]
    )
    db.add(db_request)
    db.commit()

    # Start background processing using our ProcessRunner
    asyncio.create_task(
        ProcessRunner.run_youtube_process(
            youtube_url=url,
            request_id=request_id,
            user_id=user.id
        )
    )

    return {"request_id": request_id}


@router.get("/api/status/{request_id}", tags=["api"])
async def get_status(
    request_id: str, 
    include_logs: bool = Query(False, description="Whether to include logs in the response"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get the status and optionally logs of a processing request"""
    request = (
        db.query(ProcessingRequest).filter(
            ProcessingRequest.id == request_id,
            ProcessingRequest.user_id == user.id  # Ensure user can only see their own requests
        ).first()
    )
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    response = {
        "status": request.status, 
        "result": request.result, 
        "url": request.url,
        "created_at": request.created_at.isoformat() if request.created_at else None
    }
    
    # Include logs if requested
    if include_logs and request.logs:
        response["logs"] = request.logs
        
    return response


# Test Endpoints
@router.get("/test/health", tags=["test"])
async def health_check():
    """Test endpoint to check if the API is alive"""
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0.0",
    }


@router.get("/test/echo", tags=["test"])
async def echo(message: str = "Hello, World!"):
    """Test endpoint that echoes back the provided message"""
    return {"message": message, "timestamp": datetime.datetime.now().isoformat()}


class MockProcessRequest(BaseModel):
    urls: List[str] = Field(..., description="List of YouTube URLs to mock process")


@router.post("/test/mock-process", tags=["test"])
async def mock_process(request: MockProcessRequest):
    """Test endpoint that simulates processing without authentication"""
    request_id = str(uuid.uuid4())
    return {
        "request_id": request_id,
        "urls": request.urls,
        "status": "processing",
        "message": "Mock processing started",
    }


@router.get("/test/mock-status/{request_id}", tags=["test"])
async def mock_status(request_id: str):
    """Test endpoint that simulates checking status without database"""
    # Simulate status based on the first character of the request_id
    status_options = ["processing", "completed", "failed"]
    status_index = int(request_id[0], 16) % 3

    return {
        "request_id": request_id,
        "status": status_options[status_index],
        "progress": min(100, int(request_id[-2:], 16) % 101),
        "timestamp": datetime.datetime.now().isoformat(),
    }


@router.get("/test/auth", tags=["test"])
async def test_auth(user: User = Depends(get_current_user)):
    """Test endpoint to verify authentication is working"""
    return {"message": f"Hello, {user.name}!", "user_id": user.id, "email": user.email}
