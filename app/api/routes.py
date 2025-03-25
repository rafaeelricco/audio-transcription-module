import asyncio
import uuid

from typing import List
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Body,
    Query,
)
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from app.db.database import get_db
from app.model.user import User
from app.model.request import ProcessingRequest
from app.auth.utils import get_current_user

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

    asyncio.create_task(
        ProcessRunner.run_youtube_process(
            youtube_url=url, request_id=request_id, user_id=user.id
        )
    )

    return {"request_id": request_id}


@router.get("/api/status/{request_id}", tags=["api"])
async def get_status(
    request_id: str,
    include_logs: bool = Query(
        False, description="Whether to include logs in the response"
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get the status and optionally logs of a processing request"""
    request = (
        db.query(ProcessingRequest)
        .filter(
            ProcessingRequest.id == request_id,
            ProcessingRequest.user_id
            == user.id,  # Ensure user can only see their own requests
        )
        .first()
    )
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    response = {
        "status": request.status,
        "result": request.result,
        "url": request.url,
        "created_at": request.created_at.isoformat() if request.created_at else None,
    }

    # Include logs if requested
    if include_logs and request.logs:
        response["logs"] = request.logs

    return response
