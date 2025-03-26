import asyncio
import uuid
import subprocess
import json
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
from app.ws import manager
from datetime import datetime

router = APIRouter()


class ProcessRequest(BaseModel):
    urls: List[str] = Field(..., description="List of YouTube URLs to process")


class ProcessRunner:
    @staticmethod
    async def run_youtube_process(youtube_url: str, request_id: str, user_id: str, db: Session):
        """Run the youtube transcription process and send updates via websocket"""
        try:
            # Update status to processing
            db.query(ProcessingRequest).filter(
                ProcessingRequest.id == request_id,
                ProcessingRequest.user_id == user_id
            ).update({"status": "processing"})
            db.commit()

            # Create subprocess to run the transcription
            process = subprocess.Popen(
                ["python3", "run.py", "--youtube", youtube_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            # Read output line by line and send via websocket
            for line in process.stdout:
                if line.strip():
                    await manager.send_status_update(
                        user_id,
                        request_id,
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "status": "processing",
                            "message": line.strip(),
                            "data": {"url": youtube_url}
                        }
                    )

            # Wait for process to complete
            return_code = process.wait()

            # Update final status
            if return_code == 0:
                status = "completed"
                message = "Transcription completed successfully"
            else:
                status = "failed"
                message = f"Transcription failed with code {return_code}"

            db.query(ProcessingRequest).filter(
                ProcessingRequest.id == request_id,
                ProcessingRequest.user_id == user_id
            ).update({"status": status})
            db.commit()

            await manager.send_status_update(
                user_id,
                request_id,
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": status,
                    "message": message,
                    "data": {"url": youtube_url}
                }
            )

        except Exception as e:
            db.query(ProcessingRequest).filter(
                ProcessingRequest.id == request_id,
                ProcessingRequest.user_id == user_id
            ).update({"status": "failed", "result": {"error": str(e)}})
            db.commit()

            await manager.send_status_update(
                user_id,
                request_id,
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "error",
                    "message": f"Error during processing: {str(e)}",
                    "data": {"url": youtube_url}
                }
            )


@router.post("/api/process", tags=["api"])
async def process_video(
    url: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Process a YouTube URL by running the transcription command.

    This endpoint will execute the command equivalent to:
    python3 run.py --youtube "<URL>"

    The logs and status can be tracked via the WebSocket endpoint at /ws/status/{request_id}
    """
    request_id = str(uuid.uuid4())
    db_request = ProcessingRequest(
        id=request_id, url=url, user_id=user.id, status="pending", logs=[]
    )
    db.add(db_request)
    db.commit()

    asyncio.create_task(
        ProcessRunner.run_youtube_process(
            youtube_url=url, request_id=request_id, user_id=user.id, db=db
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
