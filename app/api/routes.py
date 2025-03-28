import asyncio
import uuid
import sys

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Body, Query, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from app.db.database import get_db
from app.model.user import User
from app.model.request import ProcessingRequest
from app.utils.auth import get_current_user
from app.ws import manager
from datetime import datetime

router = APIRouter()


class ProcessRequest(BaseModel):
    urls: List[str] = Field(..., description="List of YouTube URLs to process")


class ProcessRunner:
    @staticmethod
    async def run_youtube_process(
        youtube_url: str, request_id: str, user_id: str, db: Session
    ):
        """Run the youtube transcription process and send updates via websocket"""
        db_session = db
        logs = []

        try:
            db_session.query(ProcessingRequest).filter(
                ProcessingRequest.id == request_id, ProcessingRequest.user_id == user_id
            ).update({"status": "processing", "logs": logs})
            db_session.commit()

            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "run.py",
                "--youtube",
                youtube_url,
                "--verbose",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            async for line_bytes in process.stdout:
                line = line_bytes.decode("utf-8").strip()
                if line:
                    log_entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "processing",
                        "message": line,
                    }

                    logs.append(log_entry)

                    if len(logs) % 10 == 0:
                        db_session.query(ProcessingRequest).filter(
                            ProcessingRequest.id == request_id,
                            ProcessingRequest.user_id == user_id,
                        ).update({"logs": logs})
                        db_session.commit()

                    await manager.send_status_update(
                        user_id,
                        request_id,
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "status": "processing",
                            "message": line,
                            "data": {"url": youtube_url},
                        },
                        db_session,
                    )

            return_code = await process.wait()

            if return_code == 0:
                status = "completed"
                message = "Transcription completed successfully"
            else:
                status = "failed"
                message = f"Transcription failed with code {return_code}"

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "message": message,
            }
            logs.append(log_entry)

            db_session.query(ProcessingRequest).filter(
                ProcessingRequest.id == request_id, ProcessingRequest.user_id == user_id
            ).update({"status": status, "logs": logs})
            db_session.commit()

            await manager.send_status_update(
                user_id,
                request_id,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "message": message,
                    "data": {"url": youtube_url},
                },
                db_session,
            )

        except Exception as e:
            error_message = f"Error during processing: {str(e)}"
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "message": error_message,
            }
            logs.append(log_entry)

            db_session.query(ProcessingRequest).filter(
                ProcessingRequest.id == request_id, ProcessingRequest.user_id == user_id
            ).update({"status": "failed", "result": {"error": str(e)}, "logs": logs})
            db_session.commit()

            await manager.send_status_update(
                user_id,
                request_id,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                    "message": error_message,
                    "data": {"url": youtube_url},
                },
                db_session,
            )


@router.post("/api/process", tags=["api"])
async def process_video(
    background_tasks: BackgroundTasks,
    url: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Process a YouTube URL by running the transcription command.

    This endpoint will execute the command equivalent to:
    python3 run.py --youtube "<URL>"

    The processing happens in the background and the request returns immediately.
    The logs and status can be tracked via the WebSocket endpoint at /ws/status/{request_id}
    """
    request_id = str(uuid.uuid4())
    db_request = ProcessingRequest(
        id=request_id, url=url, user_id=user.id, status="pending", logs=[]
    )
    db.add(db_request)
    db.commit()

    background_tasks.add_task(
        ProcessRunner.run_youtube_process,
        youtube_url=url,
        request_id=request_id,
        user_id=user.id,
        db=db,
    )

    return {
        "request_id": request_id,
        "status": "pending",
        "message": "Processing started. Connect to WebSocket to see live updates.",
        "websocket_url": f"/ws/status/{request_id}?token=YOUR_TOKEN_HERE",
    }


@router.get("/api/status/{request_id}", tags=["api"])
async def get_status(
    request_id: str,
    include_logs: bool = Query(
        True, description="Whether to include logs in the response"
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get the status and optionally logs of a processing request"""
    request = (
        db.query(ProcessingRequest)
        .filter(
            ProcessingRequest.id == request_id,
            ProcessingRequest.user_id == user.id,
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

    if include_logs and request.logs:
        response["logs"] = request.logs

    return response
