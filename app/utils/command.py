import asyncio
import json
import os
import sys
import subprocess
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.db.database import SessionLocal
from app.model.request import ProcessingRequest
from app.ws import manager as ws_manager

class ProcessRunner:
    """
    Utility class to run external commands and capture their output.
    Specifically designed for running the YouTube transcription process.
    """

    @staticmethod
    async def run_youtube_process(youtube_url: str, request_id: str, user_id: str) -> None:
        """
        Run the YouTube audio extraction and transcription process.
        
        Args:
            youtube_url: URL of the YouTube video to process
            request_id: ID of the request in the database
            user_id: ID of the user who initiated the request
        """
        db = SessionLocal()
        try:
            # Create command
            cmd = [sys.executable, "run.py", "--youtube", youtube_url]
            
            # Update status to processing
            db.query(ProcessingRequest)\
                .filter(ProcessingRequest.id == request_id)\
                .update({
                    "status": "processing", 
                    "logs": [{"timestamp": datetime.utcnow().isoformat(), 
                             "message": f"Starting transcription of {youtube_url}"}]
                })
            db.commit()
            
            # Send initial status update via WebSocket
            await ws_manager.send_status_update(
                user_id=user_id,
                request_id=request_id,
                status_update={
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "processing",
                    "message": f"Starting transcription of {youtube_url}"
                }
            )
            
            # Create process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )
            
            # Process output in real-time
            current_logs = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first().logs or []
            
            # Handle stdout
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                    
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": line.strip(),
                    "type": "stdout"
                }
                
                # Add to logs
                current_logs.append(log_entry)
                
                # Update database
                db.query(ProcessingRequest)\
                    .filter(ProcessingRequest.id == request_id)\
                    .update({"logs": current_logs})
                db.commit()
                
                # Send via WebSocket
                await ws_manager.send_status_update(
                    user_id=user_id,
                    request_id=request_id,
                    status_update={
                        "timestamp": log_entry["timestamp"],
                        "status": "processing",
                        "message": log_entry["message"]
                    }
                )
                
            # Handle stderr
            stderr_data = await process.stderr.read()
            if stderr_data:
                for line in stderr_data.splitlines():
                    log_entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "message": line.strip(),
                        "type": "stderr"
                    }
                    
                    # Add to logs
                    current_logs.append(log_entry)
                
                # Update database with all stderr logs
                db.query(ProcessingRequest)\
                    .filter(ProcessingRequest.id == request_id)\
                    .update({"logs": current_logs})
                db.commit()
                
                # Send errors via WebSocket
                for log_entry in [l for l in current_logs if l.get("type") == "stderr"]:
                    await ws_manager.send_status_update(
                        user_id=user_id,
                        request_id=request_id,
                        status_update={
                            "timestamp": log_entry["timestamp"],
                            "status": "processing",
                            "message": log_entry["message"],
                            "data": {"error": True}
                        }
                    )
            
            # Wait for process to complete and get exit code
            await process.wait()
            exit_code = process.returncode
            
            final_status = "completed" if exit_code == 0 else "failed"
            final_message = (
                f"Transcription completed successfully" 
                if exit_code == 0 else 
                f"Transcription failed with exit code {exit_code}"
            )
            
            # Update final status
            result = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first().result
            
            final_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "message": final_message,
                "type": "system"
            }
            current_logs.append(final_log)
            
            db.query(ProcessingRequest)\
                .filter(ProcessingRequest.id == request_id)\
                .update({
                    "status": final_status,
                    "logs": current_logs
                })
            db.commit()
            
            # Send final status via WebSocket
            await ws_manager.send_status_update(
                user_id=user_id,
                request_id=request_id,
                status_update={
                    "timestamp": final_log["timestamp"],
                    "status": final_status,
                    "message": final_message,
                    "data": {"result": result} if result else {}
                }
            )
            
        except Exception as e:
            error_message = f"Error running transcription: {str(e)}"
            
            # Log the error
            error_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "message": error_message,
                "type": "error"
            }
            
            # Get current logs
            request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
            current_logs = request.logs if request and request.logs else []
            current_logs.append(error_log)
            
            # Update database
            db.query(ProcessingRequest)\
                .filter(ProcessingRequest.id == request_id)\
                .update({
                    "status": "failed",
                    "logs": current_logs
                })
            db.commit()
            
            # Send error via WebSocket
            await ws_manager.send_status_update(
                user_id=user_id,
                request_id=request_id,
                status_update={
                    "timestamp": error_log["timestamp"],
                    "status": "failed",
                    "message": error_message,
                    "data": {"error": True}
                }
            )
            
        finally:
            db.close()
