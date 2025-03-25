"""API routes for the audio-to-text application."""

import os
import asyncio
import datetime
import json
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Security, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.db.database import get_db
from app.model.user import User
from app.model.request import ProcessingRequest
from app.auth.utils import get_current_user
from app.video.yt_downloader import YouTubeDownloader
from app.utils.logger import Logger
from app.utils.functions import ensure_dir
from app.ai.transcription import process_text
from app.ws import manager as ws_manager

router = APIRouter()


class YouTubeURLsRequest(BaseModel):
    """Request model for YouTube URLs processing."""
    
    urls: List[str] = Field(..., description="List of YouTube URLs to process")


class ProcessingResponse(BaseModel):
    """Response model for processing requests."""
    
    request_id: str = Field(..., description="ID of the processing request")
    status: str = Field(..., description="Status of the processing request")
    message: str = Field(..., description="Processing message")


@router.post("/process", response_model=ProcessingResponse)
async def process_youtube_urls(
    request: YouTubeURLsRequest,
    current_user: User = Security(get_current_user, scopes=["requests"]),
    db: Session = Depends(get_db),
):
    """
    Process YouTube URLs to extract audio and generate transcriptions.
    
    This endpoint initiates the following workflow:
    1. Download audio from YouTube videos
    2. Transcribe the audio using Whisper
    3. Process the transcription with AI
    
    Args:
        request: Request containing YouTube URLs
        current_user: Authenticated user
        db: Database session
    
    Returns:
        ProcessingResponse: Information about the created processing request
    """
    if not request.urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No YouTube URLs provided",
        )
    
    # Create a processing request in the database
    processing_request = ProcessingRequest(
        urls=request.urls,
        user_id=current_user.id,
    )
    
    db.add(processing_request)
    db.commit()
    db.refresh(processing_request)
    
    # Start processing in the background
    asyncio.create_task(
        process_videos(
            request.urls,
            processing_request.id,
            current_user.id,
            db,
        )
    )
    
    return ProcessingResponse(
        request_id=processing_request.id,
        status="processing",
        message="Processing started. Check status endpoint for updates.",
    )


@router.get("/process/{request_id}", response_model=Dict[str, Any])
async def get_processing_status(
    request_id: str,
    current_user: User = Security(get_current_user, scopes=["requests"]),
    db: Session = Depends(get_db),
):
    """
    Get the status of a processing request.
    
    Args:
        request_id: ID of the processing request
        current_user: Authenticated user
        db: Database session
    
    Returns:
        Dict: Processing request details
    """
    processing_request = db.query(ProcessingRequest).filter(
        ProcessingRequest.id == request_id,
        ProcessingRequest.user_id == current_user.id,
    ).first()
    
    if not processing_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processing request not found",
        )
    
    return processing_request.serialize


async def process_videos(
    urls: List[str],
    request_id: str,
    user_id: str,
    db: Session,
):
    """
    Process YouTube videos in the background.
    
    This function follows the workflow:
    1. Download audio from YouTube videos
    2. Transcribe the audio
    3. Process the transcription with AI
    4. Update the processing request in the database
    5. Send status updates via WebSocket
    
    Args:
        urls: List of YouTube URLs to process
        request_id: ID of the processing request
        user_id: ID of the user who initiated the request
        db: Database session
    """
    try:
        # Helper function to send status updates
        async def send_status(status: str, message: str, data: Dict = None):
            timestamp = datetime.datetime.now().isoformat()
            status_update = {
                "timestamp": timestamp,
                "status": status,
                "message": message,
                "data": data or {}
            }
            await ws_manager.send_status_update(user_id, request_id, status_update)
        
        # Send initial status update
        await send_status(
            "started", 
            f"Started processing {len(urls)} YouTube URLs",
            {"total_urls": len(urls)}
        )
        
        Logger.log(True, f"Starting processing for request {request_id}")
        
        # Create output directories
        downloads_dir = os.path.join("downloads", user_id, request_id)
        ensure_dir(downloads_dir)
        
        transcripts_dir = os.path.join("transcripts", user_id, request_id)
        ensure_dir(transcripts_dir)
        
        # Initialize YouTube downloader
        downloader = YouTubeDownloader(output_path=downloads_dir)
        
        results = []
        processed_count = 0
        
        for index, url in enumerate(urls):
            try:
                current_url_number = index + 1
                
                # Send status update for current URL
                await send_status(
                    "processing_url",
                    f"Processing URL {current_url_number}/{len(urls)}: {url}",
                    {"url": url, "progress": f"{current_url_number}/{len(urls)}"}
                )
                
                # Step 1: Download audio from YouTube
                Logger.log(True, f"Downloading audio from {url}")
                await send_status(
                    "downloading", 
                    f"Downloading audio from YouTube: {url}", 
                    {"url": url, "step": "download"}
                )
                
                download_result = downloader.download_audio(url)
                
                if not download_result["success"]:
                    error_msg = f"Failed to download audio from {url}: {download_result.get('error')}"
                    Logger.log(False, error_msg)
                    
                    # Send failure status
                    await send_status(
                        "failed", 
                        error_msg,
                        {"url": url, "step": "download", "error": download_result.get("error")}
                    )
                    
                    results.append({
                        "url": url,
                        "status": "failed",
                        "error": download_result.get("error", "Unknown error during download"),
                    })
                    continue
                
                audio_file = download_result["file_path"]
                video_title = download_result.get("title", "Unknown")
                
                # Send download success status
                await send_status(
                    "download_complete", 
                    f"Downloaded audio: {video_title}", 
                    {"url": url, "step": "download_complete", "title": video_title, "file": audio_file}
                )
                
                # Step 2: Transcribe audio
                Logger.log(True, f"Transcribing audio from {url}")
                await send_status(
                    "transcribing", 
                    f"Transcribing audio: {video_title}", 
                    {"url": url, "step": "transcribe", "title": video_title}
                )
                
                # Prepare input for transcription
                input_path = {
                    "file_path": audio_file,
                    "file_name": os.path.basename(audio_file),
                }
                
                # Import the transcribe_audio function from run.py
                from run import transcribe_audio, save_and_process_transcript
                
                # Transcribe the audio
                transcription_result = transcribe_audio(input_path, output_path=transcripts_dir)
                
                if not transcription_result.get("success", False):
                    error_msg = f"Failed to transcribe audio from {url}: {transcription_result.get('error')}"
                    Logger.log(False, error_msg)
                    
                    # Send failure status
                    await send_status(
                        "failed", 
                        error_msg,
                        {"url": url, "step": "transcribe", "error": transcription_result.get("error")}
                    )
                    
                    results.append({
                        "url": url,
                        "status": "failed",
                        "error": transcription_result.get("error", "Unknown error during transcription"),
                    })
                    continue
                
                # Send transcription success status
                await send_status(
                    "transcription_complete", 
                    f"Transcription complete: {video_title}", 
                    {"url": url, "step": "transcription_complete", "title": video_title}
                )
                
                # Step 3: Process the transcript with AI
                Logger.log(True, f"Processing transcript with AI for {url}")
                await send_status(
                    "processing_ai", 
                    f"Processing transcript with AI: {video_title}", 
                    {"url": url, "step": "process_ai", "title": video_title}
                )
                
                transcript_text = transcription_result.get("text", "")
                
                # Include text preview in status update
                text_preview = transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text
                await send_status(
                    "transcript_preview", 
                    f"Generated transcript preview", 
                    {"url": url, "preview": text_preview}
                )
                
                processed_text = process_text(transcript_text)
                
                # Save the processed transcript
                output_file = os.path.join(
                    transcripts_dir, 
                    f"{os.path.splitext(os.path.basename(audio_file))[0]}_processed.md"
                )
                
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                
                processed_count += 1
                
                # Send success status
                await send_status(
                    "success", 
                    f"Processing complete for: {video_title}", 
                    {
                        "url": url, 
                        "step": "complete", 
                        "title": video_title,
                        "transcript_file": output_file,
                        "progress": f"{processed_count}/{len(urls)}"
                    }
                )
                
                results.append({
                    "url": url,
                    "status": "success",
                    "title": video_title,
                    "transcript_file": output_file,
                })
                
            except Exception as e:
                error_msg = f"Error processing {url}: {str(e)}"
                Logger.log(False, error_msg)
                
                # Send failure status
                await send_status(
                    "failed", 
                    error_msg,
                    {"url": url, "error": str(e)}
                )
                
                results.append({
                    "url": url,
                    "status": "failed",
                    "error": str(e),
                })
        
        # Update the processing request in the database
        db_request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
        if db_request:
            db_request.result = results
            db.commit()
        
        # Send final completion status
        await send_status(
            "completed", 
            f"Completed processing {len(urls)} YouTube URLs. Success: {processed_count}, Failed: {len(urls) - processed_count}",
            {"total": len(urls), "success": processed_count, "failed": len(urls) - processed_count, "results": results}
        )
        
        Logger.log(True, f"Completed processing for request {request_id}")
        
    except Exception as e:
        error_msg = f"Error in process_videos: {str(e)}"
        Logger.log(False, error_msg)
        
        # Send error status
        try:
            await ws_manager.send_status_update(
                user_id, 
                request_id, 
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "status": "error",
                    "message": error_msg,
                    "data": {"error": str(e)}
                }
            )
        except Exception as ws_error:
            Logger.log(False, f"Failed to send error status via WebSocket: {str(ws_error)}")
        
        # Update the processing request with the error
        db_request = db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).first()
        if db_request:
            db_request.result = {"status": "failed", "error": str(e)}
            db.commit()
