"""                               
© r1cco.com

Advanced Audio Transcription Tool with AI Processing

This script implements a comprehensive audio transcription system that can:
1. Transcribe audio files using OpenAI's Whisper model
2. Process YouTube videos by extracting and transcribing their audio
3. Handle batch processing of multiple files efficiently
4. Optimize transcription with GPU acceleration when available
5. Post-process transcriptions with AI for better organization

Key Features:
- Support for various Whisper models (tiny, base, small, medium, large, turbo)
- YouTube video processing capabilities
- Parallel processing for batch transcription
- GPU acceleration (CUDA) when available
- AI-based post-processing of transcripts
- Detailed logging and error handling
- Configurable transcription parameters

Required Dependencies:
- openai-whisper: OpenAI's Whisper speech recognition model
- torch: PyTorch for model acceleration
- numpy: For audio processing
- youtube_dl/pytube: For YouTube video downloading (when using --youtube)

Usage Examples:

1. Transcribe a single audio file:
   python run.py --audio path/to/audio/file.mp3

2. Transcribe YouTube videos (single or multiple):
   python run.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID"
   python run.py --youtube "URL1" "URL2" "URL3" --output transcripts/

3. Batch process multiple audio files:
   python run.py --audio file1.mp3 file2.mp3 file3.mp3

4. Use parallel processing for faster batch transcription:
   python run.py --audio *.mp3 --parallel

Arguments:
--audio:           Path to input audio file(s) or patterns (accepts wildcards)
--youtube:         YouTube video URL(s) for transcription (one or more URLs)
--output:          Output path (file or directory)
--device:          Device to use for processing (cpu, cuda, auto) (default: auto)
--verbose, -v:     Enable verbose mode for debugging
--parallel:        Enable parallel processing for batch transcription
--config:          Path to configuration file (default: config.yml)
--only-raw:        Save only the raw transcript without AI processing

Examples:
    # Transcribe a local audio file with GPU acceleration
    python run.py --audio interview.mp3 --device cuda

    # Process multiple YouTube videos and save outputs to a directory
    python run.py --youtube "URL1" "URL2" "URL3" --output transcripts/

    # Batch process all MP3 files in a directory
    python run.py --audio "recordings/*.mp3" --output transcripts/

    # Enable verbose logging for troubleshooting
    python run.py --audio meeting.mp3 --verbose

Output Files:
- transcript.txt: Raw transcript text
- transcript_organized.md: AI-processed and organized transcript (when available)
- Additional files in the 'dist' directory by default, or in the specified output directory
"""

import os
import torch
import glob
import whisper
import numpy as np
import multiprocessing
import asyncio

from typing import Dict, Any, List
from argparse import ArgumentParser
from app.utils.logger import Logger
from app.utils.transcription import process_text
from app.utils.functions import load_config, ensure_dir, sanitize_filename
from app.utils.yt_downloader import YouTubeDownloader


class WhisperModel:
    """
    Singleton class to manage Whisper model loading and caching.

    This class implements the singleton pattern to ensure only one instance
    of the Whisper model is loaded into memory, improving efficiency when
    processing multiple files.
    """

    _instance = None
    _model = None
    _device = None
    _model_name = None

    def __new__(cls, model_name: str = None, device: str = None):
        if cls._instance is None:
            cls._instance = super(WhisperModel, cls).__new__(cls)
            cls._model = None
            cls._device = None
            cls._model_name = None
        return cls._instance

    def load_model(self, model_name: str, device: str = None) -> whisper.Whisper:
        """
        Load the Whisper model, reusing it if already loaded with the same parameters.

        This method implements lazy loading and caching of the Whisper model to
        optimize resource usage and performance.

        Args:
            model_name (str): Name of the Whisper model to load (e.g., 'tiny', 'base', 'small', 'medium', 'large', 'turbo')
            device (str, optional): Device to run the model on ('cpu', 'cuda'). Defaults to using CUDA if available.

        Returns:
            whisper.Whisper: Loaded Whisper model instance

        Raises:
            ValueError: If the model name is invalid or if model loading fails
        """
        if (
            self._model is None
            or model_name != self._model_name
            or device != self._device
        ):

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            try:
                Logger.log(True, f"Loading Whisper model: {model_name}")
                self._model = whisper.load_model(
                    model_name, device=device, in_memory=True
                )
                self._device = device
                self._model_name = model_name
            except Exception as e:
                error_msg = f"Failed to load Whisper model: {str(e)}"
                Logger.log(False, error_msg, "error")
                raise ValueError(error_msg)

        return self._model

    @property
    def device(self) -> str:
        """
        Get the current device used by the model.

        Returns:
            str: Current device ('cpu' or 'cuda')
        """
        return self._device

    @property
    def model_name(self) -> str:
        """
        Get the current model name.

        Returns:
            str: Name of the currently loaded model
        """
        return self._model_name


def get_config() -> Dict[str, Any]:
    """
    Load configuration from config.yml file or use defaults.

    Attempts to load the configuration from the config.yml file. If the file
    doesn't exist or can't be loaded, returns a default configuration.

    Returns:
        Dict[str, Any]: Configuration dictionary with transcription settings
    """
    try:
        return load_config()
    except FileNotFoundError:
        return {
            "transcription": {
                "model_name": "large-v3-turbo",
                "temperature": 0.0,
                "no_speech_threshold": 0.4,
                "beam_size": 5,
                "best_of": 5,
                "compression_ratio_threshold": 2.0,
                "logprob_threshold": -0.5,
                "word_timestamps": True,
                "prepend_punctuations": "\"'",
                "append_punctuations": '".,!?',
                "initial_prompt": "Diálogo rápido, conversa dinâmica, falas curtas",
            }
        }


def select_device(requested_device: str = None) -> str:
    """
    Select the appropriate device for transcription based on availability.

    Determines the optimal device for running the Whisper model based on the
    requested device and system capabilities.

    Args:
        requested_device (str, optional): Device requested by user ('cpu', 'cuda', 'auto').
                                         Defaults to 'auto'.

    Returns:
        str: Selected device name ('cpu' or 'cuda')
    """
    if requested_device == "auto" or requested_device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        Logger.log(False, "GPU not available", "warning")
        Logger.log(
            False, "GPU requested but not available, falling back to CPU", "warning"
        )
        device = "cpu"
    else:
        device = requested_device
    return device


async def transcribe_audio(
    input_path: Dict[str, str], output_path: str = None, device: str = None
) -> Dict[str, Any]:
    """
    Transcribe audio file using the Whisper model.

    Processes an audio file to generate a text transcription using the
    Whisper model with the specified parameters.

    Args:
        input_path (Dict[str, str]): Dictionary containing information about the input file,
                                    including 'file_path', 'file_name', etc.
        output_path (str, optional): Path to save the transcription output.
                                    If None, uses default output location.
        device (str, optional): Device to use for transcription ('cpu', 'cuda', 'auto').
                              Defaults to auto-selection.

    Returns:
        Dict[str, Any]: Transcription result including 'text' and metadata,
                       or error information in case of failure
    """
    try:
        config = get_config().get("transcription", {})
        device = select_device(device)

        model_name = config.get("model_name", "large-v3-turbo")
        model_manager = WhisperModel()
        model = model_manager.load_model(model_name, device)

        fp16 = device != "cpu" and config.get("fp16", True)

        Logger.log(True, "Processing audio file")
        try:
            audio = whisper.load_audio(input_path["file_path"])
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {str(e)}")

        if not isinstance(audio, np.ndarray):
            raise ValueError("Invalid audio format")

        Logger.log(True, "Transcribing complete audio")

        result = model.transcribe(
            audio,
            fp16=fp16,
            verbose=True,
            temperature=config.get("temperature", 0.3),
            no_speech_threshold=config.get("no_speech_threshold", 0.6),
            condition_on_previous_text=config.get("condition_on_previous_text", True),
        )

        if output_path:
            Logger.log(True, "Saving transcription")
            ensure_dir(os.path.dirname(output_path))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

        Logger.log(True, "Transcription complete")
        return result

    except ValueError as e:
        Logger.log(False, f"Transcription failed: {str(e)}", "error")
        raise ValueError(str(e))

    except Exception as e:
        Logger.log(False, f"Unexpected error during transcription: {str(e)}", "error")
        raise ValueError(str(e))


async def process_url(url: str, request_id: str):
    from app.model.request import ProcessingRequest
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).update(
            {"status": "processing"}
        )
        db.commit()

        result = await transcribe_audio({"file_path": url})

        db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).update(
            {"status": "completed", "result": result}
        )
        db.commit()
    except Exception as e:
        db.query(ProcessingRequest).filter(ProcessingRequest.id == request_id).update(
            {"status": "failed"}
        )
        db.commit()
    finally:
        db.close()


def save_and_process_transcript(
    transcript_text: str, output_path: str = None, file_name: str = None, only_raw: bool = False
) -> bool:
    """
    Save transcript to file and optionally process with AI.

    Saves the raw transcript text to a file and can optionally process it
    further with AI organization if configured to do so.

    Args:
        transcript_text (str): Raw transcript text to save and potentially process
        output_path (str, optional): Path to save the processed transcript.
                                    If None, uses a default location.
        file_name (str, optional): Name to use for the output file.
                                  If None, uses a default name.
        only_raw (bool, optional): If True, skips AI processing. Defaults to False.

    Returns:
        bool: True if saving and processing was successful, False otherwise
    """
    try:
        ensure_dir("dist")

        if output_path:
            raw_path = output_path
            base_output_path = os.path.splitext(output_path)[0]
        else:
            safe_name = "transcript"
            if file_name:
                safe_name = sanitize_filename(file_name)
            base_output_path = f"dist/{safe_name}"
            raw_path = f"{base_output_path}.txt"

        Logger.log(True, "Saving raw transcript")
        ensure_dir(os.path.dirname(raw_path))
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        Logger.log(True, f"Raw transcription saved to: {raw_path}")

        if only_raw:
            Logger.log(True, "Skipping AI processing as per --only-raw flag.")
            return True

        Logger.log(True, "Processing transcript with AI")
        try:
            processed_text = process_text(transcript_text)

            if processed_text:
                Logger.log(True, "Saving organized transcript")
                organized_path = f"{base_output_path}_organized.md"
                with open(organized_path, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                Logger.log(True, "Processing complete")
                Logger.log(True, f"Organized transcription saved to: {organized_path}")
                return True
            else:
                Logger.log(False, "AI processing failed", "error")
                Logger.log(
                    False, "Organized version not saved due to processing errors"
                )
        except Exception as e:
            Logger.log(False, f"AI processing failed: {str(e)}", "error")
            Logger.log(False, f"AI processing error: {str(e)}")

        return True
    except Exception as e:
        Logger.log(False, f"Failed to save transcript: {str(e)}", "error")
        Logger.log(False, f"Error saving transcript: {str(e)}")
        return False


def process_batch(
    files: List[str], output_dir: str = None, device: str = None, only_raw: bool = False
) -> List[Dict[str, Any]]:
    """
    Process multiple audio files in batch.

    Processes a list of audio files sequentially, optimizing for performance
    by reusing the loaded model across files.

    Args:
        files (List[str]): List of audio file paths to process
        output_dir (str, optional): Directory to save transcription outputs.
                                   If None, uses a default location.
        device (str, optional): Device to use for transcription ('cpu', 'cuda', 'auto').
                              Defaults to auto-selection.
        only_raw (bool, optional): If True, skips AI processing. Defaults to False.

    Returns:
        List[Dict[str, Any]]: List of results for each processed file, including
                             success/failure status and output paths or error details
    """
    results = []

    for file_path in files:
        try:
            Logger.log(True, f"Processing file: {file_path}")

            if output_dir:
                file_name = os.path.basename(file_path)
                name_without_ext = os.path.splitext(file_name)[0]
                output_path = os.path.join(output_dir, f"{name_without_ext}.txt")
            else:
                output_path = None

            audio_info = {"file_path": file_path, "title": os.path.basename(file_path)}
            result = asyncio.run(transcribe_audio(audio_info, output_path, device))

            if result:
                save_and_process_transcript(
                    result["text"], output_path, os.path.basename(file_path), only_raw
                )

            results.append(
                {"file": file_path, "success": bool(result), "output": output_path}
            )

        except Exception as e:
            Logger.log(False, f"Error processing {file_path}: {str(e)}", "error")
            results.append({"file": file_path, "success": False, "error": str(e)})

    return results


def process_youtube_batch(urls: List[str], output_dir: str = None, device: str = None, only_raw: bool = False) -> List[Dict[str, Any]]:
    """
    Process multiple YouTube URLs in batch.

    Args:
        urls (List[str]): List of YouTube URLs to process
        output_dir (str, optional): Directory to save transcription outputs.
                                   If None, uses a default location.
        device (str, optional): Device to use for transcription.
                              Defaults to auto-selection.
        only_raw (bool, optional): If True, skips AI processing. Defaults to False.

    Returns:
        List[Dict[str, Any]]: List of results for each processed URL
    """
    results = []
    downloader = YouTubeDownloader()

    for url in urls:
        try:
            Logger.log(True, f"Processing YouTube URL: {url}")
            
            audio_path = downloader.download_audio_only(url)
            if not audio_path.get("success", False):
                Logger.log(False, f"YouTube download failed for {url}", "error")
                results.append({"url": url, "success": False, "error": audio_path.get("error", "Download failed")})
                continue

            if output_dir:
                file_name = audio_path.get("title", "youtube_transcript")
                output_path = os.path.join(output_dir, f"{file_name}.txt")
            else:
                output_path = None

            transcript_result = asyncio.run(transcribe_audio(audio_path, output_path, device))

            if transcript_result:
                save_and_process_transcript(
                    transcript_result["text"], output_path, audio_path.get("title", "youtube_transcript"), only_raw
                )
                results.append({"url": url, "success": True, "output": output_path})
            else:
                results.append({"url": url, "success": False, "error": "Transcription failed"})

        except Exception as e:
            Logger.log(False, f"Error processing {url}: {str(e)}", "error")
            results.append({"url": url, "success": False, "error": str(e)})

    return results


def main():
    """
    Main function for audio transcription workflow.

    Parses command line arguments, sets up the environment, and orchestrates
    the transcription process for single files or batches of audio files.
    Handles output file naming and placement as well as error conditions.
    """
    parser = ArgumentParser(description="Audio transcription processor")
    parser.add_argument(
        "--audio", required=False, nargs="+", help="Input audio file(s) or patterns"
    )
    parser.add_argument(
        "--youtube", type=str, nargs="+", help="YouTube video URL(s) for transcription"
    )
    parser.add_argument("--output", help="Output path (file or directory)")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for processing (cpu, cuda, or auto for automatic detection)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose mode for debugging",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for batch transcription",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--only-raw",
        action="store_true",
        help="Save only the raw transcript without AI processing",
    )
    args = parser.parse_args()
    device = select_device(args.device)

    if args.verbose:
        Logger.set_verbose(True)

    if args.youtube:
        try:
            from app.utils.yt_downloader import YouTubeDownloader

            is_batch = len(args.youtube) > 1
            output_is_dir = args.output and (os.path.isdir(args.output) or args.output.endswith("/") or is_batch)

            if is_batch:
                Logger.log(True, f"Batch processing {len(args.youtube)} YouTube URLs")
                if output_is_dir:
                    ensure_dir(args.output)
                    output_dir = args.output
                else:
                    output_dir = "dist"
                    ensure_dir(output_dir)

                results = process_youtube_batch(args.youtube, output_dir, device, args.only_raw)
                
                success_count = sum(1 for r in results if r["success"])
                print(f"\n✓ Processed {success_count}/{len(results)} YouTube videos successfully")

                if success_count < len(results):
                    failed = [r["url"] for r in results if not r["success"]]
                    print(f"✗ Failed to process: {', '.join(failed)}")

                return results
            else:
                Logger.log(True, "Initializing YouTube downloader")
                downloader = YouTubeDownloader()

                Logger.log(True, "Downloading audio from YouTube")
                audio_path = downloader.download_audio_only(args.youtube[0])

                if not audio_path.get("success", False):
                    Logger.log(False, "YouTube download failed", "error")
                    Logger.log(
                        False,
                        f"Failed to download YouTube audio: {audio_path.get('error', 'Unknown error')}",
                    )
                    return False

                Logger.log(True, "Preparing for transcription")
                transcript_result = asyncio.run(transcribe_audio(audio_path, None))

                if transcript_result:
                    file_name = audio_path.get("title", "youtube_transcript")
                    save_and_process_transcript(
                        transcript_result["text"], args.output, file_name, args.only_raw
                    )
                    return transcript_result
                return False

        except ImportError:
            Logger.log(False, "YouTube downloader module not available", "error")
            Logger.log(
                False,
                "YouTube downloader module not available. Please check installation.",
            )
            return False

    elif args.audio:
        audio_files = []
        for pattern in args.audio:
            if "*" in pattern or "?" in pattern:
                matched = glob.glob(pattern)
                if not matched:
                    Logger.log(False, f"No files match pattern: {pattern}", "warning")
                audio_files.extend(matched)
            else:
                audio_files.append(pattern)

        if not audio_files:
            Logger.log(False, "No audio files specified", "error")
            print("\n✗ No audio files to process. Please specify input files.")
            return False

        is_batch = len(audio_files) > 1
        output_is_dir = args.output and (
            os.path.isdir(args.output) or args.output.endswith("/") or is_batch
        )

        if is_batch:
            Logger.log(True, f"Batch processing {len(audio_files)} files")

            if output_is_dir:
                ensure_dir(args.output)
                output_dir = args.output
            else:
                output_dir = "dist"
                ensure_dir(output_dir)

            if args.parallel and len(audio_files) > 1:
                max_workers = min(multiprocessing.cpu_count(), len(audio_files))
                Logger.log(True, f"Using {max_workers} parallel workers", "debug")

                tasks = []
                for file_path in audio_files:
                    if output_is_dir:
                        file_name = os.path.basename(file_path)
                        name_without_ext = os.path.splitext(file_name)[0]
                        output_path = os.path.join(
                            output_dir, f"{name_without_ext}.txt"
                        )
                    else:
                        output_path = None

                    tasks.append((file_path, output_path))

                results = []
                for file_path, output_path in tasks:
                    audio_info = {
                        "file_path": file_path,
                        "title": os.path.basename(file_path),
                    }
                    try:
                        result = asyncio.run(
                            transcribe_audio(audio_info, output_path, device)
                        )
                        if result:
                            save_and_process_transcript(
                                result["text"], output_path, os.path.basename(file_path), args.only_raw
                            )
                        results.append(
                            {
                                "file": file_path,
                                "success": bool(result),
                                "output": output_path,
                            }
                        )
                    except Exception as e:
                        Logger.log(
                            False, f"Error processing {file_path}: {str(e)}", "error"
                        )
                        results.append(
                            {"file": file_path, "success": False, "error": str(e)}
                        )
            else:
                results = process_batch(audio_files, output_dir, device, args.only_raw)

            success_count = sum(1 for r in results if r["success"])
            print(f"\n✓ Processed {success_count}/{len(results)} files successfully")

            if success_count < len(results):
                failed = [r["file"] for r in results if not r["success"]]
                print(f"✗ Failed to process: {', '.join(failed)}")

            return results
        else:
            # Single file processing
            audio_path = audio_files[0]

            if output_is_dir:
                ensure_dir(args.output)
                file_name = os.path.basename(audio_path)
                name_without_ext = os.path.splitext(file_name)[0]
                output_path = os.path.join(args.output, f"{name_without_ext}.txt")
            else:
                output_path = args.output

            audio_info = {
                "file_path": audio_path,
                "title": os.path.basename(audio_path),
            }
            result = asyncio.run(transcribe_audio(audio_info, output_path, device))

            if result:
                save_and_process_transcript(
                    result["text"], output_path, os.path.basename(audio_path), args.only_raw
                )

            return result
    else:
        parser.print_help()
        print("\n✗ Please specify either --audio or --youtube parameter")
        return False


if __name__ == "__main__":
    main()
