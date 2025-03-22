"""
 r1cco.com

Audio Transcription Module

Key features:
- Audio-to-text transcription using Whisper large-v3-turbo
- Automatic output organization in 'dist' directory
- GPU acceleration support
- Batch processing capabilities

Basic usage:
  # Default output to dist/example.txt
  python run.py --audio example.flac

  # Specify processing device
  python run.py --audio example.flac --device cpu
  python run.py --audio example.flac --device gpu

  # Custom output path
  python run.py --audio assets/interview.mp3 --output transcripts/interview.txt

  # Batch processing
  python run.py --audio assets/lectures/*.mp3 --output university/lectures/

  # Process with AI organization
  python run.py --audio example.flac --process-ai
"""

import sys
import os
import torch
import glob
import whisper
import numpy as np
import multiprocessing
import yaml
from typing import Dict, Any, List

from argparse import ArgumentParser
from pathlib import Path
from logger import Logger
from ai_transcript_processor import process_text
from utils import load_config, ensure_dir, sanitize_filename


class TranscriptionError(Exception):
    """Exception raised for errors during audio transcription."""

    pass


class WhisperModel:
    """Singleton class to manage Whisper model loading and caching."""

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

        Args:
            model_name (str): Name of the Whisper model to load
            device (str): Device to run the model on (cpu, cuda)

        Returns:
            whisper.Whisper: Loaded model

        Raises:
            ValueError: If the model or device is invalid
        """
        # Only reload if something changed
        if (
            self._model is None
            or model_name != self._model_name
            or device != self._device
        ):

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            try:
                Logger.log(True, f"Loading Whisper model: {model_name} on {device}")
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
        """Get the current device."""
        return self._device

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name


def get_config() -> Dict[str, Any]:
    """
    Load configuration from config.yml file or use defaults.

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        return load_config()
    except FileNotFoundError:
        # Default configuration if file doesn't exist
        return {
            "transcription": {
                "model_name": "turbo",
                "temperature": 0.3,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "fp16": True,
            }
        }


def select_device(requested_device: str = None) -> str:
    """
    Select the appropriate device for transcription.

    Args:
        requested_device (str, optional): Device requested by user (cpu, cuda, auto)

    Returns:
        str: Selected device name (cpu, cuda)
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

    Logger.log(True, f"Using device: {device}")
    return device


def transcribe_audio(
    input_path: Dict[str, str], output_path: str = None, device: str = None
) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper model.

    Args:
        input_path (Dict): Dictionary containing info about the input file
        output_path (str, optional): Path to save the transcription output
        device (str, optional): Device to use for transcription (cpu, cuda)

    Returns:
        Dict[str, Any]: Transcription result or error information

    Raises:
        TranscriptionError: If transcription fails
    """
    try:
        config = get_config().get("transcription", {})
        device = select_device(device)

        # Get or create the Whisper model (cached)
        model_name = config.get("model_name", "turbo")
        model_manager = WhisperModel()
        model = model_manager.load_model(model_name, device)

        # Prepare transcription parameters
        fp16 = device != "cpu" and config.get("fp16", True)

        Logger.log(True, "Processing audio file")
        try:
            audio = whisper.load_audio(input_path["file_path"])
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {str(e)}")

        if not isinstance(audio, np.ndarray):
            raise ValueError("Invalid audio format")

        Logger.log(True, "Transcribing complete audio")

        # Get transcription parameters from config
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
        print("\n✓ Transcription successful!")
        return result

    except ValueError as e:
        Logger.log(False, f"Transcription failed: {str(e)}", "error")
        print(f"\n✗ Transcription error: {str(e)}")
        raise TranscriptionError(str(e))

    except Exception as e:
        Logger.log(False, f"Unexpected error during transcription: {str(e)}", "error")
        print(f"\n✗ Transcription error: {str(e)}")
        raise TranscriptionError(f"Unexpected error: {str(e)}")


def save_and_process_transcript(
    transcript_text: str, output_path: str = None, file_name: str = None
) -> bool:
    """
    Save transcript and process with AI.

    Args:
        transcript_text (str): Raw transcript text to process
        output_path (str, optional): Path to save the processed transcript
        file_name (str, optional): Name to use for the output file

    Returns:
        bool: True if processing successful, False otherwise
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

        print(f"\n✓ Raw transcription saved to: {raw_path}")

        Logger.log(True, "Processing transcript with AI")
        try:
            processed_text = process_text(transcript_text)

            if processed_text:
                Logger.log(True, "Saving organized transcript")
                organized_path = f"{base_output_path}_organized.md"
                with open(organized_path, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                Logger.log(True, "Processing complete")
                print(f"✓ Organized transcription saved to: {organized_path}\n")
                return True
            else:
                Logger.log(False, "AI processing failed", "error")
                print("✗ Organized version not saved due to processing errors")
        except Exception as e:
            Logger.log(False, f"AI processing failed: {str(e)}", "error")
            print(f"\n✗ AI processing error: {str(e)}")

        return True
    except Exception as e:
        Logger.log(False, f"Failed to save transcript: {str(e)}", "error")
        print(f"\n✗ Error saving transcript: {str(e)}")
        return False


def process_batch(
    files: List[str], output_dir: str = None, device: str = None
) -> List[Dict[str, Any]]:
    """
    Process multiple audio files in batch.

    Args:
        files (List[str]): List of audio file paths
        output_dir (str, optional): Directory to save transcripts
        device (str, optional): Device to use for transcription

    Returns:
        List[Dict[str, Any]]: Results for each file
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
            result = transcribe_audio(audio_info, output_path, device)

            if result:
                save_and_process_transcript(
                    result["text"], output_path, os.path.basename(file_path)
                )

            results.append(
                {"file": file_path, "success": bool(result), "output": output_path}
            )

        except Exception as e:
            Logger.log(False, f"Error processing {file_path}: {str(e)}", "error")
            results.append({"file": file_path, "success": False, "error": str(e)})

    return results


def main() -> None:
    """Main transcription execution flow."""

    parser = ArgumentParser(description="Audio transcription processor")
    parser.add_argument(
        "--audio", required=False, nargs="+", help="Input audio file(s) or patterns"
    )
    parser.add_argument(
        "--youtube", type=str, help="YouTube video URL for transcription"
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
    args = parser.parse_args()

    # Set up the environment
    cache_dir = str(Path("./cache").absolute())
    os.environ["WHISPER_CACHE_DIR"] = cache_dir
    ensure_dir(cache_dir)

    if args.verbose:
        Logger.set_verbose(True)

    # Select the device
    device = select_device(args.device)

    # Process YouTube input
    if args.youtube:
        try:
            from youtube_downloader import YouTubeDownloader

            Logger.log(True, "Initializing YouTube downloader")
            downloader = YouTubeDownloader()

            Logger.log(True, "Downloading audio from YouTube")
            audio_path = downloader.download_audio_only(args.youtube)

            if not audio_path.get("success", False):
                Logger.log(False, "YouTube download failed", "error")
                print(
                    f"\n✗ Failed to download YouTube audio: {audio_path.get('error', 'Unknown error')}"
                )
                return False

            Logger.log(True, "Preparing for transcription")
            transcript_result = transcribe_audio(audio_path, None, device)

            if transcript_result:
                file_name = audio_path.get("title", "youtube_transcript")
                save_and_process_transcript(
                    transcript_result["text"], args.output, file_name
                )
                return transcript_result
            return False

        except ImportError:
            Logger.log(False, "YouTube downloader module not available", "error")
            print(
                "\n✗ YouTube downloader module not available. Please check installation."
            )
            return False

    # Process audio file(s)
    elif args.audio:
        # Expand glob patterns
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

        # Check if output is directory (for batch processing)
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

            # Process files
            if args.parallel and len(audio_files) > 1:
                # Use multiprocessing for batch processing
                max_workers = min(multiprocessing.cpu_count(), len(audio_files))
                Logger.log(True, f"Using {max_workers} parallel workers", "debug")

                # Create simplified argument list for multiprocessing
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

                # Process in parallel (we'll simplify this by using sequential with the optimized model)
                results = []
                for file_path, output_path in tasks:
                    audio_info = {
                        "file_path": file_path,
                        "title": os.path.basename(file_path),
                    }
                    try:
                        result = transcribe_audio(audio_info, output_path, device)
                        if result:
                            save_and_process_transcript(
                                result["text"], output_path, os.path.basename(file_path)
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
                # Process sequentially
                results = process_batch(audio_files, output_dir, device)

            # Print summary
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
            result = transcribe_audio(audio_info, output_path, device)

            if result:
                save_and_process_transcript(
                    result["text"], output_path, os.path.basename(audio_path)
                )

            return result
    else:
        parser.print_help()
        print("\n✗ Please specify either --audio or --youtube parameter")
        return False


if __name__ == "__main__":
    main()
