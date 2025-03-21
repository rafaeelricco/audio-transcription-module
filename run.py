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
import warnings
import glob
import whisper
from pathlib import Path
import numpy as np

from argparse import ArgumentParser
from logger import Logger
from ai_transcript_processor import process_text


def transcribe_audio(input_path, output_path=None, device=None):
    try:
        # Determine the best device to use
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set appropriate floating point precision based on device
        fp16 = device != "cpu"

        Logger.log(True, "Loading Whisper model")
        # Load the model with the appropriate device and precision settings
        model = whisper.load_model("turbo", device=device, in_memory=True)

        Logger.log(True, "Processing audio file")
        audio = whisper.load_audio(input_path["file_path"])
        audio = whisper.pad_or_trim(audio)
        if not isinstance(audio, np.ndarray):
            raise ValueError("Invalid audio format")

        Logger.log(True, "Transcribing audio")
        # Pass fp16 parameter to transcribe method, not to load_model
        result = model.transcribe(audio, fp16=fp16)

        if output_path:
            Logger.log(True, "Saving transcription")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

        Logger.log(True, "Transcription complete")
        print("\n✓ Transcription successful!")
        return result
    except Exception as e:
        Logger.log(False, "Transcription failed")
        print(f"\n✗ Transcription error: {str(e)}")
        return False


def save_and_process_transcript(transcript_text, output_path=None, file_name=None):
    """Save transcript and process with AI"""
    try:
        # Ensure dist directory exists
        os.makedirs("dist", exist_ok=True)

        # Determine output path for raw transcript
        if output_path:
            raw_path = output_path
            base_output_path = os.path.splitext(output_path)[0]
        else:
            # Create a default filename if none provided
            safe_name = "transcript"
            if file_name:
                safe_name = "".join(
                    c if c.isalnum() or c in [" ", "-", "_"] else "_" for c in file_name
                )
            base_output_path = f"dist/{safe_name}"
            raw_path = f"{base_output_path}.txt"

        # Save raw transcript
        Logger.log(True, "Saving raw transcript")
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
                Logger.log(False, "AI processing failed")
                print("✗ Organized version not saved due to processing errors")
        except Exception as e:
            Logger.log(False, "AI processing failed")
            print(f"\n✗ AI processing error: {str(e)}")

        return True
    except Exception as e:
        Logger.log(False, "Failed to save transcript")
        print(f"\n✗ Error saving transcript: {str(e)}")
        return False


def main():
    """Main transcription execution flow"""

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
    args = parser.parse_args()

    os.environ["WHISPER_CACHE_DIR"] = str(Path("./cache").absolute())
    Path("./cache").mkdir(exist_ok=True)

    # Determine device based on user preference and availability
    device = None
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        Logger.log(False, "GPU not available")
        print("\n⚠️ GPU requested but not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    print(f"\n✓ Using device: {device}")

    if args.youtube:
        from youtube_downloader import YouTubeDownloader

        Logger.log(True, "Initializing YouTube downloader")
        downloader = YouTubeDownloader()
        
        Logger.log(True, "Downloading audio from YouTube")
        audio_path = downloader.download_audio_only(args.youtube)

        if not audio_path.get("success", False):
            Logger.log(False, "YouTube download failed")
            print(f"\n✗ Failed to download YouTube audio: {audio_path.get('error', 'Unknown error')}")
            return False

        Logger.log(True, "Preparing for transcription")
        transcript_result = transcribe_audio(audio_path, None, device)

        if transcript_result:
            # Save transcript and process with AI if requested
            file_name = audio_path.get("title", "youtube_transcript")
            save_and_process_transcript(
                transcript_result["text"], args.output, file_name
            )
            return transcript_result
        return False

    if not args.audio:
        Logger.log(False, "No input specified")
        print(
            "\n✗ No input specified. Please provide either --audio or --youtube argument."
        )
        sys.exit(1)

    expanded_files = []

    Logger.log(True, "Finding audio files")
    for pattern in args.audio:
        expanded_files.extend(glob.glob(pattern, recursive=True))

    if not expanded_files:
        Logger.log(False, "No audio files found")
        print("\n✗ No audio files found matching the provided patterns.")
        sys.exit(1)

    Logger.log(True, "Files found")
    print(f"\n✓ Found {len(expanded_files)} files to process:")
    for f in expanded_files:
        print(f" - {f}")

    for index, file in enumerate(expanded_files):
        Logger.log(True, f"Processing file {index+1}/{len(expanded_files)}")
        print(f"\n✓ Transcribing audio file {index+1}/{len(expanded_files)}...")

        # Determine output path for this file
        file_output = None
        if args.output:
            if len(expanded_files) > 1 and not args.output.endswith((".txt", ".md")):
                # If output is a directory, create one file per input
                os.makedirs(args.output, exist_ok=True)
                file_base = os.path.splitext(os.path.basename(file))[0]
                file_output = os.path.join(args.output, f"{file_base}.txt")
            else:
                # Use the specified output directly
                file_output = args.output

        transcript_result = transcribe_audio({"file_path": file}, None, device)

        if transcript_result:
            # Extract filename without extension for naming
            file_name = os.path.splitext(os.path.basename(file))[0]

            # Save transcript and process with AI if requested
            save_and_process_transcript(
                transcript_result["text"], file_output, file_name
            )
            print(f"\n✓ Transcription of {file} successful!")
        else:
            print(f"\n✗ Transcription of {file} failed.")

    sys.exit(0)


if __name__ == "__main__":
    main()
