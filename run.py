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
from ui import ProgressBar


def transcribe_audio(input_path, output_path=None, device=None):
    try:
        # Determine the best device to use
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set appropriate floating point precision based on device
        fp16 = device != "cpu"

        # Load the model with the appropriate device and precision settings
        model = whisper.load_model("turbo", device=device, in_memory=True)

        audio = whisper.load_audio(input_path["file_path"])
        audio = whisper.pad_or_trim(audio)
        if not isinstance(audio, np.ndarray):
            raise ValueError("Invalid audio format")

        # Pass fp16 parameter to transcribe method, not to load_model
        result = model.transcribe(audio, fp16=fp16)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

        print("\n✓ Transcription successful!")
        return result
    except Exception as e:
        print(f"\n✗ Transcription error: {str(e)}")
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
        "--process-ai",
        action="store_true",
        help="Process transcript with AI for summarization and organization",
    )
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
        print("\n⚠️ GPU requested but not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    print(f"\n✓ Using device: {device}")

    if args.youtube:
        from youtube_downloader import YouTubeDownloader
        from ai_transcript_processor import process_text

        print("\n✓ Downloading audio from YouTube...")
        downloader = YouTubeDownloader()
        audio_path = downloader.download_audio_only(args.youtube)

        print("\n✓ Transcribing audio...")
        transcript_result = transcribe_audio(audio_path, args.output, device)

        if transcript_result and args.process_ai:
            print("\n✓ Processing transcript with AI...")
            try:
                # Process the text with AI
                processed_text = process_text(transcript_result["text"])
                ai_success = processed_text is not None

                # Determine output paths
                if args.output:
                    base_output_path = os.path.splitext(args.output)[0]
                    raw_path = args.output
                else:
                    # Use video title as filename
                    video_title = audio_path.get("title", "transcript")
                    safe_title = "".join(
                        c if c.isalnum() or c in [" ", "-", "_"] else "_"
                        for c in video_title
                    )
                    os.makedirs("dist", exist_ok=True)
                    base_output_path = f"dist/{safe_title}"
                    raw_path = f"{base_output_path}.txt"
                    # Ensure raw text is saved if not already
                    if not args.output:
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(transcript_result["text"])

                # Save results and display messages based on success
                if ai_success:
                    organized_path = f"{base_output_path}_organized.md"
                    with open(organized_path, "w", encoding="utf-8") as f:
                        f.write(processed_text)
                    print(f"\n✓ Raw transcription saved to: {raw_path}")
                    print(f"✓ Organized transcription saved to: {organized_path}\n")
                else:
                    print(f"\n✓ Raw transcription saved to: {raw_path}")
                    print("✗ Organized version not saved due to processing errors")

            except Exception as e:
                print(f"\n✗ AI processing error: {str(e)}")
                print(
                    f"✓ Raw transcription saved to: {args.output if args.output else 'dist/transcript.txt'}"
                )

        return transcript_result

    expanded_files = []
    for pattern in args.audio:
        expanded_files.extend(glob.glob(pattern, recursive=True))

    if not expanded_files:
        print("\n✗ No audio files found matching the provided patterns.")
        sys.exit(1)

    print(f"\n✓ Found {len(expanded_files)} files to process:")
    for f in expanded_files:
        print(f" - {f}")

    for file in expanded_files:
        print("\n✓ Transcribing audio...")
        success = transcribe_audio({"file_path": file}, args.output, device)
        if success:
            print(f"\n✓ Transcription of {file} successful!")
        else:
            print(f"\n✗ Transcription of {file} failed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
