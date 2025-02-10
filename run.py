"""                                
© r1cco.com

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
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from argparse import ArgumentParser
import warnings
from transformers import logging as transformers_logging
from ui import ProgressBar


def transcribe_audio(input_path, output_path=None, device=None, torch_dtype=None, model="anthropic/claude-3.5-sonnet"):
    """
    Transcribe audio file to text using Whisper large-v3-turbo model.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    transformers_logging.set_verbosity_error()
    progress = ProgressBar()

    try:
        progress.simulate_progress("Loading model...", start_from=0, until=40)

        os.makedirs("dist", exist_ok=True)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3-turbo"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)
        progress.update("Loading model", 100)

        progress.simulate_progress("Initializing pipeline...", start_from=0, until=70)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
        )
        progress.update("Initializing pipeline", 100)

        progress.simulate_progress("Transcribing audio...", start_from=0, until=90)
        generate_kwargs = {
            "task": "transcribe",
            "language": "portuguese",
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": True,
        }

        result = pipe(input_path, generate_kwargs=generate_kwargs)
        progress.update("Transcribing audio", 100)

        print("\nTranscription complete. What would you like to do?")
        print("1. Save raw transcription")
        print("2. Process and organize text before saving")
        choice = input("Your choice [1/2]: ").strip()

        if choice == "2":
            progress.simulate_progress("Processing text with AI...", start_from=0, until=90)
            from ai_transcript_processor import process_text
            try:
                organized_text = process_text(result["text"])
                ai_success = True
            except Exception as e:
                error_info = e.args[0] if hasattr(e, 'args') and isinstance(e.args[0], dict) else {
                    "type": "Unknown Error",
                    "message": str(e)
                }
                print(f"\n❌ AI Processing Failed")
                print(f"Type: {error_info['type']}")
                print(f"Message: {error_info['message']}")
                print("\nFalling back to raw transcription only.")
                organized_text = None
                ai_success = False
            
            if not organized_text:
                print("AI processing failed. Only raw transcription will be saved.")
                ai_success = False
        else:
            ai_success = True 

        progress.update("Processing text with AI", 100)

        progress.simulate_progress("Saving transcription...", start_from=0, until=90)
        
        if output_path:
            base_output_path = os.path.join("dist", output_path)
            output_dir = os.path.dirname(base_output_path)
        else:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = os.path.join("dist", base_name)
            base_output_path = os.path.join(output_dir, f"{base_name}")

        os.makedirs(output_dir, exist_ok=True)

        raw_path = f"{base_output_path}_raw.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        if choice == "2":
            if ai_success:
                organized_path = f"{base_output_path}_organized.txt"
                with open(organized_path, "w", encoding="utf-8") as f:
                    f.write(organized_text)
                print(f"\n✓ Raw transcription saved to: {raw_path}")
                print(f"✓ Organized transcription saved to: {organized_path}")
            else:
                print(f"\n✓ Raw transcription saved to: {raw_path}")
                print("✗ Organized version not saved due to processing errors")
                return False  # Indicate failure to main program
        else:
            print(f"\n✓ Transcription saved to: {raw_path}")

        progress.update("Saving transcription", 100)
        return True

    except Exception as e:
        print(f"\n✗ Transcription error: {str(e)}")
        return False


def check_ffmpeg_installation():
    """Checks ffmpeg installation and provides instructions if missing"""
    try:
        import subprocess

        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\nError: ffmpeg is required but not installed.")
        print("Install on macOS with:\n  brew install ffmpeg")
        sys.exit(1)


def check_and_install_cuda():
    """
    Checks CUDA support and installs PyTorch with CUDA if needed.
    Returns True if CUDA is available or installation was successful.
    """
    if not torch.cuda.is_available():
        print("\n⚠️ GPU not detected. Attempting to reinstall PyTorch with CUDA support...")
        try:
            import subprocess
            import platform

            # Add platform-specific PyTorch installation command
            if platform.system() == "Windows":
                torch_command = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu121",
                    "--no-cache-dir"
                ]
            else:
                torch_command = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu121"
                ]

            # Clean uninstall first
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=True)
            
            # Install with platform-specific command
            subprocess.run(torch_command, check=True)
            
            print("\n✓ PyTorch reinstalled with CUDA support")
            print("🔄 Please restart the script to apply changes")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error installing PyTorch: {str(e)}")
            return False
    return True


def main():
    """Main transcription execution flow"""
    print("\nPyTorch CUDA Diagnostics:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        if not check_and_install_cuda():
            print("\n⚠️ Warning: PyTorch is not detecting the GPU. Please verify:")
            print("1. PyTorch is installed with CUDA support")
            print("2. CUDA version is compatible with your drivers")
            print("\nTo install PyTorch with CUDA 12.1 support, run:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    parser = ArgumentParser(description="Audio transcription processor")
    parser.add_argument("--audio", required=True, help="Input audio file(s)")
    parser.add_argument("--output", help="Output path (relative to dist directory)")
    parser.add_argument(
        "--device", choices=["cpu", "gpu"], help="Specify processing device (cpu/gpu)"
    )
    args = parser.parse_args()

    check_ffmpeg_installation()

    use_gpu = False

    if args.device:
        if args.device == "gpu":
            if torch.cuda.is_available():
                use_gpu = True
                print("Using GPU acceleration 🚀")
            else:
                print("GPU not available, falling back to CPU ⚠️")
        else:
            print("Using CPU for processing")
    else:
        print("\nSelect processing device:")
        print("1. CPU (recommended if no NVIDIA GPU)")
        print("2. GPU (faster but requires CUDA)")

        choice = input("Your choice [1/2]: ").strip()
        if choice == "2":
            if torch.cuda.is_available():
                use_gpu = True
                print("Using GPU acceleration 🚀")
            else:
                print("GPU not available, falling back to CPU ⚠️")

    success = transcribe_audio(
        args.audio,
        args.output,
        device="cuda" if use_gpu else "cpu",
        torch_dtype=torch.float16 if use_gpu else torch.float32,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
