"""
            _                      
   _____   (_)  _____  _____  ____ 
  / ___/  / /  / ___/ / ___/ / __ \
 / /     / /  / /__  / /__  / /_/ /
/_/     /_/   \___/  \___/  \____/ 
                                   
© r1cco.com

Audio Transcription Module

Key features:
- Audio-to-text transcription using Whisper large-v3-turbo
- Automatic output organization in 'dist' directory
- GPU acceleration support
- Batch processing capabilities

Basic usage:
  # Default output to dist/input_name.txt
  python run.py --audio assets/meeting.mp3

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
from datasets import Audio
import soundfile as sf
import librosa


def transcribe_audio(input_path, output_path=None, device=None, torch_dtype=None):
    """
    Transcribe audio file to text using Whisper large-v3-turbo model.
    """
    try:
        print(f"\nTranscribing audio: {os.path.basename(input_path)}...")

        # Create dist directory if not exists
        os.makedirs("dist", exist_ok=True)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Carregar modelo e processador
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

        # Criar pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,  # chunk length in seconds
            batch_size=16,  # batch size for processing chunks
            torch_dtype=torch_dtype,
            device=device,
        )

        # Configurações de geração
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

        # Realizar a transcrição
        result = pipe(input_path, generate_kwargs=generate_kwargs)

        # Set output path
        if output_path:
            output_path = os.path.join("dist", output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join("dist", f"{base_name}.txt")

        # Salvar resultado
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"✓ Transcription saved to: {output_path}")
        return True

    except Exception as e:
        print(f"✗ Transcription error: {str(e)}")
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
        if sys.platform == "darwin":
            print("Install on macOS with:\n  brew install ffmpeg")
        else:
            print("Download from: https://ffmpeg.org/download.html")
        sys.exit(1)


def main():
    """Main transcription execution flow"""
    parser = ArgumentParser(description="Audio transcription processor")
    parser.add_argument("--audio", required=True, help="Input audio file(s)")
    parser.add_argument("--output", help="Output path (relative to dist directory)")
    args = parser.parse_args()

    # Check ffmpeg first
    check_ffmpeg_installation()

    # Device selection
    print("\nSelect processing device:")
    print("1. CPU (recommended if no NVIDIA GPU)")
    print("2. GPU (faster but requires CUDA)")

    choice = input("Your choice [1/2]: ").strip()
    use_gpu = False

    if choice == "2":
        if torch.cuda.is_available():
            use_gpu = True
            print("Using GPU acceleration 🚀")
        else:
            print("GPU not available, falling back to CPU ⚠️")

    # Pass configuration to transcription
    success = transcribe_audio(
        args.audio,
        args.output,
        device="cuda" if use_gpu else "cpu",
        torch_dtype=torch.float16 if use_gpu else torch.float32,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
