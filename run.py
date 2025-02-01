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

    Args:
        input_path (str): Path to input audio file
        output_path (str, optional): Output path relative to dist directory
        device (str, optional): Device to use for transcription
        torch_dtype (torch.dtype, optional): Torch data type to use for transcription

    Returns:
        bool: True if transcription succeeded, False otherwise
    """
    try:
        print(f"\nTranscribing audio: {os.path.basename(input_path)}...")

        # Create dist directory if not exists
        os.makedirs("dist", exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_dtype is None:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # Primeiro carregar o processador para obter a taxa de amostragem correta
        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")

        # Carregar áudio com reamostragem para 16kHz usando soundfile
        audio_array, original_sr = sf.read(input_path)

        # Reamostrar para 16kHz se necessário
        if original_sr != processor.feature_extractor.sampling_rate:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=original_sr,
                target_sr=processor.feature_extractor.sampling_rate,
            )
            sampling_rate = processor.feature_extractor.sampling_rate
        else:
            sampling_rate = original_sr

        # Verificar e converter para mono se necessário
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3-turbo",
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation="sdpa" if device == "cuda" else "eager",
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        # Processar o áudio
        inputs = processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=False,
            padding="longest",
        ).to(device=device, dtype=torch_dtype)

        # Configurações de geração atualizadas
        generate_kwargs = {
            "language": "pt",
            "task": "transcribe",  # Adicionar especificação explícita da tarefa
            "return_timestamps": False,  # Remover timestamps se não for necessário
            "no_speech_threshold": 0.6,
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        }

        result = model.generate(inputs.input_features, **generate_kwargs)

        # Decodificar com o processador
        transcription = processor.batch_decode(
            result,
            skip_special_tokens=True,
            decode_with_timestamps=False,  # Desativar se não usar timestamps
        )[0]

        # Set output path
        if output_path:
            output_path = os.path.join("dist", output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join("dist", f"{base_name}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcription)

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
