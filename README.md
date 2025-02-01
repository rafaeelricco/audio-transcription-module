# Audio Transcription Module

### Audio Transcription Module (`/audio_transcription`)

- `run.py` - Main transcription executor
- `assets/` - Sample audio files directory
- `dist/` - Generated transcriptions output

## Prerequisites

```bash
# Audio processing dependencies
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux

# Python packages
torch>=2.0.1
transformers>=4.37.0
soundfile>=0.12.1
librosa>=0.10.0
datasets>=2.14.0
```

## Key Features

- **Advanced Audio Transcription**
  - Whisper large-v3-turbo model
  - Automatic audio resampling (16kHz)
  - Stereo-to-mono conversion
  - GPU acceleration support
  - Batch processing capabilities

## Command Reference

### Basic Usage

```bash
# Single file transcription
python run.py --audio assets/meeting.mp3

# Custom output path
python run.py --audio assets/interview.mp3 --output transcripts/interview.txt

# Batch processing
python run.py --audio assets/lectures/*.mp3 --output university/lectures/
```

### Advanced Options

```bash
# Force CPU usage
python run.py --audio input.mp3 --device cpu

# High-precision mode (FP32)
python run.py --audio input.mp3 --precision float32

# Long-form audio processing
python run.py --audio long_recording.mp3 --chunk_length 30
```

## Command Arguments

| Argument       | Description                   | Example Value      | Notes                            |
| -------------- | ----------------------------- | ------------------ | -------------------------------- |
| --audio        | Input audio file path/pattern | \*.mp3             | Supports common audio formats    |
| --output       | Custom output path            | results/output.txt | Relative to dist directory       |
| --device       | Processing device             | cuda/cpu           | Auto-detects GPU by default      |
| --precision    | Computation precision         | float16/float32    | float16 for faster GPU inference |
| --chunk_length | Audio chunk length (seconds)  | 30                 | For long audio processing        |

## Output Files

| File Pattern     | Description                   |
| ---------------- | ----------------------------- |
| {input_name}.txt | Transcribed text output       |
| processing.log   | Transcription metrics         |
| error_logs.txt   | Failed transcription attempts |

## Error Handling

- FFmpeg installation issues
- Audio file corruption
- CUDA out-of-memory errors
- Audio resampling failures
- Model loading errors

## Performance Monitoring

- Real-time transcription progress
- GPU memory utilization
- Audio processing latency
- Model inference speed

## Future Improvements

- [ ] Real-time microphone input support
- [ ] Speaker diarization capabilities
- [ ] Automated punctuation correction
- [ ] Multi-language support toggle
- [ ] Audio preprocessing optimization
  - [ ] Noise reduction filters
  - [ ] Voice activity detection
  - [ ] Automatic gain control
