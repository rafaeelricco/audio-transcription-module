# Audio Transcription Module

### Audio Transcription Module (`/audio_transcription`)

- `run.py` - Main transcription executor
- `ai_transcript_processor.py` - AI-powered text organization
- `ui.py` - Progress tracking interface
- `activate.ps1` - Environment setup script
- `assets/` - Sample audio files directory
- `dist/` - Generated transcriptions output

## Prerequisites

```bash
# Audio processing dependencies
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux
# Windows: Automatically installed by activate.ps1

# Environment setup
.\activate.ps1  # Windows PowerShell setup
```

## Key Features

- **Advanced Audio Transcription**
  - Whisper large-v3-turbo model
  - Automatic audio resampling (16kHz)
  - Stereo-to-mono conversion
  - GPU acceleration support (CUDA)
  - Batch processing capabilities
  - Real-time progress tracking
  - AI-powered text organization

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

# Force GPU usage
python run.py --audio input.mp3 --device gpu
```

## Command Arguments

| Argument | Description               | Example Value      | Notes                       |
|----------|---------------------------|-------------------|----------------------------|
| --audio  | Input audio file path    | *.mp3            | Supports common formats    |
| --output | Custom output path       | results/output.txt| Relative to dist directory|
| --device | Processing device        | cpu/gpu          | Auto-detects GPU by default|

## Output Files

| File Pattern           | Description                 |
|-----------------------|----------------------------|
| {input_name}_raw.txt  | Raw transcription output   |
| {input_name}_organized.txt | AI-processed text (optional) |

## AI Text Processing

The module includes an AI-powered text processor that:
- Organizes content into thematic sections
- Adds markdown formatting
- Creates topic flowcharts
- Corrects grammar and typos
- Highlights technical terms
- Requires OpenRouter API key (set via OPENROUTER_API_KEY)

## Error Handling

- FFmpeg installation verification
- CUDA/GPU availability checks
- Automatic PyTorch CUDA installation
- Audio processing error recovery
- AI processing fallback options

## Performance Features

- Real-time progress tracking
- GPU memory optimization
- Automatic device selection
- Batch processing support
- Progress visualization

## Future Improvements

- [ ] Real-time microphone input
- [ ] Speaker diarization
- [ ] Multi-language support toggle
- [ ] Audio preprocessing filters
  - [ ] Noise reduction
  - [ ] Voice activity detection
  - [ ] Automatic gain control
