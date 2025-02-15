### Project Structure
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

# For YouTube video downloads:
# Recommended: Use YouTube Video Downloader (@https://transkriptor.com/pt-br/downloader-de-video-do-youtube/)
# Ensure compliance with copyright laws when using third-party content
```

## Supported Formats
**Audio Formats**  
MP3, WAV, FLAC, OGG, AAC, M4A  

**Video Formats**  
MP4, MOV, AVI, MKV, WMV  

*Note: All video files will have audio automatically extracted*

## Key Features

- **Advanced Audio Transcription**
  - Whisper large-v3-turbo model
  - Automatic audio resampling (16kHz)
  - Stereo-to-mono conversion
  - GPU acceleration support (CUDA)
  - Batch processing capabilities
  - Real-time progress tracking
  - AI-powered text organization
- **Extended Format Support**
  - 15+ audio/video formats supported
  - Automatic format conversion
  - Embedded audio extraction from video files
  - Variable bitrate handling

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

### Enhanced Basic Usage
```bash
# Transcribe video files
python run.py --audio assets/presentation.mp4

# Multiple file types in batch
python run.py --audio inputs/*.mp3 inputs/*.mov

# Directory processing 
python run.py --audio recordings/2024-07/ --output processed_transcripts/
```

### Advanced Options

```bash
# Force CPU usage
python run.py --audio input.mp3 --device cpu

# Force GPU usage
python run.py --audio input.mp3 --device gpu
```

### New Advanced Options
```bash
# Specify transcript language
python run.py --audio interview.mp3 --language spanish

# Set minimum confidence threshold (0.1-1.0)
python run.py --audio lecture.mp3 --confidence 0.8

# Process while keeping original timestamps
python run.py --audio meeting.mp3 --keep-timestamps
```

## Updated Command Arguments
| Argument       | Description               | Example Values       |
|----------------|---------------------------|----------------------|
| --audio        | Input path(s)             | *.mp3, *.mp4, dir/  |
| --output       | Custom output path       | results/final.txt   |
| --device       | Processing device        | cpu/gpu             |
| --language     | Transcription language    | english, japanese   | 
| --confidence   | Minimum confidence       | 0.5 (50% confidence)|

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
- Uses Claude 3.5 Sonnet as default model
- Supports custom processing templates
- Generates interactive HTML reports
- Adds automatic section numbering
- Creates summary bullet points
- Formats code blocks properly

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
- Automatic format detection
- Parallel file processing
- Smart memory management
- Failed file retry system
- Output validation checks

## Future Improvements

- [ ] Real-time microphone input
- [ ] Speaker diarization
- [ ] Multi-language support toggle
- [ ] Audio preprocessing filters
  - [ ] Noise reduction
  - [ ] Voice activity detection
  - [ ] Automatic gain control
