# YouTube Shorts Automation

This project automates the process of converting local video files into YouTube Shorts with Indonesian subtitles using OpenAI Whisper and FFmpeg.

## Features

- Transcribes Indonesian speech using OpenAI Whisper
- Automatically generates karaoke-style subtitles
- Processes full videos into short segments suitable for YouTube Shorts
- Dockerized setup with support for docker-compose
- Input/output directories configurable via environment variables

## Directory Structure

.

├── youtube_to_shorts.py      # Main script for processing and transcription

├── requirements.txt          # Python dependencies

├── docker-compose.yml        # Compose file for service orchestration

├── Dockerfile                # Final image for running the app

├── service-account.json      # Google service account (if needed)

## Getting Started

1. Build the Base Image
```
   docker build -f Dockerfile.base -t whisper-base:latest .
```
2. Build the Full Application
```
   docker compose build
```
3. Run the Application
```
   docker compose run app
```
   By default, videos are read from /input and outputs are saved to /output.

## Environment Variables

| Variable     | Description                        | Default  |
|--------------|------------------------------------|----------|
| FFMPEG_PATH  | Path to ffmpeg binary              | ffmpeg   |
| INPUT_DIR    | Path to directory with input files | /input   |
| OUTPUT_DIR   | Path to output directory           | /output  |

## Dependencies

- Python 3.10+
- FFmpeg
- OpenAI Whisper
- spaCy (with multilingual model)

## Authors

Built by [fauj1x]
