# Voice Test Project

This is a text-to-speech project that uses F5-TTS to generate audio from text. The project supports multiple characters with different voice configurations.

## Purpose

This project aims to address the lack of a streaming mode in F5-TTS's HTTP requests. It借鉴es the streaming approach from the socket_server in the F5-TTS project. However, it still requires waiting for the first chunk to be processed before audio can be returned, so the initial audio latency cannot be avoided (1-3 seconds).

## Prerequisites

Before running this project, you must have the F5-TTS project installed and configured properly. Please follow the installation instructions at:
https://github.com/SWivid/F5-TTS

Make sure you can run F5-TTS examples successfully before proceeding with this project setup.

## Dependencies

Before running the API server, you need to install the following dependencies:

```bash
pip install torch numpy soundfile sounddevice transformers librosa pydub
```

Additionally, you may need to install system-level dependencies for audio processing:

- **Windows**: Usually satisfied automatically with pip installation
- **macOS**: `brew install portaudio` (for sounddevice)
- **Linux**: `sudo apt-get install portaudio19-dev` (for sounddevice)

## Setup Instructions

### 1. Initial Setup
1. Clone the repository
2. Install dependencies as per project requirements
3. Make sure your F5-TTS installation is working properly

### 2. Configure Characters
1. Copy the `characters.json.template` file to create a new file named `characters.json`:
   ```bash
   cp characters.json.template characters.json
   ```
2. Modify the `characters.json` file to update the audio file paths and reference text as needed for your own audio files
3. Ensure the audio files referenced in the configuration exist in the specified paths

### 3. Audio Files
- Add your reference audio files to your `path/to/audio` directory
- Supported formats: WAV, MP3 (WAV recommended for best quality)
- Recommended length: 5-10 seconds of clear speech
- The reference text should match the content of the audio file

## Configuration File Structure

The `characters.json` file contains character definitions in the following format:

```json
{
    "character_name": {
        "ref_file": "path/to/audio/file.mp3",
        "ref_text": "Text that matches the reference audio content"
    }
}
```

- `ref_file`: Path to the reference audio file
- `ref_text`: Text content that matches the reference audio

## Usage

Run the API server:

```bash
python api_server.py
```

The API server provides endpoints for:
- Streaming audio generation: `/tts/stream`
- Full file generation: `/tts`
- Listing available characters: `/characters`
- Health check: `/health`

## API Endpoints

- `POST /tts/stream` - Stream generated audio
- `POST /tts` - Generate complete audio file
- `GET /characters` - List available characters
- `GET /health` - Check server health

## API Usage Examples

### 1. Streaming Audio Generation (`/tts/stream`)

Generate and stream audio in real-time:

```bash
curl -X POST http://localhost:8080/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the voice synthesis system.",
    "character": "default"
  }' \
  --output output.wav
```

### 2. Full Audio File Generation (`/tts`)

Generate a complete audio file:

```bash
curl -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the voice synthesis system.",
    "character": "default"
  }' \
  --output output.wav
```

### 3. List Available Characters (`/characters`)

Get a list of all available characters:

```bash
curl -X GET http://localhost:8080/characters
```

### 4. Health Check (`/health`)

Check if the server is running:

```bash
curl -X GET http://localhost:8080/health
```

## Notes

1. The `characters.json` file is ignored by Git to protect personal configuration
2. Make sure audio file paths in the configuration are correct
3. Reference text should match the content of the reference audio
4. The configuration is loaded dynamically, so changes to `characters.json` take effect without restarting the server