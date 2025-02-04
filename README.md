# Real-time Speech Transcription using WebSockets

This project provides a real-time speech-to-text transcription system using Faster Whisper, WebSockets, and FastAPI for speech recognition. It consists of a Python-based WebSocket server and a client that captures audio from a microphone and sends it to the server for transcription.

## Features
- **Real-time Streaming**: Streams audio to the server in small chunks.
- **Language Detection**: Automatically detects the spoken language.
- **Confidence Scoring**: Provides confidence scores for detected languages.
- **Whisper AI Integration**: Uses Faster Whisper for speech-to-text conversion.
- **WebSocket Communication**: Low-latency streaming between client and server.
- **Multi-Device Support**: Can be run on different devices with GPU acceleration (if available).

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed on your system. You also need `ffmpeg` installed for audio processing.

### Clone the Repository
```sh
git clone https://github.com/Shivanand-Pandey/your-repo.git
cd your-repo
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Running the Server
The server uses FastAPI with WebSockets to process incoming audio data.

```sh
python server.py
```

The server will start on `127.0.0.1:8000/ws`.

## Running the Client
The client captures real-time audio and sends it to the server for transcription.

```sh
python client.py
```

## How It Works
1. The client records microphone audio and sends small chunks (0.5s) to the WebSocket server.
2. The server accumulates 5-second audio chunks and transcribes them using Faster Whisper.
3. The detected language, confidence score, and transcribed text are sent back to the client.
4. The client displays the real-time transcriptions on the console.

## Expected Output
When running the client, you will see output similar to this:
```
✅ Connected to server...
🎤 Recording audio...
🌍 Detected Language: English (Confidence: 0.97)
📜 Transcription: Hello, this is a test of the real-time speech-to-text system.

🌍 Detected Language: English (Confidence: 0.98)
📜 Transcription: It is able to process streaming audio and return accurate transcriptions.
```
If the language detected changes, the output will reflect that along with the updated transcription.

## Technologies Used
- **Python** (FastAPI, asyncio, sounddevice, soundfile)
- **Faster Whisper** (for speech-to-text transcription)
- **WebSockets** (for real-time audio streaming)
- **Torch** (for running Whisper models on CPU/GPU)

## Customization
- Modify `CHUNK_DURATION` in `server.py` to change the transcription chunk size.
- Adjust `SAMPLE_RATE` in `client.py` to match your microphone’s settings.
- Use a different Whisper model (`tiny`, `medium`, `large-v2`, `large-v3`) in `server.py`.
- Change the `beam_size` in `server.py` to adjust transcription accuracy vs. speed.

## Troubleshooting
- If the client fails to connect, ensure the server is running on `127.0.0.1:8000`.
- If transcriptions are inaccurate, try using a higher-quality microphone.
- For better performance, use a GPU-supported environment with `torch.cuda` enabled.

## Acknowledgments
- [OpenAI Whisper](https://github.com/openai/whisper)
- [FastAPI](https://fastapi.tiangolo.com/)
- [WebSockets](https://websockets.readthedocs.io/)

## License
This project is licensed under the MIT License.

---
🚀 **Developed by Shivanand Pandey**

