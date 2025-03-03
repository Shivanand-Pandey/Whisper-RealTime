import asyncio
import torch
import numpy as np
import soundfile as sf
import faster_whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import io
import uvicorn
from collections import deque
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier

app = FastAPI()


device = "cuda" if torch.cuda.is_available() else "cpu"


whisper_model = faster_whisper.WhisperModel("large-v3", device=device, compute_type="float16")


speechbrain_model = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="/tmp"
)

# Constants
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  
FRAMES_PER_CHUNK = SAMPLE_RATE * CHUNK_DURATION
CONFIDENCE_THRESHOLD = 0.2 


ALLOWED_LANGUAGE = ["hindi", "tamil", "telugu", "marathi", "punjabi","english"]



client_buffers = {}

async def detect_with_speechbrain(audio_chunk):
    """Runs SpeechBrain language detection on the audio chunk."""
    audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0)
    emb = speechbrain_model.encode_batch(audio_tensor)
    out_prob = speechbrain_model.mods.classifier(emb).squeeze(1)
    probabilities = F.softmax(out_prob, dim=-1)
    top_value, top_index = torch.topk(probabilities, k=1)
    detected_lang = speechbrain_model.hparams.label_encoder.decode_torch(
        torch.tensor([top_index.item()])
    )[0].split(":")[-1].strip().lower()
    
    if detected_lang in ALLOWED_LANGUAGE or detected_lang == "english":
        return detected_lang, top_value.item()
    else:
        return "unknown", 0.0

async def detect_with_whisper(audio_chunk):
    """Runs Faster-Whisper language detection on the audio chunk."""
    _, info = whisper_model.transcribe(audio_chunk, beam_size=1, language=None)
    
    if info.language in ALLOWED_LANGUAGE or info.language == "en":
        return info.language, info.language_probability
    else:
        return "unknown", 0.0

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket Connection Request Received")
    await websocket.accept()
    print("Client Connected!")

    client_buffers[websocket] = deque()
    
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            audio_buffer = io.BytesIO(audio_bytes)
            audio, sample_rate = sf.read(audio_buffer, dtype='float32')
            
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)  
            
            client_buffers[websocket].extend(audio)
            
            if len(client_buffers[websocket]) >= FRAMES_PER_CHUNK:
                chunk_to_process = np.array(list(client_buffers[websocket]))
                client_buffers[websocket].clear()
                
                print("Running Language Detection...")
                
                
                if "english" in ALLOWED_LANGUAGE:
                    speechbrain_task = detect_with_speechbrain(chunk_to_process)
                    whisper_task = detect_with_whisper(chunk_to_process)
                    detected_speechbrain, sb_confidence = await speechbrain_task
                    detected_whisper, whisper_confidence = await whisper_task
                else:
                    detected_speechbrain, sb_confidence = await detect_with_speechbrain(chunk_to_process)
                    detected_whisper, whisper_confidence = "unknown", 0.0
                
                print(f"SpeechBrain Detected: {detected_speechbrain} (Confidence: {sb_confidence:.2f})")
                print(f"Whisper Detected: {detected_whisper} (Confidence: {whisper_confidence:.2f})")
                
                response = {}
                
                if sb_confidence > CONFIDENCE_THRESHOLD:
                    response["speechbrain_language"] = detected_speechbrain
                    response["speechbrain_confidence"] = round(sb_confidence, 2)
                else:
                    response["speechbrain_language"] = "unknown"
                    response["speechbrain_confidence"] = 0.0
                
                if whisper_confidence > CONFIDENCE_THRESHOLD:
                    response["whisper_language"] = detected_whisper
                    response["whisper_confidence"] = round(whisper_confidence, 2)
                else:
                    response["whisper_language"] = "unknown"
                    response["whisper_confidence"] = 0.0
                
                await websocket.send_json(response)
                
    except WebSocketDisconnect:
        print("WebSocket Disconnected")
        if websocket in client_buffers:
            del client_buffers[websocket]
    except Exception as e:
        print(f"Error: {e}")
        if websocket in client_buffers:
            del client_buffers[websocket]
    finally:
        await websocket.close()

if __name__ == "__main__":
    print("Server starting...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug", timeout_keep_alive=6000)
