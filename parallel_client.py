import asyncio
import websockets
import sounddevice as sd
import numpy as np
import io
import soundfile as sf
import json 


SERVER_URL = "ws://127.0.0.1:8000/ws"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 3 
 
async def send_audio():
    async with websockets.connect(SERVER_URL) as websocket:
        print("Connected to server...")
 
        loop = asyncio.get_running_loop()
 
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}", flush=True)
            buffer = io.BytesIO()
            sf.write(buffer, indata, SAMPLE_RATE, format='WAV')
 
            loop.run_in_executor(None, asyncio.run, websocket.send(buffer.getvalue()))
            
            
 
        device_info = sd.query_devices(kind="input")
        
 
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
            while True:
                response = await websocket.recv()
               
                
                try:
                    data = json.loads(response)
                    speechbrain_lang = data.get("speechbrain_language", "unknown")
                    speechbrain_conf = data.get("speechbrain_confidence", 0.0)
                    whisper_lang = data.get("whisper_language", "unknown")
                    whisper_conf = data.get("whisper_confidence", 0.0)
 
                    
                    print(f"Prediction [(SpeechBrain): {speechbrain_lang} (Confidence: {speechbrain_conf:.2f}) | (Whisper): {whisper_lang} (Confidence: {whisper_conf:.2f})]")
 
                except json.JSONDecodeError:
                    print(f"Invalid response from server: {response}")
                    
  
 
asyncio.run(send_audio())