from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


import uuid
import edge_tts
from pydub import AudioSegment
import os


app = FastAPI()

class TextInput(BaseModel):
    text: str

def split_text(text, max_words=50):
    words = text.split()
    for i in range(0, len(words), max_words):   
        yield ' '.join(words[i:i + max_words])
        
        
ALLOWED_SPECIAL_CHARS = {".", ",", "!", ":", ";", "-"}

SPOKEN_SPECIALS = {
    "!": " exclamation mark ",
    "@": " at ",
    "&": " and ",
    "%": " percent ",
    "$": " dollar "
}

model = WhisperModel("small", compute_type="int8")

def clean_text(text: str, allowed_chars: set, spoken_map: dict) -> str:
    result = []
    for char in text:
        if char in spoken_map:
            result.append(spoken_map[char])
        elif char.isalnum() or char.isspace() or char in allowed_chars:
            result.append(char)
        # Else: remove it
    return ''.join(result)

@app.post("/tts/")
async def tts_api(item: TextInput):
    static_voice = "en-US-JennyNeural"
    base_filename = f"temp_{uuid.uuid4().hex}"
    temp_audio_files = []

    # Clean the text first
    cleaned_text = clean_text(item.text, ALLOWED_SPECIAL_CHARS, SPOKEN_SPECIALS)

    # Generate audio for each chunk
    for idx, chunk in enumerate(split_text(cleaned_text)):
        chunk_filename = f"{base_filename}_{idx}.mp3"
        communicate = edge_tts.Communicate(text=chunk, voice=static_voice)
        await communicate.save(chunk_filename)
        temp_audio_files.append(chunk_filename)

    # Combine audio chunks into one file
    combined = AudioSegment.empty()
    for fname in temp_audio_files:
        combined += AudioSegment.from_file(fname)

    final_filename = f"{base_filename}_final.mp3"
    combined.export(final_filename, format="mp3")

    # Clean up temp chunk files
    for f in temp_audio_files:
        os.remove(f)

    return FileResponse(
        final_filename,
        media_type="audio/mpeg",
        filename="output.mp3",
        background=None
    )


@app.post("/stt/")
async def speech_to_text(audio: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.mp3"
    
    with open(temp_filename, "wb") as f:
        f.write(await audio.read())

    segments, info = model.transcribe(temp_filename)
    full_text = " ".join([segment.text for segment in segments])

    os.remove(temp_filename)

    return {"transcription": full_text}


@app.get("/record", response_class=HTMLResponse)
async def get_recorder():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Audio Recorder</title>
</head>
<body>
    <h3>Hold to Record</h3>
    <button id="record-btn">🎙️ Hold to Record</button>
    <p id="result"></p>

    <script>
        let mediaRecorder;
        let chunks = [];

        const btn = document.getElementById("record-btn");
        const result = document.getElementById("result");

        btn.onmousedown = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            chunks = [];

            mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
            mediaRecorder.onstop = async () => {
                const blob = new Blob(chunks, { type: "audio/webm" });
                const formData = new FormData();
                formData.append("audio", blob, "recording.webm");

                const res = await fetch("/stt/", {
                    method: "POST",
                    body: formData
                });

                const data = await res.json();
                result.innerText = "📝 " + data.transcription;
            };

            mediaRecorder.start();
        };

        btn.onmouseup = () => mediaRecorder.stop();
    </script>
</body>
</html>
"""
