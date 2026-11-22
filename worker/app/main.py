import os
import shutil
import tempfile
from pathlib import Path
from functools import lru_cache
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from app.checker import WhisperXChecker
from app.models import CheckResponse, WordResult

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB limit
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

app = FastAPI(title="PatchWord checker worker")

@lru_cache
def get_checker():
    # built on the first request, so importing this file (tests, CI) doesn't load any models
    return WhisperXChecker(
        whisper_model=os.getenv("WHISPER_MODEL", "small"),
        device=os.getenv("WORKER_DEVICE", "cpu"),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check", response_model=CheckResponse)
def check(audio: UploadFile = File(...), text: str = Form(...), checker=Depends(get_checker)):
    reference = text.strip()
    if not reference:
        raise HTTPException(status_code=422, detail="text must not be empty")

    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".wav"

    with tempfile.TemporaryDirectory() as tmp:
        temp_path = os.path.join(tmp, "upload" + suffix)

        with open(temp_path, "wb") as out_file:
            shutil.copyfileobj(audio.file, out_file)

        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise HTTPException(status_code=422, detail="audio file is empty")
        if file_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="audio file is too large")

        result = checker.check(temp_path, reference)

    word_results = []
    for w in result.words:
        word_results.append(
            WordResult(word=w.word, start=w.start, end=w.end, score=w.score)
        )

    return CheckResponse(
        duration_seconds=result.duration_seconds,
        elapsed_seconds=result.elapsed_seconds,
        transcript=result.transcript,
        words=word_results,
    )