# Checker worker

Small FastAPI service around WhisperX. The Spring Boot API sends it a recording and the book text, and it
returns a score and timestamps for every word plus a transcript of what it heard.

## Run it locally (Windows, Python 3.12)

    py -3.12 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt -r requirements-ml.txt
    uvicorn app.main:app --port 8000

Try it with a clip of your own voice:

    curl.exe -F "audio=@clip.wav" -F "text=The words you read go here." http://localhost:8000/check

## Tests

The tests use a stub checker, so they don't need torch or any model downloads.

    pip install -r requirements-dev.txt
    pytest

## Contract

`POST /check` takes multipart form data with `audio` (file) and `text` (string). It returns:

    {
      "durationSeconds": 14.2,
      "elapsedSeconds": 4.6,
      "transcript": "what the model heard",
      "words": [{"word": "The", "start": 0.12, "end": 0.31, "score": 0.91}]
    }

`start`, `end` and `score` can be null for a word the aligner couldn't place. `GET /health` returns `{"status": "ok"}`.
