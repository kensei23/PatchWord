import threading
import time
from typing import Protocol

SAMPLE_RATE = 16000

class Checker(Protocol):
    def check(self, audio_path, text): ...

class WordScore:
    def __init__(self, word, start, end, score):
        self.word = word
        self.start = start
        self.end = end
        self.score = score


class CheckResult:
    def __init__(self, duration_seconds, elapsed_seconds, transcript, words):
        self.duration_seconds = duration_seconds
        self.elapsed_seconds = elapsed_seconds
        self.transcript = transcript
        self.words = words


class WhisperXChecker:
    def __init__(self, whisper_model="small", device="cpu", language="en"):
        import whisperx
        self.whisperx = whisperx
        self.lock = threading.Lock()
        self.device = device
        self.language = language
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=self.language,
            device=self.device,
        )

        compute_type = "int8" if self.device == "cpu" else "float16"
        self.asr_model = whisperx.load_model(
            whisper_model,
            self.device,
            compute_type=compute_type,
            language=self.language,
        )

    def check(self, audio_path, text):
        with self.lock:
            start_time = time.time()

            audio = self.whisperx.load_audio(audio_path)
            duration = len(audio) / SAMPLE_RATE

            segments = [{"text": text, "start": 0.0, "end": duration}]
            aligned = self.whisperx.align(
                segments,
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            # Transcribe what narrator said
            heard = self.asr_model.transcribe(audio, batch_size=8, language=self.language)
            transcript_parts = []
            for segment in heard["segments"]:
                transcript_parts.append(segment["text"].strip())
            transcript = " ".join(transcript_parts)

            # Word-level alignment scores
            words = []
            for w in aligned["word_segments"]:
                start = round(float(w["start"]), 3) if w.get("start") is not None else None
                end = round(float(w["end"]), 3) if w.get("end") is not None else None
                score = round(float(w["score"]), 3) if w.get("score") is not None else None
                words.append(WordScore(w["word"], start, end, score))

            elapsed = round(time.time() - start_time, 3)
            return CheckResult(round(duration, 3), elapsed, transcript, words)