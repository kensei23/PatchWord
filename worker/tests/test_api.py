from fastapi.testclient import TestClient

from app.checker import CheckResult, WordScore
from app.main import app, get_checker


class StubChecker:
    def check(self, audio_path: str, text: str) -> CheckResult:
        words = []
        for word in text.split():
            words.append(WordScore(word, 0.0, 0.5, 0.9))
        return CheckResult(1.0, 0.1, text, words)

def get_stub_checker():
    return StubChecker()

def make_client():
    app.dependency_overrides[get_checker] = get_stub_checker
    return TestClient(app)


def test_health():
    assert make_client().get("/health").json() == {"status": "ok"}


def test_check_returns_a_score_per_word_and_a_transcript():
    response = make_client().post(
        "/check",
        files={"audio": ("clip.wav", b"not really audio", "audio/wav")},
        data={"text": "the quick brown fox"},
    )
    assert response.status_code == 200
    body = response.json()
    assert [w["word"] for w in body["words"]] == ["the", "quick", "brown", "fox"]
    assert body["words"][0]["score"] == 0.9
    assert body["transcript"] == "the quick brown fox"
    assert body["durationSeconds"] == 1.0


def test_blank_text_is_rejected():
    response = make_client().post(
        "/check",
        files={"audio": ("clip.wav", b"data", "audio/wav")},
        data={"text": "   "},
    )
    assert response.status_code == 422


def test_empty_audio_is_rejected():
    response = make_client().post(
        "/check",
        files={"audio": ("clip.wav", b"", "audio/wav")},
        data={"text": "hello there"},
    )
    assert response.status_code == 422


def test_missing_audio_is_rejected():
    assert make_client().post("/check", data={"text": "hello"}).status_code == 422
