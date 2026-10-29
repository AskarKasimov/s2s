"""Isolated real-network HTTP test. The fake is never used in production Compose."""

import base64
import io
import json
import struct
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx


def sample_wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(struct.pack("<h", 1200) * 2400)
    return buffer.getvalue()


class Fake(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["model"] == "qwen3-omni-speech"
        messages = body["messages"]
        sys_text = messages[0]["content"] if isinstance(messages[0].get("content"), str) else ""

        if body["modalities"] == ["text"]:
            if "json-объект" in sys_text.lower() or "transcript" in sys_text.lower():
                content = json.dumps({
                    "transcript": "Предельные издержки — это прирост издержек.",
                    "analysis": {
                        "verdict": "partial",
                        "answer_complete": False,
                        "rationale": "Определение верное, но краткое.",
                        "feedback": "Приведите пример.",
                    },
                }, ensure_ascii=False)
            elif "распознавания речи" in sys_text.lower():
                content = "Какого цвета ясное дневное небо?"
            else:
                content = "Четыре."
            result = {
                "id": "chatcmpl-fake123",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            }
        else:
            assert body["modalities"] == ["text", "audio"]
            audio_b64 = base64.b64encode(sample_wav_bytes()).decode("ascii")
            if "<tool_call>" in sys_text:
                text_out = (
                    "Хороший ответ! Давай уточним детали.\n"
                    '<tool_call>{"transcript": "Мой ответ", "feedback": {"verdict": "good"}}</tool_call>'
                )
            else:
                text_out = "Ответ заглушки."
            result = {
                "id": "chatcmpl-fakeaudio",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text_out}, "finish_reason": "stop"},
                    {"index": 1, "message": {"role": "assistant", "audio": {"data": audio_b64}}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
            }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())


def probe():
    with httpx.Client(base_url="http://api:8000", timeout=30) as client:
        for _ in range(100):
            try:
                if client.get("/health/ready").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("API did not become ready")

        assert client.get("/health/live").status_code == 200
        caps = client.get("/capabilities")
        assert caps.status_code == 200
        assert caps.headers.get("x-request-id")
        assert caps.json()["voices"] == ["ethan", "chelsie", "aiden"]

        data = sample_wav_bytes()

        # 1. /transcribe
        resp = client.post("/transcribe", files={"file": ("test.wav", data, "audio/wav")}, data={"language": "ru"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["text"] == "Какого цвета ясное дневное небо?"
        assert resp.json()["language"] == "ru"
        assert resp.json()["duration_seconds"] > 0

        # 2. /synthesize (wav & json)
        resp = client.post("/synthesize", json={"text": "Привет, студент!", "speaker": "ethan", "response_format": "wav"})
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/wav"

        resp = client.post("/synthesize", json={"text": "Привет, студент!", "speaker": "chelsie", "response_format": "json"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["audio"]["channels"] == 1
        assert resp.json()["audio"]["sample_rate"] == 24000

        # 3. /v1/chat/completions
        resp = client.post("/v1/chat/completions", json={
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "Сколько будет два плюс два?"}],
            "modalities": ["text"],
            "temperature": 0.0,
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "Четыре."

        # 4. /analyze-audio
        resp = client.post(
            "/analyze-audio",
            files={"file": ("test.wav", data, "audio/wav")},
            data={"prompt": "Оцени ответ по критериям.", "response_format": "json_object"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["transcript"] == "Предельные издержки — это прирост издержек."
        assert resp.json()["analysis"]["verdict"] == "partial"

        # 5. /speech-to-speech (default JSON + feedback + WAV)
        resp = client.post("/speech-to-speech", files={"file": ("test.wav", data, "audio/wav")})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["text"] == "Ответ заглушки."
        assert body["transcript"] == "Какого цвета ясное дневное небо?"
        assert body["feedback"] is None
        assert body["audio"]["channels"] == 1

        resp = client.post(
            "/speech-to-speech",
            files={"file": ("test.wav", data, "audio/wav")},
            data={"include_feedback": "true", "speaker": "aiden"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["text"] == "Хороший ответ! Давай уточним детали."
        assert resp.json()["feedback"] == {"verdict": "good"}

        resp = client.post(
            "/speech-to-speech",
            files={"file": ("test.wav", data, "audio/wav")},
            data={"response_format": "wav"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/wav"

        # 6. Error contract check
        bad = client.post("/transcribe", files={"file": ("bad.wav", b"broken", "audio/wav")})
        assert bad.status_code == 422
        assert bad.json()["error"]["code"] == "INVALID_AUDIO"
        assert bad.json()["request_id"]

        assert "speech_requests_total" in client.get("/metrics").text
        assert client.get("/docs").status_code == 200
        print("PASS: full Qwen3-Omni gateway contract verified over real HTTP.")


if __name__ == "__main__":
    if sys.argv[1] == "fake":
        HTTPServer(("0.0.0.0", 8091), Fake).serve_forever()
    else:
        probe()
