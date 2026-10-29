"""Run after the direct-model self test, through the real HTTP gateway API."""

import os
from pathlib import Path

import httpx

from client import validate_wav


def main():
    with httpx.Client(base_url="http://api:8000", timeout=900) as client:
        live = client.get("/health/live")
        live.raise_for_status()
        assert live.headers.get("x-request-id")

        ready = client.get("/health/ready")
        ready.raise_for_status()

        caps = client.get("/capabilities")
        caps.raise_for_status()
        assert "ethan" in caps.json()["voices"]

        # 1. OpenAI-compatible /v1/chat/completions
        chat_resp = client.post("/v1/chat/completions", json={
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "Сколько будет два плюс два? Ответь кратко."}],
            "modalities": ["text"],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": False,
        })
        chat_resp.raise_for_status()
        chat_data = chat_resp.json()
        text = chat_data["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Текстовая ручка /v1/chat/completions вернула некорректный ответ.")
        Path("/output/api-chat-answer.txt").write_text(text, encoding="utf-8")
        print("HTTP /v1/chat/completions: ответ сохранён в output/api-chat-answer.txt.")

        question = Path("/output/test-question.wav").read_bytes()

        # 2. /transcribe (STT)
        stt_resp = client.post(
            "/transcribe",
            files={"file": ("question.wav", question, "audio/wav")},
            data={"language": "ru"},
        )
        stt_resp.raise_for_status()
        stt_data = stt_resp.json()
        if not isinstance(stt_data.get("text"), str) or not stt_data["text"].strip():
            raise RuntimeError("Ручка /transcribe вернула пустой текст.")
        Path("/output/api-transcript.txt").write_text(stt_data["text"], encoding="utf-8")
        print("HTTP /transcribe: расшифровка сохранена в output/api-transcript.txt.")

        # 3. /speech-to-speech (WAV mode)
        s2s_resp = client.post(
            "/speech-to-speech",
            files={"file": ("question.wav", question, "audio/wav")},
            data={"response_format": "wav", "speaker": "ethan"},
        )
        s2s_resp.raise_for_status()
        validate_wav(s2s_resp.content)
        Path("/output/api-answer.wav").write_bytes(s2s_resp.content)

    uid, gid = os.getenv("HOST_UID"), os.getenv("HOST_GID")
    if uid and gid:
        uid_int, gid_int = int(uid), int(gid)
        for item in [Path("/output"), *Path("/output").rglob("*")]:
            try:
                os.chown(item, uid_int, gid_int, follow_symlinks=False)
            except OSError:
                pass
    print("HTTP API: реальный WAV-ответ сохранён в output/api-answer.wav.")


if __name__ == "__main__":
    main()
