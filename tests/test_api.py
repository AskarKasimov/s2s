import asyncio
import base64
import io
import json
import struct
import sys
import unittest
import wave
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import MAX_BODY_BYTES, MAX_CHAT_BODY_BYTES, MAX_FILE_BYTES, create_app


def audio(seconds=0.1, *, silence=False, sample_rate=24000):
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<h", 0 if silence else 1200) * int(seconds * sample_rate))
    return data.getvalue()


def reply(data=None, text="Четыре.", finish_reason="stop"):
    return {
        "id": "chatcmpl-test1",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason},
            {"index": 1, "message": {"role": "assistant", "audio": {"data": base64.b64encode(data or audio()).decode()}}, "finish_reason": finish_reason},
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
    }


class ApiTests(unittest.TestCase):
    def client(self, handler):
        client = TestClient(create_app(transport=httpx.MockTransport(handler)))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_request_id_header_propagated_and_generated(self):
        client = self.client(lambda request: httpx.Response(200, json={}))
        resp1 = client.get("/health/live")
        self.assertEqual(resp1.status_code, 200)
        self.assertTrue(resp1.headers.get("X-Request-ID", "").startswith("req-"))

        resp2 = client.get("/health/live", headers={"X-Request-ID": "custom-req-42"})
        self.assertEqual(resp2.headers.get("X-Request-ID"), "custom-req-42")

    def test_capabilities_and_health_endpoints(self):
        client = self.client(lambda request: httpx.Response(200, json={}))
        self.assertEqual(client.get("/health/live").status_code, 200)
        self.assertEqual(client.get("/health/ready").status_code, 200)
        caps = client.get("/capabilities").json()
        self.assertEqual(caps["voices"], ["ethan", "chelsie", "aiden"])
        self.assertEqual(caps["limits"]["max_audio_duration_seconds"], 90)
        self.assertEqual(caps["limits"]["max_audio_file_bytes"], 25 * 1024**2)

    def test_transcribe_stt_success_and_no_speech(self):
        seen = []

        def engine(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "Предельные издержки — это прирост общих издержек."}}]
            })

        client = self.client(engine)
        resp = client.post("/transcribe", files={"file": ("student.wav", audio(1.5), "audio/wav")}, data={"language": "ru"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["text"], "Предельные издержки — это прирост общих издержек.")
        self.assertEqual(body["language"], "ru")
        self.assertAlmostEqual(body["duration_seconds"], 1.5, places=1)
        self.assertEqual(seen[0]["modalities"], ["text"])
        self.assertEqual(seen[0]["temperature"], 0.0)

        # Silence must return 422 NO_SPEECH_DETECTED without calling engine
        silence_resp = client.post("/transcribe", files={"file": ("silence.wav", audio(0.5, silence=True), "audio/wav")})
        self.assertEqual(silence_resp.status_code, 422)
        self.assertEqual(silence_resp.json()["error"]["code"], "NO_SPEECH_DETECTED")

    def test_synthesize_wav_and_json_and_limits(self):
        seen = []

        def engine(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=reply(text="Объясните разницу."))

        client = self.client(engine)
        wav_resp = client.post("/synthesize", json={"text": "Объясните разницу.", "speaker": "chelsie", "response_format": "wav"})
        self.assertEqual(wav_resp.status_code, 200)
        self.assertEqual(wav_resp.headers["content-type"], "audio/wav")
        self.assertEqual(seen[0]["speaker"], "chelsie")
        self.assertEqual(seen[0]["modalities"], ["text", "audio"])
        self.assertIn("<read>Объясните разницу.</read>", seen[0]["messages"][1]["content"])

        json_resp = client.post("/synthesize", json={"text": "Объясните разницу.", "response_format": "json"})
        self.assertEqual(json_resp.status_code, 200)
        data = json_resp.json()
        self.assertEqual(data["text"], "Объясните разницу.")
        self.assertEqual(data["audio"]["format"], "wav")
        self.assertEqual(data["audio"]["sample_rate"], 24000)
        self.assertEqual(data["audio"]["channels"], 1)

        # Over 500 chars rejected
        too_long = client.post("/synthesize", json={"text": "а" * 501})
        self.assertEqual(too_long.status_code, 400)
        self.assertEqual(too_long.json()["error"]["code"], "INVALID_REQUEST")

        # Invalid speaker rejected with UNSUPPORTED_PARAMETER
        bad_speaker = client.post("/synthesize", json={"text": "Привет", "speaker": "unknown_voice"})
        self.assertEqual(bad_speaker.status_code, 400)
        self.assertEqual(bad_speaker.json()["error"]["code"], "UNSUPPORTED_PARAMETER")

    def test_v1_chat_completions_text_and_json_object_and_audio(self):
        def engine(request):
            payload = json.loads(request.content)
            if "audio" in payload["modalities"]:
                return httpx.Response(200, json=reply(text="Публичный ответ преподавателя."))
            return httpx.Response(200, json={
                "id": "chatcmpl-999",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": '```json\n{"verdict":"partial","feedback":"ok"}\n```'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            })

        client = self.client(engine)
        resp = client.post("/v1/chat/completions", json={
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "messages": [{"role": "system", "content": "Ты преподаватель."}, {"role": "user", "content": "Задача"}],
            "modalities": ["text"],
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "stream": False,
        })
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        self.assertEqual(json.loads(body["choices"][0]["message"]["content"]), {"verdict": "partial", "feedback": "ok"})
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

        # Multimodal ["text", "audio"] normalized into single choice with audio metadata
        audio_resp = client.post("/v1/chat/completions", json={
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "Привет"}],
            "modalities": ["text", "audio"],
            "speaker": "aiden",
        })
        self.assertEqual(audio_resp.status_code, 200)
        msg = audio_resp.json()["choices"][0]["message"]
        self.assertEqual(msg["content"], "Публичный ответ преподавателя.")
        self.assertEqual(msg["audio"]["sample_rate"], 24000)
        self.assertEqual(msg["audio"]["channels"], 1)

        # reasoning_effort explicitly rejected with UNSUPPORTED_PARAMETER
        re_resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Привет"}],
            "reasoning_effort": "high",
        })
        self.assertEqual(re_resp.status_code, 400)
        self.assertEqual(re_resp.json()["error"]["code"], "UNSUPPORTED_PARAMETER")

        # stream=true explicitly rejected
        st_resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Привет"}],
            "stream": True,
        })
        self.assertEqual(st_resp.status_code, 400)
        self.assertEqual(st_resp.json()["error"]["code"], "UNSUPPORTED_PARAMETER")

    def test_analyze_audio_json_and_text_modes(self):
        def engine(request):
            return httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "transcript": "Точный текст ответа студента...",
                            "analysis": {
                                "verdict": "partial",
                                "answer_complete": False,
                                "criteria": {"k1": 2, "k2": 1},
                                "feedback": "Уточните почему.",
                            },
                        }, ensure_ascii=False)
                    }
                }]
            })

        client = self.client(engine)
        resp = client.post(
            "/analyze-audio",
            files={"file": ("ans.wav", audio(), "audio/wav")},
            data={
                "prompt": "Вопрос и критерии К1-К2",
                "system_prompt": "Ты экзаменатор.",
                "messages": json.dumps([{"role": "assistant", "content": "Первый вопрос"}]),
                "response_format": "json_object",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["transcript"], "Точный текст ответа студента...")
        self.assertEqual(data["analysis"]["verdict"], "partial")

    def test_speech_to_speech_json_default_and_feedback_and_wav(self):
        def engine(request):
            payload = json.loads(request.content)
            if payload["modalities"] == ["text"]:
                return httpx.Response(200, json={"choices": [{"message": {"content": "Ответ студента вслух"}}]})
            sys_prompt = payload["messages"][0]["content"]
            if "<tool_call>" in sys_prompt:
                text = (
                    "Публичная реплика наставника.\n"
                    '<tool_call>{"transcript": "Ответ студента из тулколла", "feedback": {"verdict": "partial", "score": 3}}</tool_call>'
                )
            else:
                text = "Публичная реплика наставника."
            return httpx.Response(200, json=reply(text=text))

        client = self.client(engine)

        # 1. Default response_format is JSON without feedback
        r1 = client.post("/speech-to-speech", files={"file": ("q.wav", audio(), "audio/wav")})
        self.assertEqual(r1.status_code, 200)
        b1 = r1.json()
        self.assertEqual(b1["transcript"], "Ответ студента вслух")
        self.assertEqual(b1["text"], "Публичная реплика наставника.")
        self.assertIsNone(b1["feedback"])
        self.assertEqual(b1["audio"]["channels"], 1)
        self.assertEqual(b1["audio"]["sample_rate"], 24000)

        # 2. With include_feedback=true, separates <tool_call> from spoken text
        r2 = client.post(
            "/speech-to-speech",
            files={"file": ("q.wav", audio(), "audio/wav")},
            data={"include_feedback": "true", "speaker": "ethan"},
        )
        self.assertEqual(r2.status_code, 200)
        b2 = r2.json()
        self.assertEqual(b2["text"], "Публичная реплика наставника.")
        self.assertNotIn("<tool_call>", b2["text"])
        self.assertEqual(b2["transcript"], "Ответ студента из тулколла")
        self.assertEqual(b2["feedback"], {"verdict": "partial", "score": 3})

        # 3. WAV response_format
        r3 = client.post(
            "/speech-to-speech",
            files={"file": ("q.wav", audio(), "audio/wav")},
            data={"response_format": "wav"},
        )
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.headers["content-type"], "audio/wav")

    def test_error_codes_and_formats(self):
        client = self.client(lambda request: self.fail("Не следует вызывать модель"))

        # 415 UNSUPPORTED_AUDIO_FORMAT
        r415 = client.post("/transcribe", files={"file": ("doc.pdf", b"123", "application/pdf")})
        self.assertEqual(r415.status_code, 415)
        self.assertEqual(r415.json()["error"]["code"], "UNSUPPORTED_AUDIO_FORMAT")

        # 422 INVALID_AUDIO
        r422 = client.post("/transcribe", files={"file": ("bad.wav", b"not audio", "audio/wav")})
        self.assertEqual(r422.status_code, 422)
        self.assertEqual(r422.json()["error"]["code"], "INVALID_AUDIO")

        # 413 AUDIO_TOO_LARGE (duration > 90s)
        r413_dur = client.post("/transcribe", files={"file": ("long.wav", audio(90.5, sample_rate=8000), "audio/wav")})
        self.assertEqual(r413_dur.status_code, 413)
        self.assertEqual(r413_dur.json()["error"]["code"], "AUDIO_TOO_LARGE")

        # 413 oversized body
        r413_body = client.post("/speech-to-speech", content=bytes(MAX_BODY_BYTES + 1))
        self.assertEqual(r413_body.status_code, 413)
        self.assertEqual(r413_body.json()["error"]["code"], "AUDIO_TOO_LARGE")

    def test_upstream_errors_and_timeouts(self):
        client_503 = self.client(lambda request: httpx.Response(503))
        r503 = client_503.get("/health/ready")
        self.assertEqual(r503.status_code, 503)
        self.assertEqual(r503.json()["error"]["code"], "MODEL_NOT_READY")

        def timeout_engine(request):
            raise httpx.ReadTimeout("timeout", request=request)

        r504 = self.client(timeout_engine).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Тест"}]})
        self.assertEqual(r504.status_code, 504)
        self.assertEqual(r504.json()["error"]["code"], "INFERENCE_TIMEOUT")

    def test_busy_admission_and_release(self):
        async def check():
            entered, release = asyncio.Event(), asyncio.Event()
            in_flight = 0

            async def normalizer(data):
                nonlocal in_flight
                in_flight += 1
                if in_flight == 4:
                    entered.set()
                await release.wait()
                return data, {"seconds": 0.1, "sample_rate": 24000, "rms": 100.0}

            app = create_app(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply())), normalizer=normalizer)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                    tasks = [
                        asyncio.create_task(client.post("/speech-to-speech", files={"file": ("q.wav", audio(), "audio/wav")}, data={"response_format": "wav"}))
                        for _ in range(4)
                    ]
                    await asyncio.wait_for(entered.wait(), 5)
                    fifth = await client.post("/speech-to-speech", files={"file": ("q.wav", audio(), "audio/wav")})
                    self.assertEqual(fifth.status_code, 429)
                    self.assertEqual(fifth.json()["error"]["code"], "QUEUE_FULL")
                    self.assertEqual(fifth.headers.get("Retry-After"), "5")
                    self.assertEqual((await client.get("/health/live")).status_code, 200)
                    release.set()
                    for task in tasks:
                        self.assertEqual((await task).status_code, 200)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
