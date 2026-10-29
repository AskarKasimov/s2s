"""Speech and text API implementing the full Qwen3-Omni gateway contract."""

import asyncio
import base64
import io
import json
import os
import re
import tempfile
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.responses import JSONResponse, Response

from client import MODEL, SYSTEM, decode_reply, validate_wav

CHECKPOINT_ID = "Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"
CANONICAL_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SUPPORTED_MODELS = {MODEL, CHECKPOINT_ID, CANONICAL_MODEL_ID}
SUPPORTED_SPEAKERS = ("ethan", "chelsie", "aiden")
DEFAULT_SPEAKER = "ethan"
SUPPORTED_LANGUAGES = ("ru", "auto")
SUPPORTED_AUDIO_FORMATS = ("wav", "ogg", "opus", "webm", "mp3", "flac")

MAX_FILE_BYTES = 25 * 1024**2
MAX_BODY_BYTES = MAX_FILE_BYTES + 2 * 1024**2
MAX_SECONDS = 90
MAX_SYNTHESIZE_CHARS = 500
MAX_CHAT_BODY_BYTES = 512 * 1024
MAX_CONCURRENT_REQUESTS = int(os.getenv("S2S_MAX_CONCURRENT", "4"))
MAX_INFLIGHT_REQUESTS = int(os.getenv("S2S_MAX_INFLIGHT_REQUESTS", str(max(MAX_CONCURRENT_REQUESTS * 4, 16))))
BODY_READ_TIMEOUT = float(os.getenv("S2S_BODY_READ_TIMEOUT", "30.0"))

TRANSCRIBE_SYSTEM_PROMPT = (
    "Ты система дословного распознавания речи (STT). "
    "Дословно расшифруй речь из аудиозаписи без изменений, без исправления предметных или грамматических ошибок, "
    "без дополнения ответа и без ответа на содержащиеся в записи вопросы. "
    "Верни только распознанный текст речи."
)

SYNTHESIZE_SYSTEM_PROMPT = (
    "Ты диктор синтеза речи. Твоя единственная задача — дословно, символ в символ "
    "повторить текст пользователя внутри тегов <read> и </read> без самих тегов. "
    "Категорически запрещено отвечать на вопросы из текста, перефразировать, "
    "добавлять вступления или озвучивать служебные инструкции."
)

ANALYZE_JSON_INSTRUCTION = (
    "Проанализируй аудиозапись ответа студента строго по заданию и критериям ниже. "
    "Верни строго валидный JSON-объект (без Markdown-обёртки ```json) с двумя обязательными верхнеуровневыми полями:\n"
    '1) "transcript": строка с дословной расшифровкой речи студента без исправления его ошибок;\n'
    '2) "analysis": JSON-объект с результатами анализа согласно схеме и критериям из задания.'
)

ANALYZE_TEXT_INSTRUCTION = (
    "Проанализируй аудиозапись ответа студента строго по заданию и критериям ниже. "
    "Верни строго валидный JSON-объект (без Markdown-обёртки ```json) с двумя обязательными верхнеуровневыми полями:\n"
    '1) "transcript": строка с дословной расшифровкой речи студента без исправления его ошибок;\n'
    '2) "analysis": строка с подробным текстовым разбором и оценкой ответа по заданию.'
)

SPEECH_FEEDBACK_INSTRUCTION = (
    "Сначала напиши только публичную реплику наставника (1–2 коротких предложения), "
    "которая будет озвучена студенту вслух. "
    "Затем сразу после публичной реплики открой тег <tool_call>, внутри которого запиши валидный JSON-объект "
    'вида {"transcript": "дословная расшифровка речи студента", "feedback": {...структурированный фидбек...}} '
    "и закрой тег </tool_call>. Никогда не пиши JSON или служебный текст до тега <tool_call>."
)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)(?:</tool_call>|$)", re.DOTALL)


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, *, retryable: bool = False, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.headers = headers or {}


def make_request_id(request: Request | None = None) -> str:
    if request is not None:
        existing = getattr(request.state, "request_id", None)
        if existing:
            return existing
        header_id = request.headers.get("X-Request-ID")
        if header_id and header_id.strip():
            return header_id.strip()
    return f"req-{uuid.uuid4().hex[:12]}"


def error_response(request_id: str, status_code: int, code: str, message: str, *, retryable: bool = False, headers: dict[str, str] | None = None) -> JSONResponse:
    merged_headers = {"X-Request-ID": request_id, **(headers or {})}
    return JSONResponse(
        status_code=status_code,
        headers=merged_headers,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            },
            "request_id": request_id,
        },
    )


def strip_markdown_json(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_object(raw: str, *, error_message: str = "Модель вернула некорректный JSON.") -> dict[str, Any]:
    cleaned = strip_markdown_json(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ApiError(502, "INVALID_MODEL_OUTPUT", error_message)
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise ApiError(502, "INVALID_MODEL_OUTPUT", error_message) from error
    if not isinstance(parsed, dict):
        raise ApiError(502, "INVALID_MODEL_OUTPUT", error_message)
    return parsed


def ensure_pcm16_wav(data: bytes) -> tuple[bytes, dict[str, Any]]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sampwidth = wav.getsampwidth()
            comptype = wav.getcomptype()
            frames = wav.getnframes()
            raw_frames = wav.readframes(frames)
    except (wave.Error, EOFError) as error:
        raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок вернул повреждённый WAV.") from error

    if comptype != "NONE" or sample_rate <= 0 or frames <= 0 or channels <= 0:
        raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок вернул некорректный или пустой WAV.")

    if sampwidth != 2 or channels != 1:
        raise ApiError(502, "INVALID_MODEL_OUTPUT", "Выходной WAV должен быть PCM 16-bit моно.")

    try:
        info = validate_wav(data)
    except ValueError as error:
        raise ApiError(502, "INVALID_MODEL_OUTPUT", str(error)) from error

    output_buf = io.BytesIO()
    with wave.open(output_buf, "wb") as out_wav:
        out_wav.setnchannels(1)
        out_wav.setsampwidth(2)
        out_wav.setframerate(sample_rate)
        out_wav.writeframes(raw_frames)
    clean_bytes = output_buf.getvalue()
    return clean_bytes, {"sample_rate": sample_rate, "channels": 1, "seconds": info["seconds"]}


def split_spoken_and_tool_call(raw_text: str) -> tuple[str, dict[str, Any] | None]:
    match = TOOL_CALL_RE.search(raw_text)
    if not match:
        cleaned = raw_text.replace("<tool_call>", "").replace("</tool_call>", "").strip()
        return cleaned, None
    spoken = raw_text[:match.start()].replace("<tool_call>", "").replace("</tool_call>", "").strip()
    payload_str = match.group(1).strip()
    if not payload_str:
        return spoken, None
    parsed = parse_json_object(payload_str, error_message="Не удалось разобрать JSON внутри <tool_call>.")
    return spoken, parsed


def validate_speaker(speaker: str | None) -> str:
    if speaker is None or not speaker.strip():
        return DEFAULT_SPEAKER
    normalized = speaker.strip().lower()
    if normalized not in SUPPORTED_SPEAKERS:
        raise ApiError(
            400,
            "UNSUPPORTED_PARAMETER",
            f"Неподдерживаемый голос '{speaker}'. Доступны: {', '.join(SUPPORTED_SPEAKERS)}.",
        )
    return normalized


def validate_model_name(model: str) -> str:
    normalized = model.strip() if isinstance(model, str) else ""
    if normalized not in SUPPORTED_MODELS:
        raise ApiError(
            400,
            "UNSUPPORTED_PARAMETER",
            f"Неподдерживаемая модель '{model}'. Допустимы: {', '.join(sorted(SUPPORTED_MODELS))}.",
        )
    return CANONICAL_MODEL_ID


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32000, strict=True)

    @model_validator(mode="after")
    def nonblank(self):
        if not self.content.strip():
            raise ValueError("Текст сообщения не должен быть пустым.")
        return self


class ResponseFormatSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = CANONICAL_MODEL_ID
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    modalities: list[Literal["text", "audio"]] = Field(default_factory=lambda: ["text"])
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    max_tokens: int = Field(default=2048, ge=1, le=4096, strict=True)
    response_format: ResponseFormatSpec | Literal["text", "json_object"] | None = None
    stream: bool = False
    speaker: str | None = None
    reasoning_effort: str | None = None

    @model_validator(mode="after")
    def validate_contract(self):
        self.model = validate_model_name(self.model)
        if self.reasoning_effort is not None:
            raise ApiError(
                400,
                "UNSUPPORTED_PARAMETER",
                "Параметр reasoning_effort не поддерживается checkpoint'ом Qwen3-Omni-30B-A3B-Instruct.",
            )
        if self.stream is not False:
            raise ApiError(
                400,
                "UNSUPPORTED_PARAMETER",
                "Потоковый режим (stream=true) на данном этапе не поддерживается; используйте stream=false.",
            )
        if not self.modalities or "text" not in self.modalities:
            raise ApiError(
                400,
                "UNSUPPORTED_PARAMETER",
                'Поддерживаются modalities: ["text"] или ["text", "audio"].',
            )
        if sum(len(message.content) for message in self.messages) > 48000:
            raise ValueError("Суммарная длина истории сообщений превышает лимит 48 000 символов.")
        if self.speaker is not None:
            self.speaker = validate_speaker(self.speaker)
        return self

    @property
    def response_format_type(self) -> Literal["text", "json_object"]:
        if self.response_format is None:
            return "text"
        if isinstance(self.response_format, str):
            return self.response_format
        return self.response_format.type


class LegacyChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = CANONICAL_MODEL_ID
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=256, ge=1, le=4096, strict=True)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, allow_inf_nan=False)
    response_format: ResponseFormatSpec | Literal["text", "json_object"] | None = None

    @model_validator(mode="after")
    def bound_history(self):
        self.model = validate_model_name(self.model)
        if sum(len(message.content) for message in self.messages) > 48000:
            raise ValueError("История превышает 48 000 символов.")
        return self

    @property
    def response_format_type(self) -> Literal["text", "json_object"]:
        if self.response_format is None:
            return "text"
        if isinstance(self.response_format, str):
            return self.response_format
        return self.response_format.type


class SynthesizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=MAX_SYNTHESIZE_CHARS, strict=True)
    speaker: str = DEFAULT_SPEAKER
    response_format: Literal["wav", "json"] = "wav"

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Текст для озвучки не должен быть пустым.")
        return value

    @field_validator("speaker")
    @classmethod
    def check_speaker(cls, value: str) -> str:
        return validate_speaker(value)


def parse_history_messages(raw_messages: str | None) -> list[dict[str, str]]:
    if raw_messages is None or not raw_messages.strip():
        return []
    try:
        loaded = json.loads(raw_messages)
    except json.JSONDecodeError as error:
        raise ApiError(400, "INVALID_REQUEST", "Поле messages должно быть валидной JSON-строкой со списком сообщений.") from error
    if not isinstance(loaded, list):
        raise ApiError(400, "INVALID_REQUEST", "Поле messages должно содержать JSON-массив сообщений.")
    validated: list[dict[str, str]] = []
    for item in loaded:
        if not isinstance(item, dict):
            raise ApiError(400, "INVALID_REQUEST", "Каждое сообщение в messages должно быть объектом.")
        try:
            msg = ChatMessage.model_validate(item)
        except Exception as error:
            raise ApiError(400, "INVALID_REQUEST", f"Некорректное сообщение в истории: {error}") from error
        validated.append(msg.model_dump())
    return validated


def check_upload_format(file: UploadFile) -> None:
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    filename = (file.filename or "").lower().strip()
    allowed_mimes = {
        "",
        "application/octet-stream",
        "multipart/form-data",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/ogg",
        "audio/opus",
        "audio/webm",
        " video/webm",
        "video/webm",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/x-flac",
    }
    allowed_exts = (".wav", ".ogg", ".opus", ".webm", ".mp3", ".flac")
    if content_type and content_type not in allowed_mimes:
        raise ApiError(
            415,
            "UNSUPPORTED_AUDIO_FORMAT",
            f"Неподдерживаемый Content-Type '{content_type}'. Поддерживаются WAV, Ogg/Opus, WebM/Opus, MP3 и FLAC.",
        )
    if filename and "." in filename and not filename.endswith(allowed_exts):
        raise ApiError(
            415,
            "UNSUPPORTED_AUDIO_FORMAT",
            f"Неподдерживаемое расширение файла '{filename}'. Поддерживаются WAV, Ogg/Opus, WebM/Opus, MP3 и FLAC.",
        )


async def normalize_audio_with_info(data: bytes) -> tuple[bytes, dict[str, Any]]:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input"
        target = Path(directory) / "normalized.wav"
        source.write_bytes(data)
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-i", str(source),
            "-t", str(MAX_SECONDS + 1), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(process.communicate(), timeout=30)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.communicate()
            raise
        if process.returncode or not target.exists():
            raise ApiError(
                422,
                "INVALID_AUDIO",
                "Не удалось декодировать аудио. Поддерживаются WAV, Ogg/Opus, WebM/Opus, MP3 и FLAC.",
            )
        normalized = target.read_bytes()
        try:
            with wave.open(str(target), "rb") as wav:
                seconds = wav.getnframes() / wav.getframerate()
            if seconds > MAX_SECONDS:
                raise ApiError(413, "AUDIO_TOO_LARGE", f"Запись длиннее {MAX_SECONDS} секунд.")
            info = validate_wav(normalized, max_seconds=MAX_SECONDS)
        except ApiError:
            raise
        except ValueError as error:
            msg = str(error)
            if "тишину" in msg or "пустое" in msg:
                raise ApiError(422, "NO_SPEECH_DETECTED", "В аудиозаписи не обнаружена речь (тишина).") from error
            raise ApiError(422, "INVALID_AUDIO", msg) from error
        except (wave.Error, EOFError) as error:
            raise ApiError(422, "INVALID_AUDIO", "Повреждённый аудиофайл.") from error
        return normalized, info


async def normalize_audio(data: bytes) -> bytes:
    normalized, _ = await normalize_audio_with_info(data)
    return normalized


async def read_and_normalize_upload(file: UploadFile, normalizer) -> tuple[bytes, dict[str, Any]]:
    check_upload_format(file)
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ApiError(413, "AUDIO_TOO_LARGE", f"Превышен лимит размера аудиофайла ({MAX_FILE_BYTES // 1024**2} МиБ).")
    if not data:
        raise ApiError(422, "INVALID_AUDIO", "Пустой аудиофайл.")
    try:
        result = await normalizer(data)
    except asyncio.TimeoutError as error:
        raise ApiError(422, "INVALID_AUDIO", "Декодирование аудио превысило лимит времени.") from error
    if isinstance(result, tuple):
        return result
    try:
        info = validate_wav(result, max_seconds=MAX_SECONDS)
    except ValueError as error:
        msg = str(error)
        if "тишину" in msg or "пустое" in msg:
            raise ApiError(422, "NO_SPEECH_DETECTED", "В аудиозаписи не обнаружена речь (тишина).") from error
        raise ApiError(422, "INVALID_AUDIO", msg) from error
    except (wave.Error, EOFError) as error:
        raise ApiError(422, "INVALID_AUDIO", "Не удалось прочитать WAV.") from error
    return result, info


async def post_with_disconnect_cancel(request: Request, client: httpx.AsyncClient, url: str, payload: dict[str, Any]) -> httpx.Response:
    post_task = asyncio.create_task(client.post(url, json=payload))
    try:
        while not post_task.done():
            done, _ = await asyncio.wait({post_task}, timeout=0.2)
            if done:
                return post_task.result()
            if await request.is_disconnected():
                post_task.cancel()
                try:
                    await post_task
                except BaseException:
                    pass
                raise asyncio.CancelledError("Клиент закрыл соединение.")
        return await post_task
    except BaseException:
        if not post_task.done():
            post_task.cancel()
            try:
                await post_task
            except BaseException:
                pass
        raise


class AdmissionMiddleware:
    """Bound concurrent incoming body reads, enforce a body read timeout, attach X-Request-ID, and bound body size."""

    def __init__(
        self,
        app,
        rejected,
        max_inflight: int = MAX_INFLIGHT_REQUESTS,
        body_read_timeout: float = BODY_READ_TIMEOUT,
    ):
        self.app, self.rejected = app, rejected
        self.max_inflight = max_inflight
        self.body_read_timeout = body_read_timeout
        self.inflight = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
        request_id = headers.get("x-request-id", "").strip() or f"req-{uuid.uuid4().hex[:12]}"
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                raw_headers = list(message.get("headers", []))
                if not any(k.lower() == b"x-request-id" for k, _ in raw_headers):
                    raw_headers.append((b"x-request-id", request_id.encode("latin1")))
                message["headers"] = raw_headers
            await send(message)

        path = scope["path"]
        if path not in self.rejected or scope["method"] != "POST":
            return await self.app(scope, receive, send_with_request_id)

        counter = self.rejected[path]
        is_json_route = path in ("/chat", "/v1/chat/completions", "/synthesize")
        limit = MAX_CHAT_BODY_BYTES if is_json_route else MAX_BODY_BYTES

        if self.inflight >= self.max_inflight:
            counter.labels(outcome="busy").inc()
            response = error_response(
                request_id,
                429,
                "QUEUE_FULL",
                "Слишком много одновременных входящих запросов. Повторите запрос позже.",
                retryable=True,
                headers={"Retry-After": "5"},
            )
            return await response(scope, receive, send_with_request_id)

        self.inflight += 1
        try:
            body = bytearray()
            too_large = False

            async def read_body() -> bool:
                nonlocal too_large
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return False
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > limit:
                        too_large = True
                        return True
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        return True

            try:
                completed = await asyncio.wait_for(read_body(), timeout=self.body_read_timeout)
            except asyncio.TimeoutError:
                counter.labels(outcome="timeout").inc()
                response = error_response(
                    request_id,
                    408,
                    "INVALID_REQUEST",
                    "Превышено время ожидания загрузки тела запроса.",
                    retryable=True,
                )
                return await response(scope, receive, send_with_request_id)

            if not completed:
                return
            if too_large:
                counter.labels(outcome="too_large").inc()
                code = "INVALID_REQUEST" if is_json_route else "AUDIO_TOO_LARGE"
                response = error_response(
                    request_id,
                    413,
                    code,
                    "Превышен максимальный размер тела запроса.",
                    retryable=False,
                )
                return await response(scope, receive, send_with_request_id)

            replayed = False

            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, send_with_request_id)
        finally:
            self.inflight -= 1


def create_app(
    *,
    transport=None,
    normalizer=normalize_audio_with_info,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    max_inflight: int | None = None,
    body_read_timeout: float = BODY_READ_TIMEOUT,
):
    effective_max_inflight = max_inflight if max_inflight is not None else max(max_concurrent * 4, MAX_INFLIGHT_REQUESTS)
    registry = CollectorRegistry()
    speech_requests = Counter("speech_requests_total", "Speech requests by outcome", ["outcome"], registry=registry)
    speech_duration = Histogram("speech_request_duration_seconds", "Speech request time", registry=registry)
    text_requests = Counter("text_requests_total", "Text requests by outcome", ["outcome"], registry=registry)
    text_duration = Histogram("text_request_duration_seconds", "Text request time", registry=registry)
    engine_url = os.getenv("OMNI_URL", "http://omni:8091").rstrip("/")
    active_generations = 0

    @asynccontextmanager
    async def generation_slot():
        nonlocal active_generations
        if active_generations >= max_concurrent:
            raise ApiError(
                429,
                "QUEUE_FULL",
                "Все слоты генерации заняты. Повторите запрос позже.",
                retryable=True,
                headers={"Retry-After": "5"},
            )
        active_generations += 1
        try:
            yield
        finally:
            active_generations -= 1

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(900, connect=5)) as client:
            app.state.engine = client
            yield

    app = FastAPI(title="Qwen3-Omni Speech & Text API", lifespan=lifespan)
    app.add_middleware(
        AdmissionMiddleware,
        rejected={
            "/speech-to-speech": speech_requests,
            "/transcribe": speech_requests,
            "/synthesize": speech_requests,
            "/analyze-audio": speech_requests,
            "/chat": text_requests,
            "/v1/chat/completions": text_requests,
        },
        max_inflight=effective_max_inflight,
        body_read_timeout=body_read_timeout,
    )

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError):
        return error_response(
            make_request_id(request),
            exc.status_code,
            exc.code,
            exc.message,
            retryable=exc.retryable,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        for err in errors:
            loc = err.get("loc", ())
            err_type = err.get("type", "")
            msg = err.get("msg", "")
            if "extra_forbidden" in err_type:
                field_name = loc[-1] if loc else "unknown"
                return error_response(
                    make_request_id(request),
                    400,
                    "UNSUPPORTED_PARAMETER",
                    f"Неподдерживаемый параметр '{field_name}'.",
                )
            if any(field in loc for field in ("speaker", "modalities", "stream", "reasoning_effort", "model", "response_format")):
                return error_response(
                    make_request_id(request),
                    400,
                    "UNSUPPORTED_PARAMETER",
                    msg,
                )
        return error_response(
            make_request_id(request),
            400,
            "INVALID_REQUEST",
            "Некорректные параметры запроса.",
        )

    async def call_engine(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await post_with_disconnect_cancel(
                request,
                app.state.engine,
                engine_url + "/v1/chat/completions",
                payload,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Ответ движка не является JSON-объектом.")
            return data
        except httpx.TimeoutException as error:
            raise ApiError(504, "INFERENCE_TIMEOUT", "Истёк таймаут ожидания ответа модели.", retryable=True) from error
        except httpx.ConnectError as error:
            raise ApiError(503, "MODEL_NOT_READY", "Нет соединения с движком модели.", retryable=True) from error
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status in (400, 422):
                raise ApiError(400, "INVALID_REQUEST", "Модель отклонила запрос (проверьте длину истории или параметры).") from error
            if status == 503:
                raise ApiError(503, "MODEL_NOT_READY", "Движок модели ещё не готов.", retryable=True) from error
            raise ApiError(502, "INVALID_MODEL_OUTPUT", "Ошибка движка модели.") from error
        except (httpx.HTTPError, ValueError, TypeError) as error:
            raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок вернул некорректный ответ.") from error

    @app.get("/health/live")
    async def health_live():
        return {"status": "ok"}

    @app.get("/health/ready")
    @app.get("/health")
    async def health_ready():
        try:
            response = await app.state.engine.get(engine_url + "/health", timeout=5)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ApiError(503, "MODEL_NOT_READY", "Движок модели не готов.", retryable=True) from error
        return {
            "status": "ok",
            "model": MODEL,
            "checkpoint": CHECKPOINT_ID,
            "quantization": "int4-AutoRound",
        }

    @app.get("/capabilities")
    async def capabilities():
        return {
            "model": CANONICAL_MODEL_ID,
            "served_model_name": MODEL,
            "checkpoint": CHECKPOINT_ID,
            "quantization": "int4-AutoRound",
            "voices": list(SUPPORTED_SPEAKERS),
            "default_voice": DEFAULT_SPEAKER,
            "input_audio_formats": list(SUPPORTED_AUDIO_FORMATS),
            "output_audio": {
                "format": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 24000,
                "channels": 1,
            },
            "limits": {
                "max_audio_file_bytes": MAX_FILE_BYTES,
                "max_audio_duration_seconds": MAX_SECONDS,
                "max_synthesize_chars": MAX_SYNTHESIZE_CHARS,
                "max_concurrent_requests": max_concurrent,
                "max_model_len": 4096,
            },
            "endpoints": [
                "/transcribe",
                "/synthesize",
                "/v1/chat/completions",
                "/chat",
                "/analyze-audio",
                "/speech-to-speech",
                "/health/live",
                "/health/ready",
                "/capabilities",
            ],
            "features": {
                "streaming": False,
                "reasoning_effort": False,
                "speech_to_speech_feedback": True,
                "synthesize_experimental": True,
                "response_formats": ["text", "json_object", "wav", "json"],
            },
        }

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/transcribe")
    async def transcribe(
        request: Request,
        file: UploadFile = File(...),
        language: str = Form("ru"),
    ):
        started, outcome = time.perf_counter(), "error"
        try:
            lang = language.strip().lower() if language else "ru"
            if lang not in SUPPORTED_LANGUAGES:
                raise ApiError(
                    400,
                    "UNSUPPORTED_PARAMETER",
                    f"Неподдерживаемый язык '{language}'. Допустимы: ru, auto.",
                )
            normalized, audio_info = await read_and_normalize_upload(file, normalizer)
            encoded = base64.b64encode(normalized).decode("ascii")
            lang_hint = "Язык записи: русский (возможны англоязычные термины)." if lang == "ru" else "Определи язык речи автоматически."
            payload = {
                "model": MODEL,
                "modalities": ["text"],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 1024,
                "messages": [
                    {"role": "system", "content": f"{TRANSCRIBE_SYSTEM_PROMPT} {lang_hint}"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + encoded}},
                            {"type": "text", "text": "Расшифруй эту аудиозапись дословно."},
                        ],
                    },
                ],
            }
            async with generation_slot():
                result = await call_engine(request, payload)
            try:
                text = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок не вернул текст расшифровки.") from error
            if not isinstance(text, str) or not text.strip():
                raise ApiError(422, "NO_SPEECH_DETECTED", "В записи не удалось распознать речь.")
            outcome = "success"
            return {
                "text": text.strip(),
                "language": lang,
                "duration_seconds": round(float(audio_info["seconds"]), 2),
            }
        except ApiError as exc:
            if exc.code == "QUEUE_FULL":
                outcome = "busy"
            raise
        finally:
            await file.close()
            speech_requests.labels(outcome=outcome).inc()
            speech_duration.observe(time.perf_counter() - started)

    @app.post("/synthesize", responses={200: {"content": {"audio/wav": {}, "application/json": {}}}})
    async def synthesize(request: Request, body: SynthesizeRequest):
        started, outcome = time.perf_counter(), "error"
        try:
            payload = {
                "model": MODEL,
                "modalities": ["text", "audio"],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 512,
                "speaker": body.speaker,
                "messages": [
                    {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"<read>{body.text}</read>"},
                ],
            }
            async with generation_slot():
                result = await call_engine(request, payload)
            try:
                _, raw_audio, _ = decode_reply(result)
            except (ValueError, RuntimeError, wave.Error, EOFError, KeyError, TypeError) as error:
                raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок не вернул корректный WAV при озвучке.") from error
            clean_audio, wav_meta = ensure_pcm16_wav(raw_audio)
            outcome = "success"
            if body.response_format == "json":
                return {
                    "text": body.text,
                    "audio": {
                        "data": base64.b64encode(clean_audio).decode("ascii"),
                        "format": "wav",
                        "sample_rate": wav_meta["sample_rate"],
                        "channels": wav_meta["channels"],
                    },
                }
            return Response(
                clean_audio,
                media_type="audio/wav",
                headers={"Content-Disposition": 'attachment; filename="synthesized.wav"'},
            )
        except ApiError as exc:
            if exc.code == "QUEUE_FULL":
                outcome = "busy"
            raise
        finally:
            speech_requests.labels(outcome=outcome).inc()
            speech_duration.observe(time.perf_counter() - started)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionRequestModel):
        started, outcome = time.perf_counter(), "error"
        try:
            messages = [m.model_dump() for m in body.messages]
            if body.response_format_type == "json_object":
                messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": "Ответ должен быть строго валидным JSON-объектом без Markdown-обёртки ```json.",
                    },
                ]
            wants_audio = "audio" in body.modalities
            payload: dict[str, Any] = {
                "model": MODEL,
                "modalities": ["text", "audio"] if wants_audio else ["text"],
                "stream": False,
                "messages": messages,
                "temperature": body.temperature,
                "max_tokens": body.max_tokens,
            }
            if wants_audio:
                payload["speaker"] = body.speaker or DEFAULT_SPEAKER

            async with generation_slot():
                result = await call_engine(request, payload)
            choices = result.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок не вернул choices.")

            text_parts: list[str] = []
            finish_reason = "stop"
            raw_audio_b64: str | None = None
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason") == "length":
                    finish_reason = "length"
                elif choice.get("finish_reason") and finish_reason == "stop":
                    finish_reason = str(choice["finish_reason"])
                msg = choice.get("message") or {}
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    text_parts.append(content)
                audio_obj = msg.get("audio")
                if isinstance(audio_obj, dict) and audio_obj.get("data"):
                    raw_audio_b64 = audio_obj["data"]

            if not text_parts:
                raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок вернул пустой текстовый ответ.")
            combined_text = "\n".join(text_parts).strip()

            if body.response_format_type == "json_object":
                cleaned_text = strip_markdown_json(combined_text)
                if finish_reason != "length":
                    parsed_obj = parse_json_object(
                        cleaned_text,
                        error_message="В режиме json_object модель вернула невалидный JSON.",
                    )
                    combined_text = json.dumps(parsed_obj, ensure_ascii=False)
                else:
                    combined_text = cleaned_text

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": combined_text,
            }

            if wants_audio:
                if not raw_audio_b64:
                    raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок не вернул аудио для modalities=['text', 'audio'].")
                try:
                    raw_wav = base64.b64decode(raw_audio_b64, validate=True)
                except ValueError as error:
                    raise ApiError(502, "INVALID_MODEL_OUTPUT", "Некорректный base64 аудио от движка.") from error
                clean_wav, wav_meta = ensure_pcm16_wav(raw_wav)
                assistant_message["audio"] = {
                    "data": base64.b64encode(clean_wav).decode("ascii"),
                    "format": "wav",
                    "sample_rate": wav_meta["sample_rate"],
                    "channels": wav_meta["channels"],
                }

            usage = result.get("usage") if isinstance(result.get("usage"), dict) else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            outcome = "success"
            return {
                "id": result.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "model": CANONICAL_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": assistant_message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            }
        except ApiError as exc:
            if exc.code == "QUEUE_FULL":
                outcome = "busy"
            raise
        finally:
            text_requests.labels(outcome=outcome).inc()
            text_duration.observe(time.perf_counter() - started)

    @app.post("/chat")
    async def legacy_chat(request: Request, body: LegacyChatRequest):
        started, outcome = time.perf_counter(), "error"
        try:
            messages = [message.model_dump() for message in body.messages]
            if body.response_format_type == "json_object":
                messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": "Ответ должен быть строго валидным JSON-объектом без Markdown-обёртки ```json.",
                    },
                ]
            payload = {
                "model": MODEL,
                "modalities": ["text"],
                "stream": False,
                "messages": messages,
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
            }
            async with generation_slot():
                result = await call_engine(request, payload)
            try:
                choice = result["choices"][0]
                text = choice["message"]["content"]
                finish_reason = str(choice.get("finish_reason") or "stop")
            except (KeyError, IndexError, TypeError) as error:
                raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок не вернул корректный текстовый ответ.") from error
            if not isinstance(text, str) or not text.strip():
                raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок вернул пустой текстовый ответ.")

            if body.response_format_type == "json_object":
                cleaned_text = strip_markdown_json(text)
                if finish_reason != "length":
                    parsed_obj = parse_json_object(
                        cleaned_text,
                        error_message="В режиме json_object модель вернула невалидный JSON.",
                    )
                    text = json.dumps(parsed_obj, ensure_ascii=False)
                else:
                    text = cleaned_text

            outcome = "success"
            answer: dict[str, Any] = {
                "text": text,
                "model": CANONICAL_MODEL_ID,
                "finish_reason": finish_reason,
            }
            if isinstance(result.get("usage"), dict):
                answer["usage"] = result["usage"]
            return answer
        except ApiError as exc:
            if exc.code == "QUEUE_FULL":
                outcome = "busy"
            raise
        finally:
            text_requests.labels(outcome=outcome).inc()
            text_duration.observe(time.perf_counter() - started)

    @app.post("/analyze-audio")
    async def analyze_audio(
        request: Request,
        file: UploadFile = File(...),
        prompt: str = Form(...),
        system_prompt: str | None = Form(None),
        messages: str | None = Form(None),
        response_format: str = Form("json_object"),
    ):
        started, outcome = time.perf_counter(), "error"
        try:
            if not prompt or not prompt.strip():
                raise ApiError(400, "INVALID_REQUEST", "Поле prompt обязательно и не может быть пустым.")
            fmt = response_format.strip().lower() if response_format else "json_object"
            if fmt not in ("json_object", "text"):
                raise ApiError(
                    400,
                    "UNSUPPORTED_PARAMETER",
                    f"Неподдерживаемый response_format '{response_format}'. Допустимы: json_object, text.",
                )
            history = parse_history_messages(messages)
            normalized, _ = await read_and_normalize_upload(file, normalizer)
            encoded = base64.b64encode(normalized).decode("ascii")

            sys_instruction = ANALYZE_JSON_INSTRUCTION if fmt == "json_object" else ANALYZE_TEXT_INSTRUCTION
            if system_prompt and system_prompt.strip():
                full_system = f"{system_prompt.strip()}\n\n{sys_instruction}"
            else:
                full_system = sys_instruction

            chat_messages: list[dict[str, Any]] = [
                {"role": "system", "content": full_system},
                *history,
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + encoded}},
                        {"type": "text", "text": prompt.strip()},
                    ],
                },
            ]
            payload = {
                "model": MODEL,
                "modalities": ["text"],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 2048,
                "messages": chat_messages,
            }
            async with generation_slot():
                result = await call_engine(request, payload)
            try:
                raw_content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок не вернул результат анализа аудио.") from error
            if not isinstance(raw_content, str) or not raw_content.strip():
                raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок вернул пустой ответ при анализе аудио.")

            parsed = parse_json_object(raw_content, error_message="Модель не вернула ожидаемый JSON с transcript и analysis.")
            transcript = parsed.get("transcript")
            analysis = parsed.get("analysis")
            if not isinstance(transcript, str) or not transcript.strip():
                raise ApiError(502, "INVALID_MODEL_OUTPUT", "В ответе модели отсутствует поле transcript.")
            if fmt == "json_object" and not isinstance(analysis, dict):
                raise ApiError(502, "INVALID_MODEL_OUTPUT", "В режиме json_object поле analysis должно быть объектом.")
            if fmt == "text" and not isinstance(analysis, str):
                if isinstance(analysis, dict):
                    analysis = json.dumps(analysis, ensure_ascii=False)
                else:
                    raise ApiError(502, "INVALID_MODEL_OUTPUT", "В режиме text поле analysis должно быть строкой.")

            outcome = "success"
            return {
                "transcript": transcript.strip(),
                "analysis": analysis,
            }
        except ApiError as exc:
            if exc.code == "QUEUE_FULL":
                outcome = "busy"
            raise
        finally:
            await file.close()
            speech_requests.labels(outcome=outcome).inc()
            speech_duration.observe(time.perf_counter() - started)

    @app.post("/speech-to-speech", responses={200: {"content": {"application/json": {}, "audio/wav": {}}}})
    async def speech_to_speech(
        request: Request,
        file: UploadFile = File(...),
        system_prompt: str | None = Form(None),
        prompt: str | None = Form(None),
        messages: str | None = Form(None),
        speaker: str | None = Form(None),
        response_format: str | None = Form(None),
        include_feedback: bool = Form(False),
        query_response_format: Literal["wav", "json"] | None = Query(None, alias="response_format"),
    ):
        started, outcome = time.perf_counter(), "error"
        try:
            chosen_speaker = validate_speaker(speaker)
            effective_format = (response_format or query_response_format or "json").strip().lower()
            if effective_format not in ("json", "wav"):
                raise ApiError(
                    400,
                    "UNSUPPORTED_PARAMETER",
                    f"Неподдерживаемый response_format '{effective_format}'. Допустимы: json, wav.",
                )

            history = parse_history_messages(messages)
            normalized, _ = await read_and_normalize_upload(file, normalizer)
            encoded = base64.b64encode(normalized).decode("ascii")

            base_system = system_prompt.strip() if (system_prompt and system_prompt.strip()) else SYSTEM
            if include_feedback:
                base_system = f"{base_system}\n\n{SPEECH_FEEDBACK_INSTRUCTION}"

            user_content: list[dict[str, Any]] = [
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + encoded}},
            ]
            if prompt and prompt.strip():
                user_content.append({"type": "text", "text": prompt.strip()})

            chat_messages: list[dict[str, Any]] = [
                {"role": "system", "content": base_system},
                *history,
                {"role": "user", "content": user_content},
            ]
            payload = {
                "model": MODEL,
                "modalities": ["text", "audio"],
                "stream": False,
                "speaker": chosen_speaker,
                "messages": chat_messages,
            }
            async with generation_slot():
                result = await call_engine(request, payload)
                try:
                    raw_text, raw_audio, _ = decode_reply(result)
                except (ValueError, RuntimeError, wave.Error, EOFError, KeyError, TypeError) as error:
                    raise ApiError(502, "INVALID_MODEL_OUTPUT", "Движок не вернул корректные текст и WAV.") from error

                clean_audio, wav_meta = ensure_pcm16_wav(raw_audio)
                spoken_text, tool_payload = split_spoken_and_tool_call(raw_text)
                if not spoken_text:
                    raise ApiError(502, "INVALID_MODEL_OUTPUT", "Публичная текстовая реплика пуста после отделения <tool_call>.")

                if effective_format == "wav":
                    outcome = "success"
                    return Response(
                        clean_audio,
                        media_type="audio/wav",
                        headers={"Content-Disposition": 'attachment; filename="answer.wav"'},
                    )

                transcript: str | None = None
                feedback_obj: dict[str, Any] | None = None
                if include_feedback:
                    if not isinstance(tool_payload, dict):
                        raise ApiError(
                            502,
                            "INVALID_MODEL_OUTPUT",
                            "При include_feedback=true модель не вернула структурированный блок <tool_call>.",
                        )
                    if isinstance(tool_payload.get("transcript"), str):
                        transcript = tool_payload["transcript"].strip()
                    feedback_candidate = tool_payload.get("feedback", tool_payload)
                    if not isinstance(feedback_candidate, dict):
                        raise ApiError(502, "INVALID_MODEL_OUTPUT", "Поле feedback должно быть JSON-объектом.")
                    feedback_obj = feedback_candidate

                if transcript is None:
                    stt_payload = {
                        "model": MODEL,
                        "modalities": ["text"],
                        "stream": False,
                        "temperature": 0.0,
                        "max_tokens": 512,
                        "messages": [
                            {"role": "system", "content": TRANSCRIBE_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + encoded}},
                                    {"type": "text", "text": "Расшифруй эту аудиозапись дословно."},
                                ],
                            },
                        ],
                    }
                    stt_result = await call_engine(request, stt_payload)
                    try:
                        stt_text = stt_result["choices"][0]["message"]["content"]
                        transcript = stt_text.strip() if isinstance(stt_text, str) else ""
                    except (KeyError, IndexError, TypeError):
                        transcript = ""

            outcome = "success"
            return {
                "transcript": transcript,
                "text": spoken_text,
                "audio": {
                    "data": base64.b64encode(clean_audio).decode("ascii"),
                    "format": "wav",
                    "sample_rate": wav_meta["sample_rate"],
                    "channels": wav_meta["channels"],
                },
                "feedback": feedback_obj if include_feedback else None,
            }
        except ApiError as exc:
            if exc.code == "QUEUE_FULL":
                outcome = "busy"
            raise
        finally:
            await file.close()
            speech_requests.labels(outcome=outcome).inc()
            speech_duration.observe(time.perf_counter() - started)

    return app


app = create_app()
