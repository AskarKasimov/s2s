# Контракт HTTP API сервиса Qwen3-Omni

Это API сервера модели (`http://<IP_СЕРВЕРА>:8002`). Backend продолжает работать через ваш gateway для LLM-задач. Поля `complexity`, `urgent` и `routing_profile` остаются ответственностью gateway; серверу Qwen они не нужны.

Интерактивный Swagger UI: **`http://<IP_СЕРВЕРА>:8002/docs`**.

---

## 1. Общие правила

- Все ручки используют один checkpoint `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` (`Qwen/Qwen3-Omni-30B-A3B-Instruct`).
- История передаётся в запросе; сервер модели не хранит учебные сессии.
- Имена голосов во внешнем API: **`ethan`** (по умолчанию), **`chelsie`**, **`aiden`**. Внутри сервер преобразует их в формат движка.
- STT (`/transcribe`) и анализ аудио (`/analyze-audio`) возвращают только текстовые результаты (`modalities: ["text"]`), без запуска стадий генерации речи.
- На первом этапе поддерживается только `stream: false`.
- `temperature: 0.0` относится к текстовой генерации `Thinker`. Параметры `Talker` и `Code2Wav` сервер настраивает отдельно в `deploy.yaml`.
- Каждый ответ содержит заголовок **`X-Request-ID`** (если клиент прислал свой `X-Request-ID`, сервер пробрасывает его обратно).
- Аудио на выходе — настоящий WAV с **PCM 16-bit моно**. Частоту (`sample_rate`) и число каналов (`channels: 1`) сервер возвращает в метаданных JSON; при бинарном ответе они доступны в заголовке WAV.
- Входное аудио: поддерживаются **WAV, Ogg/Opus, WebM/Opus, MP3, FLAC**, размер до **25 МиБ** и длительность до **90 секунд**. Сервер декодирует и приводит их к формату модели (16 кГц, PCM 16-bit, моно).
- При отключении клиента сервер отменяет связанный запрос к движку и освобождает слот.

---

## 2. Распознавание речи: `POST /transcribe`

`Content-Type: multipart/form-data`

| Поле | Тип | Обязательное | Значение |
| :--- | :--- | :--- | :--- |
| `file` | файл | Да | Аудио студента |
| `language` | строка | Нет | `ru` или `auto`, по умолчанию `ru` |

### Пример запроса
```bash
curl -X POST http://localhost:8002/transcribe \
  -F 'file=@student.ogg' \
  -F 'language=ru'
```

### Ответ `200 application/json`
```json
{
  "text": "Предельные издержки — это изменение общих издержек...",
  "language": "ru",
  "duration_seconds": 18.4
}
```

### Поведение
- Только дословное распознавание речи, без ответа на содержащиеся в записи вопросы.
- Без исправления предметных ошибок и без дополнения ответа студента.
- Если в записи тишина / нет речи — возвращается `422 NO_SPEECH_DETECTED`.
- Инструкцию на транскрибирование сервер задаёт сам.

---

## 3. Озвучка: `POST /synthesize`

`Content-Type: application/json`

> **Статус:** экспериментальная ручка (так как в архитектуре `Qwen3-Omni` стадия `Talker` получает скрытые состояния от `Thinker`, дословное чтение обеспечивается серверным дикторским шаблоном `<read>...</read>` при `temperature: 0.0`).

### Запрос
```json
{
  "text": "Объясните, чем постоянные издержки отличаются от переменных.",
  "speaker": "ethan",
  "response_format": "wav"
}
```

| Поле | Обязательное | Значение |
| :--- | :--- | :--- |
| `text` | Да | Точный текст для чтения (до **500 символов**) |
| `speaker` | Нет | `ethan`, `chelsie` или `aiden` (по умолчанию `ethan`) |
| `response_format` | Нет | `wav` или `json` (по умолчанию `wav`) |

### Ответ для `wav`
`HTTP/1.1 200 OK`  
`Content-Type: audio/wav`  
`<байты WAV PCM 16-bit>`

### Ответ для `json`
```json
{
  "text": "Объясните, чем постоянные издержки отличаются от переменных.",
  "audio": {
    "data": "<base64 полного WAV-файла>",
    "format": "wav",
    "sample_rate": 24000,
    "channels": 1
  }
}
```

---

## 4. Текстовые и мультимодальные задачи: `POST /v1/chat/completions`

OpenAI-совместимая ручка (`POST /chat` сохранён как упрощённый алиас).

### Запрос
```json
{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "messages": [
    {
      "role": "system",
      "content": "Ты преподаватель. Используй только переданные материалы."
    },
    {
      "role": "user",
      "content": "Материалы курса и задача..."
    }
  ],
  "modalities": ["text"],
  "temperature": 0.0,
  "max_tokens": 2048,
  "response_format": {
    "type": "json_object"
  },
  "stream": false
}
```

| Поле | Требование |
| :--- | :--- |
| `messages` | Роли `system`, `user`, `assistant`; сохранение порядка истории |
| `modalities` | `["text"]` (по умолчанию) или `["text", "audio"]` |
| `temperature` | `0.0`–`2.0` (по умолчанию `0.0`) |
| `max_tokens` | Ограничение текстового ответа (`1`–`4096`, по умолчанию `2048`) |
| `response_format` | `{"type": "text"}` / `"text"` или `{"type": "json_object"}` / `"json_object"` |
| `stream` | На первом этапе только `false` (при `true` возвращается `400 UNSUPPORTED_PARAMETER`) |
| `speaker` | `ethan`, `chelsie`, `aiden` — используется при `modalities: ["text", "audio"]` |
| `reasoning_effort` | **Не поддерживается** данным checkpoint'ом (возвращает `400 UNSUPPORTED_PARAMETER`, см. `GET /capabilities`) |

### Текстовый ответ (`modalities: ["text"]`)
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"verdict\":\"partial\",\"feedback\":\"...\"}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 180,
    "total_tokens": 1380
  }
}
```
В режиме `json_object` поле `content` гарантированно содержит валидную JSON-строку без Markdown-обёртки ````json`. При достижении лимита токенов возвращается `finish_reason: "length"`.

### Ответ с озвучкой (`modalities: ["text", "audio"]`)
Сервер нормализует ответ движка в единый `choices[0].message`:
```json
{
  "role": "assistant",
  "content": "Публичный ответ преподавателя.",
  "audio": {
    "data": "<base64 WAV>",
    "format": "wav",
    "sample_rate": 24000,
    "channels": 1
  }
}
```

---

## 5. Анализ ответа студента: `POST /analyze-audio`

`Content-Type: multipart/form-data`

| Поле | Тип | Обязательное | Описание |
| :--- | :--- | :--- | :--- |
| `file` | файл | Да | Аудиозапись ответа студента |
| `prompt` | строка | Да | Вопрос, эталон, материалы и критерии оценки |
| `system_prompt` | строка | Нет | Дополнительная системная роль |
| `messages` | JSON-строка | Нет | Предыдущая история диалога (JSON-массив сообщений) |
| `response_format` | строка | Нет | `json_object` (по умолчанию) или `text` |

### Пример запроса
```bash
curl -X POST http://localhost:8002/analyze-audio \
  -F 'file=@student_answer.webm' \
  -F 'prompt=Оцени ответ студента по критериям k1, k2 и верни вердикт.' \
  -F 'response_format=json_object'
```

### Ответ
```json
{
  "transcript": "Точный текст ответа студента...",
  "analysis": {
    "verdict": "partial",
    "answer_complete": false,
    "criteria": {
      "k1": 2,
      "k2": 1,
      "k3": null,
      "k4": null
    },
    "rationale": "Определение верное, но обоснование неполное.",
    "feedback": "Определение верное. Объясните, почему..."
  }
}
```
- `transcript` — дословно распознанный ответ студента.
- `analysis` — JSON-объект при `response_format=json_object` (структура определяется вашим `prompt`) или строка при `response_format=text`.
- Аудио не генерируется.

---

## 6. Голосовой диалог: `POST /speech-to-speech`

`Content-Type: multipart/form-data`

| Поле | Тип | Обязательное | Описание |
| :--- | :--- | :--- | :--- |
| `file` | файл | Да | Аудиозапись студента |
| `system_prompt` | строка | Нет | Инструкция для наставника |
| `prompt` | строка | Нет | Контекст текущего хода |
| `messages` | JSON-строка | Нет | История сообщений (JSON-массив) |
| `speaker` | строка | Нет | `ethan` (по умолчанию), `chelsie`, `aiden` |
| `response_format` | строка | Нет | `json` (по умолчанию) или `wav` |
| `include_feedback` | boolean | Нет | `false` (по умолчанию) или `true` |

### Ответ при `response_format=json` (по умолчанию)
```json
{
  "transcript": "Ответ студента...",
  "text": "Публичная реплика наставника.",
  "audio": {
    "data": "<base64 WAV>",
    "format": "wav",
    "sample_rate": 24000,
    "channels": 1
  },
  "feedback": null
}
```
- При `include_feedback: true` сервер использует токен-границу `<tool_call>` (на которой `Talker` в `vLLM-Omni` прекращает озвучку), проверяет и отделяет JSON-блок от публичной реплики `text`, возвращая структурированный объект в поле `feedback`. В аудио попадает только публичная реплика `text`.
- При `response_format=wav` возвращается бинарный `audio/wav`.

---

## 7. Ошибки, готовность и возможности

### Единый формат ошибок
Все ошибки возвращаются с заголовком `X-Request-ID` и единым JSON-телом:
```json
{
  "error": {
    "code": "INVALID_AUDIO",
    "message": "Не удалось декодировать аудио.",
    "retryable": false
  },
  "request_id": "req-123"
}
```

| HTTP | Код (`error.code`) | Когда возникает |
| :--- | :--- | :--- |
| `400` | `INVALID_REQUEST` / `UNSUPPORTED_PARAMETER` | Некорректная схема запроса, неподдерживаемый голос, `stream=true` или `reasoning_effort` |
| `413` | `AUDIO_TOO_LARGE` | Файл больше 25 МиБ или длиннее 90 секунд |
| `415` | `UNSUPPORTED_AUDIO_FORMAT` | Неподдерживаемый MIME-тип или расширение файла |
| `422` | `INVALID_AUDIO` / `NO_SPEECH_DETECTED` | Битый аудиофайл или запись содержит только тишину |
| `429` | `QUEUE_FULL` | Все 4 слота генерации заняты (содержит заголовок `Retry-After: 5`, `retryable: true`) |
| `502` | `INVALID_MODEL_OUTPUT` | Движок вернул невалидный JSON или повреждённый WAV |
| `503` | `MODEL_NOT_READY` | Движок ещё загружается или недоступен (`retryable: true`) |
| `504` | `INFERENCE_TIMEOUT` | Превышен таймаут ожидания ответа от модели (`retryable: true`) |

### Служебные ручки
- **`GET /health/live`** — процесс HTTP-шлюза жив (`200 OK`).
- **`GET /health/ready`** (и алиас **`GET /health`**) — все стадии модели в `omni` загружены и готовы к работе (`200 OK` или `503 MODEL_NOT_READY`).
- **`GET /capabilities`** — возвращает метаданные checkpoint'а, список голосов (`ethan`, `chelsie`, `aiden`), поддерживаемые аудиоформаты, лимиты (25 МиБ, 90 с, 500 символов для `/synthesize`, 4 параллельных слота), список ручек и флаги фич (`streaming: false`, `reasoning_effort: false`, `speech_to_speech_feedback: true`, `synthesize_experimental: true`).
- **`GET /metrics`** — метрики Prometheus.
