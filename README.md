# s2s

HTTP API распознавания речи, синтеза, анализа устных ответов и голосового диалога на базе `Qwen/Qwen3-Omni-30B-A3B-Instruct`.
Репозиторий содержит конфигурацию движка `vLLM-Omni`, HTTP-шлюз, тесты и Docker-окружение.

Используется квантованный checkpoint [`Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound`](https://huggingface.co/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound), ревизия `c3ef7bbf0b9d866866136c1d08b6314d27f7e151`, и движок [`vllm-project/vllm-omni`](https://github.com/vllm-project/vllm-omni), коммит `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`. Полный контракт запросов и ответов описан в [API.md](API.md).

## Пошаговое развёртывание на сервере с NVIDIA GPU

Для запуска нужны Linux `x86_64`, видеокарта NVIDIA 80 ГБ (A100 или H100, не менее 72 ГиБ свободной VRAM, режим MIG отключён), драйвер NVIDIA `580.95.05+`, Docker с NVIDIA Container Toolkit и от 80 ГиБ свободного места на диске.

### Шаг 1. Клонирование и настройка `.env`

```bash
git clone https://github.com/AskarKasimov/s2s.git
cd s2s
git checkout feat/qwen3-omni-speech-to-speech
cp -n .env.example .env
```

По умолчанию используется `GPU_ID=0`, внешний порт API — `8002`, каталог весов — `./data`. При необходимости измените `GPU_ID`, `S2S_API_PORT`, `S2S_DATA_DIR` или добавьте `HF_TOKEN` в `.env`.

### Шаг 2. Предпусковая проверка сервера

```bash
./run.sh check
```

Если сервер готов, команда выведет: `Проверка хоста пройдена`.

#### Возможные ошибки на Шаге 2 и их решение

- **Не установлен Docker (`Не найдена команда docker`):**
  Установите Docker и добавьте текущего пользователя в группу `docker`:
  ```bash
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  newgrp docker
  ```

- **Нет прав доступа к Docker (`Нет доступа для проверки диска Docker` или `permission denied while trying to connect to the Docker daemon`):**
  Добавьте пользователя в группу `docker` или запустите команду через `sudo`:
  ```bash
  sudo usermod -aG docker "$USER"
  newgrp docker
  ```

- **Старый драйвер (`Драйвер ... слишком старый. Для выбранного CUDA 13 стека нужен 580.95.05 или новее`):**
  Обновите драйвер NVIDIA и перезагрузите сервер:
  ```bash
  sudo apt-get update
  sudo apt-get install -y nvidia-driver-580-server
  sudo reboot
  ```

- **На видеокарте включён режим разделения памяти (`На GPU включён MIG`):**
  Отключите MIG и перезагрузите сервер:
  ```bash
  sudo nvidia-smi -mig 0
  sudo reboot
  ```

- **Занята видеопамять (`Для запуска Qwen3-Omni на чистой GPU нужно не менее 72.0 GiB свободной VRAM`):**
  Проверьте, какие процессы или контейнеры занимают GPU, и остановите их (либо укажите номер другой свободной карты в `GPU_ID` внутри `.env`):
  ```bash
  nvidia-smi
  docker ps
  ```

- **Не хватает места на диске (`На общем диске весов и Docker нужно 80 GiB свободно`):**
  Очистите старый кеш Docker или укажите путь к другому диску в переменной `S2S_DATA_DIR` в `.env`:
  ```bash
  docker system prune -af
  ```

- **Занят порт (`Порт 8002 (или 8091) занят`):**
  Проверьте, не запущен ли сервис уже, либо укажите другой порт в `S2S_API_PORT` / `S2S_PORT` в `.env`:
  ```bash
  ./run.sh status
  ```

### Шаг 3. Сборка, скачивание весов и запуск

Запустите установку и старт сервиса:

```bash
./run.sh install
```

Команда сама соберёт Docker-образ, скачает 26,3 ГиБ весов в `./data` с проверкой SHA-256 всех 13 шардов, поднимет контейнеры `omni` и `api` и прогонит контрольные запросы.

Первый запуск занимает 15–30 минут. Если соединение по SSH нестабильное, команду можно запустить в фоновом режиме через `nohup` (тогда установка продолжится даже при закрытии терминала, а лог будет писаться в файл `install.log`):

```bash
nohup ./run.sh install > install.log 2>&1 &
tail -f install.log
```

#### Возможные ошибки на Шаге 3 и их решение

- **Docker не видит видеокарту (`could not select device driver "nvidia" with capabilities: [[gpu]]`):**
  Установите и подключите `nvidia-container-toolkit`, затем перезапустите Docker:
  ```bash
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

- **Обрыв скачивания весов или ошибка `HTTP 429` от Hugging Face:**
  Добавьте токен Hugging Face в `.env` и запустите установку повторно — уже скачанные в `./data` файлы не удаляются, докачка продолжится с того же места:
  ```bash
  echo "HF_TOKEN=hf_ваш_токен" >> .env
  ./run.sh install
  ```

- **Контейнер `omni` не переходит в статус `healthy` или упал при старте:**
  Соберите диагностический отчёт и посмотрите последние строки лога движка:
  ```bash
  ./run.sh doctor
  tail -n 100 reports/omni.log
  ```

### Шаг 4. Проверка работы и управление сервисом

Проверка готовности после запуска:

```bash
curl http://localhost:8002/health/live
curl http://localhost:8002/health/ready
curl http://localhost:8002/capabilities
```

Команды управления из корня репозитория:

```bash
./run.sh status   # статус контейнеров и память GPU
./run.sh logs     # логи в реальном времени
./run.sh stop     # остановить контейнеры и освободить GPU (веса в data/ сохраняются)
./run.sh start    # запустить заново без повторного скачивания весов
./run.sh verify   # прогнать контрольные запросы к движку и HTTP API
./run.sh doctor   # собрать логи и диагностику в reports/
```

## Ручки HTTP API

Все ручки работают через один запущенный экземпляр модели (до 4 параллельных генераций одновременно; при превышении возвращается `429 QUEUE_FULL` с заголовком `Retry-After: 5`). Входное аудио: WAV, Ogg/Opus, WebM/Opus, MP3, FLAC до 25 МиБ и до 90 секунд. Подробные схемы — в [API.md](API.md), Swagger UI — `http://<адрес-сервера>:8002/docs`.

### Распознавание: `POST /transcribe`

Возвращает дословную расшифровку речи (`text`, `language`, `duration_seconds`) без генерации аудио. Запись без речи отклоняется с `422 NO_SPEECH_DETECTED`:

```bash
curl -X POST http://localhost:8002/transcribe \
  -F 'file=@output/test-question.wav' \
  -F 'language=ru'
```

### Синтез речи: `POST /synthesize`

Дословно озвучивает переданный текст (до 500 символов). Голоса (`speaker`): `ethan` (по умолчанию), `chelsie`, `aiden`. Формат ответа (`response_format`): `wav` или `json`:

```bash
curl -X POST http://localhost:8002/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Объясните, чем постоянные издержки отличаются от переменных.","speaker":"ethan","response_format":"wav"}' \
  --output speech.wav
```

### Чат: `POST /v1/chat/completions` и `POST /chat`

OpenAI-совместимая ручка (`POST /chat` — упрощённый алиас). Поддерживает `modalities: ["text"]` и `["text", "audio"]`, `response_format: {"type": "text"}` и `{"type": "json_object"}`:

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [
      {"role": "system", "content": "Ты преподаватель."},
      {"role": "user", "content": "Верни JSON с полями verdict и feedback."}
    ],
    "modalities": ["text"],
    "temperature": 0.0,
    "max_tokens": 2048,
    "response_format": {"type": "json_object"},
    "stream": false
  }'
```

### Анализ устного ответа: `POST /analyze-audio`

Принимает аудиофайл и критерии оценки в `prompt`. За один проход возвращает расшифровку `transcript` и оценку `analysis` (`json_object` или `text`):

```bash
curl -X POST http://localhost:8002/analyze-audio \
  -F 'file=@output/test-question.wav' \
  -F 'prompt=Оцени ответ студента и верни JSON с полями verdict и feedback.' \
  -F 'response_format=json_object'
```

### Голосовой диалог: `POST /speech-to-speech`

По умолчанию (`response_format=json`) возвращает `transcript`, `text`, `audio` (base64 WAV PCM 16-bit) и `feedback`. При `include_feedback=true` отделяет озвучиваемую реплику от JSON-оценки. При `response_format=wav` возвращает бинарный WAV:

```bash
curl -X POST http://localhost:8002/speech-to-speech \
  -F 'file=@output/test-question.wav' \
  -F 'speaker=ethan' \
  -F 'include_feedback=true'

curl -X POST http://localhost:8002/speech-to-speech \
  -F 'file=@output/test-question.wav' \
  -F 'response_format=wav' \
  --output answer.wav
```

### Метрики

Метрики Prometheus доступны на `GET /metrics`:

```bash
curl http://localhost:8002/metrics
```
