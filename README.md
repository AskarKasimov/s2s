# s2s

HTTP API распознавания речи, синтеза, анализа устных ответов и голосового диалога на базе единой модели `Qwen/Qwen3-Omni-30B-A3B-Instruct`.
Репозиторий содержит конфигурацию движка `vLLM-Omni`, HTTP-шлюз, автопроверки и Docker-окружение для развёртывания на выделенном GPU-сервере.

Документация и контракты репетитора находятся в [ai-tutor](https://github.com/AskarKasimov/ai-tutor). Полная спецификация запросов, ответов и кодов ошибок описана в [API.md](API.md), эксплуатационные детали — в [OPERATIONS.md](OPERATIONS.md), закреплённые версии и хеши весов — в [SOURCES.md](SOURCES.md) и [versions.json](versions.json).

## Архитектура и модель

Сервис запускает один общий checkpoint [`Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound`](https://huggingface.co/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound) (ревизия `c3ef7bbf0b9d866866136c1d08b6314d27f7e151`, 13 шардов `safetensors`, 26,3 ГиБ) через движок [`vllm-project/vllm-omni`](https://github.com/vllm-project/vllm-omni) (коммит `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`).

Внутри контейнера `omni` работают три стадии модели на одной видеокарте 80 ГБ:
- **Thinker** (`0.60` памяти GPU) — распознавание входного аудио, текстовый диалог и проверка ответов;
- **Talker** (`0.20` памяти GPU) — генерация речевых кодов (голоса `ethan`, `chelsie`, `aiden`);
- **Code2Wav** (`0.08` памяти GPU) — декодирование речевых кодов в WAV PCM 16-bit (24 кГц, моно).

Суммарный бюджет трех стадий составляет 88% VRAM (~70,4 ГиБ), до 4 параллельных генераций одновременно (`max_num_seqs: 4`, контекст `4096` токенов). Контейнер `api` публикует внешний HTTP-шлюз на порту `8002` без прямого доступа к GPU.

## Требования к серверу

- **ОС и платформа:** Ubuntu 24.04 LTS (`linux/amd64`).
- **GPU:** 1× NVIDIA A100-SXM4-80GB или H100 80GB (свободно не менее 72 ГиБ VRAM, режим MIG отключён).
- **Драйвер NVIDIA:** `580.95.05` или новее (в базовом образе используется стек CUDA 13.0).
- **ОЗУ:** от 64 ГиБ (рекомендуется 128 ГиБ).
- **Диск:** не менее 80 ГиБ свободного места (рекомендуется 150 ГиБ на SSD/NVMe под Docker-образ и кеш весов).

## Пошаговый запуск на сервере с нуля

### Шаг 1. Установка драйвера NVIDIA и Docker

Проверьте текущее окружение на сервере:

```bash
nvidia-smi
docker info
docker compose version
```

Если драйвер NVIDIA версии `580.95.05` или новее уже установлен, видеокарта видна, а Docker и Docker Compose работают, переходите сразу к **Шагу 2**.

Если сервер чистый или версия драйвера ниже `580.95.05`, выполните установку драйвера:

```bash
sudo apt-get update && sudo apt-get install -y git curl python3 ca-certificates gnupg

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y nvidia-driver-580-server
sudo reboot
```

После перезагрузки проверьте вывод `nvidia-smi`. Если на карте включён режим `MIG Mode: Enabled`, отключите его:

```bash
sudo nvidia-smi -mig 0
sudo reboot
```

Установите Docker Engine и NVIDIA Container Toolkit для проброса GPU в контейнеры:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
newgrp docker
```

### Шаг 2. Клонирование репозитория и настройка окружения

```bash
git clone https://github.com/AskarKasimov/s2s.git
cd s2s
git checkout feat/qwen3-omni-speech-to-speech
cp -n .env.example .env
```

По умолчанию в `.env` выбрана карта `GPU_ID=0`, внешний порт API `S2S_API_PORT=8002` и каталог весов `S2S_DATA_DIR=./data`. При необходимости измените эти параметры в `.env` (например, укажите `HF_TOKEN` для ускорения загрузки с Hugging Face или перенесите `S2S_DATA_DIR` на отдельный диск).

Выполните предпусковую проверку хоста (проверяет драйвер, объём свободной VRAM, отключение MIG, свободное место на диске и порты):

```bash
./run.sh check
```

### Шаг 3. Сборка, загрузка весов и запуск

Запустите полную установку из корня репозитория:

```bash
./run.sh install
```

Команда выполняет последовательно:
1. проверку сервера (`check`);
2. сборку Docker-образа с проверкой дерева Python-зависимостей (`build`);
3. скачивание весов модели в `data/hf` и побайтовую сверку SHA-256 всех 13 шардов (`prepare`);
4. запуск сервисов `omni` и `api` (`start`);
5. сквозную проверку генерации текста и речи через движок и HTTP API (`verify`).

Первый запуск занимает 15–30 минут (скачивание 26,3 ГиБ весов и компиляция CUDA-графов при старте движка). Чтобы процесс не прервался при разрыве SSH-сессии, рекомендуется запускать его в фоне:

```bash
nohup ./run.sh install > install.log 2>&1 &
tail -f install.log
```

После успешного запуска сервис слушает порт `8002`. Проверить готовность можно запросами:

```bash
curl http://localhost:8002/health/live
curl http://localhost:8002/health/ready
curl http://localhost:8002/capabilities
```

## Использование HTTP API

Подробное описание всех параметров и кодов ошибок приведено в [API.md](API.md). Интерактивная документация OpenAPI доступна по адресу `http://<адрес-сервера>:8002/docs`.

### Распознавание речи: `POST /transcribe`

Принимает `multipart/form-data` с полем `file` (WAV, Ogg/Opus, WebM/Opus, MP3, FLAC до 25 МиБ и до 90 секунд) и необязательным полем `language` (`ru` по умолчанию или `auto`). Возвращает дословную расшифровку без генерации аудио:

```bash
curl -X POST http://localhost:8002/transcribe \
  -F 'file=@output/test-question.wav' \
  -F 'language=ru'
```

Ответ содержит `text`, `language` и `duration_seconds`. Запись с тишиной отклоняется с кодом `422 NO_SPEECH_DETECTED`.

### Синтез речи: `POST /synthesize`

Озвучивает переданный текст дословно (до 500 символов). Доступные голоса (`speaker`): `ethan` (по умолчанию), `chelsie`, `aiden`. Поле `response_format` принимает `wav` (по умолчанию) или `json`:

```bash
curl -X POST http://localhost:8002/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Объясните, чем постоянные издержки отличаются от переменных.","speaker":"ethan","response_format":"wav"}' \
  --output speech.wav
```

### Текстовые и мультимодальные задачи: `POST /v1/chat/completions`

OpenAI-совместимая ручка (`POST /chat` доступен как упрощённый алиас). Поддерживает `modalities: ["text"]` и `["text", "audio"]`, `response_format: {"type": "text"}` и `{"type": "json_object"}`:

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [
      {"role": "system", "content": "Ты преподаватель. Используй только переданные материалы."},
      {"role": "user", "content": "Оцени ответ ученика и верни JSON с полями verdict и feedback."}
    ],
    "modalities": ["text"],
    "temperature": 0.0,
    "max_tokens": 2048,
    "response_format": {"type": "json_object"},
    "stream": false
  }'
```

### Анализ устного ответа: `POST /analyze-audio`

Принимает аудиофайл ответа студента и критерии проверки в `prompt` (а также необязательные `system_prompt`, `messages` и `response_format: json_object | text`). За один вызов возвращает дословную расшифровку `transcript` и результат разбора `analysis` без синтеза речи:

```bash
curl -X POST http://localhost:8002/analyze-audio \
  -F 'file=@output/test-question.wav' \
  -F 'prompt=Оцени ответ по критериям и верни JSON с полями verdict и feedback.' \
  -F 'response_format=json_object'
```

### Голосовой диалог: `POST /speech-to-speech`

Принимает аудиореплику студента (`file`), необязательные `system_prompt`, `prompt`, `messages`, `speaker` (`ethan`, `chelsie`, `aiden`), `response_format` (`json` по умолчанию или `wav`) и флаг `include_feedback` (`false` по умолчанию). При `include_feedback=true` публичная голосовая реплика наставника отделяется от внутреннего JSON-блока оценки:

```bash
curl -X POST http://localhost:8002/speech-to-speech \
  -F 'file=@output/test-question.wav' \
  -F 'speaker=ethan' \
  -F 'include_feedback=true'
```

Для получения только WAV-файла укажите `response_format=wav`:

```bash
curl -X POST http://localhost:8002/speech-to-speech \
  -F 'file=@output/test-question.wav' \
  -F 'response_format=wav' \
  --output answer.wav
```

## Управление сервисом и мониторинг

Все команды выполняются из корня репозитория через `./run.sh`:

```bash
./run.sh status   # состояние контейнеров и использование памяти GPU
./run.sh logs     # просмотр логов omni и api в реальном времени
./run.sh verify   # повторный прогон сквозной проверки на живом сервисе
./run.sh stop     # остановка контейнеров и освобождение VRAM (веса в data/ сохраняются)
./run.sh start    # быстрый запуск сервиса с уже подготовленными весами
./run.sh test     # запуск CPU unit- и интеграционных тестов в изолированном контейнере
./run.sh doctor   # сбор диагностического отчёта и логов в каталог reports/
```

Метрики Prometheus доступны по адресу `http://localhost:8002/metrics`.

## Решение частых проблем при развёртывании

- **Недостаточно свободной видеопамяти (`нужно не менее 72.0 GiB свободной VRAM`):** на выбранной видеокарте запущены сторонние процессы. Проверьте список процессов в `nvidia-smi` и `docker ps`, остановите лишние контейнеры или укажите индекс другой свободной 80 ГБ карты в `GPU_ID` в файле `.env`.
- **Старый драйвер NVIDIA (`нужен 580.95.05 или новее`):** обновите драйвер по командам из **Шага 1** и перезагрузите сервер.
- **Ошибка Docker `could not select device driver "nvidia" with capabilities: [[gpu]]`:** не настроен `nvidia-container-toolkit`. Выполните команды настройки runtime из **Шага 1** и перезапустите демон Docker (`sudo systemctl restart docker`).
- **Нехватка места на диске (`нужно 80 GiB свободно`):** очистите неиспользуемые образы (`docker system prune -af`) либо укажите путь к разделу с достаточным объёмом в `S2S_DATA_DIR` внутри `.env`.
- **Обрыв загрузки весов или `HTTP 429` от Hugging Face:** добавьте токен `HF_TOKEN=hf_...` в `.env` и повторно запустите `./run.sh prepare` (или `./run.sh install`) — докачка продолжится с места остановки.
- **Занят порт `8002` или `8091`:** проверьте, не запущен ли сервис ранее (`./run.sh status`), либо измените `S2S_API_PORT` / `S2S_PORT` в `.env`.
- **Контейнер `omni` не переходит в статус `healthy`:** выполните `./run.sh doctor` и проверьте конец файла `reports/omni.log`.
