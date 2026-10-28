# Qwen3-Omni Speech-to-Speech: развёртывание с нуля на чистой GPU 80 ГБ

Самодостаточный производственный комплект для развёртывания **Qwen3-Omni (`Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound`)** на одной чистой видеокарте **NVIDIA A100 80 ГБ / H100 80 ГБ**.

На одной GPU запускаются все три стадии модели (понимание речи и LLM `Thinker` в INT4 W4A16 + генерация аудиокодов `Talker` в BF16 + синтез звука `Code2Wav` в BF16), а также HTTP API-шлюз:
- **`POST /speech-to-speech`** (порт **`8002`**) — голосовой диалог: принимает аудиофайл (WAV, MP3, OGG, WebM, FLAC) и возвращает голосовой ответ в WAV (или JSON с текстом и аудио в base64).
- **`POST /chat`** (порт **`8002`**) — текстовый диалог с LLM без генерации звука.
- **`GET /health`**, **`GET /metrics`**, **`GET /docs`** (порт **`8002`**) — проверка готовности, метрики Prometheus и интерактивный Swagger UI.
- **`127.0.0.1:8091`** — внутренний OpenAI-compatible движок `vLLM-Omni`.

Бюджет видеопамяти 80 ГБ распределён под чистую карту: **88% (~70,4 GiB)** выделено под три стадии `Qwen3-Omni` (`0.60 + 0.20 + 0.08` в [`deploy.yaml`](deploy.yaml)), а **12% (~9,6 GiB)** оставлено под системный резерв CUDA, контексты и буферы активаций. Перед стартом автоматически проверяется, что на карте свободно **не менее 90% VRAM (~72 GiB)**.

---

## 1. Требования к серверу

| Компонент | Требование |
| :--- | :--- |
| **ОС** | Ubuntu 24.04 LTS (Linux `x86_64`) |
| **GPU** | 1× NVIDIA A100 80 ГБ или H100 80 ГБ (Compute Capability $\ge 8.0$, поддержка BF16, **режим MIG отключён**) |
| **Свободная VRAM** | **$\ge 90\%$ (~72 GiB)** свободной видеопамяти перед стартом (чистая карта без сторонних процессов) |
| **Драйвер NVIDIA** | **`580.95.05` или новее** (внутри контейнера используется стек CUDA 13.0.2 / PyTorch 2.13) |
| **RAM** | **128 ГБ** (минимум 110 GiB доступной оперативной памяти) |
| **Диск** | Минимум **80 GiB** свободно на общем диске (40 GiB под веса + 40 GiB под Docker-образ), рекомендуется **150 ГБ** |
| **ПО на хосте** | Docker Engine, Docker Compose v2 (с поддержкой `--wait-timeout`), NVIDIA Container Toolkit, `git`, `python3`, `curl` |

---

## 2. Подготовка чистого сервера с нуля (если на сервере ещё нет Docker и драйвера)

Если сервер уже настроен (установлены драйвер NVIDIA $\ge 580.95.05$, Docker и NVIDIA Container Toolkit) — **сразу переходите к [Разделу 3](#3-запуск-qwen3-omni-с-первой-попытки-под-ключ)**.

Если перед вами полностью «голая» Ubuntu 24.04 LTS, выполните следующие шаги под администратором (`sudo`):

### Шаг 2.1. Установка базовых утилит и драйвера NVIDIA ($\ge 580.95.05$)
```bash
sudo apt-get update && sudo apt-get install -y git curl python3 ca-certificates gnupg

# Подключение официального репозитория CUDA для Ubuntu 24.04 и установка драйвера 580+
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y nvidia-driver-580-server
sudo reboot
```
После перезагрузки проверьте драйвер и отключите MIG (если он был включён):
```bash
nvidia-smi
# Если в выводе MIG Mode: Enabled, отключите его:
sudo nvidia-smi -mig 0 && sudo reboot
```

### Шаг 2.2. Установка Docker Engine и Docker Compose v2
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

### Шаг 2.3. Установка NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
> **Важно:** после добавления пользователя в группу `docker` переподключитесь по SSH (или выполните `newgrp docker`), чтобы команды `docker info` и `nvidia-smi` выполнялись без `sudo`.

---

## 3. Запуск Qwen3-Omni с первой попытки («под ключ»)

Весь процесс развёртывания полностью автоматизирован скриптом `./run.sh` и не требует ручной сборки окружений, квантования или сторонних утилит (включая `tmux`).

### Шаг 1. Клонируйте репозиторий
```bash
git clone https://github.com/AskarKasimov/s2s.git
cd s2s
```

*(Опционально)* Если на сервере несколько видеокарт или нужно хранить веса на отдельном диске (например, `/mnt/nvme/qwen`), создайте `.env`:
```bash
cp -n .env.example .env
# Отредактируйте GPU_ID или S2S_DATA_DIR при необходимости
```
На стандартном однокарточном сервере (`GPU_ID=0`) менять `.env` **не нужно**.

### Шаг 2. Проверьте готовность сервера (предварительный чек)
```bash
./run.sh check
```
Скрипт мгновенно проверит архитектуру ОС, версию драйвера NVIDIA ($\ge 580.95.05$), объём и чистоту GPU ($\ge 76$ GiB всего, $\ge 90\%$ свободно), оперативную память, свободное место на диске и доступность портов `8002` / `8091`. Ожидаемый вывод:
```text
Проверка хоста пройдена. Доступ GPU из контейнера проверяется после сборки образа.
```

### Шаг 3. Запустите полную установку и верификацию одной командой
```bash
./run.sh install
```
> *Примечание: при первом запуске скачиваются Docker-образ и 26,3 ГБ весов модели (5–15 минут в зависимости от канала), после чего до 10–20 минут занимает компиляция и прогрев 3 стадий на GPU. Не закрывайте SSH-терминал. Если соединение нестабильно, запустите команду в фоне:*
> ```bash
> nohup ./run.sh install > install.log 2>&1 &
> tail -f install.log
> ```

Что делает `./run.sh install` автоматически:
1. **`check`** — проверяет хост, драйвер, свободную VRAM ($\ge 90\%$), RAM, диски и порты.
2. **`build`** — собирает Docker-образ с зафиксированным `vllm-omni` (`9842ebc`), `vLLM 0.32.0` и `CUDA 13.0.2`.
3. **`gpu-check`** — проверяет проброс GPU внутрь контейнера, поддержку `BF16` и матричное умножение на CUDA.
4. **`prepare`** — скачивает 26,3 ГБ весов `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` в `./data` и побайтово сверяет заголовки и размеры всех 13 `safetensors`-шардов.
5. **`start`** — поднимает контейнеры `omni` и `api` и дожидается прохождения healthcheck.
6. **`verify`** — проводит сквозной тест `TEXT → AUDIO → TEXT+AUDIO` напрямую через движок и через HTTP API (`/chat` и `/speech-to-speech`).

**Развёртывание на 100% завершено**, когда в терминале появилась строка:
```text
Сервис запущен, техническая проверка аудио пройдена. Прослушайте output/test-answer.wav.
```

Артефакты контрольной проверки сохраняются в каталог `output/`:
- `output/test-question.wav` — сгенерированный моделью контрольный голосовой вопрос;
- `output/test-answer.wav` и `output/test-answer.txt` — голосовой и текстовый ответ движка;
- `output/api-chat-answer.txt` — ответ текстовой ручки `POST /chat`;
- `output/api-answer.wav` — голосовой ответ ручки `POST /speech-to-speech`.

---

## 4. Использование и проверка HTTP API

Интерактивная документация Swagger UI доступна в браузере по адресу:
**`http://<АДРЕС_СЕРВЕРА>:8002/docs`** (полное описание контракта — в [API.md](API.md)).

### Проверка статуса (`/health`)
```bash
curl --fail http://localhost:8002/health
```
Ожидаемый ответ:
```json
{"status":"ok","model":"qwen3-omni-speech","quantization":"int4-AutoRound"}
```

### Голосовой диалог (`POST /speech-to-speech`)
Принимает аудиофайл в формате WAV, MP3, OGG, WebM или FLAC (до 30 секунд и 10 MiB), автоматически конвертирует его в 16 кГц моно PCM и возвращает готовый WAV-файл с ответом:
```bash
# Ответ чистым WAV-файлом:
curl --fail-with-body -X POST http://localhost:8002/speech-to-speech \
  -F 'file=@output/test-question.wav' \
  --output output/my-reply.wav

# Ответ в JSON (текст ответа + аудио WAV в base64):
curl --fail-with-body -X POST 'http://localhost:8002/speech-to-speech?response_format=json' \
  -F 'file=@output/test-question.wav'
```

### Текстовый чат с LLM (`POST /chat`)
```bash
curl --fail-with-body -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "Отвечай по-русски кратко и по делу."},
      {"role": "user", "content": "Привет! Расскажи в двух предложениях, чем ты можешь помочь."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

---

## 5. Повседневное управление сервисом

Все команды выполняются из корня репозитория:

| Команда | Описание |
| :--- | :--- |
| `./run.sh status` | Показать статус контейнеров `omni` и `api` |
| `./run.sh logs` | Просмотр логов загрузки и работы модели в реальном времени (`Ctrl+C` для выхода) |
| `./run.sh verify` | Повторно прогнать боевой тест `/chat` и `/speech-to-speech` |
| `./run.sh stop` | Остановить контейнеры и освободить GPU (скачанные веса в `data/` сохраняются) |
| `./run.sh start` | Быстрый запуск уже установленного сервиса (без повторного скачивания весов) |
| `./run.sh doctor` | Собрать полную диагностику (`nvidia-smi`, статусы и логи контейнеров) в папку `reports/` |

Также управлять запуском после первой установки можно напрямую через Docker Compose:
```bash
docker compose -f compose.yaml up -d
docker compose -f compose.yaml ps
docker compose -f compose.yaml logs -f
docker compose -f compose.yaml down
```

---

## 6. Решение возможных проблем (Troubleshooting)

| Симптом / сообщение об ошибке | Причина и точное решение |
| :--- | :--- |
| **`Для запуска Qwen3-Omni на чистой GPU нужно не менее 72.0 GiB свободной VRAM (90%)`** | На видеокарте запущены сторонние процессы или старые контейнеры. Проверьте `nvidia-smi` и `docker ps`, остановите лишние контейнеры (`docker stop ...`), чтобы освободить карту. |
| **`Драйвер ... слишком старый. Для выбранного CUDA 13 стека нужен 580.95.05 или новее`** | На хосте установлен старый драйвер NVIDIA. Обновите драйвер по шагу 2.1 (`sudo apt-get install -y nvidia-driver-580-server && sudo reboot`) и повторите `./run.sh check`. |
| **`could not select device driver "nvidia" with capabilities: [[gpu]]`** | Не установлен или не настроен NVIDIA Container Toolkit. Выполните команды из шага 2.3 (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`). |
| **`На общем диске весов и Docker нужно 80 GiB свободно`** | Недостаточно места на корневом разделе. Очистите кэш Docker (`docker system prune -af`) или укажите путь к большому диску в `.env` (`S2S_DATA_DIR=/mnt/bigdisk/qwen-data`). |
| **Обрыв сети или `HTTP 429` при скачивании с Hugging Face** | Просто запустите `./run.sh install` заново — уже скачанные шарды в `data/` сохраняются и докачаются с места обрыва. При жестких rate-лимитах добавьте токен `HF_TOKEN=hf_...` в `.env`. |
| **`Порт 8002 (или 8091) занят`** | Если сервис уже запущен — проверьте `./run.sh status` и `./run.sh verify`. Если порт занят другим приложением на сервере — измените `S2S_API_PORT` или `S2S_PORT` в `.env`. |
| **`HTTP 429: Модель занята. Повторите запрос позже.`** | Шлюз защищает GPU от OOM и пропускает ровно 1 активную генерацию одновременно (общий слот для `/speech-to-speech` и `/chat`). Повторите запрос через 5 секунд. |

---

## 7. Дополнительная документация репозитория

- **[ADMIN.md](ADMIN.md)** — чеклист системного администратора по подготовке сервера и приёмке.
- **[OPERATIONS.md](OPERATIONS.md)** — подробный регламент эксплуатации, предзагрузки без GPU (`build` / `prepare` / `deploy`) и диагностики.
- **[API.md](API.md)** — полная спецификация HTTP API, кодов ответа, ограничений и метрик Prometheus.
- **[SOURCES.md](SOURCES.md)** — обоснование выбора INT4 AutoRound, закреплённых версий `vLLM-Omni` и конфигурации трёх стадий.
- **[docs/qwen3-omni-deployment.md](docs/qwen3-omni-deployment.md)** — архивное исследование архитектуры Qwen3-Omni и расчёта видеопамяти.
