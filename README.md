# Qwen3-Omni Speech-to-Speech: пошаговая инструкция развёртывания с нуля на GPU 80 ГБ

Полный производственный комплект для запуска голосовой и текстовой модели **Qwen3-Omni (`Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound`)** на одной чистой видеокарте **NVIDIA A100 80 ГБ / H100 80 ГБ**.

После развёртывания на сервере поднимаются:
- **Порт `8002` (`http://<IP_СЕРВЕРА>:8002`)** — внешний HTTP API для приложения:
  - `POST /speech-to-speech` — голосовой диалог (принимает аудио WAV/MP3/OGG/WebM/FLAC $\rightarrow$ возвращает голосовой ответ WAV или JSON с текстом и аудио в `base64`).
  - `POST /chat` — текстовый диалог с LLM без генерации аудио.
  - `GET /health` — проверка готовности всех стадий модели.
  - `GET /docs` — интерактивный Swagger UI (можно протестировать прямо в браузере).
  - `GET /metrics` — метрики Prometheus.
- **Порт `127.0.0.1:8091`** — внутренний OpenAI-compatible движок `vLLM-Omni` (три стадии на одной GPU: `Thinker` INT4 W4A16 + `Talker` BF16 + `Code2Wav` BF16).

Поддерживается до **4 параллельных запросов одновременно** (`max_num_seqs: 4`). Память карты 80 ГБ распределена так: **88% (~70,4 GiB)** под три стадии модели (`0.60 + 0.20 + 0.08` в [`deploy.yaml`](deploy.yaml)) и **12% (~9,6 GiB)** под системный резерв CUDA.

---

## Быстрый старт (шпаргалка на 4 команды, если на сервере уже есть драйвер NVIDIA 580+ и Docker)

Если вы арендовали готовый GPU-сервер с установленным драйвером NVIDIA ($\ge 580.95.05$) и Docker, просто выполните:

```bash
git clone https://github.com/AskarKasimov/s2s.git
cd s2s
git checkout feat/qwen3-omni-speech-to-speech
./run.sh check && ./run.sh install
```

Если же сервер **полностью чистый («голая» Ubuntu)** или вы хотите пройти весь путь по шагам — следуйте подробной инструкции ниже.

---

## Полная пошаговая инструкция развёртывания с нуля

### Шаг 0. Проверьте характеристики арендованного сервера

Перед началом убедитесь, что сервер соответствует минимальным требованиям:
- **ОС:** Ubuntu 24.04 LTS (`Linux x86_64`)
- **Видеокарта:** 1× NVIDIA A100 80 ГБ или H100 80 ГБ (свободно не менее **90% VRAM / ~72 GiB**, режим MIG отключён)
- **Оперативная память (RAM):** **128 ГБ** (минимум 110 GiB доступно)
- **Свободное место на диске:** минимум **80 GiB** на одном диске (рекомендуется **150 ГБ**: ~40 GiB под Docker-образ и ~40 GiB под веса модели)
- **Драйвер NVIDIA:** **`580.95.05` или новее** (внутри контейнера используется стек CUDA 13.0.2)

Подключитесь к серверу по SSH:
```bash
ssh <ПОЛЬЗОВАТЕЛЬ>@<IP_СЕРВЕРА>
```

---

### Шаг 1. Проверьте, установлены ли драйвер NVIDIA и Docker

Выполните на сервере:
```bash
nvidia-smi
docker info
docker compose version
```

- **Если все 3 команды отработали без ошибок**, а в первой строке `nvidia-smi` указана версия **`Driver Version: 580.95.05` (или выше, например `580.x` / `590.x`)** — **переходите сразу к [Шагу 3](#шаг-3-клонируйте-репозиторий-и-настройте-окружение)**.
- **Если команды не найдены или драйвер старый** (например, `535.x` / `550.x` / `570.x`) — выполните **Шаг 2**.

---

### Шаг 2. Установка драйвера NVIDIA ($\ge 580.95.05$), Docker и NVIDIA Container Toolkit на чистую Ubuntu 24.04

#### 2.1. Установка базовых пакетов и свежего драйвера NVIDIA 580+
```bash
sudo apt-get update && sudo apt-get install -y git curl python3 ca-certificates gnupg

# Подключаем официальный репозиторий NVIDIA CUDA для Ubuntu 24.04
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update

# Устанавливаем серверный драйвер ветки 580 и перезагружаем сервер
sudo apt-get install -y nvidia-driver-580-server
sudo reboot
```

Подождите 1–2 минуты, снова зайдите по SSH и проверьте `nvidia-smi`:
```bash
nvidia-smi
```
> **Важно про режим MIG:** если в таблице `nvidia-smi` в графе `MIG M.` написано `Enabled`, отключите его, чтобы вся память 80 ГБ была доступна целиком:
> ```bash
> sudo nvidia-smi -mig 0
> sudo reboot
> ```

#### 2.2. Установка Docker Engine и Docker Compose v2
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

#### 2.3. Установка NVIDIA Container Toolkit (чтобы Docker видел видеокарту)
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Примените группу `docker` без переподключения по SSH:
```bash
newgrp docker
```

---

### Шаг 3. Клонируйте репозиторий и настройте окружение

Склонируйте репозиторий и перейдите в папку проекта:
```bash
git clone https://github.com/AskarKasimov/s2s.git
cd s2s
git checkout feat/qwen3-omni-speech-to-speech
```

Создайте конфигурационный файл `.env` из шаблона:
```bash
cp -n .env.example .env
```

- **Если у вас стандартный сервер с одной видеокартой (`GPU 0`) и достаточно места на текущем диске ($\ge 80$ GiB)** — **менять в `.env` ничего не нужно**.
- Если нужно выбрать другую видеокарту (например, `GPU_ID=1`) или сохранить веса на другой примонтированный диск (например, `S2S_DATA_DIR=/mnt/nvme/qwen-data`), откройте `.env` через `nano .env` и измените нужную строку.

---

### Шаг 4. Запустите предпусковую проверку сервера (`./run.sh check`)

Перед тем как качать тяжелые образы и 26,3 ГБ весов, проверьте готовность сервера за 1 секунду:
```bash
./run.sh check
```

Скрипт проверит ОС, версию драйвера, объём и чистоту видеокарты ($\ge 90\%$ свободной VRAM), оперативную память, свободное место на диске и порты `8002` / `8091`.

**Ожидаемый успешный вывод:**
```json
{
  "gpu": "NVIDIA A100-SXM4-80GB",
  "gpu_id": "0",
  "driver": "580.95.05",
  "qwen_budget_gib": 70.4,
  "headroom_gib": 9.6,
  "minimum_free_gib": 72.0,
  "ram_gib": 125.8,
  "data_dir": "./data",
  "data_free_gib": 240.5,
  "docker_free_gib": 240.5,
  "port": 8091
}
Проверка хоста пройдена. Доступ GPU из контейнера проверяется после сборки образа.
```
*(Если на этом шаге вылезла ошибка — посмотрите точную команду для её исправления в [Разделе «Частые ошибки»](#шаг-7-частые-ошибки-и-их-решение-troubleshooting) ниже).*

---

### Шаг 5. Запустите полную установку и запуск одной командой (`./run.sh install`)

Запустите установку:
```bash
./run.sh install
```

> **Совет на случай плохого интернета:** первый запуск занимает **15–30 минут** (сборка образа + скачивание 26,3 ГБ весов с Hugging Face + загрузка и компиляция 3 стадий на GPU). Чтобы случайный разрыв SSH-соединения с вашего ноутбука не прервал установку, можете запустить её в фоновом режиме и смотреть лог:
> ```bash
> nohup ./run.sh install > install.log 2>&1 &
> tail -f install.log
> ```

**Что `./run.sh install` делает сам без вашего участия:**
1. **`check`** — проверяет хост и свободную память GPU.
2. **`build`** — собирает Docker-образ `s2s/qwen3-omni-int4:9842ebc` со всеми закреплёнными библиотеками.
3. **`gpu-check`** — запускает тестовый контейнер с GPU и проверяет реальное матричное умножение `BF16` на CUDA.
4. **`prepare`** — скачивает 26,3 ГБ весов модели `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` в папку `./data` и побайтово проверяет все 13 файлов `.safetensors`.
5. **`start`** — запускает контейнеры `omni` и `api` и ждёт, пока все 3 стадии загрузятся в видеокарту.
6. **`verify`** — проводит боевую самопроверку: генерирует контрольный голосовой вопрос, отправляет его в модель и проверяет как текстовый ответ (`POST /chat`), так и голосовой ответ (`POST /speech-to-speech`).

**Успешный финал установки выглядит ровно так:**
```text
Сервис запущен, техническая проверка аудио пройдена. Прослушайте output/test-answer.wav.
```

В папке `output/` появятся файлы тестового прогона:
- `output/test-question.wav` — контрольный вопрос голосом;
- `output/test-answer.wav` и `output/test-answer.txt` — голосовой и текстовый ответ напрямую от движка;
- `output/api-chat-answer.txt` — текстовый ответ от HTTP API `POST /chat`;
- `output/api-answer.wav` — голосовой ответ от HTTP API `POST /speech-to-speech`.

---

### Шаг 6. Как пользоваться запущенным API (проверка запросов)

Откройте в браузере интерактивную документацию Swagger UI:
**`http://<IP_СЕРВЕРА>:8002/docs`** (там можно нажать **Try it out**, загрузить любой аудиофайл с компьютера и скачать голосовой ответ).

Либо проверьте запросы через `curl` в терминале:

#### 1. Проверка статуса (`/health`)
```bash
curl --fail http://localhost:8002/health
```
Ответ:
```json
{"status":"ok","model":"qwen3-omni-speech","quantization":"int4-AutoRound"}
```

#### 2. Голосовой диалог: отправить аудио $\rightarrow$ получить аудио (`POST /speech-to-speech`)
Поддерживаются форматы **WAV, MP3, OGG, WebM, FLAC** (до 30 секунд и 10 МБ):
```bash
curl --fail-with-body -X POST http://localhost:8002/speech-to-speech \
  -F 'file=@output/test-question.wav' \
  --output output/my-reply.wav
```

#### 3. Голосовой диалог: получить сразу и ТЕКСТ ответа, и АУДИО в одном JSON (`?response_format=json`)
Удобно для бэкенда, чтобы одновременно отобразить текст реплики тьютора и проиграть звук:
```bash
curl --fail-with-body -X POST 'http://localhost:8002/speech-to-speech?response_format=json' \
  -F 'file=@output/test-question.wav'
```
Формат ответа:
```json
{
  "text": "Ясное дневное небо обычно голубого или синего цвета.",
  "model": "qwen3-omni-speech",
  "audio": {
    "format": "wav",
    "sample_rate": 24000,
    "data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."
  }
}
```

#### 4. Текстовый запрос к LLM без генерации звука (`POST /chat`)
```bash
curl --fail-with-body -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "Отвечай по-русски кратко и по делу."},
      {"role": "user", "content": "Сколько будет два плюс два?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

---

## Управление сервисом после установки

Все команды выполняются из корня папки `s2s`:

| Команда | Что делает |
| :--- | :--- |
| `./run.sh status` | Показать статус контейнеров `omni` и `api` |
| `./run.sh logs` | Смотреть логи модели и API в реальном времени (`Ctrl+C` для выхода) |
| `./run.sh verify` | Повторно прогнать тестовые запросы `/chat` и `/speech-to-speech` |
| `./run.sh stop` | Остановить контейнеры и освободить GPU (скачанные веса в `data/` сохраняются) |
| `./run.sh start` | Быстро запустить сервис заново без пересборки и без повторного скачивания весов |
| `./run.sh doctor` | Собрать полную диагностику (`nvidia-smi`, логи, статусы) в папку `reports/` |

---

## Шаг 7. Частые ошибки и их решение (Troubleshooting)

| Ошибка / ситуация | Причина и что сделать |
| :--- | :--- |
| **`Для запуска Qwen3-Omni на чистой GPU нужно не менее 72.0 GiB свободной VRAM (90%)`** | На видеокарте запущены сторонние процессы или старые контейнеры. Проверьте `nvidia-smi` и `docker ps`, остановите лишние контейнеры (`docker stop ...`), чтобы освободить видеопамять. |
| **`Драйвер ... слишком старый. Для выбранного CUDA 13 стека нужен 580.95.05 или новее`** | На сервере старый драйвер NVIDIA. Установите драйвер 580+ по [Шагу 2.1](#21-установка-базовых-пакетов-и-свежего-драйвера-nvidia-580) (`sudo apt-get install -y nvidia-driver-580-server && sudo reboot`). |
| **`could not select device driver "nvidia" with capabilities: [[gpu]]`** | Не настроен NVIDIA Container Toolkit. Выполните [Шаг 2.3](#23-установка-nvidia-container-toolkit-чтобы-docker-видел-видеокарту) (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`). |
| **`На общем диске весов и Docker нужно 80 GiB свободно`** | Мало места на текущем диске. Очистите старый кэш Docker (`docker system prune -af`) или укажите в `.env` путь к другому диску (`S2S_DATA_DIR=/mnt/disk/qwen-data`). |
| **Обрыв сети или `HTTP 429` при скачивании весов с Hugging Face** | Просто запустите `./run.sh install` ещё раз — уже скачанные файлы в `data/` сохранены и докачаются с места обрыва. При жёстких лимитах HF укажите токен `HF_TOKEN=hf_...` в `.env`. |
| **`Порт 8002 (или 8091) занят`** | Если Qwen уже запущен — проверьте `./run.sh status` и `./run.sh verify`. Если порт занят другим процессом — поменяйте `S2S_API_PORT` в `.env`. |
| **`HTTP 429: Модель занята. Повторите запрос позже.`** | Одновременно уже выполняются 4 запроса (максимальный пул `S2S_MAX_CONCURRENT=4`). Повторите запрос через 5 секунд. |

---

## Дополнительная документация

- **[API.md](API.md)** — полная спецификация HTTP API, форматов запросов/ответов и метрик Prometheus.
- **[ADMIN.md](ADMIN.md)** — краткий чеклист системных требований для администратора сервера.
- **[OPERATIONS.md](OPERATIONS.md)** — регламент обслуживания, предзагрузки без GPU (`build` / `prepare` / `deploy`) и разбора логов.
- **[SOURCES.md](SOURCES.md)** — техническое описание архитектуры стадий, квантования INT4 AutoRound и закреплённых версий.
