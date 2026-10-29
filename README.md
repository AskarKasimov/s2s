# s2s — Qwen3-Omni (Speech-to-Speech)

Голосовой и текстовый сервис на базе `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` для одной видеокарты **NVIDIA 80 ГБ (A100 / H100)**.
Держит до **4 параллельных запросов** одновременно.

После запуска работает HTTP API на порту **`8002`**:
- `POST /speech-to-speech` — отправить аудио, получить голосовой ответ
- `POST /chat` — обычный текстовый чат с моделью (без озвучки)
- `GET /health` — проверка что сервис жив
- `GET /docs` — Swagger (можно потыкать запросы прямо в браузере)

---

## Что нужно от сервера

- **ОС:** Ubuntu 24.04 (`x86_64`)
- **GPU:** 1× NVIDIA A100 80 ГБ или H100 80 ГБ (свободно от 72 ГБ VRAM, MIG выключен)
- **RAM:** 128 ГБ
- **Диск:** минимум 80 ГБ свободно (лучше 150 ГБ — под Docker-образ и 26.3 ГБ весов)
- **Драйвер NVIDIA:** `580.95.05` или новее (внутри контейнера CUDA 13.0)

---

## Шаг 1. Подготовка сервера (если он совсем пустой)

Зайди на сервер по SSH и проверь, стоят ли уже драйвер и Docker:
```bash
nvidia-smi
docker info
docker compose version
```
Если всё работает и в `nvidia-smi` версия драйвера **`580.95.05` или выше** — **пропускай этот шаг и иди сразу к Шагу 2**.

Если сервер голый или драйвер старый, выполни по очереди:

### 1.1. Поставь драйвер NVIDIA 580+
```bash
sudo apt-get update && sudo apt-get install -y git curl python3 ca-certificates gnupg

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y nvidia-driver-580-server
sudo reboot
```
После перезагрузки зайди снова и проверь `nvidia-smi`. Если там написано `MIG Mode: Enabled`, выключи его:
```bash
sudo nvidia-smi -mig 0 && sudo reboot
```

### 1.2. Поставь Docker и NVIDIA Container Toolkit
```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

# NVIDIA Container Toolkit (чтобы Docker видел видеокарту)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
newgrp docker
```

---

## Шаг 2. Скачай репозиторий и проверь сервер

```bash
git clone https://github.com/AskarKasimov/s2s.git
cd s2s
git checkout feat/qwen3-omni-speech-to-speech
cp -n .env.example .env
```
*(Файл `.env` трогать не надо, если на сервере одна карта `GPU 0` и хватает места на основном диске).*

Запусти проверку сервера:
```bash
./run.sh check
```
Если всё ок, увидишь:
```text
Проверка хоста пройдена. Доступ GPU из контейнера проверяется после сборки образа.
```

---

## Шаг 3. Установка и запуск

Запусти одну команду:
```bash
./run.sh install
```
Она сама соберёт Docker-образ, проверит видеокарту, скачает 26.3 ГБ весов модели в `./data`, поднимет контейнеры и прогонит тестовый голосовой запрос.

Первый запуск занимает **15–30 минут**. Чтобы ничего не оборвалось, если отвалится SSH, лучше запускать так:
```bash
nohup ./run.sh install > install.log 2>&1 &
tail -f install.log
```

Когда всё поднимется и пройдёт проверку, в конце лога будет строка:
```text
Сервис запущен, техническая проверка аудио пройдена. Прослушайте output/test-answer.wav.
```

---

## Шаг 4. Как отправлять запросы

В браузере можно открыть Swagger: `http://<IP_СЕРВЕРА>:8002/docs`

Примеры через `curl`:

### 1. Проверить статус
```bash
curl http://localhost:8002/health
```

### 2. Голосовой запрос -> получить WAV файл с ответом
Принимает WAV, MP3, OGG, WebM, FLAC (до 30 сек и 10 МБ):
```bash
curl -X POST http://localhost:8002/speech-to-speech \
  -F 'file=@output/test-question.wav' \
  --output reply.wav
```

### 3. Голосовой запрос -> получить JSON (текст ответа + аудио в base64)
```bash
curl -X POST 'http://localhost:8002/speech-to-speech?response_format=json' \
  -F 'file=@output/test-question.wav'
```

### 4. Обычный текстовый чат (без генерации звука)
```bash
curl -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "Отвечай кратко по-русски."},
      {"role": "user", "content": "Привет! Сколько будет дважды два?"}
    ]
  }'
```

---

## Команды управления

Запускаются из папки `s2s`:
- `./run.sh status` — посмотреть статус контейнеров
- `./run.sh logs` — смотреть логи в реальном времени (`Ctrl+C` чтобы выйти)
- `./run.sh stop` — остановить сервис и освободить GPU (скачанные веса в `data/` останутся)
- `./run.sh start` — быстро запустить сервис заново (без скачивания весов)
- `./run.sh verify` — ещё раз прогнать тестовые запросы
- `./run.sh doctor` — собрать все логи и состояние GPU в папку `reports/`

---

## Что делать, если вылезла ошибка

**1. `Для запуска Qwen3-Omni на чистой GPU нужно не менее 72.0 GiB свободной VRAM`**
Видеокарта чем-то занята. Посмотри процессы через `nvidia-smi` и `docker ps`, останови лишние контейнеры (`docker stop ...`), чтобы освободить память, и запусти `./run.sh install` заново.

**2. `Драйвер ... слишком старый. Для выбранного CUDA 13 стека нужен 580.95.05 или новее`**
На сервере старый драйвер. Выполни пункт **1.1** выше (`sudo apt-get install -y nvidia-driver-580-server && sudo reboot`).

**3. `could not select device driver "nvidia" with capabilities: [[gpu]]`**
Docker не видит видеокарту. Выполни пункт **1.2** выше (установка `nvidia-container-toolkit` и `sudo systemctl restart docker`).

**4. `На общем диске весов и Docker нужно 80 GiB свободно`**
Не хватает места на диске. Почисти старый мусор Docker (`docker system prune -af`) или пропиши в `.env` путь к другому большому диску: `S2S_DATA_DIR=/mnt/disk/qwen-data`.

**5. Оборвалось скачивание весов или ошибка `HTTP 429` от Hugging Face**
Просто запусти `./run.sh install` ещё раз — то, что уже скачалось в `data/`, не пропадёт и докачается с того же места. Если Hugging Face режет скорость, добавь свой токен в `.env`: `HF_TOKEN=hf_...`.

**6. `Порт 8002 (или 8091) занят`**
Либо сервис уже запущен (проверь через `./run.sh status`), либо порт занят чем-то другим на сервере. Во втором случае поменяй `S2S_API_PORT` в файле `.env`.

**7. API отвечает `HTTP 429: Модель занята`**
Сейчас одновременно уже идут 4 запроса. Подожди пару секунд и отправь запрос снова.

**8. Контейнер `omni` упал или долго висит в `unhealthy`**
Запусти `./run.sh doctor` и открой файл `reports/omni.log` — там в самом низу будет точная причина.
