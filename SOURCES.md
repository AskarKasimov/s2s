# Выбор модели и воспроизводимость

Проверено по опубликованным исходникам 27–28 октября 2026 года.

## Почему INT4 AutoRound

В [руководстве vLLM-Omni по AutoRound](https://github.com/vllm-project/vllm-omni/blob/9842ebc17cf1d9c6b440ecb6a79039162f68a1dd/docs/user_guide/quantization/autoround.md)
явно указан `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` и поддержка
NVIDIA Ampere/Hopper (A100/H100). Формат квантования определяется конфигурацией checkpoint;
отдельный глобальный `--quantization` не нужен.

[Конфигурация модели Intel](https://huggingface.co/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound/blob/c3ef7bbf0b9d866866136c1d08b6314d27f7e151/config.json)
задаёт 4 бита, группу 128, `auto_round:auto_gptq` и область
`thinker.model.layers`; `enable_audio_output=true`.
[Индекс весов](https://huggingface.co/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound/blob/c3ef7bbf0b9d866866136c1d08b6314d27f7e151/model.safetensors.index.json)
содержит `Thinker`, `Talker` и `Code2Wav`; суммарный размер тензоров —
26 258 064 482 байта (~24,46 GiB). Использование INT4 для языкового ядра `Thinker` при сохранении `BF16` для голосовых стадий `Talker` и `Code2Wav` гарантирует запуск всех трёх стадий на одной карте 80 ГБ с большим запасом под KV-кеш и рабочие буферы без риска OOM при инициализации.

NVFP4 W4A4 не выбран: его документированный нативный путь для этой модели
рассчитан на архитектуру Blackwell, тогда как целевые 80-ГБ A100/H100 относятся к Ampere/Hopper.
Готовый INT4 AutoRound checkpoint позволяет развернуть модель без калибровки на сервере.

## Зафиксированный runtime

Базовый образ — официальный `vllm/vllm-openai:v0.32.0`, Linux amd64 manifest
`sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9`.

Omni устанавливается из коммита `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`.
Его [официальный Dockerfile](https://github.com/vllm-project/vllm-omni/blob/9842ebc17cf1d9c6b440ecb6a79039162f68a1dd/docker/Dockerfile.cuda)
использует ту же базовую версию vLLM. Мы дополнительно закрепили digest,
предотвратили замену CUDA binary stack constraints-файлом и сохраняем
`/opt/s2s/installed-packages.txt` внутри построенного образа.
Версии OpenTelemetry согласованы на `1.40.0` с учётом требований LMCache; удалены два
неиспользуемых transport/common-пакета от другой линии OpenTelemetry,
попавших в базовый образ. NCCL `2.30.7` сохранён: официальный
[Dockerfile vLLM](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/docker/Dockerfile#L852)
намеренно заменяет строгий pin PyTorch `2.29.7` ради DeepEPv2, которому нужен
NCCL от `2.30.4`. `check_dependencies.py` проверяет все установленные зависимости
и допускает только это точное документированное исключение; остальные
несовместимости прерывают сборку.
Фактически разрешённые версии Python-зависимостей закреплены в `constraints.lock`
и используются при сборке.

Минимальный драйвер `580.95.05` выбран для CUDA 13.0 Update 2 по
[таблице NVIDIA](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html).
Контейнерная проверка выполняет BF16 matmul и синхронизацию CUDA на реальной GPU.

## Конфигурация для чистой GPU 80 ГБ

Основа — [конфиг трёх стадий Omni](https://github.com/vllm-project/vllm-omni/blob/9842ebc17cf1d9c6b440ecb6a79039162f68a1dd/vllm_omni/deploy/qwen3_omni_moe.yaml).
В [`deploy.yaml`](deploy.yaml) все три стадии размещены на `devices: "0"`, а доли `gpu_memory_utilization` заданы под выделенную карту 80 ГБ:
- **Stage 0 (`Thinker`, INT4 W4A16):** `0.60` (~48 GiB)
- **Stage 1 (`Talker`, BF16):** `0.20` (~16 GiB)
- **Stage 2 (`Code2Wav`, BF16):** `0.08` (~6,4 GiB)

Суммарный бюджет стадий составляет **`0.88` (88% VRAM, ~70,4 GiB)**, а **12% (~9,6 GiB)** остаётся свободным запасом для CUDA-контекстов, буферов активаций и межстадийного обмена. Перед стартом `gpu_policy.py` проверяет, что на карте свободно не менее **90% VRAM (~72 GiB)**.

Для BF16-стадий `Talker` и `Code2Wav` метаданные квантования checkpoint'а очищены через `hf_overrides: {quantization_config: null}`.
`enforce_eager: true` отключает компиляцию тяжелых CUDA graphs для предсказуемого потребления памяти и гарантированного старта с первой попытки.
Сохранён upstream CUDA rotary custom op (`+rotary_embedding`), рекомендованный авторами для качества звука.

HTTP-шлюз на порту `8002` принимает multipart upload (`/speech-to-speech`) и JSON-запросы (`/chat`), конвертирует входное аудио через `ffmpeg` и сериализует доступ к GPU (1 активный запрос одновременно, при конкурентном запросе возвращается `HTTP 429` с `Retry-After: 5`).

## Что проверяется автоматически

1. **До загрузки (`./run.sh check`):** ОС (`Linux x86_64`), наличие `docker` и `nvidia-smi`, версия драйвера ($\ge 580.95.05$), объём видеопамяти ($\ge 76$ GiB), чистота GPU ($\ge 90\%$ свободно), объём RAM ($\ge 110$ GiB), свободное место на диске ($\ge 80$ GiB) и свободные порты (`8002`, `8091`).
2. **В контейнере (`gpu-check`):** версия `vLLM`, доступность ровно одной GPU, поддержка `BF16`, контрольное матричное умножение на CUDA и соответствие `deploy.yaml` политике `gpu_policy.json`.
3. **После скачивания (`prepare`):** закреплённая ревизия модели, конфигурация INT4 AutoRound для `Thinker`, наличие всех трёх стадий в индексе, побайтовое соответствие заголовков и размеров всех 13 файлов `safetensors`.
4. **После старта (`verify`):** сквозной тест `TEXT → AUDIO → TEXT+AUDIO` напрямую через движок и проверка обеих ручек HTTP API (`POST /chat` и `POST /speech-to-speech`) с валидацией непустого звукового сигнала (RMS).

Архитектурный разбор первоначального BF16-профиля сохранён в [docs/qwen3-omni-deployment.md](docs/qwen3-omni-deployment.md).
