"""Dedicated 80GB GPU budget validation for Qwen3-Omni."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "gpu_policy.json").read_text())


def validate_gpu(total_bytes, free_bytes, capability):
    if total_bytes < POLICY["minimum_total_gib"] * 1024**3:
        raise RuntimeError("Профиль Qwen3-Omni рассчитан на одну NVIDIA GPU 80 ГБ без MIG.")
    if tuple(capability) < tuple(POLICY["minimum_compute_capability"]):
        raise RuntimeError("Нужна NVIDIA GPU с compute capability от 8.0 для BF16 и INT4-профиля.")
    required = total_bytes * POLICY["minimum_free_fraction"]
    if free_bytes < required:
        raise RuntimeError(
            f"Для запуска Qwen3-Omni на чистой GPU нужно не менее {required / 1024**3:.1f} GiB свободной VRAM (90%); "
            f"доступно {free_bytes / 1024**3:.1f} GiB из {total_bytes / 1024**3:.1f} GiB. "
            "Проверьте процессы на GPU через nvidia-smi и остановите сторонние сервисы."
        )
    return {
        "qwen_budget_gib": round(total_bytes * POLICY["qwen_memory_fraction"] / 1024**3, 2),
        "headroom_gib": round(total_bytes * (1 - POLICY["qwen_memory_fraction"]) / 1024**3, 2),
        "minimum_free_gib": round(required / 1024**3, 2),
    }


def validate_stage_budgets(stages):
    if [(s.stage_id, s.devices) for s in stages] != [(0, "0"), (1, "0"), (2, "0")]:
        raise RuntimeError("Неверное размещение стадий в deploy.yaml.")
    total = sum(stage.gpu_memory_utilization for stage in stages)
    if not math.isclose(total, POLICY["qwen_memory_fraction"], abs_tol=1e-8):
        raise RuntimeError("Бюджеты deploy.yaml не совпадают с GPU-профилем gpu_policy.json.")
