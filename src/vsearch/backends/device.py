"""Єдина точка вибору обчислювального пристрою.

Розробка йде на Mac (MPS), прогони — на CPU та CUDA, продакшн — тільки CUDA.
Щоб три середовища не розʼїхалися, вибір пристрою й dtype живе тут, а не
розсипається по модулях через `.cuda()` чи `torch.device("mps")` на місцях.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

DeviceName = Literal["cuda", "mps", "cpu"]

_ENV_OVERRIDE = "VSEARCH_DEVICE"


@dataclass(frozen=True)
class DeviceSpec:
    """Пристрій разом із dtype, який на ньому реально працює."""

    device: DeviceName
    dtype: str  # "float16" | "bfloat16" | "float32"

    @property
    def is_accelerated(self) -> bool:
        return self.device in ("cuda", "mps")


def available_devices() -> list[DeviceName]:
    """Які пристрої доступні тут і зараз. Без torch повертає лише CPU."""
    try:
        import torch
    except ImportError:
        return ["cpu"]

    found: list[DeviceName] = []
    if torch.cuda.is_available():
        found.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        found.append("mps")
    found.append("cpu")
    return found


def resolve(preferred: DeviceName | None = None) -> DeviceSpec:
    """Обрати пристрій: явний аргумент → env VSEARCH_DEVICE → автовизначення.

    dtype підбирається під пристрій, а не навпаки:
      * cuda — bfloat16, найширший динамічний діапазон із половинної точності;
      * mps  — float16, бо bfloat16 на Metal підтриманий нерівно;
      * cpu  — float32, half на CPU здебільшого повільніший за full.
    """
    choice = preferred or os.environ.get(_ENV_OVERRIDE) or None

    if choice is None:
        choice = available_devices()[0]
    elif choice not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Невідомий пристрій {choice!r}; очікується cuda, mps або cpu")

    dtype = {"cuda": "bfloat16", "mps": "float16", "cpu": "float32"}[choice]
    return DeviceSpec(device=choice, dtype=dtype)  # type: ignore[arg-type]


def torch_device_and_dtype(spec: DeviceSpec):
    """Перекласти DeviceSpec у пару обʼєктів torch. Імпорт torch — лениво."""
    import torch

    return torch.device(spec.device), getattr(torch, spec.dtype)
