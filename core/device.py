"""Accelerator selection shared by every module: cuda > mps > cpu (override with CLASSIC_ALGOS_DEVICE)."""

from __future__ import annotations

import functools
import os
import platform

import torch


@functools.lru_cache(maxsize=1)
def accel() -> str:
    env = os.environ.get("CLASSIC_ALGOS_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev: str | None = None) -> None:
    dev = dev or accel()
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


@functools.lru_cache(maxsize=1)
def hardware_tag() -> dict:
    """Recorded in every result so Mac and 5090 runs are never silently mixed (wall-clock!)."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ("apple-mps" if torch.backends.mps.is_available() else None)
    return {"host": platform.node(), "machine": platform.machine(), "system": platform.system(),
            "cpu_count": os.cpu_count(), "gpu": gpu, "torch": torch.__version__}
