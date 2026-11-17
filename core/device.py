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


def gpu_ports() -> tuple[str, ...]:
    """Hot loops ported to torch CUDA that are switched on: CLASSIC_ALGOS_GPU_PORTS=gmm,lda,... or 'all'.

    Off by default: a port is used for results only after its parity test (tests/test_gpu_ports.py) passes.
    CUDA only (the ports compute in float64, which MPS lacks).
    """
    env = os.environ.get("CLASSIC_ALGOS_GPU_PORTS", "").strip().lower()
    if not env or accel() != "cuda":
        return ()
    names = ("gmm", "dictlearn", "lda", "hmm", "als")
    return names if env == "all" else tuple(n for n in names if n in {p.strip() for p in env.split(",")})


def gpu_port(name: str) -> bool:
    return name in gpu_ports()


@functools.lru_cache(maxsize=1)
def hardware_tag() -> dict:
    """Recorded in every result so Mac and 5090 runs are never silently mixed (wall-clock!)."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ("apple-mps" if torch.backends.mps.is_available() else None)
    return {"host": platform.node(), "machine": platform.machine(), "system": platform.system(),
            "cpu_count": os.cpu_count(), "gpu": gpu, "torch": torch.__version__, "gpu_ports": list(gpu_ports())}
