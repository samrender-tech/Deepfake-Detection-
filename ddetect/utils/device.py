"""Device selection.

The Mac (MPS) is for authoring and smoke tests; CUDA is authoritative for every
reported number (the known-issues list, "MPS kernel gaps"). This module is the single
place that knows the difference, so model code never branches on platform.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import torch

DeviceStr = Literal["cuda", "mps", "cpu"]


@lru_cache(maxsize=1)
def pick_device(prefer: str | None = None) -> torch.device:
    """Resolve a device. ``prefer`` or ``$DDETECT_DEVICE``, else best available.

    MPS emits a warning rather than failing when a kernel is missing, so we set
    the CPU fallback env var *before* any op runs. Without it, a missing kernel
    aborts a training run several epochs in.
    """
    want = (prefer or os.environ.get("DDETECT_DEVICE") or "auto").lower()

    if want == "auto":
        if torch.cuda.is_available():
            want = "cuda"
        elif torch.backends.mps.is_available():
            want = "mps"
        else:
            want = "cpu"

    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DDETECT_DEVICE=cuda but torch.cuda.is_available() is False")
    if want == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("DDETECT_DEVICE=mps but MPS is unavailable")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    return torch.device(want)


def device_report(device: torch.device | None = None) -> dict[str, object]:
    """Provenance for the run directory. Stamped into every ``config.yaml``."""
    dev = device or pick_device()
    rep: dict[str, object] = {
        "device": dev.type,
        "torch": torch.__version__,
        "authoritative": dev.type == "cuda",
    }
    if dev.type == "cuda":
        rep["gpu_name"] = torch.cuda.get_device_name(0)
        rep["gpu_count"] = torch.cuda.device_count()
        rep["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    elif dev.type == "mps":
        rep["note"] = "MPS: authoring/smoke only, not authoritative for reported metrics"
    return rep


def supports_amp(device: torch.device | None = None) -> bool:
    """AMP is CUDA-only here. MPS autocast is still unreliable for our ops."""
    return (device or pick_device()).type == "cuda"


def amp_dtype(device: torch.device | None = None) -> torch.dtype:
    dev = device or pick_device()
    if dev.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
