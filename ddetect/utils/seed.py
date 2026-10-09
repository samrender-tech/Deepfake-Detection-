"""Determinism.

Every reported number carries a seed (the experiment grid: three seeds, mean +/- std on
headline rows), so seeding has to cover the dataloader workers too -- an
unseeded worker reshuffles the frame sample and the "same" run drifts.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    ``deterministic=True`` additionally pins cuDNN algorithm choice. It costs
    throughput, so it is off for training and on for the parity tests.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Give each dataloader worker a distinct, reproducible stream."""
    base = torch.initial_seed() % 2**31
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def frame_sample_seed(video_id: str, epoch: int = 0) -> int:
    """Stable per-video seed for frame sampling.

    Derived from the video id rather than global RNG state so the frames chosen
    for a given video are identical in training, evaluation and serving -- which
    is what makes ``test_detector_parity.py`` possible.
    """
    import hashlib

    h = hashlib.sha256(f"{video_id}:{epoch}".encode()).digest()
    return int.from_bytes(h[:4], "big")
