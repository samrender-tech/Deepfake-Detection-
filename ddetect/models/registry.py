"""Model registry.

``build_model(cfg)`` is the single construction path, used by training,
``Detector``, the API and ``tests/test_contracts.py``. Named presets map the
project's experiment IDs onto configurations so an experiment is reproduced by
name rather than by remembering eight flags.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ddetect.models.audio import AudioConfig
from ddetect.models.avforge import AVForge, AVForgeConfig
from ddetect.models.fusion import FusionConfig
from ddetect.models.syncnet import SyncConfig
from ddetect.models.visual import VisualConfig

#: Experiment presets. Keep these in sync with configs/exp/.
PRESETS: dict[str, dict[str, Any]] = {
    # --- E1 / E2: the honest baseline. Plain CNN on face crops, mean pooled.
    "baseline": {
        "visual": {"backbone": "xception", "temporal": "mean"},
    },
    # --- E5: visual generalisation stack
    "v_full": {
        "visual": {
            "backbone": "effb4",
            "use_freq": True,
            "use_blend_head": True,
            "temporal": "transformer",
        },
    },
    # --- E6 / E7: the audio-visual proposal
    "avforge": {
        "visual": {
            "backbone": "effb4",
            "use_freq": True,
            "use_blend_head": True,
            "temporal": "transformer",
        },
        "audio": {"kind": "logmel"},
        "sync": {"kind": "contrastive", "freeze": False},
        "fusion": {"mode": "coattn", "modality_dropout": 0.3, "use_gate": True},
    },
    "avforge_wavlm": {
        "visual": {
            "backbone": "effb4",
            "use_freq": True,
            "use_blend_head": True,
            "temporal": "transformer",
        },
        "audio": {"kind": "wavlm"},
        "sync": {"kind": "contrastive", "freeze": False},
        "fusion": {"mode": "coattn"},
    },
    # --- the CPU serving profile
    "lite": {
        "visual": {"backbone": "effb0", "image_size": 224, "temporal": "attention"},
        "audio": {"kind": "logmel", "width": 16},
        "sync": {"kind": "contrastive", "freeze": True},
        "fusion": {"mode": "concat", "dim": 128},
    },
    # --- smoke tests: random weights, tiny, no downloads
    "smoke": {
        "visual": {
            "backbone": "effb0",
            "pretrained": False,
            "image_size": 224,
            "temporal": "attention",
            "embed_dim": 128,
        },
        "audio": {"kind": "logmel", "width": 8, "out_dim": 32},
        "sync": {"kind": "contrastive", "freeze": False, "embed_dim": 32},
        "fusion": {"mode": "coattn", "dim": 64, "depth": 1, "heads": 2},
    },
}


def make_config(spec: str | dict[str, Any] | AVForgeConfig) -> AVForgeConfig:
    """Resolve a preset name or a plain dict into an ``AVForgeConfig``."""
    if isinstance(spec, AVForgeConfig):
        return spec
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise KeyError(f"unknown preset {spec!r}; known: {sorted(PRESETS)}")
        spec = PRESETS[spec]

    d = dict(spec)
    return AVForgeConfig(
        visual=VisualConfig(**d.get("visual", {})),
        audio=AudioConfig(**d["audio"]) if d.get("audio") else None,
        sync=SyncConfig(**d["sync"]) if d.get("sync") else None,
        fusion=FusionConfig(**d.get("fusion", {})),
    )


def build_model(spec: str | dict[str, Any] | AVForgeConfig) -> nn.Module:
    """The single model construction path."""
    return AVForge(make_config(spec))


def describe(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    streams = []
    if getattr(model, "visual", None) is not None:
        streams.append(f"visual({model.cfg.visual.backbone},{model.cfg.visual.temporal})")  # type: ignore[union-attr]
    if getattr(model, "audio", None) is not None:
        streams.append(f"audio({model.cfg.audio.kind})")  # type: ignore[union-attr]
    if getattr(model, "sync", None) is not None:
        streams.append(f"sync({model.cfg.sync.kind})")  # type: ignore[union-attr]
    if getattr(model, "fusion", None) is not None:
        streams.append(f"fusion({model.cfg.fusion.mode})")  # type: ignore[union-attr]
    return f"{' + '.join(streams)}  {total / 1e6:.1f}M params ({train / 1e6:.1f}M trainable)"
