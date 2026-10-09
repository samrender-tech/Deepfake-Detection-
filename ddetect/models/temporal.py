"""F7d - temporal aggregation over frame embeddings.

A per-frame CNN plus a temporal head beats a 3-D/video backbone at our data
scale: the spatial backbone can be ImageNet-pretrained and the temporal part
stays small enough to train on ~1k videos. (A VideoMAE variant is benchmarked
against this in the ablation grid.)

Three poolings, selected by config, all reported:

  mean / max    ablation baselines
  attention     learned per-frame weights -- a forgery is often visible in only
                a handful of frames (a bad blend on one head turn), and mean
                pooling dilutes exactly that evidence by 1/T
  transformer   attention pooling plus self-attention across frames, so
                inconsistency *between* frames becomes usable evidence

The attention weights are returned and surfaced in the UI as the per-frame
timeline, so the pooling choice is also an explainability choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MeanPool(nn.Module):
    kind = "mean"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        return x.mean(dim=1), x.new_full((B, T), 1.0 / T)


class MaxPool(nn.Module):
    kind = "max"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, idx = x.max(dim=1)
        w = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        # Attribution for max pooling: how often each frame won a dimension.
        w.scatter_add_(1, idx, torch.ones_like(idx, dtype=x.dtype))
        return pooled, w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)


class AttentionPool(nn.Module):
    kind = "attention"

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)  # (B,T)
        return (x * w.unsqueeze(-1)).sum(dim=1), w


class PositionalEncoding(nn.Module):
    """Learned positional embedding over frame index.

    Learned rather than sinusoidal: T is small and fixed (32), and the
    informative structure is "early/middle/late in the clip", which a learned
    table captures directly.
    """

    def __init__(self, dim: int, max_len: int = 128) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class TemporalTransformer(nn.Module):
    """Self-attention across frames, then attention pooling."""

    kind = "transformer"

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.pos = PositionalEncoding(dim, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: trains stably without warmup tricks
            activation="gelu",
        )
        # enable_nested_tensor is incompatible with norm_first and only warns.
        self.enc = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.pool = AttentionPool(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(self.pos(x))
        return self.pool(self.norm(h))


def build_temporal(kind: str, dim: int, **kw: object) -> nn.Module:
    kind = kind.lower()
    if kind == "mean":
        return MeanPool()
    if kind == "max":
        return MaxPool()
    if kind == "attention":
        return AttentionPool(dim, int(kw.get("hidden", 128)))  # type: ignore[arg-type]
    if kind == "transformer":
        return TemporalTransformer(
            dim,
            depth=int(kw.get("depth", 4)),  # type: ignore[arg-type]
            heads=int(kw.get("heads", 8)),  # type: ignore[arg-type]
            dropout=float(kw.get("dropout", 0.1)),  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown temporal head {kind!r}; expected mean|max|attention|transformer")


def pick_heads(dim: int, want: int = 8) -> int:
    """Largest head count <= want that divides dim (nn.Transformer requires it)."""
    for h in range(min(want, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1
