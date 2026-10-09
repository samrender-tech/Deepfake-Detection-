"""F7 - the visual stream.

    frames -> spatial backbone -> (+ frequency branch) -> per-frame embeddings
           -> temporal head -> video logit
                            -> blending-boundary head (auxiliary)

The first objective is to reproduce the literature's in-dataset accuracy
with a plain CNN on cropped faces, so ``VisualStream`` with ``freq=False``,
``temporal="mean"`` and an Xception backbone IS Baseline A -- the same class,
configured down, rather than a separate throwaway model. That keeps E1 and E5
comparable by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ddetect.models.blend_head import BlendBoundaryHead
from ddetect.models.freq import FrequencyBranch
from ddetect.models.temporal import build_temporal, pick_heads

#: timm names for the backbones in the ablation grid.
BACKBONES = {
    "xception": "legacy_xception",
    "effb4": "tf_efficientnet_b4_ns",
    "effb0": "tf_efficientnet_b0_ns",  # the CPU serving profile
    "effv2s": "tf_efficientnetv2_s",
    "resnet50": "resnet50",
}


@dataclass
class VisualConfig:
    backbone: str = "xception"
    pretrained: bool = True
    image_size: int = 299
    dropout: float = 0.3
    #: F7b frequency branch
    use_freq: bool = False
    freq_dim: int = 128
    #: F7c blending-boundary head
    use_blend_head: bool = False
    #: F7d temporal aggregation
    temporal: str = "mean"
    temporal_depth: int = 4
    temporal_dropout: float = 0.1
    #: Projection applied to backbone features before the temporal head.
    #: Keeps the temporal transformer's parameter count independent of the
    #: backbone (Xception is 2048-d, EffNet-B4 1792-d).
    embed_dim: int = 512
    freeze_backbone: bool = False
    extra: dict = field(default_factory=dict)


class VisualStream(nn.Module):
    def __init__(self, cfg: VisualConfig | None = None) -> None:
        super().__init__()
        import timm

        self.cfg = cfg = cfg or VisualConfig()
        if cfg.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(BACKBONES)}")

        self.backbone = timm.create_model(
            BACKBONES[cfg.backbone],
            pretrained=cfg.pretrained,
            num_classes=0,  # feature extractor
            global_pool="",  # keep the spatial map for the blend head
        )
        self.feat_ch = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.freq = FrequencyBranch(cfg.freq_dim) if cfg.use_freq else None
        self.blend_head = BlendBoundaryHead(self.feat_ch) if cfg.use_blend_head else None

        in_dim = self.feat_ch + (cfg.freq_dim if self.freq else 0)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.temporal = build_temporal(
            cfg.temporal,
            cfg.embed_dim,
            depth=cfg.temporal_depth,
            heads=pick_heads(cfg.embed_dim),
            dropout=cfg.temporal_dropout,
        )
        self.classifier = nn.Linear(cfg.embed_dim, 1)
        #: Per-frame classifier shares the projection but scores frames
        #: independently -- this is what the UI timeline plots, and it must not
        #: be the temporal attention weight (that is a weight, not a score).
        self.frame_classifier = nn.Linear(cfg.embed_dim, 1)
        self.out_dim = cfg.embed_dim

        if cfg.freeze_backbone:
            self.set_backbone_frozen(True)

    # ---- staged training ------------------------------------
    def set_backbone_frozen(self, frozen: bool) -> None:
        """Stage I freezes the backbone and trains heads only; Stage II unfreezes."""
        for p in self.backbone.parameters():
            p.requires_grad = not frozen
        self.backbone.eval() if frozen else self.backbone.train()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        faces = batch["faces"]  # (B,T,3,H,W)
        B, T = faces.shape[:2]
        x = faces.flatten(0, 1)  # (B*T,3,H,W)

        fmap = self.backbone(x)  # (B*T,C,h,w)
        if fmap.ndim == 2:  # some timm models pre-pool
            feat = fmap
            fmap = None
        else:
            feat = self.pool(fmap).flatten(1)  # (B*T,C)

        if self.freq is not None:
            feat = torch.cat([feat, self.freq(x)], dim=1)

        emb = self.proj(feat).view(B, T, -1)  # (B,T,D)
        frame_logits = self.frame_classifier(emb).squeeze(-1)  # (B,T)

        pooled, attn = self.temporal(emb)  # (B,D), (B,T)
        logit = self.classifier(pooled).squeeze(-1)  # (B,)

        out: dict[str, torch.Tensor] = {
            "logit": logit,
            "frame_logits": frame_logits,
            "emb": pooled,
            "attn": attn,
        }
        if self.blend_head is not None and fmap is not None:
            out["pred_mask"] = self.blend_head(fmap).view(B, T, 1, *fmap.shape[-2:])
        return out


def build_visual(cfg: VisualConfig | dict | None = None) -> VisualStream:
    if isinstance(cfg, dict):
        cfg = VisualConfig(**cfg)
    return VisualStream(cfg)
