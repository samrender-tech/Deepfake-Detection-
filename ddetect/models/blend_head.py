"""F7c - blending-boundary head (Face X-ray style, Li et al. CVPR 2020).

Predicts *where* the blend seam is, supervised by the masks SBI (F4) produces
for free. Two reasons this earns its place:

1. Localisation is a much stronger training signal than a single global label.
   "This 28x28 region is a seam" constrains the features far more than "this
   video is fake", which matters at our data scale.
2. It is the honest form of explainability. A Grad-CAM over a binary classifier
   shows where the gradient is large; this head shows where the model believes
   the *boundary* is, which is a claim that can be checked against the mask.

Trained only on the SBI subset: real frames carry a NaN target (contract 5.3)
and are masked out of the loss rather than pushed toward an all-zero map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlendBoundaryHead(nn.Module):
    """Feature map -> single-channel boundary logit map at 1/8 resolution."""

    def __init__(self, in_ch: int, mid: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid // 2, 3, padding=1),
            nn.BatchNorm2d(mid // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid // 2, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """(B,C,h,w) -> (B,1,h,w) logits."""
        return self.net(feat)


def boundary_loss(
    pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 4.0
) -> torch.Tensor:
    """Masked BCE over frames that actually have an SBI target.

    ``target`` is (B,T,1,h,w) with NaN where no mask exists. Returns 0 (not
    NaN) when a batch happens to contain no SBI samples, so the training step
    stays finite.
    """
    B, T = target.shape[:2]
    tgt = target.reshape(B * T, *target.shape[2:])
    valid = torch.isfinite(tgt).flatten(1).all(dim=1)
    if not valid.any():
        return pred.sum() * 0.0

    p = pred[valid]
    t = tgt[valid]
    if p.shape[-2:] != t.shape[-2:]:
        t = F.interpolate(t, size=p.shape[-2:], mode="bilinear", align_corners=False)

    # The seam is a thin structure -- a few percent of pixels (measured: ~5%) --
    # so an unweighted BCE is minimised by predicting "no boundary" everywhere.
    return F.binary_cross_entropy_with_logits(
        p, t.clamp(0, 1), pos_weight=torch.tensor(pos_weight, device=p.device)
    )
