"""F23 - the explainability payload.

Everything a user (and A7) is allowed to reason from. Produced for one video at
a time by ``Detector``, and nothing downstream may invent a number that did not
come from here.

Grad-CAM++ is computed against the per-frame classifier rather than the video
logit: a gradient taken through the temporal attention pooling attributes to
whichever frames the attention already favoured, which makes the heat map a
picture of the attention weights rather than of the evidence.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ExplainArtifacts:
    gradcam_png: dict[str, str] = field(default_factory=dict)  # frame idx -> base64 PNG
    spectrum_png: str | None = None
    top_frames: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------
class GradCAMPlusPlus:
    """Grad-CAM++ (Chattopadhay et al., 2018) on a chosen conv layer.

    ++ rather than plain Grad-CAM because a blend seam is a thin structure that
    often appears in several disconnected places (jawline, hairline). Plain
    Grad-CAM's global-average weighting collapses multiple occurrences into
    whichever is strongest; ++ uses per-pixel second-order weights and keeps
    them all.
    """

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module) -> None:
        self.model = model
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self._h = [
            layer.register_forward_hook(self._save_act),
            layer.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, _m: object, _i: object, out: torch.Tensor) -> None:
        self.acts = out.detach()

    def _save_grad(self, _m: object, _gi: object, go: tuple[torch.Tensor, ...]) -> None:
        self.grads = go[0].detach()

    def remove(self) -> None:
        for h in self._h:
            h.remove()

    def __call__(self, score: torch.Tensor) -> np.ndarray | None:
        """Backprop ``score`` and return the (N,h,w) normalised CAM."""
        self.model.zero_grad(set_to_none=True)
        score.sum().backward(retain_graph=True)
        if self.acts is None or self.grads is None:
            return None

        a, g = self.acts, self.grads
        g2, g3 = g.pow(2), g.pow(3)
        denom = 2 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True)
        alpha = g2 / denom.clamp(min=1e-8)
        weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * a).sum(dim=1))  # (N,h,w)

        flat = cam.flatten(1)
        lo = flat.min(dim=1).values.view(-1, 1, 1)
        hi = flat.max(dim=1).values.view(-1, 1, 1)
        return ((cam - lo) / (hi - lo).clamp(min=1e-8)).cpu().numpy()


# --------------------------------------------------------------------------
def overlay_cam(frame_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a CAM over a frame as a JET heat map."""
    import cv2

    h, w = frame_rgb.shape[:2]
    c = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip(frame_rgb * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)


def png_b64(img_rgb: np.ndarray) -> str:
    """RGB array -> base64 PNG, for embedding in the API response."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def spectrum_figure(wav: np.ndarray, sr: int = 16_000) -> str | None:
    """Log-magnitude spectrum plot as base64 PNG.

    Shown because voice cloning and codec round-trips leave visible spectral
    signatures (a hard cutoff, missing high-frequency energy) that a user can
    often judge for themselves -- a rare case where the raw evidence is
    legible without expertise.
    """
    if wav is None or len(wav) < 512:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
        freqs = np.fft.rfftfreq(len(wav), 1 / sr)

        # Transparent background with light ink: the UI is dark, and a white
        # figure dropped into it reads as a broken image rather than a chart.
        fig, ax = plt.subplots(figsize=(5, 1.9), dpi=110)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.semilogy(freqs, spec + 1e-8, lw=0.7, color="#38bdf8")
        ax.set_xlabel("Hz", fontsize=8, color="#a1a1aa")
        ax.set_ylabel("magnitude", fontsize=8, color="#a1a1aa")
        ax.tick_params(labelsize=7, colors="#71717a")
        for spine in ax.spines.values():
            spine.set_color("#3f3f46")
        ax.grid(alpha=0.18, lw=0.4, color="#a1a1aa")
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def pick_top_frames(frame_scores: list[float], k: int = 3) -> list[int]:
    """Indices of the most-suspicious frames, for the CAM overlays."""
    if not frame_scores:
        return []
    return [int(i) for i in np.argsort(np.asarray(frame_scores))[::-1][:k]]


def find_cam_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    """Locate the last conv layer of the visual backbone for Grad-CAM.

    Walks the backbone and returns the final Conv2d. Returns None rather than
    raising: a missing heat map degrades the UI, it must not fail the verdict.
    """
    backbone = getattr(getattr(model, "visual", None), "backbone", None)
    if backbone is None:
        return None
    last = None
    for m in backbone.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last
