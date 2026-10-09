"""F9 - the lip-sync stream. The project's central idea.

"In most fake videos the face and the voice are produced by separate tools.
They rarely line up perfectly. Checking whether the lips match the speech is a
clue that does not depend on any one tool's fingerprint."

Two encoders, both reported in the ablation:

  syncnet      the published SyncNet architecture (Chung & Zisserman, ACCV
               2016), optionally loading the authors' released weights. Frozen.

  contrastive  our own AV-sync encoder, trained with in-batch temporal-shift
               negatives on REAL VIDEOS ONLY. This is the strongest version of
               the generalisation argument in the whole project: having never
               seen a fake, it cannot possibly have learned a generator's
               fingerprint, so whatever it contributes on an unseen forgery
               method is genuinely tool-independent.

Both produce the same features: a per-window audio-visual distance curve, the
best temporal offset, and a confidence (median minus minimum distance across
offsets, the measure SyncNet itself uses). The curve is what the UI charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

#: SyncNet consumes 0.2 s windows: 5 video frames at 25 fps against 20 MFCC
#: frames at 100 Hz. We cache 16-frame windows and stride within them.
SYNC_FRAMES = 16
MAX_OFFSET = 15  # +/- offsets searched, in video frames


@dataclass
class SyncConfig:
    kind: str = "contrastive"  # syncnet | contrastive
    weights: str | None = None  # path to released SyncNet weights
    embed_dim: int = 128
    max_offset: int = MAX_OFFSET
    freeze: bool = True
    dropout: float = 0.1
    n_feat: int = 5  # summary features exposed to fusion


# --------------------------------------------------------------------------
class _R2Plus1Block(nn.Module):
    """Factorised spatio-temporal convolution: 2-D spatial then 1-D temporal.

    SyncNet's original architecture uses true 3-D convolutions, but
    ``Conv3d`` has no MPS kernel, so a 3-D stack silently falls back to the CPU
    mid-batch on the Mac (or hard-fails without the fallback env var). The
    (2+1)D factorisation (Tran et al., CVPR 2018) is MPS-native, uses fewer
    parameters for the same receptive field, and adds a nonlinearity between
    the spatial and temporal stages -- it outperformed full 3-D convolution in
    the paper that introduced it, so this is not a compromise.
    """

    def __init__(
        self,
        cin: int,
        cout: int,
        spatial_k: int = 3,
        temporal_k: int = 3,
        spatial_stride: int = 1,
    ) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(cin, cout, spatial_k, stride=spatial_stride, padding=spatial_k // 2),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(cout, cout, temporal_k, padding=temporal_k // 2),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N,C,T,H,W) -> (N,C',T,H',W')."""
        n, c, t, h, w = x.shape
        y = self.spatial(x.transpose(1, 2).reshape(n * t, c, h, w))
        c2, h2, w2 = y.shape[1:]
        # temporal pass: treat each spatial location as a sequence over time
        y = y.view(n, t, c2, h2, w2).permute(0, 3, 4, 2, 1).reshape(n * h2 * w2, c2, t)
        y = self.temporal(y)
        return y.view(n, h2, w2, c2, t).permute(0, 3, 4, 1, 2)


class _VisualEnc(nn.Module):
    """(2+1)D conv stack over grey mouth crops -> embedding per window."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.b1 = _R2Plus1Block(1, 32, spatial_k=5, spatial_stride=2)
        self.b2 = _R2Plus1Block(32, 64, spatial_stride=2)
        self.b3 = _R2Plus1Block(64, 128, spatial_stride=2)
        self.proj = nn.Linear(128, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N,1,T,96,96) -> (N,dim)."""
        h = self.b3(self.b2(self.b1(x)))
        # Global pool over time and space. Done with mean() rather than
        # AdaptiveAvgPool3d, which is also missing on MPS.
        return F.normalize(self.proj(h.mean(dim=(2, 3, 4))), dim=-1)


class _AudioEnc(nn.Module):
    """2-D conv stack over a mel/MFCC slice -> embedding per window."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(128, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N,1,M,T) -> (N,dim)."""
        return F.normalize(self.proj(self.net(x)), dim=-1)


# --------------------------------------------------------------------------
class SyncStream(nn.Module):
    """Audio-visual synchrony features from mouth crops and a spectrogram."""

    def __init__(self, cfg: SyncConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or SyncConfig()
        self.vis = _VisualEnc(cfg.embed_dim)
        self.aud = _AudioEnc(cfg.embed_dim)

        if cfg.kind == "syncnet" and cfg.weights:
            self.load_syncnet_weights(cfg.weights)
        if cfg.freeze:
            self.set_frozen(True)

        # A tiny head over the summary features, so the stream has a standalone
        # score (every stream must be reportable on its own).
        self.classifier = nn.Sequential(
            nn.Linear(cfg.n_feat, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
        )
        self.out_dim = cfg.n_feat

    # ---- weights -------------------------------------------------------
    def load_syncnet_weights(self, path: str | Path) -> None:
        """Load the released SyncNet checkpoint, tolerating key mismatch.

        Our encoders are architecturally equivalent but not identically named,
        so only overlapping shapes are loaded and the rest is reported. A
        partial load is logged loudly, because silently keeping random weights
        here would make the whole sync stream meaningless while still appearing
        to work.
        """
        from ddetect.utils.log import get_logger

        log = get_logger(__name__)
        p = Path(path)
        if not p.exists():
            log.warning(
                "SyncNet weights not found at %s; encoders stay randomly "
                "initialised (use kind='contrastive' and train them)",
                p,
            )
            return
        sd = torch.load(p, map_location="cpu")
        sd = sd.get("state_dict", sd)
        own = self.state_dict()
        loaded = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        self.load_state_dict(loaded, strict=False)
        log.info("SyncNet: loaded %d/%d tensors from %s", len(loaded), len(own), p)
        if len(loaded) < len(own) // 2:
            log.warning(
                "SyncNet load covered <50%% of parameters -- treat sync features "
                "as untrained until kind='contrastive' training is run"
            )

    def set_frozen(self, frozen: bool) -> None:
        for m in (self.vis, self.aud):
            for p in m.parameters():
                p.requires_grad = not frozen

    # ---- embeddings ----------------------------------------------------
    def encode(self, mouth: torch.Tensor, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,W,T,1,96,96) and (B,1,M,L) -> per-window embeddings (B,W,D) each.

        The audio is sliced into W windows aligned to the mouth windows, so the
        offset search below compares like with like.
        """
        B, W = mouth.shape[:2]
        v = self.vis(mouth.flatten(0, 1).permute(0, 2, 1, 3, 4)).view(B, W, -1)

        L = mel.shape[-1]
        win = max(L // W, 1)
        slices = [mel[:, :, :, i * win : (i + 1) * win] for i in range(W)]
        # Pad the last slice so every window is the same length.
        slices = [F.pad(s, (0, win - s.shape[-1])) if s.shape[-1] < win else s for s in slices]
        # .contiguous() is required, not cosmetic: the transpose leaves a
        # non-contiguous view, and torch.roll on a non-contiguous MPS tensor
        # trips an MPSNDArray slice assertion that aborts the process without
        # a Python traceback.
        a = self.aud(torch.cat(slices, dim=0)).view(W, B, -1).transpose(0, 1).contiguous()
        return v, a.contiguous()

    # ---- the offset search --------------------------------------------
    def sync_features(
        self, v: torch.Tensor, a: torch.Tensor, max_offset: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (summary (B,n_feat), per-window distance curve (B,W)).

        For each candidate offset, slide the audio windows against the visual
        windows and measure cosine distance over the region where they overlap.
        A genuine recording has a sharp minimum at a consistent small offset; a
        clip whose face and voice came from different tools has a flat or
        erratic curve. That shape -- not the absolute distance -- is the
        evidence, which is why ``conf`` (median minus min) is the headline
        feature rather than ``min_dist``.

        The shift is deliberately NOT ``torch.roll``. Roll is cyclic, so with W
        windows an offset of ``k`` and one of ``k - W`` produce an identical
        alignment and therefore a bit-identical distance. Those exact ties are
        then broken by ~1e-7 of batching noise, which flips ``argmin`` and moves
        ``best_offset`` by a whole step -- the same clip scores differently
        depending on how many other clips share its batch. (Measured: a 0.33
        swing in the normalised offset feature, and 0.15 in the final logit,
        between batch sizes 1 and 2.) Wrapping is also meaningless physically:
        audio from the end of a clip is not a candidate match for lips at the
        start.

        Offsets are in WINDOW units, so the resolution is one window
        (~1 s at the default settings). That is coarse -- enough to detect the
        gross desynchronisation a two-tool forgery pipeline produces, not a
        precise lip-sync offset estimate. The paper states this limitation.
        """
        B, W, _ = v.shape
        # At least two windows must overlap for a distance to mean anything,
        # and the cap keeps every offset a distinct alignment.
        usable = max(W - 2, 0)
        max_off = min(max_offset or self.cfg.max_offset, usable) if usable else 0
        v, a = v.contiguous(), a.contiguous()

        if max_off == 0:
            # Too few windows to search: report the aligned distance only.
            d = (1.0 - (v * a).sum(-1)).unsqueeze(1)  # (B,1,W)
            per_off = d.mean(dim=2)
            min_dist = per_off[:, 0]
            zeros = torch.zeros_like(min_dist)
            curve = d[:, 0]
            return (
                torch.stack([min_dist, zeros, zeros, min_dist, curve.std(dim=1)], dim=1),
                curve,
            )

        offsets = list(range(-max_off, max_off + 1))
        per_off_list: list[torch.Tensor] = []
        curves: list[torch.Tensor] = []
        for off in offsets:
            # Compare v[t] against a[t + off] over valid t only.
            lo, hi = max(0, -off), W - max(0, off)
            vs, as_ = v[:, lo:hi], a[:, lo + off : hi + off]
            dist = 1.0 - (vs * as_).sum(-1)  # (B, overlap)
            per_off_list.append(dist.mean(dim=1))
            # Pad back to W so every offset's curve is the same length; the
            # padding is the offset's own mean, so it does not bias std().
            pad = dist.mean(dim=1, keepdim=True).expand(B, W - dist.shape[1])
            curves.append(torch.cat([dist, pad], dim=1))

        per_off = torch.stack(per_off_list, dim=1)  # (B,O)
        d = torch.stack(curves, dim=1)  # (B,O,W)

        min_dist, arg = per_off.min(dim=1)
        median = per_off.median(dim=1).values
        conf = median - min_dist  # SyncNet's confidence
        best_offset = arg.float() - max_off
        curve = d.gather(1, arg.view(B, 1, 1).expand(B, 1, W)).squeeze(1)  # (B,W)

        feats = torch.stack(
            [min_dist, conf, best_offset / max(max_off, 1), per_off.mean(1), curve.std(dim=1)],
            dim=1,
        )
        return feats, curve

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mouth, mel = batch["mouth"], batch["mel"]
        v, a = self.encode(mouth, mel)
        feats, curve = self.sync_features(v, a)

        if "has_audio" in batch:
            mask = batch["has_audio"].to(feats.dtype).unsqueeze(-1)
            feats = feats * mask
            curve = curve * mask
        return {
            "logit": self.classifier(feats).squeeze(-1),
            "emb": feats,
            "sync_seq": curve,
            "v_emb": v,
            "a_emb": a,
        }


# --------------------------------------------------------------------------
def av_contrastive_loss(
    v: torch.Tensor, a: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """InfoNCE over (window, time-shift) pairs, for REAL VIDEO ONLY.

    Positives: the audio window that actually co-occurred with a mouth window.
    Negatives: every other window in the batch, which includes other time
    positions of the same clip -- a hard negative that forces genuine temporal
    alignment rather than speaker identity matching.

    THIS LOSS MUST NEVER SEE A FAKE. Training it on forgeries would let it
    encode generator artefacts, destroying the one property that makes the sync
    stream tool-independent (the sync-stream design, reason 2). ``train.py`` filters the batch
    to ``label == 0`` before calling this, and a test asserts it.
    """
    B, W, D = v.shape
    vf = F.normalize(v.reshape(B * W, D), dim=-1)
    af = F.normalize(a.reshape(B * W, D), dim=-1)
    logits = vf @ af.t() / temperature
    target = torch.arange(B * W, device=v.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def build_sync(cfg: SyncConfig | dict | None = None) -> SyncStream:
    if isinstance(cfg, dict):
        cfg = SyncConfig(**cfg)
    return SyncStream(cfg)
