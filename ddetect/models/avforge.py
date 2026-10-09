"""The assembled detector: visual + audio + sync -> fusion.

One class covers every experiment in the experiment grid by configuration alone:

  E1/E2 Baseline A/B   visual only, mean pool, no freq/blend/SBI
  E5    V-full         visual + freq + blend + transformer temporal
  E6    AV-proposed    all three streams + co-attention fusion
  E7    Full system    the same, trained on FF++ + DFDC

That matters for honesty as much as tidiness: a baseline implemented in a
separate file drifts from the proposed model, and then the reported gap partly
measures the difference between two codebases instead of the difference between
two methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ddetect.models.audio import AudioConfig, AudioStream
from ddetect.models.fusion import FusionConfig, FusionHead
from ddetect.models.syncnet import SyncConfig, SyncStream
from ddetect.models.visual import VisualConfig, VisualStream


@dataclass
class AVForgeConfig:
    visual: VisualConfig = field(default_factory=VisualConfig)
    audio: AudioConfig | None = None  # None = visual-only
    sync: SyncConfig | None = None
    fusion: FusionConfig = field(default_factory=FusionConfig)

    @property
    def is_multimodal(self) -> bool:
        return self.audio is not None or self.sync is not None


class AVForge(nn.Module):
    def __init__(self, cfg: AVForgeConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or AVForgeConfig()
        self.visual = VisualStream(cfg.visual)
        self.audio = AudioStream(cfg.audio) if cfg.audio else None
        self.sync = SyncStream(cfg.sync) if cfg.sync else None

        self.fusion = (
            FusionHead(
                self.visual.out_dim,
                self.audio.out_dim if self.audio else 1,
                self.sync.out_dim if self.sync else 1,
                cfg.fusion,
            )
            if cfg.is_multimodal
            else None
        )

    # ---- staged training ------------------------------------
    def stage(self, name: str) -> None:
        """Apply one of the three training stages.

        I   heads only, backbones frozen
        II  backbones unfrozen, end-to-end
        III fusion only, streams frozen
        """
        if name == "I":
            self.visual.set_backbone_frozen(True)
            if self.sync:
                self.sync.set_frozen(True)
        elif name == "II":
            self.visual.set_backbone_frozen(False)
            if self.sync:
                self.sync.set_frozen(not self.cfg.sync.freeze if self.cfg.sync else True)
        elif name == "III":
            for m in (self.visual, self.audio, self.sync):
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad = False
        else:
            raise ValueError(f"stage must be I|II|III, got {name!r}")

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- forward --------------------------------------------------------
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        vis = self.visual(batch)

        if self.fusion is None:
            # Visual-only: expose the same keys so downstream code (metrics,
            # Detector, the API) needs no branch.
            return {
                "logit": vis["logit"],
                "frame_logits": vis["frame_logits"],
                "emb": vis["emb"],
                "attn": vis["attn"],
                "stream_logits": {"visual": vis["logit"]},
                **({"pred_mask": vis["pred_mask"]} if "pred_mask" in vis else {}),
            }

        aud = self.audio(batch) if self.audio else None
        syn = self.sync(batch) if self.sync else None
        fus = self.fusion(vis, aud, syn, batch)

        out: dict[str, torch.Tensor] = {
            "logit": fus["logit"],
            "emb": fus["emb"],
            "frame_logits": vis["frame_logits"],
            "attn": vis["attn"],
            "gate_weights": fus["gate_weights"],
            "stream_logits": fus["stream_logits"],
        }
        if syn is not None:
            out["sync_seq"] = syn["sync_seq"]
            out["v_emb"] = syn["v_emb"]
            out["a_emb"] = syn["a_emb"]
        if "pred_mask" in vis:
            out["pred_mask"] = vis["pred_mask"]
        return out
