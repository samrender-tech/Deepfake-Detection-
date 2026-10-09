"""F10 - fusion. Objective 3's payoff, and the part that must survive a
missing modality.

Three fusion modes, all reported:

  concat     MLP on concatenated embeddings. The obvious baseline.
  score      logistic regression on the three stream logits. The cheap floor;
             the insurance policy -- "the project still works even if fusion
             underperforms" means this must be measured, not assumed.
  coattn     cross-modal co-attention transformer. Each stream attends to the
             others, so "the lips disagree with the speech" can be represented
             as an interaction rather than a sum of two independent opinions.

MODALITY DROPOUT is what makes one checkpoint serve all four test sets. During
training, audio and sync are randomly dropped (p=0.3) even when present, so the
model is forced to produce a calibrated visual-only verdict. Without it, the
model learns to depend on audio and collapses on silent FF++/Celeb-DF -- which
the ablation demonstrates deliberately (the modality-dropout ablation: "the -modality-dropout
ablation must fail on silent video, proving the mask works").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ddetect.models.reliability import STREAMS, ReliabilityGate


@dataclass
class FusionConfig:
    mode: str = "coattn"  # coattn | concat | score
    dim: int = 256
    depth: int = 2
    heads: int = 4
    dropout: float = 0.2
    #: Modality dropout probability, applied to audio and sync independently.
    modality_dropout: float = 0.3
    use_gate: bool = True


class CoAttentionBlock(nn.Module):
    """One stream attends to the concatenation of the others."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(B,N,D) -> (B,N,D). ``key_padding_mask`` True = ignore that token."""
        q = self.norm_q(tokens)
        kv = self.norm_kv(tokens)
        a, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        h = tokens + a
        return h + self.ff(self.norm_ff(h))


class FusionHead(nn.Module):
    """Combines the three streams into one calibrated-ready logit."""

    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        sync_dim: int,
        cfg: FusionConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg = cfg or FusionConfig()
        d = cfg.dim

        self.proj = nn.ModuleDict(
            {
                "visual": nn.Sequential(nn.Linear(visual_dim, d), nn.LayerNorm(d)),
                "audio": nn.Sequential(nn.Linear(audio_dim, d), nn.LayerNorm(d)),
                "sync": nn.Sequential(nn.Linear(sync_dim, d), nn.LayerNorm(d)),
            }
        )
        # Learned per-stream type embedding: without it the attention cannot
        # tell which token is which, and "audio disagrees with visual" is not
        # representable.
        self.type_emb = nn.Parameter(torch.zeros(1, len(STREAMS), d))
        nn.init.trunc_normal_(self.type_emb, std=0.02)

        self.gate = ReliabilityGate() if cfg.use_gate else None

        if cfg.mode == "coattn":
            self.blocks = nn.ModuleList(
                [CoAttentionBlock(d, cfg.heads, cfg.dropout) for _ in range(cfg.depth)]
            )
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.cls, std=0.02)
            self.norm = nn.LayerNorm(d)
            self.classifier = nn.Linear(d, 1)
        elif cfg.mode == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(d * len(STREAMS), d),
                nn.LayerNorm(d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d, 1),
            )
        elif cfg.mode == "score":
            # Exactly a logistic regression on the three stream logits, plus
            # the gate weights so it can still discount an absent stream.
            self.classifier = nn.Linear(len(STREAMS) * 2, 1)
        else:
            raise ValueError(f"fusion mode must be coattn|concat|score, got {cfg.mode!r}")
        self.out_dim = d

    # ------------------------------------------------------------------
    def _modality_mask(self, has_audio: torch.Tensor) -> torch.Tensor:
        """(B,) -> (B,3) keep-mask for [visual, audio, sync].

        Audio and sync are dropped together with probability p and
        independently of each other with probability p, which covers both
        "no audio at all" (FF++) and "audio present but useless" (music bed).
        """
        B = has_audio.shape[0]
        keep = torch.ones(B, len(STREAMS), device=has_audio.device)
        a = has_audio.to(keep.dtype)
        keep[:, 1] = a
        keep[:, 2] = a

        if self.training and self.cfg.modality_dropout > 0:
            p = self.cfg.modality_dropout
            drop_both = (torch.rand(B, 1, device=keep.device) < p).to(keep.dtype)
            drop_a = (torch.rand(B, device=keep.device) < p).to(keep.dtype)
            drop_s = (torch.rand(B, device=keep.device) < p).to(keep.dtype)
            keep[:, 1] = keep[:, 1] * (1 - drop_a) * (1 - drop_both[:, 0])
            keep[:, 2] = keep[:, 2] * (1 - drop_s) * (1 - drop_both[:, 0])
        return keep

    def forward(
        self,
        visual: dict[str, torch.Tensor],
        audio: dict[str, torch.Tensor] | None,
        sync: dict[str, torch.Tensor] | None,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        v_emb = visual["emb"]
        B = v_emb.shape[0]
        dev, dt = v_emb.device, v_emb.dtype

        a_emb = (
            audio["emb"]
            if audio
            else torch.zeros(B, self.proj["audio"][0].in_features, device=dev, dtype=dt)
        )
        s_emb = (
            sync["emb"]
            if sync
            else torch.zeros(B, self.proj["sync"][0].in_features, device=dev, dtype=dt)
        )

        has_audio = batch.get("has_audio", torch.ones(B, dtype=torch.bool, device=dev))
        keep = self._modality_mask(has_audio)

        gate = (
            self.gate(batch["reliab"], has_audio)
            if (self.gate is not None and "reliab" in batch)
            else torch.ones(B, len(STREAMS), device=dev, dtype=dt)
        )
        weight = keep * gate  # (B,3)

        stream_logits = {
            "visual": visual["logit"],
            "audio": audio["logit"] if audio else torch.zeros(B, device=dev, dtype=dt),
            "sync": sync["logit"] if sync else torch.zeros(B, device=dev, dtype=dt),
        }

        if self.cfg.mode == "score":
            feats = torch.stack(
                [stream_logits[s] * weight[:, i] for i, s in enumerate(STREAMS)], dim=1
            )
            logit = self.classifier(torch.cat([feats, weight], dim=1)).squeeze(-1)
            return {
                "logit": logit,
                "emb": feats,
                "gate_weights": weight,
                "stream_logits": stream_logits,
            }

        tokens = (
            torch.stack(
                [
                    self.proj["visual"](v_emb),
                    self.proj["audio"](a_emb),
                    self.proj["sync"](s_emb),
                ],
                dim=1,
            )
            + self.type_emb
        )  # (B,3,D)
        tokens = tokens * weight.unsqueeze(-1)

        if self.cfg.mode == "concat":
            logit = self.classifier(tokens.flatten(1)).squeeze(-1)
            return {
                "logit": logit,
                "emb": tokens.flatten(1),
                "gate_weights": weight,
                "stream_logits": stream_logits,
            }

        # co-attention: a CLS token plus the three stream tokens. Dropped
        # streams are masked out of attention entirely rather than merely
        # zeroed, so they contribute no key/value at all.
        cls = self.cls.expand(B, 1, -1)
        seq = torch.cat([cls, tokens], dim=1)  # (B,4,D)
        pad = torch.cat([torch.zeros(B, 1, device=dev, dtype=torch.bool), keep <= 0], dim=1)
        for blk in self.blocks:
            seq = blk(seq, key_padding_mask=pad)
        pooled = self.norm(seq[:, 0])
        return {
            "logit": self.classifier(pooled).squeeze(-1),
            "emb": pooled,
            "gate_weights": weight,
            "stream_logits": stream_logits,
        }
