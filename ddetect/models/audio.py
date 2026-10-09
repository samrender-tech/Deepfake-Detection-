"""F8 - the audio stream. Two tiers, both reported as an ablation.

  logmel   80-bin log-mel -> 6-block CNN. A small design, ~1M params,
           trains on the Mac. This is the tier that always works.

  wavlm    frozen WavLM-base features -> AASIST-style graph-attention backend
           (Jung et al., ICASSP 2022). The audio anti-spoofing standard. A
           voice clone's giveaways are prosodic and phase-related, and an SSL
           encoder trained on thousands of hours of speech represents those far
           better than a mel-CNN trained on a few hundred DFDC clips.

The AASIST backend here is a faithful-in-spirit, reduced implementation: a
graph attention layer over spectral and temporal node sets with a learned
master node. It is not a line-by-line port of the authors' code, and the paper
says so -- what matters for our claim is the ablation logmel-vs-SSL, not
reproducing AASIST's own leaderboard number.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioConfig:
    kind: str = "logmel"  # logmel | wavlm
    mel_bins: int = 80
    width: int = 32
    out_dim: int = 128
    dropout: float = 0.3
    # wavlm tier
    wavlm_name: str = "microsoft/wavlm-base"
    freeze_layers: int = 12  # 12 = fully frozen; lower to fine-tune the top
    sample_rate: int = 16_000


# --------------------------------------------------------------------------
# tier 1: log-mel CNN
# --------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.pool = nn.MaxPool2d(stride) if stride > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.conv(x) + self.skip(x), inplace=True))


class LogMelCNN(nn.Module):
    """Residual CNN over the log-mel spectrogram."""

    def __init__(self, cfg: AudioConfig) -> None:
        super().__init__()
        w = cfg.width
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 5, stride=1, padding=2), nn.BatchNorm2d(w), nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            _Block(w, w),
            _Block(w, w * 2),
            _Block(w * 2, w * 4),
            _Block(w * 4, w * 4),
            _Block(w * 4, w * 8, stride=1),
        )
        # Pool frequency away but keep time, then attention-pool over time:
        # a splice or clone artefact is usually localised to part of the
        # utterance, and mean-pooling over time dilutes it.
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.attn = nn.Sequential(nn.Conv1d(w * 8, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(w * 8, cfg.out_dim),
            nn.LayerNorm(cfg.out_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.out_dim = cfg.out_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(mel))  # (B,C,F,T)
        h = self.freq_pool(h).squeeze(2)  # (B,C,T)
        w = torch.softmax(self.attn(h), dim=-1)  # (B,1,T)
        return self.proj((h * w).sum(-1))


# --------------------------------------------------------------------------
# tier 2: WavLM + AASIST-style graph attention
# --------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """Single-head graph attention over a set of nodes (AASIST's GAT block)."""

    def __init__(self, dim: int, out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, out)
        self.att = nn.Linear(2 * out, 1)
        self.norm = nn.LayerNorm(out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,N,D) -> (B,N,out)."""
        h = self.proj(x)
        B, N, dim = h.shape
        a = torch.cat(
            [h.unsqueeze(2).expand(B, N, N, dim), h.unsqueeze(1).expand(B, N, N, dim)],
            dim=-1,
        )
        e = F.leaky_relu(self.att(a).squeeze(-1), 0.2)  # (B,N,N)
        w = self.drop(torch.softmax(e, dim=-1))
        return self.norm(F.gelu(w @ h) + h)


class AASISTBackend(nn.Module):
    """Spectral + temporal node graphs joined by a learned master node."""

    def __init__(
        self, in_dim: int, out_dim: int = 128, nodes: int = 24, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.to_spec = nn.Linear(in_dim, out_dim)
        self.to_temp = nn.Linear(in_dim, out_dim)
        self.gat_s = GraphAttentionLayer(out_dim, out_dim, dropout)
        self.gat_t = GraphAttentionLayer(out_dim, out_dim, dropout)
        self.master = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.gat_m = GraphAttentionLayer(out_dim, out_dim, dropout)
        self.nodes = nodes
        self.head = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.out_dim = out_dim
        nn.init.trunc_normal_(self.master, std=0.02)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(B,T,D) frame features -> (B,out_dim)."""
        B, T, _ = feats.shape
        n = min(self.nodes, T)
        # Temporal nodes: segment-pooled over time.
        seg = torch.linspace(0, T, n + 1).long()
        temp = torch.stack(
            [feats[:, seg[i] : max(seg[i + 1], seg[i] + 1)].mean(1) for i in range(n)], dim=1
        )
        # Spectral nodes: pooled over time, split across the feature axis.
        pooled = feats.mean(dim=1)  # (B,D)
        spec = self.to_spec(pooled).unsqueeze(1).expand(B, n, -1)

        hs = self.gat_s(spec)
        ht = self.gat_t(self.to_temp(temp))
        joined = torch.cat([hs, ht, self.master.expand(B, 1, -1)], dim=1)
        out = self.gat_m(joined)
        master = out[:, -1]
        rest = out[:, :-1].mean(dim=1)
        return self.head(torch.cat([master, rest], dim=-1))


class WavLMAASIST(nn.Module):
    """Frozen WavLM encoder + AASIST backend."""

    def __init__(self, cfg: AudioConfig) -> None:
        super().__init__()
        from transformers import WavLMModel

        self.encoder = WavLMModel.from_pretrained(cfg.wavlm_name)
        self.encoder.config.apply_spec_augment = False
        total = len(self.encoder.encoder.layers)
        n_freeze = min(cfg.freeze_layers, total)

        self.encoder.feature_extractor._freeze_parameters()
        for i, layer in enumerate(self.encoder.encoder.layers):
            for p in layer.parameters():
                p.requires_grad = i >= n_freeze
        self.backend = AASISTBackend(
            self.encoder.config.hidden_size, cfg.out_dim, dropout=cfg.dropout
        )
        self.out_dim = cfg.out_dim

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(wav).last_hidden_state  # (B,T,H)
        return self.backend(feats)


# --------------------------------------------------------------------------
class AudioStream(nn.Module):
    """Dispatches to a tier and adds the stream's own logit head.

    Each stream carries its own classifier so it can be trained and reported
    standalone: the streams are trained separately first, so the project
    still works even if fusion underperforms.
    """

    def __init__(self, cfg: AudioConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or AudioConfig()
        if cfg.kind == "logmel":
            self.net: nn.Module = LogMelCNN(cfg)
            self._input = "mel"
        elif cfg.kind == "wavlm":
            self.net = WavLMAASIST(cfg)
            self._input = "wav"
        else:
            raise ValueError(f"audio kind must be logmel|wavlm, got {cfg.kind!r}")
        self.classifier = nn.Linear(self.net.out_dim, 1)
        self.out_dim = self.net.out_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch[self._input]
        emb = self.net(x)
        # Zero the embedding where there is no audio, so a silent clip cannot
        # contribute a spurious constant to the fusion head. The mask is also
        # passed through so fusion can drop the stream entirely.
        if "has_audio" in batch:
            emb = emb * batch["has_audio"].to(emb.dtype).unsqueeze(-1)
        return {"logit": self.classifier(emb).squeeze(-1), "emb": emb}


def build_audio(cfg: AudioConfig | dict | None = None) -> AudioStream:
    if isinstance(cfg, dict):
        cfg = AudioConfig(**cfg)
    return AudioStream(cfg)
