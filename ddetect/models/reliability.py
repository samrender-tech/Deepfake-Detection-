"""F10 (part) - the reliability gate.

A core problem is that "real uploads are compressed and low quality".
The consequence for a multi-stream model is that streams become unreliable
*independently*: a clip can have a crisp face and a music-bed audio track, or
clean speech and a 40px blurred face. A fusion head that weights streams by a
fixed learned constant cannot express that.

The gate maps the per-sample reliability vector (contract: RELIABILITY_FIELDS =
audio_snr, face_conf, mouth_vis, blur) to one weight per stream. It is a
3-layer MLP, deliberately tiny -- it must not have the capacity to classify,
only to decide who to listen to.

The weights are returned and shown in the UI, so a user can see *why* a verdict
leaned on the visual stream alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ddetect.contracts import RELIABILITY_FIELDS

STREAMS = ("visual", "audio", "sync")


class ReliabilityGate(nn.Module):
    def __init__(self, n_feat: int = len(RELIABILITY_FIELDS), hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, len(STREAMS)),
        )
        # Start near-uniform: a randomly initialised gate that happens to
        # silence a stream at step 0 may never recover, since that stream then
        # gets no gradient.
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, reliab: torch.Tensor, has_audio: torch.Tensor | None = None) -> torch.Tensor:
        """(B,n_feat) -> (B,3) weights in [0,1], one per stream.

        Sigmoid rather than softmax: the streams are not competing for a fixed
        budget. "The face is clear AND the audio is clear" should weight both
        highly, which a softmax forbids.
        """
        w = torch.sigmoid(self.net(reliab))
        if has_audio is not None:
            m = has_audio.to(w.dtype).unsqueeze(-1)
            # Hard-zero the audio and sync columns when there is no audio at
            # all: no amount of learned weighting should resurrect a stream
            # whose input does not exist.
            keep = torch.cat([torch.ones_like(m), m, m], dim=-1)
            w = w * keep
        return w
