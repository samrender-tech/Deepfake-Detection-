"""F7b - high-frequency / frequency-domain branch.

One project objective is robustness to compression. The subtlety is that
compression *erases* the fine artefacts detectors rely
on, so a branch that looks at the frequency domain explicitly is useful only if
it looks at the bands that survive a re-encode.

Two complementary views:

  SRM filters   fixed high-pass residual kernels from steganalysis (Fridrich &
                Kodovsky). They suppress image content and expose local noise
                inconsistency, which a blended face has and a camera-captured
                face does not. Fixed, not learned: with ~1k training videos a
                learned high-pass filter just over-fits the dataset's camera.

  DCT bands     block-DCT energy pooled into low/mid/high bands. GAN upsampling
                leaves periodic spectral peaks; heavy compression flattens the
                high band. Giving the model the band energies directly means it
                does not have to rediscover them through a spatial conv stack.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def srm_kernels() -> torch.Tensor:
    """Three classic SRM high-pass residual kernels, as a (3,1,5,5) tensor."""
    # 1st-order horizontal difference
    k1 = np.zeros((5, 5), dtype=np.float32)
    k1[2, 1], k1[2, 2] = -1.0, 1.0

    # 2nd-order (SQUARE 3x3) laplacian-like residual
    k2 = (
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        / 4.0
    )

    # 3rd-order (SQUARE 5x5) residual
    k3 = (
        np.array(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=np.float32,
        )
        / 12.0
    )

    return torch.from_numpy(np.stack([k1, k2, k3])).unsqueeze(1)


class SRMConv(nn.Module):
    """Fixed SRM high-pass filter bank applied to luminance.

    ``requires_grad=False`` is the point: these are priors, not parameters.
    """

    def __init__(self, clip: float = 3.0) -> None:
        super().__init__()
        self.register_buffer("weight", srm_kernels())
        self.register_buffer("rgb2y", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B,3,H,W) residual maps."""
        y = (x * self.rgb2y).sum(dim=1, keepdim=True)
        r = F.conv2d(y, self.weight, padding=2)
        # Truncate: SRM residuals are heavy-tailed and a few extreme pixels
        # (specular highlights, compression ringing) otherwise dominate the
        # statistics the head pools over.
        return torch.clamp(r, -self.clip, self.clip)


class DCTBandPool(nn.Module):
    """Block-DCT band energies as a compact, resolution-independent feature.

    Returns ``n_bands`` log-energies per image. Computed with a fixed DCT-II
    matrix so it runs on MPS (``torch.fft`` coverage there is patchy and
    silently falls back to CPU mid-batch).
    """

    def __init__(self, block: int = 8, n_bands: int = 8) -> None:
        super().__init__()
        self.block = block
        self.n_bands = n_bands
        self.register_buffer("dct_m", self._dct_matrix(block))
        self.register_buffer("band_idx", self._band_map(block, n_bands))

    @staticmethod
    def _dct_matrix(n: int) -> torch.Tensor:
        k = torch.arange(n, dtype=torch.float32)
        m = torch.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
        m[0] *= 1 / np.sqrt(2)
        return m * np.sqrt(2.0 / n)

    @staticmethod
    def _band_map(block: int, n_bands: int) -> torch.Tensor:
        """Assign each DCT coefficient to a radial band by (u+v)."""
        u = torch.arange(block).view(-1, 1).expand(block, block)
        v = torch.arange(block).view(1, -1).expand(block, block)
        radial = (u + v).float()
        idx = (radial / radial.max() * (n_bands - 1)).round().long()
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, n_bands)."""
        b = self.block
        y = x.mean(dim=1, keepdim=True)  # luminance proxy
        B, _, H, W = y.shape
        H2, W2 = (H // b) * b, (W // b) * b
        y = y[:, :, :H2, :W2]

        # (B, nH, nW, b, b) blocks
        patches = y.squeeze(1).unfold(1, b, b).unfold(2, b, b).contiguous()
        d = self.dct_m
        coef = d @ patches @ d.transpose(0, 1)  # 2-D DCT per block
        energy = coef.pow(2).mean(dim=(1, 2))  # (B, b, b)

        out = []
        for i in range(self.n_bands):
            m = self.band_idx == i
            out.append(energy[:, m].mean(dim=1) if m.any() else energy.new_zeros(B))
        return torch.log1p(torch.stack(out, dim=1))


class FrequencyBranch(nn.Module):
    """SRM residual CNN + DCT band energies -> a compact embedding."""

    def __init__(self, out_dim: int = 128, width: int = 32) -> None:
        super().__init__()
        self.srm = SRMConv()
        self.dct = DCTBandPool()

        self.conv = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 4 + self.dct.n_bands, out_dim), nn.ReLU(inplace=True)
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, out_dim)."""
        return self.head(torch.cat([self.conv(self.srm(x)), self.dct(x)], dim=1))
