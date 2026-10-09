"""F13 - out-of-distribution / unseen-generator flagging.

The honest response to a forgery method the model has never seen is not a
confident verdict: it is "this does not look like anything I was trained on".
The project's whole premise is that new generation methods appear every few months,
so this flag is the system's answer to its own central limitation.

Two scores, both fitted on train/val features only:

  energy       -logsumexp over class logits (Liu et al., NeurIPS 2020). For a
               binary head, logsumexp(0, z). Cheap, needs no stored statistics.
  mahalanobis  distance to the training feature mean under a shared covariance.
               Catches feature-space novelty the logit cannot express -- a
               video can produce a confident logit from features nothing like
               the training set.

Validated by treating each held-out dataset as the novel set and reporting the
AUROC of the novelty detector itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ddetect.utils.log import get_logger

log = get_logger(__name__)


def energy_score(logits: np.ndarray | torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """Free-energy OOD score; HIGHER = more novel.

    Liu et al. define the energy as ``-T * logsumexp(z_k / T)`` over class
    logits. A single-logit binary head has no explicit second logit, and the
    obvious completion ``(0, z)`` is asymmetric: it reports a confidently
    *authentic* video (z very negative) as maximally novel, because
    ``logsumexp(0, z) -> 0`` there. That is backwards, and it is why a first
    version of this scored ~0.5 AUROC.

    The symmetric completion ``(+z/2, -z/2)`` represents the same binary
    posterior and gives ``-T * log(2 cosh(z / 2T))``: maximal (= most novel)
    at z = 0 where the model is ambivalent, and falling as |z| grows in either
    direction. That is the intended semantics.
    """
    z = np.asarray(logits, dtype=float) / max(temperature, 1e-6)
    # log(2 cosh(z/2)) computed stably as logaddexp(z/2, -z/2).
    lse = np.logaddexp(z / 2.0, -z / 2.0)
    return np.asarray(-temperature * lse, dtype=float)


@dataclass
class MahalanobisOOD:
    """Shared-covariance Mahalanobis novelty detector."""

    mean: list[float] | None = None
    precision: list[list[float]] | None = None
    dim: int = 0
    n_fit: int = 0

    def fit(self, feats: np.ndarray, shrinkage: float = 0.1) -> MahalanobisOOD:
        """Fit on TRAIN features only.

        Ledoit-Wolf style shrinkage toward a scaled identity: the fusion
        embedding is 256-d and a few thousand training videos give a covariance
        that is near-singular, so the raw inverse is numerically meaningless.
        """
        x = np.asarray(feats, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected (N,D) features, got {x.shape}")
        mu = x.mean(axis=0)
        xc = x - mu
        cov = xc.T @ xc / max(len(x) - 1, 1)
        cov = (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[
            0
        ]
        self.mean = mu.tolist()
        self.precision = np.linalg.pinv(cov).tolist()
        self.dim = x.shape[1]
        self.n_fit = len(x)
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance; HIGHER = more novel."""
        if self.mean is None or self.precision is None:
            return np.zeros(len(feats))
        x = np.asarray(feats, dtype=np.float64) - np.asarray(self.mean)
        p = np.asarray(self.precision)
        return np.asarray(np.einsum("ij,jk,ik->i", x, p, x), dtype=float)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)))
        return path

    @classmethod
    def load(cls, path: str | Path) -> MahalanobisOOD | None:
        p = Path(path)
        if not p.exists():
            return None
        return cls(**json.loads(p.read_text()))


def novelty_auroc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """How well the OOD score separates in-distribution from novel data.

    This is the validation of F13 itself: treat the training dataset as
    in-distribution and a held-out forgery dataset as novel, and report whether
    the flag actually fires more often on the novel one. Without this number
    the flag is decoration.
    """
    from ddetect.metrics import auc

    y = np.r_[np.zeros(len(in_scores)), np.ones(len(out_scores))]
    s = np.r_[np.asarray(in_scores, float), np.asarray(out_scores, float)]
    return auc(y, s)
