"""Domain generalisation objectives (ablation grid).

The cross-dataset gap IS a domain-shift gap, so the domain generalisation
literature applies directly. Three objectives, each switchable and each
individually ablatable -- the point is to measure which (if any) earns its
place, not to stack them and claim the sum.

A caution worth stating in the paper: DomainBed (Gulrajani & Lopez-Paz, 2021)
found that under a fair protocol almost none of these reliably beat plain ERM.
We therefore report them as ablations against an honest ERM baseline rather
than adopting one and claiming the gain. SWAD (already implemented in
train.py) was the method that *did* survive that scrutiny, which is why it is
the default and these are the comparisons.

Domains here are forgery methods and source datasets -- the axes along which
the test distribution actually shifts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# DANN -- Ganin et al., JMLR 2016
# ==========================================================================
class _GradientReversal(torch.autograd.Function):
    """Identity forward, negated gradient backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)  # type: ignore[no-any-return]


class DomainDiscriminator(nn.Module):
    """Predicts which domain an embedding came from.

    Trained to succeed; the feature extractor is trained (through the reversed
    gradient) to make it fail. The equilibrium is a representation that cannot
    tell FF++ from Celeb-DF -- which is the property we want, since a feature
    that identifies the dataset is by definition a generator fingerprint.
    """

    def __init__(self, dim: int, n_domains: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, n_domains),
        )
        self.n_domains = n_domains

    def forward(self, emb: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(emb, lambd))


def dann_lambda(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """Ganin's schedule: ramp 0 -> 1 over training.

    Reversing the gradient at full strength from step 0 destroys the features
    before the classifier has learned anything, and training never recovers.
    """
    p = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))).item() - 1.0)


def dann_loss(
    discriminator: DomainDiscriminator,
    emb: torch.Tensor,
    domain_ids: torch.Tensor,
    lambd: float = 1.0,
) -> torch.Tensor:
    if domain_ids.numel() == 0 or int(domain_ids.max()) >= discriminator.n_domains:
        return emb.sum() * 0.0
    return F.cross_entropy(discriminator(emb, lambd), domain_ids.long())


# ==========================================================================
# GroupDRO -- Sagawa et al., ICLR 2020
# ==========================================================================
@dataclass
class GroupDRO:
    """Minimises the WORST group's loss rather than the average.

    Directly aimed at the failure the project measures: a model can average
    well while being useless on one forgery method, and the average is what a
    single-number accuracy reports. Group weights follow an exponentiated-
    gradient update, so a group that is doing badly is upweighted.
    """

    n_groups: int
    eta: float = 0.01
    weights: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = torch.ones(self.n_groups) / self.n_groups

    def __call__(
        self, per_sample_loss: torch.Tensor, group_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.weights is not None
        device = per_sample_loss.device
        w = self.weights.to(device)

        group_loss = torch.zeros(self.n_groups, device=device)
        counts = torch.zeros(self.n_groups, device=device)
        group_loss.index_add_(0, group_ids, per_sample_loss)
        counts.index_add_(0, group_ids, torch.ones_like(per_sample_loss))
        present = counts > 0
        group_loss = torch.where(present, group_loss / counts.clamp(min=1), group_loss)

        # Exponentiated gradient ascent on the weights, renormalised over the
        # groups actually present in this batch.
        with torch.no_grad():
            w = w * torch.exp(self.eta * group_loss.detach())
            w = torch.where(present, w, torch.zeros_like(w))
            w = w / w.sum().clamp(min=1e-12)
            self.weights = w.detach().cpu()

        loss = (w * group_loss).sum()
        return loss, {
            "worst_group_loss": float(group_loss[present].max()) if present.any() else 0.0,
            "mean_group_loss": float(group_loss[present].mean()) if present.any() else 0.0,
            "n_groups_in_batch": int(present.sum()),
        }


# ==========================================================================
# IRM -- Arjovsky et al., 2019
# ==========================================================================
def irm_penalty(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """IRMv1 penalty: squared gradient of the loss w.r.t. a dummy scale of 1.

    Penalises a classifier whose optimum differs between environments. Needs
    ``create_graph=True``, so it costs a second backward pass -- real, and
    worth stating when reporting the ablation's wall-clock.
    """
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits * scale, targets)
    (g,) = torch.autograd.grad(loss, [scale], create_graph=True)
    return (g**2).sum()


def irm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    env_ids: torch.Tensor,
    penalty_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean ERM loss across environments plus the IRM penalty."""
    envs = torch.unique(env_ids)
    if envs.numel() < 2:
        # One environment: IRM reduces to ERM, and the penalty is meaningless.
        return F.binary_cross_entropy_with_logits(logits, targets), {"irm_penalty": 0.0}

    losses, penalties = [], []
    for e in envs:
        m = env_ids == e
        if m.sum() < 2:
            continue
        losses.append(F.binary_cross_entropy_with_logits(logits[m], targets[m]))
        penalties.append(irm_penalty(logits[m], targets[m]))
    if not losses:
        return F.binary_cross_entropy_with_logits(logits, targets), {"irm_penalty": 0.0}

    erm = torch.stack(losses).mean()
    pen = torch.stack(penalties).mean()
    total = erm + penalty_weight * pen
    # Arjovsky's rescaling: without it a large penalty weight makes the total
    # loss explode and the optimiser stalls.
    if penalty_weight > 1.0:
        total = total / penalty_weight
    return total, {"irm_erm": float(erm.detach()), "irm_penalty": float(pen.detach())}


# ==========================================================================
def domain_ids_from_batch(
    batch: dict, key: str = "forgery_method", vocab: dict[str, int] | None = None
) -> tuple[torch.Tensor, dict[str, int]]:
    """Map a batch's metadata strings to integer domain ids.

    The vocabulary is built lazily and returned so the caller can keep it
    stable across batches -- a domain that changes index between steps makes
    both DANN and GroupDRO meaningless.
    """
    vocab = dict(vocab or {})
    values = batch.get(key) or []
    ids = []
    for v in values:
        if v not in vocab:
            vocab[v] = len(vocab)
        ids.append(vocab[v])
    device = batch["label"].device if "label" in batch else "cpu"
    return torch.tensor(ids, dtype=torch.long, device=device), vocab
