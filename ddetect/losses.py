"""Loss functions for the composite objective.

    focal + label_smoothing + lambda_m * mask_BCE + lambda_c * supcon
                            + lambda_s * av_contrastive

Each term is weighted by config and each is individually ablatable, because a
composite loss whose terms cannot be switched off is untestable: there would be
no way to show which part earned the gain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = -1.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss with optional label smoothing.

    Focal because the hard cases are the point: once a detector gets the
    obvious FF++ Deepfakes right, plain BCE is dominated by examples it has
    already solved, and the gradient from the near-miss cases that actually
    determine cross-dataset performance is swamped.

    Label smoothing because a confidently wrong detector is the failure mode
    section 13 is written to avoid -- it keeps the logits from running away and
    makes the downstream temperature scaling better behaved.
    """
    t = targets.float()
    if label_smoothing > 0:
        t = t * (1 - label_smoothing) + 0.5 * label_smoothing

    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p_t = p * t + (1 - p) * (1 - t)
    loss = ce * (1 - p_t).clamp(min=1e-6).pow(gamma)

    if alpha >= 0:
        a_t = alpha * t + (1 - alpha) * (1 - t)
        loss = a_t * loss

    if reduction == "mean":
        return torch.as_tensor(loss.mean())
    if reduction == "sum":
        return torch.as_tensor(loss.sum())
    return torch.as_tensor(loss)


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020) on the pooled embedding.

    Pulls fakes toward fakes and reals toward reals regardless of which
    generator made them. That is directly aimed at the research gap: a
    classifier head can separate FF++ Deepfakes from real without ever forming
    a general notion of "manipulated", and this term penalises exactly that by
    requiring the four FF++ methods to share a region of feature space.

    Returns 0 when a batch is single-class (no positives exist).
    """
    z = F.normalize(emb, dim=-1)
    y = labels.view(-1)
    n = z.shape[0]
    if n < 2 or y.unique().numel() < 2:
        return emb.sum() * 0.0

    sim = z @ z.t() / temperature
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))

    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    n_pos = pos.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return emb.sum() * 0.0

    # torch.where, not multiplication: log_prob holds -inf on the diagonal
    # (masked out of the similarity above) and -inf * 0 is NaN, which would
    # silently poison the whole loss.
    masked = torch.where(pos, log_prob, torch.zeros_like(log_prob))
    mean_log_prob = masked.sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob.mean()


class CompositeLoss:
    """Assembles the weighted objective and reports each term separately.

    Returning the breakdown is not cosmetic: a composite loss that silently
    becomes dominated by one auxiliary term looks like healthy training in the
    aggregate while the main task stops improving. The per-term values go into
    ``metrics.jsonl`` and A3 watches them.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = -1.0,
        label_smoothing: float = 0.05,
        lambda_mask: float = 0.0,
        lambda_supcon: float = 0.0,
        lambda_sync: float = 0.0,
        mask_pos_weight: float = 4.0,
        supcon_temperature: float = 0.1,
    ) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.lambda_mask = lambda_mask
        self.lambda_supcon = lambda_supcon
        self.lambda_sync = lambda_sync
        self.mask_pos_weight = mask_pos_weight
        self.supcon_temperature = supcon_temperature

    def __call__(
        self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        from ddetect.models.blend_head import boundary_loss
        from ddetect.models.syncnet import av_contrastive_loss

        y = batch["label"]
        main = focal_bce(out["logit"], y, self.gamma, self.alpha, self.label_smoothing)
        total = main
        parts: dict[str, float] = {"main": float(main.detach())}

        if self.lambda_mask > 0 and "pred_mask" in out and "mask" in batch:
            pm = out["pred_mask"]
            # The model emits (B,T,1,h,w); boundary_loss wants (B*T,1,h,w).
            # Accept either, so a stream that already flattened is not
            # silently reshaped into the wrong channel layout.
            if pm.ndim == 5:
                pm = pm.flatten(0, 1)
            ml = boundary_loss(pm, batch["mask"], self.mask_pos_weight)
            total = total + self.lambda_mask * ml
            parts["mask"] = float(ml.detach())

        if self.lambda_supcon > 0 and "emb" in out:
            sc = supcon_loss(out["emb"], y, self.supcon_temperature)
            total = total + self.lambda_supcon * sc
            parts["supcon"] = float(sc.detach())

        if self.lambda_sync > 0 and "v_emb" in out and "a_emb" in out:
            # CRITICAL: the AV-sync encoder trains on REAL
            # video only. Letting a fake into this loss would let the encoder
            # learn generator artefacts, which destroys the single property
            # that makes the sync stream tool-independent.
            real = (y == 0) & batch.get("has_audio", torch.ones_like(y, dtype=torch.bool))
            if real.sum() >= 2:
                sl = av_contrastive_loss(out["v_emb"][real], out["a_emb"][real])
                total = total + self.lambda_sync * sl
                parts["sync"] = float(sl.detach())

        parts["total"] = float(total.detach())
        return total, parts
