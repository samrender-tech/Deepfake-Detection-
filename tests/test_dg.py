"""Domain generalisation objectives.

Each of these is an ablation the paper reports, so each has to actually do
what its name says. The failure mode they share is silence: an objective that
is wired up but contributes nothing still trains, still logs a falling loss,
and still produces a number for the table.
"""

from __future__ import annotations

import pytest
import torch

from ddetect.dg import (
    DomainDiscriminator,
    GroupDRO,
    dann_lambda,
    dann_loss,
    domain_ids_from_batch,
    grad_reverse,
    irm_loss,
    irm_penalty,
)


# ==========================================================================
# DANN
# ==========================================================================
def test_gradient_reversal_negates_the_gradient():
    x = torch.randn(4, 8, requires_grad=True)
    grad_reverse(x, 1.0).sum().backward()
    reversed_grad = x.grad.clone()

    x2 = x.detach().clone().requires_grad_(True)
    x2.sum().backward()
    torch.testing.assert_close(reversed_grad, -x2.grad)


def test_dann_lambda_ramps_from_zero_to_one():
    # Full-strength reversal from step 0 destroys the features before the
    # classifier has learned anything.
    assert dann_lambda(0, 100) == pytest.approx(0.0, abs=1e-6)
    assert dann_lambda(100, 100) == pytest.approx(1.0, abs=1e-3)
    assert dann_lambda(25, 100) < dann_lambda(75, 100)


def test_dann_loss_reaches_the_encoder():
    d = DomainDiscriminator(32, n_domains=3)
    emb = torch.randn(8, 32, requires_grad=True)
    dann_loss(d, emb, torch.randint(0, 3, (8,)), lambd=1.0).backward()
    assert emb.grad is not None and float(emb.grad.abs().sum()) > 0


def test_dann_loss_is_safe_on_an_out_of_range_domain():
    d = DomainDiscriminator(32, n_domains=2)
    emb = torch.randn(4, 32, requires_grad=True)
    # A lazily-grown vocabulary can exceed the discriminator's class count;
    # that must degrade to a no-op, not crash a training run mid-epoch.
    assert float(dann_loss(d, emb, torch.tensor([0, 1, 5, 9]))) == 0.0


# ==========================================================================
# GroupDRO
# ==========================================================================
def test_groupdro_upweights_the_worst_group():
    dro = GroupDRO(n_groups=3, eta=0.1)
    per_sample = torch.tensor([0.1, 0.1, 0.1, 5.0, 5.0, 0.1])
    groups = torch.tensor([0, 0, 1, 2, 2, 1])
    for _ in range(30):
        _, stats = dro(per_sample, groups)
    assert dro.weights.argmax().item() == 2, "the worst group was not upweighted"
    assert stats["worst_group_loss"] == pytest.approx(5.0, abs=1e-5)
    assert stats["n_groups_in_batch"] == 3


def test_groupdro_weights_stay_a_distribution():
    dro = GroupDRO(n_groups=4, eta=0.05)
    for _ in range(20):
        dro(torch.rand(16), torch.randint(0, 4, (16,)))
    assert float(dro.weights.sum()) == pytest.approx(1.0, abs=1e-5)
    assert bool((dro.weights >= 0).all())


def test_groupdro_ignores_groups_absent_from_the_batch():
    dro = GroupDRO(n_groups=5, eta=0.1)
    _, stats = dro(torch.rand(6), torch.tensor([0, 0, 1, 1, 1, 0]))
    assert stats["n_groups_in_batch"] == 2
    # Absent groups must not soak up weight they did not earn.
    assert float(dro.weights[2:].sum()) == pytest.approx(0.0, abs=1e-6)


# ==========================================================================
# IRM
# ==========================================================================
def test_irm_penalty_is_zero_for_an_already_invariant_predictor():
    # A predictor at the optimum for this environment has zero gradient
    # w.r.t. the dummy scale.
    logits = torch.tensor([10.0, -10.0, 10.0, -10.0], requires_grad=True)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert float(irm_penalty(logits, y)) < 1e-3


def test_irm_penalty_is_positive_when_the_predictor_is_off_optimum():
    logits = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert float(irm_penalty(logits, y)) > 1e-4


def test_irm_falls_back_to_erm_with_one_environment():
    logits = torch.randn(10, requires_grad=True)
    y = (torch.rand(10) > 0.5).float()
    _, stats = irm_loss(logits, y, torch.zeros(10, dtype=torch.long))
    assert stats["irm_penalty"] == 0.0


def test_irm_loss_is_differentiable_across_environments():
    logits = torch.randn(12, requires_grad=True)
    y = (torch.rand(12) > 0.5).float()
    env = torch.tensor([0] * 6 + [1] * 6)
    loss, stats = irm_loss(logits, y, env, penalty_weight=1.0)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert stats["irm_penalty"] >= 0.0


# ==========================================================================
# domain ids
# ==========================================================================
def test_domain_vocabulary_is_stable_across_batches():
    """A domain that changes index between steps makes DANN and DRO meaningless."""
    b1 = {"forgery_method": ["a", "b"], "label": torch.zeros(2)}
    b2 = {"forgery_method": ["b", "c"], "label": torch.zeros(2)}
    ids1, vocab = domain_ids_from_batch(b1)
    ids2, vocab = domain_ids_from_batch(b2, vocab=vocab)
    assert vocab["a"] == 0 and vocab["b"] == 1 and vocab["c"] == 2
    assert ids1.tolist() == [0, 1]
    assert ids2.tolist() == [1, 2], "an existing domain was re-indexed"
