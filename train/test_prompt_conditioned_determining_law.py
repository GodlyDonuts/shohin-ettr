import torch
import torch.nn.functional as F

from prompt_conditioned_determining_law import (
    DeterminingLawConfig,
    PromptConditionedDeterminingLaw,
)


def _inputs(config: DeterminingLawConfig, batch: int = 7, evidence: int = 6):
    torch.manual_seed(83)
    probes = torch.randn(batch, evidence, config.width)
    outcomes = torch.randint(config.outcome_classes, (batch, evidence))
    mask = torch.ones(batch, evidence, dtype=torch.bool)
    query = torch.randn(batch, config.width)
    return probes, outcomes, mask, query


def test_law_solver_reconstructs_full_rank_witnesses() -> None:
    config = DeterminingLawConfig(width=12, rank=3, heads=3, ridge=1e-5)
    basis = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]
    )
    outcomes = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(outcomes, dtype=torch.bool)
    targets = F.one_hot(outcomes, num_classes=config.outcome_classes).float()
    weighted = basis * mask.unsqueeze(-1)
    gram = torch.einsum("ber,bes->brs", weighted, basis)
    rhs = torch.einsum("ber,bec->brc", weighted, targets)
    coefficients = torch.linalg.solve(
        gram + config.ridge * torch.eye(config.rank).unsqueeze(0), rhs
    )
    prediction = torch.einsum("ber,brc->bec", basis, coefficients).argmax(-1)
    assert torch.equal(prediction, outcomes)


def test_law_and_dense_arms_have_equal_parameters() -> None:
    config = DeterminingLawConfig(width=24, rank=6, heads=3)
    counts = {
        arm: sum(
            parameter.numel()
            for parameter in PromptConditionedDeterminingLaw(config, arm).parameters()
        )
        for arm in ("law", "dense")
    }
    assert len(set(counts.values())) == 1


def test_both_paths_execute_and_train_finitely() -> None:
    config = DeterminingLawConfig(width=24, rank=6, heads=3)
    model = PromptConditionedDeterminingLaw(config, "law")
    probes, outcomes, mask, query = _inputs(config)
    result = model(probes, outcomes, mask, query)
    target = torch.randint(config.outcome_classes, (probes.shape[0],))
    context_loss = F.cross_entropy(
        result.context_logits.reshape(-1, config.outcome_classes),
        outcomes.reshape(-1),
    )
    loss = F.cross_entropy(result.selected_logits, target) + 0.5 * context_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert model.law.basis.network[1].weight.grad is not None
    assert model.dense.answer[-1].weight.grad is None
    assert result.law_logits.shape == result.dense_logits.shape


def test_destroy_law_changes_only_law_output() -> None:
    config = DeterminingLawConfig(width=24, rank=6, heads=3)
    model = PromptConditionedDeterminingLaw(config, "law")
    inputs = _inputs(config)
    normal = model(*inputs)
    destroyed = model(*inputs, destroy_law=True)
    assert not torch.allclose(normal.law_logits, destroyed.law_logits)
    assert torch.allclose(normal.dense_logits, destroyed.dense_logits)


def test_shuffled_outcomes_change_both_paths() -> None:
    config = DeterminingLawConfig(width=24, rank=6, heads=3)
    model = PromptConditionedDeterminingLaw(config, "law")
    inputs = _inputs(config)
    normal = model(*inputs)
    shuffled = model(*inputs, shuffle_outcomes=True)
    assert not torch.allclose(normal.law_logits, shuffled.law_logits)
    assert not torch.allclose(normal.dense_logits, shuffled.dense_logits)
