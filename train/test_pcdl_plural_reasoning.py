from dataclasses import replace

import torch
import torch.nn.functional as F

from fcpt_plural_reasoning import BoardConfig, generate_batch
from pcdl_plural_reasoning import (
    DeterminingLawReasoner,
    context_loss,
    parameter_count,
)
from prompt_conditioned_determining_law import DeterminingLawConfig


def test_reasoner_trains_one_finite_step() -> None:
    torch.manual_seed(89)
    board = BoardConfig(width=24)
    law = DeterminingLawConfig(width=24, rank=6, heads=3)
    model = DeterminingLawReasoner(board, law, "law")
    batch = generate_batch(9, 3, board, seed=97)
    result = model(batch)
    loss = F.cross_entropy(result.selected_logits, batch.answer) + 0.5 * context_loss(
        result, batch
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.core.law.basis.network[1].weight.grad is not None


def test_reasoner_arms_are_parameter_matched() -> None:
    board = BoardConfig(width=24)
    law = DeterminingLawConfig(width=24, rank=6, heads=3)
    counts = {
        arm: parameter_count(DeterminingLawReasoner(board, law, arm))
        for arm in ("law", "dense")
    }
    assert len(set(counts.values())) == 1


def test_late_query_is_not_part_of_law_solve() -> None:
    board = BoardConfig(width=24)
    law = DeterminingLawConfig(width=24, rank=6, heads=3)
    model = DeterminingLawReasoner(board, law, "law")
    batch = generate_batch(6, 3, board, seed=101)
    normal = model(batch)
    changed = replace(batch, query_fields=batch.query_fields.roll(1, 0))
    shifted = model(changed)
    assert torch.allclose(normal.law_coefficients, shifted.law_coefficients)
    assert not torch.allclose(normal.law_logits, shifted.law_logits)
