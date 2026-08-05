import torch
import torch.nn.functional as F

from cgsgr_plural_reasoning import (
    RevisionReasoner,
    final_behavior_loss,
)
from counterexample_guided_revision import RevisionConfig
from fcpt_plural_reasoning import BoardConfig, generate_batch


def test_revision_reasoner_one_step_is_finite() -> None:
    torch.manual_seed(43)
    board = BoardConfig(width=24)
    revision = RevisionConfig(
        width=24,
        heads=3,
        slots=6,
        rounds=2,
        outcome_classes=board.modulus,
        answer_classes=board.modulus,
    )
    model = RevisionReasoner(board, revision, "guided")
    batch = generate_batch(6, 3, board, seed=47)
    logits, trajectory = model(batch)
    loss = F.cross_entropy(logits, batch.answer) + 0.5 * final_behavior_loss(
        trajectory, batch
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.core.revision.gate.weight.grad is not None
    assert torch.isfinite(model.core.revision.gate.weight.grad).all()


def test_guided_and_fixed_reasoners_have_equal_parameters() -> None:
    board = BoardConfig(width=24)
    revision = RevisionConfig(width=24, heads=3)
    guided = RevisionReasoner(board, revision, "guided")
    fixed = RevisionReasoner(board, revision, "fixed")
    guided_count = sum(parameter.numel() for parameter in guided.parameters())
    fixed_count = sum(parameter.numel() for parameter in fixed.parameters())
    assert guided_count == fixed_count
