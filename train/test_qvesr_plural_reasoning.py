import torch
import torch.nn.functional as F

from counterexample_guided_revision import RevisionConfig
from fcpt_plural_reasoning import BoardConfig, generate_batch
from qvesr_plural_reasoning import ValueRevisionReasoner, final_behavior_loss


def test_value_reasoner_trains_one_finite_step() -> None:
    torch.manual_seed(59)
    board = BoardConfig(width=24)
    revision = RevisionConfig(
        width=24,
        heads=3,
        slots=6,
        rounds=2,
        outcome_classes=board.modulus,
        answer_classes=board.modulus,
    )
    model = ValueRevisionReasoner(board, revision, "utility")
    batch = generate_batch(6, 3, board, seed=61)
    logits, trajectory = model(batch)
    loss = F.cross_entropy(logits, batch.answer) + 0.5 * final_behavior_loss(
        trajectory, batch
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.core.selector.output.weight.grad is not None


def test_value_reasoner_arms_have_equal_parameters() -> None:
    board = BoardConfig(width=24)
    revision = RevisionConfig(width=24, heads=3)
    counts = {
        arm: sum(
            parameter.numel()
            for parameter in ValueRevisionReasoner(board, revision, arm).parameters()
        )
        for arm in ("utility", "fixed", "residual")
    }
    assert len(set(counts.values())) == 1
