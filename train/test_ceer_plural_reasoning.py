import torch
import torch.nn.functional as F

from ceer_plural_reasoning import EquilibriumReasoner, final_behavior_loss
from counterexample_guided_revision import RevisionConfig
from fcpt_plural_reasoning import BoardConfig, generate_batch


def test_equilibrium_reasoner_trains_one_finite_step() -> None:
    torch.manual_seed(71)
    board = BoardConfig(width=24)
    revision = RevisionConfig(
        width=24,
        heads=3,
        slots=6,
        rounds=2,
        outcome_classes=board.modulus,
        answer_classes=board.modulus,
    )
    model = EquilibriumReasoner(board, revision, "energy", 0.5)
    batch = generate_batch(6, 3, board, seed=73)
    logits, trajectory = model(batch)
    loss = F.cross_entropy(logits, batch.answer) + 0.5 * final_behavior_loss(
        trajectory, batch
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.core.energy_preconditioner.scale[0].weight.grad is not None


def test_equilibrium_reasoner_arms_have_equal_parameters() -> None:
    board = BoardConfig(width=24)
    revision = RevisionConfig(width=24, heads=3)
    counts = {
        arm: sum(
            parameter.numel()
            for parameter in EquilibriumReasoner(
                board, revision, arm, 0.5
            ).parameters()
        )
        for arm in ("energy", "recurrent")
    }
    assert len(set(counts.values())) == 1
