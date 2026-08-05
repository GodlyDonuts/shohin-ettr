import torch
import torch.nn.functional as F

from pspa_presented_reasoning import (
    NeuralConfig,
    PresentedAlgebraConfig,
    PresentedReasoner,
    batch_sha256,
    generate_batch,
)


def test_generated_board_is_deterministic() -> None:
    config = PresentedAlgebraConfig()
    first = generate_batch(18, 8, config, seed=107)
    second = generate_batch(18, 8, config, seed=107)
    assert batch_sha256(first) == batch_sha256(second)


def test_structured_presentation_solves_all_families_at_length_twelve() -> None:
    config = PresentedAlgebraConfig()
    model = PresentedReasoner(config, NeuralConfig(width=24, heads=3, layers=1))
    for family in range(3):
        batch = generate_batch(32, 12, config, seed=109 + family, family=family)
        result = model.structured(batch)
        assert result.answer.eq(batch.answer).all()
        assert result.challenge_exact.all()
        assert result.selected_tables.argmax(-1).eq(batch.true_tables.long()).all()


def test_challenge_and_lineage_interventions_reduce_exactness() -> None:
    config = PresentedAlgebraConfig()
    model = PresentedReasoner(config, NeuralConfig(width=24, heads=3, layers=1))
    batch = generate_batch(96, 12, config, seed=113)
    normal = model.structured(batch).answer.eq(batch.answer).float().mean()
    shuffled = (
        model.structured(batch, shuffle_challenges=True)
        .answer.eq(batch.answer)
        .float()
        .mean()
    )
    swapped = (
        model.structured(batch, lineage_swap=True)
        .answer.eq(batch.answer)
        .float()
        .mean()
    )
    assert normal.item() == 1.0
    assert shuffled.item() < 0.8
    assert swapped.item() < 0.3


def test_neural_controls_train_one_finite_step() -> None:
    torch.manual_seed(127)
    config = PresentedAlgebraConfig()
    model = PresentedReasoner(config, NeuralConfig(width=24, heads=3, layers=1))
    batch = generate_batch(12, 3, config, seed=131)
    recurrent = model.recurrent(batch)
    transformer = model.transformer(batch)
    loss = F.cross_entropy(recurrent, batch.answer) + F.cross_entropy(
        transformer, batch.answer
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.recurrent.answer.weight.grad is not None
    assert model.transformer.answer.weight.grad is not None
