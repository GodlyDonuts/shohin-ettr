import inspect

import torch

from prompt_selected_presented_algebra import (
    PresentedAlgebraConfig,
    PromptSelectedPresentedAlgebra,
)


def _single_cycle_batch(batch: int = 1):
    config = PresentedAlgebraConfig(
        carrier_size=3,
        maximum_generators=1,
        maximum_observations=1,
        maximum_challenges=1,
        maximum_word_length=2,
    )
    generator = torch.zeros(batch, 1, dtype=torch.long)
    observed_input = torch.zeros(batch, 1, dtype=torch.long)
    observed_output = torch.tensor([[1], [2]][:batch], dtype=torch.long)
    observed_mask = torch.ones(batch, 1, dtype=torch.bool)
    generator_mask = torch.ones(batch, 1, dtype=torch.bool)
    challenge_start = torch.ones(batch, 1, dtype=torch.long)
    challenge_word = torch.zeros(batch, 1, 2, dtype=torch.long)
    challenge_word_mask = torch.zeros(batch, 1, 2, dtype=torch.bool)
    challenge_word_mask[..., 0] = True
    challenge_outcome = torch.tensor([[2], [0]][:batch], dtype=torch.long)
    challenge_mask = torch.ones(batch, 1, dtype=torch.bool)
    query_start = torch.zeros(batch, dtype=torch.long)
    query_word = torch.zeros(batch, 2, dtype=torch.long)
    query_word_mask = torch.ones(batch, 2, dtype=torch.bool)
    return (
        config,
        generator,
        observed_input,
        observed_output,
        observed_mask,
        generator_mask,
        challenge_start,
        challenge_word,
        challenge_word_mask,
        challenge_outcome,
        challenge_mask,
        query_start,
        query_word,
        query_word_mask,
    )


def test_source_challenge_selects_complete_generator_and_executes_query() -> None:
    config, *inputs = _single_cycle_batch()
    result = PromptSelectedPresentedAlgebra(config)(*inputs)
    assert result.challenge_exact.item()
    assert result.selection_margin.item() == 1
    assert result.answer.item() == 2
    assert result.selected_candidate.item() == 1


def test_every_candidate_is_a_whole_permutation() -> None:
    config, *inputs = _single_cycle_batch()
    result = PromptSelectedPresentedAlgebra(config)(*inputs)
    row_sums = result.candidate_tables.sum(-1)
    column_sums = result.candidate_tables.sum(-2)
    assert torch.equal(row_sums, torch.ones_like(row_sums))
    assert torch.equal(column_sums, torch.ones_like(column_sums))


def test_late_query_is_absent_from_compile_interface() -> None:
    parameters = inspect.signature(
        PromptSelectedPresentedAlgebra.compile
    ).parameters
    assert "query_start" not in parameters
    assert "query_word" not in parameters


def test_shuffled_challenges_and_lineage_swap_are_predicted_failures() -> None:
    config, *inputs = _single_cycle_batch(batch=2)
    model = PromptSelectedPresentedAlgebra(config)
    normal = model(*inputs)
    shuffled = model(*inputs, shuffle_challenges=True)
    swapped = model(*inputs, lineage_swap=True)
    assert normal.challenge_exact.all()
    assert not shuffled.challenge_exact.all()
    assert not torch.equal(normal.answer, swapped.answer)
