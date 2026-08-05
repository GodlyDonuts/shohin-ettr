import torch

from evaluate_counterexample_selected_closure import (
    binary_completion_candidates,
    select_with_challenges,
)
from learned_pspa_language_reasoning import execute_word


def test_binary_candidates_are_whole_and_cover_both_completions() -> None:
    probabilities = torch.full((1, 1, 4, 4), 0.01)
    probabilities[0, 0, 0, 1] = 0.97
    probabilities[0, 0, 1, 3] = 0.97
    probabilities[0, 0, 2, 0] = 0.49
    probabilities[0, 0, 2, 2] = 0.48
    probabilities[0, 0, 3, 2] = 0.49
    probabilities[0, 0, 3, 0] = 0.48
    candidates = binary_completion_candidates(
        probabilities, torch.ones(1, 1, dtype=torch.bool)
    )
    assert candidates.shape == (1, 2, 1, 4, 4)
    assert torch.equal(candidates.sum(-1), torch.ones_like(candidates.sum(-1)))
    assert torch.equal(candidates.sum(-2), torch.ones_like(candidates.sum(-2)))
    assert not torch.equal(candidates[:, 0], candidates[:, 1])


def test_challenge_selects_the_matching_complete_lineage() -> None:
    candidates = torch.zeros(1, 2, 1, 3, 3)
    candidates[0, 0, 0, torch.arange(3), torch.tensor([1, 2, 0])] = 1
    candidates[0, 1, 0, torch.arange(3), torch.tensor([2, 1, 0])] = 1
    start = torch.tensor([[0]])
    word = torch.zeros(1, 1, 2, dtype=torch.long)
    word_mask = torch.tensor([[[True, False]]])
    outcome = torch.tensor([[2]])
    challenge_mask = torch.ones(1, 1, dtype=torch.bool)
    selected, index, _ = select_with_challenges(
        candidates, start, word, word_mask, outcome, challenge_mask
    )
    assert index.item() == 1
    prediction = execute_word(selected, start, word, word_mask).argmax(-1)
    assert prediction.item() == 2

