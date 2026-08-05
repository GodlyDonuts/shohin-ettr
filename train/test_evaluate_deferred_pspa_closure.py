import torch

from evaluate_deferred_pspa_closure import close_presentation


def test_deferred_closure_returns_whole_permutations() -> None:
    torch.manual_seed(197)
    probabilities = torch.rand(5, 3, 11, 11).softmax(-1)
    generator_mask = torch.tensor(
        [[True, False, False], [True, True, False], [True, True, True]]
        + [[True, True, True]] * 2
    )
    tables = close_presentation(probabilities, generator_mask)
    assert torch.equal(tables.sum(-1), torch.ones_like(tables.sum(-1)))
    assert torch.equal(tables.sum(-2), torch.ones_like(tables.sum(-2)))

