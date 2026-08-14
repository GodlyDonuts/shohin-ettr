from __future__ import annotations

import itertools

import pytest
import torch

from q36_mtr_setwise_head import SetwiseCommitHead, setwise_selection_loss


def test_setwise_head_is_permutation_equivariant() -> None:
    torch.manual_seed(7)
    head = SetwiseCommitHead(16, width=12, projection=6)
    hidden = torch.randn(4, 3, 16)
    direct = head(hidden)
    for order in itertools.permutations(range(3)):
        permutation = torch.tensor(order)
        permuted = head(hidden[:, permutation])
        assert torch.equal(permuted, direct[:, permutation])


def test_setwise_head_requires_exactly_three_owners() -> None:
    head = SetwiseCommitHead(8, width=8, projection=4)
    with pytest.raises(ValueError, match="geometry"):
        head(torch.zeros(1, 2, 8))


def test_setwise_head_uses_competing_trajectory_context() -> None:
    torch.manual_seed(11)
    head = SetwiseCommitHead(8, width=8, projection=4)
    own = torch.randn(1, 1, 8)
    first_context = torch.cat((own, torch.zeros(1, 2, 8)), dim=1)
    second_context = torch.cat((own, torch.randn(1, 2, 8) * 4), dim=1)
    assert not torch.equal(head(first_context)[:, 0], head(second_context)[:, 0])


def test_setwise_loss_rewards_correct_candidate_ranking() -> None:
    correct = torch.tensor([[True, False, False], [False, True, True]])
    good = torch.tensor([[4.0, -2.0, -3.0], [-3.0, 3.0, 2.0]])
    bad = -good
    assert setwise_selection_loss(good, correct) < setwise_selection_loss(bad, correct)


def test_setwise_loss_handles_all_wrong_rows() -> None:
    correct = torch.zeros((2, 3), dtype=torch.bool)
    low = torch.full((2, 3), -4.0)
    high = torch.full((2, 3), 4.0)
    assert setwise_selection_loss(low, correct) < setwise_selection_loss(high, correct)


@pytest.mark.parametrize(
    "scores,correct",
    [
        (torch.zeros(3), torch.zeros(3, dtype=torch.bool)),
        (torch.zeros(1, 3), torch.zeros(1, 2, dtype=torch.bool)),
        (torch.zeros(1, 3), torch.zeros(1, 3)),
    ],
)
def test_setwise_loss_rejects_wrong_geometry(
    scores: torch.Tensor, correct: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        setwise_selection_loss(scores, correct)
