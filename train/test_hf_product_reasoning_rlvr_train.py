import pytest
import torch

from hf_product_reasoning_rlvr_train import (
    ProductRLVRTrainError,
    policy_objective,
    standardized_group_advantages,
)


def test_group_advantages_center_and_scale_mixed_rewards() -> None:
    advantages = standardized_group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-7)
    assert float(advantages.square().mean()) == pytest.approx(1.0)
    assert advantages[0] > 0
    assert torch.all(advantages[1:] < 0)


def test_group_advantages_suppress_uniform_reward_groups() -> None:
    assert torch.equal(
        standardized_group_advantages(torch.ones(4)),
        torch.zeros(4),
    )
    assert torch.equal(
        standardized_group_advantages(torch.zeros(4)),
        torch.zeros(4),
    )


def test_group_advantages_reject_singleton() -> None:
    with pytest.raises(ProductRLVRTrainError, match="at least two"):
        standardized_group_advantages(torch.ones(1))


def test_policy_objective_raises_positive_reward_log_probability() -> None:
    positive_logp = torch.tensor(-2.0, requires_grad=True)
    policy_objective(positive_logp, torch.tensor(1.5)).backward()
    assert positive_logp.grad is not None
    assert float(positive_logp.grad) < 0


def test_policy_objective_lowers_negative_reward_log_probability() -> None:
    negative_logp = torch.tensor(-2.0, requires_grad=True)
    policy_objective(negative_logp, torch.tensor(-0.5)).backward()
    assert negative_logp.grad is not None
    assert float(negative_logp.grad) > 0
