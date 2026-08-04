import torch

from hf_product_reasoning_preference_train import (
    preference_gradient_coefficient,
    preference_loss_value,
)


def test_sequential_coefficients_match_joint_preference_gradient() -> None:
    chosen = torch.tensor(-1.2, requires_grad=True)
    rejected = torch.tensor(-1.0, requires_grad=True)
    loss = preference_loss_value(chosen, rejected, beta=2.0, margin=0.3)
    loss.backward()
    joint_chosen_gradient = chosen.grad.detach().clone()
    joint_rejected_gradient = rejected.grad.detach().clone()

    coefficient = preference_gradient_coefficient(
        chosen.detach(), rejected.detach(), beta=2.0, margin=0.3
    )
    assert torch.allclose(joint_chosen_gradient, -coefficient)
    assert torch.allclose(joint_rejected_gradient, coefficient)


def test_preference_loss_falls_when_chosen_gap_increases() -> None:
    narrow = preference_loss_value(
        torch.tensor(-1.1), torch.tensor(-1.0), beta=2.0, margin=0.0
    )
    wide = preference_loss_value(
        torch.tensor(-0.5), torch.tensor(-1.0), beta=2.0, margin=0.0
    )
    assert wide < narrow
