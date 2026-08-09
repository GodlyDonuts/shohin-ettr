import torch

from dseo1_revision import DSEO1ObjectiveError, draft_specific_edit_loss


def _fixture():
    labels = torch.tensor(
        [
            [-100, -100, 1, 2, 3, 4],
            [-100, 2, 1, 3, 4, -100],
        ]
    )
    logits = torch.zeros(2, 6, 8, requires_grad=True)
    return logits, labels


def test_dseo1_loss_normalizes_action_and_final_per_row() -> None:
    logits, labels = _fixture()
    loss = draft_specific_edit_loss(logits, labels, [2, 1])
    expected = torch.tensor(8.0).log()
    assert torch.allclose(loss.action, expected)
    assert torch.allclose(loss.final, expected)
    assert torch.allclose(loss.total, expected)
    assert loss.action_tokens == 3
    assert loss.final_tokens == 5


def test_dseo1_action_errors_receive_half_the_objective() -> None:
    logits, labels = _fixture()
    with torch.no_grad():
        # The first valid target of row zero is label 1, predicted at logit slot 1.
        logits[0, 1, 1] = 8.0
    loss = draft_specific_edit_loss(logits, labels, [2, 1])
    assert loss.action < loss.final
    assert torch.allclose(loss.total, 0.5 * loss.action + 0.5 * loss.final)


def test_final_only_removes_action_gradient_and_preserves_loss_scale() -> None:
    logits, labels = _fixture()
    loss = draft_specific_edit_loss(logits, labels, [2, 1], final_only=True)
    loss.total.backward()
    assert loss.weighted_action.item() == 0.0
    assert torch.allclose(loss.total.detach(), loss.final.detach())
    # Row zero action targets are predicted at positions one and two.
    assert torch.count_nonzero(logits.grad[0, 1:3]).item() == 0
    assert torch.count_nonzero(logits.grad).item() > 0


def test_dseo1_rejects_empty_final_span() -> None:
    logits, labels = _fixture()
    try:
        draft_specific_edit_loss(logits, labels, [4, 1])
    except DSEO1ObjectiveError:
        pass
    else:
        raise AssertionError("empty DSEO1 final span was accepted")
