"""Normalized autoregressive edit-action objective for DSEO1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class DSEO1ObjectiveError(RuntimeError):
    """The action/final loss geometry differs from the frozen contract."""


@dataclass(frozen=True)
class DSEO1Loss:
    total: torch.Tensor
    action: torch.Tensor
    final: torch.Tensor
    weighted_action: torch.Tensor
    weighted_final: torch.Tensor
    action_tokens: int
    final_tokens: int


def draft_specific_edit_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_lengths: list[int],
    *,
    action_weight: float = 0.5,
    final_only: bool = False,
) -> DSEO1Loss:
    """Give each row's action and final spans separately normalized weight."""

    if (
        logits.ndim != 3
        or labels.ndim != 2
        or logits.shape[:2] != labels.shape
        or len(action_lengths) != logits.shape[0]
        or not 0.0 < action_weight < 1.0
    ):
        raise DSEO1ObjectiveError("DSEO1 loss geometry differs")
    shifted_labels = labels[:, 1:]
    token_loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shifted_labels.shape)
    action_rows = []
    final_rows = []
    action_tokens = final_tokens = 0
    for row_index, action_length in enumerate(action_lengths):
        valid = shifted_labels[row_index] != -100
        row_loss = token_loss[row_index][valid]
        if action_length <= 0 or action_length >= row_loss.numel():
            raise DSEO1ObjectiveError("DSEO1 action/final split is empty")
        action_rows.append(row_loss[:action_length].mean())
        final_rows.append(row_loss[action_length:].mean())
        action_tokens += action_length
        final_tokens += int(row_loss.numel()) - action_length
    action = torch.stack(action_rows).mean()
    final = torch.stack(final_rows).mean()
    if final_only:
        # Keep the total objective scale at one while removing all action gradient.
        weighted_action = action.detach() * 0.0
        weighted_final = final
    else:
        weighted_action = action_weight * action
        weighted_final = (1.0 - action_weight) * final
    return DSEO1Loss(
        total=weighted_action + weighted_final,
        action=action,
        final=final,
        weighted_action=weighted_action,
        weighted_final=weighted_final,
        action_tokens=action_tokens,
        final_tokens=final_tokens,
    )
