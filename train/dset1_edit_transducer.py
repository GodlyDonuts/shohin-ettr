"""Strict script parsing, execution, and normalized CE for DSET1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


KEEP = "<KEEP>"
REPLACE_LAST = "<REPLACE_LAST>"


class DSET1Error(RuntimeError):
    """A DSET1 script, edit, or objective violates the frozen contract."""


@dataclass(frozen=True)
class EditScript:
    action: str
    old: str | None = None
    new: str | None = None


def render_script(action: str, old: str | None = None, new: str | None = None) -> str:
    if action == KEEP and old is None and new is None:
        return f"{KEEP}\n"
    if (
        action == REPLACE_LAST
        and old
        and new
        and "\n" not in old
        and "\n" not in new
    ):
        return f"{REPLACE_LAST}\n{old}\n{new}\n"
    raise DSET1Error("DSET1 script fields differ")


def parse_script(text: str) -> EditScript:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines == [KEEP]:
        return EditScript(KEEP)
    if len(lines) == 3 and lines[0] == REPLACE_LAST and lines[1] and lines[2]:
        return EditScript(REPLACE_LAST, lines[1], lines[2])
    raise DSET1Error("DSET1 generated script is malformed")


def execute_script(draft: str, script: EditScript) -> str:
    if script.action == KEEP:
        return draft
    if script.action != REPLACE_LAST or not script.old or not script.new:
        raise DSET1Error("DSET1 script action differs")
    index = draft.rfind(script.old)
    if index < 0:
        raise DSET1Error("DSET1 old surface is absent")
    return draft[:index] + script.new + draft[index + len(script.old) :]


def normalized_script_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Average token CE per presentation, then equally across presentations."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise DSET1Error("DSET1 loss geometry differs")
    shifted = labels[:, 1:]
    losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        shifted.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shifted.shape)
    rows = []
    for row_index in range(labels.shape[0]):
        valid = shifted[row_index] != -100
        if not torch.any(valid):
            raise DSET1Error("DSET1 script target is empty")
        rows.append(losses[row_index][valid].mean())
    return torch.stack(rows).mean()
