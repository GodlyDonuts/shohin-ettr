#!/usr/bin/env python3
"""Focused data and mechanics tests for DIVERGE-NCP1."""

from __future__ import annotations

import torch

from diverge_eal2_data import build_evaluation_episode
from diverge_ncp1_data import (
    DEVELOPMENT_SEED,
    augment_evaluation_episode,
    build_training_record,
    command_renderer_pairs,
)
from diverge_ncp1_runtime import (
    NaturalCommandPointer,
    greedy_ctc_decode,
    tensorize_commands,
)


def main() -> None:
    assert set(command_renderer_pairs("train")).isdisjoint(
        command_renderer_pairs("development")
    )
    assert {left for left, _ in command_renderer_pairs("train")} == set(range(4))
    assert {right for _, right in command_renderer_pairs("train")} == set(range(4))

    rows = [build_training_record(index) for index in range(4)]
    model = NaturalCommandPointer()
    command_ids, command_mask, alias_ids, alias_mask, lengths = tensorize_commands(
        rows, torch.device("cpu")
    )
    logits = model(command_ids, command_mask, alias_ids, alias_mask)
    assert logits.shape == (4, command_ids.shape[1], 9)
    targets = torch.tensor(
        [value for row in rows for value in row["targets"]], dtype=torch.long
    )
    target_lengths = torch.tensor([len(row["targets"]) for row in rows])
    loss = torch.nn.functional.ctc_loss(
        logits.log_softmax(dim=-1).transpose(0, 1),
        targets,
        lengths,
        target_lengths,
        blank=8,
        zero_infinity=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )
    decoded = greedy_ctc_decode(logits, lengths)
    assert len(decoded) == len(rows)
    synthetic = torch.full((1, 8, 9), -10.0)
    for position, token in enumerate((8, 1, 1, 8, 1, 2, 2, 8)):
        synthetic[0, position, token] = 10.0
    assert greedy_ctc_decode(synthetic, torch.tensor([8])) == [(1, 1, 2)]

    base_public, base_assessor = build_evaluation_episode(0, seed=DEVELOPMENT_SEED)
    public, assessor = augment_evaluation_episode(
        base_public, base_assessor, seed=DEVELOPMENT_SEED, serial=0
    )
    assert all("symbols" not in transfer for transfer in public["transfer"])
    assert len(assessor["command_targets"]) == len(public["transfer"])
    assert public["aliases"] != public["renamed_aliases"]
    print("diverge NCP1 mechanics tests passed")


if __name__ == "__main__":
    main()
