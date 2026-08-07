#!/usr/bin/env python3
"""Focused mechanics tests for the PQI1 stage-typed composite."""

from __future__ import annotations

import torch
import torch.nn as nn

from diverge_pqi1_composite import PretrainedStageTypedMachine


class TinyQueryOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> torch.Tensor:
        score = symbol_masks[:, 0].sum(-1).float() * self.weight[0]
        return torch.stack((score, -score), dim=-1)


def main() -> None:
    model = PretrainedStageTypedMachine(
        nn.Linear(2, 2),
        nn.Linear(3, 2),
        TinyQueryOwner(),
        tokenizer_sha256="a" * 64,
    )
    assert not any(parameter.requires_grad for parameter in model.parameters())
    ids = torch.zeros((2, 4), dtype=torch.long)
    attention = torch.ones_like(ids, dtype=torch.bool)
    symbols = torch.zeros((2, 2, 4), dtype=torch.bool)
    symbols[:, 0, 1] = True
    assert model.forward_query(ids, attention, symbols).shape == (2, 2)
    before = model.owner_hashes()
    with torch.no_grad():
        model.query_owner.weight.add_(1.0)
    after = model.owner_hashes()
    assert before["WORLD"] == after["WORLD"]
    assert before["EVIDENCE"] == after["EVIDENCE"]
    assert before["QUERY"] != after["QUERY"]
    print("PQI1 composite tests passed")


if __name__ == "__main__":
    main()
