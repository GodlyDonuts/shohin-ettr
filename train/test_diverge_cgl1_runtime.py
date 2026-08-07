#!/usr/bin/env python3
"""Focused tests for DIVERGE-CGL1 claims and compressed losses."""

from __future__ import annotations

import torch

from diverge_cgl1_runtime import CGL1Config, render_claim_prompt
from train_diverge_cgl1 import CompressedPair, _pair_losses


def main() -> None:
    record = {
        "source_text": "Use register cedar; reject register amber.",
        "symbols": ["amber", "cedar"],
    }
    left = render_claim_prompt(record, 0)
    right = render_claim_prompt(record, 1)
    assert "alpha is the requested answer source" in left
    assert "beta is the requested answer source" in right
    assert left != right
    CGL1Config().validate()

    pair = CompressedPair(
        records=(dict(record), dict(record)),
        targets=(1, 1),
        physical_to_candidate=((1, 0), (1, 0)),
    )
    scores = torch.tensor([[0.0, 4.0], [0.0, 4.0]], requires_grad=True)
    outcome, consistency = _pair_losses(
        scores, [pair], flip_outcomes=False, device=torch.device("cpu")
    )
    flipped, _ = _pair_losses(
        scores, [pair], flip_outcomes=True, device=torch.device("cpu")
    )
    assert float(outcome.detach()) < float(flipped.detach())
    assert float(consistency.detach()) == 0.0
    (outcome + consistency).backward()
    assert scores.grad is not None
    print("DIVERGE-CGL1 runtime tests passed")


if __name__ == "__main__":
    main()
