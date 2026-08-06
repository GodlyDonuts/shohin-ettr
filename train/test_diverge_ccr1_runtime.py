#!/usr/bin/env python3
"""Tests for the DIVERGE-CCR1 candidate-relative owner."""

from __future__ import annotations

import torch

from diverge_ccr1_runtime import CCR1Config, CounterfactualCandidateReferent
from diverge_iem1_runtime import CLS_ID


def main() -> None:
    torch.manual_seed(2026080623)
    owner = CounterfactualCandidateReferent(CCR1Config())
    ids = torch.zeros((2, 192), dtype=torch.long)
    ids[:, 0] = CLS_ID
    ids[:, 1:20] = torch.arange(1, 20)
    mask = torch.zeros((2, 192), dtype=torch.bool)
    mask[:, :20] = True
    symbols = torch.zeros((2, 2, 192), dtype=torch.bool)
    symbols[:, 0, 3:7] = True
    symbols[:, 1, 12:16] = True
    normal = owner(ids, mask, symbols)
    swapped_groups = owner(ids, mask, symbols.flip(1))
    assert normal.shape == (2, 2, 2)
    assert torch.allclose(normal[:, 0], swapped_groups[:, 1], atol=1e-6)
    assert torch.allclose(normal[:, 1], swapped_groups[:, 0], atol=1e-6)
    marker_swap = owner(ids, mask, symbols, marker_control="swap")
    assert not torch.allclose(normal, marker_swap)
    deleted = owner(ids, mask, symbols, marker_control="delete")
    assert torch.allclose(deleted, torch.zeros_like(deleted), atol=1e-6)

    mutated = ids.clone()
    mutated[symbols[:, 0] | symbols[:, 1]] = 65
    assert torch.allclose(owner(mutated, mask, symbols), normal, atol=1e-6)
    print("DIVERGE-CCR1 runtime tests passed")


if __name__ == "__main__":
    main()
