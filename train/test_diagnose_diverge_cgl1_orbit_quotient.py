#!/usr/bin/env python3
"""Unit checks for the frozen CGL1 permutation-orbit attribution."""

import torch

from diagnose_diverge_cgl1_orbit_quotient import mapped_swap_scores


def main() -> None:
    scores = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    mapped = mapped_swap_scores(scores)
    assert torch.equal(mapped, torch.tensor([[2.0, 1.0], [4.0, -3.0]]))
    assert torch.equal(mapped_swap_scores(mapped), scores)
    for malformed in (torch.zeros(2), torch.zeros(2, 3), torch.zeros(1, 2, 1)):
        try:
            mapped_swap_scores(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed orbit scores were accepted")
    print("CGL1 orbit-attribution tests passed")


if __name__ == "__main__":
    main()
