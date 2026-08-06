#!/usr/bin/env python3
"""CPU mechanics tests for DIVERGE-SRP1 semantic ownership."""

from __future__ import annotations

import torch

from diverge_iem1_runtime import tensorize_queries
from diverge_srp1_runtime import (
    SRP1Config,
    SemanticPrimitiveEpistemicMachine,
    validate_owner_contract,
)


def main() -> None:
    torch.manual_seed(7)
    model = SemanticPrimitiveEpistemicMachine(SRP1Config())
    model.freeze_qualified_owners()
    validate_owner_contract(model)
    record = {
        "source_text": "Reject omega and answer from register alpha.",
        "symbols": ["alpha", "beta", "gamma", "delta", "omega"],
        "symbol_role_ids": [1, 0],
    }
    ids, mask, symbols, _ = tensorize_queries([record], torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        logits = model.forward_query(ids, mask, symbols)
        swapped = model.forward_query(ids, mask, symbols.flip(1))
    assert logits.shape == (1, 2, 2)
    assert torch.allclose(swapped, logits.flip(1), atol=1e-6, rtol=1e-6)
    numeric_bounds = torch.tensor([[[1, 2], [2, 3]]], dtype=torch.long)
    with torch.no_grad():
        numeric, referent = model.evidence_owner(
            ids,
            mask,
            numeric_bounds,
            symbols,
        )
    assert numeric.shape == referent.shape == (1, 2, 2)
    hashes = model.owner_hashes()
    assert hashes["QUERY"] == hashes["REFERENT"]
    assert hashes["EVIDENCE"] != hashes["QUERY"]
    print("DIVERGE-SRP1 runtime tests passed")


if __name__ == "__main__":
    main()

