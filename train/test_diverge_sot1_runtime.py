#!/usr/bin/env python3
"""Focused tests for SOT1 owner isolation and query computation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from diverge_iem1_runtime import tensorize_queries
from diverge_sot1_runtime import (
    SOT1Config,
    StageOwnedEpistemicMachine,
    module_state_sha256,
    query_owner_parameters,
    validate_owner_isolation,
)


def main() -> None:
    torch.manual_seed(7)
    model = StageOwnedEpistemicMachine(SOT1Config())
    model.freeze_qualified_owners()
    validate_owner_isolation(model)
    before = model.owner_hashes()
    records = [
        {
            "source_text": "Read talos and disregard vesper.",
            "symbols": ["rhea", "talos", "vesper", "yarrow", "zephyr"],
            "symbol_role_ids": [0, 1],
        },
        {
            "source_text": "Ignore talos; report vesper.",
            "symbols": ["rhea", "talos", "vesper", "yarrow", "zephyr"],
            "symbol_role_ids": [1, 0],
        },
    ]
    ids, mask, groups, targets = tensorize_queries(records, torch.device("cpu"))
    logits = model.forward_query(ids, mask, groups)
    assert logits.shape == (2, 2, 2)
    parameters = query_owner_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1)).backward()
    optimizer.step()
    after = model.owner_hashes()
    assert before["WORLD"] == after["WORLD"]
    assert before["EVIDENCE"] == after["EVIDENCE"]
    assert before["QUERY"] != after["QUERY"]
    assert module_state_sha256(model)
    print("diverge SOT1 runtime tests passed")


if __name__ == "__main__":
    main()
