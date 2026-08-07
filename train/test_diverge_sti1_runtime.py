#!/usr/bin/env python3
"""Tests for the zero-training DIVERGE-STI1 routing boundary."""

from __future__ import annotations

import torch

from diverge_iem1_runtime import tensorize_queries
from diverge_nve1_runtime import tensorize_sources
from diverge_rrg1_runtime import RRG1Config, RelationalReferentMachine
from diverge_sti1_runtime import StageTypedInterfaceMachine, validate_owner_contract


def main() -> None:
    torch.manual_seed(2026080625)
    source = RelationalReferentMachine(RRG1Config())
    target = StageTypedInterfaceMachine(RRG1Config())
    target.load_state_dict(source.state_dict(), strict=True)
    target.freeze_owners()
    validate_owner_contract(target)
    assert not any(parameter.requires_grad for parameter in target.parameters())

    evidence = {
        "source_text": (
            "Value 3 is certified for verified register alpha; reject decoy "
            "register beta after instruction 6."
        ),
        "symbols": ["alpha", "beta", "gamma", "delta", "epsilon"],
        "numeric_role_ids": [0, 1],
        "symbol_role_ids": [0, 1],
    }
    ids, mask, bounds, symbols, _, _ = tensorize_sources(
        [evidence], torch.device("cpu")
    )
    expected_evidence = target.numeric_evidence_owner(ids, mask, bounds, symbols)
    actual_evidence = target.evidence_owner(ids, mask, bounds, symbols)
    assert torch.equal(actual_evidence[0], expected_evidence[0])
    assert torch.equal(actual_evidence[1], expected_evidence[1])

    query = {
        "source_text": "Report register alpha; ignore decoy register beta.",
        "symbols": ["alpha", "beta", "gamma", "delta", "epsilon"],
        "symbol_role_ids": [0, 1],
    }
    q_ids, q_mask, q_symbols, _ = tensorize_queries([query], torch.device("cpu"))
    assert torch.equal(
        target.forward_query(q_ids, q_mask, q_symbols),
        target.referent_owner(q_ids, q_mask, q_symbols),
    )
    manifest = target.owner_manifest()
    assert manifest["trainable_parameters"] == 0
    assert manifest["cross_stage_parameter_sharing"] is False
    print("DIVERGE-STI1 runtime tests passed")


if __name__ == "__main__":
    main()
