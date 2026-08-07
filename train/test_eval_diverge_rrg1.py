#!/usr/bin/env python3
"""Contract tests for DIVERGE-RRG1 evaluation accounting."""

from __future__ import annotations

import torch

from eval_diverge_rrg1 import _fresh_conditions


def _score(*, query_exact: int, logits: torch.Tensor):
    return {
        "by_stage": {
            "EVIDENCE": {"total": 3072, "exact": 3072},
            "QUERY": {"total": 768, "exact": query_exact},
        },
        "query_by_mode": {
            mode: {"total": 256, "exact": 256}
            for mode in ("sensitive", "invariant", "underdetermined")
        },
        "query_by_renderer": {
            str(renderer): {"total": 128, "exact": 128}
            for renderer in range(6)
        },
        "_predictions": [(0, 1)] * 3840,
        "_logits": logits,
    }


def main() -> None:
    logits = torch.arange(3840 * 4, dtype=torch.float32).reshape(3840, 2, 2)
    normal = _score(query_exact=768, logits=logits)
    slot_swap = _score(query_exact=600, logits=logits.flip(2))
    deleted = _score(query_exact=700, logits=torch.zeros_like(logits))
    renamed = _score(query_exact=768, logits=logits.clone())
    report = {
        "natural_query_path": {
            "counts": {
                "source_program_exact": 256,
                "evidence_exact": 3072,
                "episodes_fully_sealed": 256,
                "sensitive_exact": 256,
                "extensional_parity": 256,
                "no_evidence_abstain": 256,
                "invariant_exact": 256,
                "partial_underdetermined_abstain": 256,
                "packet_query_swap_reject": 256,
                "post_seal_poison_invariant": 256,
                "invalid_queries_accepted": 0,
            }
        },
        "fresh_nve1": {
            "counts": {
                "learned_exact": 256,
                "shuffled_exact": 0,
                "state_reset_exact": 0,
                "operation_shift_exact": 0,
                "false_commitment": 0,
                "malformed_accepted": 0,
                "learned_gold_preserved": 256,
                "overflow": 0,
            }
        },
    }
    baseline = {
        "direct_query_exact": 764,
        "natural_query_path": {"counts": {"evidence_exact": 3072}},
    }
    conditions = _fresh_conditions(
        report,
        baseline,
        normal,
        slot_swap,
        deleted,
        renamed,
        owner_hashes_exact=True,
    )
    assert all(conditions.values()), conditions
    changed = dict(renamed)
    changed["_logits"] = logits.clone()
    changed["_logits"][0, 0, 0] += 1
    assert not _fresh_conditions(
        report,
        baseline,
        normal,
        slot_swap,
        deleted,
        changed,
        owner_hashes_exact=True,
    )["entity_rename_all_logits_bit_exact"]
    print("DIVERGE-RRG1 evaluator tests passed")


if __name__ == "__main__":
    main()
