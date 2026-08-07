#!/usr/bin/env python3
"""Contract tests for DIVERGE-CCR1 evaluation accounting."""

from __future__ import annotations

import torch

from diverge_ccr1_data import query_text
from diverge_iem1_runtime import tensorize_queries
from eval_diverge_ccr1 import _fresh_conditions, _rename_records


def _score(*, query_exact: int, predictions: list[tuple[int, int]]):
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
        "_predictions": predictions,
    }


def main() -> None:
    records = [
        {
            "source_text": query_text(
                0, target="asteroid", distractor="birchwood"
            ),
            "symbols": ["asteroid", "birchwood", "coralstone"],
            "symbol_role_ids": [0, 1],
            "stage": "QUERY",
            "mode": "sensitive",
            "renderer": 0,
        }
    ]
    renamed = _rename_records(records)
    assert "asteroid" not in renamed[0]["source_text"]
    assert "birchwood" not in renamed[0]["source_text"]
    assert renamed[0]["symbol_role_ids"] == [0, 1]
    tensorize_queries(renamed, torch.device("cpu"))

    predictions = [(0, 1)] * 3840
    normal = _score(query_exact=768, predictions=predictions)
    swap = _score(query_exact=0, predictions=[(1, 0)] * 3840)
    deleted = _score(query_exact=384, predictions=[(0, 1)] * 3840)
    renamed_score = _score(query_exact=768, predictions=predictions)
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
        swap,
        deleted,
        renamed_score,
        owner_hashes_exact=True,
    )
    assert all(conditions.values()), conditions
    changed = dict(renamed_score)
    changed["_predictions"] = [(1, 0), *predictions[1:]]
    assert not _fresh_conditions(
        report,
        baseline,
        normal,
        swap,
        deleted,
        changed,
        owner_hashes_exact=True,
    )["entity_rename_all_assignments_invariant"]
    print("DIVERGE-CCR1 evaluator tests passed")


if __name__ == "__main__":
    main()
