from __future__ import annotations

import json
from pathlib import Path

from audit_variable_topology_curriculum import (
    CELLS,
    FAMILIES,
    SEEDS,
    audit_reports,
)


def test_audit_accepts_complete_variable_topology_matrix(
    tmp_path: Path,
) -> None:
    for seed in SEEDS:
        for family in FAMILIES:
            report = {
                "candidate_time_oracle_calls": 0,
                "candidate_time_search_calls": 0,
                "candidate_time_verifier_calls": 0,
                "development": {
                    "direction_shuffled": {
                        "exact": 17,
                        "invalid": 0,
                        "total": 24,
                    },
                    "treatment": {
                        "cell_exact": {
                            cell: {"correct": 4, "total": 4}
                            for cell in CELLS
                        },
                        "collision_exact": 8,
                        "collision_total": 8,
                        "exact": 24,
                        "invalid": 0,
                        "total": 24,
                    },
                    "type_shuffled": {
                        "exact": 20,
                        "invalid": 4,
                        "total": 24,
                    },
                },
                "equal_budget": {
                    "base_rows": 40,
                    "counterfactual_rows": 160,
                    "initialization_identical": True,
                    "optimizer_updates_per_arm": 300,
                },
                "held_out_family": family,
                "parameter_receipt": {
                    "complete_system": 125_234_597,
                    "learned_compiler": 152_933,
                },
                "preparation_exact_parser_calls": 40,
                "seed": seed,
                "status": "variable_topology_semantic_type_curriculum",
            }
            path = (
                tmp_path
                / f"semantic_type_holdout_{family}_seed{seed}.json"
            )
            path.write_text(json.dumps(report), encoding="ascii")
    audit = audit_reports(tmp_path)
    assert audit["totals"]["treatment"]["correct"] == 360
    assert audit["totals"]["direction_shuffled"]["correct"] == 255
    assert audit["totals"]["type_shuffled"]["correct"] == 300
    assert audit["positive_seed_fold_directions"]["type_shuffled"] == {
        "correct": 15,
        "total": 15,
    }
