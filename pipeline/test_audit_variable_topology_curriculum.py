from __future__ import annotations

import json
from pathlib import Path

from audit_variable_topology_curriculum import (
    CELLS,
    FAMILIES,
    SEEDS,
    _board_manifest,
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
                "candidate_source_bytes_absent_from_deployed_wire": True,
                "candidate_uses_learned_global_partition_on_all_rows": True,
                "candidate_uses_query_role_logits": True,
                "board_manifest_sha256": _board_manifest(seed),
                "development": {
                    "same_weights_direction_swapped": {
                        "cell_exact": {
                            cell: {
                                "correct": 0 if cell == "joint" else 2,
                                "total": 4,
                            }
                            for cell in CELLS
                        },
                        "collision_exact": 3,
                        "exact": 11,
                        "invalid": 0,
                        "sealed_wire_roundtrip": True,
                        "total": 24,
                    },
                    "same_weights_key_scores_negated": {
                        "cell_exact": {
                            cell: {"correct": 0, "total": 4}
                            for cell in CELLS
                        },
                        "collision_exact": 0,
                        "exact": 0,
                        "invalid": 24,
                        "sealed_wire_roundtrip": True,
                        "total": 24,
                    },
                    "same_weights_query_roles_swapped": {
                        "cell_exact": {
                            cell: {"correct": 0, "total": 4}
                            for cell in CELLS
                        },
                        "collision_exact": 0,
                        "exact": 0,
                        "invalid": 24,
                        "sealed_wire_roundtrip": True,
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
                        "sealed_wire_roundtrip": True,
                        "total": 24,
                    },
                },
                "equal_budget": {
                    "base_rows": 40,
                    "control_additional_updates": 0,
                    "counterfactual_rows": 160,
                    "models_trained": 1,
                    "optimizer_updates": 100,
                    "same_weights_controls": True,
                },
                "held_out_family": family,
                "parameter_receipt": {
                    "complete_system": 125_142_277,
                    "learned_compiler": 60_613,
                },
                "preparation_exact_parser_calls": 40,
                "preparation_query_parser_calls": 40,
                "preparation_source_parser_calls": 40,
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
    assert audit["positive_seed_fold_directions"][
        "same_weights_key_scores_negated"
    ] == {
        "correct": 15,
        "total": 15,
    }
