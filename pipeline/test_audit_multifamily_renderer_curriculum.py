from __future__ import annotations

import json
from pathlib import Path

from audit_multifamily_renderer_curriculum import (
    FAMILIES,
    SEEDS,
    audit_reports,
)


def test_audit_accepts_complete_positive_matrix(tmp_path: Path) -> None:
    for seed in SEEDS:
        for family in FAMILIES:
            control = 5 if family == "affine_modular" else 4
            report = {
                "candidate_time_oracle_calls": 0,
                "candidate_time_search_calls": 0,
                "candidate_time_verifier_calls": 0,
                "development": {
                    "direction_shuffled_control": {
                        "exact": control,
                        "invalid": 0,
                        "total": 8,
                    },
                    "renderer_curriculum_treatment": {
                        "cell_exact": {
                            cell: {"correct": 2, "total": 2}
                            for cell in (
                                "composition",
                                "joint",
                                "law",
                                "renderer",
                            )
                        },
                        "exact": 8,
                        "invalid": 0,
                        "total": 8,
                    },
                },
                "equal_budget": {
                    "base_rows": 24,
                    "counterfactual_rows": 24,
                    "initialization_identical": True,
                    "optimizer_updates_per_arm": 300,
                },
                "held_out_family": family,
                "parameter_receipt": {
                    "complete_system": 125_234_597,
                    "learned_compiler": 152_933,
                },
                "preparation_exact_parser_calls": 24,
                "seed": seed,
                "status": "target_first_renderer_curriculum_smoke",
            }
            path = (
                tmp_path
                / f"renderer_curriculum_holdout_{family}_seed{seed}.json"
            )
            path.write_text(json.dumps(report), encoding="ascii")
    audit = audit_reports(tmp_path)
    assert audit["treatment"]["correct"] == 120
    assert audit["control"]["correct"] == 65
    assert audit["positive_seed_fold_directions"]["correct"] == 15
