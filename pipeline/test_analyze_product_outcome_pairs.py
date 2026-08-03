import json
from pathlib import Path

import pytest

from pipeline.analyze_product_outcome_pairs import OutcomePairError
from pipeline.analyze_product_outcome_pairs import analyze_reports


def _report(path: Path, task: str, outcomes: list[bool]) -> Path:
    rows = [
        {
            "identity_sha256": f"{task}-{index}",
            "question": f"question {index}",
            "gold": str(index),
            "correct": outcome,
        }
        for index, outcome in enumerate(outcomes)
    ]
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "task": task,
                "data_sha256": f"data-{task}",
                "model_revision": "revision",
                "selection_sha256": f"selection-{task}",
                "results": rows,
            }
        )
    )
    return path


def test_outcome_pair_gate_opens_for_material_complementarity(tmp_path: Path) -> None:
    baseline = _report(
        tmp_path / "baseline.json",
        "math500",
        [True] * 6 + [False] * 4,
    )
    dense = _report(
        tmp_path / "dense.json",
        "math500",
        [True] * 5 + [False] + [True] * 3 + [False],
    )

    report = analyze_reports(
        [baseline],
        [dense],
        minimum_oracle_lift=0.05,
        minimum_exclusive_rate=0.05,
    )

    assert report["aggregate"]["counts"] == {
        "both_correct": 5,
        "baseline_only": 1,
        "dense_only": 3,
        "both_wrong": 1,
    }
    assert report["aggregate"]["paired_oracle_accuracy"] == 0.9
    assert report["decision"] == "train-outcome-gate"


def test_outcome_pair_gate_closes_when_one_expert_dominates(tmp_path: Path) -> None:
    baseline = _report(tmp_path / "baseline.json", "math500", [False] * 10)
    dense = _report(tmp_path / "dense.json", "math500", [True] * 8 + [False] * 2)

    report = analyze_reports(
        [baseline],
        [dense],
        minimum_oracle_lift=0.05,
        minimum_exclusive_rate=0.05,
    )

    assert report["aggregate"]["oracle_lift_over_best"] == 0.0
    assert report["decision"] == "close-outcome-routing"


def test_outcome_pair_rejects_mismatched_identity_sets(tmp_path: Path) -> None:
    baseline = _report(tmp_path / "baseline.json", "math500", [True, False])
    dense = _report(tmp_path / "dense.json", "math500", [False, True])
    payload = json.loads(dense.read_text())
    payload["results"][1]["identity_sha256"] = "different"
    dense.write_text(json.dumps(payload))

    with pytest.raises(OutcomePairError, match="identity sets differ"):
        analyze_reports(
            [baseline],
            [dense],
            minimum_oracle_lift=0.05,
            minimum_exclusive_rate=0.05,
        )
