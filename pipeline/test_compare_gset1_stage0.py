import json
from pathlib import Path
from types import SimpleNamespace

from compare_gset1_stage0 import run
from merge_gset1_evaluation_shards import MERGED_SCHEMA


def write_report(path: Path, arm: str, action: int, execution: int) -> Path:
    rows = 100
    payload = {
        "schema": MERGED_SCHEMA,
        "status": "complete",
        "arm": arm,
        "intervention": "predicted",
        "holdout_used": False,
        "row_count": rows,
        "pair_count": 50,
        "data_sha256": "data",
        "dset_checkpoint_sha256": "dset",
        "gate_action_correct": action,
        "execution_correct": execution,
        "counterfactual_consistency": 0.96 if arm == "aligned" else 0.5,
        "family_metrics": {
            "numeric_final": {"gate_action_correct_accuracy": 0.96},
            "choice_final": {"gate_action_correct_accuracy": 0.95},
        },
        "member_metrics": {
            "clean": {"execution_correct_accuracy": 1.0},
            "fault": {"execution_correct_accuracy": 0.98},
        },
        "execution_errors": {},
        "max_token_exhausted": 0,
    }
    path.write_text(json.dumps(payload))
    return path


def test_passing_gate(tmp_path: Path) -> None:
    args = SimpleNamespace(
        aligned=write_report(tmp_path / "a.json", "aligned", 96, 99),
        swapped=write_report(tmp_path / "s.json", "swapped", 10, 20),
        hidden=write_report(tmp_path / "h.json", "hidden", 50, 60),
        output=tmp_path / "out.json",
    )
    report = run(args)
    assert report["passed"]
    assert report["margins"] == {"aligned_minus_swapped": 79, "aligned_minus_hidden": 39}
