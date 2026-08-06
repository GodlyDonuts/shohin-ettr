import argparse
import json
from pathlib import Path

from score_diverge_crp1_gate import ARMS, REPORT_SCHEMA, score


def _report(arm: str, ablation: str, variant: str, exact: int, packet: int | None):
    family_exact = exact // 3
    remainder = exact - 3 * family_exact
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "arm": arm,
        "ablation": ablation,
        "variant": variant,
        "data_sha256": "a" * 64,
        "input_rows": 480,
        "evaluated_rows": 480,
        "skipped_length": 0,
        "exact_answers": exact,
        "error_localizations": 400,
        "packet_localizations": packet,
        "joint_correct": min(exact, 390),
        "exhausted": 0,
        "family_metrics": {
            family: {
                "rows": 160,
                "exact_answers": family_exact + (index < remainder),
                "error_localizations": 130,
                "joint": 120,
            }
            for index, family in enumerate(("register", "scalar", "symbolic"))
        },
        "results": [{}] * 480,
    }


def test_frozen_gate_passes_only_material_guarded_advantage(tmp_path: Path) -> None:
    exact = {
        "plain_wrong": 210,
        "guarded_wrong": 330,
        "unguarded_wrong": 285,
        "reset_wrong": 240,
        "shift_wrong": 250,
        "packet_swap_wrong": 260,
        "plain_correct": 450,
        "guarded_correct": 445,
        "unguarded_correct": 440,
    }
    namespace = {"output": tmp_path / "gate.json"}
    for name, (arm, ablation, variant) in ARMS.items():
        path = tmp_path / f"{name}.json"
        packet = None if arm == "plain" else (410 if variant == "wrong" else 450)
        path.write_text(json.dumps(_report(arm, ablation, variant, exact[name], packet)))
        namespace[name] = path
    report = score(argparse.Namespace(**namespace))
    assert report["gate_pass"] is True
    assert report["exact_answers"]["guarded_wrong"] == 330


def test_frozen_gate_rejects_unguarded_tie(tmp_path: Path) -> None:
    namespace = {"output": tmp_path / "gate.json"}
    for name, (arm, ablation, variant) in ARMS.items():
        exact = 330 if name in {"guarded_wrong", "unguarded_wrong"} else 250
        if variant == "correct":
            exact = 445
        path = tmp_path / f"{name}.json"
        packet = None if arm == "plain" else 440
        path.write_text(json.dumps(_report(arm, ablation, variant, exact, packet)))
        namespace[name] = path
    report = score(argparse.Namespace(**namespace))
    assert report["gate_pass"] is False
    assert report["checks"]["guarded_beats_unguarded_by_five_points"] is False
