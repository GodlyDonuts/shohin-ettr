from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from score_diverge_vcr1_gate import (
    DRAFT_SCHEMA,
    REPORT_SCHEMA,
    VCR1GateError,
    score,
)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _identity_hash() -> str:
    digest = hashlib.sha256()
    for index in range(100):
        digest.update(f"{index:064x}".encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _draft(path: Path, task: str, correct: int) -> None:
    rows = [
        {"identity_sha256": f"{index:064x}", "source_correct": index < correct}
        for index in range(100)
    ]
    _write(
        path,
        {
            "schema": DRAFT_SCHEMA,
            "status": "complete",
            "task": task,
            "count": 100,
            "correct": correct,
            "ordered_identity_sha256": _identity_hash(),
            "rows": rows,
        },
    )


def _report(
    path: Path,
    drafts: Path,
    task: str,
    arm: str,
    ablation: str,
    source: int,
    corrected: int,
) -> None:
    wrong_to_right = max(0, corrected - source)
    right_to_wrong = max(0, source - corrected)
    _write(
        path,
        {
            "schema": REPORT_SCHEMA,
            "status": "complete",
            "task": task,
            "arm": arm,
            "ablation": ablation,
            "drafts_sha256": hashlib.sha256(drafts.read_bytes()).hexdigest(),
            "input_rows": 100,
            "evaluated_rows": 100,
            "skipped_length": 0,
            "source_correct": source,
            "corrected_correct": corrected,
            "net_correction": corrected - source,
            "transitions": {
                "wrong_to_right": wrong_to_right,
                "right_to_right": source - right_to_wrong,
                "right_to_wrong": right_to_wrong,
                "wrong_to_wrong": 100 - source - wrong_to_right,
            },
            "draft_ordered_identity_sha256": _identity_hash(),
            "evaluated_ordered_identity_sha256": _identity_hash(),
            "validity_accuracy": 0.8 if arm != "plain" else None,
            "validity_brier": 0.1 if arm != "plain" else None,
            "generated_tokens": 100,
            "exhausted": 0,
        },
    )


def _bundle(tmp_path: Path) -> Namespace:
    values: dict[str, Path] = {"output": tmp_path / "gate.json"}
    for domain, task, source in (
        ("math", "math500", 50),
        ("science", "preformatted_short_answer", 30),
    ):
        drafts = tmp_path / f"{domain}_drafts.json"
        _draft(drafts, task, source)
        values[f"{domain}_drafts"] = drafts
        settings = {
            "plain": ("plain", "normal", source + 1),
            "treatment": ("vcr1", "normal", source + 4),
            "role_blind": ("role_blind", "normal", source + 2),
            "reset": ("vcr1", "reset", source + 1),
            "swap_roles": ("vcr1", "swap_roles", source + 2),
        }
        for name, (arm, ablation, corrected) in settings.items():
            path = tmp_path / f"{domain}_{name}.json"
            _report(path, drafts, task, arm, ablation, source, corrected)
            values[f"{domain}_{name}"] = path
    return Namespace(**values)


def test_frozen_gate_passes_a_complete_causal_bundle(tmp_path: Path) -> None:
    report = score(_bundle(tmp_path))
    assert report["gate_pass"]
    assert report["aggregate"]["treatment_correct"] == 88
    assert report["aggregate"]["source_correct"] == 80


def test_frozen_gate_rejects_identity_drift(tmp_path: Path) -> None:
    args = _bundle(tmp_path)
    payload = json.loads(args.math_plain.read_text(encoding="utf-8"))
    payload["evaluated_ordered_identity_sha256"] = "0" * 64
    _write(args.math_plain, payload)
    with pytest.raises(VCR1GateError, match="accounting"):
        score(args)
