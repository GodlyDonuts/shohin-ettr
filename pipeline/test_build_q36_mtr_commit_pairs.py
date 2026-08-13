from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_q36_mtr_commit_pairs import (
    CALIBRATION_SEED,
    PAIR_SCHEMA,
    REPORT_SCHEMA,
    build,
    calibration_split,
    expected_outcome,
)
from hf_q36_mtr_evaluate import CANDIDATE_SCHEMA, DATA_SCHEMA, TASKS
from merge_q36_mtr_evaluations import SCHEMA as MERGED_REPORT_SCHEMA
from q36_mtr_roles import MODEL_REVISION


def _write_lines(path: Path, rows: list[dict[str, object]]) -> str:
    encoded = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows
    )
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _arm(tmp_path: Path, identities: list[str], arm: str) -> tuple[Path, Path]:
    candidates = tmp_path / f"{arm}.jsonl"
    rows = [
        {
            "schema": CANDIDATE_SCHEMA,
            "arm": arm,
            "identity_sha256": identity,
            "task": TASKS[index % len(TASKS)],
            "completion": f"{arm}-{index}",
            "generated_tokens": 1,
            "max_token_exhausted": False,
        }
        for index, identity in enumerate(identities)
    ]
    digest = _write_lines(candidates, rows)
    report = tmp_path / f"{arm}.report.json"
    report.write_text(
        json.dumps(
            {
                "schema": MERGED_REPORT_SCHEMA,
                "status": "complete",
                "arm": arm,
                "split": "development",
                "model_revision": MODEL_REVISION,
                "rows": len(rows),
                "exact_identity_coverage": True,
                "duplicate_identities": 0,
                "assessor_board_access_count": 0,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
                "output": str(candidates.resolve()),
                "output_sha256": digest,
                "metrics": None,
                "data_sha256": "filled-by-test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return report, candidates


def test_q36_development_pairs_remain_label_free(tmp_path: Path) -> None:
    identities = [
        hashlib.sha256(f"id-{index}".encode()).hexdigest() for index in range(1289)
    ]
    data = tmp_path / "development.jsonl"
    _write_lines(
        data,
        [
            {
                "schema": DATA_SCHEMA,
                "split": "development",
                "identity_sha256": identity,
                "task": TASKS[index % len(TASKS)],
                "question": f"question-{index}",
            }
            for index, identity in enumerate(identities)
        ],
    )
    revision_report, revision_candidates = _arm(tmp_path, identities, "revision")
    unchanged_report, unchanged_candidates = _arm(tmp_path, identities, "unchanged")
    data_sha256 = hashlib.sha256(data.read_bytes()).hexdigest()
    for arm_report in (revision_report, unchanged_report):
        payload = json.loads(arm_report.read_text())
        payload["data_sha256"] = data_sha256
        arm_report.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output = tmp_path / "pairs.jsonl"
    report = tmp_path / "pairs.report.json"
    payload = build(
        argparse.Namespace(
            split="development",
            data=data,
            revision_report=revision_report,
            revision_candidates=revision_candidates,
            unchanged_report=unchanged_report,
            unchanged_candidates=unchanged_candidates,
            candidates_root=tmp_path,
            output=output,
            report=report,
            seed=CALIBRATION_SEED,
        )
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert payload["schema"] == REPORT_SCHEMA
    assert payload["labels_or_correctness_fields"] == 0
    assert len(rows) == 1289
    assert all(row["schema"] == PAIR_SCHEMA for row in rows)
    assert all("outcome_class" not in row for row in rows)
    assert all(
        set(candidate) == {"lineage", "completion"}
        for row in rows
        for candidate in row["candidates"]
    )


def test_q36_commit_outcomes_and_split_are_deterministic() -> None:
    assert expected_outcome(True, True) == "both_correct"
    assert expected_outcome(True, False) == "revision_only"
    assert expected_outcome(False, True) == "unchanged_only"
    assert expected_outcome(False, False) == "both_wrong"
    identity = "a" * 64
    assert calibration_split(identity, CALIBRATION_SEED) == calibration_split(
        identity, CALIBRATION_SEED
    )
