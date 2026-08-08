#!/usr/bin/env python3
"""AQC1 pair custody tests."""

from __future__ import annotations

import argparse
import hashlib
import json

from build_aqc1_pairs import OUTCOMES, build


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_preserves_holdout_and_candidate_binding(tmp_path) -> None:
    tasks = ("math500", "bbh_logic", "mbpp")
    paths = {}
    for source_split in ("development", "holdout"):
        data = []
        idr1 = []
        control = []
        for index in range(500):
            identity = hashlib.sha256(f"{source_split}:{index}".encode()).hexdigest()
            task = tasks[index % len(tasks)]
            outcome = OUTCOMES[index % len(OUTCOMES)]
            left_correct = outcome in {"both_correct", "idr1_only"}
            right_correct = outcome in {"both_correct", "control_only"}
            data.append(
                {
                    "identity_sha256": identity,
                    "split": source_split,
                    "task": task,
                    "question": f"question {index}",
                }
            )
            common = {
                "identity_sha256": identity,
                "task": task,
                "generated_tokens": 8,
                "max_token_exhausted": False,
            }
            idr1.append({**common, "completion": "left", "correct": left_correct})
            control.append(
                {**common, "completion": "right", "correct": right_correct}
            )
        for label, rows in (("data", data), ("idr1", idr1), ("control", control)):
            path = tmp_path / f"{source_split}_{label}.jsonl"
            _write_jsonl(path, rows)
            paths[(source_split, label)] = path

    output = tmp_path / "pairs.jsonl"
    report_path = tmp_path / "report.json"
    report = build(
        argparse.Namespace(
            development_data=paths[("development", "data")],
            holdout_data=paths[("holdout", "data")],
            idr1_development=paths[("development", "idr1")],
            idr1_holdout=paths[("holdout", "idr1")],
            control_development=paths[("development", "control")],
            control_holdout=paths[("holdout", "control")],
            output=output,
            report=report_path,
            seed=2026080820,
        )
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["status"] == "complete"
    assert report["rows"] == 1_000
    assert report["split_counts"]["holdout"] == 500
    assert {row["source_split"] for row in rows if row["split"] == "holdout"} == {
        "holdout"
    }
    assert all(
        [candidate["lineage"] for candidate in row["candidates"]]
        == ["idr1", "control"]
        for row in rows
    )
