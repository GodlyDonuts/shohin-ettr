from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_idr1_revision_data import EVAL_SCHEMA, REPORT_SCHEMA
from build_kr2_predecessor_evaluation import build as build_predecessors
from build_kr2_training_data import KEEP_ACTION, build as build_training
from build_vcr1_revision_data import sha256_file
from compare_kr2_stage_owner import (
    ABSOLUTE_FLOOR,
    CONTROL_MARGIN,
    KEEP_PRECISION_FLOOR,
    RETENTION_FLOOR,
)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_kr2_thresholds_and_keep_action_are_frozen() -> None:
    assert KEEP_ACTION == "<KEEP_PREVIOUS>"
    assert ABSOLUTE_FLOOR == 615
    assert CONTROL_MARGIN == 26
    assert RETENTION_FLOOR == 0.98
    assert KEEP_PRECISION_FLOOR == 0.95


def test_kr2_builders_preserve_unique_and_presentation_geometry(tmp_path: Path) -> None:
    banks = [tmp_path / f"bank_{index}.jsonl" for index in range(3)]
    sources: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    drafts: list[dict[str, object]] = []
    train: list[dict[str, object]] = []
    for index in range(5824):
        identity = f"{index:064x}"
        task = ("math500", "bbh_logic", "mbpp")[index % 3]
        source = (
            {
                "identity_sha256": identity,
                "task": task,
                "text": "return one",
                "test_list": ["f()==1"],
                "code": "def f(): return 1",
            }
            if task == "mbpp"
            else {
                "identity_sha256": identity,
                "task": task,
                "question": "What is one?",
                "answer": "1",
            }
        )
        sources.append(source)
        pairs.append(
            {
                "identity_sha256": identity,
                "task": task,
                "outcome_class": "both_wrong",
                "candidates": [
                    {"lineage": "base", "correct": False, "completion": "0"},
                    {"lineage": "expert", "correct": False, "completion": "2"},
                ],
            }
        )
        drafts.append(
            {
                "identity_sha256": identity,
                "task": task,
                "completion": "draft",
            }
        )
        train.append(
            {
                "identity_sha256": f"{index + 10000:064x}",
                "source_identity_sha256": identity,
                "outcome_class": "both_wrong",
                "presentation": 0,
                "question": "source Internal draft:\ndraft end",
                "response": "def f(): return 1" if task == "mbpp" else "\\boxed{1}",
            }
        )
    for index in range(9655 - 5824):
        duplicate = dict(train[index])
        duplicate["identity_sha256"] = f"{index + 20000:064x}"
        duplicate["presentation"] = 1
        train.append(duplicate)
    for bank_index, bank in enumerate(banks):
        _jsonl(bank, sources[bank_index::3])
    pairs_path, drafts_path, train_path = (
        tmp_path / "pairs.jsonl",
        tmp_path / "drafts.jsonl",
        tmp_path / "train.jsonl",
    )
    _jsonl(pairs_path, pairs)
    _jsonl(drafts_path, drafts)
    _jsonl(train_path, train)
    idr_report_path = tmp_path / "idr_report.json"
    idr_report_path.write_text(
        json.dumps(
            {
                "schema": REPORT_SCHEMA,
                "status": "complete",
                "pairs": str(pairs_path),
                "pairs_sha256": sha256_file(pairs_path),
                "drafts": str(drafts_path),
                "drafts_sha256": sha256_file(drafts_path),
                "banks": [
                    {"path": str(path), "sha256": sha256_file(path)} for path in banks
                ],
                "outputs": {
                    "train": {
                        "path": str(train_path.resolve()),
                        "sha256": sha256_file(train_path),
                        "rows": 9655,
                    }
                },
            }
        )
    )
    predecessor_root = tmp_path / "predecessor"
    predecessor_report = build_predecessors(
        argparse.Namespace(
            idr_train=train_path,
            idr_report=idr_report_path,
            output=predecessor_root,
        )
    )
    assert predecessor_report["outputs"]["development"]["rows"] == 5824

    candidate_path = tmp_path / "predecessor_candidates.jsonl"
    candidates = [
        {
            "identity_sha256": row["identity_sha256"],
            "task": row["task"],
            "completion": "def f(): return 1" if row["task"] == "mbpp" else "\\boxed{1}",
            "correct": index % 2 == 0,
        }
        for index, row in enumerate(
            json.loads(line)
            for line in (predecessor_root / "development.jsonl").read_text().splitlines()
        )
    ]
    _jsonl(candidate_path, candidates)
    candidate_report = tmp_path / "candidate_report.json"
    candidate_report.write_text(
        json.dumps(
            {
                "status": "complete",
                "candidates_sha256": sha256_file(candidate_path),
            }
        )
    )
    recursive_path = tmp_path / "recursive.jsonl"
    recursive_rows = []
    for index, source in enumerate(sources[:1289]):
        identity = str(source["identity_sha256"])
        recursive_rows.append(
            {
                "schema": EVAL_SCHEMA,
                "identity_sha256": identity,
                "split": "development",
                "task": source["task"],
                "question": "recursive",
                "internal_draft": {
                    "identity_sha256": identity,
                    "completion": candidates[index]["completion"],
                    "correct": candidates[index]["correct"],
                },
                "candidates": pairs[index]["candidates"],
                "assessor": source,
                "runtime_fields": ["question"],
                "internal_draft_visible": True,
                "external_candidate_text_visible": False,
            }
        )
    _jsonl(recursive_path, recursive_rows)
    recursive_report = tmp_path / "recursive_report.json"
    recursive_report.write_text(
        json.dumps(
            {
                "status": "complete",
                "outputs": {
                    "development": {"sha256": sha256_file(recursive_path)}
                },
            }
        )
    )
    output = tmp_path / "kr2"
    summary = build_training(
        argparse.Namespace(
            idr_train=train_path,
            idr_report=idr_report_path,
            predecessor_data=predecessor_root / "development.jsonl",
            predecessor_candidates=candidate_path,
            predecessor_report=candidate_report,
            recursive_development=recursive_path,
            recursive_report=recursive_report,
            output=output,
        )
    )
    assert summary["counts"]["pairs"] == 9655
    treatment = [
        json.loads(line)
        for line in (output / "keep_or_repair" / "train.jsonl").read_text().splitlines()
    ]
    direct = [
        json.loads(line)
        for line in (output / "direct_rewrite" / "train.jsonl").read_text().splitlines()
    ]
    assert len(treatment) == len(direct) == 9655
    assert treatment[0]["response"] == KEEP_ACTION
    assert direct[0]["response"] != KEEP_ACTION
