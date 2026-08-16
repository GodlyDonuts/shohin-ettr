import argparse
import json
from pathlib import Path

import pytest

import apply_q36_mtr_owner_commit as module


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, monkeypatch) -> argparse.Namespace:
    monkeypatch.setattr(module, "DEVELOPMENT_ROWS", 16)
    first = []
    second = []
    selections = []
    for index in range(16):
        identity = f"{index:064x}"
        task = "math500" if index % 2 else "bbh_logic"
        for owner, paths in (("first", first), ("second", second)):
            path = tmp_path / f"{owner}_{index:02d}.jsonl"
            _write(
                path,
                [
                    {
                        "schema": module.CANDIDATE_SCHEMA,
                        "identity_sha256": identity,
                        "split": "development",
                        "task": task,
                        "completion": f"{owner}-{index}",
                        "generated_tokens": 10,
                        "max_token_exhausted": False,
                    }
                ],
            )
            paths.append(path)
        selected = index % 2
        selections.append(
            {
                "schema": module.SELECTION_SCHEMA,
                "identity_sha256": identity,
                "task": task,
                "selected_index": selected,
                "selected_lineage": ("revision", "unchanged")[selected],
                "order_consistent": True,
                "margin": 1.0 if selected == 0 else -1.0,
            }
        )
    selection_path = tmp_path / "selections.jsonl"
    _write(selection_path, selections)
    return argparse.Namespace(
        first_candidates=first,
        second_candidates=second,
        selections=selection_path,
        output=tmp_path / "selected.jsonl",
        report=tmp_path / "report.json",
    )


def test_materializes_exact_selected_owner_rows(tmp_path: Path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    report = module.apply(args)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert report["selected"] == {"first": 8, "second": 8}
    assert report["label_or_assessor_reads"] == 0
    assert [row["completion"] for row in rows] == [
        f"{'first' if index % 2 == 0 else 'second'}-{index}" for index in range(16)
    ]


def test_rejects_nonfinite_selection_margin(tmp_path: Path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in args.selections.read_text().splitlines()]
    rows[0]["margin"] = float("inf")
    _write(args.selections, rows)
    with pytest.raises(module.Q36MTROwnerCommitApplyError, match="selection differs"):
        module.apply(args)
