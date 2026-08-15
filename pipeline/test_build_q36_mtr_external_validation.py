from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import build_q36_mtr_external_validation as module
from build_pcf1_data import PAIR_SCHEMA


def _identity(index: int) -> str:
    return hashlib.sha256(f"external-{index}".encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path, rows: int = 40) -> tuple[Path, list[Path], int]:
    pairs = []
    sources = []
    holdout = 0
    index = 0
    while len(pairs) < rows:
        identity = _identity(index)
        index += 1
        task = "math500" if index % 2 else "bbh_logic"
        pairs.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": identity,
                "task": task,
            }
        )
        sources.append(
            {
                "schema": "synthetic-source-v1",
                "identity_sha256": identity,
                "task": task,
                "question": f"Question {index}?",
                "answer": str(index),
            }
        )
        holdout += module.assigned_split(identity) == "holdout"
    pair_path = tmp_path / "pairs.jsonl"
    bank_path = tmp_path / "bank.jsonl"
    _write_jsonl(pair_path, pairs)
    _write_jsonl(bank_path, sources)
    return pair_path, [bank_path], holdout


def test_build_materializes_only_external_partition(tmp_path, monkeypatch):
    pairs, banks, holdout = _fixture(tmp_path)
    monkeypatch.setattr(module, "ROWS", holdout)
    monkeypatch.setattr(module, "SCREEN_ROWS", min(3, holdout))
    monkeypatch.setattr(module, "VALIDATION_ROWS", holdout - min(3, holdout))
    report = module.build(pairs, banks, tmp_path / "output", verify_hashes=False)
    sources = [
        json.loads(line)
        for line in (tmp_path / "output" / "external_sources.jsonl")
        .read_text()
        .splitlines()
    ]
    assessors = [
        json.loads(line)
        for line in (tmp_path / "output" / "external_assessors.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(sources) == holdout
    assert len(assessors) == holdout
    assert all(row["split"] == "external_validation" for row in sources)
    assert all("assessor" not in row and "answer" not in row for row in sources)
    assert report["development_identity_overlap"] == 0
    assert report["screen_rows"] == min(3, holdout)
    assert report["validation_rows"] == holdout - min(3, holdout)
    screen = {
        json.loads(line)["identity_sha256"]
        for line in (tmp_path / "output" / "screen_sources.jsonl")
        .read_text()
        .splitlines()
    }
    validation = {
        json.loads(line)["identity_sha256"]
        for line in (tmp_path / "output" / "validation_sources.jsonl")
        .read_text()
        .splitlines()
    }
    assert len(validation) == holdout - min(3, holdout)
    assert not screen & validation


def test_existing_output_fails_closed(tmp_path):
    pairs, banks, holdout = _fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(module.Q36MTRExternalValidationError, match="output exists"):
        module.build(pairs, banks, output, verify_hashes=False)


def test_source_pair_mismatch_fails(tmp_path, monkeypatch):
    pairs, banks, holdout = _fixture(tmp_path)
    rows = [json.loads(line) for line in banks[0].read_text().splitlines()]
    rows[0]["task"] = "mbpp"
    _write_jsonl(banks[0], rows)
    monkeypatch.setattr(module, "ROWS", holdout)
    monkeypatch.setattr(module, "SCREEN_ROWS", min(3, holdout))
    monkeypatch.setattr(module, "VALIDATION_ROWS", holdout - min(3, holdout))
    with pytest.raises(module.Q36MTRExternalValidationError, match="binding"):
        module.build(pairs, banks, tmp_path / "output", verify_hashes=False)
