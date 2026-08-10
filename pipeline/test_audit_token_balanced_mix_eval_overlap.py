from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.audit_token_balanced_mix_eval_overlap import (
    EvalOverlapAuditError,
    run,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, candidate: str) -> argparse.Namespace:
    reference = tmp_path / "development.jsonl"
    reference.write_text(
        json.dumps(
            {
                "assessor": {
                    "question": (
                        "one two three four five six seven eight nine ten eleven "
                        "twelve thirteen"
                    )
                }
            }
        )
        + "\n"
    )
    data = tmp_path / "mix.jsonl"
    data.write_text(
        json.dumps(
            {
                "question": candidate,
                "response": "verified response",
                "training_group": "math",
            }
        )
        + "\n"
    )
    report = tmp_path / "mix.report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-token-balanced-reasoning-mix-v1",
                "status": "complete",
                "output": str(data.resolve()),
                "output_sha256": _sha(data),
                "selected_rows": 1,
                "eval_overlap_filter": {
                    "ngram_size": 13,
                    "references": [
                        {"path": str(reference.resolve()), "sha256": _sha(reference)}
                    ],
                },
            }
        )
        + "\n"
    )
    return argparse.Namespace(
        data=data,
        report=report,
        eval_reference=[reference],
        ngram=13,
        output=tmp_path / "audit.json",
    )


def test_independent_audit_accepts_disjoint_mix(tmp_path: Path) -> None:
    args = _fixture(tmp_path, "a source-disjoint short question")
    result = run(args)
    assert result["status"] == "complete"
    assert result["exact_overlap_rows"] == 0
    assert result["protected_unique_ngram_overlap_rows"] == 0


def test_independent_audit_rejects_protected_ngram(tmp_path: Path) -> None:
    args = _fixture(
        tmp_path,
        "prefix one two three four five six seven eight nine ten eleven twelve thirteen",
    )
    with pytest.raises(EvalOverlapAuditError, match="overlaps"):
        run(args)
    failed = json.loads(args.output.read_text())
    assert failed["status"] == "failed"
    assert failed["protected_unique_ngram_overlap_rows"] == 1
