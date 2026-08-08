from __future__ import annotations

import argparse
import hashlib
import json

import build_vcr1_product_data as product


def test_build_uses_original_task_renderer_for_humaneval(tmp_path, monkeypatch) -> None:
    identity = "a" * 64
    pairs = tmp_path / "pairs.jsonl"
    pair = {
        "schema": product.PAIR_SCHEMA,
        "identity_sha256": identity,
        "task": "humaneval",
        "candidates": [
            {"lineage": "base", "completion": "base", "correct": False},
            {"lineage": "expert", "completion": "expert", "correct": True},
        ],
    }
    encoded = (json.dumps(pair) + "\n").encode()
    pairs.write_bytes(encoded)
    pair_report = tmp_path / "pair_report.json"
    pair_report.write_text(
        json.dumps(
            {
                "schema": product.PAIR_SCHEMA,
                "status": "complete",
                "pairs": str(pairs.resolve()),
                "pairs_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        product,
        "assessor_rows",
        lambda _: {
            identity: {
                "task": "humaneval",
                "prompt": "def answer(x):\n    pass",
            }
        },
    )
    output = tmp_path / "product.jsonl"
    report = tmp_path / "report.json"
    product.build(
        argparse.Namespace(
            pairs=pairs,
            pair_report=pair_report,
            output=output,
            report=report,
        )
    )
    built = json.loads(output.read_text(encoding="utf-8"))
    assert "Complete the Python function" in built["question"]
    assert "def answer(x)" in built["question"]
    assert built["runtime_fields"] == ["question"]
