import hashlib
import json
from pathlib import Path

import pytest

from pipeline.build_blinded_source_comparison import (
    BLINDED_SCHEMA,
    BlindedComparisonError,
    build_comparison,
)


def _write_arm(root: Path, arm: str, *, overlap: str | None = None):
    packet = root / f"{arm}.jsonl"
    rows = []
    for index in range(150):
        identity = hashlib.sha256(f"{arm}-{index}".encode()).hexdigest()
        if overlap is not None and index == 0:
            identity = overlap
        tokens = (1_000, 4_000, 12_000, 40_000)[index % 4]
        rows.append(
            {
                "schema": "shohin-private-selected-source-review-v1",
                "dataset": "test/source",
                "config": "default",
                "stable_identity_sha256": identity,
                "document_sha256": hashlib.sha256(
                    f"document-{arm}-{index}".encode()
                ).hexdigest(),
                "metadata": {"score_that_must_not_leak": index},
                "selection": {
                    "tokens": tokens,
                    "document_policy_tier": arm,
                },
                "review_text": f"Review text for document {index}.",
                "review_text_truncated": False,
            }
        )
    with packet.open("w", encoding="ascii") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    receipt = root / f"{arm}-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "shohin-selected-source-review-receipt-v1",
                "dataset": "test/source",
                "config": "default",
                "review_rows": len(rows),
                "private_packet_bytes": packet.stat().st_size,
                "private_packet_sha256": hashlib.sha256(
                    packet.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="ascii",
    )
    return packet, receipt, rows


def test_comparison_is_balanced_deterministic_and_blind(tmp_path):
    core_packet, core_receipt, _rows = _write_arm(tmp_path, "core")
    residual_packet, residual_receipt, _rows = _write_arm(tmp_path, "residual")
    arms = {
        "core": (core_packet, core_receipt),
        "residual": (residual_packet, residual_receipt),
    }
    first = build_comparison(arms, rows_per_arm=100)
    second = build_comparison(dict(reversed(list(arms.items()))), rows_per_arm=100)
    assert first == second
    blinded, key, receipt = first
    assert len(blinded) == 200
    assert len({row["blind_id"] for row in blinded}) == 200
    assert all(row["schema"] == BLINDED_SCHEMA for row in blinded)
    assert "core" not in json.dumps(blinded)
    assert "residual" not in json.dumps(blinded)
    assert "score_that_must_not_leak" not in json.dumps(blinded)
    assert receipt["selected_counts"]["core"] == receipt["selected_counts"]["residual"]
    assert sum(receipt["matched_length_bucket_quotas"].values()) == 100
    assert {row["arm"] for row in key["rows"]} == {"core", "residual"}


def test_comparison_rejects_cross_arm_identity_overlap(tmp_path):
    overlap = hashlib.sha256(b"overlap").hexdigest()
    first_packet, first_receipt, _rows = _write_arm(
        tmp_path,
        "first",
        overlap=overlap,
    )
    second_packet, second_receipt, _rows = _write_arm(
        tmp_path,
        "second",
        overlap=overlap,
    )
    with pytest.raises(BlindedComparisonError, match="share document"):
        build_comparison(
            {
                "first": (first_packet, first_receipt),
                "second": (second_packet, second_receipt),
            },
            rows_per_arm=100,
        )


def test_comparison_rejects_packet_receipt_drift(tmp_path):
    packet, receipt, _rows = _write_arm(tmp_path, "first")
    packet.write_text(packet.read_text() + "\n", encoding="ascii")
    second_packet, second_receipt, _rows = _write_arm(tmp_path, "second")
    with pytest.raises(BlindedComparisonError, match="does not bind"):
        build_comparison(
            {
                "first": (packet, receipt),
                "second": (second_packet, second_receipt),
            },
            rows_per_arm=100,
        )
