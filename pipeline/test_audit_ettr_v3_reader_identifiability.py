from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from audit_ettr_v3_reader_identifiability import (
    ReaderIdentifiabilityAuditError,
    audit,
)


def _row(*operators: str) -> dict[str, object]:
    return {
        "assessor_only": {
            "semantic_factors": {
                "queries": [{"op": operator} for operator in operators]
            }
        }
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="ascii", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_audit_quantifies_address_sensitive_query_support(tmp_path: Path) -> None:
    _write(
        tmp_path / "local_rewrite-test.jsonl.gz",
        [
            _row("slot_is", "type_count_ge"),
            _row("adjacent_is", "pattern_exists"),
        ],
    )
    _write(
        tmp_path / "resource-test.jsonl.gz",
        [_row("resource_place_ge", "resource_halt")],
    )
    report = audit(tmp_path)
    assert report["query_count"] == 6
    assert report["address_sensitive_query_count"] == 4
    assert report["address_sensitive_query_rate"] == pytest.approx(2 / 3)
    assert report["reader_without_addresses_representable_count"] == 2
    assert report["shard_count"] == 2


def test_audit_rejects_an_unclassified_operator(tmp_path: Path) -> None:
    _write(tmp_path / "horn-test.jsonl.gz", [_row("unknown", "horn_has")])
    with pytest.raises(
        ReaderIdentifiabilityAuditError,
        match="unclassified",
    ):
        audit(tmp_path)
