from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline.materialize_jsonl_prefix import (
    PrefixMaterializationError,
    materialize_prefix,
)


def _fixture(tmp_path: Path, count: int = 5) -> tuple[Path, Path, str]:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"index": index}) + "\n" for index in range(count)),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    report = tmp_path / "source.report.json"
    report.write_text(
        json.dumps(
            {"status": "complete", "rows": count, "output_sha256": source_sha256}
        ),
        encoding="utf-8",
    )
    return source, report, source_sha256


def test_materialize_prefix_preserves_order_and_provenance(tmp_path: Path) -> None:
    source, source_report, source_sha256 = _fixture(tmp_path)
    output = tmp_path / "prefix.jsonl"
    report = materialize_prefix(
        source,
        source_report,
        output,
        tmp_path / "prefix.report.json",
        rows=3,
        expected_source_sha256=source_sha256,
    )
    assert [json.loads(line)["index"] for line in output.read_text().splitlines()] == [
        0,
        1,
        2,
    ]
    assert report["rows"] == 3
    assert report["source_declared_sha256"] == source_sha256
    assert report["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_materialize_prefix_rejects_wrong_parent_hash(tmp_path: Path) -> None:
    source, source_report, _ = _fixture(tmp_path)
    with pytest.raises(PrefixMaterializationError, match="does not match"):
        materialize_prefix(
            source,
            source_report,
            tmp_path / "prefix.jsonl",
            tmp_path / "prefix.report.json",
            rows=3,
            expected_source_sha256="0" * 64,
        )


def test_materialize_prefix_removes_partial_output_on_bad_json(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"index": 0}\nnot-json\n', encoding="utf-8")
    source_report = tmp_path / "source.report.json"
    declared = hashlib.sha256(source.read_bytes()).hexdigest()
    source_report.write_text(
        json.dumps({"status": "complete", "rows": 2, "output_sha256": declared}),
        encoding="utf-8",
    )
    output = tmp_path / "prefix.jsonl"
    with pytest.raises(PrefixMaterializationError, match="malformed"):
        materialize_prefix(
            source,
            source_report,
            output,
            tmp_path / "prefix.report.json",
            rows=2,
            expected_source_sha256=declared,
        )
    assert not output.exists()
