import hashlib
import io
import json
from pathlib import Path

import pytest
import zstandard as zstd

from pipeline.audit_cross_source_exact_dedup import (
    CorpusSpec,
    audit_exact_duplicates,
)
from pipeline.materialize_cross_source_exact_residual import (
    _write_shard,
    ExactResidualError,
    materialize_exact_residual,
)
from pipeline.test_verify_tokenized_shards import build_corpus
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    canonical_payload_sha256,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


SELECTION_CODE = Path(__file__).with_name(
    "materialize_cross_source_exact_residual.py"
)


def test_new_shard_receipt_is_bound_to_written_bytes(tmp_path: Path) -> None:
    receipt = _write_shard(
        tmp_path,
        index=7,
        payload=bytearray(b"\x01\x00\x02\x00"),
    )
    path = tmp_path / receipt["path"]
    assert receipt["bytes"] == path.stat().st_size
    assert receipt["sha256"] == sha256_file(path)
    assert receipt["tokens"] == 2


def _make_two_document_corpus(corpus: Path) -> None:
    raw = b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00"
    shard = corpus / "shard_00000.u16.zst"
    shard.write_bytes(zstd.ZstdCompressor(level=3).compress(raw))
    rows = [
        {
            "allowed_value": "CCBY",
            "chars": 100,
            "document_sha256": "a" * 64,
            "domain": "duplicate.example",
            "schema": DOCUMENT_LEDGER_SCHEMA,
            "shard": shard.name,
            "source_row_index": 7,
            "stable_identity_sha256": "b" * 64,
            "token_end": 3,
            "token_sha256": hashlib.sha256(raw[:6]).hexdigest(),
            "token_start": 0,
            "tokens": 3,
        },
        {
            "allowed_value": "CCBY",
            "chars": 80,
            "document_sha256": "c" * 64,
            "domain": "unique.example",
            "schema": DOCUMENT_LEDGER_SCHEMA,
            "shard": shard.name,
            "source_row_index": 8,
            "stable_identity_sha256": "d" * 64,
            "token_end": 5,
            "token_sha256": hashlib.sha256(raw[6:]).hexdigest(),
            "token_start": 3,
            "tokens": 2,
        },
    ]
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("ascii")
    ledger = corpus / DOCUMENT_LEDGER_NAME
    ledger.write_bytes(zstd.ZstdCompressor(level=3).compress(payload))
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tokens"] = 5
    manifest["kept"] = 2
    manifest["filters"] = {"exact_dedup": True}
    manifest["shard_files"] = [
        {
            "path": shard.name,
            "bytes": shard.stat().st_size,
            "tokens": 5,
            "sha256": sha256_file(shard),
        }
    ]
    manifest["document_ledger"] = {
        "path": DOCUMENT_LEDGER_NAME,
        "bytes": ledger.stat().st_size,
        "sha256": sha256_file(ledger),
        "rows": 2,
        "tokens": 5,
        "contains_document_text": False,
        "schema": DOCUMENT_LEDGER_SCHEMA,
    }
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, first_selection = build_corpus(first_root, schema="v3")
    second, _second_selection = build_corpus(second_root, schema="v3")
    _make_two_document_corpus(second)
    for corpus in (first,):
        manifest_path = corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["filters"] = {"exact_dedup": True}
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = canonical_payload_sha256(manifest)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    dedup = tmp_path / "dedup"
    audit_exact_duplicates(
        (
            CorpusSpec("incumbent", first),
            CorpusSpec("challenger", second),
        ),
        selection_code=first_selection,
        output_dir=dedup,
        require_external_inputs=True,
    )
    return second, dedup, first_selection


def test_exact_removal_is_applied_and_residual_reverifies(tmp_path):
    source, dedup, _selection = _prepare(tmp_path)
    output = tmp_path / "residual"
    result = materialize_exact_residual(
        source_dir=source,
        dedup_dir=dedup,
        corpus_name="challenger",
        selection_code=SELECTION_CODE,
        output_dir=output,
        shard_tokens=1,
    )
    assert result["documents"] == 1
    assert result["tokens"] == 2
    assert result["dropped_documents"] == 1
    assert result["dropped_tokens"] == 3
    verification = verify_manifest(
        output,
        selection_code=SELECTION_CODE,
        require_external_inputs=True,
    )
    assert verification["document_rows"] == 1
    with (output / DOCUMENT_LEDGER_NAME).open("rb") as source_file:
        with zstd.ZstdDecompressor().stream_reader(source_file) as reader:
            row = json.loads(
                io.TextIOWrapper(reader, encoding="ascii").read()
            )
    assert row["stable_identity_sha256"] == "d" * 64
    assert row["token_start"] == 0
    assert row["token_end"] == 2


def test_tampered_removal_artifact_fails_without_output(tmp_path):
    source, dedup, _selection = _prepare(tmp_path)
    removal_path = dedup / "exact_duplicate_removals.jsonl.zst"
    removal_path.write_bytes(b"tampered")
    output = tmp_path / "residual"
    with pytest.raises(ExactResidualError, match="removal artifact differs"):
        materialize_exact_residual(
            source_dir=source,
            dedup_dir=dedup,
            corpus_name="challenger",
            selection_code=SELECTION_CODE,
            output_dir=output,
        )
    assert not output.exists()


def test_wrong_corpus_binding_fails_closed(tmp_path):
    source, dedup, _selection = _prepare(tmp_path)
    with pytest.raises(ExactResidualError, match="corpus name"):
        materialize_exact_residual(
            source_dir=source,
            dedup_dir=dedup,
            corpus_name="missing",
            selection_code=SELECTION_CODE,
            output_dir=tmp_path / "residual",
        )
