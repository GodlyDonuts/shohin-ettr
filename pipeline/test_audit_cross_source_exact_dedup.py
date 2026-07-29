import json
import io
from pathlib import Path

import pytest
import zstandard as zstd

from pipeline.audit_cross_source_exact_dedup import (
    CorpusSpec,
    CrossSourceDedupError,
    audit_exact_duplicates,
)
from pipeline.test_verify_tokenized_shards import build_corpus
from pipeline.tokenize_shards import canonical_payload_sha256


def _set_document_hash(corpus: Path, value: str) -> None:
    ledger_path = corpus / "documents.jsonl.zst"
    row = json.loads(
        zstd.ZstdDecompressor().decompress(ledger_path.read_bytes()).decode("ascii")
    )
    row["document_sha256"] = value
    payload = (
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    ledger_path.write_bytes(zstd.ZstdCompressor(level=3).compress(payload))
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["document_ledger"]["bytes"] = ledger_path.stat().st_size
    from pipeline.tokenize_shards import sha256_file

    manifest["document_ledger"]["sha256"] = sha256_file(ledger_path)
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_first_corpus_wins_and_removal_receipt_is_text_free(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, selection_code = build_corpus(first_root, schema="v3")
    second, _ = build_corpus(second_root, schema="v3")
    for corpus in (first, second):
        manifest_path = corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["filters"] = {"exact_dedup": True}
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = canonical_payload_sha256(manifest)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    output = tmp_path / "dedup"
    report = audit_exact_duplicates(
        (
            CorpusSpec("incumbent", first),
            CorpusSpec("challenger", second),
        ),
        selection_code=selection_code,
        output_dir=output,
        require_external_inputs=False,
    )
    assert report["totals"]["exact_duplicate_documents_dropped"] == 1
    assert report["totals"]["exact_duplicate_tokens_dropped"] == 3
    assert report["corpora"][0]["residual_documents"] == 1
    assert report["corpora"][1]["residual_documents"] == 0
    with (output / "exact_duplicate_removals.jsonl.zst").open("rb") as source:
        with zstd.ZstdDecompressor().stream_reader(source) as reader:
            removal_payload = io.TextIOWrapper(
                reader,
                encoding="ascii",
            ).read()
    removal = json.loads(removal_payload)
    assert removal["keep"]["corpus"] == "incumbent"
    assert removal["drop"]["corpus"] == "challenger"
    assert "review_text" not in removal_payload


def test_unique_documents_remain(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, selection_code = build_corpus(first_root, schema="v3")
    second, _ = build_corpus(second_root, schema="v3")
    _set_document_hash(second, "c" * 64)
    for corpus in (first, second):
        manifest_path = corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["filters"] = {"exact_dedup": True}
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = canonical_payload_sha256(manifest)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    report = audit_exact_duplicates(
        (CorpusSpec("first", first), CorpusSpec("second", second)),
        selection_code=selection_code,
        output_dir=tmp_path / "dedup",
        require_external_inputs=False,
    )
    assert report["totals"]["exact_duplicate_documents_dropped"] == 0
    assert report["totals"]["residual_documents"] == 2


def test_v2_or_non_deduplicated_corpus_fails(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, selection_code = build_corpus(first_root, schema="v3")
    second, _ = build_corpus(second_root, schema="v2")
    output = tmp_path / "dedup"
    with pytest.raises(CrossSourceDedupError, match="v3 payload"):
        audit_exact_duplicates(
            (CorpusSpec("first", first), CorpusSpec("second", second)),
            selection_code=selection_code,
            output_dir=output,
            require_external_inputs=False,
        )
    assert not output.exists()
