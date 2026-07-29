import hashlib
import io
import json
from pathlib import Path

import pytest
import zstandard as zstd

from pipeline.audit_cross_source_exact_dedup import CorpusSpec
from pipeline.audit_cross_source_near_dedup import (
    NearDedupError,
    audit_near_duplicates,
)
from pipeline.test_verify_tokenized_shards import build_corpus
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    canonical_payload_sha256,
    sha256_file,
)


def _replace_corpus(corpus: Path, documents: list[tuple[str, str, list[int]]]) -> None:
    raw = bytearray()
    rows = []
    offset = 0
    for index, (identity, document_hash, tokens) in enumerate(documents):
        payload = b"".join(int(token).to_bytes(2, "little") for token in tokens)
        raw.extend(payload)
        rows.append(
            {
                "allowed_value": "CCBY",
                "chars": len(tokens) * 5,
                "document_sha256": document_hash,
                "domain": f"domain-{index}.example",
                "schema": DOCUMENT_LEDGER_SCHEMA,
                "shard": "shard_00000.u16.zst",
                "source_row_index": index,
                "stable_identity_sha256": identity,
                "token_end": offset + len(tokens),
                "token_sha256": hashlib.sha256(payload).hexdigest(),
                "token_start": offset,
                "tokens": len(tokens),
            }
        )
        offset += len(tokens)
    shard = corpus / "shard_00000.u16.zst"
    shard.write_bytes(zstd.ZstdCompressor(level=3).compress(bytes(raw)))
    ledger_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("ascii")
    ledger = corpus / DOCUMENT_LEDGER_NAME
    ledger.write_bytes(zstd.ZstdCompressor(level=3).compress(ledger_payload))
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tokens"] = offset
    manifest["kept"] = len(rows)
    manifest["filters"] = {"exact_dedup": True}
    manifest["tokenizer"]["eos_id"] = None
    manifest["shard_files"] = [
        {
            "path": shard.name,
            "bytes": shard.stat().st_size,
            "tokens": offset,
            "sha256": sha256_file(shard),
        }
    ]
    manifest["document_ledger"] = {
        "path": DOCUMENT_LEDGER_NAME,
        "bytes": ledger.stat().st_size,
        "sha256": sha256_file(ledger),
        "rows": len(rows),
        "tokens": offset,
        "contains_document_text": False,
        "schema": DOCUMENT_LEDGER_SCHEMA,
    }
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, selection = build_corpus(first_root, schema="v3")
    second, _ = build_corpus(second_root, schema="v3")
    base = list(range(100, 228))
    near = base.copy()
    near[60] = 999
    _replace_corpus(
        first,
        [
            ("1" * 64, "a" * 64, base),
            ("2" * 64, "b" * 64, list(range(500, 580))),
        ],
    )
    _replace_corpus(
        second,
        [
            ("3" * 64, "c" * 64, near),
            ("4" * 64, "d" * 64, list(range(1000, 1090))),
        ],
    )
    return first, second, selection


def test_near_duplicate_is_exactly_confirmed_and_removed(tmp_path):
    first, second, selection = _fixture(tmp_path)
    output = tmp_path / "near"
    report = audit_near_duplicates(
        (
            CorpusSpec("incumbent", first),
            CorpusSpec("challenger", second),
        ),
        selection_code=selection,
        output_dir=output,
        minimum_tokens=16,
        batch_documents=1,
        require_external_inputs=True,
    )
    assert report["totals"]["input_documents"] == 4
    assert report["totals"]["near_duplicate_documents_dropped"] == 1
    assert report["corpora"][0]["residual_documents"] == 2
    assert report["corpora"][1]["residual_documents"] == 1
    with (output / "near_duplicate_removals.jsonl.zst").open("rb") as source:
        with zstd.ZstdDecompressor().stream_reader(source) as reader:
            removal = json.loads(
                io.TextIOWrapper(reader, encoding="ascii").read()
            )
    assert removal["keep"]["stable_identity_sha256"] == "1" * 64
    assert removal["drop"]["stable_identity_sha256"] == "3" * 64
    assert removal["comparison"]["jaccard"] >= 0.8


def test_same_batch_near_duplicate_is_removed(tmp_path):
    first, second, selection = _fixture(tmp_path)
    report = audit_near_duplicates(
        (
            CorpusSpec("incumbent", first),
            CorpusSpec("challenger", second),
        ),
        selection_code=selection,
        output_dir=tmp_path / "near",
        minimum_tokens=16,
        batch_documents=100,
        require_external_inputs=True,
    )
    assert report["totals"]["near_duplicate_documents_dropped"] == 1


def test_each_corpus_can_bind_its_own_selection_code(tmp_path):
    first, second, selection = _fixture(tmp_path)
    second_selection = tmp_path / "second-selection.py"
    second_selection.write_text("print('second selection')\n")
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_code_sha256"] = sha256_file(second_selection)
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = audit_near_duplicates(
        (
            CorpusSpec("incumbent", first, selection),
            CorpusSpec("challenger", second, second_selection),
        ),
        selection_code=selection,
        output_dir=tmp_path / "near",
        minimum_tokens=16,
        batch_documents=1,
        require_external_inputs=True,
    )
    assert report["corpora"][0]["selection_code_sha256"] == sha256_file(
        selection
    )
    assert report["corpora"][1]["selection_code_sha256"] == sha256_file(
        second_selection
    )


def test_exact_duplicate_fails_before_publication(tmp_path):
    first, second, selection = _fixture(tmp_path)
    first_manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    with (second / DOCUMENT_LEDGER_NAME).open("rb") as source:
        rows = [
            json.loads(line)
            for line in zstd.ZstdDecompressor()
            .decompress(source.read())
            .decode("ascii")
            .splitlines()
        ]
    rows[0]["document_sha256"] = "a" * 64
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("ascii")
    ledger = second / DOCUMENT_LEDGER_NAME
    ledger.write_bytes(zstd.ZstdCompressor(level=3).compress(payload))
    second_manifest["document_ledger"]["bytes"] = ledger.stat().st_size
    second_manifest["document_ledger"]["sha256"] = sha256_file(ledger)
    second_manifest.pop("payload_sha256")
    second_manifest["payload_sha256"] = canonical_payload_sha256(second_manifest)
    (second / "manifest.json").write_text(
        json.dumps(second_manifest, indent=2, sort_keys=True) + "\n"
    )
    assert first_manifest["payload_sha256"] != second_manifest["payload_sha256"]
    output = tmp_path / "near"
    with pytest.raises(NearDedupError, match="exact duplicate"):
        audit_near_duplicates(
            (
                CorpusSpec("incumbent", first),
                CorpusSpec("challenger", second),
            ),
            selection_code=selection,
            output_dir=output,
            minimum_tokens=16,
            batch_documents=1,
            require_external_inputs=True,
        )
    assert not output.exists()
