import hashlib
import io
import json
from pathlib import Path

import pytest
import zstandard as zstd

from pipeline.materialize_v3_holdout_split import (
    SPLIT_NAMES,
    classify_document,
    materialize_holdout_split,
)
from pipeline.test_verify_tokenized_shards import build_corpus
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    canonical_payload_sha256,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest
from pipeline.verify_v3_holdout_split import (
    HoldoutVerificationError,
    verify_holdout_split,
)


SELECTION_CODE = Path(__file__).with_name("materialize_v3_holdout_split.py")
SEED = "phase2-test-seed"


def _candidate(
    index: int,
    *,
    domain: str | None,
) -> dict:
    return {
        "allowed_value": "CCBY",
        "chars": 100 + index,
        "document_sha256": hashlib.sha256(
            f"document-{index}".encode()
        ).hexdigest(),
        "domain": domain,
        "schema": DOCUMENT_LEDGER_SCHEMA,
        "source_row_index": index,
        "stable_identity_sha256": hashlib.sha256(
            f"identity-{index}".encode()
        ).hexdigest(),
        "tokens": 2,
    }


def _rows_for_all_splits() -> list[dict]:
    selected: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for index in range(10_000):
        domain = None if index % 7 == 0 else f"domain-{index % 53}.example"
        row = _candidate(index, domain=domain)
        split = classify_document(
            row,
            seed=SEED,
            document_validation_bps=2_500,
            domain_validation_bps=2_500,
        )
        if len(selected[split]) < 4:
            selected[split].append(row)
        if all(len(values) == 4 for values in selected.values()):
            break
    assert all(len(values) == 4 for values in selected.values())
    return sorted(
        [row for values in selected.values() for row in values],
        key=lambda row: row["source_row_index"],
    )


def _build_source(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    source, source_selection = build_corpus(tmp_path, schema="v3")
    rows = _rows_for_all_splits()
    raw = bytearray()
    for offset, row in enumerate(rows):
        payload = bytes((offset + 1, 0, offset + 2, 0))
        row["shard"] = "shard_00000.u16.zst"
        row["token_start"] = len(raw) // 2
        row["token_end"] = row["token_start"] + 2
        row["token_sha256"] = hashlib.sha256(payload).hexdigest()
        raw.extend(payload)
    shard = source / "shard_00000.u16.zst"
    shard.write_bytes(zstd.ZstdCompressor(level=3).compress(bytes(raw)))
    ledger_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("ascii")
    ledger = source / DOCUMENT_LEDGER_NAME
    ledger.write_bytes(zstd.ZstdCompressor(level=3).compress(ledger_payload))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["filters"] = {"exact_dedup": True}
    manifest["tokens"] = len(raw) // 2
    manifest["kept"] = len(rows)
    manifest["shard_files"] = [
        {
            "path": shard.name,
            "bytes": shard.stat().st_size,
            "tokens": len(raw) // 2,
            "sha256": sha256_file(shard),
        }
    ]
    manifest["document_ledger"] = {
        "path": DOCUMENT_LEDGER_NAME,
        "bytes": ledger.stat().st_size,
        "sha256": sha256_file(ledger),
        "rows": len(rows),
        "tokens": len(raw) // 2,
        "contains_document_text": False,
        "schema": DOCUMENT_LEDGER_SCHEMA,
    }
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return source, source_selection, rows


def _ledger_rows(path: Path) -> list[dict]:
    with path.open("rb") as source:
        with zstd.ZstdDecompressor().stream_reader(source) as reader:
            return [
                json.loads(line)
                for line in io.TextIOWrapper(reader, encoding="ascii")
            ]


def test_holdout_split_partitions_and_reverifies(tmp_path):
    source, source_selection, source_rows = _build_source(tmp_path)
    output = tmp_path / "split"
    receipt = materialize_holdout_split(
        source_dir=source,
        source_selection_code=source_selection,
        selection_code=SELECTION_CODE,
        output_dir=output,
        seed=SEED,
        document_validation_bps=2_500,
        domain_validation_bps=2_500,
        shard_tokens=3,
    )
    assert receipt["source"]["documents"] == 12
    assert receipt["source"]["tokens"] == 24
    observed: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        verification = verify_manifest(
            output / split_name,
            selection_code=SELECTION_CODE,
            require_external_inputs=True,
        )
        assert verification["document_rows"] == 4
        rows = _ledger_rows(output / split_name / DOCUMENT_LEDGER_NAME)
        assert [row["source_row_index"] for row in rows] == sorted(
            row["source_row_index"] for row in rows
        )
        for row in rows:
            assert row["stable_identity_sha256"] not in observed
            observed[row["stable_identity_sha256"]] = split_name
    assert set(observed) == {
        row["stable_identity_sha256"] for row in source_rows
    }
    independent = verify_holdout_split(
        output_dir=output,
        source_selection_code=source_selection,
        selection_code=SELECTION_CODE,
    )
    assert independent["partition_verified"]
    assert independent["documents"] == 12
    assert independent["tokens"] == 24


def test_missing_domains_never_enter_domain_holdout():
    for index in range(100):
        row = _candidate(index, domain=None)
        assert classify_document(
            row,
            seed=SEED,
            document_validation_bps=5_000,
            domain_validation_bps=4_000,
        ) != "domain_validation"


def test_wrong_source_selection_code_fails_without_output(tmp_path):
    source, _source_selection, _rows = _build_source(tmp_path)
    wrong = tmp_path / "wrong.py"
    wrong.write_text("print('wrong')\n")
    output = tmp_path / "split"
    with pytest.raises(Exception, match="selection code"):
        materialize_holdout_split(
            source_dir=source,
            source_selection_code=wrong,
            selection_code=SELECTION_CODE,
            output_dir=output,
            seed=SEED,
            document_validation_bps=2_500,
            domain_validation_bps=2_500,
        )
    assert not output.exists()


def test_existing_output_is_never_replaced(tmp_path):
    source, source_selection, _rows = _build_source(tmp_path)
    output = tmp_path / "split"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("do not replace")
    with pytest.raises(FileExistsError):
        materialize_holdout_split(
            source_dir=source,
            source_selection_code=source_selection,
            selection_code=SELECTION_CODE,
            output_dir=output,
            seed=SEED,
            document_validation_bps=2_500,
            domain_validation_bps=2_500,
        )
    assert marker.read_text() == "do not replace"


def test_independent_verifier_rejects_manifest_substitution(tmp_path):
    source, source_selection, _rows = _build_source(tmp_path)
    output = tmp_path / "split"
    materialize_holdout_split(
        source_dir=source,
        source_selection_code=source_selection,
        selection_code=SELECTION_CODE,
        output_dir=output,
        seed=SEED,
        document_validation_bps=2_500,
        domain_validation_bps=2_500,
        shard_tokens=3,
    )
    manifest = output / "train" / "manifest.json"
    manifest.write_text("{}\n")
    with pytest.raises(
        HoldoutVerificationError,
        match="receipt differs",
    ):
        verify_holdout_split(
            output_dir=output,
            source_selection_code=source_selection,
            selection_code=SELECTION_CODE,
        )
