import hashlib
import json
from pathlib import Path

import pytest
import zstandard as zstd

from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    canonical_payload_sha256,
    sha256_file,
)
from pipeline.verify_tokenized_shards import (
    ShardVerificationError,
    verify_manifest,
)


def build_corpus(root: Path, *, schema: str = "v2") -> tuple[Path, Path]:
    shard_dir = root / "corpus"
    shard_dir.mkdir()
    selection_code = root / "selection.py"
    selection_code.write_text("print('pinned selection')\n")
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text('{"version":"test"}\n')
    eval_file = root / "eval.jsonl"
    eval_file.write_text('{"question":"held out"}\n')
    eval_pickle = root / "evalgrams.pkl"
    eval_pickle.write_bytes(b"frozen-eval-index")

    raw = b"\x01\x00\x02\x00\x03\x00"
    compressed = zstd.ZstdCompressor(level=3).compress(raw)
    shard = shard_dir / "shard_00000.u16.zst"
    shard.write_bytes(compressed)
    manifest = {
        "schema": f"shohin-tokenized-shards-{schema}",
        "selection_code_sha256": sha256_file(selection_code),
        "source_files": [],
        "tokenizer": {
            "path": str(tokenizer),
            "bytes": tokenizer.stat().st_size,
            "sha256": sha256_file(tokenizer),
        },
        "tokens": 3,
        "shards": 1,
        "shard_files": [
            {
                "path": shard.name,
                "bytes": shard.stat().st_size,
                "tokens": 3,
                "sha256": sha256_file(shard),
            }
        ],
        "decontamination": {
            "pickle_path": str(eval_pickle),
            "pickle_bytes": eval_pickle.stat().st_size,
            "pickle_sha256": sha256_file(eval_pickle),
            "eval_files": [
                {
                    "path": str(eval_file),
                    "bytes": eval_file.stat().st_size,
                    "sha256": sha256_file(eval_file),
                }
            ],
        },
        "filters": {"document_policy": None},
    }
    if schema == "v3":
        rows = [
            {
                "allowed_value": "CCBY",
                "chars": 100,
                "document_sha256": "a" * 64,
                "domain": "example.org",
                "schema": DOCUMENT_LEDGER_SCHEMA,
                "shard": shard.name,
                "source_row_index": 7,
                "stable_identity_sha256": "b" * 64,
                "token_end": 3,
                "token_sha256": hashlib.sha256(raw).hexdigest(),
                "token_start": 0,
                "tokens": 3,
            }
        ]
        ledger_payload = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("ascii")
        ledger_path = shard_dir / DOCUMENT_LEDGER_NAME
        ledger_path.write_bytes(
            zstd.ZstdCompressor(level=3).compress(ledger_payload)
        )
        manifest["kept"] = 1
        manifest["document_ledger"] = {
            "path": DOCUMENT_LEDGER_NAME,
            "bytes": ledger_path.stat().st_size,
            "sha256": sha256_file(ledger_path),
            "rows": 1,
            "tokens": 3,
            "contains_document_text": False,
            "schema": DOCUMENT_LEDGER_SCHEMA,
        }
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    (shard_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return shard_dir, selection_code


def rewrite_document_ledger(
    shard_dir: Path,
    *,
    document_policy_tier: str | None,
    include_tier_field: bool,
) -> None:
    ledger_path = shard_dir / DOCUMENT_LEDGER_NAME
    rows = [
        json.loads(line)
        for line in zstd.ZstdDecompressor()
        .decompress(ledger_path.read_bytes())
        .decode("ascii")
        .splitlines()
    ]
    for row in rows:
        if include_tier_field:
            row["document_policy_tier"] = document_policy_tier
        else:
            row.pop("document_policy_tier", None)
    ledger_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("ascii")
    ledger_path.write_bytes(
        zstd.ZstdCompressor(level=3).compress(ledger_payload)
    )
    manifest_path = shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["document_ledger"]["bytes"] = ledger_path.stat().st_size
    manifest["document_ledger"]["sha256"] = sha256_file(ledger_path)
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_complete_bound_corpus_verifies(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path)
    receipt = verify_manifest(
        shard_dir,
        selection_code=selection_code,
        require_external_inputs=True,
    )
    assert receipt["tokens"] == 3
    assert receipt["shards"] == 1
    assert receipt["all_shards_hash_and_token_verified"]
    assert receipt["external_inputs_verified"]
    assert not receipt["document_ledger_verified"]


def test_v3_document_ledger_reconciles_exact_token_ranges(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path, schema="v3")
    receipt = verify_manifest(shard_dir, selection_code=selection_code)
    assert receipt["document_ledger_verified"]
    assert receipt["document_rows"] == 1


def test_v3_generic_document_policy_tier_none_verifies(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path, schema="v3")
    rewrite_document_ledger(
        shard_dir,
        document_policy_tier=None,
        include_tier_field=True,
    )
    receipt = verify_manifest(shard_dir, selection_code=selection_code)
    assert receipt["document_ledger_verified"]


def test_v3_bound_document_policy_tier_and_count_verify(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path, schema="v3")
    manifest_path = shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["filters"]["document_policy"] = {
        "name": "finepdf_core_v1",
        "allowed_tiers": ["core"],
        "retained_tiers": {"core": 1},
    }
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    rewrite_document_ledger(
        shard_dir,
        document_policy_tier="core",
        include_tier_field=True,
    )
    receipt = verify_manifest(shard_dir, selection_code=selection_code)
    assert receipt["document_ledger_verified"]


def test_v3_bound_document_policy_requires_tier_field(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path, schema="v3")
    manifest_path = shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["filters"]["document_policy"] = {
        "name": "finepdf_core_v1",
        "allowed_tiers": ["core"],
        "retained_tiers": {"core": 1},
    }
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(
        ShardVerificationError,
        match="document-policy tier differs",
    ):
        verify_manifest(shard_dir, selection_code=selection_code)


def test_v3_bound_document_policy_reconciles_retained_counts(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path, schema="v3")
    manifest_path = shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["filters"]["document_policy"] = {
        "name": "finepdf_core_v1",
        "allowed_tiers": ["core"],
        "retained_tiers": {"core": 2},
    }
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    rewrite_document_ledger(
        shard_dir,
        document_policy_tier="core",
        include_tier_field=True,
    )
    with pytest.raises(
        ShardVerificationError,
        match="does not reconcile",
    ):
        verify_manifest(shard_dir, selection_code=selection_code)


def test_v3_document_ledger_substitution_fails(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path, schema="v3")
    ledger = shard_dir / DOCUMENT_LEDGER_NAME
    ledger.write_bytes(b"substituted")
    with pytest.raises(ShardVerificationError, match="byte count|SHA-256"):
        verify_manifest(shard_dir, selection_code=selection_code)


def test_external_source_substitution_fails(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path)
    source = tmp_path / "source.json.gz"
    source.write_bytes(b"pinned source")
    manifest_path = shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_files"] = [
        {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    ]
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    source.write_bytes(b"substituted")
    with pytest.raises(ShardVerificationError, match="source file 0"):
        verify_manifest(
            shard_dir,
            selection_code=selection_code,
            require_external_inputs=True,
        )


def test_document_policy_source_substitution_fails(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path)
    policy = tmp_path / "policy.py"
    policy.write_text("POLICY = 'pinned'\n")
    manifest_path = shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["filters"]["document_policy"] = {
        "name": "test-policy",
        "source": {
            "path": str(policy),
            "bytes": policy.stat().st_size,
            "sha256": sha256_file(policy),
        },
    }
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    policy.write_text("POLICY = 'substituted'\n")
    with pytest.raises(ShardVerificationError, match="document-policy source"):
        verify_manifest(
            shard_dir,
            selection_code=selection_code,
            require_external_inputs=True,
        )


def test_shard_substitution_fails(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path)
    (shard_dir / "shard_00000.u16.zst").write_bytes(b"substituted")
    with pytest.raises(ShardVerificationError, match="byte count|SHA-256"):
        verify_manifest(shard_dir, selection_code=selection_code)


def test_unbound_extra_shard_fails(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path)
    (shard_dir / "shard_00001.u16.zst").write_bytes(
        (shard_dir / "shard_00000.u16.zst").read_bytes()
    )
    with pytest.raises(ShardVerificationError, match="unbound"):
        verify_manifest(shard_dir, selection_code=selection_code)


def test_manifest_payload_mutation_fails(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path)
    manifest_path = shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tokens"] = 4
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ShardVerificationError, match="payload SHA-256 differs"):
        verify_manifest(shard_dir, selection_code=selection_code)


def test_symlink_or_hardlinked_shard_fails(tmp_path):
    shard_dir, selection_code = build_corpus(tmp_path)
    shard = shard_dir / "shard_00000.u16.zst"
    payload = shard.read_bytes()
    shard.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(payload)
    shard.symlink_to(outside)
    with pytest.raises(ShardVerificationError, match="single-link"):
        verify_manifest(shard_dir, selection_code=selection_code)

    shard.unlink()
    shard.hardlink_to(outside)
    with pytest.raises(ShardVerificationError, match="single-link"):
        verify_manifest(shard_dir, selection_code=selection_code)
