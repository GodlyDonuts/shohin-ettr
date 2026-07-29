import json
from pathlib import Path

import pytest
import zstandard as zstd

from pipeline.tokenize_shards import canonical_payload_sha256, sha256_file
from pipeline.verify_tokenized_shards import (
    ShardVerificationError,
    verify_manifest,
)


def build_corpus(root: Path) -> tuple[Path, Path]:
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
        "schema": "shohin-tokenized-shards-v2",
        "selection_code_sha256": sha256_file(selection_code),
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
    }
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    (shard_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return shard_dir, selection_code


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
