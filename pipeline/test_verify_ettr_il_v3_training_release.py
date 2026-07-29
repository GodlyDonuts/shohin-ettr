from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from build_ettr_il_v3_training_release import build_training_release
from ettr_il_v3_protocol import canonical_json_bytes
from test_build_ettr_il_v3_training_release import (
    SOURCE_COMMIT,
    _make_writable,
    _write_inputs,
)
from verify_ettr_il_v3_training_release import (
    ETTRV3ReleaseVerificationError,
    SCHEMA,
    verify_training_release,
)


PROTECTED = "4" * 64
VERIFIER_COMMIT = "6" * 40


def _source_root(tmp_path: Path, *, suffix: bytes = b"") -> Path:
    root = tmp_path / f"source-{len(suffix)}"
    destination = root / "pipeline" / "build_ettr_il_v3_training_release.py"
    destination.parent.mkdir(parents=True)
    source = Path(build_training_release.__code__.co_filename)
    destination.write_bytes(source.read_bytes() + suffix)
    return root


def _release(tmp_path: Path):
    tokenizer, data_root, audit, separation = _write_inputs(tmp_path)
    output = tmp_path / "release"
    build_training_release(
        main_audit_path=audit,
        separation_path=separation,
        data_root=data_root,
        tokenizer_path=tokenizer,
        protected_checkpoint_sha256=PROTECTED,
        source_commit=SOURCE_COMMIT,
        output=output,
        expected_split_counts={
            "development": 1,
            "development_reserve": 0,
            "train": 1,
            "train_reserve": 0,
        },
    )
    return tokenizer, data_root, output


def test_independent_release_verifier_binds_stream_and_packet_index(tmp_path):
    tokenizer, data_root, release = _release(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    result = verify_training_release(
        release_root=release,
        data_root=data_root,
        tokenizer_path=tokenizer,
        release_source_root=_source_root(tmp_path),
        expected_release_source_commit=SOURCE_COMMIT,
        verifier_source_commit=VERIFIER_COMMIT,
        expected_protected_checkpoint_sha256=PROTECTED,
        receipt_path=receipt_path,
    )
    assert result["schema"] == SCHEMA
    assert result["status"] == "pass"
    assert result["packet_index"]["train_batches"] == 4
    assert result["packet_index"]["validation_batches"] == 4
    assert result["source_verification"]["core_rows"] == 2
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_bytes())
    claimed = receipt.pop("payload_sha256")
    assert claimed == result["payload_sha256"]
    assert claimed == hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    _make_writable(release)


def test_independent_release_verifier_rejects_wrong_builder_source(tmp_path):
    tokenizer, data_root, release = _release(tmp_path)
    with pytest.raises(
        ETTRV3ReleaseVerificationError,
        match="release builder source differs",
    ):
        verify_training_release(
            release_root=release,
            data_root=data_root,
            tokenizer_path=tokenizer,
            release_source_root=_source_root(tmp_path, suffix=b"\n# changed\n"),
            expected_release_source_commit=SOURCE_COMMIT,
            verifier_source_commit=VERIFIER_COMMIT,
            expected_protected_checkpoint_sha256=PROTECTED,
            receipt_path=tmp_path / "receipt.json",
        )
    _make_writable(release)
