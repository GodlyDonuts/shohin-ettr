from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

import ettr_il_v2_token_native_surface as token_surface
from build_ettr_il_v3_training_release import (
    ETTRV3ReleaseError,
    RELEASE_SCHEMA,
    build_training_release,
)
from ettr_data_contract import ETTRContinuationManifest
from ettr_il_v2_token_native_surface import (
    DEFAULT_TOKENIZER_PATH,
    TokenNativeSurfaceCodec,
)
from ettr_il_v3_materialize import materialize_candidate
from ettr_il_v3_protocol import PROTOCOL, canonical_json_bytes
from ettr_il_v3_shards import SemanticCoreRecord
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import (
    ETTRV3StreamingError,
    ETTRV3StreamingRelease,
)
from materialize_ettr_il_v3_corpus import AUDIT_SCHEMA, SEPARATION_SCHEMA
from test_ettr_il_v3_materialize import _row


SOURCE_COMMIT = "5" * 40


def _write_record_shard(
    data_root: Path,
    *,
    split: str,
    family: str,
    row: object,
    tokenizer: Tokenizer,
) -> dict[str, object]:
    record = materialize_candidate(row, tokenizer)
    record = replace(
        record,
        identity=replace(record.identity, split=split),
    )
    relative = f"{split}/{family}.jsonl.gz"
    path = data_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(
        record.canonical_bytes(),
        compresslevel=6,
        mtime=0,
    )
    path.write_bytes(payload)
    path.chmod(0o400)
    return {
        "bytes": len(payload),
        "path": relative,
        "report_sha256": hashlib.sha256(
            f"report-{split}".encode("ascii")
        ).hexdigest(),
        "rows": 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "split": split,
    }


def _write_inputs(tmp_path: Path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_bytes(DEFAULT_TOKENIZER_PATH.read_bytes())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    codec = TokenNativeSurfaceCodec(tokenizer_path)
    data_root = tmp_path / "data"
    train = _write_record_shard(
        data_root,
        split="train",
        family="horn",
        row=_row("horn"),
        tokenizer=tokenizer,
    )
    development = _write_record_shard(
        data_root,
        split="development",
        family="resource",
        row=_row("resource"),
        tokenizer=tokenizer,
    )
    audit = {
        "audit_sha256": "",
        "codebook_sha256": codec.codebook_sha256,
        "core_rows": 2,
        "materializer_freeze_sha256": "1" * 64,
        "protocol": PROTOCOL,
        "qualification_freeze_sha256": "2" * 64,
        "role": "main",
        "schema": AUDIT_SCHEMA,
        "shards": [train, development],
        "split_counts": {
            "development": 1,
            "development_reserve": 0,
            "train": 1,
            "train_reserve": 0,
        },
        "status": "pass",
        "tokenizer_sha256": codec.tokenizer_sha256,
    }
    del audit["audit_sha256"]
    audit["audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(audit)
    ).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_path.write_bytes(canonical_json_bytes(audit))

    separation = {
        "confirmation_audit_sha256": "3" * 64,
        "confirmation_core_rows": 1,
        "main_audit_sha256": audit["audit_sha256"],
        "main_core_rows": 2,
        "overlap_counts": {
            "core IDs": 0,
            "graph-isomorphism hashes": 0,
            "semantic hashes": 0,
            "source-view hashes": 0,
        },
        "protocol": PROTOCOL,
        "schema": SEPARATION_SCHEMA,
        "status": "pass",
    }
    separation["separation_sha256"] = hashlib.sha256(
        canonical_json_bytes(separation)
    ).hexdigest()
    separation_path = tmp_path / "separation.json"
    separation_path.write_bytes(canonical_json_bytes(separation))
    return tokenizer_path, data_root, audit_path, separation_path


def _make_writable(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o600)
    root.chmod(0o700)


def _release_inventory(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_release(
    output: Path,
    mutate,
) -> str:
    _make_writable(output)
    path = output / "release.json"
    release = json.loads(path.read_bytes())
    mutate(release)
    unsigned = dict(release)
    unsigned.pop("release_payload_sha256", None)
    release["release_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    payload = canonical_json_bytes(release)
    path.write_bytes(payload)
    path.chmod(0o400)
    output.chmod(0o500)
    return hashlib.sha256(payload).hexdigest()


def test_training_release_reconstructs_and_binds_complete_stream(tmp_path):
    tokenizer, data_root, audit, separation = _write_inputs(tmp_path)
    output = tmp_path / "release"
    result = build_training_release(
        main_audit_path=audit,
        separation_path=separation,
        data_root=data_root,
        tokenizer_path=tokenizer,
        protected_checkpoint_sha256="4" * 64,
        source_commit=SOURCE_COMMIT,
        output=output,
        expected_split_counts={
            "development": 1,
            "development_reserve": 0,
            "train": 1,
            "train_reserve": 0,
        },
    )
    assert result["schema"] == RELEASE_SCHEMA
    assert result["status"] == "pass"
    assert result["source_commit"] == SOURCE_COMMIT
    assert result["release_builder"] == {
        "bytes": Path(
            build_training_release.__code__.co_filename
        ).stat().st_size,
        "path": "pipeline/build_ettr_il_v3_training_release.py",
        "sha256": hashlib.sha256(
            Path(build_training_release.__code__.co_filename).read_bytes()
        ).hexdigest(),
    }
    assert result["stream_index"]["rows"] == 8
    assert result["training_batches_per_core"] == 4
    assert result["training_rows_per_batch"] == 16
    assert result["training_split_core_counts"] == {
        "development": 1,
        "train": 1,
    }

    manifest_payload = (output / "continuation-manifest.json").read_bytes()
    manifest = ETTRContinuationManifest(**json.loads(manifest_payload))
    manifest.validate()
    assert manifest_payload == json.dumps(
        asdict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    assert manifest.sha256() == result["continuation_manifest_sha256"]
    with ETTRDiskPacketSufficiencyIndex(output / "packet-index") as index:
        assert index.train_batches == 4
        assert index.validation_batches == 4
        assert index.receipt.rows == 128
        assert (
            index.receipt.context_sha256
            == manifest.packet_sufficiency_context_sha256
        )
        stream = ETTRV3StreamingRelease(
            output,
            expected_release_sha256=result["release_file_sha256"],
            data_root=data_root,
            tokenizer_path=tokenizer,
        )
        assert stream.verify_source_shards()["core_rows"] == 2
        train_batches = tuple(
            stream.iter_batches(
                "train",
                rank=0,
                world_size=1,
                epoch=0,
                seed=7,
            )
        )
        validation_batches = tuple(
            stream.iter_batches(
                "development",
                rank=0,
                world_size=1,
                epoch=0,
                seed=7,
            )
        )
        assert len(train_batches) == len(validation_batches) == 4
        assert all(
            batch.episodes.world.tokens.shape[0] == 16
            for batch in train_batches + validation_batches
        )
        index.verify_train(train_batches)
        index.verify_validation(validation_batches)

        positioned = tuple(
            stream.iter_positioned_batches(
                "train",
                rank=0,
                world_size=1,
                epoch=0,
                seed=7,
                start_position=1,
            )
        )
        assert len(positioned) == 3
        assert positioned[0][0] == 1
        with pytest.raises(
            ValueError,
            match="cursor exceeds the usable epoch",
        ):
            tuple(
                stream.iter_positioned_batches(
                    "train",
                    rank=0,
                    world_size=1,
                    epoch=0,
                    seed=7,
                    start_position=5,
                )
            )
    _make_writable(output)


def test_release_and_streaming_do_not_depend_on_default_tokenizer_path(
    tmp_path,
    monkeypatch,
):
    tokenizer, data_root, audit, separation = _write_inputs(tmp_path)
    monkeypatch.setattr(
        token_surface,
        "DEFAULT_TOKENIZER_PATH",
        tmp_path / "missing-default-tokenizer.json",
    )
    output = tmp_path / "release"
    result = build_training_release(
        main_audit_path=audit,
        separation_path=separation,
        data_root=data_root,
        tokenizer_path=tokenizer,
        protected_checkpoint_sha256="4" * 64,
        source_commit=SOURCE_COMMIT,
        output=output,
        expected_split_counts={
            "development": 1,
            "development_reserve": 0,
            "train": 1,
            "train_reserve": 0,
        },
    )
    stream = ETTRV3StreamingRelease(
        output,
        expected_release_sha256=result["release_file_sha256"],
        data_root=data_root,
        tokenizer_path=tokenizer,
    )
    assert len(
        tuple(
            stream.iter_batches(
                "train",
                rank=0,
                world_size=1,
                epoch=0,
                seed=7,
            )
        )
    ) == 4


def test_stream_canonicalizes_each_semantic_core_once(
    tmp_path,
    monkeypatch,
):
    tokenizer, data_root, audit, separation = _write_inputs(tmp_path)
    output = tmp_path / "release"
    result = build_training_release(
        main_audit_path=audit,
        separation_path=separation,
        data_root=data_root,
        tokenizer_path=tokenizer,
        protected_checkpoint_sha256="4" * 64,
        source_commit=SOURCE_COMMIT,
        output=output,
        expected_split_counts={
            "development": 1,
            "development_reserve": 0,
            "train": 1,
            "train_reserve": 0,
        },
    )
    stream = ETTRV3StreamingRelease(
        output,
        expected_release_sha256=result["release_file_sha256"],
        data_root=data_root,
        tokenizer_path=tokenizer,
    )
    original = SemanticCoreRecord.canonical_bytes
    calls = 0

    def counted(record):
        nonlocal calls
        calls += 1
        return original(record)

    monkeypatch.setattr(SemanticCoreRecord, "canonical_bytes", counted)
    batches = tuple(
        stream.iter_batches(
            "train",
            rank=0,
            world_size=1,
            epoch=0,
            seed=7,
        )
    )
    assert len(batches) == 4
    assert calls == 1


def test_parallel_release_is_byte_identical_to_serial_release(tmp_path):
    tokenizer, data_root, audit, separation = _write_inputs(tmp_path)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    common = {
        "main_audit_path": audit,
        "separation_path": separation,
        "data_root": data_root,
        "tokenizer_path": tokenizer,
        "protected_checkpoint_sha256": "4" * 64,
        "source_commit": SOURCE_COMMIT,
        "expected_split_counts": {
            "development": 1,
            "development_reserve": 0,
            "train": 1,
            "train_reserve": 0,
        },
    }
    build_training_release(output=serial, workers=1, **common)
    build_training_release(output=parallel, workers=2, **common)
    assert _release_inventory(serial) == _release_inventory(parallel)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda release: release.__setitem__(
                "source_commit",
                "not-a-commit",
            ),
            "release source commit",
        ),
        (
            lambda release: release["release_builder"].__setitem__(
                "path",
                "pipeline/other_builder.py",
            ),
            "release builder receipt",
        ),
    ),
)
def test_streaming_release_rejects_self_rehashed_provenance_tampering(
    tmp_path,
    mutate,
    message,
):
    tokenizer, data_root, audit, separation = _write_inputs(tmp_path)
    output = tmp_path / "release"
    build_training_release(
        main_audit_path=audit,
        separation_path=separation,
        data_root=data_root,
        tokenizer_path=tokenizer,
        protected_checkpoint_sha256="4" * 64,
        source_commit=SOURCE_COMMIT,
        output=output,
        expected_split_counts={
            "development": 1,
            "development_reserve": 0,
            "train": 1,
            "train_reserve": 0,
        },
    )
    release_sha256 = _rewrite_release(output, mutate)
    with pytest.raises(ETTRV3StreamingError, match=message):
        ETTRV3StreamingRelease(
            output,
            expected_release_sha256=release_sha256,
            data_root=data_root,
            tokenizer_path=tokenizer,
        )
    _make_writable(output)


def test_training_release_rejects_writable_materialized_shard(tmp_path):
    tokenizer, data_root, audit, separation = _write_inputs(tmp_path)
    shard = next(data_root.rglob("*.jsonl.gz"))
    shard.chmod(0o600)
    with pytest.raises(
        ETTRV3ReleaseError,
        match="immutable single-link",
    ):
        build_training_release(
            main_audit_path=audit,
            separation_path=separation,
            data_root=data_root,
            tokenizer_path=tokenizer,
            protected_checkpoint_sha256="4" * 64,
            source_commit=SOURCE_COMMIT,
            output=tmp_path / "release",
            expected_split_counts={
                "development": 1,
                "development_reserve": 0,
                "train": 1,
                "train_reserve": 0,
            },
        )


def test_training_release_rejects_malformed_source_commit(tmp_path):
    tokenizer, data_root, audit, separation = _write_inputs(tmp_path)
    with pytest.raises(
        ETTRV3ReleaseError,
        match="release source commit differs",
    ):
        build_training_release(
            main_audit_path=audit,
            separation_path=separation,
            data_root=data_root,
            tokenizer_path=tokenizer,
            protected_checkpoint_sha256="4" * 64,
            source_commit="not-a-commit",
            output=tmp_path / "release",
            expected_split_counts={
                "development": 1,
                "development_reserve": 0,
                "train": 1,
                "train_reserve": 0,
            },
        )
