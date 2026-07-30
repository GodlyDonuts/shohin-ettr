from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from audit_ettr_il_v3_packet_context_separation import (
    audit_packet_context_separation,
)
from ettr_il_v2_token_native_surface import (
    DEFAULT_TOKENIZER_PATH,
    TokenNativeSurfaceCodec,
)
from ettr_il_v3_materialize import materialize_candidate
from ettr_il_v3_protocol import PROTOCOL, canonical_json_bytes
from materialize_ettr_il_v3_corpus import AUDIT_SCHEMA
from test_ettr_il_v3_materialize import _row


SOURCE_COMMIT = "7" * 40


def _write_shard(
    root: Path,
    *,
    split: str,
    family: str,
    record: object,
) -> dict[str, object]:
    changed = replace(
        record,
        identity=replace(
            record.identity,
            core_id=hashlib.sha256(
                f"{split}|{record.identity.core_id}".encode()
            ).hexdigest(),
            split=split,
        ),
    )
    relative = f"{split}/{family}.jsonl.gz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(
        changed.canonical_bytes(),
        compresslevel=6,
        mtime=0,
    )
    path.write_bytes(payload)
    path.chmod(0o400)
    return {
        "bytes": len(payload),
        "path": relative,
        "report_sha256": hashlib.sha256(relative.encode()).hexdigest(),
        "rows": 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "split": split,
    }


def _inputs(tmp_path: Path, *, overlap: bool):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_bytes(DEFAULT_TOKENIZER_PATH.read_bytes())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    codec = TokenNativeSurfaceCodec(tokenizer_path)
    data_root = tmp_path / "data"
    train_record = materialize_candidate(_row("horn"), tokenizer)
    development_record = (
        train_record
        if overlap
        else materialize_candidate(_row("resource"), tokenizer)
    )
    shards = [
        _write_shard(
            data_root,
            split="train",
            family="horn",
            record=train_record,
        ),
        _write_shard(
            data_root,
            split="development",
            family=("horn" if overlap else "resource"),
            record=development_record,
        ),
    ]
    audit = {
        "codebook_sha256": codec.codebook_sha256,
        "core_rows": 2,
        "materializer_freeze_sha256": "1" * 64,
        "protocol": PROTOCOL,
        "qualification_freeze_sha256": "2" * 64,
        "role": "main",
        "schema": AUDIT_SCHEMA,
        "shards": shards,
        "split_counts": {
            "development": 1,
            "development_reserve": 0,
            "train": 1,
            "train_reserve": 0,
        },
        "status": "pass",
        "tokenizer_sha256": codec.tokenizer_sha256,
    }
    audit["audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(audit)
    ).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_path.write_bytes(canonical_json_bytes(audit))
    return tokenizer_path, data_root, audit_path


def test_packet_context_audit_passes_independent_families(tmp_path: Path) -> None:
    tokenizer, data_root, audit = _inputs(tmp_path, overlap=False)
    first = audit_packet_context_separation(
        main_audit_path=audit,
        data_root=data_root,
        tokenizer_path=tokenizer,
        source_commit=SOURCE_COMMIT,
        output=tmp_path / "report-one.json",
        workers=1,
    )
    second = audit_packet_context_separation(
        main_audit_path=audit,
        data_root=data_root,
        tokenizer_path=tokenizer,
        source_commit=SOURCE_COMMIT,
        output=tmp_path / "report-two.json",
        workers=2,
    )
    assert first == second
    assert first["status"] == "pass"
    assert first["cross_owner_context_count"] == 0
    assert first["target_conflict_context_count"] == 0


def test_packet_context_audit_finds_exact_cross_split_overlap(
    tmp_path: Path,
) -> None:
    tokenizer, data_root, audit = _inputs(tmp_path, overlap=True)
    report = audit_packet_context_separation(
        main_audit_path=audit,
        data_root=data_root,
        tokenizer_path=tokenizer,
        source_commit=SOURCE_COMMIT,
        output=tmp_path / "report.json",
        workers=1,
    )
    assert report["status"] == "fail"
    assert report["cross_owner_context_count"] == 64
    assert report["cross_owner_component_count"] == 1
    payload = json.loads((tmp_path / "report.json").read_bytes())
    assert payload["report_sha256"] == report["report_sha256"]
