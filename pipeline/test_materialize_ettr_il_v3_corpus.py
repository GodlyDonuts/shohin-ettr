from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import ettr_il_v2_token_native_surface
from tokenizers import Tokenizer

from ettr_il_v2_token_native_surface import (
    DEFAULT_TOKENIZER_PATH,
    TokenNativeSurfaceCodec,
)
from ettr_il_v3_horn_resource import CurriculumStage
from ettr_il_v3_production import (
    ProductionCell,
    _candidate_row,
    generate_production_cell,
)
from ettr_il_v3_protocol import PROTOCOL, canonical_json_bytes
from materialize_ettr_il_v3_corpus import (
    AUDIT_SCHEMA,
    SEPARATION_SCHEMA,
    audit_main_confirmation_separation,
    build_task_manifest,
    audit_materialization,
    materialize_task,
    prepare_publication,
)
from freeze_ettr_il_v3_protocol import build_freeze
from select_ettr_il_v3 import MANIFEST_SCHEMA as SELECTED_MANIFEST_SCHEMA


def _selected_root(
    tmp_path: Path,
    *,
    name: str = "selected",
    role: str = "main",
    split: str = "train",
) -> Path:
    root = tmp_path / name
    root.mkdir()
    stage = CurriculumStage.ATOMIC_TRANSITIONS
    cell = ProductionCell(
        index=0,
        split=split,
        family="horn",
        stage=stage.value,
        depth=1,
        selected_quota=1,
        candidate_target=1,
        owner_skip=0,
    )
    row = _candidate_row(
        cell,
        generate_production_cell(cell, beam_width=8)[0],
        ordinal=0,
    )
    payload = gzip.compress(canonical_json_bytes(row), compresslevel=6, mtime=0)
    shard_name = f"{split}-horn-{stage.value}.jsonl.gz"
    (root / shard_name).write_bytes(payload)
    manifest: dict[str, object] = {
        "candidate_root_sha256": "c" * 64,
        "codebook_sha256": TokenNativeSurfaceCodec(
            DEFAULT_TOKENIZER_PATH
        ).codebook_sha256,
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": "b" * 64,
        "qualification_admitted_rows": 2,
        "qualification_freeze_sha256": "9" * 64,
        "qualification_input_rows": 3,
        "qualification_rejected_rows": 1,
        "qualification_source_commit": "8" * 40,
        "role": role,
        "schema": SELECTED_MANIFEST_SCHEMA,
        "selector_freeze_sha256": "e" * 64,
        "selector_source_commit": "f" * 40,
        "shards": [
            {
                "bytes": len(payload),
                "family": "horn",
                "path": shard_name,
                "rows": 1,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "split": split,
                "stage": stage.value,
            }
        ],
        "source_commit": "a" * 40,
        "tokenizer_sha256": TokenNativeSurfaceCodec(
            DEFAULT_TOKENIZER_PATH
        ).tokenizer_sha256,
        "total_rows": 1,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return root


def test_task_worker_global_audit_and_publication_inventory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_bytes(DEFAULT_TOKENIZER_PATH.read_bytes())
    monkeypatch.setattr(
        ettr_il_v2_token_native_surface,
        "DEFAULT_TOKENIZER_PATH",
        tmp_path / "developer-default-does-not-exist.json",
    )
    selected = _selected_root(tmp_path)
    task_manifest = tmp_path / "tasks.json"
    source_root = Path(__file__).parent.parent
    freeze_value = build_freeze(
        source_root,
        (
            "pipeline/ettr_il_v3_protocol.py",
            "pipeline/materialize_ettr_il_v3_corpus.py",
        ),
        source_commit="d" * 40,
    )
    tasks = build_task_manifest(
        selected,
        task_manifest,
        materializer_source_commit="d" * 40,
        materializer_freeze_sha256=str(freeze_value["freeze_sha256"]),
    )
    assert tasks["task_count"] == 1

    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze_value))

    dataset = tmp_path / "dataset"
    shards = dataset / "shards"
    reports = tmp_path / "reports"
    worker = materialize_task(
        task_manifest,
        selected,
        shards,
        reports,
        tokenizer_path,
        source_root,
        freeze_path,
        task_index=0,
    )
    assert worker["row_count"] == 1
    assert worker["expanded_rows"] == 64

    audit_path = tmp_path / "audit.json"
    audit = audit_materialization(
        task_manifest,
        shards,
        reports,
        tokenizer_path,
        source_root,
        freeze_path,
        audit_path,
    )
    assert audit["schema"] == AUDIT_SCHEMA
    assert audit["core_rows"] == 1
    assert audit["expanded_rows"] == 64
    assert audit["unique_core_ids"] == 1

    card = tmp_path / "card.md"
    card.write_text("# Frozen test card\n", encoding="ascii")
    publication_path = dataset / "publication_manifest.json"
    publication = prepare_publication(
        audit_path,
        dataset,
        card,
        publication_path,
    )
    assert publication["dataset_protocol"] == PROTOCOL
    assert publication["shards"][0]["split"] == "train"

    confirmation_selected = _selected_root(
        tmp_path,
        name="confirmation-selected",
        role="sealed_confirmation",
        split="confirmation",
    )
    confirmation_tasks = tmp_path / "confirmation-tasks.json"
    build_task_manifest(
        confirmation_selected,
        confirmation_tasks,
        materializer_source_commit="d" * 40,
        materializer_freeze_sha256=str(freeze_value["freeze_sha256"]),
    )
    key = tmp_path / "confirmation.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o400)
    confirmation_shards = tmp_path / "confirmation-shards"
    confirmation_reports = tmp_path / "confirmation-reports"
    materialize_task(
        confirmation_tasks,
        confirmation_selected,
        confirmation_shards,
        confirmation_reports,
        tokenizer_path,
        source_root,
        freeze_path,
        task_index=0,
        confirmation_key_file=key,
    )
    confirmation_audit_path = tmp_path / "confirmation-audit.json"
    audit_materialization(
        confirmation_tasks,
        confirmation_shards,
        confirmation_reports,
        tokenizer_path,
        source_root,
        freeze_path,
        confirmation_audit_path,
    )
    separation_path = tmp_path / "separation.json"
    separation = audit_main_confirmation_separation(
        audit_path,
        confirmation_audit_path,
        shards,
        confirmation_shards,
        separation_path,
    )
    assert separation["schema"] == SEPARATION_SCHEMA
    assert separation["main_core_rows"] == 1
    assert separation["confirmation_core_rows"] == 1
    assert set(separation["overlap_counts"].values()) == {0}


def test_tokenizer_fixture_is_loadable() -> None:
    assert Tokenizer.from_file(str(DEFAULT_TOKENIZER_PATH)).get_vocab_size() > 0
