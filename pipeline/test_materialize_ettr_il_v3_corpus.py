from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

from tokenizers import Tokenizer

from ettr_il_v2_token_native_surface import DEFAULT_TOKENIZER_PATH
from ettr_il_v3_horn_resource import CurriculumStage, generate_horn_episodes
from ettr_il_v3_production import ProductionCell, _candidate_row
from ettr_il_v3_protocol import PROTOCOL, canonical_json_bytes
from materialize_ettr_il_v3_corpus import (
    AUDIT_SCHEMA,
    build_task_manifest,
    audit_materialization,
    materialize_task,
    prepare_publication,
)
from select_ettr_il_v3 import MANIFEST_SCHEMA as SELECTED_MANIFEST_SCHEMA


def _selected_root(tmp_path: Path) -> Path:
    root = tmp_path / "selected"
    root.mkdir()
    stage = CurriculumStage.ATOMIC_TRANSITIONS
    cell = ProductionCell(
        index=0,
        split="train",
        family="horn",
        stage=stage.value,
        depth=1,
        selected_quota=1,
        candidate_target=1,
        owner_skip=0,
    )
    row = _candidate_row(
        cell,
        generate_horn_episodes(stage=stage, theory_index=0, limit=1)[0],
        ordinal=0,
    )
    payload = gzip.compress(canonical_json_bytes(row), compresslevel=6, mtime=0)
    name = f"train-horn-{stage.value}.jsonl.gz"
    (root / name).write_bytes(payload)
    manifest: dict[str, object] = {
        "candidate_root_sha256": "c" * 64,
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": "b" * 64,
        "role": "main",
        "schema": SELECTED_MANIFEST_SCHEMA,
        "shards": [
            {
                "bytes": len(payload),
                "family": "horn",
                "path": name,
                "rows": 1,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "split": "train",
                "stage": stage.value,
            }
        ],
        "source_commit": "a" * 40,
        "total_rows": 1,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return root


def test_task_worker_global_audit_and_publication_inventory(tmp_path: Path) -> None:
    selected = _selected_root(tmp_path)
    task_manifest = tmp_path / "tasks.json"
    tasks = build_task_manifest(
        selected,
        task_manifest,
        materializer_source_commit="d" * 40,
        materializer_freeze_sha256="e" * 64,
    )
    assert tasks["task_count"] == 1

    dataset = tmp_path / "dataset"
    shards = dataset / "shards"
    reports = tmp_path / "reports"
    worker = materialize_task(
        task_manifest,
        selected,
        shards,
        reports,
        DEFAULT_TOKENIZER_PATH,
        task_index=0,
    )
    assert worker["row_count"] == 1
    assert worker["expanded_rows"] == 64

    audit_path = tmp_path / "audit.json"
    audit = audit_materialization(
        task_manifest,
        shards,
        reports,
        DEFAULT_TOKENIZER_PATH,
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


def test_tokenizer_fixture_is_loadable() -> None:
    assert Tokenizer.from_file(str(DEFAULT_TOKENIZER_PATH)).get_vocab_size() > 0
