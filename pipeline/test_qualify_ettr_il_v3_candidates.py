from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import shutil

import pytest

import qualify_ettr_il_v3_candidates as qualifier
import select_ettr_il_v3 as selector
from ettr_il_v2_materialize import MaterializationError
from ettr_il_v2_candidate_search import semantic_world_value
from ettr_il_v2_token_native_surface import DEFAULT_TOKENIZER_PATH
from ettr_il_v3_horn_resource import CurriculumStage, generate_horn_episodes
from ettr_il_v3_production import (
    ROW_SCHEMA,
    ProductionCell,
    base_owner_split,
    write_production_cell,
)
from ettr_il_v3_protocol import PROTOCOL, canonical_json_bytes, orbit_owner
from freeze_ettr_il_v3_protocol import build_freeze


def _cell(split: str = "train") -> ProductionCell:
    return ProductionCell(
        index=0,
        split=split,
        family="horn",
        stage="compiler_grounding",
        depth=1,
        selected_quota=1,
        candidate_target=2,
        owner_skip=0,
    )


def _raw_root(tmp_path: Path, cell: ProductionCell) -> tuple[Path, list[bytes]]:
    root = tmp_path / "raw"
    (root / "reports").mkdir(parents=True)
    (root / "shards").mkdir()
    owner = base_owner_split(cell.split)
    rows: list[bytes] = []
    for ordinal in range(cell.candidate_target):
        episode = {
            "command": {"op": f"op-{ordinal}"},
            "coverage": {"disposition": "apply"},
            "queries": [{"op": "lookup", "slot": ordinal}],
            "world": {"ordinal": ordinal, "owner": owner},
        }
        episode_id = hashlib.sha256(canonical_json_bytes(episode)).hexdigest()
        rows.append(
            canonical_json_bytes(
                {
                    "cell": cell.to_value(),
                    "episode": episode,
                    "episode_id": episode_id,
                    "ordinal": ordinal,
                    "owner": owner,
                    "protocol": PROTOCOL,
                    "schema": ROW_SCHEMA,
                }
            )
        )
    uncompressed = b"".join(rows)
    compressed = gzip.compress(uncompressed, compresslevel=6, mtime=0)
    shard_name = "cell-0.jsonl.gz"
    (root / "shards" / shard_name).write_bytes(compressed)
    report: dict[str, object] = {
        "cell": cell.to_value(),
        "compressed_bytes": len(compressed),
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": "b" * 64,
        "row_count": len(rows),
        "schema": "r12-ettr-il-v3-production-cell-v1",
        "shard_name": shard_name,
        "shard_sha256": hashlib.sha256(compressed).hexdigest(),
        "source_commit": "a" * 40,
        "status": "pass",
        "uncompressed_bytes": len(uncompressed),
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    (root / "reports" / "cell-0.json").write_bytes(canonical_json_bytes(report))
    return root, rows


def _custody(tmp_path: Path) -> tuple[Path, Path, str]:
    repository = Path(__file__).parent.parent
    source_root = tmp_path / "source"
    required = (
        *sorted(qualifier._REQUIRED_PIPELINE_SOURCES),
        *sorted(qualifier._REQUIRED_TRAIN_SOURCES),
    )
    for relative in required:
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    commit = "c" * 40
    freeze = build_freeze(
        source_root,
        required,
        source_commit=commit,
    )
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze))
    return source_root, freeze_path, commit


def test_qualification_rejects_unmeasured_or_incomplete_runtime_source(
    tmp_path: Path,
) -> None:
    source_root, freeze_path, _commit = _custody(tmp_path)
    freeze = qualifier.load_and_verify_freeze(
        source_root,
        freeze_path,
        source_commit="c" * 40,
    )
    (source_root / "pipeline" / "unmeasured.py").write_text(
        "raise RuntimeError\n",
        encoding="ascii",
    )
    with pytest.raises(
        qualifier.QualificationError,
        match="source tree and freeze inventory differ",
    ):
        qualifier._measured_runtime_sources(source_root, freeze)

    (source_root / "pipeline" / "unmeasured.py").unlink()
    incomplete = dict(freeze)
    inventory = [
        entry
        for entry in freeze["source_inventory"]
        if entry["path"] != "train/model.py"
    ]
    incomplete["source_inventory"] = inventory
    (source_root / "train" / "model.py").unlink()
    with pytest.raises(
        qualifier.QualificationError,
        match="train closure differs",
    ):
        qualifier._measured_runtime_sources(source_root, incomplete)


def test_receiver_qualification_preserves_only_admitted_original_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _cell()
    raw, rows = _raw_root(tmp_path, cell)
    monkeypatch.setattr(qualifier, "production_cells", lambda: (cell,))
    monkeypatch.setattr(
        selector,
        "orbit_owner",
        lambda value: value["world"]["owner"],
    )
    calls = 0

    def receiver(value, _codec, *, confirmation_key=None):
        nonlocal calls
        assert confirmation_key is None
        calls += 1
        if value["ordinal"] == 1:
            raise MaterializationError(
                "rectangle 0 corner 01 step 6 has no generic state effect"
            )
        return object()

    monkeypatch.setattr(qualifier, "materialize_candidate", receiver)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(DEFAULT_TOKENIZER_PATH.read_bytes())
    source_root, freeze, commit = _custody(tmp_path)
    output = tmp_path / "qualified"
    report = qualifier.qualify_cell(
        raw,
        output,
        tokenizer,
        source_root,
        freeze,
        matrix_index=0,
        role=qualifier.MAIN_ROLE,
        qualifier_source_commit=commit,
    )

    assert calls == 2
    assert report["admitted_row_count"] == 1
    assert report["rejected_row_count"] == 1
    assert report["rejection_histogram"] == [
        ["MaterializationError:no_generic_state_effect", 1]
    ]
    assert (
        gzip.decompress((output / "shards" / "cell-0.jsonl.gz").read_bytes()) == rows[0]
    )


def test_qualification_key_boundary_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_cell = _cell()
    raw, _rows = _raw_root(tmp_path, main_cell)
    monkeypatch.setattr(qualifier, "production_cells", lambda: (main_cell,))
    monkeypatch.setattr(
        selector,
        "orbit_owner",
        lambda value: value["world"]["owner"],
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(DEFAULT_TOKENIZER_PATH.read_bytes())
    source_root, freeze, commit = _custody(tmp_path)
    key = tmp_path / "confirmation.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o400)

    with pytest.raises(qualifier.QualificationError, match="received a sealed key"):
        qualifier.qualify_cell(
            raw,
            tmp_path / "main-output",
            tokenizer,
            source_root,
            freeze,
            matrix_index=0,
            role=qualifier.MAIN_ROLE,
            qualifier_source_commit=commit,
            confirmation_key_file=key,
        )

    confirmation_cell = _cell("confirmation")
    confirmation_raw, _ = _raw_root(tmp_path / "confirmation-case", confirmation_cell)
    monkeypatch.setattr(
        qualifier,
        "production_cells",
        lambda: (confirmation_cell,),
    )
    with pytest.raises(qualifier.QualificationError, match="requires a sealed key"):
        qualifier.qualify_cell(
            confirmation_raw,
            tmp_path / "confirmation-output",
            tokenizer,
            source_root,
            freeze,
            matrix_index=0,
            role=qualifier.CONFIRMATION_ROLE,
            qualifier_source_commit=commit,
        )


def test_real_receiver_admits_a_generated_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episodes = generate_horn_episodes(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        theory_index=0,
        limit=12,
    )
    episode = next(
        item
        for item in episodes
        if orbit_owner(
            {
                "family": "horn",
                "world": dict(semantic_world_value(item.world)),
            }
        )
        in {"train", "development"}
    )
    split = orbit_owner(
        {
            "family": "horn",
            "world": dict(semantic_world_value(episode.world)),
        }
    )
    cell = ProductionCell(
        index=0,
        split=split,
        family="horn",
        stage=CurriculumStage.ATOMIC_TRANSITIONS.value,
        depth=1,
        selected_quota=1,
        candidate_target=1,
        owner_skip=0,
    )
    raw = tmp_path / "raw"
    write_production_cell(
        cell,
        (episode,),
        source_commit="a" * 40,
        protocol_freeze_sha256="b" * 64,
        shard_path=raw / "shards" / "cell-0.jsonl.gz",
        report_path=raw / "reports" / "cell-0.json",
    )
    monkeypatch.setattr(qualifier, "production_cells", lambda: (cell,))
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(DEFAULT_TOKENIZER_PATH.read_bytes())
    source_root, freeze, commit = _custody(tmp_path)
    report = qualifier.qualify_cell(
        raw,
        tmp_path / "qualified",
        tokenizer,
        source_root,
        freeze,
        matrix_index=0,
        role=qualifier.MAIN_ROLE,
        qualifier_source_commit=commit,
    )
    assert report["admitted_row_count"] == 1
    assert report["rejected_row_count"] == 0
