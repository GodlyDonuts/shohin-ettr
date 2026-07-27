"""Tests for deterministic ETTR-IL-v3 candidate selection."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import select_ettr_il_v3 as selector
from ettr_il_v3_production import (
    ROW_SCHEMA,
    ProductionCell,
    base_owner_split,
)
from ettr_il_v3_protocol import PROTOCOL, canonical_json_bytes
from select_ettr_il_v3 import (
    Candidate,
    SelectionError,
    audit_and_select,
    select_pool,
)

import pytest


def _candidate(
    index: int,
    *,
    factor: str | None = None,
    tokens: tuple[str, ...] = ("common",),
) -> Candidate:
    digest = f"{index:064x}"
    return Candidate(
        episode_id=digest,
        split="train",
        family="horn",
        stage="atomic_transactions",
        depth=1,
        row={"episode_id": digest},
        semantic_factor_sha256=factor or digest,
        coverage_tokens=tokens,
    )


def test_selection_is_exact_deterministic_and_coverage_weighted() -> None:
    candidates = (
        _candidate(1, tokens=("common",)),
        _candidate(2, tokens=("common",)),
        _candidate(3, tokens=("common", "rare")),
        _candidate(4, tokens=("common",)),
    )
    first_forbidden: set[str] = set()
    first = select_pool(
        candidates,
        quota=2,
        group_context={"group": 1},
        forbidden_semantic_factors=first_forbidden,
    )
    second = select_pool(
        candidates,
        quota=2,
        group_context={"group": 1},
        forbidden_semantic_factors=set(),
    )
    assert tuple(item.episode_id for item in first) == tuple(
        item.episode_id for item in second
    )
    assert candidates[2] in first
    assert len(first) == 2
    assert first_forbidden == {
        item.semantic_factor_sha256 for item in first
    }


def test_selection_rejects_duplicate_or_forbidden_semantic_factors() -> None:
    candidates = (
        _candidate(1, factor="a" * 64),
        _candidate(2, factor="a" * 64),
        _candidate(3, factor="b" * 64),
    )
    with pytest.raises(SelectionError, match="exhausted"):
        select_pool(
            candidates,
            quota=2,
            group_context={},
            forbidden_semantic_factors={"b" * 64},
        )


def test_selection_rejects_impossible_quota() -> None:
    with pytest.raises(SelectionError, match="quota"):
        select_pool(
            (_candidate(1),),
            quota=2,
            group_context={},
            forbidden_semantic_factors=set(),
        )


def _write_test_candidates(
    root: Path,
    cells: tuple[ProductionCell, ...],
) -> None:
    (root / "reports").mkdir(parents=True)
    (root / "shards").mkdir()
    for cell in cells:
        rows: list[bytes] = []
        owner = base_owner_split(cell.split)
        for ordinal in range(cell.candidate_target):
            episode = {
                "command": {"op": f"op-{cell.index}-{ordinal}"},
                "coverage": {
                    "disposition": "apply",
                    "query_ops": ["lookup"],
                },
                "queries": [{"op": "lookup", "slot": ordinal}],
                "world": {
                    "cell": cell.index,
                    "ordinal": ordinal,
                    "owner": owner,
                },
            }
            episode_id = hashlib.sha256(
                canonical_json_bytes(episode)
            ).hexdigest()
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
        shard_name = f"cell-{cell.index}.jsonl.gz"
        (root / "shards" / shard_name).write_bytes(compressed)
        report: dict[str, object] = {
            "cell": cell.to_value(),
            "compressed_bytes": len(compressed),
            "protocol": PROTOCOL,
            "protocol_freeze_sha256": "b" * 64,
            "row_count": len(rows),
            "schema": selector.PRODUCTION_SCHEMA,
            "shard_name": shard_name,
            "shard_sha256": hashlib.sha256(compressed).hexdigest(),
            "source_commit": "a" * 40,
            "status": "pass",
            "uncompressed_bytes": len(uncompressed),
        }
        report["report_sha256"] = hashlib.sha256(
            canonical_json_bytes(report)
        ).hexdigest()
        (root / "reports" / f"cell-{cell.index}.json").write_bytes(
            canonical_json_bytes(report)
        )


def _patch_tiny_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProductionCell, ...]:
    splits = (
        "train",
        "development",
        "confirmation",
        "train_reserve",
        "development_reserve",
        "confirmation_reserve",
    )
    cells = tuple(
        ProductionCell(
            index=index,
            split=split,
            family="horn",
            stage="compiler_grounding",
            depth=1,
            selected_quota=1,
            candidate_target=2,
            owner_skip=0,
        )
        for index, split in enumerate(splits)
    )
    monkeypatch.setattr(selector, "production_cells", lambda: cells)
    monkeypatch.setattr(selector, "FAMILIES", ("horn",))
    monkeypatch.setattr(
        selector,
        "CURRICULUM_STAGES",
        ("compiler_grounding",),
    )
    monkeypatch.setattr(selector, "SPLIT_CORES", dict.fromkeys(splits, 1))
    monkeypatch.setattr(
        selector,
        "split_stage_family_allocation",
        lambda _split: {"compiler_grounding": {"horn": 1}},
    )
    monkeypatch.setattr(
        selector,
        "orbit_owner",
        lambda value: value["world"]["owner"],
    )
    return cells


def test_audit_selects_exact_splits_and_separates_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _patch_tiny_protocol(monkeypatch)
    candidates = tmp_path / "candidates"
    _write_test_candidates(candidates, cells)
    main = tmp_path / "main"
    confirmation = tmp_path / "sealed"

    report = audit_and_select(
        candidates,
        main_output=main,
        confirmation_output=confirmation,
    )

    assert report["candidate_rows"] == 12
    assert report["selected_rows"] == 6
    assert len(tuple(main.glob("*.jsonl.gz"))) == 4
    assert len(tuple(confirmation.glob("*.jsonl.gz"))) == 2
    assert (main / "manifest.json").is_file()
    assert (confirmation / "manifest.json").is_file()


def test_audit_writes_nothing_before_all_candidates_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _patch_tiny_protocol(monkeypatch)
    candidates = tmp_path / "candidates"
    _write_test_candidates(candidates, cells)
    (candidates / "shards" / "cell-5.jsonl.gz").write_bytes(b"corrupt")
    main = tmp_path / "main"
    confirmation = tmp_path / "sealed"

    with pytest.raises(SelectionError, match="hash differs"):
        audit_and_select(
            candidates,
            main_output=main,
            confirmation_output=confirmation,
        )

    assert not main.exists()
    assert not confirmation.exists()
