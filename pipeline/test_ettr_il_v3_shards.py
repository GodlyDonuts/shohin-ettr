from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat

import pytest

from ettr_il_v3_shards import (
    AssessorOnly,
    AuditRecord,
    CoreIdentity,
    CounterfactualGroups,
    CoverageRecord,
    MANIFEST_FILENAME,
    OracleChannel,
    OracleRecord,
    SemanticCoreRecord,
    SemanticFactors,
    ShardError,
    SourceView,
    SourceVisible,
    TargetRecord,
    canonical_json_bytes,
    load_manifest,
    reload_sharded_dataset,
    semantic_factors_sha256,
    shard_index_for_record,
    validate_sharded_dataset,
    write_sharded_dataset,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _record(index: int) -> SemanticCoreRecord:
    factors = SemanticFactors(
        theory={"family": "local_rewrite", "laws": [index % 6, (index + 2) % 6]},
        worlds=(
            {"registers": [index, 1, 2, 3, 0, 1]},
            {"registers": [index + 1, 1, 2, 3, 0, 1]},
        ),
        commands=(
            {"direction": 0, "site": index % 5},
            {"direction": 1, "site": (index + 1) % 5},
        ),
        queries=(
            {"kind": "slot_equal", "left": 0, "right": 2},
            {"kind": "symbol_count_at_least", "symbol": 1, "threshold": 2},
        ),
    )
    oracle_primary = OracleChannel(
        executions=({"world": 0, "command": 0, "disposition": "answer"},),
        intermediate_snapshots=({"step": 0, "registers": [index, 1, 2]},),
        terminal_observations=({"answer": bool(index % 2)},),
    )
    oracle_replay = OracleChannel(
        executions=({"command": 0, "world": 0, "disposition": "answer"},),
        intermediate_snapshots=({"registers": [index, 1, 2], "step": 0},),
        terminal_observations=({"answer": bool(index % 2)},),
    )
    assessor = AssessorOnly(
        semantic_factors=factors,
        oracle=OracleRecord(primary=oracle_primary, replay=oracle_replay),
        targets=TargetRecord(
            initial_packets=({"slots": [index, 1, 2]},),
            terminal_packets=({"slots": [index + 1, 1, 2]},),
            transaction_traces=([{"opcode": "WRITE", "source": 0, "target": 1}],),
            answer_matrix=([bool(index % 2), not bool(index % 2)],),
        ),
        counterfactual_groups=CounterfactualGroups(
            invariant_orbit_id=f"orbit-{index}",
            world_counterfactual_id=f"world-cf-{index}",
            command_counterfactual_id=f"command-cf-{index}",
            query_counterfactual_id=f"query-cf-{index}",
            hard_negative_ids=(f"negative-{index}-0", f"negative-{index}-1"),
        ),
        coverage=CoverageRecord(
            depth=1 + index % 6,
            trace_length=1,
            opcode_histogram={"WRITE": 1},
            active_slot_bin=2,
            edge_count_bin=0,
            topology_signature=f"path-{index % 3}",
        ),
        audit=AuditRecord(
            raw_hashes=(_digest(f"raw-{index}"),),
            semantic_hash=semantic_factors_sha256(factors),
            graph_iso_hash=_digest(f"graph-{index}"),
            token_hashes=(_digest(f"token-{index}"),),
            materialization_hash=_digest(f"materialization-{index}"),
            replay_hash=_digest(f"replay-{index}"),
        ),
    )
    return SemanticCoreRecord(
        identity=CoreIdentity(
            core_id=f"core-{index:05d}",
            generator_version="r12-ettr-v3-test",
            split="train",
            curriculum_stage="dependent_composition",
            generator_ordinal=index,
        ),
        source_visible=SourceVisible(
            views=(
                SourceView(
                    view_id=f"view-{index}-base",
                    presentation="base",
                    renderer=0,
                    world_sources=(f"World {index}.",),
                    command_sources=("Rewrite site zero.",),
                    query_sources=("Are slots zero and two equal?",),
                ),
                SourceView(
                    view_id=f"view-{index}-alias",
                    presentation="alpha_reorder",
                    renderer=1,
                    world_sources=(f"Register board {index}.",),
                    command_sources=("At position zero, apply the rewrite.",),
                    query_sources=("Do register 0 and register 2 match?",),
                ),
            )
        ),
        assessor_only=assessor,
    )


def test_record_is_strict_canonical_and_round_trips() -> None:
    record = _record(7)
    payload = record.canonical_bytes()
    assert payload == canonical_json_bytes(record.to_value())
    assert payload.endswith(b"\n")
    assert payload.isascii()
    assert SemanticCoreRecord.from_canonical_bytes(payload) == record
    assert record.core_sha256() == hashlib.sha256(payload).hexdigest()

    decoded = json.loads(payload)
    assert set(decoded) == {
        "assessor_only",
        "identity",
        "schema",
        "source_visible",
    }
    assert set(decoded["source_visible"]) == {"views"}
    assert "semantic_factors" not in decoded["source_visible"]
    assert "targets" not in decoded["source_visible"]
    assert "oracle" not in decoded["source_visible"]
    assert set(decoded["assessor_only"]) == {
        "audit",
        "counterfactual_groups",
        "coverage",
        "oracle",
        "semantic_factors",
        "targets",
    }


def test_record_rejects_schema_drift_and_unbound_semantic_hash() -> None:
    record = _record(1)
    value = record.to_value()
    value["unexpected"] = True
    with pytest.raises(ShardError, match="fields differ"):
        SemanticCoreRecord.from_value(value)

    broken_audit = replace(
        record.assessor_only.audit,
        semantic_hash=_digest("wrong semantics"),
    )
    broken = replace(
        record,
        assessor_only=replace(record.assessor_only, audit=broken_audit),
    )
    with pytest.raises(ShardError, match="does not bind"):
        broken.validate()

    with pytest.raises(ShardError, match="floating-point"):
        SemanticCoreRecord.from_canonical_bytes(b'{"bad":1.5}\n')
    with pytest.raises(ShardError, match="strict ASCII"):
        SemanticCoreRecord.from_canonical_bytes('{"bad":"é"}\n'.encode())


def test_shards_are_deterministic_independent_of_input_order(
    tmp_path: Path,
) -> None:
    records = tuple(_record(index) for index in range(17))
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first = write_sharded_dataset(records, first_path, shard_count=5)
    second = write_sharded_dataset(reversed(records), second_path, shard_count=5)
    assert first == second
    assert first.dataset_root_sha256 == second.dataset_root_sha256
    assert first.manifest_sha256() == second.manifest_sha256()
    assert (first_path / MANIFEST_FILENAME).read_bytes() == (
        second_path / MANIFEST_FILENAME
    ).read_bytes()
    for shard in first.shards:
        assert (first_path / shard.filename).read_bytes() == (
            second_path / shard.filename
        ).read_bytes()
        assert stat.S_IMODE((first_path / shard.filename).stat().st_mode) == 0o444
    assert stat.S_IMODE((first_path / MANIFEST_FILENAME).stat().st_mode) == 0o444


def test_reload_verifies_assignment_hashes_counts_and_merkle_root(
    tmp_path: Path,
) -> None:
    records = tuple(_record(index) for index in range(12))
    destination = tmp_path / "dataset"
    written = write_sharded_dataset(records, destination, shard_count=4)
    reloaded, manifest = reload_sharded_dataset(destination)
    assert manifest == written
    assert validate_sharded_dataset(destination) == written
    assert load_manifest(destination) == written
    assert {record.identity.core_id for record in reloaded} == {
        record.identity.core_id for record in records
    }
    assert sum(shard.row_count for shard in manifest.shards) == len(records)
    for record in reloaded:
        assert shard_index_for_record(record, manifest.shard_count) in range(
            manifest.shard_count
        )


def test_writer_rejects_duplicate_core_and_semantic_identity(
    tmp_path: Path,
) -> None:
    first = _record(1)
    duplicate_id = replace(
        _record(2),
        identity=replace(_record(2).identity, core_id=first.identity.core_id),
    )
    with pytest.raises(ShardError, match="duplicate core ID"):
        write_sharded_dataset(
            (first, duplicate_id),
            tmp_path / "duplicate-id",
            shard_count=2,
        )
    assert not list((tmp_path / "duplicate-id").glob("*.jsonl.gz"))

    second = _record(2)
    duplicate_semantics = replace(
        second,
        assessor_only=replace(
            second.assessor_only,
            semantic_factors=first.assessor_only.semantic_factors,
            audit=replace(
                second.assessor_only.audit,
                semantic_hash=first.assessor_only.audit.semantic_hash,
            ),
        ),
    )
    with pytest.raises(ShardError, match="duplicate semantic hash"):
        write_sharded_dataset(
            (first, duplicate_semantics),
            tmp_path / "duplicate-semantic",
            shard_count=2,
        )


def test_publication_is_no_replace_and_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dataset"
    manifest = write_sharded_dataset(
        (_record(1), _record(2)),
        destination,
        shard_count=2,
    )
    with pytest.raises(ShardError, match="already contains"):
        write_sharded_dataset((_record(3),), destination, shard_count=2)

    shard_path = destination / manifest.shards[0].filename
    shard_path.chmod(0o600)
    shard_path.write_bytes(shard_path.read_bytes() + b"tamper")
    with pytest.raises(ShardError, match="identity differs"):
        reload_sharded_dataset(destination)


def test_empty_input_and_non_ascii_or_mutable_numeric_types_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ShardError, match="empty"):
        write_sharded_dataset((), tmp_path / "empty", shard_count=2)
    bad_view = replace(
        _record(0).source_visible.views[0],
        world_sources=("München",),
    )
    bad_record = replace(
        _record(0),
        source_visible=SourceVisible(views=(bad_view,)),
    )
    with pytest.raises(ShardError, match="ASCII"):
        bad_record.validate()
    bad_coverage = replace(_record(0).assessor_only.coverage, depth=True)
    with pytest.raises(ShardError, match="non-negative integer"):
        bad_coverage.validate()
