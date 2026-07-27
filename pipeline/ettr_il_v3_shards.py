"""Strict, deterministic ETTR v3 semantic-core JSONL shard storage.

This module is intentionally storage-only.  It accepts already generated and
audited semantic cores, keeps model-visible sources structurally separate from
assessor-only semantics, and writes deterministic gzip-compressed JSONL
shards.  It does not import torch, contact a network, execute semantics, fit
weights, or write outside the caller-provided destination.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Iterable, Mapping, Sequence


RECORD_SCHEMA = "r12-ettr-initializer-v3-semantic-core-v1"
MANIFEST_SCHEMA = "r12-ettr-initializer-v3-shard-manifest-v1"
SHARD_SCHEMA = "r12-ettr-initializer-v3-shard-v1"
DATASET_ROOT_SCHEMA = "r12-ettr-initializer-v3-dataset-root-v1"
MANIFEST_FILENAME = "manifest.json"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHARD_NAME = re.compile(r"^data-[0-9]{5}-of-[0-9]{5}\.jsonl\.gz$")


class ShardError(ValueError):
    """An ETTR v3 record or shard publication violates the frozen contract."""


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_ascii(value: object, name: str, *, allow_empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or (not value and not allow_empty)
        or not value.isascii()
    ):
        raise ShardError(f"{name} must be ASCII text")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if not _plain_int(value) or value < 0:
        raise ShardError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if not _plain_int(value) or value < 1:
        raise ShardError(f"{name} must be a positive integer")
    return value


def _require_hex(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ShardError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_json_value(value: object, name: str = "JSON value") -> None:
    if value is None or isinstance(value, bool):
        return
    if _plain_int(value):
        return
    if isinstance(value, str):
        if not value.isascii():
            raise ShardError(f"{name} contains non-ASCII text")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _strict_json_value(item, f"{name}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key or not key.isascii():
                raise ShardError(f"{name} contains an invalid object key")
            _strict_json_value(item, f"{name}.{key}")
        return
    raise ShardError(f"{name} contains a non-canonical type: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Return strict canonical ASCII JSON with exactly one trailing LF."""

    _strict_json_value(value)
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ShardError("canonical JSON rendering failed") from exc
    return (rendered + "\n").encode("ascii")


def canonical_sha256(value: object) -> str:
    """Hash one canonical JSON value, including its final line feed."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_loads(payload: bytes, name: str) -> object:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise ShardError(f"{name} is not one canonical JSON line")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ShardError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_float=lambda _: (_ for _ in ()).throw(
                ShardError(f"{name} contains a floating-point value")
            ),
            parse_constant=lambda _: (_ for _ in ()).throw(
                ShardError(f"{name} contains a non-finite value")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShardError(f"{name} is not strict ASCII JSON") from exc
    if canonical_json_bytes(value) != payload:
        raise ShardError(f"{name} is not canonical JSON")
    return value


def _exact_object(
    value: object,
    expected: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ShardError(f"{name} fields differ")
    return value


def _json_sequence(
    value: object, name: str, *, nonempty: bool = True
) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ShardError(f"{name} must be a{' non-empty' if nonempty else ''} list")
    result = tuple(value)
    _strict_json_value(result, name)
    return result


def _ascii_sequence(
    value: object,
    name: str,
    *,
    nonempty: bool = True,
) -> tuple[str, ...]:
    values = _json_sequence(value, name, nonempty=nonempty)
    return tuple(_require_ascii(item, f"{name} item") for item in values)


def _hash_sequence(
    value: object,
    name: str,
    *,
    nonempty: bool = True,
) -> tuple[str, ...]:
    values = _json_sequence(value, name, nonempty=nonempty)
    return tuple(_require_hex(item, f"{name} item") for item in values)


def _json_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ShardError(f"{name} must be an object")
    result = dict(value)
    _strict_json_value(result, name)
    return result


@dataclass(frozen=True, slots=True)
class CoreIdentity:
    """Stable generation identity; it contains no model-visible content."""

    core_id: str
    generator_version: str
    split: str
    curriculum_stage: str
    generator_ordinal: int

    def validate(self) -> None:
        _require_ascii(self.core_id, "identity.core_id")
        _require_ascii(self.generator_version, "identity.generator_version")
        _require_ascii(self.split, "identity.split")
        _require_ascii(self.curriculum_stage, "identity.curriculum_stage")
        _require_nonnegative_int(self.generator_ordinal, "identity.generator_ordinal")

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "core_id": self.core_id,
            "curriculum_stage": self.curriculum_stage,
            "generator_ordinal": self.generator_ordinal,
            "generator_version": self.generator_version,
            "split": self.split,
        }

    @classmethod
    def from_value(cls, value: object) -> "CoreIdentity":
        item = _exact_object(
            value,
            {
                "core_id",
                "curriculum_stage",
                "generator_ordinal",
                "generator_version",
                "split",
            },
            "identity",
        )
        result = cls(
            core_id=_require_ascii(item["core_id"], "identity.core_id"),
            generator_version=_require_ascii(
                item["generator_version"], "identity.generator_version"
            ),
            split=_require_ascii(item["split"], "identity.split"),
            curriculum_stage=_require_ascii(
                item["curriculum_stage"], "identity.curriculum_stage"
            ),
            generator_ordinal=_require_nonnegative_int(
                item["generator_ordinal"], "identity.generator_ordinal"
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class SemanticFactors:
    """Assessor-only latent theory, worlds, commands, and queries."""

    theory: object
    worlds: tuple[object, ...]
    commands: tuple[object, ...]
    queries: tuple[object, ...]

    def validate(self) -> None:
        _strict_json_value(self.theory, "semantic_factors.theory")
        _json_sequence(self.worlds, "semantic_factors.worlds")
        _json_sequence(self.commands, "semantic_factors.commands")
        _json_sequence(self.queries, "semantic_factors.queries")

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "commands": list(self.commands),
            "queries": list(self.queries),
            "theory": self.theory,
            "worlds": list(self.worlds),
        }

    @classmethod
    def from_value(cls, value: object) -> "SemanticFactors":
        item = _exact_object(
            value,
            {"commands", "queries", "theory", "worlds"},
            "semantic_factors",
        )
        result = cls(
            theory=item["theory"],
            worlds=_json_sequence(item["worlds"], "semantic_factors.worlds"),
            commands=_json_sequence(item["commands"], "semantic_factors.commands"),
            queries=_json_sequence(item["queries"], "semantic_factors.queries"),
        )
        result.validate()
        return result

    def sha256(self) -> str:
        return canonical_sha256(self.to_value())


@dataclass(frozen=True, slots=True)
class OracleChannel:
    """One independently implemented semantic execution channel."""

    executions: tuple[object, ...]
    intermediate_snapshots: tuple[object, ...]
    terminal_observations: tuple[object, ...]

    def validate(self) -> None:
        _json_sequence(self.executions, "oracle.executions")
        _json_sequence(
            self.intermediate_snapshots,
            "oracle.intermediate_snapshots",
            nonempty=False,
        )
        _json_sequence(
            self.terminal_observations,
            "oracle.terminal_observations",
        )

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "executions": list(self.executions),
            "intermediate_snapshots": list(self.intermediate_snapshots),
            "terminal_observations": list(self.terminal_observations),
        }

    @classmethod
    def from_value(cls, value: object, name: str) -> "OracleChannel":
        item = _exact_object(
            value,
            {
                "executions",
                "intermediate_snapshots",
                "terminal_observations",
            },
            name,
        )
        result = cls(
            executions=_json_sequence(item["executions"], f"{name}.executions"),
            intermediate_snapshots=_json_sequence(
                item["intermediate_snapshots"],
                f"{name}.intermediate_snapshots",
                nonempty=False,
            ),
            terminal_observations=_json_sequence(
                item["terminal_observations"],
                f"{name}.terminal_observations",
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class OracleRecord:
    """Primary and replay executions retained strictly assessor-side."""

    primary: OracleChannel
    replay: OracleChannel

    def validate(self) -> None:
        if not isinstance(self.primary, OracleChannel) or not isinstance(
            self.replay, OracleChannel
        ):
            raise ShardError("oracle channels have invalid types")
        self.primary.validate()
        self.replay.validate()

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "primary": self.primary.to_value(),
            "replay": self.replay.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> "OracleRecord":
        item = _exact_object(value, {"primary", "replay"}, "oracle")
        result = cls(
            primary=OracleChannel.from_value(item["primary"], "oracle.primary"),
            replay=OracleChannel.from_value(item["replay"], "oracle.replay"),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class SourceView:
    """Only these rendered sources may be exposed to the model."""

    view_id: str
    presentation: str
    renderer: int
    world_sources: tuple[str, ...]
    command_sources: tuple[str, ...]
    query_sources: tuple[str, ...]

    def validate(self) -> None:
        _require_ascii(self.view_id, "view.view_id")
        _require_ascii(self.presentation, "view.presentation")
        _require_nonnegative_int(self.renderer, "view.renderer")
        _ascii_sequence(self.world_sources, "view.world_sources")
        _ascii_sequence(self.command_sources, "view.command_sources")
        _ascii_sequence(self.query_sources, "view.query_sources")

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "command_sources": list(self.command_sources),
            "presentation": self.presentation,
            "query_sources": list(self.query_sources),
            "renderer": self.renderer,
            "view_id": self.view_id,
            "world_sources": list(self.world_sources),
        }

    @classmethod
    def from_value(cls, value: object) -> "SourceView":
        item = _exact_object(
            value,
            {
                "command_sources",
                "presentation",
                "query_sources",
                "renderer",
                "view_id",
                "world_sources",
            },
            "view",
        )
        result = cls(
            view_id=_require_ascii(item["view_id"], "view.view_id"),
            presentation=_require_ascii(item["presentation"], "view.presentation"),
            renderer=_require_nonnegative_int(item["renderer"], "view.renderer"),
            world_sources=_ascii_sequence(item["world_sources"], "view.world_sources"),
            command_sources=_ascii_sequence(
                item["command_sources"], "view.command_sources"
            ),
            query_sources=_ascii_sequence(item["query_sources"], "view.query_sources"),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class SourceVisible:
    """The complete and exclusive model-visible section of one core."""

    views: tuple[SourceView, ...]

    def validate(self) -> None:
        if (
            not isinstance(self.views, tuple)
            or not self.views
            or any(not isinstance(view, SourceView) for view in self.views)
        ):
            raise ShardError("source_visible.views must contain SourceView records")
        for view in self.views:
            view.validate()
        identifiers = [view.view_id for view in self.views]
        if len(set(identifiers)) != len(identifiers):
            raise ShardError("source_visible contains duplicate view IDs")

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {"views": [view.to_value() for view in self.views]}

    @classmethod
    def from_value(cls, value: object) -> "SourceVisible":
        item = _exact_object(value, {"views"}, "source_visible")
        raw_views = _json_sequence(item["views"], "source_visible.views")
        result = cls(views=tuple(SourceView.from_value(view) for view in raw_views))
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class TargetRecord:
    """All exact packets, traces, and answers; never model-visible inputs."""

    initial_packets: tuple[object, ...]
    terminal_packets: tuple[object, ...]
    transaction_traces: tuple[object, ...]
    answer_matrix: tuple[object, ...]

    def validate(self) -> None:
        _json_sequence(self.initial_packets, "targets.initial_packets")
        _json_sequence(self.terminal_packets, "targets.terminal_packets")
        _json_sequence(self.transaction_traces, "targets.transaction_traces")
        _json_sequence(self.answer_matrix, "targets.answer_matrix")

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "answer_matrix": list(self.answer_matrix),
            "initial_packets": list(self.initial_packets),
            "terminal_packets": list(self.terminal_packets),
            "transaction_traces": list(self.transaction_traces),
        }

    @classmethod
    def from_value(cls, value: object) -> "TargetRecord":
        item = _exact_object(
            value,
            {
                "answer_matrix",
                "initial_packets",
                "terminal_packets",
                "transaction_traces",
            },
            "targets",
        )
        result = cls(
            initial_packets=_json_sequence(
                item["initial_packets"], "targets.initial_packets"
            ),
            terminal_packets=_json_sequence(
                item["terminal_packets"], "targets.terminal_packets"
            ),
            transaction_traces=_json_sequence(
                item["transaction_traces"], "targets.transaction_traces"
            ),
            answer_matrix=_json_sequence(
                item["answer_matrix"], "targets.answer_matrix"
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class CounterfactualGroups:
    """Assessor-side orbit and intervention identities."""

    invariant_orbit_id: str
    world_counterfactual_id: str
    command_counterfactual_id: str
    query_counterfactual_id: str
    hard_negative_ids: tuple[str, ...]

    def validate(self) -> None:
        _require_ascii(self.invariant_orbit_id, "groups.invariant_orbit_id")
        _require_ascii(self.world_counterfactual_id, "groups.world_counterfactual_id")
        _require_ascii(
            self.command_counterfactual_id, "groups.command_counterfactual_id"
        )
        _require_ascii(self.query_counterfactual_id, "groups.query_counterfactual_id")
        _ascii_sequence(
            self.hard_negative_ids,
            "groups.hard_negative_ids",
            nonempty=False,
        )
        if len(set(self.hard_negative_ids)) != len(self.hard_negative_ids):
            raise ShardError("groups contains duplicate hard-negative IDs")

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "command_counterfactual_id": self.command_counterfactual_id,
            "hard_negative_ids": list(self.hard_negative_ids),
            "invariant_orbit_id": self.invariant_orbit_id,
            "query_counterfactual_id": self.query_counterfactual_id,
            "world_counterfactual_id": self.world_counterfactual_id,
        }

    @classmethod
    def from_value(cls, value: object) -> "CounterfactualGroups":
        item = _exact_object(
            value,
            {
                "command_counterfactual_id",
                "hard_negative_ids",
                "invariant_orbit_id",
                "query_counterfactual_id",
                "world_counterfactual_id",
            },
            "groups",
        )
        result = cls(
            invariant_orbit_id=_require_ascii(
                item["invariant_orbit_id"], "groups.invariant_orbit_id"
            ),
            world_counterfactual_id=_require_ascii(
                item["world_counterfactual_id"],
                "groups.world_counterfactual_id",
            ),
            command_counterfactual_id=_require_ascii(
                item["command_counterfactual_id"],
                "groups.command_counterfactual_id",
            ),
            query_counterfactual_id=_require_ascii(
                item["query_counterfactual_id"],
                "groups.query_counterfactual_id",
            ),
            hard_negative_ids=_ascii_sequence(
                item["hard_negative_ids"],
                "groups.hard_negative_ids",
                nonempty=False,
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    """Measured selection dimensions for coverage optimization and audits."""

    depth: int
    trace_length: int
    opcode_histogram: Mapping[str, int]
    active_slot_bin: int
    edge_count_bin: int
    topology_signature: str

    def validate(self) -> None:
        _require_nonnegative_int(self.depth, "coverage.depth")
        _require_nonnegative_int(self.trace_length, "coverage.trace_length")
        _require_nonnegative_int(self.active_slot_bin, "coverage.active_slot_bin")
        _require_nonnegative_int(self.edge_count_bin, "coverage.edge_count_bin")
        _require_ascii(self.topology_signature, "coverage.topology_signature")
        if not isinstance(self.opcode_histogram, Mapping):
            raise ShardError("coverage.opcode_histogram must be an object")
        for opcode, count in self.opcode_histogram.items():
            _require_ascii(opcode, "coverage opcode")
            _require_nonnegative_int(count, f"coverage opcode {opcode}")

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "active_slot_bin": self.active_slot_bin,
            "depth": self.depth,
            "edge_count_bin": self.edge_count_bin,
            "opcode_histogram": dict(self.opcode_histogram),
            "topology_signature": self.topology_signature,
            "trace_length": self.trace_length,
        }

    @classmethod
    def from_value(cls, value: object) -> "CoverageRecord":
        item = _exact_object(
            value,
            {
                "active_slot_bin",
                "depth",
                "edge_count_bin",
                "opcode_histogram",
                "topology_signature",
                "trace_length",
            },
            "coverage",
        )
        histogram = _json_mapping(item["opcode_histogram"], "coverage.opcode_histogram")
        result = cls(
            depth=_require_nonnegative_int(item["depth"], "coverage.depth"),
            trace_length=_require_nonnegative_int(
                item["trace_length"], "coverage.trace_length"
            ),
            opcode_histogram={
                _require_ascii(key, "coverage opcode"): _require_nonnegative_int(
                    count, f"coverage opcode {key}"
                )
                for key, count in histogram.items()
            },
            active_slot_bin=_require_nonnegative_int(
                item["active_slot_bin"], "coverage.active_slot_bin"
            ),
            edge_count_bin=_require_nonnegative_int(
                item["edge_count_bin"], "coverage.edge_count_bin"
            ),
            topology_signature=_require_ascii(
                item["topology_signature"], "coverage.topology_signature"
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Stable hashes binding raw, semantic, materialized, and replay forms."""

    raw_hashes: tuple[str, ...]
    semantic_hash: str
    graph_iso_hash: str
    token_hashes: tuple[str, ...]
    materialization_hash: str
    replay_hash: str

    def validate(self) -> None:
        _hash_sequence(self.raw_hashes, "audit.raw_hashes")
        _require_hex(self.semantic_hash, "audit.semantic_hash")
        _require_hex(self.graph_iso_hash, "audit.graph_iso_hash")
        _hash_sequence(self.token_hashes, "audit.token_hashes")
        _require_hex(self.materialization_hash, "audit.materialization_hash")
        _require_hex(self.replay_hash, "audit.replay_hash")

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "graph_iso_hash": self.graph_iso_hash,
            "materialization_hash": self.materialization_hash,
            "raw_hashes": list(self.raw_hashes),
            "replay_hash": self.replay_hash,
            "semantic_hash": self.semantic_hash,
            "token_hashes": list(self.token_hashes),
        }

    @classmethod
    def from_value(cls, value: object) -> "AuditRecord":
        item = _exact_object(
            value,
            {
                "graph_iso_hash",
                "materialization_hash",
                "raw_hashes",
                "replay_hash",
                "semantic_hash",
                "token_hashes",
            },
            "audit",
        )
        result = cls(
            raw_hashes=_hash_sequence(item["raw_hashes"], "audit.raw_hashes"),
            semantic_hash=_require_hex(item["semantic_hash"], "audit.semantic_hash"),
            graph_iso_hash=_require_hex(item["graph_iso_hash"], "audit.graph_iso_hash"),
            token_hashes=_hash_sequence(item["token_hashes"], "audit.token_hashes"),
            materialization_hash=_require_hex(
                item["materialization_hash"], "audit.materialization_hash"
            ),
            replay_hash=_require_hex(item["replay_hash"], "audit.replay_hash"),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class AssessorOnly:
    """Everything prohibited from model-visible source tensors."""

    semantic_factors: SemanticFactors
    oracle: OracleRecord
    targets: TargetRecord
    counterfactual_groups: CounterfactualGroups
    coverage: CoverageRecord
    audit: AuditRecord

    def validate(self) -> None:
        expected = (
            (self.semantic_factors, SemanticFactors, "semantic_factors"),
            (self.oracle, OracleRecord, "oracle"),
            (self.targets, TargetRecord, "targets"),
            (
                self.counterfactual_groups,
                CounterfactualGroups,
                "counterfactual_groups",
            ),
            (self.coverage, CoverageRecord, "coverage"),
            (self.audit, AuditRecord, "audit"),
        )
        for value, expected_type, name in expected:
            if not isinstance(value, expected_type):
                raise ShardError(f"assessor_only.{name} has an invalid type")
            value.validate()
        if self.audit.semantic_hash != self.semantic_factors.sha256():
            raise ShardError("audit.semantic_hash does not bind semantic_factors")

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "audit": self.audit.to_value(),
            "counterfactual_groups": self.counterfactual_groups.to_value(),
            "coverage": self.coverage.to_value(),
            "oracle": self.oracle.to_value(),
            "semantic_factors": self.semantic_factors.to_value(),
            "targets": self.targets.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> "AssessorOnly":
        item = _exact_object(
            value,
            {
                "audit",
                "counterfactual_groups",
                "coverage",
                "oracle",
                "semantic_factors",
                "targets",
            },
            "assessor_only",
        )
        result = cls(
            semantic_factors=SemanticFactors.from_value(item["semantic_factors"]),
            oracle=OracleRecord.from_value(item["oracle"]),
            targets=TargetRecord.from_value(item["targets"]),
            counterfactual_groups=CounterfactualGroups.from_value(
                item["counterfactual_groups"]
            ),
            coverage=CoverageRecord.from_value(item["coverage"]),
            audit=AuditRecord.from_value(item["audit"]),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class SemanticCoreRecord:
    """One strict ETTR v3 core with an explicit information boundary."""

    identity: CoreIdentity
    source_visible: SourceVisible
    assessor_only: AssessorOnly
    schema: str = RECORD_SCHEMA

    def validate(self) -> None:
        if self.schema != RECORD_SCHEMA:
            raise ShardError("semantic-core schema differs")
        if not isinstance(self.identity, CoreIdentity):
            raise ShardError("identity has an invalid type")
        if not isinstance(self.source_visible, SourceVisible):
            raise ShardError("source_visible has an invalid type")
        if not isinstance(self.assessor_only, AssessorOnly):
            raise ShardError("assessor_only has an invalid type")
        self.identity.validate()
        self.source_visible.validate()
        self.assessor_only.validate()

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "assessor_only": self.assessor_only.to_value(),
            "identity": self.identity.to_value(),
            "schema": self.schema,
            "source_visible": self.source_visible.to_value(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_value())

    def core_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_value(cls, value: object) -> "SemanticCoreRecord":
        item = _exact_object(
            value,
            {"assessor_only", "identity", "schema", "source_visible"},
            "semantic core",
        )
        if item["schema"] != RECORD_SCHEMA:
            raise ShardError("semantic-core schema differs")
        result = cls(
            identity=CoreIdentity.from_value(item["identity"]),
            source_visible=SourceVisible.from_value(item["source_visible"]),
            assessor_only=AssessorOnly.from_value(item["assessor_only"]),
            schema=RECORD_SCHEMA,
        )
        result.validate()
        return result

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "SemanticCoreRecord":
        return cls.from_value(_strict_loads(payload, "semantic-core row"))


def semantic_factors_sha256(factors: SemanticFactors) -> str:
    """Return the canonical semantic identity required by ``AuditRecord``."""

    if not isinstance(factors, SemanticFactors):
        raise ShardError("semantic factors have an invalid type")
    return factors.sha256()


def shard_index_for_hash(core_sha256: str, shard_count: int) -> int:
    """Assign one full core hash to a shard without platform-dependent state."""

    digest = _require_hex(core_sha256, "core_sha256")
    count = _require_positive_int(shard_count, "shard_count")
    return int(digest, 16) % count


def shard_index_for_record(record: SemanticCoreRecord, shard_count: int) -> int:
    if not isinstance(record, SemanticCoreRecord):
        raise ShardError("record has an invalid type")
    return shard_index_for_hash(record.core_sha256(), shard_count)


@dataclass(frozen=True, slots=True)
class ShardDescriptor:
    index: int
    filename: str
    row_count: int
    byte_count: int
    uncompressed_byte_count: int
    sha256: str

    def validate(self, shard_count: int) -> None:
        if not _plain_int(self.index) or not 0 <= self.index < shard_count:
            raise ShardError("shard index differs")
        expected = _shard_filename(self.index, shard_count)
        if self.filename != expected or _SHARD_NAME.fullmatch(self.filename) is None:
            raise ShardError("shard filename differs")
        _require_nonnegative_int(self.row_count, "shard row_count")
        _require_nonnegative_int(self.byte_count, "shard byte_count")
        _require_nonnegative_int(
            self.uncompressed_byte_count, "shard uncompressed_byte_count"
        )
        _require_hex(self.sha256, "shard sha256")

    def to_value(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "filename": self.filename,
            "index": self.index,
            "row_count": self.row_count,
            "sha256": self.sha256,
            "uncompressed_byte_count": self.uncompressed_byte_count,
        }

    @classmethod
    def from_value(
        cls,
        value: object,
        shard_count: int,
    ) -> "ShardDescriptor":
        item = _exact_object(
            value,
            {
                "byte_count",
                "filename",
                "index",
                "row_count",
                "sha256",
                "uncompressed_byte_count",
            },
            "shard descriptor",
        )
        result = cls(
            index=_require_nonnegative_int(item["index"], "shard index"),
            filename=_require_ascii(item["filename"], "shard filename"),
            row_count=_require_nonnegative_int(item["row_count"], "shard row_count"),
            byte_count=_require_nonnegative_int(item["byte_count"], "shard byte_count"),
            uncompressed_byte_count=_require_nonnegative_int(
                item["uncompressed_byte_count"],
                "shard uncompressed_byte_count",
            ),
            sha256=_require_hex(item["sha256"], "shard sha256"),
        )
        result.validate(shard_count)
        return result


def _merkle_shard_root(shards: Sequence[ShardDescriptor]) -> str:
    if not shards:
        raise ShardError("cannot derive a root for zero shards")
    nodes = [
        hashlib.sha256(
            b"r12-ettr-v3-shard-leaf\0" + canonical_json_bytes(shard.to_value())
        ).digest()
        for shard in shards
    ]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(
                b"r12-ettr-v3-shard-node\0" + nodes[index] + nodes[index + 1]
            ).digest()
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def dataset_root_sha256(
    shards: Sequence[ShardDescriptor],
    *,
    row_count: int,
    byte_count: int,
    uncompressed_byte_count: int,
) -> str:
    """Bind ordered shard leaves and dataset totals into one stable root."""

    values = tuple(shards)
    shard_count = len(values)
    if shard_count < 1:
        raise ShardError("dataset must contain at least one shard")
    for index, shard in enumerate(values):
        shard.validate(shard_count)
        if shard.index != index:
            raise ShardError("shards are not in canonical index order")
    payload = {
        "byte_count": _require_nonnegative_int(byte_count, "dataset byte_count"),
        "merkle_shard_root": _merkle_shard_root(values),
        "record_schema": RECORD_SCHEMA,
        "row_count": _require_nonnegative_int(row_count, "dataset row_count"),
        "schema": DATASET_ROOT_SCHEMA,
        "shard_count": shard_count,
        "uncompressed_byte_count": _require_nonnegative_int(
            uncompressed_byte_count, "dataset uncompressed_byte_count"
        ),
    }
    return hashlib.sha256(
        b"r12-ettr-v3-dataset-root\0" + canonical_json_bytes(payload)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    shard_count: int
    row_count: int
    byte_count: int
    uncompressed_byte_count: int
    shards: tuple[ShardDescriptor, ...]
    dataset_root_sha256: str
    schema: str = MANIFEST_SCHEMA
    record_schema: str = RECORD_SCHEMA

    def validate(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ShardError("manifest schema differs")
        if self.record_schema != RECORD_SCHEMA:
            raise ShardError("manifest record schema differs")
        _require_positive_int(self.shard_count, "manifest shard_count")
        if not isinstance(self.shards, tuple) or len(self.shards) != self.shard_count:
            raise ShardError("manifest shard inventory differs")
        for index, shard in enumerate(self.shards):
            if not isinstance(shard, ShardDescriptor):
                raise ShardError("manifest shard type differs")
            shard.validate(self.shard_count)
            if shard.index != index:
                raise ShardError("manifest shards are not ordered")
        expected_totals = (
            sum(shard.row_count for shard in self.shards),
            sum(shard.byte_count for shard in self.shards),
            sum(shard.uncompressed_byte_count for shard in self.shards),
        )
        observed_totals = (
            _require_nonnegative_int(self.row_count, "manifest row_count"),
            _require_nonnegative_int(self.byte_count, "manifest byte_count"),
            _require_nonnegative_int(
                self.uncompressed_byte_count,
                "manifest uncompressed_byte_count",
            ),
        )
        if expected_totals != observed_totals:
            raise ShardError("manifest totals differ from shard totals")
        _require_hex(self.dataset_root_sha256, "manifest dataset_root_sha256")
        expected_root = dataset_root_sha256(
            self.shards,
            row_count=self.row_count,
            byte_count=self.byte_count,
            uncompressed_byte_count=self.uncompressed_byte_count,
        )
        if self.dataset_root_sha256 != expected_root:
            raise ShardError("manifest dataset root differs")

    def to_value(self) -> dict[str, object]:
        self.validate()
        return {
            "byte_count": self.byte_count,
            "dataset_root_sha256": self.dataset_root_sha256,
            "record_schema": self.record_schema,
            "row_count": self.row_count,
            "schema": self.schema,
            "shard_count": self.shard_count,
            "shards": [shard.to_value() for shard in self.shards],
            "uncompressed_byte_count": self.uncompressed_byte_count,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_value())

    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_value(cls, value: object) -> "DatasetManifest":
        item = _exact_object(
            value,
            {
                "byte_count",
                "dataset_root_sha256",
                "record_schema",
                "row_count",
                "schema",
                "shard_count",
                "shards",
                "uncompressed_byte_count",
            },
            "manifest",
        )
        if item["schema"] != MANIFEST_SCHEMA:
            raise ShardError("manifest schema differs")
        if item["record_schema"] != RECORD_SCHEMA:
            raise ShardError("manifest record schema differs")
        shard_count = _require_positive_int(item["shard_count"], "manifest shard_count")
        raw_shards = _json_sequence(item["shards"], "manifest shards")
        result = cls(
            shard_count=shard_count,
            row_count=_require_nonnegative_int(item["row_count"], "manifest row_count"),
            byte_count=_require_nonnegative_int(
                item["byte_count"], "manifest byte_count"
            ),
            uncompressed_byte_count=_require_nonnegative_int(
                item["uncompressed_byte_count"],
                "manifest uncompressed_byte_count",
            ),
            shards=tuple(
                ShardDescriptor.from_value(shard, shard_count) for shard in raw_shards
            ),
            dataset_root_sha256=_require_hex(
                item["dataset_root_sha256"],
                "manifest dataset_root_sha256",
            ),
        )
        result.validate()
        return result

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "DatasetManifest":
        return cls.from_value(_strict_loads(payload, "manifest"))


def _shard_filename(index: int, shard_count: int) -> str:
    if shard_count > 99_999:
        raise ShardError("shard_count exceeds the fixed filename width")
    return f"data-{index:05d}-of-{shard_count:05d}.jsonl.gz"


def _regular_file_bytes(path: Path, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ShardError(f"{name} cannot be inspected") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or path.is_symlink():
        raise ShardError(f"{name} is not a single-link regular file")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ShardError(f"{name} cannot be read") from exc
    after = path.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != before.st_size
    ):
        raise ShardError(f"{name} changed during measurement")
    return payload


def _exclusive_open(path: Path):
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o444)
    except OSError as exc:
        raise ShardError(f"refusing to replace {path.name}") from exc
    os.fchmod(descriptor, 0o444)
    return os.fdopen(descriptor, "wb")


def _prepare_destination(destination: str | Path, shard_count: int) -> Path:
    path = Path(destination)
    if path.exists() and (not path.is_dir() or path.is_symlink()):
        raise ShardError("destination must be a real directory")
    path.mkdir(parents=True, exist_ok=True)
    targets = [path / MANIFEST_FILENAME] + [
        path / _shard_filename(index, shard_count) for index in range(shard_count)
    ]
    if any(target.exists() or target.is_symlink() for target in targets):
        raise ShardError("destination already contains a dataset target")
    return path


def write_sharded_dataset(
    records: Iterable[SemanticCoreRecord],
    destination: str | Path,
    *,
    shard_count: int,
    compresslevel: int = 9,
) -> DatasetManifest:
    """Write one deterministic no-replace shard set and canonical manifest.

    Temporary bucket files are created only below ``destination`` and removed
    before return.  Input order does not affect shard bytes or dataset root.
    """

    count = _require_positive_int(shard_count, "shard_count")
    if count > 99_999:
        raise ShardError("shard_count exceeds the fixed filename width")
    if not _plain_int(compresslevel) or not 0 <= compresslevel <= 9:
        raise ShardError("compresslevel must be an integer from 0 through 9")
    output = _prepare_destination(destination, count)
    stage = Path(tempfile.mkdtemp(prefix=".ettr-v3-stage-", dir=output))
    handles: dict[int, object] = {}
    core_ids: set[str] = set()
    semantic_hashes: set[str] = set()
    created: list[Path] = []
    try:
        for record in records:
            if not isinstance(record, SemanticCoreRecord):
                raise ShardError("records must contain SemanticCoreRecord values")
            record.validate()
            core_id = record.identity.core_id
            semantic_hash = record.assessor_only.audit.semantic_hash
            if core_id in core_ids:
                raise ShardError(f"duplicate core ID: {core_id}")
            if semantic_hash in semantic_hashes:
                raise ShardError(f"duplicate semantic hash: {semantic_hash}")
            core_ids.add(core_id)
            semantic_hashes.add(semantic_hash)
            row = record.canonical_bytes()
            core_hash = hashlib.sha256(row).hexdigest()
            index = shard_index_for_hash(core_hash, count)
            handle = handles.get(index)
            if handle is None:
                handle = (stage / f"{index:05d}.bucket").open("ab")
                handles[index] = handle
            handle.write(core_hash.encode("ascii") + b"\t" + row)
        if not core_ids:
            raise ShardError("cannot publish an empty semantic-core dataset")
        for handle in handles.values():
            handle.close()
        handles.clear()

        descriptors: list[ShardDescriptor] = []
        for index in range(count):
            bucket = stage / f"{index:05d}.bucket"
            staged_rows = (
                []
                if not bucket.exists()
                else bucket.read_bytes().splitlines(keepends=True)
            )
            staged_rows.sort()
            filename = _shard_filename(index, count)
            target = output / filename
            uncompressed_count = 0
            with _exclusive_open(target) as raw:
                created.append(target)
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=compresslevel,
                    fileobj=raw,
                    mtime=0,
                ) as compressed:
                    for staged_row in staged_rows:
                        separator = staged_row.find(b"\t")
                        if separator != 64:
                            raise ShardError("staging row framing differs")
                        row = staged_row[separator + 1 :]
                        compressed.write(row)
                        uncompressed_count += len(row)
                raw.flush()
                os.fsync(raw.fileno())
            payload = _regular_file_bytes(target, f"shard {filename}")
            descriptors.append(
                ShardDescriptor(
                    index=index,
                    filename=filename,
                    row_count=len(staged_rows),
                    byte_count=len(payload),
                    uncompressed_byte_count=uncompressed_count,
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            )

        shards = tuple(descriptors)
        row_count = sum(shard.row_count for shard in shards)
        byte_count = sum(shard.byte_count for shard in shards)
        uncompressed_byte_count = sum(shard.uncompressed_byte_count for shard in shards)
        manifest = DatasetManifest(
            shard_count=count,
            row_count=row_count,
            byte_count=byte_count,
            uncompressed_byte_count=uncompressed_byte_count,
            shards=shards,
            dataset_root_sha256=dataset_root_sha256(
                shards,
                row_count=row_count,
                byte_count=byte_count,
                uncompressed_byte_count=uncompressed_byte_count,
            ),
        )
        manifest.validate()
        manifest_path = output / MANIFEST_FILENAME
        with _exclusive_open(manifest_path) as handle:
            created.append(manifest_path)
            handle.write(manifest.canonical_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        return manifest
    except Exception:
        for handle in handles.values():
            handle.close()
        for path in reversed(created):
            try:
                path.chmod(0o600)
                path.unlink()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def load_manifest(path: str | Path) -> DatasetManifest:
    """Load and strictly validate one canonical manifest."""

    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_FILENAME
    return DatasetManifest.from_canonical_bytes(
        _regular_file_bytes(manifest_path, "manifest")
    )


def reload_sharded_dataset(
    path: str | Path,
) -> tuple[tuple[SemanticCoreRecord, ...], DatasetManifest]:
    """Reload, hash-check, assign-check, and deduplicate every shard row."""

    location = Path(path)
    manifest_path = location / MANIFEST_FILENAME if location.is_dir() else location
    manifest = load_manifest(manifest_path)
    root = manifest_path.parent
    records: list[SemanticCoreRecord] = []
    core_ids: set[str] = set()
    semantic_hashes: set[str] = set()
    for descriptor in manifest.shards:
        relative = PurePosixPath(descriptor.filename)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != descriptor.filename
        ):
            raise ShardError("manifest contains an unsafe shard path")
        shard_path = root / descriptor.filename
        payload = _regular_file_bytes(shard_path, f"shard {descriptor.filename}")
        if (
            len(payload) != descriptor.byte_count
            or hashlib.sha256(payload).hexdigest() != descriptor.sha256
        ):
            raise ShardError(f"shard identity differs: {descriptor.filename}")
        rows = 0
        uncompressed_bytes = 0
        previous_hash: str | None = None
        try:
            with gzip.open(shard_path, "rb") as handle:
                for payload_row in handle:
                    uncompressed_bytes += len(payload_row)
                    record = SemanticCoreRecord.from_canonical_bytes(payload_row)
                    core_hash = record.core_sha256()
                    if previous_hash is not None and core_hash <= previous_hash:
                        raise ShardError(
                            f"shard row order differs: {descriptor.filename}"
                        )
                    previous_hash = core_hash
                    if (
                        shard_index_for_hash(core_hash, manifest.shard_count)
                        != descriptor.index
                    ):
                        raise ShardError(
                            f"shard assignment differs: {record.identity.core_id}"
                        )
                    core_id = record.identity.core_id
                    semantic_hash = record.assessor_only.audit.semantic_hash
                    if core_id in core_ids:
                        raise ShardError(f"duplicate core ID: {core_id}")
                    if semantic_hash in semantic_hashes:
                        raise ShardError(f"duplicate semantic hash: {semantic_hash}")
                    core_ids.add(core_id)
                    semantic_hashes.add(semantic_hash)
                    records.append(record)
                    rows += 1
        except (OSError, EOFError) as exc:
            raise ShardError(
                f"shard decompression failed: {descriptor.filename}"
            ) from exc
        if (
            rows != descriptor.row_count
            or uncompressed_bytes != descriptor.uncompressed_byte_count
        ):
            raise ShardError(f"shard counts differ: {descriptor.filename}")
    if len(records) != manifest.row_count:
        raise ShardError("reloaded row count differs from manifest")
    manifest.validate()
    return tuple(records), manifest


def validate_sharded_dataset(path: str | Path) -> DatasetManifest:
    """Fully reload a shard set and return its validated manifest."""

    _, manifest = reload_sharded_dataset(path)
    return manifest


__all__ = [
    "AssessorOnly",
    "AuditRecord",
    "CoreIdentity",
    "CounterfactualGroups",
    "CoverageRecord",
    "DATASET_ROOT_SCHEMA",
    "DatasetManifest",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "OracleChannel",
    "OracleRecord",
    "RECORD_SCHEMA",
    "SHARD_SCHEMA",
    "SemanticCoreRecord",
    "SemanticFactors",
    "ShardDescriptor",
    "ShardError",
    "SourceView",
    "SourceVisible",
    "TargetRecord",
    "canonical_json_bytes",
    "canonical_sha256",
    "dataset_root_sha256",
    "load_manifest",
    "reload_sharded_dataset",
    "semantic_factors_sha256",
    "shard_index_for_hash",
    "shard_index_for_record",
    "validate_sharded_dataset",
    "write_sharded_dataset",
]
