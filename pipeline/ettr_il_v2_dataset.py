"""Deterministic ETTR-IL-v2 dataset manifest and freeze-plan sidecar.

This module sits strictly after semantic selection and tensor materialization.
It does not enumerate candidates, infer dynamic generator feasibility, fit
weights, load a checkpoint, or submit work.  Its responsibilities are:

* validate the exact frozen split/view quotas;
* bind every selected semantic rectangle to one validated CPU materialized
  batch;
* validate the exact invariant-pair population and derive the frozen
  6,000-update schedule identities;
* compute split-disjointness and permitted primitive-template overlap
  metadata;
* bind source files and an upstream cardinality-validation receipt by
  independently measured SHA-256;
* derive the non-circular dataset and manifest hashes; and
* publish the canonical manifest with an exclusive no-replace write.

Production mode is intentionally unusable with a partial population.  Canary
mode is intentionally marked as non-claim-bearing and does not assert dynamic
cardinality or emit a training schedule.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence

from ettr_il_v2_schedule import (
    FIT_ONTOLOGIES,
    InvariantPairRecord,
    MODEL_SEEDS,
    PAIRS_PER_FOLD,
    build_pair_schedule,
)


PROTOCOL = "R12-ETTR-IL-v2"
MANIFEST_SCHEMA = "r12-ettr-il-v2-dataset-manifest-v1"
PAYLOAD_INDEX_SCHEMA = "r12-ettr-il-v2-dataset-payload-index-v1"
CARDINALITY_RECEIPT_SCHEMA = (
    "r12-ettr-il-v2-cardinality-validation-receipt-v1"
)
MASTER_SEED_SHA256 = (
    "f6edaccd75ba80763540b990fcd0d1c85016e2d62a79cc3bbe328a206db925dd"
)
TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)
TOKENIZER_BYTES = 2_309_567
WORLD_WIDTH = 192
COMMAND_WIDTH = 96
QUERY_WIDTH = 48
TRANSACTION_STEPS = 64
ROWS_PER_RECTANGLE = 16
CAUSAL_RECTANGLES_PER_RECTANGLE = 4
TRAIN_RECTANGLES_PER_FOLD = 2_304
SCORE_RECTANGLES_PER_FOLD_SPLIT = 2_304
RECTANGLES_PER_PRODUCTION_DATASET = 20_736

FOLDS = (0, 1, 2)
SPLITS = ("train", "development", "confirmation")
ONTOLOGIES = ("horn", "rewrite", "resource")
STRATA = (
    "seen_id",
    "rule",
    "composition",
    "renderer",
    "rule_composition",
    "rule_renderer",
    "composition_renderer",
    "all_axes",
)
PRESENTATIONS = (
    "base",
    "alpha_reorder",
    "alias_split",
    "relation_reification",
    "type_twin",
    "execution_semantics_twin",
)
FIT_PRESENTATIONS = ("base", "alpha_reorder", "alias_split")
ALL_AXES_PRESENTATIONS = (
    "base",
    "relation_reification",
    "type_twin",
    "execution_semantics_twin",
)
FIT_THEORIES = {
    "horn": (1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 17, 18),
    "rewrite": (0, 4, 5, 7, 9, 12, 14),
    "resource": tuple(
        value
        for value in range(60)
        if value not in {0, 7, 14, 21, 30, 39, 48, 59}
    ),
}
SCORE_THEORIES = {
    "horn": (0, 2, 5, 7, 10, 12, 15, 19),
    "rewrite": (1, 2, 3, 6, 8, 10, 11, 13),
    "resource": (0, 7, 14, 21, 30, 39, 48, 59),
}
RULE_SHIFT_STRATA = frozenset(
    ("rule", "rule_composition", "rule_renderer", "all_axes")
)
RENDERER_SHIFT_STRATA = frozenset(
    ("renderer", "rule_renderer", "composition_renderer", "all_axes")
)
COMPOSITION_SHIFT_STRATA = frozenset(
    (
        "composition",
        "rule_composition",
        "composition_renderer",
        "all_axes",
    )
)
REQUIRED_SOURCE_ROLES = frozenset(
    (
        "arms_statistics_spec",
        "cardinality_report",
        "cardinality_validation_receipt",
        "cardinality_validator",
        "materialization_spec",
        "materializer_source",
        "semantic_generator_source",
        "semantic_generator_spec",
        "tokenizer",
    )
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LOGICAL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class DatasetError(ValueError):
    """The selected/materialized population cannot be frozen as ETTR-IL-v2."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_hex(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise DatasetError(f"{name} differs")
    return value


def _require_ascii(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or any(ord(character) < 32 or ord(character) > 126 for character in value)
    ):
        raise DatasetError(f"{name} differs")
    return value


def _strict_json_value(value: object, name: str = "canonical JSON") -> None:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and not value.isascii():
            raise DatasetError(f"{name} contains non-ASCII text")
        return
    if _plain_int(value):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _strict_json_value(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key.isascii():
                raise DatasetError(f"{name} has a non-ASCII string key")
            _strict_json_value(item, f"{name}.{key}")
        return
    raise DatasetError(f"{name} contains a non-canonical value")


def canonical_json_bytes(value: object) -> bytes:
    """Return the protocol's strict canonical JSON bytes, including final LF."""

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
        raise DatasetError("canonical JSON rendering failed") from exc
    return (rendered + "\n").encode("ascii")


def _strict_loads(payload: bytes, name: str) -> object:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise DatasetError(f"{name} is not canonical JSON")

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DatasetError(f"{name} has a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicate,
            parse_float=lambda _: (_ for _ in ()).throw(
                DatasetError(f"{name} contains a float")
            ),
            parse_constant=lambda _: (_ for _ in ()).throw(
                DatasetError(f"{name} contains a non-finite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{name} is not strict ASCII JSON") from exc
    if canonical_json_bytes(value) != payload:
        raise DatasetError(f"{name} is not canonical JSON")
    return value


def _logical_name(value: object, name: str) -> str:
    text = _require_ascii(value, name)
    if _LOGICAL_NAME.fullmatch(text) is None:
        raise DatasetError(f"{name} differs")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetError(f"{name} is not a safe relative path")
    return text


def _read_regular_file(path: Path, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise DatasetError(f"{name} cannot be inspected") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise DatasetError(f"{name} is not a single-link regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DatasetError(f"{name} cannot be opened without following links") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise DatasetError(f"{name} changed before measurement")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise DatasetError(f"{name} changed during measurement")
        payload = b"".join(chunks)
        if len(payload) != opened.st_size:
            raise DatasetError(f"{name} byte count changed")
        return payload
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """One independently remeasured source identity."""

    role: str
    logical_name: str
    path: Path
    sha256: str
    byte_count: int

    @classmethod
    def from_path(
        cls,
        *,
        role: str,
        logical_name: str,
        path: str | Path,
    ) -> "SourceArtifact":
        source = Path(path)
        payload = _read_regular_file(source, f"source {logical_name}")
        return cls(
            role=_require_ascii(role, "source role"),
            logical_name=_logical_name(logical_name, "source logical_name"),
            path=source,
            sha256=_sha256(payload),
            byte_count=len(payload),
        )

    def verify(self) -> bytes:
        _require_ascii(self.role, "source role")
        _logical_name(self.logical_name, "source logical_name")
        _require_hex(self.sha256, "source sha256")
        if not _plain_int(self.byte_count) or self.byte_count < 1:
            raise DatasetError("source byte_count differs")
        payload = _read_regular_file(self.path, f"source {self.logical_name}")
        if len(payload) != self.byte_count or _sha256(payload) != self.sha256:
            raise DatasetError(f"source identity changed: {self.logical_name}")
        return payload

    def manifest_value(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "logical_name": self.logical_name,
            "role": self.role,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SelectedRectangle:
    """Assessor-side identity and leakage metadata for one selected view."""

    semantic_rectangle_id: str
    semantic_core_id: str
    fold: int
    split: str
    ontology: str
    stratum: str
    depth: int
    renderer: int
    presentation: str
    theory_pool_index: int
    law_signature: str
    world_ids: tuple[str, str]
    command_instance_ids: tuple[str, str]
    command_template_ids: tuple[str, str]
    opaque_name_ids: tuple[str, ...]
    raw_row_sha256: tuple[str, ...]
    world_source_sha256: tuple[str, str, str, str]
    command_source_sha256: tuple[str, str, str, str]
    query_prefix_sha256: tuple[str, str, str, str]

    def validate(self) -> None:
        _require_hex(self.semantic_rectangle_id, "semantic_rectangle_id")
        _require_hex(self.semantic_core_id, "semantic_core_id")
        if self.fold not in FOLDS:
            raise DatasetError("rectangle fold differs")
        if self.split not in SPLITS:
            raise DatasetError("rectangle split differs")
        if self.ontology not in ONTOLOGIES:
            raise DatasetError("rectangle ontology differs")
        if self.stratum not in STRATA:
            raise DatasetError("rectangle stratum differs")
        if not _plain_int(self.depth) or not 1 <= self.depth <= 6:
            raise DatasetError("rectangle depth differs")
        if self.renderer not in (0, 1, 2, 3):
            raise DatasetError("rectangle renderer differs")
        if self.presentation not in PRESENTATIONS:
            raise DatasetError("rectangle presentation differs")
        if not _plain_int(self.theory_pool_index) or self.theory_pool_index < 0:
            raise DatasetError("rectangle theory_pool_index differs")
        _require_hex(self.law_signature, "rectangle law_signature")
        theory_pool = (
            FIT_THEORIES[self.ontology]
            if self.split == "train" or self.stratum not in RULE_SHIFT_STRATA
            else SCORE_THEORIES[self.ontology]
        )
        if self.theory_pool_index not in theory_pool:
            raise DatasetError("rectangle theory leaves its frozen pool")
        for field_name, values, expected_count in (
            ("world_ids", self.world_ids, 2),
            ("command_instance_ids", self.command_instance_ids, 2),
            ("command_template_ids", self.command_template_ids, 2),
            ("raw_row_sha256", self.raw_row_sha256, ROWS_PER_RECTANGLE),
            ("world_source_sha256", self.world_source_sha256, 4),
            ("command_source_sha256", self.command_source_sha256, 4),
            ("query_prefix_sha256", self.query_prefix_sha256, 4),
        ):
            if (
                not isinstance(values, tuple)
                or len(values) != expected_count
                or any(_HEX64.fullmatch(value) is None for value in values)
            ):
                raise DatasetError(f"rectangle {field_name} differs")
        if (
            not isinstance(self.opaque_name_ids, tuple)
            or not self.opaque_name_ids
            or any(_HEX64.fullmatch(value) is None for value in self.opaque_name_ids)
        ):
            raise DatasetError("rectangle opaque_name_ids differ")
        if len(set(self.raw_row_sha256)) != ROWS_PER_RECTANGLE:
            raise DatasetError("rectangle raw rows are not unique")
        if self.split == "train":
            if (
                self.stratum != "seen_id"
                or self.depth not in (1, 2, 3)
                or self.renderer not in (0, 1)
                or self.presentation not in FIT_PRESENTATIONS
                or self.ontology not in FIT_ONTOLOGIES[self.fold]
            ):
                raise DatasetError("training rectangle leaves the fit population")

    def manifest_value(self) -> dict[str, object]:
        source_receipt = {
            "command_source_sha256": list(self.command_source_sha256),
            "query_prefix_sha256": list(self.query_prefix_sha256),
            "raw_row_sha256": list(self.raw_row_sha256),
            "world_source_sha256": list(self.world_source_sha256),
        }
        return {
            "candidate_source_receipt_sha256": _sha256(
                canonical_json_bytes(source_receipt)
            ),
            "depth": self.depth,
            "fold": self.fold,
            "ontology": self.ontology,
            "presentation": self.presentation,
            "renderer": self.renderer,
            "semantic_core_id": self.semantic_core_id,
            "semantic_rectangle_id": self.semantic_rectangle_id,
            "split": self.split,
            "stratum": self.stratum,
            "theory_pool_index": self.theory_pool_index,
            "law_signature": self.law_signature,
        }


@dataclass(frozen=True, slots=True)
class MaterializedBatchInput:
    """One already materialized CPU batch plus its selected-view ownership."""

    fold: int
    split: str
    rectangle_ids: tuple[str, ...]
    batch: Any


@dataclass(frozen=True, slots=True)
class FrozenBatchRecord:
    batch_id: str
    fold: int
    split: str
    rectangle_ids: tuple[str, ...]
    row_count: int
    causal_rectangle_count: int
    equivariance_pair_count: int
    payload_sha256: str

    def manifest_value(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "causal_rectangle_count": self.causal_rectangle_count,
            "equivariance_pair_count": self.equivariance_pair_count,
            "fold": self.fold,
            "payload_sha256": self.payload_sha256,
            "rectangle_ids": list(self.rectangle_ids),
            "row_count": self.row_count,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class DatasetBuildRequest:
    mode: str
    vocab_size: int
    rectangles: tuple[SelectedRectangle, ...]
    invariant_pairs: tuple[InvariantPairRecord, ...]
    materialized_batches: tuple[MaterializedBatchInput, ...]
    sources: tuple[SourceArtifact, ...]


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    mode: str
    dataset_sha256: str
    manifest_sha256: str
    manifest: dict[str, object]
    manifest_bytes: bytes
    frozen_batches: tuple[Any, ...]


def _all_tensors(value: object) -> Iterable[Any]:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise DatasetError("torch is required to inspect materialized batches") from exc
    if torch.is_tensor(value):
        yield value
        return
    if hasattr(value, "__dataclass_fields__"):
        for field in fields(value):
            yield from _all_tensors(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _all_tensors(item)


def _freeze_materialized_batch(
    value: MaterializedBatchInput,
    *,
    vocab_size: int,
    final_manifest_sha256: str | None = None,
    final_dataset_sha256: str | None = None,
) -> tuple[FrozenBatchRecord, Any]:
    if (
        not isinstance(value, MaterializedBatchInput)
        or value.fold not in FOLDS
        or value.split not in SPLITS
        or not isinstance(value.rectangle_ids, tuple)
        or not value.rectangle_ids
        or any(_HEX64.fullmatch(item) is None for item in value.rectangle_ids)
        or len(set(value.rectangle_ids)) != len(value.rectangle_ids)
    ):
        raise DatasetError("materialized batch descriptor differs")
    if not _plain_int(vocab_size) or vocab_size <= 1:
        raise DatasetError("vocab_size differs")

    train = Path(__file__).resolve().parents[1] / "train"
    if str(train) not in sys.path:
        sys.path.insert(0, str(train))
    try:
        from endogenous_typed_theory_reactor import (  # noqa: PLC0415
            TheoryReactorConfig,
        )
        from ettr_data_contract import (  # noqa: PLC0415
            ETTRContinuationBatch,
            continuation_batch_payload_sha256,
        )
        from ettr_objectives import ETTRObjectiveConfig  # noqa: PLC0415
    except ImportError as exc:
        raise DatasetError("ETTR receiving contracts are unavailable") from exc
    if not isinstance(value.batch, ETTRContinuationBatch):
        raise DatasetError("materialized batch type differs")
    batch = value.batch
    if final_manifest_sha256 is not None or final_dataset_sha256 is not None:
        batch = replace(
            batch,
            manifest_sha256=_require_hex(
                final_manifest_sha256, "final manifest sha256"
            ),
            dataset_sha256=_require_hex(
                final_dataset_sha256, "final dataset sha256"
            ),
        )
    try:
        batch.validate(
            TheoryReactorConfig(),
            ETTRObjectiveConfig(
                vocab_size=vocab_size,
                require_equivariance_pairs=value.split == "train",
            ),
        )
    except Exception as exc:
        raise DatasetError("materialized batch validation failed") from exc
    if any(tensor.device.type != "cpu" for tensor in _all_tensors(batch)):
        raise DatasetError("materialized batch is not CPU-only")
    rows = int(batch.episodes.world.tokens.shape[0])
    causal = int(batch.causal_rectangles.rows.shape[0])
    equivariance = (
        0
        if batch.equivariance is None
        else int(batch.equivariance.left_index.numel())
    )
    expected_rows = ROWS_PER_RECTANGLE * len(value.rectangle_ids)
    expected_causal = CAUSAL_RECTANGLES_PER_RECTANGLE * len(value.rectangle_ids)
    if rows != expected_rows or causal != expected_causal:
        raise DatasetError("materialized batch rectangle geometry differs")
    if value.split == "train":
        if len(value.rectangle_ids) != 2 or equivariance != ROWS_PER_RECTANGLE:
            raise DatasetError("training batch is not one invariant pair")
    elif equivariance != 0:
        raise DatasetError("score-only batch manufactures equivariance labels")
    payload_sha256 = continuation_batch_payload_sha256(batch)
    batch_value = {
        "causal_rectangle_count": causal,
        "equivariance_pair_count": equivariance,
        "fold": value.fold,
        "payload_sha256": payload_sha256,
        "rectangle_ids": list(value.rectangle_ids),
        "row_count": rows,
        "split": value.split,
    }
    batch_id = _sha256(canonical_json_bytes(batch_value))
    return (
        FrozenBatchRecord(
            batch_id=batch_id,
            fold=value.fold,
            split=value.split,
            rectangle_ids=value.rectangle_ids,
            row_count=rows,
            causal_rectangle_count=causal,
            equivariance_pair_count=equivariance,
            payload_sha256=payload_sha256,
        ),
        batch,
    )


def _selected_core_root(rectangles: Sequence[SelectedRectangle]) -> tuple[int, str]:
    core_keys = sorted(
        {
            (value.fold, value.split, value.semantic_core_id)
            for value in rectangles
        }
    )
    values = [
        {
            "fold": fold,
            "semantic_core_id": core_id,
            "split": split,
        }
        for fold, split, core_id in core_keys
    ]
    return len(values), _sha256(canonical_json_bytes(values))


def _validate_sources(
    sources: Sequence[SourceArtifact],
    rectangles: Sequence[SelectedRectangle],
    *,
    mode: str,
) -> tuple[dict[str, bytes], list[dict[str, object]], dict[str, object]]:
    if not sources:
        raise DatasetError("source inventory is empty")
    roles: dict[str, SourceArtifact] = {}
    names: set[str] = set()
    payloads: dict[str, bytes] = {}
    for source in sources:
        if not isinstance(source, SourceArtifact):
            raise DatasetError("source inventory type differs")
        if source.role in roles:
            raise DatasetError(f"duplicate source role: {source.role}")
        if source.logical_name in names:
            raise DatasetError(f"duplicate source logical name: {source.logical_name}")
        roles[source.role] = source
        names.add(source.logical_name)
        payloads[source.role] = source.verify()
    if mode == "production" and not REQUIRED_SOURCE_ROLES <= set(roles):
        missing = sorted(REQUIRED_SOURCE_ROLES - set(roles))
        raise DatasetError(f"production source roles are incomplete: missing={missing}")
    if "tokenizer" in roles and (
        roles["tokenizer"].sha256 != TOKENIZER_SHA256
        or roles["tokenizer"].byte_count != TOKENIZER_BYTES
    ):
        raise DatasetError("tokenizer source identity differs")

    count, core_root = _selected_core_root(rectangles)
    cardinality = {
        "dynamic_counts_recomputed_by_dataset_builder": False,
        "theory_remainder_offset_recomputed_by_dataset_builder": False,
        "selected_semantic_core_count": count,
        "selected_semantic_core_ids_sha256": core_root,
        "status": "canary_only_not_asserted",
    }
    required_cardinality = {
        "cardinality_report",
        "cardinality_validation_receipt",
        "cardinality_validator",
    }
    present_cardinality = required_cardinality & set(roles)
    if present_cardinality and present_cardinality != required_cardinality:
        raise DatasetError("cardinality source bundle is incomplete")
    if required_cardinality <= set(roles):
        receipt_payload = payloads["cardinality_validation_receipt"]
        receipt = _strict_loads(receipt_payload, "cardinality validation receipt")
        expected_fields = {
            "protocol",
            "report_sha256",
            "result",
            "schema",
            "selected_semantic_core_count",
            "selected_semantic_core_ids_sha256",
            "validator_sha256",
        }
        if not isinstance(receipt, dict) or set(receipt) != expected_fields:
            raise DatasetError("cardinality validation receipt fields differ")
        expected_result = "pass" if mode == "production" else "canary"
        if (
            receipt["schema"] != CARDINALITY_RECEIPT_SCHEMA
            or receipt["protocol"] != PROTOCOL
            or receipt["result"] != expected_result
            or receipt["report_sha256"] != roles["cardinality_report"].sha256
            or receipt["validator_sha256"] != roles["cardinality_validator"].sha256
            or receipt["selected_semantic_core_count"] != count
            or receipt["selected_semantic_core_ids_sha256"] != core_root
        ):
            raise DatasetError("cardinality validation receipt identity differs")
        cardinality = {
            "dynamic_counts_recomputed_by_dataset_builder": False,
            "theory_remainder_offset_recomputed_by_dataset_builder": False,
            "report_sha256": receipt["report_sha256"],
            "selected_semantic_core_count": count,
            "selected_semantic_core_ids_sha256": core_root,
            "status": (
                "bound_to_upstream_validated_report"
                if mode == "production"
                else "canary_only_not_asserted"
            ),
            "validation_receipt_sha256": roles[
                "cardinality_validation_receipt"
            ].sha256,
            "validator_sha256": receipt["validator_sha256"],
        }
    elif mode == "production":
        raise DatasetError("production cardinality sources are incomplete")

    manifest_sources = [
        source.manifest_value()
        for source in sorted(sources, key=lambda item: (item.role, item.logical_name))
    ]
    return payloads, manifest_sources, cardinality


def _validate_rectangle_identity(rectangles: Sequence[SelectedRectangle]) -> None:
    if not rectangles:
        raise DatasetError("selected rectangle population is empty")
    rectangle_ids: set[str] = set()
    core_owners: dict[tuple[int, str], tuple[str, str, str, int, int, str]] = {}
    raw_rows: set[str] = set()
    command_instance_owners: dict[tuple[int, str], str] = {}
    for rectangle in rectangles:
        if not isinstance(rectangle, SelectedRectangle):
            raise DatasetError("selected rectangle type differs")
        rectangle.validate()
        if rectangle.semantic_rectangle_id in rectangle_ids:
            raise DatasetError("semantic rectangle ID is duplicated")
        rectangle_ids.add(rectangle.semantic_rectangle_id)
        core_key = (rectangle.fold, rectangle.semantic_core_id)
        metadata = (
            rectangle.split,
            rectangle.ontology,
            rectangle.stratum,
            rectangle.depth,
            rectangle.theory_pool_index,
            rectangle.law_signature,
        )
        prior = core_owners.setdefault(core_key, metadata)
        if prior != metadata:
            raise DatasetError("semantic core metadata differs within a fold")
        overlap = raw_rows.intersection(rectangle.raw_row_sha256)
        if overlap:
            raise DatasetError("raw row hash is duplicated")
        raw_rows.update(rectangle.raw_row_sha256)
        for instance_id in rectangle.command_instance_ids:
            prior_core = command_instance_owners.setdefault(
                (rectangle.fold, instance_id), rectangle.semantic_core_id
            )
            if prior_core != rectangle.semantic_core_id:
                raise DatasetError(
                    "command instance ID occurs in multiple cores within a fold"
                )


def _validate_theory_balance(
    core_views: Mapping[str, Sequence[SelectedRectangle]],
    *,
    pool: Sequence[int],
    name: str,
) -> None:
    counts: Counter[int] = Counter()
    for core_id, views in core_views.items():
        theories = {value.theory_pool_index for value in views}
        laws = {value.law_signature for value in views}
        if len(theories) != 1 or len(laws) != 1:
            raise DatasetError(f"{name} core theory identity differs: {core_id}")
        theory = next(iter(theories))
        if theory not in pool:
            raise DatasetError(f"{name} core leaves its theory pool: {core_id}")
        counts[theory] += 1
    quotient, remainder = divmod(len(core_views), len(pool))
    observed = sorted(counts.get(theory, 0) for theory in pool)
    expected = sorted(
        [quotient] * (len(pool) - remainder)
        + [quotient + 1] * remainder
    )
    if observed != expected:
        raise DatasetError(f"{name} theory remainder quota differs")


def _validate_train_quotas(
    rectangles: Sequence[SelectedRectangle],
    *,
    production: bool,
) -> None:
    by_fold = defaultdict(list)
    for rectangle in rectangles:
        if rectangle.split == "train":
            by_fold[rectangle.fold].append(rectangle)
    if not production:
        return
    if set(by_fold) != set(FOLDS):
        raise DatasetError("training folds differ")
    expected_views = {
        ("base", 0): 96,
        ("alpha_reorder", 1): 96,
        ("base", 1): 96,
        ("alias_split", 0): 96,
    }
    for fold in FOLDS:
        values = by_fold[fold]
        if len(values) != TRAIN_RECTANGLES_PER_FOLD:
            raise DatasetError(f"fold {fold} training rectangle count differs")
        expected_ontologies = set(FIT_ONTOLOGIES[fold])
        if {value.ontology for value in values} != expected_ontologies:
            raise DatasetError(f"fold {fold} fit ontology membership differs")
        counts = Counter(
            (
                value.ontology,
                value.depth,
                value.presentation,
                value.renderer,
            )
            for value in values
        )
        expected = {
            (ontology, depth, presentation, renderer): count
            for ontology in FIT_ONTOLOGIES[fold]
            for depth in (1, 2, 3)
            for (presentation, renderer), count in expected_views.items()
        }
        if dict(counts) != expected:
            raise DatasetError(f"fold {fold} exact training view quotas differ")
        cores: dict[str, list[SelectedRectangle]] = defaultdict(list)
        for value in values:
            cores[value.semantic_core_id].append(value)
        if len(cores) != 576:
            raise DatasetError(f"fold {fold} training core count differs")
        for core_id, views in cores.items():
            if len(views) != 4:
                raise DatasetError(f"training core view count differs: {core_id}")
            observed = Counter((view.presentation, view.renderer) for view in views)
            if observed != Counter(expected_views.keys()):
                raise DatasetError(f"training core view bundle differs: {core_id}")
        for ontology in FIT_ONTOLOGIES[fold]:
            for depth in (1, 2, 3):
                cell_cores: dict[str, list[SelectedRectangle]] = defaultdict(list)
                for value in values:
                    if value.ontology == ontology and value.depth == depth:
                        cell_cores[value.semantic_core_id].append(value)
                _validate_theory_balance(
                    cell_cores,
                    pool=FIT_THEORIES[ontology],
                    name=f"fold {fold} train {ontology} depth {depth}",
                )


def _validate_score_quotas(
    rectangles: Sequence[SelectedRectangle],
    *,
    production: bool,
) -> None:
    if not production:
        return
    for fold in FOLDS:
        for split in ("development", "confirmation"):
            split_values = [
                value
                for value in rectangles
                if value.fold == fold and value.split == split
            ]
            if len(split_values) != SCORE_RECTANGLES_PER_FOLD_SPLIT:
                raise DatasetError(
                    f"fold {fold} {split} scored rectangle count differs"
                )
            for ontology in ONTOLOGIES:
                for stratum in STRATA:
                    values = [
                        value
                        for value in split_values
                        if value.ontology == ontology and value.stratum == stratum
                    ]
                    name = f"fold {fold} {split} {ontology} {stratum}"
                    if len(values) != 96:
                        raise DatasetError(f"{name} rectangle quota differs")
                    cores: dict[str, list[SelectedRectangle]] = defaultdict(list)
                    for value in values:
                        cores[value.semantic_core_id].append(value)
                    expected_core_count = 24 if stratum == "all_axes" else 32
                    expected_views = 4 if stratum == "all_axes" else 3
                    if len(cores) != expected_core_count or any(
                        len(core_views) != expected_views
                        for core_views in cores.values()
                    ):
                        raise DatasetError(f"{name} core/view quota differs")
                    presentations = Counter(value.presentation for value in values)
                    expected_presentations = (
                        ALL_AXES_PRESENTATIONS
                        if stratum == "all_axes"
                        else FIT_PRESENTATIONS
                    )
                    if presentations != Counter(
                        {presentation: expected_core_count for presentation in expected_presentations}
                    ):
                        raise DatasetError(f"{name} presentation quota differs")

                    shifted_renderer = stratum in RENDERER_SHIFT_STRATA
                    if split == "confirmation" and shifted_renderer:
                        expected_renderers = Counter({3: 96})
                    elif shifted_renderer:
                        expected_renderers = Counter({2: 48, 3: 48})
                    else:
                        expected_renderers = Counter({0: 48, 1: 48})
                    if Counter(value.renderer for value in values) != expected_renderers:
                        raise DatasetError(f"{name} renderer quota differs")

                    if stratum == "all_axes":
                        expected_depths = Counter({4: 32, 5: 32, 6: 32})
                    else:
                        domain = (
                            (4, 5, 6)
                            if stratum in COMPOSITION_SHIFT_STRATA
                            else (1, 2, 3)
                        )
                        depth_counts = Counter(value.depth for value in values)
                        if set(depth_counts) != set(domain) or sorted(
                            depth_counts.values()
                        ) != [30, 33, 33]:
                            raise DatasetError(f"{name} depth quota differs")
                        expected_depths = depth_counts
                    if Counter(value.depth for value in values) != expected_depths:
                        raise DatasetError(f"{name} depth quota differs")
                    for core_id, core_views in cores.items():
                        if len({value.renderer for value in core_views}) != 1:
                            raise DatasetError(
                                f"{name} core renderer differs: {core_id}"
                            )
                        if len({value.depth for value in core_views}) != 1:
                            raise DatasetError(
                                f"{name} core depth differs: {core_id}"
                            )
                        if {value.presentation for value in core_views} != set(
                            expected_presentations
                        ):
                            raise DatasetError(
                                f"{name} core presentation bundle differs: {core_id}"
                            )
                    _validate_theory_balance(
                        cores,
                        pool=(
                            SCORE_THEORIES[ontology]
                            if stratum in RULE_SHIFT_STRATA
                            else FIT_THEORIES[ontology]
                        ),
                        name=name,
                    )


def _validate_pairs(
    rectangles: Sequence[SelectedRectangle],
    pairs: Sequence[InvariantPairRecord],
    *,
    production: bool,
) -> dict[int, tuple[InvariantPairRecord, ...]]:
    rectangle_map = {
        value.semantic_rectangle_id: value
        for value in rectangles
        if value.split == "train"
    }
    pair_ids: set[str] = set()
    used_rectangles: set[str] = set()
    by_fold: dict[int, list[InvariantPairRecord]] = defaultdict(list)
    core_pairs: dict[tuple[int, str], list[tuple[tuple[str, int], tuple[str, int]]]] = (
        defaultdict(list)
    )
    for pair in pairs:
        if not isinstance(pair, InvariantPairRecord):
            raise DatasetError("invariant pair type differs")
        try:
            pair.validate()
        except Exception as exc:
            raise DatasetError("invariant pair validation failed") from exc
        if pair.pair_id in pair_ids:
            raise DatasetError("invariant pair ID is duplicated")
        pair_ids.add(pair.pair_id)
        try:
            left = rectangle_map[pair.left_semantic_rectangle_id]
            right = rectangle_map[pair.right_semantic_rectangle_id]
        except KeyError as exc:
            raise DatasetError("invariant pair references a non-training rectangle") from exc
        if (
            left.fold != right.fold
            or left.semantic_core_id != right.semantic_core_id
            or left.semantic_core_id != pair.semantic_core_id
            or left.ontology != right.ontology
            or left.ontology != pair.ontology
            or left.depth != right.depth
            or left.depth != pair.depth
        ):
            raise DatasetError("invariant pair crosses semantic ownership")
        for rectangle_id in (
            pair.left_semantic_rectangle_id,
            pair.right_semantic_rectangle_id,
        ):
            if rectangle_id in used_rectangles:
                raise DatasetError("training rectangle occurs in multiple pairs")
            used_rectangles.add(rectangle_id)
        by_fold[left.fold].append(pair)
        core_pairs[(left.fold, pair.semantic_core_id)].append(
            (
                (left.presentation, left.renderer),
                (right.presentation, right.renderer),
            )
        )
    if used_rectangles != set(rectangle_map):
        raise DatasetError("invariant pairs do not partition training rectangles")
    allowed_pair_bundles = {
        frozenset((("base", 0), ("alpha_reorder", 1))),
        frozenset((("base", 1), ("alias_split", 0))),
    }
    for key, bundles in core_pairs.items():
        observed = {frozenset(bundle) for bundle in bundles}
        if (
            len(observed) != len(bundles)
            or not observed
            or not observed <= allowed_pair_bundles
            or (production and observed != allowed_pair_bundles)
        ):
            raise DatasetError(f"training invariant pair bundle differs: {key}")
    if production:
        if set(by_fold) != set(FOLDS) or any(
            len(by_fold[fold]) != PAIRS_PER_FOLD for fold in FOLDS
        ):
            raise DatasetError("production invariant-pair population differs")
    return {
        fold: tuple(sorted(values, key=lambda value: value.pair_id))
        for fold, values in by_fold.items()
    }


def _validate_batch_partition(
    rectangles: Sequence[SelectedRectangle],
    batch_records: Sequence[FrozenBatchRecord],
    pairs: Sequence[InvariantPairRecord],
) -> None:
    rectangle_map = {value.semantic_rectangle_id: value for value in rectangles}
    pair_by_rectangles = {
        (pair.left_semantic_rectangle_id, pair.right_semantic_rectangle_id)
        for pair in pairs
    }
    covered: set[str] = set()
    batch_ids: set[str] = set()
    payloads: set[str] = set()
    for batch in batch_records:
        if batch.batch_id in batch_ids:
            raise DatasetError("materialized batch ID is duplicated")
        if batch.payload_sha256 in payloads:
            raise DatasetError("materialized batch payload is duplicated")
        batch_ids.add(batch.batch_id)
        payloads.add(batch.payload_sha256)
        for rectangle_id in batch.rectangle_ids:
            if rectangle_id in covered:
                raise DatasetError("rectangle occurs in multiple materialized batches")
            try:
                rectangle = rectangle_map[rectangle_id]
            except KeyError as exc:
                raise DatasetError("batch references an unselected rectangle") from exc
            if rectangle.fold != batch.fold or rectangle.split != batch.split:
                raise DatasetError("batch crosses rectangle fold/split ownership")
            covered.add(rectangle_id)
        if batch.split == "train" and tuple(batch.rectangle_ids) not in pair_by_rectangles:
            raise DatasetError("training batch order is not an invariant pair")
    if covered != set(rectangle_map):
        raise DatasetError("materialized batches do not partition selected rectangles")


def _split_disjointness(
    rectangles: Sequence[SelectedRectangle],
) -> dict[str, object]:
    by_fold: dict[int, dict[str, dict[str, set[str]]]] = {
        fold: {
            split: {
                "world_ids": set(),
                "command_instance_ids": set(),
            "command_template_ids": set(),
            "law_signatures": set(),
                "opaque_name_ids": set(),
                "raw_row_sha256": set(),
            }
            for split in SPLITS
        }
        for fold in FOLDS
    }
    for value in rectangles:
        target = by_fold[value.fold][value.split]
        target["world_ids"].update(value.world_ids)
        target["command_instance_ids"].update(value.command_instance_ids)
        target["command_template_ids"].update(value.command_template_ids)
        target["law_signatures"].add(value.law_signature)
        target["opaque_name_ids"].update(value.opaque_name_ids)
        target["raw_row_sha256"].update(value.raw_row_sha256)

    required_disjoint = (
        "world_ids",
        "command_instance_ids",
        "opaque_name_ids",
        "raw_row_sha256",
    )
    folds: dict[str, object] = {}
    for fold in FOLDS:
        pairwise: dict[str, object] = {}
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                name = f"{left}__{right}"
                overlaps: dict[str, object] = {}
                for field_name in required_disjoint + (
                    "command_template_ids",
                    "law_signatures",
                ):
                    intersection = sorted(
                        by_fold[fold][left][field_name]
                        & by_fold[fold][right][field_name]
                    )
                    overlaps[field_name] = {
                        "count": len(intersection),
                        "sha256": _sha256(canonical_json_bytes(intersection)),
                    }
                    if field_name in required_disjoint and intersection:
                        raise DatasetError(
                            f"fold {fold} split-disjoint {field_name} overlap: {name}"
                        )
                pairwise[name] = overlaps
        split_roots: dict[str, object] = {}
        for split in SPLITS:
            split_roots[split] = {
                field_name: {
                    "count": len(values),
                    "sha256": _sha256(canonical_json_bytes(sorted(values))),
                }
                for field_name, values in sorted(by_fold[fold][split].items())
            }
        folds[str(fold)] = {
            "pairwise": pairwise,
            "split_roots": split_roots,
        }
    return {
        "folds": folds,
        "primitive_template_overlap_is_report_only": True,
        "law_signature_overlap_is_report_only": True,
        "required_disjoint_fields": list(required_disjoint),
    }


def _schedule_receipts(
    pairs_by_fold: Mapping[int, Sequence[InvariantPairRecord]],
    *,
    production: bool,
) -> list[dict[str, object]]:
    if not production:
        return []
    receipts: list[dict[str, object]] = []
    for fold in FOLDS:
        population = pairs_by_fold.get(fold)
        if population is None:
            raise DatasetError(f"fold {fold} pair population is absent")
        for seed in MODEL_SEEDS:
            try:
                schedule = build_pair_schedule(population, fold=fold, seed=seed)
            except Exception as exc:
                raise DatasetError(
                    f"fold {fold} seed {seed} schedule validation failed"
                ) from exc
            receipts.append(schedule.receipt())
    return receipts


def _packet_sufficiency_receipts(
    inputs: Sequence[MaterializedBatchInput],
    batches: Sequence[Any],
) -> dict[str, object]:
    if len(inputs) != len(batches):
        raise DatasetError("packet sufficiency input geometry differs")
    train = Path(__file__).resolve().parents[1] / "train"
    if str(train) not in sys.path:
        sys.path.insert(0, str(train))
    try:
        from ettr_data_contract import ETTRPacketSufficiencyIndex  # noqa: PLC0415
    except ImportError as exc:
        raise DatasetError("ETTR packet sufficiency contract is unavailable") from exc

    fold_receipts: dict[str, object] = {}
    present_folds = sorted({value.fold for value in inputs})
    for fold in present_folds:
        train_batches = tuple(
            batch
            for value, batch in zip(inputs, batches, strict=True)
            if value.fold == fold and value.split == "train"
        )
        validation_batches = tuple(
            batch
            for value, batch in zip(inputs, batches, strict=True)
            if value.fold == fold
            and value.split in {"development", "confirmation"}
        )
        if not train_batches:
            raise DatasetError(f"fold {fold} packet sufficiency train set is empty")
        try:
            index = ETTRPacketSufficiencyIndex.from_splits(
                train_batches,
                validation_batches,
            )
            index.verify_train(train_batches)
            if validation_batches:
                index.verify_validation(validation_batches)
        except Exception as exc:
            raise DatasetError(
                f"fold {fold} global packet sufficiency validation failed"
            ) from exc
        receipt = index.receipt
        fold_receipts[str(fold)] = {
            "batches": receipt.batches,
            "context_sha256": receipt.context_sha256,
            "rows": receipt.rows,
            "schema": receipt.schema,
            "target_bound_sha256": receipt.target_bound_sha256,
            "train_batches": index.train_batches,
            "train_payload_sha256": index.train_payload_sha256,
            "train_rows": index.train_rows,
            "unique_contexts": receipt.unique_contexts,
            "validation_batches": index.validation_batches,
            "validation_payload_sha256": index.validation_payload_sha256,
            "validation_rows": index.validation_rows,
        }
    return {
        "folds": fold_receipts,
        "validation_population": "development_plus_confirmation",
    }


def _counts(rectangles: Sequence[SelectedRectangle]) -> dict[str, object]:
    counts: Counter[tuple[int, str, str, str]] = Counter(
        (value.fold, value.split, value.ontology, value.stratum)
        for value in rectangles
    )
    cells = [
        {
            "count": count,
            "fold": fold,
            "ontology": ontology,
            "split": split,
            "stratum": stratum,
        }
        for (fold, split, ontology, stratum), count in sorted(counts.items())
    ]
    return {
        "cells": cells,
        "causal_rectangles": len(rectangles) * CAUSAL_RECTANGLES_PER_RECTANGLE,
        "query_rows": len(rectangles) * ROWS_PER_RECTANGLE,
        "semantic_cores": len(
            {
                (value.fold, value.split, value.semantic_core_id)
                for value in rectangles
            }
        ),
        "semantic_rectangles": len(rectangles),
    }


def build_dataset(request: DatasetBuildRequest) -> DatasetBuildResult:
    """Validate and freeze one production dataset or non-claim canary.

    The function is CPU-only and performs no publication.  Production mode
    returns batches rebound to the final manifest/dataset hashes; payload
    digests are rechecked to prove the receipt fields are excluded.
    """

    if not isinstance(request, DatasetBuildRequest) or request.mode not in {
        "production",
        "canary",
    }:
        raise DatasetError("dataset build mode differs")
    if not _plain_int(request.vocab_size) or request.vocab_size <= 1:
        raise DatasetError("vocab_size differs")
    production = request.mode == "production"
    rectangles = tuple(request.rectangles)
    _validate_rectangle_identity(rectangles)
    if production and len(rectangles) != RECTANGLES_PER_PRODUCTION_DATASET:
        raise DatasetError("production rectangle population count differs")
    _validate_train_quotas(rectangles, production=production)
    _validate_score_quotas(rectangles, production=production)
    pairs_by_fold = _validate_pairs(
        rectangles,
        request.invariant_pairs,
        production=production,
    )
    _, source_inventory, cardinality = _validate_sources(
        request.sources,
        rectangles,
        mode=request.mode,
    )
    disjointness = _split_disjointness(rectangles)

    frozen_records: list[FrozenBatchRecord] = []
    provisional_batches: list[Any] = []
    for value in request.materialized_batches:
        record, batch = _freeze_materialized_batch(
            value,
            vocab_size=request.vocab_size,
        )
        frozen_records.append(record)
        provisional_batches.append(batch)
    frozen_records.sort(key=lambda value: (value.fold, value.split, value.batch_id))
    _validate_batch_partition(
        rectangles,
        frozen_records,
        request.invariant_pairs,
    )
    provisional_sufficiency = _packet_sufficiency_receipts(
        request.materialized_batches,
        provisional_batches,
    )

    schedules = _schedule_receipts(pairs_by_fold, production=production)
    rectangle_index = [
        value.manifest_value()
        for value in sorted(
            rectangles,
            key=lambda item: (
                item.fold,
                item.split,
                item.ontology,
                item.stratum,
                item.semantic_rectangle_id,
            ),
        )
    ]
    pair_index = [
        {
            "depth": value.depth,
            "left_semantic_rectangle_id": value.left_semantic_rectangle_id,
            "ontology": value.ontology,
            "pair_id": value.pair_id,
            "right_semantic_rectangle_id": value.right_semantic_rectangle_id,
            "semantic_core_id": value.semantic_core_id,
        }
        for value in sorted(request.invariant_pairs, key=lambda item: item.pair_id)
    ]
    payload_index = {
        "batches": [value.manifest_value() for value in frozen_records],
        "invariant_pairs_sha256": _sha256(canonical_json_bytes(pair_index)),
        "protocol": PROTOCOL,
        "schema": PAYLOAD_INDEX_SCHEMA,
        "selected_rectangles_sha256": _sha256(
            canonical_json_bytes(rectangle_index)
        ),
        "split_disjointness_sha256": _sha256(
            canonical_json_bytes(disjointness)
        ),
    }
    dataset_sha256 = _sha256(canonical_json_bytes(payload_index))
    manifest: dict[str, object] = {
        "batch_index": [value.manifest_value() for value in frozen_records],
        "cardinality": cardinality,
        "claim_status": (
            "production_manifest_complete_no_weights_fitted"
            if production
            else "canary_only_no_fit_no_dynamic_feasibility_claim"
        ),
        "counts": _counts(rectangles),
        "dataset_sha256": dataset_sha256,
        "geometry": {
            "causal_rectangles_per_semantic_rectangle": (
                CAUSAL_RECTANGLES_PER_RECTANGLE
            ),
            "command_width": COMMAND_WIDTH,
            "query_width": QUERY_WIDTH,
            "rows_per_semantic_rectangle": ROWS_PER_RECTANGLE,
            "transaction_steps": TRANSACTION_STEPS,
            "world_width": WORLD_WIDTH,
        },
        "invariant_pair_index": pair_index,
        "master_seed_sha256": MASTER_SEED_SHA256,
        "mode": request.mode,
        "payload_index": payload_index,
        "packet_sufficiency": provisional_sufficiency,
        "protocol": PROTOCOL,
        "schedule_receipts": schedules,
        "schedule_status": (
            "frozen_15_fold_seed_identities"
            if production
            else "not_emitted_for_partial_canary"
        ),
        "schema": MANIFEST_SCHEMA,
        "selected_rectangle_index": rectangle_index,
        "source_inventory": source_inventory,
        "split_disjointness": disjointness,
        "tokenizer": {
            "byte_count": TOKENIZER_BYTES,
            "pad_token_id": 0,
            "sha256": TOKENIZER_SHA256,
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = _sha256(manifest_bytes)

    final_batches: list[Any] = []
    final_records_by_key: dict[tuple[int, str, tuple[str, ...]], FrozenBatchRecord] = {
        (value.fold, value.split, value.rectangle_ids): value
        for value in frozen_records
    }
    for value in request.materialized_batches:
        record, batch = _freeze_materialized_batch(
            value,
            vocab_size=request.vocab_size,
            final_manifest_sha256=manifest_sha256,
            final_dataset_sha256=dataset_sha256,
        )
        prior = final_records_by_key.get((record.fold, record.split, record.rectangle_ids))
        if prior is None or prior.payload_sha256 != record.payload_sha256:
            raise DatasetError("final batch binding changed its payload digest")
        final_batches.append(batch)
    final_sufficiency = _packet_sufficiency_receipts(
        request.materialized_batches,
        final_batches,
    )
    if final_sufficiency != provisional_sufficiency:
        raise DatasetError("final batch binding changed packet sufficiency")
    return DatasetBuildResult(
        mode=request.mode,
        dataset_sha256=dataset_sha256,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        frozen_batches=tuple(final_batches),
    )


def dry_run_dataset(request: DatasetBuildRequest) -> DatasetBuildResult:
    """Alias making the no-publication/no-fitting boundary explicit."""

    return build_dataset(request)


def publish_manifest_no_replace(
    result: DatasetBuildResult,
    destination: str | Path,
) -> str:
    """Publish one canonical manifest with O_EXCL, fsync, and mode 0444."""

    if not isinstance(result, DatasetBuildResult):
        raise DatasetError("dataset build result differs")
    if _sha256(result.manifest_bytes) != result.manifest_sha256:
        raise DatasetError("manifest bytes changed after build")
    if canonical_json_bytes(result.manifest) != result.manifest_bytes:
        raise DatasetError("manifest is not canonical at publication")
    path = Path(destination)
    if path.name in {"", ".", ".."} or path.parent == path:
        raise DatasetError("manifest destination differs")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise DatasetError("manifest parent cannot be opened safely") from exc
    descriptor: int | None = None
    created = False
    try:
        try:
            descriptor = os.open(path.name, flags, 0o400, dir_fd=parent_fd)
            created = True
        except OSError as exc:
            raise DatasetError("manifest destination already exists or is unsafe") from exc
        payload = memoryview(result.manifest_bytes)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise DatasetError("manifest write made no progress")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        measured = os.fstat(descriptor)
        if (
            not stat.S_ISREG(measured.st_mode)
            or measured.st_nlink != 1
            or stat.S_IMODE(measured.st_mode) != 0o444
            or measured.st_size != len(result.manifest_bytes)
        ):
            raise DatasetError("published manifest inode differs")
        os.fsync(descriptor)
        os.fsync(parent_fd)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    return result.manifest_sha256


def audit_published_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    """Re-read one published manifest and reject noncanonical substitution."""

    expected = _require_hex(expected_sha256, "expected manifest sha256")
    payload = _read_regular_file(Path(path), "published manifest")
    if _sha256(payload) != expected:
        raise DatasetError("published manifest SHA-256 differs")
    value = _strict_loads(payload, "published manifest")
    if (
        not isinstance(value, dict)
        or value.get("schema") != MANIFEST_SCHEMA
        or value.get("protocol") != PROTOCOL
    ):
        raise DatasetError("published manifest identity differs")
    return value


__all__ = [
    "CARDINALITY_RECEIPT_SCHEMA",
    "DatasetBuildRequest",
    "DatasetBuildResult",
    "DatasetError",
    "MANIFEST_SCHEMA",
    "MaterializedBatchInput",
    "SelectedRectangle",
    "SourceArtifact",
    "audit_published_manifest",
    "build_dataset",
    "canonical_json_bytes",
    "dry_run_dataset",
    "publish_manifest_no_replace",
]
