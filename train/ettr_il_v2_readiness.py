"""Read-only trainer/evaluator readiness checks for R12-ETTR-IL-v2.

This module validates the inputs that a later, explicitly authorized fitting
process would consume.  It has no model, optimizer, checkpoint writer,
backward pass, training loop, or job-launch surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Mapping, Sequence

from endogenous_typed_theory_reactor import (
    SYSTEM_PARAMETER_CAP,
    TheoryReactorConfig,
    TheoryReactorError,
)
from ettr_checkpoint import (
    DataStreamState,
    EpisodeLifecycleState,
    TrainingProgress,
    _capture_rng_state,
    _validate_rng_state,
    tree_sha256,
)
from ettr_data_contract import (
    ETTRContinuationBatch,
    ETTRContinuationManifest,
    ETTRPacketSufficiencyIndex,
)
from ettr_model_assembly import (
    ETTRModelAssemblyReceipt,
)
from ettr_objectives import (
    ETTRObjectiveConfig,
    ETTRObjectiveWeights,
)
from ettr_train_step import ETTRTrainStepConfig
from model import GPTConfig


_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _ROOT / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from ettr_il_v2_controls import (  # noqa: E402
    BindingDerangement,
    canonical_json_bytes as control_json_bytes,
)
from ettr_il_v2_schedule import (  # noqa: E402
    FOLDS,
    MICROSTEPS_PER_UPDATE,
    MODEL_SEEDS,
    PAIR_EXPOSURES,
    UPDATES,
    PairSchedule,
    canonical_json_bytes as schedule_json_bytes,
)


PROTOCOL = "R12-ETTR-IL-v2"
READINESS_SCHEMA = "r12-ettr-il-v2-readiness-v1"
DATASET_INDEX_SCHEMA = "r12-ettr-il-v2-dataset-index-v1"
RESUME_CURSOR_SCHEMA = "r12-ettr-il-v2-resume-cursor-v1"
PRIMARY_ARMS = (
    "treatment",
    "state_reset",
    "binding_deranged",
    "query_only",
    "dense_state",
)
ARM_MECHANISMS = {
    "treatment": "native_ettr",
    "state_reset": "reset_before_each_reactor_position",
    "binding_deranged": "matched_target_bundle_derangement",
    "query_only": "canonical_empty_reader_packet",
    "dense_state": "parameter_matched_dense_controller",
}
OBJECTIVE_FAMILIES = (
    "token_lm",
    "packet",
    "world_intervention",
    "command_intervention",
    "world_query_binding",
    "command_query_binding",
    "transaction",
    "equivariance",
    "commit_halt",
    "sparsity",
    "anti_bypass",
)

PROTECTED_BASE_STEP = 300_000
PROTECTED_BASE_PARAMETERS = 125_081_664
ARCHITECTURE_PARAMETERS = 67_697_771
COMPLETE_SYSTEM_PARAMETERS = 192_779_435
ROWS_PER_MICROSTEP = 32
ROWS_PER_UPDATE = 128
SEMANTIC_RECTANGLES_PER_MICROSTEP = 2
SEMANTIC_RECTANGLES_PER_UPDATE = 8
CAUSAL_RECTANGLES_PER_MICROSTEP = 8
CAUSAL_RECTANGLES_PER_UPDATE = 32
ALIGNMENT_PAIRS_PER_MICROSTEP = 16
WORLD_WIDTH = 192
COMMAND_WIDTH = 96
QUERY_WIDTH = 48
TRANSACTION_WIDTH = 64
SUPERVISED_POSITIONS_PER_ROW = (
    WORLD_WIDTH - 1 + COMMAND_WIDTH - 1 + QUERY_WIDTH - 1
)
SUPERVISED_POSITIONS_PER_UPDATE = (
    ROWS_PER_UPDATE * SUPERVISED_POSITIONS_PER_ROW
)
ENCODED_POSITIONS_PER_ROW = WORLD_WIDTH + 2 * COMMAND_WIDTH + 3 * QUERY_WIDTH
ENCODED_POSITIONS_PER_UPDATE = ROWS_PER_UPDATE * ENCODED_POSITIONS_PER_ROW
BINDING_DERANGEMENT_ASSIGNMENTS = 2_304

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ETTRILV2ReadinessError(ValueError):
    """A Phase-1 training or evaluation readiness invariant failed."""


@dataclass(frozen=True, slots=True)
class ImmutableArtifactReceipt:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArmReadinessReceipt:
    """Static receipt emitted independently by one prospective arm."""

    arm: str
    mechanism: str
    arm_config_sha256: str
    control_receipt_sha256: str
    manifest_sha256: str
    dataset_sha256: str
    schedule_sha256: str
    source_payload_sha256: str
    objective_weights_sha256: str
    objective_families: tuple[str, ...]
    microsteps_per_update: int
    rows_per_microstep: int
    rows_per_update: int
    semantic_rectangles_per_update: int
    causal_rectangles_per_update: int
    encoded_positions_per_update: int
    supervised_positions_per_update: int
    updates: int
    trainable_parameters: int
    complete_system_parameters: int


@dataclass(frozen=True, slots=True)
class ResumeReadinessReceipt:
    """Exact between-update cursor and RNG receipt without restoration."""

    progress: TrainingProgress
    data_stream: DataStreamState
    episode_lifecycle: EpisodeLifecycleState
    rng_state: Mapping[str, object]
    rng_state_sha256: str


@dataclass(frozen=True, slots=True)
class ETTRILV2ReadinessRequest:
    manifest_path: Path
    expected_manifest_sha256: str
    dataset_path: Path
    expected_dataset_artifact_sha256: str
    model_assembly_receipt_path: Path
    expected_model_assembly_receipt_sha256: str
    reactor_config_path: Path
    expected_reactor_config_sha256: str
    base_config: GPTConfig
    objective_config: ETTRObjectiveConfig
    step_config: ETTRTrainStepConfig
    packet_sufficiency: ETTRPacketSufficiencyIndex
    update_batches: tuple[ETTRContinuationBatch, ...]
    update_pair_ids: tuple[str, ...]
    schedule: PairSchedule
    arms: tuple[ArmReadinessReceipt, ...]
    binding_derangement: BindingDerangement
    resume: ResumeReadinessReceipt
    evaluator_batch: ETTRContinuationBatch


@dataclass(frozen=True, slots=True)
class ETTRILV2ReadinessReport:
    schema: str
    protocol: str
    status: str
    mode: str
    weight_updates: int
    manifest: ImmutableArtifactReceipt
    dataset: ImmutableArtifactReceipt
    logical_dataset_sha256: str
    model_assembly_receipt_sha256: str
    model_sha256: str
    reactor_config_sha256: str
    schedule_sha256: str
    rng_state_sha256: str
    update_pair_ids: tuple[str, ...]
    update_rows: int
    supervised_positions_per_update: int
    encoded_positions_per_update: int
    arm_count: int
    evaluator_rows: int
    evaluator_query_shape: tuple[int, int]
    complete_system_parameters: int
    remaining_under_cap: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json_bytes(value: object, *, newline: bool) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (payload + ("\n" if newline else "")).encode("ascii")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ETTRILV2ReadinessError(f"{label} SHA-256 differs")
    return value


def _read_immutable(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[bytes, ImmutableArtifactReceipt]:
    _require_sha256(expected_sha256, f"expected {label}")
    path = path.resolve()
    try:
        before = path.lstat()
    except OSError as exc:
        raise ETTRILV2ReadinessError(f"{label} is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_mode & 0o222
        or before.st_nlink != 1
    ):
        raise ETTRILV2ReadinessError(
            f"{label} is not an immutable single-link regular file"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ETTRILV2ReadinessError(f"{label} cannot be read") from exc
    after = path.lstat()
    identity = lambda value: (  # noqa: E731
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise ETTRILV2ReadinessError(f"{label} changed while being read")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ETTRILV2ReadinessError(f"{label} content hash differs")
    return payload, ImmutableArtifactReceipt(
        path=str(path),
        bytes=len(payload),
        sha256=digest,
    )


def _load_continuation_manifest(
    path: Path,
    expected_sha256: str,
) -> tuple[ETTRContinuationManifest, ImmutableArtifactReceipt]:
    payload, artifact = _read_immutable(
        path,
        expected_sha256=expected_sha256,
        label="continuation manifest",
    )
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRILV2ReadinessError(
            "continuation manifest is malformed"
        ) from exc
    if not isinstance(value, dict):
        raise ETTRILV2ReadinessError("continuation manifest is not an object")
    expected_fields = {field.name for field in fields(ETTRContinuationManifest)}
    if set(value) != expected_fields:
        raise ETTRILV2ReadinessError("continuation manifest fields differ")
    if not isinstance(value.get("family_label_fields"), list):
        raise ETTRILV2ReadinessError(
            "continuation manifest family-label field differs"
        )
    value["family_label_fields"] = tuple(value["family_label_fields"])
    try:
        manifest = ETTRContinuationManifest(**value)
        manifest.validate()
    except (TypeError, TheoryReactorError) as exc:
        raise ETTRILV2ReadinessError(
            "continuation manifest contract differs"
        ) from exc
    canonical = _canonical_json_bytes(asdict(manifest), newline=False)
    if payload != canonical or manifest.sha256() != artifact.sha256:
        raise ETTRILV2ReadinessError(
            "continuation manifest is not canonical or self-identical"
        )
    return manifest, artifact


def _load_reactor_config(
    path: Path,
    expected_sha256: str,
    *,
    base_config: GPTConfig,
) -> tuple[TheoryReactorConfig, str]:
    payload, artifact = _read_immutable(
        path,
        expected_sha256=expected_sha256,
        label="reactor configuration",
    )
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRILV2ReadinessError(
            "reactor configuration is malformed"
        ) from exc
    expected_fields = {field.name for field in fields(TheoryReactorConfig)}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ETTRILV2ReadinessError("reactor configuration fields differ")
    try:
        config = TheoryReactorConfig(**value)
        config.validate(n_layer=base_config.n_layer)
    except (TypeError, TheoryReactorError) as exc:
        raise ETTRILV2ReadinessError(
            "reactor configuration is incompatible"
        ) from exc
    if (
        payload != _canonical_json_bytes(asdict(config), newline=True)
        or artifact.sha256 != expected_sha256
    ):
        raise ETTRILV2ReadinessError(
            "reactor configuration is not canonical"
        )
    return config, artifact.sha256


def _load_dataset_index(
    path: Path,
    expected_sha256: str,
    *,
    manifest: ETTRContinuationManifest,
) -> ImmutableArtifactReceipt:
    payload, artifact = _read_immutable(
        path,
        expected_sha256=expected_sha256,
        label="dataset index",
    )
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRILV2ReadinessError("dataset index is malformed") from exc
    expected_fields = {
        "dataset_sha256",
        "manifest_sha256",
        "schema",
        "train_batch_payload_sha256s",
        "train_payload_sha256",
        "validation_batch_payload_sha256s",
        "validation_payload_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ETTRILV2ReadinessError("dataset index fields differ")
    train = value["train_batch_payload_sha256s"]
    validation = value["validation_batch_payload_sha256s"]
    if (
        value["schema"] != DATASET_INDEX_SCHEMA
        or value["manifest_sha256"] != manifest.sha256()
        or value["dataset_sha256"] != manifest.dataset_sha256
        or value["train_payload_sha256"] != manifest.train_payload_sha256
        or value["validation_payload_sha256"]
        != manifest.validation_payload_sha256
        or not isinstance(train, list)
        or not isinstance(validation, list)
        or len(train) != manifest.packet_sufficiency_train_batches
        or len(validation)
        != manifest.packet_sufficiency_validation_batches
        or train != sorted(set(train))
        or validation != sorted(set(validation))
        or set(train) & set(validation)
        or any(_SHA256.fullmatch(item) is None for item in (*train, *validation))
        or hashlib.sha256(
            _canonical_json_bytes(train, newline=False)
        ).hexdigest()
        != manifest.train_payload_sha256
        or hashlib.sha256(
            _canonical_json_bytes(validation, newline=False)
        ).hexdigest()
        != manifest.validation_payload_sha256
        or payload != _canonical_json_bytes(value, newline=True)
    ):
        raise ETTRILV2ReadinessError(
            "dataset index identity or payload population differs"
        )
    return artifact


def _validate_model_and_config(
    request: ETTRILV2ReadinessRequest,
    manifest: ETTRContinuationManifest,
) -> tuple[ETTRModelAssemblyReceipt, TheoryReactorConfig, str]:
    receipt = ETTRModelAssemblyReceipt.from_path(
        request.model_assembly_receipt_path
    )
    if receipt.sha256() != _require_sha256(
        request.expected_model_assembly_receipt_sha256,
        "model assembly receipt",
    ):
        raise ETTRILV2ReadinessError(
            "model assembly receipt identity differs"
        )
    config, config_sha256 = _load_reactor_config(
        request.reactor_config_path,
        request.expected_reactor_config_sha256,
        base_config=request.base_config,
    )
    objective = request.objective_config
    step = request.step_config
    try:
        step.validate()
    except TheoryReactorError as exc:
        raise ETTRILV2ReadinessError(
            "train-step configuration differs"
        ) from exc
    if (
        receipt.config_sha256 != config_sha256
        or receipt.checkpoint_sha256
        != manifest.protected_checkpoint_sha256
        or receipt.checkpoint_step != PROTECTED_BASE_STEP
        or receipt.base_parameters != PROTECTED_BASE_PARAMETERS
        or receipt.architecture_parameters != ARCHITECTURE_PARAMETERS
        or receipt.total_parameters != COMPLETE_SYSTEM_PARAMETERS
        or receipt.parameter_cap != SYSTEM_PARAMETER_CAP
        or receipt.remaining_under_cap
        != SYSTEM_PARAMETER_CAP - COMPLETE_SYSTEM_PARAMETERS
        or request.base_config.d_model != config.d_model
        or request.base_config.vocab_size != objective.vocab_size
        or request.base_config.seq_len < max(
            WORLD_WIDTH,
            COMMAND_WIDTH,
            QUERY_WIDTH,
        )
        or objective.num_slots != config.num_slots
        or objective.num_types != config.num_types
        or objective.num_relations != config.num_relations
        or objective.num_value_codes != config.num_value_codes
        or objective.relation_edge_budget != config.max_edges
        or config.max_steps != TRANSACTION_WIDTH
        or config != TheoryReactorConfig()
        or step.gradient_accumulation_steps != MICROSTEPS_PER_UPDATE
        or not step.hard_transactions
    ):
        raise ETTRILV2ReadinessError(
            "model parameter cap or trainer/config compatibility differs"
        )
    return receipt, config, config_sha256


def _validate_schedule(
    schedule: PairSchedule,
    *,
    optimizer_step: int,
    update_pair_ids: tuple[str, ...],
) -> None:
    if not isinstance(schedule, PairSchedule):
        raise ETTRILV2ReadinessError("pair schedule type differs")
    receipt = schedule.receipt()
    schedule_payload_sha256 = hashlib.sha256(
        schedule_json_bytes(
            [value.as_dict() for value in schedule.exposures]
        )
    ).hexdigest()
    if (
        receipt["protocol"] != PROTOCOL
        or schedule.fold not in FOLDS
        or schedule.seed not in MODEL_SEEDS
        or _SHA256.fullmatch(schedule.population_sha256) is None
        or _SHA256.fullmatch(schedule.schedule_sha256) is None
        or schedule_payload_sha256 != schedule.schedule_sha256
        or receipt["updates"] != UPDATES
        or receipt["microsteps_per_update"] != MICROSTEPS_PER_UPDATE
        or receipt["pair_exposures"] != PAIR_EXPOSURES
        or len(schedule.exposures) != PAIR_EXPOSURES
        or not 0 <= optimizer_step < UPDATES
        or len(update_pair_ids) != MICROSTEPS_PER_UPDATE
    ):
        raise ETTRILV2ReadinessError("pair schedule geometry differs")
    start = optimizer_step * MICROSTEPS_PER_UPDATE
    expected = schedule.exposures[start : start + MICROSTEPS_PER_UPDATE]
    if (
        tuple(value.pair_id for value in expected) != update_pair_ids
        or tuple(value.update for value in expected)
        != (optimizer_step,) * MICROSTEPS_PER_UPDATE
        or tuple(value.microstep for value in expected)
        != tuple(range(MICROSTEPS_PER_UPDATE))
    ):
        raise ETTRILV2ReadinessError(
            "next update does not match the deterministic pair cursor"
        )


def _validate_resume(
    resume: ResumeReadinessReceipt,
    *,
    manifest: ETTRContinuationManifest,
    schedule: PairSchedule,
    update_pair_ids: tuple[str, ...],
) -> None:
    progress = resume.progress
    stream = resume.data_stream
    lifecycle = resume.episode_lifecycle
    expected_exposure = progress.optimizer_step * MICROSTEPS_PER_UPDATE
    expected_sampler = {
        "next_pair_ids": list(update_pair_ids),
        "pair_exposure_index": expected_exposure,
        "schedule_sha256": schedule.schedule_sha256,
        "schema": RESUME_CURSOR_SCHEMA,
    }
    integer_progress = (
        progress.global_step,
        progress.optimizer_step,
        progress.micro_step,
        progress.gradient_accumulation_steps,
        progress.tokens_seen,
    )
    expected_epoch = schedule.exposures[expected_exposure].epoch
    if (
        any(type(value) is not int for value in integer_progress)
        or progress.global_step
        != PROTECTED_BASE_STEP + progress.optimizer_step
        or not 0 <= progress.optimizer_step < UPDATES
        or progress.micro_step != 0
        or progress.gradient_accumulation_steps != MICROSTEPS_PER_UPDATE
        or progress.tokens_seen
        != progress.optimizer_step * ENCODED_POSITIONS_PER_UPDATE
        or stream.manifest_sha256 != manifest.sha256()
        or stream.dataset_sha256 != manifest.dataset_sha256
        or stream.seed != schedule.seed
        or stream.generation != 0
        or stream.epoch != expected_epoch
        or stream.shard_index != 0
        or stream.sample_index != expected_exposure
        or stream.token_offset != 0
        or stream.sampler_state != expected_sampler
        or lifecycle.episode_index != expected_exposure
        or lifecycle.phase != "between_episodes"
        or lifecycle.episode_sha256 is not None
        or lifecycle.token_offset != 0
        or lifecycle.reactor_step != 0
        or lifecycle.source_deleted
        or lifecycle.committed
        or lifecycle.halted
    ):
        raise ETTRILV2ReadinessError(
            "deterministic resume cursor differs"
        )
    _validate_schedule(
        schedule,
        optimizer_step=progress.optimizer_step,
        update_pair_ids=update_pair_ids,
    )
    _require_sha256(resume.rng_state_sha256, "RNG receipt")
    global_before = tree_sha256(_capture_rng_state())
    try:
        _validate_rng_state(resume.rng_state)
    except Exception as exc:
        raise ETTRILV2ReadinessError("RNG state contract differs") from exc
    global_after = tree_sha256(_capture_rng_state())
    if (
        tree_sha256(resume.rng_state) != resume.rng_state_sha256
        or global_before != global_after
    ):
        raise ETTRILV2ReadinessError(
            "RNG receipt differs or validation mutated global RNG"
        )


def _batch_supervised_positions(batch: ETTRContinuationBatch) -> int:
    return sum(
        int(segment.supervised_tokens.item())
        for segment in (
            batch.episodes.world,
            batch.episodes.command,
            batch.episodes.query,
        )
    )


def _validate_batch_geometry(
    batch: ETTRContinuationBatch,
    *,
    manifest: ETTRContinuationManifest,
    reactor_config: TheoryReactorConfig,
    objective_config: ETTRObjectiveConfig,
    training: bool,
) -> int:
    if not isinstance(batch, ETTRContinuationBatch):
        raise ETTRILV2ReadinessError("continuation batch type differs")
    try:
        batch.validate(reactor_config, objective_config)
    except (TheoryReactorError, RuntimeError) as exc:
        raise ETTRILV2ReadinessError(
            "continuation batch validation failed"
        ) from exc
    rows = batch.episodes.world.tokens.shape[0]
    expected_rows = ROWS_PER_MICROSTEP if training else rows
    segments = (
        (batch.episodes.world, WORLD_WIDTH),
        (batch.episodes.command, COMMAND_WIDTH),
        (batch.episodes.query, QUERY_WIDTH),
    )
    if (
        rows != expected_rows
        or (not training and (rows < 16 or rows % 16 != 0))
        or batch.manifest_sha256 != manifest.sha256()
        or batch.dataset_sha256 != manifest.dataset_sha256
        or any(
            segment.tokens.shape != (rows, width)
            or segment.attention_mask.shape != (rows, width)
            or not bool(segment.attention_mask.all())
            for segment, width in segments
        )
        or batch.transaction_targets.opcode.shape
        != (rows, TRANSACTION_WIDTH)
        or batch.causal_rectangles.rows.shape
        != (
            (
                CAUSAL_RECTANGLES_PER_MICROSTEP
                if training
                else rows // 4
            ),
            2,
            2,
        )
        or batch.packet_targets.active.shape
        != (rows, reactor_config.num_slots)
        or batch.terminal_packet_targets.active.shape
        != (rows, reactor_config.num_slots)
        or batch.terminal_packet_targets.relations.shape
        != (
            rows,
            reactor_config.num_relations,
            reactor_config.num_slots,
            reactor_config.num_slots,
        )
    ):
        raise ETTRILV2ReadinessError(
            "continuation or evaluator input shape differs"
        )
    if training:
        alignment = batch.equivariance
        if (
            alignment is None
            or alignment.left_index.shape
            != (ALIGNMENT_PAIRS_PER_MICROSTEP,)
            or alignment.right_index.shape
            != (ALIGNMENT_PAIRS_PER_MICROSTEP,)
            or bool(alignment.left_index.eq(alignment.right_index).any())
        ):
            raise ETTRILV2ReadinessError(
                "training invariant-pair geometry differs"
            )
    return _batch_supervised_positions(batch)


def objective_weights_sha256(
    weights: ETTRObjectiveWeights | None = None,
) -> str:
    value = ETTRObjectiveWeights() if weights is None else weights
    return hashlib.sha256(
        _canonical_json_bytes(dict(value.items()), newline=False)
    ).hexdigest()


def _validate_derangement(
    value: BindingDerangement,
    *,
    fold: int,
) -> None:
    if not isinstance(value, BindingDerangement):
        raise ETTRILV2ReadinessError("binding derangement type differs")
    assignments = value.assignments
    recipient_ids = tuple(item.recipient_id for item in assignments)
    donor_ids = tuple(item.donor_id for item in assignments)
    payload_sha256 = hashlib.sha256(
        control_json_bytes([item.as_dict() for item in assignments])
    ).hexdigest()
    if (
        value.fold != fold
        or len(assignments) != BINDING_DERANGEMENT_ASSIGNMENTS
        or len(set(recipient_ids)) != len(assignments)
        or len(set(donor_ids)) != len(assignments)
        or any(left == right for left, right in zip(recipient_ids, donor_ids))
        or any(
            _SHA256.fullmatch(item.recipient_id) is None
            or _SHA256.fullmatch(item.donor_id) is None
            or _SHA256.fullmatch(item.donor_digest) is None
            or type(item.donor_rank) is not int
            or item.donor_rank < 0
            for item in assignments
        )
        or payload_sha256 != value.assignment_sha256
        or value.receipt()["fixed_points"] != 0
    ):
        raise ETTRILV2ReadinessError(
            "binding-deranged control receipt differs"
        )


def _validate_arms(
    arms: Sequence[ArmReadinessReceipt],
    *,
    manifest: ETTRContinuationManifest,
    schedule: PairSchedule,
    model_receipt: ETTRModelAssemblyReceipt,
    binding_derangement: BindingDerangement,
) -> None:
    values = tuple(arms)
    if (
        len(values) != len(PRIMARY_ARMS)
        or {value.arm for value in values} != set(PRIMARY_ARMS)
        or len({value.arm_config_sha256 for value in values}) != len(values)
    ):
        raise ETTRILV2ReadinessError(
            "primary arm inventory differs"
        )
    weights_sha256 = objective_weights_sha256()
    common = None
    for value in values:
        for digest_name in (
            "arm_config_sha256",
            "control_receipt_sha256",
            "manifest_sha256",
            "dataset_sha256",
            "schedule_sha256",
            "source_payload_sha256",
            "objective_weights_sha256",
        ):
            _require_sha256(getattr(value, digest_name), digest_name)
        expected_common = (
            value.manifest_sha256,
            value.dataset_sha256,
            value.schedule_sha256,
            value.source_payload_sha256,
            value.objective_weights_sha256,
            value.objective_families,
            value.microsteps_per_update,
            value.rows_per_microstep,
            value.rows_per_update,
            value.semantic_rectangles_per_update,
            value.causal_rectangles_per_update,
            value.encoded_positions_per_update,
            value.supervised_positions_per_update,
            value.updates,
            value.trainable_parameters,
            value.complete_system_parameters,
        )
        if common is None:
            common = expected_common
        elif expected_common != common:
            raise ETTRILV2ReadinessError(
                "matched arm budget or supervision differs"
            )
        if (
            value.mechanism != ARM_MECHANISMS[value.arm]
            or value.manifest_sha256 != manifest.sha256()
            or value.dataset_sha256 != manifest.dataset_sha256
            or value.schedule_sha256 != schedule.schedule_sha256
            or value.source_payload_sha256 != manifest.train_payload_sha256
            or value.objective_weights_sha256 != weights_sha256
            or value.objective_families != OBJECTIVE_FAMILIES
            or value.microsteps_per_update != MICROSTEPS_PER_UPDATE
            or value.rows_per_microstep != ROWS_PER_MICROSTEP
            or value.rows_per_update != ROWS_PER_UPDATE
            or value.semantic_rectangles_per_update
            != SEMANTIC_RECTANGLES_PER_UPDATE
            or value.causal_rectangles_per_update
            != CAUSAL_RECTANGLES_PER_UPDATE
            or value.encoded_positions_per_update
            != ENCODED_POSITIONS_PER_UPDATE
            or value.supervised_positions_per_update
            != SUPERVISED_POSITIONS_PER_UPDATE
            or value.updates != UPDATES
            or value.trainable_parameters != ARCHITECTURE_PARAMETERS
            or value.complete_system_parameters
            != model_receipt.total_parameters
        ):
            raise ETTRILV2ReadinessError(
                f"{value.arm} arm contract differs"
            )
    deranged = next(value for value in values if value.arm == "binding_deranged")
    if deranged.control_receipt_sha256 != binding_derangement.assignment_sha256:
        raise ETTRILV2ReadinessError(
            "binding-deranged arm is not bound to its control"
        )


def validate_readiness(
    request: ETTRILV2ReadinessRequest,
) -> ETTRILV2ReadinessReport:
    """Validate Phase-1 readiness without constructing a trainable object."""

    if not isinstance(request, ETTRILV2ReadinessRequest):
        raise ETTRILV2ReadinessError("readiness request type differs")
    manifest, manifest_artifact = _load_continuation_manifest(
        request.manifest_path,
        request.expected_manifest_sha256,
    )
    dataset_artifact = _load_dataset_index(
        request.dataset_path,
        request.expected_dataset_artifact_sha256,
        manifest=manifest,
    )
    model_receipt, reactor_config, config_sha256 = _validate_model_and_config(
        request,
        manifest,
    )
    _validate_resume(
        request.resume,
        manifest=manifest,
        schedule=request.schedule,
        update_pair_ids=request.update_pair_ids,
    )
    if len(request.update_batches) != MICROSTEPS_PER_UPDATE:
        raise ETTRILV2ReadinessError("update microbatch count differs")
    supervised = sum(
        _validate_batch_geometry(
            batch,
            manifest=manifest,
            reactor_config=reactor_config,
            objective_config=request.objective_config,
            training=True,
        )
        for batch in request.update_batches
    )
    try:
        request.packet_sufficiency.verify_train(request.update_batches)
    except TheoryReactorError as exc:
        raise ETTRILV2ReadinessError(
            "update batches are absent from the immutable dataset index"
        ) from exc
    if (
        request.packet_sufficiency.receipt
        != manifest.packet_sufficiency_receipt()
        or request.packet_sufficiency.train_payload_sha256
        != manifest.train_payload_sha256
        or request.packet_sufficiency.validation_payload_sha256
        != manifest.validation_payload_sha256
        or supervised != SUPERVISED_POSITIONS_PER_UPDATE
    ):
        raise ETTRILV2ReadinessError(
            "dataset identity or update supervision differs"
        )
    _validate_derangement(
        request.binding_derangement,
        fold=request.schedule.fold,
    )
    _validate_arms(
        request.arms,
        manifest=manifest,
        schedule=request.schedule,
        model_receipt=model_receipt,
        binding_derangement=request.binding_derangement,
    )
    _validate_batch_geometry(
        request.evaluator_batch,
        manifest=manifest,
        reactor_config=reactor_config,
        objective_config=request.objective_config,
        training=False,
    )
    evaluator_rows = request.evaluator_batch.episodes.query.tokens.shape[0]
    return ETTRILV2ReadinessReport(
        schema=READINESS_SCHEMA,
        protocol=PROTOCOL,
        status="pass",
        mode="validate_only",
        weight_updates=0,
        manifest=manifest_artifact,
        dataset=dataset_artifact,
        logical_dataset_sha256=manifest.dataset_sha256,
        model_assembly_receipt_sha256=model_receipt.sha256(),
        model_sha256=model_receipt.complete_model_sha256,
        reactor_config_sha256=config_sha256,
        schedule_sha256=request.schedule.schedule_sha256,
        rng_state_sha256=request.resume.rng_state_sha256,
        update_pair_ids=request.update_pair_ids,
        update_rows=sum(
            batch.episodes.world.tokens.shape[0]
            for batch in request.update_batches
        ),
        supervised_positions_per_update=supervised,
        encoded_positions_per_update=ENCODED_POSITIONS_PER_UPDATE,
        arm_count=len(request.arms),
        evaluator_rows=evaluator_rows,
        evaluator_query_shape=tuple(
            request.evaluator_batch.episodes.query.tokens.shape
        ),
        complete_system_parameters=model_receipt.total_parameters,
        remaining_under_cap=model_receipt.remaining_under_cap,
    )


__all__ = [
    "ARCHITECTURE_PARAMETERS",
    "ARM_MECHANISMS",
    "ArmReadinessReceipt",
    "BINDING_DERANGEMENT_ASSIGNMENTS",
    "COMPLETE_SYSTEM_PARAMETERS",
    "DATASET_INDEX_SCHEMA",
    "ENCODED_POSITIONS_PER_UPDATE",
    "ETTRILV2ReadinessError",
    "ETTRILV2ReadinessReport",
    "ETTRILV2ReadinessRequest",
    "OBJECTIVE_FAMILIES",
    "PRIMARY_ARMS",
    "PROTECTED_BASE_PARAMETERS",
    "READINESS_SCHEMA",
    "RESUME_CURSOR_SCHEMA",
    "ROWS_PER_UPDATE",
    "ResumeReadinessReceipt",
    "SUPERVISED_POSITIONS_PER_UPDATE",
    "objective_weights_sha256",
    "validate_readiness",
]
