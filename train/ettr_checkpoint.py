"""Fail-closed exact-resume checkpoints for the ETTR architecture.

An ETTR checkpoint is a new, no-replace artifact.  It binds a complete
continuation state to the immutable step-300k Shohin trust root without ever
writing to, replacing, or embedding that protected checkpoint as an output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.optim import Optimizer

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
    TheoryReactorError,
)
from model import GPT, GPTConfig
from ettr_optimization import ETTROptimizerBundle
from workspace_checkpoint import (
    PROTECTED_BASE_CONFIG,
    PROTECTED_BASE_PARAMETERS,
    PROTECTED_BASE_STATE_SHA256,
    PROTECTED_CHECKPOINT_BYTES,
    PROTECTED_CHECKPOINT_KEYS,
    PROTECTED_CHECKPOINT_SHA256,
    PROTECTED_CHECKPOINT_STEP,
    PROTECTED_CONFIG_SHA256,
    PROTECTED_DATA_SEED,
    PROTECTED_DATA_STREAM_GENERATION,
    PROTECTED_DATA_STREAM_SEED,
    PROTECTED_STATE_KEY_COUNT,
    PROTECTED_STATE_KEY_SHA256,
    file_sha256,
    json_sha256,
    state_dict_sha256,
)


ETTR_CHECKPOINT_SCHEMA = "shohin_ettr_exact_resume_v1"
RNG_SCHEMA = "shohin_rng_state_v1"
DATA_STREAM_SCHEMA = "shohin_ettr_data_stream_v1"
EPISODE_LIFECYCLE_SCHEMA = "shohin_ettr_episode_lifecycle_v1"
INTEGRITY_SCHEMA = "shohin_ettr_integrity_v1"

_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "protected_base",
        "runtime_source_manifest",
        "ettr_config",
        "training_progress",
        "model_state",
        "optimizer",
        "scheduler",
        "rng_state",
        "data_stream_state",
        "episode_lifecycle_state",
        "integrity",
    }
)
_OPTIMIZER_KEYS = frozenset({"contract", "state"})
_SCHEDULER_KEYS = frozenset({"contract", "state"})
_INTEGRITY_KEYS = frozenset(
    {
        "schema",
        "model_state_sha256",
        "optimizer_state_sha256",
        "scheduler_state_sha256",
        "rng_state_sha256",
        "data_stream_state_sha256",
        "episode_lifecycle_state_sha256",
        "model_state_key_sha256",
        "model_state_key_count",
    }
)
_EPISODE_PHASES = frozenset(
    {
        "between_episodes",
        "source",
        "source_deleted",
        "reactor",
        "query",
        "complete",
    }
)

_SOURCE_PATHS = {
    "endogenous_typed_theory_reactor.py": Path(__file__).with_name(
        "endogenous_typed_theory_reactor.py"
    ),
    "ettr_checkpoint.py": Path(__file__),
    "ettr_data_contract.py": Path(__file__).with_name("ettr_data_contract.py"),
    "ettr_episode.py": Path(__file__).with_name("ettr_episode.py"),
    "ettr_objectives.py": Path(__file__).with_name("ettr_objectives.py"),
    "ettr_optimization.py": Path(__file__).with_name("ettr_optimization.py"),
    "ettr_train_step.py": Path(__file__).with_name("ettr_train_step.py"),
    "model.py": Path(__file__).with_name("model.py"),
    "workspace_checkpoint.py": Path(__file__).with_name("workspace_checkpoint.py"),
}
_IMPORTED_SOURCE_MANIFEST = {
    name: file_sha256(path) for name, path in _SOURCE_PATHS.items()
}


class ETTRCheckpointError(ValueError):
    """An ETTR checkpoint or exact-resume invariant failed."""


@dataclass(frozen=True, slots=True)
class BaseProvenance:
    """Cryptographic identity of the immutable model-only trust root."""

    checkpoint_path: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    step: int
    data_seed: int
    data_stream_generation: int
    data_stream_seed: int
    base_config: dict[str, object]
    config_sha256: str
    base_state_sha256: str
    state_key_sha256: str
    state_key_count: int
    base_parameters: int


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """Counters whose loss would change continuation behavior."""

    global_step: int
    optimizer_step: int
    micro_step: int
    gradient_accumulation_steps: int
    tokens_seen: int


@dataclass(frozen=True, slots=True)
class DataStreamState:
    """Exact cursor and identity of the deterministic training stream."""

    manifest_sha256: str
    dataset_sha256: str
    generation: int
    seed: int
    epoch: int
    shard_index: int
    sample_index: int
    token_offset: int
    sampler_state: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EpisodeLifecycleState:
    """Resume boundary inside the ETTR source/delete/react/query lifecycle."""

    episode_index: int
    phase: str
    episode_sha256: str | None
    token_offset: int
    reactor_step: int
    source_deleted: bool
    committed: bool
    halted: bool


@dataclass(frozen=True, slots=True)
class ETTRResumeState:
    """Validated non-model state restored from one checkpoint."""

    checkpoint_sha256: str
    progress: TrainingProgress
    data_stream: DataStreamState
    episode_lifecycle: EpisodeLifecycleState


def runtime_source_manifest() -> dict[str, str]:
    """Bind continuation to the exact imported architecture implementation."""

    current = {name: file_sha256(path) for name, path in _SOURCE_PATHS.items()}
    if current != _IMPORTED_SOURCE_MANIFEST:
        raise ETTRCheckpointError(
            "ETTR runtime source files changed after process import"
        )
    return dict(_IMPORTED_SOURCE_MANIFEST)


def _load_protected_base(
    checkpoint_path: Path,
) -> tuple[GPT, BaseProvenance]:
    """Validate and load the immutable production step-300k base once."""

    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise ETTRCheckpointError(
            f"protected base is not a regular file: {checkpoint_path}"
        )
    if checkpoint_path.stat().st_size != PROTECTED_CHECKPOINT_BYTES:
        raise ETTRCheckpointError("protected base byte count differs")
    payload, digest = _torch_load_verified(
        checkpoint_path,
        expected_sha256=PROTECTED_CHECKPOINT_SHA256,
    )
    if not isinstance(payload, dict):
        raise ETTRCheckpointError("protected base is not a dictionary")
    _require_exact_keys(payload, PROTECTED_CHECKPOINT_KEYS, "protected base")
    expected_metadata = {
        "step": PROTECTED_CHECKPOINT_STEP,
        "data_seed": PROTECTED_DATA_SEED,
        "data_stream_generation": PROTECTED_DATA_STREAM_GENERATION,
        "data_stream_seed": PROTECTED_DATA_STREAM_SEED,
    }
    for name, expected in expected_metadata.items():
        if payload[name] != expected:
            raise ETTRCheckpointError(f"protected base {name} differs from trust root")
    if payload["cfg"] != PROTECTED_BASE_CONFIG:
        raise ETTRCheckpointError("protected base configuration differs")
    if json_sha256(payload["cfg"]) != PROTECTED_CONFIG_SHA256:
        raise ETTRCheckpointError("protected base configuration hash differs")
    model_state = payload["model"]
    if not isinstance(model_state, Mapping) or not model_state:
        raise ETTRCheckpointError("protected base model state is missing")
    _validate_tensor_tree(model_state, "protected base model state")
    if len(model_state) != PROTECTED_STATE_KEY_COUNT:
        raise ETTRCheckpointError("protected base state key count differs")
    if json_sha256(sorted(model_state)) != PROTECTED_STATE_KEY_SHA256:
        raise ETTRCheckpointError("protected base state key hash differs")
    if state_dict_sha256(model_state) != PROTECTED_BASE_STATE_SHA256:
        raise ETTRCheckpointError("protected base tensor hash differs")
    try:
        base = GPT(GPTConfig(**payload["cfg"]))
        incompatibility = base.load_state_dict(model_state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ETTRCheckpointError("protected base failed strict model loading") from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRCheckpointError(
            "protected base strict load returned incompatible keys"
        )
    if base.num_params() != PROTECTED_BASE_PARAMETERS:
        raise ETTRCheckpointError("protected base parameter count differs")
    provenance = BaseProvenance(
        checkpoint_path=str(checkpoint_path),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        checkpoint_sha256=digest,
        step=payload["step"],
        data_seed=payload["data_seed"],
        data_stream_generation=payload["data_stream_generation"],
        data_stream_seed=payload["data_stream_seed"],
        base_config=dict(payload["cfg"]),
        config_sha256=PROTECTED_CONFIG_SHA256,
        base_state_sha256=PROTECTED_BASE_STATE_SHA256,
        state_key_sha256=PROTECTED_STATE_KEY_SHA256,
        state_key_count=PROTECTED_STATE_KEY_COUNT,
        base_parameters=PROTECTED_BASE_PARAMETERS,
    )
    return base, provenance


def load_protected_base_provenance(
    checkpoint_path: Path,
) -> BaseProvenance:
    """Validate and receipt only the immutable production step-300k base."""

    _, provenance = _load_protected_base(checkpoint_path)
    return provenance


def load_protected_base_model(
    checkpoint_path: Path,
) -> tuple[GPT, BaseProvenance]:
    """Return the strict protected model and its cryptographic provenance."""

    return _load_protected_base(checkpoint_path)


def save_ettr_checkpoint(
    path: Path,
    *,
    model: EndogenousTypedTheoryReactorGPT,
    protected_base: BaseProvenance,
    optimizer: Optimizer | ETTROptimizerBundle,
    scheduler: Any | None,
    progress: TrainingProgress,
    data_stream: DataStreamState,
    episode_lifecycle: EpisodeLifecycleState,
) -> str:
    """Atomically publish a complete ETTR exact-resume checkpoint."""

    return _save_ettr_checkpoint(
        path,
        model=model,
        protected_base=protected_base,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        data_stream=data_stream,
        episode_lifecycle=episode_lifecycle,
        require_protected_constants=True,
    )


def load_ettr_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    model: EndogenousTypedTheoryReactorGPT,
    protected_base: BaseProvenance,
    optimizer: Optimizer | ETTROptimizerBundle,
    scheduler: Any | None,
) -> ETTRResumeState:
    """Validate every continuation contract, then restore exact state."""

    return _load_ettr_checkpoint(
        path,
        expected_sha256=expected_sha256,
        model=model,
        protected_base=protected_base,
        optimizer=optimizer,
        scheduler=scheduler,
        require_protected_constants=True,
    )


def _save_ettr_checkpoint_for_test(
    path: Path,
    *,
    model: EndogenousTypedTheoryReactorGPT,
    protected_base: BaseProvenance,
    optimizer: Optimizer | ETTROptimizerBundle,
    scheduler: Any | None,
    progress: TrainingProgress,
    data_stream: DataStreamState,
    episode_lifecycle: EpisodeLifecycleState,
) -> str:
    """Exercise the production wire against a synthetic trust root."""

    return _save_ettr_checkpoint(
        path,
        model=model,
        protected_base=protected_base,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        data_stream=data_stream,
        episode_lifecycle=episode_lifecycle,
        require_protected_constants=False,
    )


def _load_ettr_checkpoint_for_test(
    path: Path,
    *,
    expected_sha256: str,
    model: EndogenousTypedTheoryReactorGPT,
    protected_base: BaseProvenance,
    optimizer: Optimizer | ETTROptimizerBundle,
    scheduler: Any | None,
) -> ETTRResumeState:
    """Load the production wire against a synthetic trust root."""

    return _load_ettr_checkpoint(
        path,
        expected_sha256=expected_sha256,
        model=model,
        protected_base=protected_base,
        optimizer=optimizer,
        scheduler=scheduler,
        require_protected_constants=False,
    )


def _save_ettr_checkpoint(
    path: Path,
    *,
    model: EndogenousTypedTheoryReactorGPT,
    protected_base: BaseProvenance,
    optimizer: Optimizer | ETTROptimizerBundle,
    scheduler: Any | None,
    progress: TrainingProgress,
    data_stream: DataStreamState,
    episode_lifecycle: EpisodeLifecycleState,
    require_protected_constants: bool,
) -> str:
    path = path.resolve()
    _validate_destination(path, protected_base)
    _validate_base_provenance(
        protected_base,
        require_protected_constants=require_protected_constants,
    )
    base_digest_before = _verify_protected_base_file(protected_base)
    ettr_config = asdict(model.config)
    _validate_ettr_config(model, ettr_config)
    _validate_progress(progress, protected_base)
    _validate_data_stream(data_stream)
    _validate_episode_lifecycle(episode_lifecycle, model.config)
    model_state = _cpu_snapshot(model.state_dict())
    _validate_model_state(model_state, model)
    optimizer_contract = _optimizer_contract(optimizer, model)
    _validate_optimizer_progress(optimizer, progress)
    optimizer_state = _snapshot_tree(optimizer.state_dict())
    _validate_optimizer_state(
        optimizer_state,
        optimizer_contract,
        "optimizer state",
    )
    scheduler_contract = _scheduler_contract(
        scheduler,
        optimizer,
        optimizer_contract,
    )
    scheduler_state = _snapshot_tree(_scheduler_state(scheduler, optimizer))
    _validate_tree(scheduler_state, "scheduler state")
    rng_state = _capture_rng_state()
    data_payload = _dataclass_payload(
        data_stream,
        schema=DATA_STREAM_SCHEMA,
    )
    episode_payload = _dataclass_payload(
        episode_lifecycle,
        schema=EPISODE_LIFECYCLE_SCHEMA,
    )
    progress_payload = asdict(progress)
    integrity = _integrity_payload(
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        rng_state=rng_state,
        data_stream_state=data_payload,
        episode_lifecycle_state=episode_payload,
    )
    payload = {
        "schema": ETTR_CHECKPOINT_SCHEMA,
        "protected_base": asdict(protected_base),
        "runtime_source_manifest": runtime_source_manifest(),
        "ettr_config": ettr_config,
        "training_progress": progress_payload,
        "model_state": model_state,
        "optimizer": {
            "contract": optimizer_contract,
            "state": optimizer_state,
        },
        "scheduler": {
            "contract": scheduler_contract,
            "state": scheduler_state,
        },
        "rng_state": rng_state,
        "data_stream_state": data_payload,
        "episode_lifecycle_state": episode_payload,
        "integrity": integrity,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_noreplace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    base_digest_after = _verify_protected_base_file(protected_base)
    if base_digest_after != base_digest_before:
        raise ETTRCheckpointError(
            "protected base changed during ETTR checkpoint publication"
        )
    return file_sha256(path)


def _load_ettr_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    model: EndogenousTypedTheoryReactorGPT,
    protected_base: BaseProvenance,
    optimizer: Optimizer | ETTROptimizerBundle,
    scheduler: Any | None,
    require_protected_constants: bool,
) -> ETTRResumeState:
    path = path.resolve()
    _validate_base_provenance(
        protected_base,
        require_protected_constants=require_protected_constants,
    )
    base_digest_before = _verify_protected_base_file(protected_base)
    payload, actual_sha256 = _torch_load_verified(
        path,
        expected_sha256=expected_sha256,
    )
    if not isinstance(payload, dict):
        raise ETTRCheckpointError("ETTR checkpoint is not a dictionary")
    _require_exact_keys(payload, _CHECKPOINT_KEYS, "ETTR checkpoint")
    if payload["schema"] != ETTR_CHECKPOINT_SCHEMA:
        raise ETTRCheckpointError("ETTR checkpoint schema is invalid")
    if payload["runtime_source_manifest"] != runtime_source_manifest():
        raise ETTRCheckpointError("ETTR checkpoint runtime implementation differs")
    _validate_stored_base(payload["protected_base"], protected_base)
    expected_config = asdict(model.config)
    if payload["ettr_config"] != expected_config:
        raise ETTRCheckpointError("ETTR configuration differs")
    _validate_ettr_config(model, payload["ettr_config"])
    progress = _parse_dataclass(
        TrainingProgress,
        payload["training_progress"],
        "training progress",
    )
    _validate_progress(progress, protected_base)
    data_stream = _parse_schema_dataclass(
        DataStreamState,
        payload["data_stream_state"],
        DATA_STREAM_SCHEMA,
        "data stream",
    )
    _validate_data_stream(data_stream)
    episode_lifecycle = _parse_schema_dataclass(
        EpisodeLifecycleState,
        payload["episode_lifecycle_state"],
        EPISODE_LIFECYCLE_SCHEMA,
        "episode lifecycle",
    )
    _validate_episode_lifecycle(episode_lifecycle, model.config)

    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping):
        raise ETTRCheckpointError("ETTR model state is missing")
    _validate_model_state(model_state, model)
    optimizer_payload = payload["optimizer"]
    scheduler_payload = payload["scheduler"]
    if not isinstance(optimizer_payload, dict):
        raise ETTRCheckpointError("optimizer payload is invalid")
    if not isinstance(scheduler_payload, dict):
        raise ETTRCheckpointError("scheduler payload is invalid")
    _require_exact_keys(
        optimizer_payload,
        _OPTIMIZER_KEYS,
        "optimizer payload",
    )
    _require_exact_keys(
        scheduler_payload,
        _SCHEDULER_KEYS,
        "scheduler payload",
    )
    expected_optimizer_contract = _optimizer_contract(optimizer, model)
    if optimizer_payload["contract"] != expected_optimizer_contract:
        raise ETTRCheckpointError("optimizer contract differs")
    _validate_optimizer_progress_payload(
        optimizer_payload["state"],
        progress,
        optimizer,
    )
    _validate_optimizer_state(
        optimizer_payload["state"],
        expected_optimizer_contract,
        "optimizer state",
    )
    expected_scheduler_contract = _scheduler_contract(
        scheduler,
        optimizer,
        expected_optimizer_contract,
    )
    if scheduler_payload["contract"] != expected_scheduler_contract:
        raise ETTRCheckpointError("scheduler contract differs")
    current_scheduler_keys = frozenset(_scheduler_state(scheduler, optimizer))
    saved_scheduler = scheduler_payload["state"]
    if not isinstance(saved_scheduler, Mapping):
        raise ETTRCheckpointError("scheduler state is invalid")
    if frozenset(saved_scheduler) != current_scheduler_keys:
        raise ETTRCheckpointError("scheduler state keys differ")
    _validate_tree(saved_scheduler, "scheduler state")
    if isinstance(optimizer, ETTROptimizerBundle) and saved_scheduler != {
        "next_update": optimizer_payload["state"]["next_update"]
    }:
        raise ETTRCheckpointError("embedded ETTR scheduler cursor differs")
    _validate_rng_state(payload["rng_state"])
    _validate_integrity(payload)
    _validate_alias_consistency(model_state, model.state_dict())

    optimizer.load_state_dict(optimizer_payload["state"])
    if isinstance(optimizer, ETTROptimizerBundle):
        if scheduler is not None or saved_scheduler != {
            "next_update": optimizer.next_update
        }:
            raise ETTRCheckpointError("embedded ETTR scheduler state differs")
    else:
        scheduler.load_state_dict(saved_scheduler)
    target_state = model.state_dict()
    with torch.no_grad():
        for name, source in model_state.items():
            target_state[name].copy_(
                source.to(
                    device=target_state[name].device,
                    dtype=target_state[name].dtype,
                )
            )
    _restore_rng_state(payload["rng_state"])
    base_digest_after = _verify_protected_base_file(protected_base)
    if base_digest_after != base_digest_before:
        raise ETTRCheckpointError(
            "protected base changed during ETTR checkpoint restoration"
        )
    return ETTRResumeState(
        checkpoint_sha256=actual_sha256,
        progress=progress,
        data_stream=data_stream,
        episode_lifecycle=episode_lifecycle,
    )


def _validate_destination(
    destination: Path,
    protected_base: BaseProvenance,
) -> None:
    base_path = Path(protected_base.checkpoint_path).resolve()
    if destination == base_path:
        raise ETTRCheckpointError(
            "refusing to use the protected base as an ETTR destination"
        )
    if destination.exists():
        try:
            if os.path.samefile(destination, base_path):
                raise ETTRCheckpointError("ETTR destination aliases the protected base")
        except FileNotFoundError:
            pass
        raise FileExistsError(f"refusing to overwrite {destination}")


def _validate_base_provenance(
    provenance: BaseProvenance,
    *,
    require_protected_constants: bool,
) -> None:
    _require_hex_digest(provenance.checkpoint_sha256, "base checkpoint")
    _require_hex_digest(provenance.config_sha256, "base config")
    _require_hex_digest(provenance.base_state_sha256, "base state")
    _require_hex_digest(provenance.state_key_sha256, "base state keys")
    if provenance.checkpoint_bytes <= 0:
        raise ETTRCheckpointError("base checkpoint byte count is invalid")
    if provenance.step < 0 or provenance.base_parameters <= 0:
        raise ETTRCheckpointError("base step or parameter count is invalid")
    if not isinstance(provenance.base_config, dict):
        raise ETTRCheckpointError("base configuration is invalid")
    if json_sha256(provenance.base_config) != provenance.config_sha256:
        raise ETTRCheckpointError("base configuration hash is invalid")
    if require_protected_constants:
        expected = {
            "checkpoint_bytes": PROTECTED_CHECKPOINT_BYTES,
            "checkpoint_sha256": PROTECTED_CHECKPOINT_SHA256,
            "step": PROTECTED_CHECKPOINT_STEP,
            "data_seed": PROTECTED_DATA_SEED,
            "data_stream_generation": PROTECTED_DATA_STREAM_GENERATION,
            "data_stream_seed": PROTECTED_DATA_STREAM_SEED,
            "base_config": PROTECTED_BASE_CONFIG,
            "config_sha256": PROTECTED_CONFIG_SHA256,
            "base_state_sha256": PROTECTED_BASE_STATE_SHA256,
            "state_key_sha256": PROTECTED_STATE_KEY_SHA256,
            "state_key_count": PROTECTED_STATE_KEY_COUNT,
            "base_parameters": PROTECTED_BASE_PARAMETERS,
        }
        actual = asdict(provenance)
        differing = sorted(
            name
            for name, expected_value in expected.items()
            if actual[name] != expected_value
        )
        if differing:
            raise ETTRCheckpointError(
                "base provenance is not the protected trust root: "
                + ",".join(differing)
            )


def _verify_protected_base_file(provenance: BaseProvenance) -> str:
    path = Path(provenance.checkpoint_path).resolve()
    if not path.is_file():
        raise ETTRCheckpointError("protected base file is unavailable")
    if path.stat().st_size != provenance.checkpoint_bytes:
        raise ETTRCheckpointError("protected base file size changed")
    digest = file_sha256(path)
    if digest != provenance.checkpoint_sha256:
        raise ETTRCheckpointError("protected base file hash changed")
    return digest


def _portable_base(provenance: BaseProvenance) -> dict[str, object]:
    payload = asdict(provenance)
    payload.pop("checkpoint_path")
    return payload


def _validate_stored_base(
    stored: object,
    expected: BaseProvenance,
) -> None:
    if not isinstance(stored, dict):
        raise ETTRCheckpointError("stored base provenance is invalid")
    expected_keys = {field.name for field in fields(BaseProvenance)}
    _require_exact_keys(stored, frozenset(expected_keys), "base provenance")
    try:
        parsed = BaseProvenance(**stored)
    except TypeError as exc:
        raise ETTRCheckpointError("stored base provenance is invalid") from exc
    _validate_base_provenance(parsed, require_protected_constants=False)
    if _portable_base(parsed) != _portable_base(expected):
        raise ETTRCheckpointError("ETTR checkpoint references another protected base")


def _validate_ettr_config(
    model: EndogenousTypedTheoryReactorGPT,
    payload: object,
) -> None:
    if payload != asdict(model.config):
        raise ETTRCheckpointError("ETTR configuration is not exact")
    try:
        model.config.validate(n_layer=model.base.cfg.n_layer)
    except TheoryReactorError as exc:
        raise ETTRCheckpointError("ETTR configuration is invalid") from exc


def _validate_progress(
    progress: TrainingProgress,
    protected_base: BaseProvenance,
) -> None:
    values = (
        progress.global_step,
        progress.optimizer_step,
        progress.micro_step,
        progress.gradient_accumulation_steps,
        progress.tokens_seen,
    )
    if any(type(value) is not int for value in values):
        raise ETTRCheckpointError("training counters must be exact integers")
    if (
        progress.global_step < protected_base.step
        or progress.optimizer_step < 0
        or progress.tokens_seen < 0
        or progress.gradient_accumulation_steps <= 0
        or progress.micro_step != 0
    ):
        raise ETTRCheckpointError(
            "training progress is invalid; exact resume requires "
            "an optimizer boundary with micro_step=0"
        )


def _validate_optimizer_progress(
    optimizer: Optimizer | ETTROptimizerBundle,
    progress: TrainingProgress,
) -> None:
    if (
        isinstance(optimizer, ETTROptimizerBundle)
        and optimizer.next_update != progress.optimizer_step
    ):
        raise ETTRCheckpointError(
            "ETTR optimizer cursor differs from training progress"
        )


def _validate_optimizer_progress_payload(
    payload: object,
    progress: TrainingProgress,
    optimizer: Optimizer | ETTROptimizerBundle,
) -> None:
    if not isinstance(optimizer, ETTROptimizerBundle):
        return
    if (
        not isinstance(payload, Mapping)
        or payload.get("next_update") != progress.optimizer_step
    ):
        raise ETTRCheckpointError(
            "saved ETTR optimizer cursor differs from training progress"
        )


def _validate_data_stream(state: DataStreamState) -> None:
    _require_hex_digest(state.manifest_sha256, "stream manifest")
    _require_hex_digest(state.dataset_sha256, "stream dataset")
    counters = (
        state.generation,
        state.seed,
        state.epoch,
        state.shard_index,
        state.sample_index,
        state.token_offset,
    )
    if any(type(value) is not int or value < 0 for value in counters):
        raise ETTRCheckpointError("data-stream counters are invalid")
    if not isinstance(state.sampler_state, Mapping):
        raise ETTRCheckpointError("sampler state is invalid")
    _validate_tree(state.sampler_state, "sampler state")


def _validate_episode_lifecycle(
    state: EpisodeLifecycleState,
    config: TheoryReactorConfig,
) -> None:
    del config
    integer_values = (
        state.episode_index,
        state.token_offset,
        state.reactor_step,
    )
    if any(type(value) is not int or value < 0 for value in integer_values):
        raise ETTRCheckpointError("episode lifecycle counters are invalid")
    if state.phase not in _EPISODE_PHASES:
        raise ETTRCheckpointError("episode lifecycle phase is invalid")
    if state.phase != "between_episodes":
        raise ETTRCheckpointError(
            "exact resume currently permits only a between-episodes boundary"
        )
    expected = (
        state.episode_sha256 is None
        and state.token_offset == 0
        and state.reactor_step == 0
        and not state.source_deleted
        and not state.committed
        and not state.halted
    )
    if not expected:
        raise ETTRCheckpointError("between-episode lifecycle state is inconsistent")


def _optimizer_contract(
    optimizer: Optimizer | ETTROptimizerBundle,
    model: EndogenousTypedTheoryReactorGPT,
) -> dict[str, object]:
    if isinstance(optimizer, ETTROptimizerBundle):
        return {
            "kind": "shohin-ettr-optimizer-bundle-v1",
            "class_module": type(optimizer).__module__,
            "class_qualname": type(optimizer).__qualname__,
            "config": asdict(optimizer.config),
            "receipt": asdict(optimizer.receipt),
            "muon": (
                None
                if optimizer.muon is None
                else _standard_optimizer_contract(
                    optimizer.muon,
                    model,
                )
            ),
            "adam": _standard_optimizer_contract(
                optimizer.adam,
                model,
            ),
        }
    return _standard_optimizer_contract(optimizer, model)


def _standard_optimizer_contract(
    optimizer: Optimizer,
    model: EndogenousTypedTheoryReactorGPT,
) -> dict[str, object]:
    named = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[dict[str, object]] = []
    observed: set[int] = set()
    for index, group in enumerate(optimizer.param_groups):
        names: list[str] = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in named:
                raise ETTRCheckpointError(
                    f"optimizer group {index} contains a foreign parameter"
                )
            if identity in observed:
                raise ETTRCheckpointError(
                    "optimizer contains a duplicate model parameter"
                )
            observed.add(identity)
            names.append(named[identity])
        options = _optimizer_group_contract_options(
            group,
            "optimizer",
        )
        groups.append({"parameter_names": names, "options": options})
    return {
        "kind": "torch-optimizer-v1",
        "class_module": type(optimizer).__module__,
        "class_qualname": type(optimizer).__qualname__,
        "groups": groups,
    }


def _scheduler_contract(
    scheduler: Any,
    optimizer: Optimizer | ETTROptimizerBundle,
    optimizer_contract: Mapping[str, object],
) -> dict[str, object]:
    if isinstance(optimizer, ETTROptimizerBundle):
        if scheduler is not None:
            raise ETTRCheckpointError("native ETTR optimizer embeds its scheduler")
        return {
            "class_module": type(optimizer).__module__,
            "class_qualname": "ETTROptimizerBundle.embedded_schedule",
            "state_keys": ["next_update"],
            "optimizer_contract_sha256": tree_sha256(optimizer_contract),
        }
    if not hasattr(scheduler, "state_dict") or not hasattr(
        scheduler, "load_state_dict"
    ):
        raise ETTRCheckpointError("scheduler lacks checkpoint methods")
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise ETTRCheckpointError("scheduler is not bound to the supplied optimizer")
    state = scheduler.state_dict()
    if not isinstance(state, Mapping):
        raise ETTRCheckpointError("scheduler state is not a mapping")
    return {
        "class_module": type(scheduler).__module__,
        "class_qualname": type(scheduler).__qualname__,
        "state_keys": sorted(state),
        "optimizer_contract_sha256": tree_sha256(optimizer_contract),
    }


def _scheduler_state(
    scheduler: Any | None,
    optimizer: Optimizer | ETTROptimizerBundle,
) -> Mapping[str, Any]:
    if isinstance(optimizer, ETTROptimizerBundle):
        if scheduler is not None:
            raise ETTRCheckpointError("native ETTR optimizer embeds its scheduler")
        return {"next_update": optimizer.next_update}
    if scheduler is None:
        raise ETTRCheckpointError("scheduler is missing")
    return scheduler.state_dict()


def _validate_optimizer_state(
    state: object,
    contract: Mapping[str, object],
    label: str,
) -> None:
    if contract.get("kind") == "shohin-ettr-optimizer-bundle-v1":
        if not isinstance(state, Mapping):
            raise ETTRCheckpointError(f"{label} is not a mapping")
        _require_exact_keys(
            state,
            frozenset(
                {
                    "schema",
                    "config",
                    "receipt",
                    "next_update",
                    "muon",
                    "adam",
                }
            ),
            label,
        )
        if (
            state["schema"] != "shohin-ettr-optimizer-v1"
            or state["config"] != contract["config"]
            or state["receipt"] != contract["receipt"]
            or type(state["next_update"]) is not int
            or state["next_update"] < 0
        ):
            raise ETTRCheckpointError(f"{label} bundle contract differs")
        muon_contract = contract["muon"]
        if (state["muon"] is None) != (muon_contract is None):
            raise ETTRCheckpointError(f"{label} Muon presence differs")
        if muon_contract is not None:
            _validate_optimizer_state(
                state["muon"],
                muon_contract,
                f"{label} Muon",
            )
        _validate_optimizer_state(
            state["adam"],
            contract["adam"],
            f"{label} AdamW",
        )
        _validate_tree(state, label)
        return
    if not isinstance(state, Mapping):
        raise ETTRCheckpointError(f"{label} is not a mapping")
    _require_exact_keys(state, frozenset({"state", "param_groups"}), label)
    groups = state["param_groups"]
    slots = state["state"]
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise ETTRCheckpointError(f"{label} parameter groups are invalid")
    if not isinstance(slots, Mapping):
        raise ETTRCheckpointError(f"{label} slots are invalid")
    expected_groups = contract["groups"]
    if len(groups) != len(expected_groups):
        raise ETTRCheckpointError(f"{label} group count differs")
    parameter_ids: set[int] = set()
    for saved_group, expected_group in zip(groups, expected_groups):
        if not isinstance(saved_group, Mapping):
            raise ETTRCheckpointError(f"{label} group is invalid")
        params = saved_group.get("params")
        if not isinstance(params, list):
            raise ETTRCheckpointError(f"{label} parameter IDs are invalid")
        if len(params) != len(expected_group["parameter_names"]):
            raise ETTRCheckpointError(f"{label} parameter count differs")
        if any(type(value) is not int for value in params):
            raise ETTRCheckpointError(f"{label} parameter ID is invalid")
        parameter_ids.update(params)
        saved_options = _optimizer_group_contract_options(
            saved_group,
            label,
        )
        if saved_options != expected_group["options"]:
            raise ETTRCheckpointError(f"{label} options differ")
    if any(
        type(parameter_id) is not int or parameter_id not in parameter_ids
        for parameter_id in slots
    ):
        raise ETTRCheckpointError(f"{label} contains a foreign state slot")
    _validate_tree(state, label)


def _optimizer_group_contract_options(
    group: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    options = {
        key: _contract_value(value, f"{label} option {key}")
        for key, value in group.items()
        if key != "params"
    }
    if "initial_lr" in options:
        options["lr"] = options["initial_lr"]
    return options


def _contract_value(value: Any, label: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ETTRCheckpointError(f"{label} tensor is not scalar")
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ETTRCheckpointError(f"{label} is nonfinite")
        return value
    if isinstance(value, tuple):
        return [_contract_value(item, label) for item in value]
    if isinstance(value, list):
        return [_contract_value(item, label) for item in value]
    raise ETTRCheckpointError(f"{label} has unsupported type")


def _capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "schema": RNG_SCHEMA,
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        "torch_cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
    }


def _validate_rng_state(state: object) -> None:
    if not isinstance(state, Mapping):
        raise ETTRCheckpointError("RNG state is invalid")
    expected_keys = frozenset(
        {
            "schema",
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
            "torch_cuda_device_count",
        }
    )
    _require_exact_keys(state, expected_keys, "RNG state")
    if state["schema"] != RNG_SCHEMA:
        raise ETTRCheckpointError("RNG schema is invalid")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise ETTRCheckpointError("NumPy RNG state is invalid")
    _require_exact_keys(
        numpy_state,
        frozenset(
            {
                "bit_generator",
                "keys",
                "position",
                "has_gauss",
                "cached_gaussian",
            }
        ),
        "NumPy RNG state",
    )
    _validate_tree(state, "RNG state")
    if (
        not isinstance(state["torch_cpu"], torch.Tensor)
        or state["torch_cpu"].dtype != torch.uint8
        or state["torch_cpu"].ndim != 1
    ):
        raise ETTRCheckpointError("torch CPU RNG state is invalid")
    cuda_states = state["torch_cuda"]
    device_count = state["torch_cuda_device_count"]
    if type(device_count) is not int or device_count < 0:
        raise ETTRCheckpointError("CUDA RNG device count is invalid")
    if not isinstance(cuda_states, list) or len(cuda_states) != device_count:
        raise ETTRCheckpointError("CUDA RNG state count differs")
    if device_count != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ETTRCheckpointError(
            "CUDA device count differs from exact-resume checkpoint"
        )
    if any(
        not isinstance(item, torch.Tensor)
        or item.dtype != torch.uint8
        or item.ndim != 1
        for item in cuda_states
    ):
        raise ETTRCheckpointError("CUDA RNG tensor is invalid")
    try:
        probe = random.Random()
        probe.setstate(state["python"])
        np.random.RandomState().set_state(
            (
                numpy_state["bit_generator"],
                numpy_state["keys"].cpu().numpy(),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
    except (TypeError, ValueError) as exc:
        raise ETTRCheckpointError("RNG state failed validation") from exc


def _restore_rng_state(state: Mapping[str, object]) -> None:
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _integrity_payload(
    *,
    model_state: Mapping[str, torch.Tensor],
    optimizer_state: object,
    scheduler_state: object,
    rng_state: object,
    data_stream_state: object,
    episode_lifecycle_state: object,
) -> dict[str, object]:
    return {
        "schema": INTEGRITY_SCHEMA,
        "model_state_sha256": _model_state_sha256(model_state),
        "optimizer_state_sha256": tree_sha256(optimizer_state),
        "scheduler_state_sha256": tree_sha256(scheduler_state),
        "rng_state_sha256": tree_sha256(rng_state),
        "data_stream_state_sha256": tree_sha256(data_stream_state),
        "episode_lifecycle_state_sha256": tree_sha256(episode_lifecycle_state),
        "model_state_key_sha256": json_sha256(sorted(model_state)),
        "model_state_key_count": len(model_state),
    }


def _validate_integrity(payload: Mapping[str, object]) -> None:
    integrity = payload["integrity"]
    if not isinstance(integrity, Mapping):
        raise ETTRCheckpointError("checkpoint integrity payload is invalid")
    _require_exact_keys(integrity, _INTEGRITY_KEYS, "checkpoint integrity")
    if integrity["schema"] != INTEGRITY_SCHEMA:
        raise ETTRCheckpointError("checkpoint integrity schema is invalid")
    expected = _integrity_payload(
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer"]["state"],
        scheduler_state=payload["scheduler"]["state"],
        rng_state=payload["rng_state"],
        data_stream_state=payload["data_stream_state"],
        episode_lifecycle_state=payload["episode_lifecycle_state"],
    )
    if integrity != expected:
        raise ETTRCheckpointError("checkpoint section integrity differs")


def _validate_model_state(
    state: Mapping[str, torch.Tensor],
    model: EndogenousTypedTheoryReactorGPT,
) -> None:
    target = model.state_dict()
    if frozenset(state) != frozenset(target):
        missing = sorted(set(target) - set(state))
        unexpected = sorted(set(state) - set(target))
        raise ETTRCheckpointError(
            f"ETTR model state keys differ; missing={missing}, unexpected={unexpected}"
        )
    _validate_tensor_tree(state, "ETTR model state")
    for name, tensor in state.items():
        expected = target[name]
        if tensor.shape != expected.shape:
            raise ETTRCheckpointError(f"ETTR model state shape differs for {name}")
        if tensor.dtype != expected.dtype:
            raise ETTRCheckpointError(f"ETTR model state dtype differs for {name}")


def _validate_alias_consistency(
    state: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> None:
    aliases: dict[tuple[object, ...], list[str]] = {}
    for name, tensor in target.items():
        key = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        aliases.setdefault(key, []).append(name)
    for names in aliases.values():
        reference = state[names[0]]
        if any(not torch.equal(reference, state[name]) for name in names[1:]):
            raise ETTRCheckpointError(
                "ETTR model state disagrees across tied parameters"
            )


def _cpu_snapshot(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu", copy=True).contiguous()
        for name, tensor in state.items()
    }


def _snapshot_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True).contiguous()
    if isinstance(value, Mapping):
        return {key: _snapshot_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_tree(item) for item in value)
    return value


def _validate_tensor_tree(
    state: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ETTRCheckpointError(f"{label} entry is invalid")
        if tensor.layout != torch.strided:
            raise ETTRCheckpointError(f"{label} {name} is not strided")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ETTRCheckpointError(f"{label} {name} contains nonfinite values")


def _validate_tree(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ETTRCheckpointError(f"{label} tensor is not strided")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ETTRCheckpointError(f"{label} contains nonfinite values")
        return
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ETTRCheckpointError(f"{label} contains nonfinite values")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise ETTRCheckpointError(f"{label} contains an unsupported key")
            _validate_tree(item, label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_tree(item, label)
        return
    raise ETTRCheckpointError(f"{label} contains an unsupported value")


def tree_sha256(value: Any) -> str:
    """Hash a safe nested tensor tree independently of torch serialization."""

    digest = hashlib.sha256()
    _update_tree_hash(digest, value)
    return digest.hexdigest()


def _model_state_sha256(
    state: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to(device="cpu").contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _update_bytes(digest, metadata)
        _update_bytes(
            digest,
            memoryview(tensor.reshape(-1).view(torch.uint8).numpy()),
        )
    return digest.hexdigest()


def _update_tree_hash(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        metadata = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(b"T")
        _update_bytes(digest, metadata)
        _update_bytes(
            digest,
            memoryview(tensor.reshape(-1).view(torch.uint8).numpy()),
        )
        return
    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return
    if isinstance(value, int):
        digest.update(b"I")
        _update_bytes(digest, str(value).encode("ascii"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ETTRCheckpointError("cannot hash a nonfinite value")
        digest.update(b"F")
        _update_bytes(digest, value.hex().encode("ascii"))
        return
    if isinstance(value, str):
        digest.update(b"S")
        _update_bytes(digest, value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        digest.update(b"Y")
        _update_bytes(digest, value)
        return
    if isinstance(value, Mapping):
        digest.update(b"D")
        ordered = sorted(
            value.items(),
            key=lambda item: (
                type(item[0]).__name__,
                repr(item[0]),
            ),
        )
        for key, item in ordered:
            _update_tree_hash(digest, key)
            _update_tree_hash(digest, item)
        digest.update(b"d")
        return
    if isinstance(value, list):
        digest.update(b"L")
        for item in value:
            _update_tree_hash(digest, item)
        digest.update(b"l")
        return
    if isinstance(value, tuple):
        digest.update(b"Q")
        for item in value:
            _update_tree_hash(digest, item)
        digest.update(b"q")
        return
    raise ETTRCheckpointError("cannot hash unsupported checkpoint value")


def _update_bytes(digest: Any, value: Any) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _dataclass_payload(value: Any, *, schema: str) -> dict[str, object]:
    return {"schema": schema, **_snapshot_tree(asdict(value))}


def _parse_dataclass(
    cls: Any,
    payload: object,
    label: str,
) -> Any:
    if not isinstance(payload, dict):
        raise ETTRCheckpointError(f"{label} is not a dictionary")
    expected = frozenset(field.name for field in fields(cls))
    _require_exact_keys(payload, expected, label)
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ETTRCheckpointError(f"{label} is invalid") from exc


def _parse_schema_dataclass(
    cls: Any,
    payload: object,
    schema: str,
    label: str,
) -> Any:
    if not isinstance(payload, dict):
        raise ETTRCheckpointError(f"{label} is not a dictionary")
    expected = frozenset({"schema", *(field.name for field in fields(cls))})
    _require_exact_keys(payload, expected, label)
    if payload["schema"] != schema:
        raise ETTRCheckpointError(f"{label} schema is invalid")
    values = dict(payload)
    values.pop("schema")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ETTRCheckpointError(f"{label} is invalid") from exc


def _require_exact_keys(
    payload: Mapping[Any, Any],
    expected: frozenset[Any],
    label: str,
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected - actual, key=str)
        unexpected = sorted(actual - expected, key=str)
        raise ETTRCheckpointError(
            f"{label} keys differ; missing={missing}, unexpected={unexpected}"
        )


def _require_hex_digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ETTRCheckpointError(f"{label} SHA-256 is invalid")


def _torch_load_verified(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[object, str]:
    _require_hex_digest(expected_sha256, "expected checkpoint")
    if not path.is_file():
        raise ETTRCheckpointError(f"checkpoint is unavailable: {path}")
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ETTRCheckpointError(f"ETTR checkpoint file hash mismatch: {actual}")
        handle.seek(0)
        try:
            payload = torch.load(
                handle,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ETTRCheckpointError("ETTR checkpoint deserialization failed") from exc
    return payload, actual


def _publish_noreplace(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BaseProvenance",
    "DataStreamState",
    "ETTRCheckpointError",
    "ETTRResumeState",
    "EpisodeLifecycleState",
    "TrainingProgress",
    "load_ettr_checkpoint",
    "load_protected_base_model",
    "load_protected_base_provenance",
    "runtime_source_manifest",
    "save_ettr_checkpoint",
    "tree_sha256",
]
