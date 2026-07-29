#!/usr/bin/env python3
"""Isolated BF16 profiler for the ETTR factorial-intervention objective.

This program is not a trainer. It uses deterministic synthetic token tensors,
runs a bounded number of in-memory optimizer microsteps, writes one JSON
receipt, and never writes model or optimizer state. H100 mode may read a
hash-bound base checkpoint, but re-hashes it after profiling and refuses any
output directory that aliases the checkpoint or a protected repository path.
Matched eager and ``torch.compile`` arms start from identical model
initialization and consume identical immutable 2x2 causal rectangles. Every
update executes factual WORLD/COMMAND/QUERY episodes, both intervention arms,
the complete composite objective with hard transactions, backward, and one
``ETTROptimizerBundle`` step.

Synchronization policy
----------------------
The Python profiling loop contains no ``Tensor.item()`` calls. CUDA events are
recorded asynchronously. Explicit CUDA host synchronization occurs only:

1. after warmup, before peak-memory reset and measured event recording;
2. once after all measured updates, before timing/gradient receipt extraction;
3. between eager and compiled arms before allocator cleanup.

The intervention runner retains its current batch-validation path. Its tensor
assertions are asynchronous on CUDA and therefore do not add a host sync.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Iterable, Mapping, Sequence

import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
    TypedTheoryState,
)
from ettr_data_contract import (
    ETTRCausalRectangle,
    ETTRContinuationBatch,
    terminal_packet_sufficiency_receipt,
)
from ettr_episode import (
    CausalETTREpisodeRunner,
    ETTREpisodeBatch,
    ETTREpisodeSegment,
)
from ettr_objectives import (
    ETTRCompositeObjective,
    ETTRObjectiveConfig,
    ETTRObjectiveWeights,
    ETTRPacketTargets,
    ETTRTransactionTargets,
)
from ettr_optimization import (
    ETTROptimizerBundle,
    ETTROptimizerConfig,
)
from model import GPT, GPTConfig


SCHEMA = "shohin-ettr-h100-profile-v5"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTECTED_PATHS = (REPOSITORY_ROOT / "train" / "flagship_out",)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SYNC_POINTS = (
    "cuda_after_warmup_before_measurement",
    "cuda_after_all_measured_updates_before_receipt",
    "cuda_between_arms_before_allocator_cleanup",
)
OBJECTIVE_LOSS_NAMES = (
    "total",
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


class ETTRProfileError(ValueError):
    """A profiling, custody, or device contract failed."""


@dataclass(frozen=True, slots=True)
class ProfileSettings:
    mode: str
    batch_size: int
    microsteps: int
    warmup_updates: int
    measured_updates: int
    world_tokens: int
    command_tokens: int
    query_tokens: int
    reactor_steps: int
    learning_rate: float
    seed: int
    train_scope: str
    compile_mode: str

    def validate(self) -> None:
        if self.mode not in {"dry-run", "cpu-validation", "h100"}:
            raise ETTRProfileError("profile mode differs")
        integer_values = (
            self.batch_size,
            self.microsteps,
            self.measured_updates,
            self.world_tokens,
            self.command_tokens,
            self.query_tokens,
            self.reactor_steps,
        )
        if any(value <= 0 for value in integer_values):
            raise ETTRProfileError("profile dimensions must be positive")
        if self.batch_size < 4 or self.batch_size % 4:
            raise ETTRProfileError(
                "batch size must be at least four and divisible by four"
            )
        if self.warmup_updates < 0:
            raise ETTRProfileError("warmup updates must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ETTRProfileError("learning rate must be finite and positive")
        if not 0 <= self.seed < 2**63:
            raise ETTRProfileError("profile seed differs")
        if self.train_scope not in {"architecture", "all"}:
            raise ETTRProfileError("train scope differs")
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
        }:
            raise ETTRProfileError("compile mode differs")
        if self.mode == "cpu-validation" and (
            self.batch_size > 4
            or self.microsteps > 2
            or self.warmup_updates > 1
            or self.measured_updates > 2
            or self.world_tokens > 16
            or self.command_tokens > 16
            or self.query_tokens > 16
            or self.reactor_steps > 2
        ):
            raise ETTRProfileError(
                "CPU validation geometry exceeds the bounded smoke contract"
            )


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    path: str
    sha256_before: str
    sha256_after: str
    bytes: int
    step: int
    strict_state_load: bool
    unchanged_after_profile: bool
    opened_read_only: bool


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_physical_parent(path: Path) -> None:
    if not path.is_absolute():
        raise ETTRProfileError("output directory must be absolute")
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        raise ETTRProfileError("output parent must be an existing directory")
    current = Path(path.anchor)
    for part in parent.parts[1:]:
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ETTRProfileError("output parent may not contain symlinks")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ETTRProfileError("output parent chain differs")


def validate_output_directory(
    output_dir: Path,
    *,
    protected_paths: Sequence[Path],
) -> Path:
    _require_physical_parent(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ETTRProfileError("refusing existing output directory")
    resolved = output_dir.resolve(strict=False)
    for protected in protected_paths:
        protected_resolved = protected.resolve(strict=False)
        if (
            resolved == protected_resolved
            or _is_relative_to(resolved, protected_resolved)
            or _is_relative_to(protected_resolved, resolved)
        ):
            raise ETTRProfileError("output directory aliases a protected path")
    return resolved


def reserve_output_directory(
    output_dir: Path,
    *,
    protected_paths: Sequence[Path],
) -> Path:
    resolved = validate_output_directory(
        output_dir,
        protected_paths=protected_paths,
    )
    resolved.mkdir(mode=0o700)
    return resolved


def write_report_once(output_dir: Path, report: Mapping[str, object]) -> Path:
    path = output_dir / "report.json"
    payload = canonical_json_bytes(report)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise ETTRProfileError("profile report already exists") from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o400)
    output_dir.chmod(0o500)
    return path


def _verify_checkpoint_path(path: Path) -> os.stat_result:
    if not path.is_absolute():
        raise ETTRProfileError("checkpoint path must be absolute")
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ETTRProfileError("checkpoint must be a physical regular file")
    return metadata


def load_checkpoint_read_only(
    path: Path,
    *,
    expected_sha256: str,
    expected_step: int,
) -> tuple[dict[str, object], os.stat_result]:
    """Load one hash-bound checkpoint through a read-only, no-follow fd."""

    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise ETTRProfileError("checkpoint SHA-256 differs")
    initial = _verify_checkpoint_path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != initial.st_dev
            or opened.st_ino != initial.st_ino
            or opened.st_size != initial.st_size
        ):
            raise ETTRProfileError("checkpoint identity changed before load")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for chunk in iter(
                lambda: handle.read(8 * 1024 * 1024),
                b"",
            ):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ETTRProfileError("checkpoint hash differs")
            handle.seek(0)
            payload = torch.load(
                handle,
                map_location="cpu",
                weights_only=False,
            )
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        final.st_dev != opened.st_dev
        or final.st_ino != opened.st_ino
        or final.st_size != opened.st_size
        or final.st_mtime_ns != opened.st_mtime_ns
    ):
        raise ETTRProfileError("checkpoint changed during read")
    if not isinstance(payload, dict):
        raise ETTRProfileError("checkpoint payload differs")
    if payload.get("step") != expected_step:
        raise ETTRProfileError("checkpoint step differs")
    if not isinstance(payload.get("cfg"), dict):
        raise ETTRProfileError("checkpoint config differs")
    if not isinstance(payload.get("model"), Mapping):
        raise ETTRProfileError("checkpoint model state differs")
    return payload, opened


def synthetic_batches(
    settings: ProfileSettings,
    *,
    reactor_config: TheoryReactorConfig,
    objective_config: ETTRObjectiveConfig,
) -> tuple[tuple[ETTRContinuationBatch, ...], str]:
    """Build deterministic factual 2x2 rectangles with no authored CF labels."""

    vocab_size = objective_config.vocab_size
    if vocab_size < 8:
        raise ETTRProfileError("synthetic profile vocabulary is too small")
    if (
        reactor_config.num_value_codes < 4
        or reactor_config.num_slots < 1
        or objective_config.num_slots != reactor_config.num_slots
        or objective_config.num_types != reactor_config.num_types
        or objective_config.num_relations != reactor_config.num_relations
        or objective_config.num_value_codes != reactor_config.num_value_codes
    ):
        raise ETTRProfileError("synthetic profile objective geometry differs")
    batches: list[ETTRContinuationBatch] = []
    digest_payload: list[dict[str, object]] = []
    for microstep in range(settings.microsteps):
        offset = settings.seed + 104_729 * microstep
        row = torch.arange(settings.batch_size, dtype=torch.long)
        rectangle = torch.div(row, 4, rounding_mode="floor")
        within = row.remainder(4)
        world_factor = torch.div(within, 2, rounding_mode="floor")
        command_factor = within.remainder(2)

        def factor_tokens(
            length: int,
            factor: torch.Tensor,
            rendering: torch.Tensor,
            *,
            multiplier: int,
            bias: int,
        ) -> torch.Tensor:
            positions = torch.arange(length, dtype=torch.long)[None, :]
            values = (
                positions * multiplier
                + factor[:, None] * (multiplier + 12)
                + rendering[:, None] * (multiplier + 7)
                + offset
                + bias
            ).remainder(vocab_size)
            return values

        world_key = rectangle * 2 + world_factor
        command_key = rectangle * 2 + command_factor
        world = factor_tokens(
            settings.world_tokens,
            world_key,
            within,
            multiplier=17,
            bias=11,
        )
        command = factor_tokens(
            settings.command_tokens,
            command_key,
            within.roll(1),
            multiplier=29,
            bias=23,
        )
        query = factor_tokens(
            settings.query_tokens,
            rectangle * 4 + within,
            within.flip(0),
            multiplier=37,
            bias=31,
        )
        world[:, 0] = 1 + world_factor
        command[:, 0] = 3 + command_factor
        query_read_index = torch.zeros(
            settings.batch_size,
            dtype=torch.long,
        )
        query[:, 0] = (
            rectangle + offset + 41
        ).remainder(vocab_size)
        query[:, 1] = 4 + within
        episode_ids = tuple(
            hashlib.sha256(
                f"synthetic-{settings.seed}-{microstep}-{row}".encode("ascii")
            ).hexdigest()
            for row in range(settings.batch_size)
        )
        episodes = ETTREpisodeBatch(
            episode_ids=episode_ids,
            reset_mask=torch.ones(
                settings.batch_size,
                dtype=torch.bool,
            ),
            query_read_index=query_read_index,
            world=ETTREpisodeSegment.from_tokens(world),
            command=ETTREpisodeSegment.from_tokens(command),
            query=ETTREpisodeSegment.from_tokens(query),
        )
        active = torch.zeros(
            settings.batch_size,
            reactor_config.num_slots,
            dtype=torch.bool,
        )
        active[:, 0] = True
        root = active.clone()
        slot_mask = torch.ones_like(active)
        relations = torch.zeros(
            settings.batch_size,
            reactor_config.num_relations,
            reactor_config.num_slots,
            reactor_config.num_slots,
            dtype=torch.bool,
        )
        relation_mask = torch.ones_like(relations)
        initial_values = torch.zeros_like(active, dtype=torch.long)
        initial_values[:, 0] = world_key.remainder(
            reactor_config.num_value_codes
        )
        initial_types = torch.zeros_like(active, dtype=torch.long)
        initial_types[:, 0] = world_factor.remainder(
            reactor_config.num_types
        )
        zeros = torch.zeros(settings.batch_size, dtype=torch.bool)
        packet_targets = ETTRPacketTargets(
            value_code=initial_values,
            type_index=initial_types,
            relations=relations,
            active=active,
            root=root,
            committed=zeros,
            halted=zeros,
            slot_mask=slot_mask,
            relation_mask=relation_mask,
        )
        terminal_values = initial_values.clone()
        terminal_values[:, 0] = (
            rectangle * 4 + within
        ).remainder(reactor_config.num_value_codes)
        terminal_types = initial_types.clone()
        committed = (within == 1) | (within == 3)
        halted = (within == 2) | (within == 3)
        terminal_packet_targets = ETTRPacketTargets(
            value_code=terminal_values,
            type_index=terminal_types,
            relations=relations.clone(),
            active=active.clone(),
            root=root.clone(),
            committed=committed,
            halted=halted,
            slot_mask=slot_mask.clone(),
            relation_mask=relation_mask.clone(),
        )
        opcodes = torch.ones(
            settings.batch_size,
            settings.reactor_steps,
            dtype=torch.long,
        )
        terminal_opcode = torch.tensor(
            (1, 6, 7, 8),
            dtype=torch.long,
        ).index_select(0, within)
        opcodes[:, -1] = terminal_opcode
        transaction_committed = torch.zeros_like(opcodes, dtype=torch.bool)
        transaction_halted = torch.zeros_like(opcodes, dtype=torch.bool)
        transaction_committed[:, -1] = committed
        transaction_halted[:, -1] = halted
        transaction_targets = ETTRTransactionTargets(
            opcode=opcodes,
            source=torch.zeros_like(opcodes),
            target=torch.zeros_like(opcodes),
            relation=torch.zeros_like(opcodes),
            type_index=terminal_types[:, :1].expand_as(opcodes).clone(),
            value_code=terminal_values[:, :1].expand_as(opcodes).clone(),
            committed=transaction_committed,
            halted=transaction_halted,
            step_mask=torch.ones_like(opcodes, dtype=torch.bool),
        )
        rectangles = ETTRCausalRectangle(
            rows=torch.arange(
                settings.batch_size,
                dtype=torch.long,
            ).view(-1, 2, 2)
        )
        manifest_sha256 = hashlib.sha256(
            f"manifest-{settings.seed}-{microstep}".encode("ascii")
        ).hexdigest()
        dataset_sha256 = hashlib.sha256(
            f"dataset-{settings.seed}-{microstep}".encode("ascii")
        ).hexdigest()
        batch = ETTRContinuationBatch(
            manifest_sha256=manifest_sha256,
            dataset_sha256=dataset_sha256,
            episodes=episodes,
            packet_targets=packet_targets,
            terminal_packet_targets=terminal_packet_targets,
            causal_rectangles=rectangles,
            transaction_targets=transaction_targets,
            initial_committed=zeros.clone(),
            initial_halted=zeros.clone(),
            equivariance=None,
        )
        batch.validate(reactor_config, objective_config)
        batches.append(batch)
        digest_payload.append(
            {
                "causal_rectangles": rectangles.rows.tolist(),
                "dataset_sha256": dataset_sha256,
                "episode_ids": list(episode_ids),
                "initial_committed": batch.initial_committed.tolist(),
                "initial_halted": batch.initial_halted.tolist(),
                "manifest_sha256": manifest_sha256,
                "packet_targets": {
                    field.name: getattr(packet_targets, field.name).tolist()
                    for field in fields(packet_targets)
                },
                "reset_mask": episodes.reset_mask.tolist(),
                "query_read_index": episodes.query_read_index.tolist(),
                "segments": {
                    name: {
                        "attention_mask": getattr(
                            episodes,
                            name,
                        ).attention_mask.tolist(),
                        "targets": getattr(
                            episodes,
                            name,
                        ).targets.tolist(),
                        "tokens": getattr(
                            episodes,
                            name,
                        ).tokens.tolist(),
                    }
                    for name in ("world", "command", "query")
                },
                "terminal_packet_targets": {
                    field.name: getattr(
                        terminal_packet_targets,
                        field.name,
                    ).tolist()
                    for field in fields(terminal_packet_targets)
                },
                "transaction_targets": {
                    field.name: getattr(
                        transaction_targets,
                        field.name,
                    ).tolist()
                    for field in fields(transaction_targets)
                },
            }
        )
    digest = hashlib.sha256(canonical_json_bytes(digest_payload)).hexdigest()
    terminal_packet_sufficiency_receipt(tuple(batches))
    return tuple(batches), digest


def _tiny_model(seed: int) -> EndogenousTypedTheoryReactorGPT:
    torch.manual_seed(seed)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=32,
            zloss=0.0,
        )
    )
    return EndogenousTypedTheoryReactorGPT(
        base,
        TheoryReactorConfig(
            d_model=32,
            state_width=32,
            num_slots=6,
            num_types=3,
            num_relations=3,
            num_value_codes=16,
            max_edges=24,
            num_heads=4,
            compiler_layers=1,
            reactor_layers=1,
            query_layers=1,
            ff_multiplier=2,
            max_steps=4,
            stage_after_block=1,
            parameter_cap=1_000_000,
        ),
    )


def _model_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    seed: int,
) -> EndogenousTypedTheoryReactorGPT:
    try:
        config = GPTConfig(**checkpoint["cfg"])
    except (KeyError, TypeError) as exc:
        raise ETTRProfileError("checkpoint GPT configuration differs") from exc
    if config.n_loop != 1:
        raise ETTRProfileError("ETTR profiling requires n_loop=1")
    torch.manual_seed(seed)
    base = GPT(config)
    try:
        incompatibility = base.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
    except RuntimeError as exc:
        raise ETTRProfileError("checkpoint failed strict base loading") from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRProfileError("checkpoint strict state load differs")
    return EndogenousTypedTheoryReactorGPT(
        base,
        TheoryReactorConfig(d_model=config.d_model),
    )


def _objective_config(
    model: EndogenousTypedTheoryReactorGPT,
) -> ETTRObjectiveConfig:
    config = model.config
    return ETTRObjectiveConfig(
        vocab_size=model.base.cfg.vocab_size,
        num_slots=config.num_slots,
        num_types=config.num_types,
        num_relations=config.num_relations,
        num_value_codes=config.num_value_codes,
        active_slot_budget=config.num_slots,
        relation_edge_budget=config.max_edges,
        require_equivariance_pairs=False,
    )


def require_h100(device: torch.device) -> dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ETTRProfileError("H100 profile requires CUDA")
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    if "H100" not in name.upper() or capability[0] != 9:
        raise ETTRProfileError(f"H100 profile requires an NVIDIA H100, got {name}")
    if not torch.cuda.is_bf16_supported():
        raise ETTRProfileError("H100 profile requires native BF16")
    properties = torch.cuda.get_device_properties(device)
    return {
        "bf16_supported": True,
        "capability": list(capability),
        "name": name,
        "total_memory_bytes": int(properties.total_memory),
    }


def _component_parameters(
    model: EndogenousTypedTheoryReactorGPT,
) -> dict[str, list[torch.nn.Parameter]]:
    command_projection = list(
        model.reactor.command_projection.parameters()
    )
    command_projection_ids = {
        id(parameter) for parameter in command_projection
    }
    return {
        "base": list(model.base.parameters()),
        "command_projection": command_projection,
        "compiler": list(model.compiler.parameters()),
        "reactor": list(model.reactor.parameters()),
        "reactor_core": [
            parameter
            for parameter in model.reactor.parameters()
            if id(parameter) not in command_projection_ids
        ],
        "query_reader": list(model.query_reader.parameters()),
    }


def _parameter_sha256(
    model: EndogenousTypedTheoryReactorGPT,
) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(
            canonical_json_bytes(
                {
                    "dtype": str(value.dtype),
                    "name": name,
                    "shape": list(value.shape),
                }
            )
        )
        digest.update(
            memoryview(value.reshape(-1).view(torch.uint8).numpy())
        )
    return digest.hexdigest()


def _sample_parameters(
    parameters: Iterable[torch.nn.Parameter],
    *,
    maximum: int = 4096,
) -> torch.Tensor:
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ETTRProfileError("parameter sample budget must be a positive integer")
    trainable = tuple(
        parameter
        for parameter in parameters
        if parameter.requires_grad and parameter.numel() > 0
    )
    if len(trainable) > maximum:
        raise ETTRProfileError(
            "parameter sample budget cannot cover every trainable tensor"
        )
    allocations = [1] * len(trainable)
    remaining = maximum - len(trainable)
    while remaining:
        active = tuple(
            index
            for index, parameter in enumerate(trainable)
            if allocations[index] < parameter.numel()
        )
        if not active:
            break
        share, remainder = divmod(remaining, len(active))
        assigned = 0
        for rank, index in enumerate(active):
            capacity = trainable[index].numel() - allocations[index]
            increment = min(
                capacity,
                share + int(rank < remainder),
            )
            allocations[index] += increment
            assigned += increment
        if assigned == 0:
            break
        remaining -= assigned
    samples = []
    for parameter, take in zip(trainable, allocations, strict=True):
        flat = parameter.detach().flatten()
        coordinates = _evenly_spaced_indices(
            flat.numel(),
            take,
            device=flat.device,
        )
        samples.append(flat.index_select(0, coordinates).float().clone())
    if not samples:
        return torch.empty(0)
    return torch.cat(samples)


def _evenly_spaced_indices(
    length: int,
    count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or length < 1
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or count > length
    ):
        raise ETTRProfileError("parameter sample geometry differs")
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    numerators = torch.arange(count, dtype=torch.long, device=device)
    return numerators.mul(length - 1).div(
        count - 1,
        rounding_mode="floor",
    )


def _gradient_tensors(
    parameters: Iterable[torch.nn.Parameter],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    square = torch.zeros((), device=device, dtype=torch.float64)
    nonzero = torch.zeros((), device=device, dtype=torch.int64)
    nonfinite = torch.zeros((), device=device, dtype=torch.int64)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        square.add_(gradient.double().square().sum())
        nonzero.add_(torch.count_nonzero(gradient))
        nonfinite.add_(torch.count_nonzero(~torch.isfinite(gradient)))
    return square, nonzero, nonfinite


def _index_state(
    state: TypedTheoryState,
    index: torch.Tensor,
) -> TypedTheoryState:
    return TypedTheoryState(
        **{
            field.name: (
                getattr(state, field.name)
                if field.name == "step"
                else getattr(state, field.name).index_select(0, index)
            )
            for field in fields(state)
        }
    )


def _detach_state(state: TypedTheoryState) -> TypedTheoryState:
    return TypedTheoryState(
        **{
            field.name: (
                getattr(state, field.name)
                if field.name == "step"
                else getattr(state, field.name).detach()
            )
            for field in fields(state)
        }
    )


def _gather_query_logits(
    model: EndogenousTypedTheoryReactorGPT,
    state: TypedTheoryState,
    batch: ETTRContinuationBatch,
    row_index: torch.Tensor,
) -> torch.Tensor:
    logits, _ = model.answer_query(
        state,
        batch.episodes.query.tokens.index_select(0, row_index),
        targets=None,
        attention_mask=batch.episodes.query.attention_mask.index_select(
            0,
            row_index,
        ),
    )
    read_index = batch.episodes.query_read_index.index_select(0, row_index)
    return logits.gather(
        1,
        read_index[:, None, None].expand(-1, 1, logits.shape[-1]),
    ).squeeze(1)


def _query_binding_weights(arm: str) -> ETTRObjectiveWeights:
    if arm not in {"world", "command"}:
        raise ETTRProfileError("isolated query-binding arm differs")
    return ETTRObjectiveWeights(
        token_lm=0.0,
        packet=0.0,
        world_intervention=0.0,
        command_intervention=0.0,
        world_query_binding=1.0 if arm == "world" else 0.0,
        command_query_binding=1.0 if arm == "command" else 0.0,
        transaction=0.0,
        equivariance=0.0,
        commit_halt=0.0,
        sparsity=0.0,
        anti_bypass=0.0,
    )


def _isolated_query_binding_gradients(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    objective_config: ETTRObjectiveConfig,
    *,
    reactor_steps: int,
    device: torch.device,
) -> dict[str, dict[str, dict[str, tuple[torch.Tensor, ...] | int]]]:
    """Measure each causal query arm without support from other losses."""

    runner = CausalETTREpisodeRunner(model)
    components = _component_parameters(model)
    receipts: dict[
        str,
        dict[str, dict[str, tuple[torch.Tensor, ...] | int]],
    ] = {}
    for arm in ("world", "command"):
        receipts[arm] = {}
        for mode in ("treatment", "detached_state"):
            model.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type in {"cuda", "cpu"},
            ):
                output = runner(
                    batch.episodes,
                    reactor_steps=reactor_steps,
                    hard=True,
                    validate_batch=False,
                    compute_losses=False,
                )
                (
                    world_packet,
                    world_command,
                    world_target,
                    command_packet,
                    command_command,
                    command_target,
                ) = batch.causal_rectangles.intervention_indices()
                interventions = runner.intervene(
                    batch.episodes,
                    output.initial_state,
                    reactor_steps=reactor_steps,
                    world_packet_index=world_packet,
                    world_command_index=world_command,
                    world_query_index=world_target,
                    command_packet_index=command_packet,
                    command_command_index=command_command,
                    command_query_index=command_target,
                    hard=True,
                )
                objective_batch = batch.objective_batch(output, interventions)
                if mode == "detached_state":
                    if arm == "world":
                        correct_state = _detach_state(
                            interventions.world_terminal_state
                        )
                        correct_index = world_target
                        foil_index = world_command
                    else:
                        correct_state = _detach_state(
                            interventions.command_terminal_state
                        )
                        correct_index = command_target
                        foil_index = command_packet
                    foil_state = _detach_state(
                        _index_state(output.terminal_state, foil_index)
                    )
                    detached_pair = replace(
                        getattr(objective_batch, f"{arm}_query_binding"),
                        correct_logits=_gather_query_logits(
                            model,
                            correct_state,
                            batch,
                            correct_index,
                        ),
                        foil_logits=_gather_query_logits(
                            model,
                            foil_state,
                            batch,
                            foil_index,
                        ),
                    )
                    objective_batch = replace(
                        objective_batch,
                        **{f"{arm}_query_binding": detached_pair},
                    )
                loss = ETTRCompositeObjective(
                    objective_config,
                    weights=_query_binding_weights(arm),
                )(objective_batch).total
            loss.backward()
            receipts[arm][mode] = {
                name: (
                    *_gradient_tensors(parameters, device=device),
                    sum(parameter.grad is not None for parameter in parameters),
                )
                for name, parameters in components.items()
            }
    model.zero_grad(set_to_none=True)
    return receipts


def _encoded_tokens_per_update(settings: ProfileSettings) -> int:
    """Count actual encoder calls in the factual plus intervention path."""

    return (
        settings.batch_size
        * settings.microsteps
        * (
            settings.world_tokens
            + 2 * settings.command_tokens
            + 3 * settings.query_tokens
        )
    )


def _optimizer_state_tensor_count(
    optimizer: ETTROptimizerBundle,
) -> int:
    optimizers = [optimizer.adam]
    if optimizer.muon is not None:
        optimizers.append(optimizer.muon)
    return sum(
        int(torch.is_tensor(value))
        for instance in optimizers
        for state in instance.state.values()
        for value in state.values()
    )


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=True,
    )


class ETTRCompositeFactorialInterventionSubject(torch.nn.Module):
    """Exact factual + two-arm + composite path profiled by both arms."""

    def __init__(
        self,
        model: EndogenousTypedTheoryReactorGPT,
        objective_config: ETTRObjectiveConfig,
        *,
        reactor_steps: int,
    ) -> None:
        super().__init__()
        self.runner = CausalETTREpisodeRunner(model)
        self.objective = ETTRCompositeObjective(objective_config)
        self.reactor_steps = reactor_steps

    def forward(
        self,
        batch: ETTRContinuationBatch,
    ) -> tuple[torch.Tensor, ...]:
        output = self.runner(
            batch.episodes,
            reactor_steps=self.reactor_steps,
            hard=True,
            validate_batch=False,
            compute_losses=False,
        )
        (
            world_packet,
            world_command,
            world_target,
            command_packet,
            command_command,
            command_target,
        ) = batch.causal_rectangles.intervention_indices()
        interventions = self.runner.intervene(
            batch.episodes,
            output.initial_state,
            reactor_steps=self.reactor_steps,
            world_packet_index=world_packet,
            world_command_index=world_command,
            world_query_index=world_target,
            command_packet_index=command_packet,
            command_command_index=command_command,
            command_query_index=command_target,
            hard=True,
        )
        loss = self.objective(batch.objective_batch(output, interventions))
        return (
            *(getattr(loss, name) for name in OBJECTIVE_LOSS_NAMES),
            output.query_logits,
        )


def _run_update(
    subject: torch.nn.Module,
    optimizer: ETTROptimizerBundle,
    batches: Sequence[ETTRContinuationBatch],
    *,
    compiled: bool,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.dtype]:
    optimizer.zero_grad(set_to_none=True)
    optimizer.apply_schedule()
    accumulated = {
        name: torch.zeros((), device=device)
        for name in OBJECTIVE_LOSS_NAMES
    }
    logits_dtype = torch.float32
    for batch in batches:
        if compiled and device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        with _autocast(device):
            result = subject(batch)
            losses = result[:-1]
            logits = result[-1]
            scaled_loss = losses[0] / len(batches)
        scaled_loss.backward()
        for name, loss in zip(
            OBJECTIVE_LOSS_NAMES,
            losses,
            strict=True,
        ):
            accumulated[name].add_(loss.detach() / len(batches))
        logits_dtype = logits.dtype
    optimizer.step()
    return accumulated, logits_dtype


def _device_batches(
    batches: Sequence[ETTRContinuationBatch],
    device: torch.device,
) -> tuple[ETTRContinuationBatch, ...]:
    def segment(value: ETTREpisodeSegment) -> ETTREpisodeSegment:
        return ETTREpisodeSegment(
            tokens=value.tokens.to(device),
            targets=value.targets.to(device),
            attention_mask=value.attention_mask.to(device),
        )

    def tensor_dataclass(value):
        return type(value)(
            **{
                field.name: getattr(value, field.name).to(device)
                for field in fields(value)
            }
        )

    return tuple(
        ETTRContinuationBatch(
            manifest_sha256=batch.manifest_sha256,
            dataset_sha256=batch.dataset_sha256,
            episodes=ETTREpisodeBatch(
                episode_ids=batch.episodes.episode_ids,
                reset_mask=batch.episodes.reset_mask.to(device),
                query_read_index=batch.episodes.query_read_index.to(device),
                world=segment(batch.episodes.world),
                command=segment(batch.episodes.command),
                query=segment(batch.episodes.query),
            ),
            packet_targets=tensor_dataclass(batch.packet_targets),
            terminal_packet_targets=tensor_dataclass(
                batch.terminal_packet_targets
            ),
            causal_rectangles=ETTRCausalRectangle(
                rows=batch.causal_rectangles.rows.to(device)
            ),
            transaction_targets=tensor_dataclass(
                batch.transaction_targets
            ),
            initial_committed=batch.initial_committed.to(device),
            initial_halted=batch.initial_halted.to(device),
            equivariance=(
                None
                if batch.equivariance is None
                else tensor_dataclass(batch.equivariance)
            ),
        )
        for batch in batches
    )


def _compile_subject(
    subject: ETTRCompositeFactorialInterventionSubject,
    *,
    device: torch.device,
    compile_mode: str,
) -> torch.nn.Module:
    if device.type == "cpu":
        return torch.compile(subject, backend="eager")
    return torch.compile(subject, mode=compile_mode)


def execute_profile_arm(
    model: EndogenousTypedTheoryReactorGPT,
    settings: ProfileSettings,
    *,
    execution_arm: str,
    device: torch.device,
    device_receipt: Mapping[str, object],
) -> dict[str, object]:
    if execution_arm not in {"eager", "compiled"}:
        raise ETTRProfileError("execution arm differs")
    if settings.train_scope == "architecture":
        model.freeze_base()
    initial_parameter_sha256 = _parameter_sha256(model)
    objective_config = _objective_config(model)
    model.to(device)
    model.train()
    composite_subject = ETTRCompositeFactorialInterventionSubject(
        model,
        objective_config,
        reactor_steps=settings.reactor_steps,
    )
    subject = (
        composite_subject
        if execution_arm == "eager"
        else _compile_subject(
            composite_subject,
            device=device,
            compile_mode=settings.compile_mode,
        )
    )
    receipt = model.parameter_receipt()
    components = _component_parameters(model)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ETTRProfileError("profile has no trainable parameters")
    model_memory = (
        int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    )
    optimizer = ETTROptimizerBundle(
        model,
        ETTROptimizerConfig(
            train_base=settings.train_scope == "all",
            base_lr_muon=settings.learning_rate,
            base_lr_adam=settings.learning_rate,
            architecture_lr_muon=settings.learning_rate,
            architecture_lr_adam=settings.learning_rate,
            warmup_updates=0,
            total_updates=max(
                100,
                settings.warmup_updates + settings.measured_updates + 1,
            ),
        ),
    )
    batches, batch_sha256 = synthetic_batches(
        settings,
        reactor_config=model.config,
        objective_config=objective_config,
    )
    batches = _device_batches(batches, device)
    for batch in batches:
        batch.validate(model.config, objective_config)
    before_samples = {
        name: _sample_parameters(parameters) for name, parameters in components.items()
    }

    for _ in range(settings.warmup_updates):
        _run_update(
            subject,
            optimizer,
            batches,
            compiled=execution_arm == "compiled",
            device=device,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        optimizer_ready_memory = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        starts = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(settings.measured_updates)
        ]
        ends = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(settings.measured_updates)
        ]
        elapsed_started = None
    else:
        optimizer_ready_memory = 0
        starts = []
        ends = []
        elapsed_started = time.perf_counter_ns()

    measured_losses = {
        name: torch.zeros((), device=device)
        for name in OBJECTIVE_LOSS_NAMES
    }
    last_losses = {
        name: torch.zeros((), device=device)
        for name in OBJECTIVE_LOSS_NAMES
    }
    logits_dtype = torch.float32
    for update in range(settings.measured_updates):
        if device.type == "cuda":
            starts[update].record()
        last_losses, logits_dtype = _run_update(
            subject,
            optimizer,
            batches,
            compiled=execution_arm == "compiled",
            device=device,
        )
        for name in OBJECTIVE_LOSS_NAMES:
            measured_losses[name].add_(last_losses[name])
        if device.type == "cuda":
            ends[update].record()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        update_ms = [
            starts[index].elapsed_time(ends[index])
            for index in range(settings.measured_updates)
        ]
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        current_allocated = int(torch.cuda.memory_allocated(device))
    else:
        assert elapsed_started is not None
        total_ms = (time.perf_counter_ns() - elapsed_started) / 1_000_000
        update_ms = [total_ms]
        peak_allocated = 0
        peak_reserved = 0
        current_allocated = 0

    gradient_devices = {
        name: _gradient_tensors(parameters, device=device)
        for name, parameters in components.items()
    }
    gradient_present_tensors = {
        name: sum(parameter.grad is not None for parameter in parameters)
        for name, parameters in components.items()
    }
    after_samples = {
        name: _sample_parameters(parameters) for name, parameters in components.items()
    }
    sample_deltas = {
        name: (after_samples[name] - before_samples[name].to(device)).abs().sum()
        if before_samples[name].numel()
        else torch.zeros((), device=device)
        for name in components
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        isolated_started = None
    else:
        isolated_started = time.perf_counter_ns()
    isolated_gradient_devices = _isolated_query_binding_gradients(
        model,
        batches[0],
        objective_config,
        reactor_steps=settings.reactor_steps,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        isolated_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        isolated_peak_reserved = int(torch.cuda.max_memory_reserved(device))
        isolated_elapsed_ms = None
    else:
        assert isolated_started is not None
        isolated_peak_allocated = 0
        isolated_peak_reserved = 0
        isolated_elapsed_ms = (
            time.perf_counter_ns() - isolated_started
        ) / 1_000_000

    def host_number(tensor: torch.Tensor) -> float | int:
        value = tensor.detach().cpu().tolist()
        if isinstance(value, (float, int)):
            return value
        raise ETTRProfileError("profile scalar extraction differs")

    gradients = {}
    for name, parameters in components.items():
        square, nonzero, nonfinite = gradient_devices[name]
        present_tensors = gradient_present_tensors[name]
        gradients[name] = {
            "gradient_l2": math.sqrt(float(host_number(square))),
            "gradient_nonfinite_elements": int(host_number(nonfinite)),
            "gradient_nonzero_elements": int(host_number(nonzero)),
            "gradient_tensors": present_tensors,
            "parameter_tensors": len(parameters),
            "sampled_parameter_abs_delta": float(host_number(sample_deltas[name])),
            "trainable_parameters": sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
        }
    isolated_query_binding_gradients: dict[str, dict[str, object]] = {}
    for arm, modes in isolated_gradient_devices.items():
        mode_receipts: dict[str, object] = {}
        for mode, component_values in modes.items():
            component_receipts: dict[str, object] = {}
            for name, raw in component_values.items():
                square, nonzero, nonfinite, present_tensors = raw
                component_receipts[name] = {
                    "gradient_l2": math.sqrt(float(host_number(square))),
                    "gradient_nonfinite_elements": int(
                        host_number(nonfinite)
                    ),
                    "gradient_nonzero_elements": int(host_number(nonzero)),
                    "gradient_tensors": present_tensors,
                }
            mode_receipts[mode] = component_receipts
        isolated_query_binding_gradients[arm] = mode_receipts

    def positive(
        arm: str,
        mode: str,
        component: str,
    ) -> bool:
        value = isolated_query_binding_gradients[arm][mode][component]
        assert isinstance(value, dict)
        return (
            int(value["gradient_tensors"]) > 0
            and int(value["gradient_nonzero_elements"]) > 0
            and int(value["gradient_nonfinite_elements"]) == 0
        )

    def exact_zero(
        arm: str,
        mode: str,
        component: str,
    ) -> bool:
        value = isolated_query_binding_gradients[arm][mode][component]
        assert isinstance(value, dict)
        return (
            int(value["gradient_nonzero_elements"]) == 0
            and int(value["gradient_nonfinite_elements"]) == 0
        )

    isolated_query_binding_gate = (
        positive("world", "treatment", "compiler")
        and positive("world", "treatment", "reactor_core")
        and positive("world", "treatment", "query_reader")
        and positive("command", "treatment", "command_projection")
        and positive("command", "treatment", "reactor_core")
        and positive("command", "treatment", "query_reader")
        and positive("world", "detached_state", "query_reader")
        and positive("command", "detached_state", "query_reader")
        and exact_zero("world", "detached_state", "compiler")
        and exact_zero("world", "detached_state", "reactor_core")
        and exact_zero("command", "detached_state", "compiler")
        and exact_zero("command", "detached_state", "reactor_core")
        and exact_zero("command", "detached_state", "command_projection")
        and all(
            exact_zero(arm, mode, "base")
            for arm in ("world", "command")
            for mode in ("treatment", "detached_state")
        )
    )
    expected_gradient_components = {
        "compiler",
        "query_reader",
        "reactor",
    }
    if settings.train_scope == "all":
        expected_gradient_components.add("base")
    gradient_gate = all(
        gradients[name]["gradient_tensors"] > 0
        and gradients[name]["gradient_nonzero_elements"] > 0
        and gradients[name]["gradient_nonfinite_elements"] == 0
        and gradients[name]["sampled_parameter_abs_delta"] > 0
        for name in expected_gradient_components
    )
    measured_loss_values = {
        name: float(host_number(value))
        for name, value in measured_losses.items()
    }
    last_loss_values = {
        name: float(host_number(value))
        for name, value in last_losses.items()
    }
    loss_finite = all(
        math.isfinite(value)
        for value in (
            *measured_loss_values.values(),
            *last_loss_values.values(),
        )
    )
    total_elapsed_ms = sum(update_ms)
    encoded_tokens_per_update = _encoded_tokens_per_update(settings)
    supervised_tokens_per_update = (
        settings.batch_size
        * settings.microsteps
        * (settings.world_tokens + settings.command_tokens + settings.query_tokens - 3)
    )
    return {
        "batch": {
            "causal_rectangles_per_microstep": settings.batch_size // 4,
            "encoded_tokens_per_update": encoded_tokens_per_update,
            "episode_segments": ["WORLD", "COMMAND", "QUERY"],
            "factual_rows_per_microstep": settings.batch_size,
            "immutable_factorial_geometry": "rows[rectangle,world,command]=2x2",
            "intervention_rows_per_arm_per_microstep": settings.batch_size,
            "objective_targets": "factual_rows_gathered_by_rectangle_indices",
            "prevalidated_before_hot_path": True,
            "reset_between_segments": True,
            "sha256": batch_sha256,
            "source": (
                "validated_deterministic_synthetic_ettr_continuation_rectangles"
            ),
            "supervised_tokens_per_update": (supervised_tokens_per_update),
        },
        "device": dict(device_receipt),
        "execution": {
            "autocast_dtype": "torch.bfloat16",
            "compile_backend": (
                None
                if execution_arm == "eager"
                else ("eager" if device.type == "cpu" else "inductor")
            ),
            "compile_mode": (
                None if execution_arm == "eager" else settings.compile_mode
            ),
            "executed": True,
            "execution_arm": execution_arm,
            "factual_episode_path": True,
            "forward_backward_optimizer": True,
            "full_composite_objective": True,
            "hard_transactions": True,
            "intervention_arms": ["WORLD", "COMMAND"],
            "intervention_validate_batch_in_hot_path": True,
            "last_logits_dtype": str(logits_dtype),
            "last_loss": last_loss_values["total"],
            "loss_finite": loss_finite,
            "mean_loss": (
                measured_loss_values["total"] / settings.measured_updates
            ),
            "objective_losses": {
                name: {
                    "last": last_loss_values[name],
                    "mean": (
                        measured_loss_values[name]
                        / settings.measured_updates
                    ),
                }
                for name in OBJECTIVE_LOSS_NAMES
            },
            "optimizer": "ettr_muon_plus_adamw",
            "optimizer_state_tensors": (_optimizer_state_tensor_count(optimizer)),
            "subject": (
                "ETTRCompositeFactorialInterventionSubject"
            ),
            "factual_validate_batch_in_hot_path": False,
        },
        "gates": {
            "bf16_autocast_exercised": (logits_dtype == torch.bfloat16),
            "gradient_receipt_pass": gradient_gate,
            "isolated_query_binding_eager_bf16_gradient_receipt_pass": (
                isolated_query_binding_gate
            ),
            "loss_finite": loss_finite,
            "parameter_cap_pass": (
                receipt.complete_system_parameters <= receipt.parameter_cap
            ),
        },
        "gradients": gradients,
        "isolated_query_binding_gradients": (
            isolated_query_binding_gradients
        ),
        "isolated_query_binding_execution": {
            "autocast_dtype": "torch.bfloat16",
            "compiled": False,
            "elapsed_ms": isolated_elapsed_ms,
            "peak_allocated_bytes": isolated_peak_allocated,
            "peak_reserved_bytes": isolated_peak_reserved,
            "purpose": "causal_path_attribution_not_throughput",
        },
        "memory": {
            "current_allocated_bytes": current_allocated,
            "model_loaded_allocated_bytes": model_memory,
            "optimizer_ready_allocated_bytes": optimizer_ready_memory,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "parameters": {
            **asdict(receipt),
            "initial_parameter_sha256": initial_parameter_sha256,
            "optimizer_receipt": asdict(optimizer.receipt),
            "profile_trainable_parameters": sum(
                parameter.numel() for parameter in trainable
            ),
            "profile_trainable_tensors": len(trainable),
        },
        "throughput": {
            "encoded_tokens_per_second": (
                encoded_tokens_per_update
                * settings.measured_updates
                * 1000
                / total_elapsed_ms
            ),
            "measured_update_ms": update_ms,
            "measured_updates": settings.measured_updates,
            "reactor_state_steps_per_second": (
                settings.batch_size
                * settings.microsteps
                * settings.reactor_steps
                * 3
                * settings.measured_updates
                * 1000
                / total_elapsed_ms
            ),
            "supervised_tokens_per_second": (
                supervised_tokens_per_update
                * settings.measured_updates
                * 1000
                / total_elapsed_ms
            ),
            "total_elapsed_ms": total_elapsed_ms,
        },
    }


def execute_profile_arms(
    model_factory,
    settings: ProfileSettings,
    *,
    device: torch.device,
    device_receipt: Mapping[str, object],
) -> dict[str, object]:
    arms: dict[str, dict[str, object]] = {}
    for execution_arm in ("eager", "compiled"):
        model: EndogenousTypedTheoryReactorGPT | None = None
        print(
            "[ettr-profile] "
            f"arm={execution_arm} phase=construct "
            f"batch_size={settings.batch_size} "
            f"reactor_steps={settings.reactor_steps} "
            f"train_scope={settings.train_scope}",
            flush=True,
        )
        try:
            model = model_factory()
            print(
                f"[ettr-profile] arm={execution_arm} phase=execute",
                flush=True,
            )
            result = execute_profile_arm(
                model,
                settings,
                execution_arm=execution_arm,
                device=device,
                device_receipt=device_receipt,
            )
            arms[execution_arm] = {
                "status": "completed",
                **result,
            }
            print(
                f"[ettr-profile] arm={execution_arm} phase=completed",
                flush=True,
            )
        except Exception as exc:
            if execution_arm == "eager":
                raise
            arms[execution_arm] = {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
            print(
                "[ettr-profile] "
                f"arm={execution_arm} phase=unavailable "
                f"error_type={type(exc).__name__}",
                flush=True,
            )
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

    eager = arms["eager"]
    compiled = arms["compiled"]
    available = eager["status"] == "completed" and compiled["status"] == "completed"
    comparison: dict[str, object] = {
        "available": available,
        "compiled_attempted": True,
    }
    matched_batch = False
    matched_parameters = False
    matched_initial_parameters = False
    if available:
        eager_throughput = float(eager["throughput"]["encoded_tokens_per_second"])
        compiled_throughput = float(compiled["throughput"]["encoded_tokens_per_second"])
        eager_peak = int(eager["memory"]["peak_allocated_bytes"])
        compiled_peak = int(compiled["memory"]["peak_allocated_bytes"])
        matched_batch = eager["batch"]["sha256"] == compiled["batch"]["sha256"]
        matched_parameters = eager["parameters"] == compiled["parameters"]
        matched_initial_parameters = (
            eager["parameters"]["initial_parameter_sha256"]
            == compiled["parameters"]["initial_parameter_sha256"]
        )
        comparison.update(
            {
                "compiled_over_eager_throughput": (
                    compiled_throughput / eager_throughput
                ),
                "compiled_over_eager_peak_allocated": (
                    compiled_peak / eager_peak if eager_peak else None
                ),
                "compiled_peak_allocated_bytes": compiled_peak,
                "compiled_tokens_per_second": compiled_throughput,
                "eager_peak_allocated_bytes": eager_peak,
                "eager_tokens_per_second": eager_throughput,
                "matched_batch_sha256": matched_batch,
                "matched_initial_parameter_sha256": (
                    matched_initial_parameters
                ),
                "matched_parameter_receipt": matched_parameters,
            }
        )
    return {
        "arms": arms,
        "comparison": comparison,
        "gates": {
            "compiled_arm_completed": (compiled["status"] == "completed"),
            "eager_arm_completed": eager["status"] == "completed",
            "matched_batch_sha256": matched_batch,
            "matched_initial_parameter_sha256": matched_initial_parameters,
            "matched_parameter_receipt": matched_parameters,
        },
    }


def _resolved_settings(arguments: argparse.Namespace) -> ProfileSettings:
    cpu = arguments.mode == "cpu-validation"

    def selected(value: int | None, cpu_default: int, other: int) -> int:
        if value is not None:
            return value
        return cpu_default if cpu else other

    settings = ProfileSettings(
        mode=arguments.mode,
        batch_size=selected(arguments.batch_size, 4, 4),
        microsteps=selected(arguments.microsteps, 1, 2),
        warmup_updates=selected(arguments.warmup_updates, 0, 2),
        measured_updates=selected(arguments.measured_updates, 1, 10),
        world_tokens=selected(arguments.world_tokens, 8, 256),
        command_tokens=selected(arguments.command_tokens, 4, 64),
        query_tokens=selected(arguments.query_tokens, 5, 128),
        reactor_steps=selected(arguments.reactor_steps, 2, 4),
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        train_scope=arguments.train_scope,
        compile_mode=arguments.compile_mode,
    )
    settings.validate()
    return settings


def run(
    *,
    settings: ProfileSettings,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str | None = None,
    expected_step: int | None = None,
    protected_paths: Sequence[Path] = DEFAULT_PROTECTED_PATHS,
) -> dict[str, object]:
    settings.validate()
    if os.environ.get("SHARDS"):
        raise ETTRProfileError("live SHARDS environment is forbidden in ETTR profiling")
    checkpoint_arguments = (
        checkpoint_path,
        checkpoint_sha256,
        expected_step,
    )
    if settings.mode == "h100":
        if any(value is None for value in checkpoint_arguments):
            raise ETTRProfileError("H100 mode requires checkpoint path, hash, and step")
    elif any(value is not None for value in checkpoint_arguments):
        raise ETTRProfileError("dry-run and CPU validation may not read a checkpoint")

    custody_paths = list(protected_paths)
    if checkpoint_path is not None:
        custody_paths.append(checkpoint_path)
    reserved = reserve_output_directory(
        output_dir,
        protected_paths=custody_paths,
    )
    common: dict[str, object] = {
        "custody": {
            "checkpoint_read_only": settings.mode == "h100",
            "live_shards_read": False,
            "model_or_optimizer_state_written": False,
            "output_directory": str(reserved),
            "output_isolated": True,
            "pretraining_started": False,
            "profiling_only": True,
        },
        "mode": settings.mode,
        "schema": SCHEMA,
        "settings": asdict(settings),
        "sync_points": list(SYNC_POINTS),
    }
    checkpoint_receipt: CheckpointReceipt | None = None
    if settings.mode == "dry-run":
        report = {
            **common,
            "execution": {
                "executed": False,
                "validation_only": True,
            },
            "gates": {
                "control_plane_valid": True,
            },
        }
    elif settings.mode == "cpu-validation":
        result = execute_profile_arms(
            lambda: _tiny_model(settings.seed),
            settings,
            device=torch.device("cpu"),
            device_receipt={
                "bf16_supported": True,
                "capability": None,
                "name": "cpu-validation",
                "total_memory_bytes": None,
                "validation_only": True,
            },
        )
        report = {**common, **result}
    else:
        assert checkpoint_path is not None
        assert checkpoint_sha256 is not None
        assert expected_step is not None
        device = torch.device("cuda")
        device_receipt = require_h100(device)
        checkpoint, metadata = load_checkpoint_read_only(
            checkpoint_path,
            expected_sha256=checkpoint_sha256,
            expected_step=expected_step,
        )
        result = execute_profile_arms(
            lambda: _model_from_checkpoint(
                checkpoint,
                seed=settings.seed,
            ),
            settings,
            device=device,
            device_receipt=device_receipt,
        )
        del checkpoint
        after_hash = sha256_file(checkpoint_path)
        after_metadata = _verify_checkpoint_path(checkpoint_path)
        unchanged = (
            after_hash == checkpoint_sha256
            and after_metadata.st_dev == metadata.st_dev
            and after_metadata.st_ino == metadata.st_ino
            and after_metadata.st_size == metadata.st_size
            and after_metadata.st_mtime_ns == metadata.st_mtime_ns
        )
        if not unchanged:
            raise ETTRProfileError("checkpoint changed during profiling")
        checkpoint_receipt = CheckpointReceipt(
            path=str(checkpoint_path),
            sha256_before=checkpoint_sha256,
            sha256_after=after_hash,
            bytes=metadata.st_size,
            step=expected_step,
            strict_state_load=True,
            unchanged_after_profile=True,
            opened_read_only=True,
        )
        report = {
            **common,
            **result,
            "checkpoint": asdict(checkpoint_receipt),
        }
    report_path = write_report_once(reserved, report)
    summary = {
        "mode": settings.mode,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "schema": SCHEMA,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "cpu-validation", "h100"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--microsteps", type=int)
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--measured-updates", type=int)
    parser.add_argument("--world-tokens", type=int)
    parser.add_argument("--command-tokens", type=int)
    parser.add_argument("--query-tokens", type=int)
    parser.add_argument("--reactor-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026072501)
    parser.add_argument(
        "--train-scope",
        choices=("architecture", "all"),
        default="architecture",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    settings = _resolved_settings(arguments)
    run(
        settings=settings,
        output_dir=arguments.output_dir,
        checkpoint_path=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        expected_step=arguments.expected_step,
    )


if __name__ == "__main__":
    main()
