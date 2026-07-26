#!/usr/bin/env python3
"""Isolated BF16 resource profiler for the full causal ETTR episode runner.

This program is not a trainer. It uses deterministic synthetic token tensors,
runs a bounded number of in-memory optimizer microsteps, writes one JSON
receipt, and never writes model or optimizer state. H100 mode may read a
hash-bound base checkpoint, but re-hashes it after profiling and refuses any
output directory that aliases the checkpoint or a protected repository path.
Matched eager and ``torch.compile`` arms start from identical model
initialization and consume identical WORLD, COMMAND, and QUERY episodes.

Synchronization policy
----------------------
The Python profiling loop contains no ``Tensor.item()`` calls. CUDA events are
recorded asynchronously. Explicit host synchronization occurs only:

1. after warmup, before peak-memory reset and measured event recording;
2. once after all measured updates, before timing/gradient receipt extraction;
3. in eager mode, inside the shared LM-loss supervision validator, which uses
   a tensor-to-bool check and is intentionally included in the measured path.

The third category is existing architecture behavior, not profiler logging.
The compiled arm converts it to an asynchronous assertion. The remaining ETTR
state and episode assertions are already asynchronous on CUDA.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
)
from ettr_episode import (
    CausalETTREpisodeRunner,
    ETTREpisodeBatch,
    ETTREpisodeSegment,
)
from ettr_optimization import (
    ETTROptimizerBundle,
    ETTROptimizerConfig,
)
from model import GPT, GPTConfig


SCHEMA = "shohin-ettr-h100-profile-v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTECTED_PATHS = (REPOSITORY_ROOT / "train" / "flagship_out",)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SYNC_POINTS = (
    "cuda_after_warmup_before_measurement",
    "cuda_after_all_measured_updates_before_receipt",
    "eager_shared_lm_loss_tensor_to_bool_validation_inside_forward",
    "cuda_between_arms_before_allocator_cleanup",
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
            self.batch_size > 2
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
    vocab_size: int,
) -> tuple[tuple[ETTREpisodeBatch, ...], str]:
    """Build validated deterministic WORLD/COMMAND/QUERY episodes."""

    if vocab_size < 8:
        raise ETTRProfileError("synthetic profile vocabulary is too small")
    batches: list[ETTREpisodeBatch] = []
    digest_payload: list[dict[str, object]] = []
    for microstep in range(settings.microsteps):
        offset = settings.seed + 104_729 * microstep

        def tokens(length: int, multiplier: int, bias: int) -> torch.Tensor:
            positions = torch.arange(
                settings.batch_size * length,
                dtype=torch.long,
            ).view(settings.batch_size, length)
            rows = torch.arange(
                settings.batch_size,
                dtype=torch.long,
            )[:, None]
            return (
                positions * multiplier + rows * (multiplier + 12) + offset + bias
            ).remainder(vocab_size)

        world = tokens(settings.world_tokens, 17, 11)
        command = tokens(settings.command_tokens, 29, 23)
        query = tokens(settings.query_tokens, 37, 31)
        episode_ids = tuple(
            f"synthetic-{settings.seed}-{microstep}-{row}"
            for row in range(settings.batch_size)
        )
        batch = ETTREpisodeBatch(
            episode_ids=episode_ids,
            reset_mask=torch.ones(
                settings.batch_size,
                dtype=torch.bool,
            ),
            world=ETTREpisodeSegment.from_tokens(world),
            command=ETTREpisodeSegment.from_tokens(command),
            query=ETTREpisodeSegment.from_tokens(query),
        )
        batch.validate()
        batches.append(batch)
        digest_payload.append(
            {
                "episode_ids": list(episode_ids),
                "reset_mask": batch.reset_mask.tolist(),
                "segments": {
                    name: {
                        "attention_mask": getattr(
                            batch,
                            name,
                        ).attention_mask.tolist(),
                        "targets": getattr(
                            batch,
                            name,
                        ).targets.tolist(),
                        "tokens": getattr(
                            batch,
                            name,
                        ).tokens.tolist(),
                    }
                    for name in ("world", "command", "query")
                },
            }
        )
    digest = hashlib.sha256(canonical_json_bytes(digest_payload)).hexdigest()
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
    return {
        "base": list(model.base.parameters()),
        "compiler": list(model.compiler.parameters()),
        "reactor": list(model.reactor.parameters()),
        "query_reader": list(model.query_reader.parameters()),
    }


def _sample_parameters(
    parameters: Iterable[torch.nn.Parameter],
    *,
    maximum: int = 4096,
) -> torch.Tensor:
    samples = []
    remaining = maximum
    for parameter in parameters:
        if not parameter.requires_grad or remaining == 0:
            continue
        flat = parameter.detach().flatten()
        take = min(flat.numel(), remaining)
        samples.append(flat[:take].float().clone())
        remaining -= take
    if not samples:
        return torch.empty(0)
    return torch.cat(samples)


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


def _run_update(
    subject: torch.nn.Module,
    optimizer: ETTROptimizerBundle,
    batches: Sequence[ETTREpisodeBatch],
    *,
    compiled: bool,
    reactor_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.dtype]:
    optimizer.zero_grad(set_to_none=True)
    optimizer.apply_schedule()
    accumulated = torch.zeros((), device=device)
    logits_dtype = torch.float32
    for batch in batches:
        if compiled and device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        with _autocast(device):
            output = subject(
                batch,
                reactor_steps=reactor_steps,
                hard=False,
                validate_batch=False,
            )
            scaled_loss = output.losses.token_lm / len(batches)
        scaled_loss.backward()
        accumulated.add_(scaled_loss.detach())
        logits_dtype = output.query_logits.dtype
    optimizer.step()
    return accumulated, logits_dtype


def _device_batches(
    batches: Sequence[ETTREpisodeBatch],
    device: torch.device,
) -> tuple[ETTREpisodeBatch, ...]:
    def segment(value: ETTREpisodeSegment) -> ETTREpisodeSegment:
        return ETTREpisodeSegment(
            tokens=value.tokens.to(device),
            targets=value.targets.to(device),
            attention_mask=value.attention_mask.to(device),
        )

    return tuple(
        ETTREpisodeBatch(
            episode_ids=batch.episode_ids,
            reset_mask=batch.reset_mask.to(device),
            world=segment(batch.world),
            command=segment(batch.command),
            query=segment(batch.query),
        )
        for batch in batches
    )


def _compile_subject(
    runner: CausalETTREpisodeRunner,
    *,
    device: torch.device,
    compile_mode: str,
) -> torch.nn.Module:
    if device.type == "cpu":
        return torch.compile(runner, backend="eager")
    return torch.compile(runner, mode=compile_mode)


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
    model.to(device)
    model.train()
    runner = CausalETTREpisodeRunner(model)
    subject = (
        runner
        if execution_arm == "eager"
        else _compile_subject(
            runner,
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
        vocab_size=model.base.cfg.vocab_size,
    )
    batches = _device_batches(batches, device)
    before_samples = {
        name: _sample_parameters(parameters) for name, parameters in components.items()
    }

    for _ in range(settings.warmup_updates):
        _run_update(
            subject,
            optimizer,
            batches,
            compiled=execution_arm == "compiled",
            reactor_steps=settings.reactor_steps,
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

    measured_loss = torch.zeros((), device=device)
    last_loss = torch.zeros((), device=device)
    logits_dtype = torch.float32
    for update in range(settings.measured_updates):
        if device.type == "cuda":
            starts[update].record()
        last_loss, logits_dtype = _run_update(
            subject,
            optimizer,
            batches,
            compiled=execution_arm == "compiled",
            reactor_steps=settings.reactor_steps,
            device=device,
        )
        measured_loss.add_(last_loss)
        if device.type == "cuda":
            ends[update].record()

    gradient_devices = {
        name: _gradient_tensors(parameters, device=device)
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
        update_ms = [total_ms / settings.measured_updates]
        peak_allocated = 0
        peak_reserved = 0
        current_allocated = 0

    def host_number(tensor: torch.Tensor) -> float | int:
        value = tensor.detach().cpu().tolist()
        if isinstance(value, (float, int)):
            return value
        raise ETTRProfileError("profile scalar extraction differs")

    gradients = {}
    for name, parameters in components.items():
        square, nonzero, nonfinite = gradient_devices[name]
        present_tensors = sum(parameter.grad is not None for parameter in parameters)
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
    loss_sum = float(host_number(measured_loss))
    last_loss_value = float(host_number(last_loss))
    loss_finite = math.isfinite(loss_sum) and math.isfinite(last_loss_value)
    total_elapsed_ms = sum(update_ms)
    encoded_tokens_per_update = (
        settings.batch_size
        * settings.microsteps
        * (settings.world_tokens + settings.command_tokens + settings.query_tokens)
    )
    supervised_tokens_per_update = (
        settings.batch_size
        * settings.microsteps
        * (settings.world_tokens + settings.command_tokens + settings.query_tokens - 3)
    )
    return {
        "batch": {
            "encoded_tokens_per_update": encoded_tokens_per_update,
            "episode_segments": ["WORLD", "COMMAND", "QUERY"],
            "prevalidated_before_hot_path": True,
            "reset_between_segments": True,
            "sha256": batch_sha256,
            "source": ("validated_deterministic_synthetic_ettr_episodes"),
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
            "forward_backward_optimizer": True,
            "full_token_lm_loss": True,
            "last_logits_dtype": str(logits_dtype),
            "last_loss": last_loss_value,
            "loss_finite": loss_finite,
            "mean_loss": loss_sum / settings.measured_updates,
            "optimizer": "ettr_muon_plus_adamw",
            "optimizer_state_tensors": (_optimizer_state_tensor_count(optimizer)),
            "subject": "CausalETTREpisodeRunner",
            "validate_batch_in_hot_path": False,
        },
        "gates": {
            "bf16_autocast_exercised": (logits_dtype == torch.bfloat16),
            "gradient_receipt_pass": gradient_gate,
            "loss_finite": loss_finite,
            "parameter_cap_pass": (
                receipt.complete_system_parameters <= receipt.parameter_cap
            ),
        },
        "gradients": gradients,
        "memory": {
            "current_allocated_bytes": current_allocated,
            "model_loaded_allocated_bytes": model_memory,
            "optimizer_ready_allocated_bytes": optimizer_ready_memory,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "parameters": {
            **asdict(receipt),
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
        try:
            model = model_factory()
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
        except Exception as exc:
            if execution_arm == "eager":
                raise
            arms[execution_arm] = {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
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
    if available:
        eager_throughput = float(eager["throughput"]["encoded_tokens_per_second"])
        compiled_throughput = float(compiled["throughput"]["encoded_tokens_per_second"])
        eager_peak = int(eager["memory"]["peak_allocated_bytes"])
        compiled_peak = int(compiled["memory"]["peak_allocated_bytes"])
        matched_batch = eager["batch"]["sha256"] == compiled["batch"]["sha256"]
        matched_parameters = eager["parameters"] == compiled["parameters"]
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
        batch_size=selected(arguments.batch_size, 1, 1),
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
