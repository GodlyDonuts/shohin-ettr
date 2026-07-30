#!/usr/bin/env python3
"""Test whether ETTR state is blocked by the frozen vocabulary interface.

The treatment motor receives only the source-deleted query-reader residual at
the causal query position.  A matched query-only motor receives the frozen
query hidden state instead.  Both are trained against the protocol's two
admitted disposition tokens, first assessed with the exact terminal packet,
and then assessed with the fully autonomous compiler/reactor state.

This is an isolated component probe.  Exact packet labels are training-only;
they never enter the autonomous arm.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

from safetensors.torch import save_file
import torch
import torch.nn.functional as F
from torch import nn

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TypedTheoryState,
)
from ettr_data_contract import ETTRContinuationBatch
from ettr_objectives import (
    ETTRCausalQueryPair,
    ETTRObjectiveConfig,
    _causal_query_binding_loss,
)
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_component_assembly import load_hash_bound_component
from eval_ettr_v3 import (
    ETTRV3EvaluationError,
    _build_model,
    _parameter_sha256,
    _read_hash_bound_json,
    _validate_run_contract,
)
from probe_ettr_causal_queries import _pair_rows, _summary
from probe_ettr_oracle_interfaces import packet_targets_to_state


RUN_SCHEMA = "shohin-ettr-disposition-motor-run-v1"
REPORT_SCHEMA = "shohin-ettr-disposition-motor-report-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENTS = ("compiler", "reactor", "query_reader")


class ETTRDispositionMotorError(ETTRV3EvaluationError):
    """The disposition-motor experiment differs from its sealed contract."""


class DispositionMotor(nn.Module):
    """Small generic readout for a two-outcome source-deleted protocol."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        if width < 1 or hidden < 1:
            raise ETTRDispositionMotorError(
                "disposition motor geometry differs"
            )
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, hidden)
        self.up = nn.Linear(hidden, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ETTRDispositionMotorError(
                "disposition motor features differ"
            )
        features = features.to(dtype=self.norm.weight.dtype)
        return self.up(F.gelu(self.down(self.norm(features))))


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_no_replace(path: Path, payload: bytes, mode: int = 0o400) -> str:
    if not path.is_absolute() or not path.parent.is_dir():
        raise ETTRDispositionMotorError(
            "disposition motor output destination differs"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise ETTRDispositionMotorError(
            "refusing an existing or unsafe disposition motor output"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _query_classes(
    batch: ETTRContinuationBatch,
    *,
    negative_token_id: int,
    positive_token_id: int,
) -> torch.Tensor:
    targets = batch.episodes.query.targets.gather(
        1,
        batch.episodes.query_read_index[:, None],
    ).squeeze(1)
    admitted = (targets == negative_token_id) | (
        targets == positive_token_id
    )
    if not bool(admitted.all()):
        raise ETTRDispositionMotorError(
            "query target leaves the disposition codebook"
        )
    return targets.eq(positive_token_id).long()


def _gather_query_position(
    hidden: torch.Tensor,
    batch: ETTRContinuationBatch,
) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[:2] != batch.episodes.query.tokens.shape:
        raise ETTRDispositionMotorError(
            "query feature geometry differs"
        )
    return hidden.gather(
        1,
        batch.episodes.query_read_index[:, None, None].expand(
            -1,
            1,
            hidden.shape[-1],
        ),
    ).squeeze(1)


def _query_features(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    state: TypedTheoryState,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        query_hidden = model._encode_to_stage(
            batch.episodes.query.tokens,
            pos=0,
        )
        read = model.query_reader(
            query_hidden,
            state,
            attention_mask=batch.episodes.query.attention_mask,
        )
    return (
        _gather_query_position(query_hidden, batch).detach(),
        _gather_query_position(read, batch).detach(),
    )


def _autonomous_terminal(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
) -> TypedTheoryState:
    with torch.no_grad():
        world_hidden = model._encode_to_stage(
            batch.episodes.world.tokens,
            pos=0,
        )
        state = model.compiler(
            world_hidden,
            attention_mask=batch.episodes.world.attention_mask,
            hard=True,
        )
        command_hidden = model._encode_to_stage(
            batch.episodes.command.tokens,
            pos=0,
        )
        state, _trace = model.reactor(
            state,
            steps=batch.transaction_targets.opcode.shape[1],
            hard=True,
            command_hidden=command_hidden,
            command_attention_mask=batch.episodes.command.attention_mask,
        )
    return state.detached_clone()


def _causal_pairs(
    logits: torch.Tensor,
    classes: torch.Tensor,
    batch: ETTRContinuationBatch,
) -> Mapping[str, ETTRCausalQueryPair]:
    (
        _world_packet,
        world_command,
        world_target,
        command_packet,
        _command_command,
        command_target,
    ) = batch.causal_rectangles.intervention_indices()
    return {
        "world": ETTRCausalQueryPair(
            correct_logits=logits.index_select(0, world_target),
            foil_logits=logits.index_select(0, world_command),
            correct_target=classes.index_select(0, world_target),
            foil_target=classes.index_select(0, world_command),
        ),
        "command": ETTRCausalQueryPair(
            correct_logits=logits.index_select(0, command_target),
            foil_logits=logits.index_select(0, command_packet),
            correct_target=classes.index_select(0, command_target),
            foil_target=classes.index_select(0, command_packet),
        ),
    }


def _motor_loss(
    logits: torch.Tensor,
    classes: torch.Tensor,
    batch: ETTRContinuationBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
    pairs = _causal_pairs(logits, classes, batch)
    losses = {
        "factual": F.cross_entropy(logits.float(), classes),
        "world_binding": _causal_query_binding_loss(
            pairs["world"],
            margin=1.0,
        )[0],
        "command_binding": _causal_query_binding_loss(
            pairs["command"],
            margin=1.0,
        )[0],
    }
    return torch.stack(tuple(losses.values())).mean(), {
        name: float(value.detach().cpu())
        for name, value in losses.items()
    }


def _evaluate(
    model: EndogenousTypedTheoryReactorGPT,
    motors: Mapping[str, DispositionMotor],
    *,
    stream: ETTRV3StreamingRelease,
    packet_index: ETTRDiskPacketSufficiencyIndex,
    device: torch.device,
    data_seed: int,
    max_batches: int,
    negative_token_id: int,
    positive_token_id: int,
) -> dict[str, object]:
    rows = {
        motor: {
            state: {"world": [], "command": []}
            for state in ("exact_terminal", "autonomous")
        }
        for motor in motors
    }
    factual = {
        motor: {
            state: [0, 0]
            for state in ("exact_terminal", "autonomous")
        }
        for motor in motors
    }
    class_counts = [0, 0]
    iterator = stream.iter_positioned_batches(
        "development",
        rank=0,
        world_size=1,
        epoch=0,
        seed=data_seed,
    )
    observed = 0
    for _, cpu_batch in iterator:
        if observed >= max_batches:
            break
        packet_index.verify_validation((cpu_batch,))
        batch = move_continuation_batch(cpu_batch, device)
        classes = _query_classes(
            batch,
            negative_token_id=negative_token_id,
            positive_token_id=positive_token_id,
        )
        class_counts[0] += int(classes.eq(0).sum().detach().cpu())
        class_counts[1] += int(classes.eq(1).sum().detach().cpu())
        exact = packet_targets_to_state(
            batch.terminal_packet_targets,
            model.config,
            step=batch.transaction_targets.opcode.shape[1],
            dtype=next(model.query_reader.parameters()).dtype,
        )
        states = {
            "exact_terminal": exact,
            "autonomous": _autonomous_terminal(model, batch),
        }
        for state_name, state in states.items():
            query_features, reader_features = _query_features(
                model,
                batch,
                state,
            )
            features = {
                "query_only": query_features,
                "treatment": reader_features,
            }
            for motor_name, motor in motors.items():
                logits = motor(features[motor_name])
                predictions = logits.argmax(-1)
                factual[motor_name][state_name][0] += int(
                    predictions.eq(classes).sum().detach().cpu()
                )
                factual[motor_name][state_name][1] += classes.numel()
                for kind, pair in _causal_pairs(
                    logits,
                    classes,
                    batch,
                ).items():
                    rows[motor_name][state_name][kind].extend(
                        _pair_rows(pair)
                    )
        observed += 1
    if observed != max_batches or min(class_counts) < 1:
        raise ETTRDispositionMotorError(
            "disposition motor development support differs"
        )
    return {
        "batches": observed,
        "class_counts": {
            "negative": class_counts[0],
            "positive": class_counts[1],
        },
        "motors": {
            motor: {
                state: {
                    "factual_accuracy": (
                        factual[motor][state][0]
                        / factual[motor][state][1]
                    ),
                    "query_binding": {
                        kind: _summary(values)
                        for kind, values in rows[motor][state].items()
                    },
                }
                for state in ("exact_terminal", "autonomous")
            }
            for motor in motors
        },
    }


def _gates(report: Mapping[str, object]) -> dict[str, bool]:
    motors = report["motors"]
    if not isinstance(motors, Mapping):
        raise ETTRDispositionMotorError(
            "disposition motor summary differs"
        )
    treatment = motors["treatment"]
    query_only = motors["query_only"]
    if not isinstance(treatment, Mapping) or not isinstance(
        query_only,
        Mapping,
    ):
        raise ETTRDispositionMotorError(
            "disposition motor arms differ"
        )

    def margin(arm: Mapping[str, object], state: str, kind: str) -> float:
        state_values = arm[state]
        if not isinstance(state_values, Mapping):
            raise ETTRDispositionMotorError(
                "disposition motor state summary differs"
            )
        binding = state_values["query_binding"]
        if not isinstance(binding, Mapping):
            raise ETTRDispositionMotorError(
                "disposition motor binding summary differs"
            )
        kind_values = binding[kind]
        if not isinstance(kind_values, Mapping):
            raise ETTRDispositionMotorError(
                "disposition motor causal summary differs"
            )
        rates = kind_values["margin_rates"]
        if not isinstance(rates, Mapping):
            raise ETTRDispositionMotorError(
                "disposition motor margin summary differs"
            )
        return float(rates["0.1"])

    exact = treatment["exact_terminal"]
    autonomous = treatment["autonomous"]
    query_exact = query_only["exact_terminal"]
    if not all(
        isinstance(value, Mapping)
        for value in (exact, autonomous, query_exact)
    ):
        raise ETTRDispositionMotorError(
            "disposition motor factual summary differs"
        )
    return {
        "autonomous_factual_above_chance": (
            float(autonomous["factual_accuracy"]) > 0.5
        ),
        "autonomous_world_margin_positive": (
            margin(treatment, "autonomous", "world") > 0.0
        ),
        "exact_factual_beats_query_only": (
            float(exact["factual_accuracy"])
            > float(query_exact["factual_accuracy"])
        ),
        "exact_world_margin_beats_query_only": (
            margin(treatment, "exact_terminal", "world")
            > margin(query_only, "exact_terminal", "world")
        ),
        "exact_command_margin_beats_query_only": (
            margin(treatment, "exact_terminal", "command")
            > margin(query_only, "exact_terminal", "command")
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--run-contract-sha256", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--compiler-sha256", required=True)
    parser.add_argument("--reactor", type=Path, required=True)
    parser.add_argument("--reactor-sha256", required=True)
    parser.add_argument("--query-reader", type=Path, required=True)
    parser.add_argument("--query-reader-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--motor-seed", type=int, required=True)
    parser.add_argument("--negative-token-id", type=int, required=True)
    parser.add_argument("--positive-token-id", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    hashes = (
        args.release_sha256,
        args.run_contract_sha256,
        args.compiler_sha256,
        args.reactor_sha256,
        args.query_reader_sha256,
    )
    paths = (
        args.release_root,
        args.data_root,
        args.tokenizer,
        args.protected_checkpoint,
        args.run_contract,
        args.compiler,
        args.reactor,
        args.query_reader,
        args.output,
    )
    if (
        any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or any(not path.is_absolute() for path in paths)
        or not 0 <= args.architecture_seed < 2**63
        or not 0 <= args.data_seed < 2**63
        or not 0 <= args.motor_seed < 2**63
        or args.negative_token_id < 0
        or args.positive_token_id < 0
        or args.negative_token_id == args.positive_token_id
        or args.hidden < 1
        or args.updates < 1
        or args.eval_batches < 2
        or args.log_every < 1
        or not math.isfinite(args.learning_rate)
        or not 0.0 < args.learning_rate < 1.0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0.0
        or not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0.0
    ):
        raise ETTRDispositionMotorError(
            "disposition motor arguments differ"
        )


def _motor_state(motor: DispositionMotor) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in motor.state_dict().items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise ETTRDispositionMotorError(
            "disposition motor training requires CUDA"
        )
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRDispositionMotorError(
            "disposition motor training requires H100"
        )
    if args.output.exists() or args.output.is_symlink():
        raise ETTRDispositionMotorError(
            "refusing an existing disposition motor output"
        )
    args.output.mkdir(mode=0o700, parents=True)

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    run_contract = _read_hash_bound_json(
        args.run_contract,
        expected_sha256=args.run_contract_sha256,
        label="ETTR run contract",
    )
    model_config, _optimizer_config = _validate_run_contract(
        run_contract,
        release_sha256=args.release_sha256,
        release_source_commit=stream.release["source_commit"],
        architecture_seed=args.architecture_seed,
    )
    model, protected = _build_model(
        args.protected_checkpoint,
        architecture_seed=args.architecture_seed,
        model_config=model_config,
        device=device,
    )
    component_paths = {
        "compiler": args.compiler,
        "reactor": args.reactor,
        "query_reader": args.query_reader,
    }
    component_hashes = {
        "compiler": args.compiler_sha256,
        "reactor": args.reactor_sha256,
        "query_reader": args.query_reader_sha256,
    }
    for name in _COMPONENTS:
        load_hash_bound_component(
            getattr(model, name),
            component_paths[name],
            expected_sha256=component_hashes[name],
            label=name.replace("_", " "),
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    torch.manual_seed(args.motor_seed)
    torch.cuda.manual_seed(args.motor_seed)
    template = DispositionMotor(model.config.d_model, args.hidden).to(device)
    motors = {
        "query_only": deepcopy(template),
        "treatment": template,
    }
    motor_parameters = {
        name: sum(parameter.numel() for parameter in motor.parameters())
        for name, motor in motors.items()
    }
    if len(set(motor_parameters.values())) != 1:
        raise ETTRDispositionMotorError(
            "disposition motor control geometry differs"
        )
    treatment_parameters = motor_parameters["treatment"]
    receipt = model.parameter_receipt()
    deployment_parameters = (
        receipt.complete_system_parameters + treatment_parameters
    )
    if deployment_parameters > receipt.parameter_cap:
        raise ETTRDispositionMotorError(
            "disposition motor exceeds the complete-system parameter cap"
        )
    optimizers = {
        name: torch.optim.AdamW(
            motor.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        for name, motor in motors.items()
    }
    packet_index = ETTRDiskPacketSufficiencyIndex(stream.packet_index_root)
    try:
        before = _evaluate(
            model,
            motors,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
            negative_token_id=args.negative_token_id,
            positive_token_id=args.positive_token_id,
        )
        initial_hashes = {}
        for name, motor in motors.items():
            path = args.output / f"{name}-initial.safetensors"
            save_file(_motor_state(motor), path)
            os.chmod(path, 0o400)
            initial_hashes[name] = _sha256_file(path)
        contract = {
            "architecture_seed": args.architecture_seed,
            "component_sha256": component_hashes,
            "data_seed": args.data_seed,
            "eval_batches": args.eval_batches,
            "gradient_clip": args.gradient_clip,
            "hidden": args.hidden,
            "initial_motor_sha256": initial_hashes,
            "learning_rate": args.learning_rate,
            "motor_seed": args.motor_seed,
            "negative_token_id": args.negative_token_id,
            "oracle_at_autonomous_inference": False,
            "oracle_training_boundary": "exact_terminal_packet",
            "positive_token_id": args.positive_token_id,
            "protected_checkpoint_sha256": (
                protected.checkpoint_sha256
            ),
            "release_file_sha256": args.release_sha256,
            "run_contract_sha256": args.run_contract_sha256,
            "schema": RUN_SCHEMA,
            "source_commit": args.source_commit,
            "updates": args.updates,
            "weight_decay": args.weight_decay,
        }
        _write_no_replace(
            args.output / "run-contract.json",
            _canonical_bytes(contract),
        )
        _write_no_replace(
            args.output / "train.jsonl",
            b"",
            mode=0o600,
        )

        iterator = stream.iter_positioned_batches(
            "train",
            rank=0,
            world_size=1,
            epoch=0,
            seed=args.data_seed,
        )
        observed_rows = 0
        class_counts = [0, 0]
        for update in range(1, args.updates + 1):
            try:
                position, cpu_batch = next(iterator)
            except StopIteration as exc:
                raise ETTRDispositionMotorError(
                    "disposition motor update budget exceeds one epoch"
                ) from exc
            packet_index.verify_train((cpu_batch,))
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(
                model.config,
                ETTRObjectiveConfig(
                    vocab_size=model.base.cfg.vocab_size
                ),
            )
            classes = _query_classes(
                batch,
                negative_token_id=args.negative_token_id,
                positive_token_id=args.positive_token_id,
            )
            class_counts[0] += int(classes.eq(0).sum().detach().cpu())
            class_counts[1] += int(classes.eq(1).sum().detach().cpu())
            exact = packet_targets_to_state(
                batch.terminal_packet_targets,
                model.config,
                step=batch.transaction_targets.opcode.shape[1],
                dtype=next(model.query_reader.parameters()).dtype,
            )
            query_features, reader_features = _query_features(
                model,
                batch,
                exact,
            )
            features = {
                "query_only": query_features,
                "treatment": reader_features,
            }
            metrics = {}
            for name, motor in motors.items():
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    loss, parts = _motor_loss(
                        motor(features[name]),
                        classes,
                        batch,
                    )
                if not bool(torch.isfinite(loss)):
                    raise ETTRDispositionMotorError(
                        "disposition motor loss is non-finite"
                    )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    tuple(motor.parameters()),
                    args.gradient_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                metrics[name] = {
                    "gradient_norm_pre_clip": float(
                        gradient_norm.detach().float().cpu()
                    ),
                    "loss": float(loss.detach().cpu()),
                    "loss_parts": parts,
                }
            observed_rows += classes.numel()
            if update % args.log_every == 0 or update == args.updates:
                with (args.output / "train.jsonl").open(
                    "ab",
                    buffering=0,
                ) as log:
                    log.write(
                        _canonical_bytes(
                            {
                                "metrics": metrics,
                                "position": position,
                                "schema": (
                                    "shohin-ettr-disposition-motor-metric-v1"
                                ),
                                "update": update,
                            }
                        )
                    )
        if min(class_counts) < 1:
            raise ETTRDispositionMotorError(
                "disposition motor train support differs"
            )
        os.chmod(args.output / "train.jsonl", 0o400)

        after = _evaluate(
            model,
            motors,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
            negative_token_id=args.negative_token_id,
            positive_token_id=args.positive_token_id,
        )
        final_hashes = {}
        for name, motor in motors.items():
            path = args.output / f"{name}-final.safetensors"
            save_file(_motor_state(motor), path)
            os.chmod(path, 0o400)
            final_hashes[name] = _sha256_file(path)
        gates = _gates(after)
        gates["strict_output_interface_signal"] = all(gates.values())
        report = {
            "after": after,
            "architecture_seed": args.architecture_seed,
            "before": before,
            "class_counts": {
                "negative": class_counts[0],
                "positive": class_counts[1],
            },
            "component_sha256": component_hashes,
            "data_seed": args.data_seed,
            "device": {
                "bf16": torch.cuda.is_bf16_supported(),
                "name": torch.cuda.get_device_name(device),
            },
            "final_motor_sha256": final_hashes,
            "gates": gates,
            "initial_motor_sha256": initial_hashes,
            "model_parameter_sha256": _parameter_sha256(model),
            "observed_rows": observed_rows,
            "oracle_at_autonomous_inference": False,
            "parameter_receipt": {
                "complete_system_with_treatment_motor": (
                    deployment_parameters
                ),
                "motor_control_parameters": motor_parameters["query_only"],
                "remaining_under_cap": (
                    receipt.parameter_cap - deployment_parameters
                ),
                "treatment_motor_parameters": treatment_parameters,
                "unmodified_model": asdict(receipt),
            },
            "protected_checkpoint_sha256": protected.checkpoint_sha256,
            "release_file_sha256": args.release_sha256,
            "release_manifest_sha256": stream.manifest.sha256(),
            "schema": REPORT_SCHEMA,
            "source_commit": args.source_commit,
            "source_verification": source_verification,
            "split": "development",
        }
        digest = _write_no_replace(
            args.output / "report.json",
            _canonical_bytes(report),
        )
    finally:
        packet_index.close()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": digest,
                "strict_output_interface_signal": gates[
                    "strict_output_interface_signal"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
