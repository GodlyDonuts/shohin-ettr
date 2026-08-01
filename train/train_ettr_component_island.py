#!/usr/bin/env python3
"""Fit one ETTR component behind an exact stop-gradient interface.

This trainer is deliberately not an autonomous reasoning claim.  Offline
packet and transaction labels may enter the selected component's training
loss, but they are absent from the ordinary compiler/reactor/reader inference
path and from the held-out autonomous evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    GenericTransactionReactor,
    ReactorTrace,
    TransactionPolicy,
    TypedTheoryState,
)
from ettr_checkpoint import load_ettr_checkpoint
from ettr_data_contract import ETTRContinuationBatch
from ettr_objectives import (
    ETTRCausalQueryPair,
    ETTRObjectiveConfig,
    ETTRPacketTargets,
    _causal_query_binding_loss,
)
from ettr_optimization import ETTROptimizerBundle
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_v3 import (
    _build_model,
    _parameter_sha256,
    _read_hash_bound_json,
    _validate_checkpoint_cursor,
    _validate_run_contract,
)
from probe_ettr_causal_queries import _depth_bucket, _pair_rows, _summary
from probe_ettr_oracle_interfaces import (
    _arm_batch,
    _count_summary,
    _merge_counts,
    packet_targets_to_state,
    policy_masks,
    target_policy,
)


RUN_SCHEMA = "shohin-ettr-component-island-run-v1"
REPORT_SCHEMA = "shohin-ettr-component-island-report-v1"
_COMPONENTS = ("compiler", "reactor", "reader")
_REACTOR_REDUCTIONS = ("decision-mean", "head-class-balanced")
_READER_REDUCTIONS = ("mixed-mean", "balanced-logmeanexp")
_READER_STATE_SOURCES = ("oracle", "autonomous")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_POLICY_FIELDS = (
    "opcode",
    "source",
    "target",
    "relation",
    "type_index",
    "value_code",
)


class ETTRComponentIslandError(RuntimeError):
    """A component-island custody or optimization contract failed."""


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
        raise ETTRComponentIslandError("component output destination differs")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise ETTRComponentIslandError(
            "refusing an existing or unsafe component output"
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


def select_trainable_component(
    model: EndogenousTypedTheoryReactorGPT,
    component: str,
) -> dict[str, object]:
    """Freeze the complete system except one exact architecture module."""

    if component not in _COMPONENTS:
        raise ETTRComponentIslandError("unknown ETTR component island")
    modules = {
        "compiler": model.compiler,
        "reactor": model.reactor,
        "reader": model.query_reader,
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = modules[component]
    for parameter in selected.parameters():
        parameter.requires_grad_(True)
    selected_ids = {id(parameter) for parameter in selected.parameters()}
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if not selected_ids or trainable_ids != selected_ids:
        raise ETTRComponentIslandError(
            "component trainable parameter ownership differs"
        )
    frozen_modules = tuple(name for name in modules if name != component)
    return {
        "component": component,
        "frozen_architecture_modules": list(frozen_modules),
        "frozen_base": True,
        "trainable_parameters": sum(
            parameter.numel() for parameter in selected.parameters()
        ),
        "trainable_tensors": len(selected_ids),
    }


def _component_module(
    model: EndogenousTypedTheoryReactorGPT,
    component: str,
) -> torch.nn.Module:
    try:
        return {
            "compiler": model.compiler,
            "reactor": model.reactor,
            "reader": model.query_reader,
        }[component]
    except KeyError as exc:
        raise ETTRComponentIslandError(
            "unknown ETTR component island"
        ) from exc


def load_component_warm_start(
    model: EndogenousTypedTheoryReactorGPT,
    component: str,
    path: Path,
    *,
    expected_sha256: str,
) -> str:
    """Load one hash-bound component artifact without touching other weights."""

    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or _HEX64.fullmatch(expected_sha256) is None
    ):
        raise ETTRComponentIslandError(
            "component warm-start custody differs"
        )
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise ETTRComponentIslandError(
            "component warm-start hash differs"
        )
    module = _component_module(model, component)
    device = next(module.parameters()).device
    state = load_file(str(path), device=str(device))
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ETTRComponentIslandError(
            "component warm-start tensors differ"
        ) from exc
    return observed


def _masked_categorical_nll(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if not bool(mask.any()):
        return None
    selected = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -selected[mask].float().clamp_min(1e-7).log().mean()


def _masked_categorical_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if not bool(mask.any()):
        return None
    return F.cross_entropy(logits[mask].float(), targets[mask])


def _masked_class_balanced_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    """Average supported target classes equally within one actuator head."""

    if not bool(mask.any()):
        return None
    losses = F.cross_entropy(
        logits.float(),
        targets,
        reduction="none",
    )
    class_count = logits.shape[-1]
    numerators = torch.zeros(
        class_count,
        dtype=losses.dtype,
        device=losses.device,
    ).scatter_add_(
        0,
        targets.flatten(),
        (losses * mask).flatten(),
    )
    counts = torch.zeros_like(numerators).scatter_add_(
        0,
        targets.flatten(),
        mask.flatten().to(losses.dtype),
    )
    present = counts.gt(0)
    return (numerators[present] / counts[present]).mean()


def _reactor_policy_logits(
    reactor: GenericTransactionReactor,
    state: TypedTheoryState,
    *,
    command_hidden: torch.Tensor,
    command_attention_mask: torch.Tensor,
    hard: bool = False,
) -> tuple[TransactionPolicy, Mapping[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if name in captured:
                raise ETTRComponentIslandError(
                    f"reactor emitted {name} more than once"
                )
            captured[name] = output

        return hook

    modules = {
        "opcode": reactor.opcode_head,
        "source_query": reactor.source_query,
        "target_query": reactor.target_query,
        "slot_key": reactor.slot_key,
        "relation": reactor.relation_head,
        "type_index": reactor.type_head,
        "value_code": reactor.value_head,
    }
    handles = [
        module.register_forward_hook(capture(name))
        for name, module in modules.items()
    ]
    try:
        policy = reactor.policy(
            state,
            hard=hard,
            command_hidden=command_hidden,
            command_attention_mask=command_attention_mask,
            validate=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(modules):
        raise ETTRComponentIslandError(
            "reactor policy logit capture is incomplete"
        )
    keys = captured["slot_key"]
    logits = {
        "opcode": captured["opcode"].float(),
        "source": torch.einsum(
            "bw,bsw->bs",
            captured["source_query"],
            keys,
        ).float(),
        "target": torch.einsum(
            "bw,bsw->bs",
            captured["target_query"],
            keys,
        ).float(),
        "relation": captured["relation"].float(),
        "type_index": captured["type_index"].float(),
        "value_code": captured["value_code"].float(),
    }
    probabilities = {
        "opcode": policy.opcode_probabilities,
        "source": policy.source_probabilities,
        "target": policy.target_probabilities,
        "relation": policy.relation_probabilities,
        "type_index": policy.type_probabilities,
        "value_code": policy.value_probabilities,
    }
    for name, field_logits in logits.items():
        if field_logits.shape != probabilities[name].shape:
            raise ETTRComponentIslandError(
                f"reactor {name} logit geometry differs"
            )
    return policy, logits


def _balanced_binary_nll(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if not bool(mask.any()):
        return None
    probabilities = probabilities.float().clamp(1e-7, 1.0 - 1e-7)
    targets = targets.bool()
    losses = -torch.where(
        targets,
        probabilities.log(),
        (1.0 - probabilities).log(),
    )
    classes = []
    for support in (mask & targets, mask & ~targets):
        if bool(support.any()):
            classes.append(losses[support].mean())
    if not classes:
        return None
    return torch.stack(classes).mean()


def compiler_packet_loss(
    prediction: TypedTheoryState,
    targets: ETTRPacketTargets,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Balanced direct packet supervision for the WORLD compiler."""

    categorical = targets.slot_mask & targets.active
    values = {
        "value_code": _masked_categorical_nll(
            prediction.value_probabilities,
            targets.value_code,
            categorical,
        ),
        "type_index": _masked_categorical_nll(
            prediction.type_probabilities,
            targets.type_index,
            categorical,
        ),
        "active": _balanced_binary_nll(
            prediction.active,
            targets.active,
            targets.slot_mask,
        ),
        "root": _balanced_binary_nll(
            prediction.root,
            targets.root,
            targets.slot_mask,
        ),
        "relations": _balanced_binary_nll(
            prediction.relations,
            targets.relations,
            targets.relation_mask,
        ),
        "committed": _balanced_binary_nll(
            prediction.committed,
            targets.committed,
            torch.ones_like(targets.committed),
        ),
        "halted": _balanced_binary_nll(
            prediction.halted,
            targets.halted,
            torch.ones_like(targets.halted),
        ),
    }
    present = [value for value in values.values() if value is not None]
    if not present:
        raise ETTRComponentIslandError("compiler packet loss has no support")
    return torch.stack(present).mean(), {
        name: float(value.detach().cpu())
        for name, value in values.items()
        if value is not None
    }


def _reader_pairs_from_logits(
    logits: torch.Tensor,
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
    targets = batch.episodes.query.targets.gather(
        1,
        batch.episodes.query_read_index[:, None],
    ).squeeze(1)
    return {
        "world": ETTRCausalQueryPair(
            correct_logits=logits.index_select(0, world_target),
            foil_logits=logits.index_select(0, world_command),
            correct_target=targets.index_select(0, world_target),
            foil_target=targets.index_select(0, world_command),
        ),
        "command": ETTRCausalQueryPair(
            correct_logits=logits.index_select(0, command_target),
            foil_logits=logits.index_select(0, command_packet),
            correct_target=targets.index_select(0, command_target),
            foil_target=targets.index_select(0, command_packet),
        ),
    }


def _compiler_loss(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        hidden = model._encode_to_stage(batch.episodes.world.tokens, pos=0)
    prediction = model.compiler(
        hidden.detach(),
        attention_mask=batch.episodes.world.attention_mask,
        hard=False,
    )
    return compiler_packet_loss(prediction, batch.packet_targets)


def _reactor_loss(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    *,
    reduction: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if reduction not in _REACTOR_REDUCTIONS:
        raise ETTRComponentIslandError("reactor reduction differs")
    loss_function = (
        _masked_class_balanced_cross_entropy
        if reduction == "head-class-balanced"
        else _masked_categorical_cross_entropy
    )
    targets = batch.transaction_targets
    masks = policy_masks(targets)
    state = packet_targets_to_state(
        batch.packet_targets,
        model.config,
        step=0,
        dtype=next(model.reactor.parameters()).dtype,
    )
    with torch.no_grad():
        command_hidden = model._encode_to_stage(
            batch.episodes.command.tokens,
            pos=0,
        )
    field_losses: dict[str, list[torch.Tensor]] = {
        name: [] for name in _POLICY_FIELDS
    }
    for step in range(targets.opcode.shape[1]):
        _prediction, logits = _reactor_policy_logits(
            model.reactor,
            state,
            command_hidden=command_hidden.detach(),
            command_attention_mask=batch.episodes.command.attention_mask,
        )
        for name in _POLICY_FIELDS:
            loss = loss_function(
                logits[name],
                getattr(targets, name)[:, step],
                masks[name][:, step],
            )
            if loss is not None:
                field_losses[name].append(loss)
        with torch.no_grad():
            state = model.reactor.apply(
                state,
                target_policy(
                    targets,
                    model.config,
                    step,
                    dtype=state.active.dtype,
                ),
                hard=True,
                validate=False,
            ).detached_clone()
    means = {
        name: torch.stack(values).mean()
        for name, values in field_losses.items()
        if values
    }
    if not means:
        raise ETTRComponentIslandError("reactor loss has no support")
    return torch.stack(tuple(means.values())).mean(), {
        name: float(value.detach().cpu()) for name, value in means.items()
    }


def _reader_logits(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    state: TypedTheoryState,
    *,
    injection: str,
    trace: ReactorTrace | None = None,
) -> torch.Tensor:
    if injection not in {
        "stage",
        "late",
        "postnorm",
        "postnorm-scaled",
    }:
        raise ETTRComponentIslandError("reader injection geometry differs")
    with torch.no_grad():
        query_hidden = model._encode_to_stage(
            batch.episodes.query.tokens,
            pos=0,
        )
    read = model.query_reader(
        query_hidden.detach(),
        state,
        trace=trace,
        attention_mask=batch.episodes.query.attention_mask,
    )
    if injection == "stage":
        hidden = model._decode_from_stage(
            query_hidden.detach() + read,
            pos=0,
        )
        hidden = model.base.norm(hidden)
    elif injection == "late":
        with torch.no_grad():
            decoded = model._decode_from_stage(
                query_hidden.detach(),
                pos=0,
            )
        hidden = decoded.detach() + read
        hidden = model.base.norm(hidden)
    else:
        with torch.no_grad():
            decoded = model._decode_from_stage(
                query_hidden.detach(),
                pos=0,
            )
            normalized = model.base.norm(decoded)
        scale = (
            read.shape[-1] ** -0.5
            if injection == "postnorm-scaled"
            else 1.0
        )
        hidden = normalized.detach() + scale * read
    logits = model.base.head(hidden)
    return logits.gather(
        1,
        batch.episodes.query_read_index[:, None, None].expand(
            -1,
            1,
            logits.shape[-1],
        ),
    ).squeeze(1)


def _reader_state(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    *,
    source: str,
) -> tuple[TypedTheoryState, ReactorTrace | None]:
    if source == "oracle":
        return (
            packet_targets_to_state(
                batch.terminal_packet_targets,
                model.config,
                step=batch.transaction_targets.opcode.shape[1],
                dtype=next(model.query_reader.parameters()).dtype,
            ),
            None,
        )
    if source != "autonomous":
        raise ETTRComponentIslandError("reader state source differs")
    with torch.no_grad():
        state = model.compile_world(
            batch.episodes.world.tokens,
            attention_mask=batch.episodes.world.attention_mask,
            hard=True,
        )
        state, trace = model.execute(
            state,
            steps=batch.transaction_targets.opcode.shape[1],
            hard=True,
            command_idx=batch.episodes.command.tokens,
            command_attention_mask=batch.episodes.command.attention_mask,
        )
    return state.detached_clone(), trace


def _reader_loss(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    *,
    injection: str,
    state_source: str,
    reduction: str = "mixed-mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    state, trace = _reader_state(
        model,
        batch,
        source=state_source,
    )
    read_logits = _reader_logits(
        model,
        batch,
        state,
        injection=injection,
        trace=trace,
    )
    targets = batch.episodes.query.targets.gather(
        1,
        batch.episodes.query_read_index[:, None],
    ).squeeze(1)
    factual = F.cross_entropy(read_logits.float(), targets)
    pairs = _reader_pairs_from_logits(read_logits, batch)
    world = _causal_query_binding_loss(
        pairs["world"],
        margin=1.0,
        reduction=reduction,
    )[0]
    command = _causal_query_binding_loss(
        pairs["command"],
        margin=1.0,
        reduction=reduction,
    )[0]
    losses = {
        "factual": factual,
        "world_binding": world,
        "command_binding": command,
    }
    return torch.stack(tuple(losses.values())).mean(), {
        name: float(value.detach().cpu()) for name, value in losses.items()
    }


def component_loss(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    component: str,
    *,
    reader_injection: str = "stage",
    reader_state_source: str = "oracle",
    reader_reduction: str = "mixed-mean",
    reactor_reduction: str = "decision-mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    if component == "compiler":
        return _compiler_loss(model, batch)
    if component == "reactor":
        return _reactor_loss(
            model,
            batch,
            reduction=reactor_reduction,
        )
    if component == "reader":
        return _reader_loss(
            model,
            batch,
            injection=reader_injection,
            state_source=reader_state_source,
            reduction=reader_reduction,
        )
    raise ETTRComponentIslandError("unknown ETTR component island")


def _evaluate_interfaces(
    model: EndogenousTypedTheoryReactorGPT,
    *,
    stream: ETTRV3StreamingRelease,
    packet_index: ETTRDiskPacketSufficiencyIndex,
    device: torch.device,
    data_seed: int,
    max_batches: int,
    reader_injection: str,
    reader_state_source: str = "oracle",
    batch_transform: (
        Callable[[ETTRContinuationBatch], ETTRContinuationBatch] | None
    ) = None,
) -> dict[str, object]:
    model.eval()
    counts: dict[str, dict[str, list[int]]] = {
        "compiler": {},
        "teacher_forced_reactor": {},
    }
    reader_rows: dict[str, list[dict[str, object]]] = {
        "world": [],
        "command": [],
    }
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
        if batch_transform is not None:
            cpu_batch = batch_transform(cpu_batch)
        batch = move_continuation_batch(cpu_batch, device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            compiler, reactor, reader = _arm_batch(model, batch)
            if (
                reader_injection != "stage"
                or reader_state_source != "oracle"
            ):
                terminal, trace = _reader_state(
                    model,
                    batch,
                    source=reader_state_source,
                )
                logits = _reader_logits(
                    model,
                    batch,
                    terminal,
                    injection=reader_injection,
                    trace=trace,
                )
                pairs = _reader_pairs_from_logits(logits, batch)
                (
                    _world_packet,
                    _world_command,
                    world_target,
                    _command_packet,
                    _command_command,
                    command_target,
                ) = batch.causal_rectangles.intervention_indices()
                depths = batch.transaction_targets.step_mask.sum(-1)
                reader = {
                    kind: [
                        row
                        | {
                            "depth": int(depth.detach().cpu()),
                            "depth_bucket": _depth_bucket(
                                int(depth.detach().cpu())
                            ),
                        }
                        for row, depth in zip(
                            _pair_rows(pair),
                            depths.index_select(
                                0,
                                (
                                    world_target
                                    if kind == "world"
                                    else command_target
                                ),
                            ),
                            strict=True,
                        )
                    ]
                    for kind, pair in pairs.items()
                }
        _merge_counts(counts["compiler"], compiler)
        _merge_counts(counts["teacher_forced_reactor"], reactor)
        for kind, rows in reader.items():
            reader_rows[kind].extend(rows)
        observed += 1
    if observed != max_batches:
        raise ETTRComponentIslandError(
            "component development split is too short"
        )
    return {
        "batches": observed,
        "compiler": _count_summary(counts["compiler"]),
        "oracle_terminal_reader": {
            kind: _summary(rows) for kind, rows in reader_rows.items()
        },
        "teacher_forced_reactor": _count_summary(
            counts["teacher_forced_reactor"]
        ),
    }


def _component_state(
    model: EndogenousTypedTheoryReactorGPT,
    component: str,
) -> dict[str, torch.Tensor]:
    module = _component_module(model, component)
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in module.state_dict().items()
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=_COMPONENTS, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--run-contract-sha256", required=True)
    parser.add_argument("--initial-component", type=Path)
    parser.add_argument("--initial-component-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--reactor-reduction",
        choices=_REACTOR_REDUCTIONS,
        default="decision-mean",
    )
    parser.add_argument(
        "--reader-reduction",
        choices=_READER_REDUCTIONS,
        default="mixed-mean",
    )
    parser.add_argument(
        "--reader-injection",
        choices=("stage", "late", "postnorm", "postnorm-scaled"),
        default="stage",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    initial_component = getattr(args, "initial_component", None)
    initial_component_sha256 = getattr(
        args,
        "initial_component_sha256",
        None,
    )
    if (
        _HEX64.fullmatch(args.release_sha256) is None
        or _HEX64.fullmatch(args.checkpoint_sha256) is None
        or _HEX64.fullmatch(args.run_contract_sha256) is None
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.architecture_seed < 2**63
        or not 0 <= args.data_seed < 2**63
        or args.updates < 1
        or getattr(args, "start_position", 0) < 0
        or args.eval_batches < 2
        or args.log_every < 1
        or not math.isfinite(args.learning_rate)
        or not 0.0 < args.learning_rate < 1.0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0.0
        or not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0.0
        or (args.component != "reader" and args.reader_injection != "stage")
        or (
            args.component != "reader"
            and getattr(args, "reader_reduction", "mixed-mean")
            != "mixed-mean"
        )
        or (
            args.component != "reactor"
            and getattr(
                args,
                "reactor_reduction",
                "decision-mean",
            )
            != "decision-mean"
        )
        or ((initial_component is None) != (initial_component_sha256 is None))
        or (
            initial_component_sha256 is not None
            and _HEX64.fullmatch(initial_component_sha256) is None
        )
    ):
        raise ETTRComponentIslandError("component trainer arguments differ")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise ETTRComponentIslandError("component trainer requires CUDA")
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRComponentIslandError("component trainer requires H100")
    if args.output.exists() or args.output.is_symlink():
        raise ETTRComponentIslandError("refusing an existing output directory")
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
    model_config, optimizer_config = _validate_run_contract(
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
    resume_optimizer = ETTROptimizerBundle(model, optimizer_config)
    resumed = load_ettr_checkpoint(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        model=model,
        protected_base=protected,
        optimizer=resume_optimizer,
        scheduler=None,
    )
    _validate_checkpoint_cursor(
        resumed.progress,
        resumed.data_stream,
        run_contract=run_contract,
        stream=stream,
        release_sha256=args.release_sha256,
        protected_step=protected.step,
    )
    del resume_optimizer

    ownership = select_trainable_component(model, args.component)
    warm_start_sha256 = None
    if args.initial_component is not None:
        warm_start_sha256 = load_component_warm_start(
            model,
            args.component,
            args.initial_component,
            expected_sha256=args.initial_component_sha256,
        )
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    packet_index = ETTRDiskPacketSufficiencyIndex(stream.packet_index_root)
    try:
        initial_parameter_sha256 = _parameter_sha256(model)
        initial_component = _component_state(model, args.component)
        initial_component_path = args.output / "component-initial.safetensors"
        save_file(initial_component, initial_component_path)
        os.chmod(initial_component_path, 0o400)
        initial_component_sha256 = _sha256_file(initial_component_path)
        before = _evaluate_interfaces(
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
            reader_injection=args.reader_injection,
        )

        run_receipt = {
            "architecture_seed": args.architecture_seed,
            "checkpoint_sha256": args.checkpoint_sha256,
            "component": args.component,
            "data_seed": args.data_seed,
            "eval_batches": args.eval_batches,
            "gradient_clip": args.gradient_clip,
            "initial_component_sha256": warm_start_sha256,
            "learning_rate": args.learning_rate,
            "oracle_at_autonomous_inference": False,
            "oracle_training_boundary": {
                "compiler": "initial_packet_targets",
                "reactor": "initial_packet_and_prior_transactions_teacher_forced",
                "reader": "exact_terminal_packet",
            }[args.component],
            "ownership": ownership,
            "protected_checkpoint_sha256": protected.checkpoint_sha256,
            "reader_injection": args.reader_injection,
            "reader_reduction": args.reader_reduction,
            "reactor_reduction": args.reactor_reduction,
            "release_file_sha256": args.release_sha256,
            "run_contract_sha256": args.run_contract_sha256,
            "schema": RUN_SCHEMA,
            "source_commit": args.source_commit,
            "start_position": args.start_position,
            "updates": args.updates,
            "weight_decay": args.weight_decay,
        }
        _write_no_replace(
            args.output / "island-contract.json",
            _canonical_bytes(run_receipt),
        )
        _write_no_replace(args.output / "train.jsonl", b"", mode=0o600)

        model.train()
        epoch = 0
        position = args.start_position
        iterator = stream.iter_positioned_batches(
            "train",
            rank=0,
            world_size=1,
            epoch=epoch,
            seed=args.data_seed,
            start_position=position,
        )
        observed_rows = 0
        observed_token_positions = 0
        last_loss = None
        for update in range(1, args.updates + 1):
            try:
                position, cpu_batch = next(iterator)
            except StopIteration:
                epoch += 1
                position = 0
                iterator = stream.iter_positioned_batches(
                    "train",
                    rank=0,
                    world_size=1,
                    epoch=epoch,
                    seed=args.data_seed,
                )
                position, cpu_batch = next(iterator)
            packet_index.verify_train((cpu_batch,))
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(
                model.config,
                ETTRObjectiveConfig(vocab_size=model.base.cfg.vocab_size),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, parts = component_loss(
                    model,
                    batch,
                    args.component,
                    reader_injection=args.reader_injection,
                    reader_reduction=args.reader_reduction,
                    reactor_reduction=args.reactor_reduction,
                )
            if not bool(torch.isfinite(loss)):
                raise ETTRComponentIslandError(
                    "component training loss is non-finite"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            observed_rows += batch.episodes.world.tokens.shape[0]
            observed_token_positions += int(
                batch.episodes.world.attention_mask.sum().detach().cpu()
                + batch.episodes.command.attention_mask.sum().detach().cpu()
                + batch.episodes.query.attention_mask.sum().detach().cpu()
            )
            last_loss = float(loss.detach().cpu())
            if update % args.log_every == 0 or update == args.updates:
                with (args.output / "train.jsonl").open("ab", buffering=0) as log:
                    log.write(
                        _canonical_bytes(
                            {
                                "component": args.component,
                                "epoch": epoch,
                                "gradient_norm_pre_clip": float(
                                    gradient_norm.detach().float().cpu()
                                ),
                                "loss": last_loss,
                                "loss_parts": parts,
                                "position": position,
                                "schema": "shohin-ettr-component-metric-v1",
                                "update": update,
                            }
                        )
                    )
        os.chmod(args.output / "train.jsonl", 0o400)

        after = _evaluate_interfaces(
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
            reader_injection=args.reader_injection,
        )
        final_component = _component_state(model, args.component)
        final_component_path = args.output / "component-final.safetensors"
        save_file(final_component, final_component_path)
        os.chmod(final_component_path, 0o400)
        final_component_sha256 = _sha256_file(final_component_path)
        report = {
            "architecture_seed": args.architecture_seed,
            "checkpoint_sha256": args.checkpoint_sha256,
            "component": args.component,
            "data_seed": args.data_seed,
            "device": {
                "bf16": torch.cuda.is_bf16_supported(),
                "name": torch.cuda.get_device_name(device),
            },
            "evaluation": {"after": after, "before": before},
            "final_component_sha256": final_component_sha256,
            "final_parameter_sha256": _parameter_sha256(model),
            "initial_component_sha256": initial_component_sha256,
            "loaded_component_sha256": warm_start_sha256,
            "initial_parameter_sha256": initial_parameter_sha256,
            "last_loss": last_loss,
            "observed_rows": observed_rows,
            "observed_token_positions": observed_token_positions,
            "oracle_at_autonomous_inference": False,
            "ownership": ownership,
            "protected_checkpoint_sha256": protected.checkpoint_sha256,
            "release_file_sha256": args.release_sha256,
            "release_manifest_sha256": stream.manifest.sha256(),
            "reader_injection": args.reader_injection,
            "reader_reduction": args.reader_reduction,
            "reactor_reduction": args.reactor_reduction,
            "schema": REPORT_SCHEMA,
            "source_commit": args.source_commit,
            "start_position": args.start_position,
            "source_verification": source_verification,
            "updates": args.updates,
        }
        _write_no_replace(
            args.output / "report.json",
            _canonical_bytes(report),
        )
    finally:
        packet_index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
