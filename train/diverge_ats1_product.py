"""Training/evaluation utilities for the bounded DIVERGE-ATS1 gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from diverge_ats1_data import (
    MAX_SEGMENT_BYTES,
    PAD_ID,
    ROLE_TO_ID,
    SegmentTarget,
    build_segments,
    supervisor_states,
)
from diverge_ats1_runtime import (
    ATS1RuntimeError,
    CompiledSegment,
    SourceRoleCompiler,
    TypedState,
    compile_segment,
    execute_step,
    render_typed_state,
    shifted_operation,
)


class ATS1ProductError(RuntimeError):
    """The ATS1 training or evaluation receipt is invalid."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise ATS1ProductError(f"data hash differs for {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ATS1ProductError(f"malformed JSONL line {line_number}") from error
            if not isinstance(row, dict):
                raise ATS1ProductError("JSONL row is not an object")
            rows.append(row)
    if not rows:
        raise ATS1ProductError("ATS1 data is empty")
    return rows


def tensorize_segments(
    segments: Sequence[SegmentTarget],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(segments)
    byte_ids = torch.full(
        (batch, MAX_SEGMENT_BYTES), PAD_ID, dtype=torch.long, device=device
    )
    attention = torch.zeros_like(byte_ids, dtype=torch.bool)
    roles = torch.full_like(byte_ids, ROLE_TO_ID["OTHER"])
    operations = torch.empty(batch, dtype=torch.long, device=device)
    for index, segment in enumerate(segments):
        length = len(segment.byte_ids)
        if length > MAX_SEGMENT_BYTES or len(segment.role_ids) != length:
            raise ATS1ProductError("segment tensor contract differs")
        byte_ids[index, :length] = torch.tensor(segment.byte_ids, device=device)
        roles[index, :length] = torch.tensor(segment.role_ids, device=device)
        attention[index, :length] = True
        operations[index] = segment.operation_id
    return byte_ids, attention, roles, operations


def _expected_state(segment: SegmentTarget, side: str) -> str:
    if side not in {"lhs", "rhs"}:
        raise ATS1ProductError("state side differs")
    prefix = side
    if segment.lhs_symbol is not None:
        value = getattr(segment, f"{prefix}_symbol")
        if value is None:
            raise ATS1ProductError("symbol target is missing")
        return value
    a = getattr(segment, f"{prefix}_a")
    b = getattr(segment, f"{prefix}_b")
    if a is None:
        raise ATS1ProductError("numeric target is missing")
    return a if b is None else f"{a},{b}"


@torch.no_grad()
def compile_segments(
    model: SourceRoleCompiler,
    segments: Sequence[SegmentTarget],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[tuple[str, int, str], CompiledSegment], dict[str, Any]]:
    model.eval()
    compiled: dict[tuple[str, int, str], CompiledSegment] = {}
    counters: Counter[str] = Counter()
    per_family: dict[str, Counter[str]] = defaultdict(Counter)
    total = len(segments)
    for start in range(0, total, batch_size):
        batch = list(segments[start : start + batch_size])
        byte_ids, attention, role_targets, operation_targets = tensorize_segments(
            batch, device
        )
        role_logits, operation_logits = model(byte_ids, attention)
        role_predictions = role_logits.argmax(-1).cpu()
        operation_predictions = operation_logits.argmax(-1).cpu()
        role_targets_cpu = role_targets.cpu()
        operation_targets_cpu = operation_targets.cpu()
        attention_cpu = attention.cpu()
        byte_ids_cpu = byte_ids.cpu()
        for index, segment in enumerate(batch):
            counters["segments"] += 1
            family = ("scalar", "register", "symbolic")[segment.family_id]
            local = per_family[family]
            local["segments"] += 1
            active = attention_cpu[index]
            role_exact = bool(
                torch.equal(
                    role_predictions[index][active],
                    role_targets_cpu[index][active],
                )
            )
            operation_exact = bool(
                operation_predictions[index] == operation_targets_cpu[index]
            )
            counters["role_exact"] += role_exact
            counters["operation_exact"] += operation_exact
            local["role_exact"] += role_exact
            local["operation_exact"] += operation_exact
            length = int(active.sum())
            key = (segment.identity_sha256, segment.step_index, segment.trace_kind)
            try:
                packet = compile_segment(
                    byte_ids_cpu[index, :length].tolist(),
                    role_predictions[index, :length].tolist(),
                    int(operation_predictions[index]),
                )
                counters["valid"] += 1
                local["valid"] += 1
                lhs_exact = render_typed_state(packet.lhs) == _expected_state(segment, "lhs")
                rhs_exact = render_typed_state(packet.rhs_claim) == _expected_state(segment, "rhs")
                args_exact = packet.arguments == tuple(
                    sum((ord(character) - 48) * (10 ** power) for power, character in enumerate(reversed(value)))
                    for value in segment.arguments
                )
                counters["lhs_exact"] += lhs_exact
                counters["rhs_exact"] += rhs_exact
                counters["argument_exact"] += args_exact
                local["lhs_exact"] += lhs_exact
                local["rhs_exact"] += rhs_exact
                local["argument_exact"] += args_exact
                compiled[key] = packet
            except ATS1RuntimeError:
                counters["invalid"] += 1
                local["invalid"] += 1
    metrics = {
        "counts": dict(counters),
        "rates": {
            key: counters[key] / max(1, counters["segments"])
            for key in (
                "role_exact",
                "operation_exact",
                "valid",
                "lhs_exact",
                "rhs_exact",
                "argument_exact",
            )
        },
        "per_family": {
            family: {
                "counts": dict(values),
                "rates": {
                    key: values[key] / max(1, values["segments"])
                    for key in (
                        "role_exact",
                        "operation_exact",
                        "valid",
                        "lhs_exact",
                        "rhs_exact",
                        "argument_exact",
                    )
                },
            }
            for family, values in sorted(per_family.items())
        },
    }
    return compiled, metrics


def _rolled_states(
    rows: Sequence[dict[str, Any]],
    starts: dict[str, TypedState],
) -> dict[str, TypedState]:
    output = dict(starts)
    by_family: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(str(row["identity_sha256"]))
    for identities in by_family.values():
        if len(identities) < 2:
            raise ATS1ProductError("packet-swap control needs two rows per family")
        values = [starts[identity] for identity in identities]
        values = values[-1:] + values[:-1]
        output.update(zip(identities, values))
    return output


def evaluate_replay(
    rows: Sequence[dict[str, Any]],
    compiled: dict[tuple[str, int, str], CompiledSegment],
    *,
    ablation: str = "normal",
) -> dict[str, Any]:
    if ablation not in {"normal", "initial_swap", "operation_shift"}:
        raise ATS1ProductError("unknown ATS1 replay ablation")
    starts: dict[str, TypedState] = {}
    selections: dict[str, int] = {}
    for row in rows:
        identity = str(row["identity_sha256"])
        selection = int(row["error_index"])
        selections[identity] = selection
        key = (identity, selection - 1, "wrong")
        if key in compiled:
            starts[identity] = compiled[key].lhs
    if ablation == "initial_swap" and len(starts) == len(rows):
        starts = _rolled_states(rows, starts)

    totals: Counter[str] = Counter()
    per_family: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    for row in rows:
        totals["rows"] += 1
        family = str(row["family"])
        local = per_family[family]
        local["rows"] += 1
        identity = str(row["identity_sha256"])
        selection = selections[identity]
        expected = supervisor_states(row)
        state = starts.get(identity)
        exact_steps = 0
        invalid = state is None
        if state is not None:
            totals["initial_exact"] += render_typed_state(state) == expected[selection - 1]
            local["initial_exact"] += render_typed_state(state) == expected[selection - 1]
        try:
            if state is None:
                raise ATS1RuntimeError("initial packet missing")
            for step_index in range(selection - 1, int(row["depth"])):
                packet = compiled[(identity, step_index, "wrong")]
                operation = packet.operation_id
                if ablation == "operation_shift":
                    operation = shifted_operation(operation)
                state = execute_step(state, operation, packet.arguments)
                if render_typed_state(state) == expected[step_index + 1]:
                    exact_steps += 1
            terminal = render_typed_state(state)
        except (ATS1RuntimeError, KeyError):
            invalid = True
            terminal = "<INVALID>"
        active_steps = int(row["depth"]) - selection + 1
        trajectory_exact = not invalid and exact_steps == active_steps
        terminal_exact = not invalid and terminal == str(row["answer"])
        totals["invalid"] += invalid
        totals["exact_steps"] += exact_steps
        totals["active_steps"] += active_steps
        totals["trajectory_exact"] += trajectory_exact
        totals["terminal_exact"] += terminal_exact
        local["invalid"] += invalid
        local["exact_steps"] += exact_steps
        local["active_steps"] += active_steps
        local["trajectory_exact"] += trajectory_exact
        local["terminal_exact"] += terminal_exact
        if len(examples) < 24:
            examples.append(
                {
                    "identity_sha256": identity,
                    "family": family,
                    "selection": selection,
                    "prediction": terminal,
                    "target": str(row["answer"]),
                    "terminal_exact": terminal_exact,
                    "trajectory_exact": trajectory_exact,
                    "invalid": invalid,
                }
            )
    return {
        "ablation": ablation,
        "counts": dict(totals),
        "rates": {
            "initial_exact": totals["initial_exact"] / max(1, totals["rows"]),
            "step_exact": totals["exact_steps"] / max(1, totals["active_steps"]),
            "trajectory_exact": totals["trajectory_exact"] / max(1, totals["rows"]),
            "terminal_exact": totals["terminal_exact"] / max(1, totals["rows"]),
            "invalid": totals["invalid"] / max(1, totals["rows"]),
        },
        "per_family": {
            family: {
                "counts": dict(values),
                "rates": {
                    "initial_exact": values["initial_exact"] / max(1, values["rows"]),
                    "step_exact": values["exact_steps"] / max(1, values["active_steps"]),
                    "trajectory_exact": values["trajectory_exact"] / max(1, values["rows"]),
                    "terminal_exact": values["terminal_exact"] / max(1, values["rows"]),
                    "invalid": values["invalid"] / max(1, values["rows"]),
                },
            }
            for family, values in sorted(per_family.items())
        },
        "examples": examples,
    }


@torch.no_grad()
def evaluate_model(
    model: SourceRoleCompiler,
    rows: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    segments = build_segments(rows, trace_kinds=("wrong",))
    compiled, compiler = compile_segments(
        model, segments, device=device, batch_size=batch_size
    )
    normal = evaluate_replay(rows, compiled, ablation="normal")
    initial_swap = evaluate_replay(rows, compiled, ablation="initial_swap")
    operation_shift = evaluate_replay(rows, compiled, ablation="operation_shift")
    rhs_poison_invariant = True
    for packet in compiled.values():
        poisoned = replace(packet, rhs_claim=packet.lhs)
        if (
            execute_step(packet.lhs, packet.operation_id, packet.arguments)
            != execute_step(poisoned.lhs, poisoned.operation_id, poisoned.arguments)
        ):
            rhs_poison_invariant = False
            break
    return {
        "compiler": compiler,
        "replay": {
            "normal": normal,
            "initial_swap": initial_swap,
            "operation_shift": operation_shift,
        },
        "rhs_poison_invariant": rhs_poison_invariant,
    }


__all__ = [
    "ATS1ProductError",
    "compile_segments",
    "evaluate_model",
    "evaluate_replay",
    "load_jsonl",
    "sha256_path",
    "tensorize_segments",
]
