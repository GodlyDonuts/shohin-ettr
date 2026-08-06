"""Finite-state role projection for the zero-update DIVERGE-NTA2 gate."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import torch

from diverge_ats1_data import BYTE_OFFSET, CLS_ID, ROLE_TO_ID, SegmentTarget
from diverge_ats1_product import tensorize_segments
from diverge_ats1_runtime import ATS1RuntimeError, CompiledSegment, compile_segment


class NTA2ProductError(RuntimeError):
    """A natural source cannot be projected into the frozen transaction grammar."""


def _source_character(byte_id: int) -> str:
    value = int(byte_id)
    if value < BYTE_OFFSET or value >= BYTE_OFFSET + 128:
        raise NTA2ProductError("natural source byte is not ASCII")
    return chr(value - BYTE_OFFSET)


def project_scalar_roles(byte_ids: Sequence[int]) -> tuple[int, ...]:
    """Project one source into OTHER/LHS/ARG/RHS using maximal signed-number runs."""

    if not byte_ids or int(byte_ids[0]) != CLS_ID:
        raise NTA2ProductError("natural source CLS differs")
    characters = [_source_character(value) for value in byte_ids[1:]]
    numeric = [character.isdigit() for character in characters]
    for index, character in enumerate(characters):
        if character == "-" and index + 1 < len(characters) and characters[index + 1].isdigit():
            numeric[index] = True
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(characters):
        if not numeric[index]:
            index += 1
            continue
        start = index
        while index < len(characters) and numeric[index]:
            index += 1
        runs.append((start + 1, index + 1))
    if len(runs) != 3:
        raise NTA2ProductError("natural source does not contain three numeric fields")
    roles = [ROLE_TO_ID["OTHER"]] * len(byte_ids)
    for (start, end), role in zip(
        runs, ("LHS_A", "ARG1", "RHS_A"), strict=True
    ):
        for position in range(start, end):
            roles[position] = ROLE_TO_ID[role]
    return tuple(roles)


@torch.no_grad()
def compile_nta2_segments(
    model: torch.nn.Module,
    segments: Sequence[SegmentTarget],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[tuple[str, int, str], CompiledSegment], dict[str, Any]]:
    model.eval()
    compiled: dict[tuple[str, int, str], CompiledSegment] = {}
    counts: Counter[str] = Counter()
    for start in range(0, len(segments), batch_size):
        batch = list(segments[start : start + batch_size])
        byte_ids, attention, role_targets, operation_targets = tensorize_segments(
            batch, device
        )
        role_logits, operation_logits = model(byte_ids, attention)
        raw_roles = role_logits.argmax(-1).cpu()
        operations = operation_logits.argmax(-1).cpu()
        byte_ids_cpu = byte_ids.cpu()
        attention_cpu = attention.cpu()
        role_targets_cpu = role_targets.cpu()
        operation_targets_cpu = operation_targets.cpu()
        for index, segment in enumerate(batch):
            counts["segments"] += 1
            active = attention_cpu[index]
            length = int(active.sum())
            source = byte_ids_cpu[index, :length].tolist()
            projected = project_scalar_roles(source)
            raw_exact = torch.equal(
                raw_roles[index, :length], role_targets_cpu[index, :length]
            )
            projected_exact = tuple(role_targets_cpu[index, :length].tolist()) == projected
            operation_exact = bool(
                operations[index] == operation_targets_cpu[index]
            )
            counts["raw_role_exact"] += raw_exact
            counts["projected_role_exact"] += projected_exact
            counts["operation_exact"] += operation_exact
            try:
                packet = compile_segment(source, projected, int(operations[index]))
                compiled[
                    (segment.identity_sha256, segment.step_index, segment.trace_kind)
                ] = packet
                counts["valid"] += 1
            except ATS1RuntimeError:
                counts["invalid"] += 1
    total = max(1, counts["segments"])
    return compiled, {
        "counts": dict(counts),
        "rates": {
            key: counts[key] / total
            for key in (
                "raw_role_exact",
                "projected_role_exact",
                "operation_exact",
                "valid",
            )
        },
    }


__all__ = [
    "NTA2ProductError",
    "compile_nta2_segments",
    "project_scalar_roles",
]
