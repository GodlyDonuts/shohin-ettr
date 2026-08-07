#!/usr/bin/env python3
"""Compose confirmed spanless value events with frozen neural law synthesis."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Literal, Mapping, Sequence

import torch

from diverge_eal1_runtime import EpisodeLawPacket, LawCompilation, canonical_sha256
from diverge_nls1_runtime import NeuralLawSynthesizer, hard_rows
from diverge_sve1_runtime import decode_evidence_events


SCHEMA = "shohin-diverge-snl1-runtime-v1"
OPERATIONS = 8
DEMONSTRATIONS = 3
CompileControl = Literal["normal", "one_example", "scrub_outcomes"]


class SNL1RuntimeError(RuntimeError):
    """A spanless neural-law packet violates its frozen contract."""


def _operation_index(text: str, aliases: Sequence[str]) -> int:
    present = []
    for index, alias in enumerate(aliases):
        if not alias.isalpha() or not alias.islower():
            raise SNL1RuntimeError("SNL1 alias carrier differs")
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text):
            present.append(index)
    if len(present) != 1:
        raise SNL1RuntimeError("SNL1 evidence does not bind exactly one alias")
    return present[0]


@torch.no_grad()
def compile_neural_event_laws(
    public: Mapping[str, Any],
    event_sequences: Sequence[Sequence[int]],
    model: NeuralLawSynthesizer,
    *,
    device: torch.device,
    event_owner_sha256: str,
    model_owner_sha256: str,
    text_key: str,
    hash_key: str,
    control: CompileControl = "normal",
) -> LawCompilation:
    """Compile one hard law packet from model-emitted complete value events."""
    if control not in ("normal", "one_example", "scrub_outcomes"):
        raise SNL1RuntimeError("SNL1 compilation control differs")
    aliases = tuple(str(value) for value in public["aliases"])
    evidence = tuple(public["evidence"])
    if (
        len(aliases) != OPERATIONS
        or len(set(aliases)) != OPERATIONS
        or len(evidence) != OPERATIONS * DEMONSTRATIONS
        or len(event_sequences) != len(evidence)
    ):
        raise SNL1RuntimeError("SNL1 episode geometry differs")

    grouped: list[list[tuple[int, int, int, int]]] = [[] for _ in range(OPERATIONS)]
    commitments = []
    for record, sequence in zip(evidence, event_sequences, strict=True):
        try:
            decoded = decode_evidence_events(sequence)
        except RuntimeError:
            return LawCompilation(None, "event_not_complete", tuple(), len(commitments))
        by_role = {int(role): int(value) for role, value in decoded}
        if sorted(by_role) != [0, 1, 2, 3]:
            return LawCompilation(None, "event_not_complete", tuple(), len(commitments))
        grouped[_operation_index(str(record[text_key]), aliases)].append(
            (by_role[0], by_role[1], by_role[2], by_role[3])
        )
        commitments.append(str(record[hash_key]))
    if any(len(value) != DEMONSTRATIONS for value in grouped):
        return LawCompilation(None, "operation_not_complete", tuple(), len(commitments))

    values = torch.tensor(grouped, dtype=torch.long, device=device)
    mask = torch.ones((OPERATIONS, DEMONSTRATIONS), dtype=torch.bool, device=device)
    if control == "one_example":
        mask[:, 1:] = False
    elif control == "scrub_outcomes":
        values[:, :, 2:] = 0
    rows = tuple(hard_rows(logits) for logits in model(values, mask))
    owner_hash = canonical_sha256(
        ["snl1-owner", event_owner_sha256, model_owner_sha256]
    )
    provisional = EpisodeLawPacket(
        aliases=aliases,
        rows=rows,
        evidence_commitments=tuple(commitments),
        reader_state_sha256=owner_hash,
        commitment="",
    )
    packet = replace(provisional, commitment=canonical_sha256(provisional.payload()))
    return LawCompilation(
        packet=packet,
        error=None,
        support_sizes=tuple((1, 1) for _ in range(OPERATIONS)),
        evidence_count=len(evidence),
    )


__all__ = [
    "CompileControl",
    "SCHEMA",
    "SNL1RuntimeError",
    "compile_neural_event_laws",
]
