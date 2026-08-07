#!/usr/bin/env python3
"""Build deterministic CWC1 wrappers around frozen NPL2 WORLD programs."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Mapping, Sequence

from diverge_cwc1_data import (
    MAX_SOURCE_BYTES,
    _NEGATIVE,
    _POSITIVE,
    counterfactual_source,
)
from diverge_npl1_data import parse_program_surface, validate_natural_public_record


SCHEMA = "shohin-diverge-cwc1-npl2-wrapper-v1"


class CWC1NPL2DataError(RuntimeError):
    """Raised when a composed WORLD wrapper violates the frozen contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload: object) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )


def _word(identity: str, index: int) -> str:
    digest = hashlib.sha256(f"cwc1-npl2:{identity}:{index}".encode("ascii")).digest()
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    return "".join(
        consonants[digest[2 * offset] % len(consonants)]
        + vowels[digest[2 * offset + 1] % len(vowels)]
        for offset in range(7)
    )


def _program_text(
    initial: tuple[int, int],
    symbols: Sequence[int],
    aliases: Sequence[str],
    registers: tuple[str, str],
) -> str:
    sequence = " | ".join(aliases[index] for index in symbols)
    return (
        f"Begin with {registers[0]} = {initial[0]} and {registers[1]} = "
        f"{initial[1]}. Execute aliases in order: {sequence}."
    )


def _decoy_program(
    *,
    initial: tuple[int, int],
    symbols: tuple[int, ...],
    aliases: Sequence[str],
    registers: tuple[str, str],
    serial: int,
) -> dict[str, object]:
    decoy_initial = (
        ((initial[0] - 1 + 17 + serial % 31) % 96) + 1,
        ((initial[1] - 1 + 29 + serial % 23) % 96) + 1,
    )
    if decoy_initial == initial:
        raise CWC1NPL2DataError("decoy initial state matches the true state")
    decoy_symbols = tuple(
        (symbol + 1 + ((serial + position) % 7)) % len(aliases)
        for position, symbol in enumerate(symbols)
    )
    if decoy_symbols == symbols:
        raise CWC1NPL2DataError("decoy operation stream matches the true stream")
    return {
        "source_text": _program_text(
            decoy_initial, decoy_symbols, aliases, registers
        ),
        "initial_state": list(decoy_initial),
        "symbols": list(decoy_symbols),
        "is_true": False,
    }


def _wrapper_record(
    *,
    split: str,
    episode: Mapping[str, Any],
    phase: str,
    phase_index: int,
    program: Mapping[str, Any],
    serial: int,
) -> dict[str, Any]:
    aliases = tuple(str(value) for value in episode["aliases"])
    registers_raw = tuple(str(value) for value in episode["register_names"])
    if len(aliases) != 8 or len(registers_raw) != 2:
        raise CWC1NPL2DataError("NPL2 symbol geometry differs")
    registers = (registers_raw[0], registers_raw[1])
    initial, symbols = parse_program_surface(program, aliases, registers)
    identity = f"{split}:{episode['episode_id']}:{phase}:{phase_index}:{program['program_id']}"
    labels = (_word(identity, 0), _word(identity, 1))
    if labels[0] == labels[1] or len(labels[0]) != len(labels[1]):
        raise CWC1NPL2DataError("candidate labels are not swappable")

    pair_index = serial % 64
    cycle = serial // 64
    positive_index, negative_index = divmod(pair_index, 8)
    target_position = (pair_index + cycle) % 2
    true_program = {
        "source_text": str(program["source_text"]),
        "initial_state": list(initial),
        "symbols": list(symbols),
        "is_true": True,
    }
    decoy_program = _decoy_program(
        initial=initial,
        symbols=symbols,
        aliases=aliases,
        registers=registers,
        serial=serial,
    )
    candidate_programs = [decoy_program, decoy_program]
    candidate_programs[target_position] = true_program
    candidate_programs[1 - target_position] = decoy_program

    positive = _POSITIVE[positive_index].format(label=labels[target_position])
    negative = _NEGATIVE[negative_index].format(label=labels[1 - target_position])
    rng = random.Random(int.from_bytes(hashlib.sha256(identity.encode("ascii")).digest()[:8]))
    directive_order = rng.randrange(2)
    first, second = (positive, negative) if directive_order == 0 else (negative, positive)
    directive = f"Directive: {first}; {second}."
    blocks = [
        f"Candidate {labels[index]}: {candidate_programs[index]['source_text']}"
        for index in range(2)
    ]
    directive_position = rng.randrange(2)
    if directive_position == 0:
        source = f"{directive} {blocks[0]} {blocks[1]}"
        directive_bounds = (0, len(directive))
        first_left = len(directive) + 1
    else:
        source = f"{blocks[0]} {blocks[1]} {directive}"
        directive_bounds = (len(blocks[0]) + len(blocks[1]) + 2, len(source))
        first_left = 0
    candidate_bounds = []
    cursor = first_left
    for block in blocks:
        candidate_bounds.append((cursor, cursor + len(block)))
        cursor += len(block) + 1
    if len(source.encode("ascii")) + 1 > MAX_SOURCE_BYTES:
        raise CWC1NPL2DataError("composed CWC1 source exceeds the frozen width")

    row: dict[str, Any] = {
        "schema": SCHEMA,
        "split": split,
        "serial": serial,
        "episode_id": str(episode["episode_id"]),
        "phase": phase,
        "phase_index": phase_index,
        "program_id": str(program["program_id"]),
        "program_source_sha256": str(program["source_sha256"]),
        "renderer": [positive_index, negative_index],
        "directive_order": directive_order,
        "directive_position": directive_position,
        "source_text": source,
        "source_sha256": sha256_bytes(source.encode("ascii")),
        "directive_bounds": list(directive_bounds),
        "candidate_bounds": [list(value) for value in candidate_bounds],
        "candidate_labels": list(labels),
        "aliases": list(aliases),
        "registers": list(registers),
        "candidate_programs": candidate_programs,
        "target_position": target_position,
    }
    row["counterfactual_sha256"] = sha256_bytes(
        counterfactual_source(row).encode("ascii")
    )
    row["identity_sha256"] = canonical_sha256(row)
    validate_wrapper_record(row)
    return row


def build_wrapper_records(
    public: Sequence[Mapping[str, Any]], *, split: str
) -> list[dict[str, Any]]:
    rows = []
    serial = 0
    for episode in public:
        validate_natural_public_record(episode)
        for phase in ("acquisition", "transfer"):
            for phase_index, program in enumerate(episode[phase]):
                rows.append(
                    _wrapper_record(
                        split=split,
                        episode=episode,
                        phase=phase,
                        phase_index=phase_index,
                        program=program,
                        serial=serial,
                    )
                )
                serial += 1
    return rows


def validate_wrapper_record(row: Mapping[str, Any]) -> None:
    if row.get("schema") != SCHEMA:
        raise CWC1NPL2DataError("wrapper schema differs")
    source = str(row["source_text"])
    if sha256_bytes(source.encode("ascii")) != row["source_sha256"]:
        raise CWC1NPL2DataError("wrapper source commitment differs")
    if len(source.encode("ascii")) + 1 > MAX_SOURCE_BYTES:
        raise CWC1NPL2DataError("wrapper source width differs")
    labels = tuple(str(value) for value in row["candidate_labels"])
    if len(labels) != 2 or len(labels[0]) != len(labels[1]):
        raise CWC1NPL2DataError("wrapper label geometry differs")
    bounds = tuple(tuple(int(value) for value in item) for item in row["candidate_bounds"])
    if len(bounds) != 2 or any(not (0 <= left < right <= len(source)) for left, right in bounds):
        raise CWC1NPL2DataError("wrapper candidate bounds differ")
    for label, (left, right) in zip(labels, bounds, strict=True):
        if label not in source[left:right]:
            raise CWC1NPL2DataError("wrapper label is absent from its candidate")
    programs = row["candidate_programs"]
    if len(programs) != 2 or sum(bool(item["is_true"]) for item in programs) != 1:
        raise CWC1NPL2DataError("wrapper whole-world geometry differs")
    target = int(row["target_position"])
    if not programs[target]["is_true"] or programs[1 - target]["is_true"]:
        raise CWC1NPL2DataError("wrapper target does not identify the true world")
    if counterfactual_source(row) == source:
        raise CWC1NPL2DataError("wrapper counterfactual is inert")


def audit_wrapper_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise CWC1NPL2DataError("wrapper board is empty")
    renderers: dict[str, list[int]] = {}
    target_counts = [0, 0]
    identities = set()
    sources = set()
    labels = set()
    maximum_source_bytes = 0
    for row in rows:
        validate_wrapper_record(row)
        identity = str(row["identity_sha256"])
        source = str(row["source_sha256"])
        if identity in identities or source in sources:
            raise CWC1NPL2DataError("wrapper identity or source repeats")
        identities.add(identity)
        sources.add(source)
        for label in row["candidate_labels"]:
            if label in labels:
                raise CWC1NPL2DataError("wrapper candidate label repeats")
            labels.add(label)
        target = int(row["target_position"])
        target_counts[target] += 1
        key = f"{row['renderer'][0]}:{row['renderer'][1]}"
        renderers.setdefault(key, [0, 0])[target] += 1
        maximum_source_bytes = max(
            maximum_source_bytes, len(str(row["source_text"]).encode("ascii")) + 1
        )
    imbalance = {key: abs(value[0] - value[1]) for key, value in renderers.items()}
    return {
        "rows": len(rows),
        "target_counts": target_counts,
        "renderer_counts": dict(sorted(renderers.items())),
        "renderer_max_target_imbalance": max(imbalance.values()),
        "renderers": len(renderers),
        "unique_identities": len(identities),
        "unique_sources": len(sources),
        "unique_candidate_labels": len(labels),
        "maximum_source_bytes": maximum_source_bytes,
        "all_conditions_passed": (
            abs(target_counts[0] - target_counts[1]) <= 1
            and max(imbalance.values()) <= 1
            and len(renderers) == 64
            and maximum_source_bytes <= MAX_SOURCE_BYTES
        ),
    }


__all__ = [
    "CWC1NPL2DataError",
    "SCHEMA",
    "audit_wrapper_records",
    "build_wrapper_records",
    "canonical_sha256",
    "sha256_bytes",
    "validate_wrapper_record",
]
