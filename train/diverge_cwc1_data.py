"""Positionally symmetric counterfactual worlds for DIVERGE-CWC1."""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Mapping, Sequence


TRAIN_SCHEMA = "shohin-diverge-cwc1-training-v1"
BOARD_SCHEMA = "shohin-diverge-cwc1-board-v1"
REPORT_SCHEMA = "shohin-diverge-cwc1-data-report-v1"
TRAIN_SEED = 2026080731
DEVELOPMENT_SEED = 2026080732
CONFIRMATION_SEED = 2026080733
TRAIN_ROWS = 50_000
BOARD_ROWS = 4_096
MAX_SOURCE_BYTES = 1536

_POSITIVE = (
    "execute candidate {label}",
    "candidate {label} is approved",
    "keep candidate {label}",
    "candidate {label} is active",
    "use candidate {label}",
    "candidate {label} is the valid world",
    "accept candidate {label}",
    "candidate {label} governs execution",
)
_NEGATIVE = (
    "ignore candidate {label}",
    "candidate {label} is rejected",
    "discard candidate {label}",
    "candidate {label} is inactive",
    "do not use candidate {label}",
    "candidate {label} is the decoy world",
    "reject candidate {label}",
    "candidate {label} must not execute",
)


class CWC1DataError(RuntimeError):
    """A CWC1 row or split violates the frozen contract."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _word(domain: str, seed: int, serial: int, index: int) -> str:
    digest = hashlib.sha256(
        f"{domain}:{seed}:{serial}:{index}".encode("ascii")
    ).digest()
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    return "".join(
        consonants[digest[2 * offset] % len(consonants)]
        + vowels[digest[2 * offset + 1] % len(vowels)]
        for offset in range(7)
    )


def _pairs(bucket: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (positive, negative)
        for positive in range(8)
        for negative in range(8)
        if (2 * positive + 3 * negative) % 5 == bucket
    )


TRAIN_PAIRS = (*_pairs(0), *_pairs(1), *_pairs(2))
DEVELOPMENT_PAIRS = _pairs(3)
CONFIRMATION_PAIRS = _pairs(4)


def _program_text(
    *,
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


def _replace_in_bounds(
    text: str,
    bounds: tuple[int, int],
    first: str,
    second: str,
) -> str:
    left, right = bounds
    fragment = text[left:right]
    placeholder = "q" * len(first)
    if len(first) != len(second) or placeholder in fragment:
        raise CWC1DataError("CWC1 counterfactual labels are not swappable")
    pattern_first = re.compile(rf"(?<![a-z]){re.escape(first)}(?![a-z])")
    pattern_second = re.compile(rf"(?<![a-z]){re.escape(second)}(?![a-z])")
    fragment = pattern_first.sub(placeholder, fragment)
    fragment = pattern_second.sub(first, fragment)
    fragment = fragment.replace(placeholder, second)
    return text[:left] + fragment + text[right:]


def counterfactual_source(record: Mapping[str, Any]) -> str:
    labels = tuple(str(value) for value in record["candidate_labels"])
    bounds_raw = tuple(int(value) for value in record["directive_bounds"])
    return _replace_in_bounds(
        str(record["source_text"]),
        (bounds_raw[0], bounds_raw[1]),
        labels[0],
        labels[1],
    )


def _record(
    *,
    split: str,
    seed: int,
    serial: int,
    pair: tuple[int, int],
    rng: random.Random,
) -> dict[str, Any]:
    aliases = tuple(_word("alias", seed, serial, index) for index in range(8))
    registers_raw = tuple(_word("register", seed, serial, index) for index in range(2))
    labels = tuple(_word("candidate", seed, serial, index) for index in range(2))
    if len(set((*aliases, *registers_raw, *labels))) != 12:
        raise CWC1DataError("CWC1 generated names collide")
    registers = (registers_raw[0], registers_raw[1])
    depth = rng.randrange(3, 21)
    candidates = []
    for candidate in range(2):
        initial = (rng.randrange(1, 97), rng.randrange(1, 97))
        symbols = tuple(rng.randrange(8) for _ in range(depth))
        candidates.append(
            {
                "initial_state": list(initial),
                "symbols": list(symbols),
                "program_text": _program_text(
                    initial=initial,
                    symbols=symbols,
                    aliases=aliases,
                    registers=registers,
                ),
            }
        )
    if candidates[0]["initial_state"] == candidates[1]["initial_state"]:
        candidates[1]["initial_state"][0] = (
            int(candidates[1]["initial_state"][0]) % 96
        ) + 1
        candidates[1]["program_text"] = _program_text(
            initial=tuple(candidates[1]["initial_state"]),  # type: ignore[arg-type]
            symbols=candidates[1]["symbols"],
            aliases=aliases,
            registers=registers,
        )
    block_order = [0, 1]
    rng.shuffle(block_order)
    target_position = serial % 2
    valid_label_index = block_order[target_position]
    decoy_label_index = 1 - valid_label_index
    positive = _POSITIVE[pair[0]].format(label=labels[valid_label_index])
    negative = _NEGATIVE[pair[1]].format(label=labels[decoy_label_index])
    directive_order = rng.randrange(2)
    first, second = (positive, negative) if directive_order == 0 else (negative, positive)
    directive = f"Directive: {first}; {second}."
    blocks = [
        f"Candidate {labels[index]}: {candidates[index]['program_text']}"
        for index in block_order
    ]
    directive_position = rng.randrange(2)
    if directive_position == 0:
        source = f"{directive} {blocks[0]} {blocks[1]}"
        directive_bounds = (0, len(directive))
        first_left = len(directive) + 1
    else:
        source = f"{blocks[0]} {blocks[1]} {directive}"
        first_left = 0
        directive_bounds = (len(blocks[0]) + len(blocks[1]) + 2, len(source))
    candidate_bounds = []
    cursor = first_left
    for block in blocks:
        candidate_bounds.append((cursor, cursor + len(block)))
        cursor += len(block) + 1
    record: dict[str, Any] = {
        "schema": TRAIN_SCHEMA if split == "train" else BOARD_SCHEMA,
        "split": split,
        "seed": seed,
        "serial": serial,
        "renderer": [pair[0], pair[1]],
        "directive_order": directive_order,
        "directive_position": directive_position,
        "source_text": source,
        "source_sha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
        "directive_bounds": list(directive_bounds),
        "candidate_bounds": [list(value) for value in candidate_bounds],
        "candidate_labels": [labels[index] for index in block_order],
        "aliases": list(aliases),
        "registers": list(registers),
        "candidates": [candidates[index] for index in block_order],
        "target_position": target_position,
    }
    record["counterfactual_sha256"] = hashlib.sha256(
        counterfactual_source(record).encode("ascii")
    ).hexdigest()
    record["identity_sha256"] = canonical_sha256(record)
    validate_record(record)
    return record


def generate_records(*, split: str, seed: int, count: int) -> list[dict[str, Any]]:
    contract = {
        "train": (TRAIN_SEED, TRAIN_ROWS, TRAIN_PAIRS),
        "development": (DEVELOPMENT_SEED, BOARD_ROWS, DEVELOPMENT_PAIRS),
        "confirmation": (CONFIRMATION_SEED, BOARD_ROWS, CONFIRMATION_PAIRS),
    }
    try:
        expected_seed, expected_count, pairs = contract[split]
    except KeyError as error:
        raise CWC1DataError("CWC1 split differs") from error
    if seed != expected_seed or count != expected_count:
        raise CWC1DataError("CWC1 split geometry differs")
    return [
        _record(
            split=split,
            seed=seed,
            serial=serial,
            pair=pairs[serial % len(pairs)],
            rng=random.Random(canonical_sha256(["cwc1", split, seed, serial])),
        )
        for serial in range(count)
    ]


def validate_record(record: Mapping[str, Any]) -> None:
    split = str(record.get("split"))
    schema = TRAIN_SCHEMA if split == "train" else BOARD_SCHEMA
    if record.get("schema") != schema:
        raise CWC1DataError("CWC1 schema differs")
    text = str(record["source_text"])
    if not text or len(text.encode("ascii")) + 1 > MAX_SOURCE_BYTES:
        raise CWC1DataError("CWC1 source width differs")
    if hashlib.sha256(text.encode("ascii")).hexdigest() != record["source_sha256"]:
        raise CWC1DataError("CWC1 source commitment differs")
    bounds = tuple(tuple(int(value) for value in item) for item in record["candidate_bounds"])
    directive = tuple(int(value) for value in record["directive_bounds"])
    if len(bounds) != 2 or any(not (0 <= left < right <= len(text)) for left, right in bounds):
        raise CWC1DataError("CWC1 candidate bounds differ")
    if not (0 <= directive[0] < directive[1] <= len(text)):
        raise CWC1DataError("CWC1 directive bounds differ")
    labels = tuple(str(value) for value in record["candidate_labels"])
    if len(labels) != 2 or len(set(labels)) != 2 or len(labels[0]) != len(labels[1]):
        raise CWC1DataError("CWC1 candidate labels differ")
    for label, (left, right) in zip(labels, bounds, strict=True):
        if label not in text[left:right]:
            raise CWC1DataError("CWC1 candidate label is outside its block")
    if int(record["target_position"]) not in (0, 1):
        raise CWC1DataError("CWC1 target differs")
    counterfactual = counterfactual_source(record)
    if hashlib.sha256(counterfactual.encode("ascii")).hexdigest() != record["counterfactual_sha256"]:
        raise CWC1DataError("CWC1 counterfactual commitment differs")
    if counterfactual == text:
        raise CWC1DataError("CWC1 counterfactual is unchanged")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise CWC1DataError("CWC1 record identity differs")


def overlap_report(*groups: tuple[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    sources = {name: {str(row["source_sha256"]) for row in rows} for name, rows in groups}
    identities = {name: {str(row["identity_sha256"]) for row in rows} for name, rows in groups}
    names = {
        name: {
            value
            for row in rows
            for value in (*row["candidate_labels"], *row["aliases"], *row["registers"])
        }
        for name, rows in groups
    }
    reports = {}
    for left_index, (left, _) in enumerate(groups):
        for right, _ in groups[left_index + 1 :]:
            reports[f"{left}_{right}"] = {
                "source": len(sources[left] & sources[right]),
                "identity": len(identities[left] & identities[right]),
                "name": len(names[left] & names[right]),
            }
    balance = {}
    for name, rows in groups:
        target = {str(index): 0 for index in range(2)}
        directive_order = {str(index): 0 for index in range(2)}
        directive_position = {str(index): 0 for index in range(2)}
        renderers: dict[str, int] = {}
        maximum_source_bytes = 0
        for row in rows:
            target[str(int(row["target_position"]))] += 1
            directive_order[str(int(row["directive_order"]))] += 1
            directive_position[str(int(row["directive_position"]))] += 1
            renderer = ":".join(str(value) for value in row["renderer"])
            renderers[renderer] = renderers.get(renderer, 0) + 1
            maximum_source_bytes = max(
                maximum_source_bytes, len(str(row["source_text"]).encode("ascii")) + 1
            )
        balance[name] = {
            "target_position": target,
            "directive_order": directive_order,
            "directive_position": directive_position,
            "renderer": dict(sorted(renderers.items())),
            "maximum_source_bytes": maximum_source_bytes,
        }
    exact_target_balance = all(
        report["target_position"]["0"] == report["target_position"]["1"]
        for report in balance.values()
    )
    return {
        "schema": REPORT_SCHEMA,
        "rows": {name: len(rows) for name, rows in groups},
        "overlaps": reports,
        "balance": balance,
        "exact_target_balance": exact_target_balance,
        "all_zero": all(value == 0 for report in reports.values() for value in report.values()),
    }


__all__ = [
    "BOARD_ROWS",
    "CONFIRMATION_PAIRS",
    "CONFIRMATION_SEED",
    "CWC1DataError",
    "DEVELOPMENT_PAIRS",
    "DEVELOPMENT_SEED",
    "MAX_SOURCE_BYTES",
    "TRAIN_PAIRS",
    "TRAIN_ROWS",
    "TRAIN_SEED",
    "canonical_sha256",
    "counterfactual_source",
    "generate_records",
    "overlap_report",
    "validate_record",
]
