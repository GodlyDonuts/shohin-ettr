"""Deterministic source-disjoint data for DIVERGE-EWC1."""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Mapping, Sequence


TRAIN_SCHEMA = "shohin-diverge-ewc1-training-v1"
BOARD_SCHEMA = "shohin-diverge-ewc1-board-v1"
REPORT_SCHEMA = "shohin-diverge-ewc1-data-report-v1"
TRAIN_SEED = 2026080721
DEVELOPMENT_SEED = 2026080722
CONFIRMATION_SEED = 2026080723
TRAIN_ROWS = 50_000
BOARD_ROWS = 4_096
MAX_WORLD_BYTES = 768
MAX_NUMERIC_MENTIONS = 8
MAX_ALIAS_OCCURRENCES = 40

_INTEGER = re.compile(r"(?<![A-Za-z0-9_])[0-9]+(?![A-Za-z0-9_])")

_INITIAL_CLAUSES = (
    "Begin with {r0} = {v0} and {r1} = {v1}.",
    "Initialize {r0} to {v0}; initialize {r1} to {v1}.",
    "Starting state: {r1} has {v1}, while {r0} has {v0}.",
    "Use {v0} as the opening value of {r0} and {v1} for {r1}.",
    "At launch, register {r0} contains {v0}; register {r1} contains {v1}.",
    "Let {r1} begin at {v1} and let {r0} begin at {v0}.",
    "The starting assignments are {r0}:{v0} plus {r1}:{v1}.",
    "Before execution, set {r1}={v1}; set {r0}={v0}.",
)

_SEQUENCE_CLAUSES = (
    "Execute aliases in order: {pipe}.",
    "Then apply this alias sequence: {comma}.",
    "Run these operators from first to last: {arrow}.",
    "The program, read left to right, is {slash}.",
    "Process these aliases in sequence: {comma_then}.",
    "Execution order is {space_arrow}.",
    "Follow the ordered operator list {semicolon}.",
    "Use this program from left to right: {colon}.",
)


class EWC1DataError(RuntimeError):
    """An EWC1 source or split violates the frozen contract."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def scan_integer_spans(text: str) -> tuple[tuple[int, int], ...]:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as error:
        raise EWC1DataError("EWC1 WORLD is not ASCII") from error
    return tuple(match.span() for match in _INTEGER.finditer(text))


def scan_symbol_occurrences(
    text: str, symbols: Sequence[str]
) -> tuple[tuple[int, int, int], ...]:
    occurrences: list[tuple[int, int, int]] = []
    if len(set(symbols)) != len(symbols):
        raise EWC1DataError("EWC1 declared symbols collide")
    for symbol_index, symbol in enumerate(symbols):
        if not symbol.isalpha() or not symbol.islower():
            raise EWC1DataError("EWC1 symbol is not a lowercase word")
        pattern = re.compile(rf"(?<![a-z]){re.escape(symbol)}(?![a-z])")
        occurrences.extend(
            (match.start(), match.end(), symbol_index)
            for match in pattern.finditer(text)
        )
    occurrences.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(occurrences)


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


def _renderer_pairs(bucket: int) -> tuple[tuple[int, int], ...]:
    if bucket not in range(5):
        raise EWC1DataError("EWC1 renderer bucket differs")
    return tuple(
        (initial, sequence)
        for initial in range(len(_INITIAL_CLAUSES))
        for sequence in range(len(_SEQUENCE_CLAUSES))
        if (2 * initial + 3 * sequence) % 5 == bucket
    )


TRAIN_PAIRS = (*_renderer_pairs(0), *_renderer_pairs(1), *_renderer_pairs(2))
DEVELOPMENT_PAIRS = _renderer_pairs(3)
CONFIRMATION_PAIRS = _renderer_pairs(4)


def _sequence_fields(aliases: Sequence[str]) -> dict[str, str]:
    return {
        "pipe": " | ".join(aliases),
        "comma": ", ".join(aliases),
        "arrow": " -> ".join(aliases),
        "slash": " / ".join(aliases),
        "comma_then": ", then ".join(aliases),
        "space_arrow": " => ".join(aliases),
        "semicolon": "; ".join(aliases),
        "colon": " : ".join(aliases),
    }


def render_world(
    *,
    initial_family: int,
    sequence_family: int,
    initial_state: tuple[int, int],
    symbol_sequence: Sequence[int],
    aliases: Sequence[str],
    registers: tuple[str, str],
    clause_order: int,
    distractor_form: int,
    distractor_value: int,
    distractor_alias: str,
) -> tuple[str, tuple[int, int]]:
    if initial_family not in range(8) or sequence_family not in range(8):
        raise EWC1DataError("EWC1 renderer family differs")
    if clause_order not in (0, 1) or distractor_form not in range(3):
        raise EWC1DataError("EWC1 clause geometry differs")
    selected = [aliases[index] for index in symbol_sequence]
    initial_clause = _INITIAL_CLAUSES[initial_family].format(
        r0=registers[0],
        r1=registers[1],
        v0=initial_state[0],
        v1=initial_state[1],
    )
    sequence_clause = _SEQUENCE_CLAUSES[sequence_family].format(
        **_sequence_fields(selected)
    )
    if clause_order == 0:
        core = f"{initial_clause} {sequence_clause}"
        sequence_left = len(initial_clause) + 1
    else:
        core = f"{sequence_clause} {initial_clause}"
        sequence_left = 0
    sequence_bounds = (sequence_left, sequence_left + len(sequence_clause))
    if distractor_form == 0:
        return core, sequence_bounds
    if distractor_form == 1:
        prefix = (
            f"Discard diagnostic {distractor_value}; alias {distractor_alias} "
            "is illustrative only. "
        )
        return prefix + core, (
            sequence_bounds[0] + len(prefix),
            sequence_bounds[1] + len(prefix),
        )
    suffix = (
        f" Audit number {distractor_value} and unused alias {distractor_alias} "
        "do not participate."
    )
    return core + suffix, sequence_bounds


def _record(
    *,
    split: str,
    seed: int,
    serial: int,
    pair: tuple[int, int],
    rng: random.Random,
) -> dict[str, Any]:
    aliases = tuple(_word("alias", seed, serial, index) for index in range(8))
    registers = tuple(_word("register", seed, serial, index) for index in range(2))
    if len(set((*aliases, *registers))) != 10:
        raise EWC1DataError("EWC1 generated symbols collide")
    initial_state = (rng.randrange(1, 97), rng.randrange(1, 97))
    while initial_state[1] == initial_state[0]:
        initial_state = (initial_state[0], rng.randrange(1, 97))
    if split == "train":
        depth = rng.randrange(3, 21)
    else:
        depth = rng.randrange(3, 29)
    symbols = tuple(rng.randrange(8) for _ in range(depth))
    clause_order = rng.randrange(2)
    distractor_form = rng.randrange(3)
    distractor_value = rng.randrange(101, 997)
    distractor_symbol = rng.randrange(8)
    text, sequence_bounds = render_world(
        initial_family=pair[0],
        sequence_family=pair[1],
        initial_state=initial_state,
        symbol_sequence=symbols,
        aliases=aliases,
        registers=(registers[0], registers[1]),
        clause_order=clause_order,
        distractor_form=distractor_form,
        distractor_value=distractor_value,
        distractor_alias=aliases[distractor_symbol],
    )
    numeric_spans = scan_integer_spans(text)
    if len(numeric_spans) > MAX_NUMERIC_MENTIONS:
        raise EWC1DataError("EWC1 numeric candidate count exceeds the cap")
    numeric_targets = []
    for value in initial_state:
        matches = [
            index
            for index, (left, right) in enumerate(numeric_spans)
            if text[left:right] == str(value)
        ]
        if len(matches) != 1:
            raise EWC1DataError("EWC1 initial value is not source-identifiable")
        numeric_targets.append(matches[0])
    alias_occurrences = scan_symbol_occurrences(text, aliases)
    if len(alias_occurrences) > MAX_ALIAS_OCCURRENCES:
        raise EWC1DataError("EWC1 alias occurrence count exceeds the cap")
    operation_targets = [
        int(sequence_bounds[0] <= left and right <= sequence_bounds[1])
        for left, right, _ in alias_occurrences
    ]
    selected = tuple(
        symbol_index
        for (_, _, symbol_index), target in zip(
            alias_occurrences, operation_targets, strict=True
        )
        if target
    )
    if selected != symbols:
        raise EWC1DataError("EWC1 selected operation mentions differ")
    record: dict[str, Any] = {
        "schema": TRAIN_SCHEMA if split == "train" else BOARD_SCHEMA,
        "split": split,
        "seed": seed,
        "serial": serial,
        "renderer": [pair[0], pair[1]],
        "clause_order": clause_order,
        "distractor_form": distractor_form,
        "source_text": text,
        "source_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
        "aliases": list(aliases),
        "registers": list(registers),
        "initial_state": list(initial_state),
        "symbols": list(symbols),
        "numeric_targets": numeric_targets,
        "operation_targets": operation_targets,
    }
    record["identity_sha256"] = canonical_sha256(record)
    validate_record(record)
    return record


def generate_records(*, split: str, seed: int, count: int) -> list[dict[str, Any]]:
    expected = {
        "train": (TRAIN_SEED, TRAIN_ROWS, TRAIN_PAIRS),
        "development": (DEVELOPMENT_SEED, BOARD_ROWS, DEVELOPMENT_PAIRS),
        "confirmation": (CONFIRMATION_SEED, BOARD_ROWS, CONFIRMATION_PAIRS),
    }
    try:
        expected_seed, expected_count, pairs = expected[split]
    except KeyError as error:
        raise EWC1DataError("EWC1 split differs") from error
    if seed != expected_seed or count != expected_count:
        raise EWC1DataError("EWC1 frozen split geometry differs")
    output = []
    for serial in range(count):
        rng = random.Random(
            canonical_sha256(["diverge-ewc1-record", split, seed, serial])
        )
        pair = pairs[serial % len(pairs)]
        output.append(
            _record(split=split, seed=seed, serial=serial, pair=pair, rng=rng)
        )
    return output


def validate_record(record: Mapping[str, Any]) -> None:
    split = str(record.get("split"))
    expected_schema = TRAIN_SCHEMA if split == "train" else BOARD_SCHEMA
    if record.get("schema") != expected_schema:
        raise EWC1DataError("EWC1 schema differs")
    text = str(record["source_text"])
    encoded = text.encode("ascii")
    if not encoded or len(encoded) + 1 > MAX_WORLD_BYTES:
        raise EWC1DataError("EWC1 WORLD width differs")
    if hashlib.sha256(encoded).hexdigest() != record["source_sha256"]:
        raise EWC1DataError("EWC1 source commitment differs")
    aliases = tuple(str(value) for value in record["aliases"])
    registers_raw = tuple(str(value) for value in record["registers"])
    if len(aliases) != 8 or len(registers_raw) != 2:
        raise EWC1DataError("EWC1 symbol geometry differs")
    numeric = scan_integer_spans(text)
    occurrences = scan_symbol_occurrences(text, aliases)
    numeric_targets = tuple(int(value) for value in record["numeric_targets"])
    operation_targets = tuple(int(value) for value in record["operation_targets"])
    if (
        len(numeric_targets) != 2
        or len(set(numeric_targets)) != 2
        or any(index not in range(len(numeric)) for index in numeric_targets)
    ):
        raise EWC1DataError("EWC1 numeric supervision differs")
    initial = tuple(int(value) for value in record["initial_state"])
    if tuple(int(text[numeric[index][0] : numeric[index][1]]) for index in numeric_targets) != initial:
        raise EWC1DataError("EWC1 initial-state supervision differs")
    if len(operation_targets) != len(occurrences) or any(
        value not in (0, 1) for value in operation_targets
    ):
        raise EWC1DataError("EWC1 operation supervision differs")
    selected = tuple(
        occurrence[2]
        for occurrence, target in zip(occurrences, operation_targets, strict=True)
        if target
    )
    if selected != tuple(int(value) for value in record["symbols"]):
        raise EWC1DataError("EWC1 operation sequence differs")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise EWC1DataError("EWC1 record identity differs")


def overlap_report(
    train: Sequence[Mapping[str, Any]],
    development: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = {"train": train, "development": development, "confirmation": confirmation}
    identities = {
        name: {str(row["identity_sha256"]) for row in rows}
        for name, rows in groups.items()
    }
    sources = {
        name: {str(row["source_sha256"]) for row in rows}
        for name, rows in groups.items()
    }
    aliases = {
        name: {alias for row in rows for alias in row["aliases"]}
        for name, rows in groups.items()
    }
    registers = {
        name: {symbol for row in rows for symbol in row["registers"]}
        for name, rows in groups.items()
    }
    pairs = (("train", "development"), ("train", "confirmation"), ("development", "confirmation"))
    overlaps = {
        f"{left}_{right}": {
            "identity": len(identities[left] & identities[right]),
            "source": len(sources[left] & sources[right]),
            "alias": len(aliases[left] & aliases[right]),
            "register": len(registers[left] & registers[right]),
        }
        for left, right in pairs
    }
    return {
        "schema": REPORT_SCHEMA,
        "rows": {name: len(rows) for name, rows in groups.items()},
        "renderer_pairs": {
            "train": [list(value) for value in TRAIN_PAIRS],
            "development": [list(value) for value in DEVELOPMENT_PAIRS],
            "confirmation": [list(value) for value in CONFIRMATION_PAIRS],
        },
        "overlaps": overlaps,
        "all_zero": all(
            value == 0
            for report in overlaps.values()
            for value in report.values()
        ),
    }


__all__ = [
    "BOARD_ROWS",
    "BOARD_SCHEMA",
    "CONFIRMATION_PAIRS",
    "CONFIRMATION_SEED",
    "DEVELOPMENT_PAIRS",
    "DEVELOPMENT_SEED",
    "EWC1DataError",
    "MAX_ALIAS_OCCURRENCES",
    "MAX_NUMERIC_MENTIONS",
    "MAX_WORLD_BYTES",
    "REPORT_SCHEMA",
    "TRAIN_PAIRS",
    "TRAIN_ROWS",
    "TRAIN_SCHEMA",
    "TRAIN_SEED",
    "canonical_sha256",
    "generate_records",
    "overlap_report",
    "render_world",
    "scan_integer_spans",
    "scan_symbol_occurrences",
    "validate_record",
]
