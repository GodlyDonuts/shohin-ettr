#!/usr/bin/env python3
"""Deterministic episode generator for DIVERGE-PL1."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable


SCHEMA = "shohin-diverge-pl1-board-v1"
PRIME = 97
OP_NAMES = (
    "X_PLUS_Y",
    "Y_PLUS_X",
    "X_MINUS_Y",
    "Y_MINUS_X",
    "X_DOUBLE_PLUS_Y",
    "Y_DOUBLE_PLUS_X",
    "SWAP",
    "NEGATE_X_PLUS_Y",
)
TRAIN_SEED = 2026080701
DEVELOPMENT_SEED = 2026080702
CONFIRMATION_SEEDS = (
    2026080711,
    2026080712,
    2026080713,
    2026080714,
    2026080715,
)

State = tuple[int, int]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def commitment(domain: str, value: object) -> str:
    digest = hashlib.sha256()
    for part in (domain.encode("ascii"), canonical_json_bytes(value)):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def apply_operation(operation: int, state: State) -> State:
    """Apply one primitive transform over Z/97Z."""

    x, y = state
    if operation == 0:
        result = (x + y, y)
    elif operation == 1:
        result = (x, y + x)
    elif operation == 2:
        result = (x - y, y)
    elif operation == 3:
        result = (x, y - x)
    elif operation == 4:
        result = (2 * x + y, y)
    elif operation == 5:
        result = (x, 2 * y + x)
    elif operation == 6:
        result = (y, x)
    elif operation == 7:
        result = (-x + y, y)
    else:
        raise ValueError(f"unknown operation {operation}")
    return result[0] % PRIME, result[1] % PRIME


def operation_outputs_are_unique(state: State) -> bool:
    return len({apply_operation(operation, state) for operation in range(len(OP_NAMES))}) == len(
        OP_NAMES
    )


@dataclass(frozen=True)
class Program:
    program_id: str
    initial_state: State
    symbols: tuple[int, ...]
    trace: tuple[State, ...]

    @property
    def terminal_state(self) -> State:
        return self.trace[-1]

    def public_record(self) -> dict[str, object]:
        return {
            "program_id": self.program_id,
            "initial_state": list(self.initial_state),
            "symbols": list(self.symbols),
            "depth": len(self.symbols),
        }

    def assessor_record(self) -> dict[str, object]:
        record = self.public_record()
        record["trace"] = [list(state) for state in self.trace]
        record["terminal_state"] = list(self.terminal_state)
        return record


@dataclass(frozen=True)
class Episode:
    episode_id: str
    split: str
    seed: int
    aliases: tuple[str, ...]
    symbol_to_operation: tuple[int, ...]
    acquisition: tuple[Program, ...]
    transfer: tuple[Program, ...]

    def public_record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "episode_id": self.episode_id,
            "split": self.split,
            "aliases": list(self.aliases),
            "acquisition": [program.public_record() for program in self.acquisition],
            "transfer": [program.public_record() for program in self.transfer],
        }

    def assessor_record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "episode_id": self.episode_id,
            "split": self.split,
            "seed": self.seed,
            "aliases": list(self.aliases),
            "symbol_to_operation": list(self.symbol_to_operation),
            "acquisition": [program.assessor_record() for program in self.acquisition],
            "transfer": [program.assessor_record() for program in self.transfer],
        }


@dataclass(frozen=True)
class Verification:
    passed: bool
    first_error: int | None
    receipt: str


def verify_trace(episode: Episode, program: Program, candidate: Iterable[State]) -> Verification:
    """Return a source-owned outcome without disclosing the correct state."""

    candidate_trace = tuple(candidate)
    expected_length = len(program.trace)
    first_error: int | None = None
    if len(candidate_trace) != expected_length:
        first_error = min(len(candidate_trace), expected_length - 1)
    else:
        for index, (observed, expected) in enumerate(zip(candidate_trace, program.trace, strict=True)):
            normalized = observed[0] % PRIME, observed[1] % PRIME
            if normalized != expected:
                first_error = index
                break
    passed = first_error is None
    receipt = commitment(
        "diverge-pl1-verification",
        {
            "episode_id": episode.episode_id,
            "program_id": program.program_id,
            "candidate_commitment": commitment("diverge-pl1-candidate", candidate_trace),
            "passed": passed,
            "first_error": first_error,
        },
    )
    return Verification(passed=passed, first_error=first_error, receipt=receipt)


def execute_mapping(mapping: tuple[int, ...], program: Program) -> tuple[State, ...]:
    if sorted(mapping) != list(range(len(OP_NAMES))):
        raise ValueError("mapping must be a complete operation permutation")
    state = program.initial_state
    trace = [state]
    for symbol in program.symbols:
        state = apply_operation(mapping[symbol], state)
        trace.append(state)
    return tuple(trace)


def _alias(episode_id: str, symbol: int) -> str:
    digest = hashlib.sha256(f"{episode_id}:{symbol}".encode("ascii")).hexdigest()
    alphabet = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    chunks = []
    for offset in range(0, 12, 3):
        consonant = alphabet[int(digest[offset : offset + 2], 16) % len(alphabet)]
        vowel = vowels[int(digest[offset + 2], 16) % len(vowels)]
        chunks.append(consonant + vowel)
    return "".join(chunks)


def _program(
    *,
    rng: random.Random,
    episode_id: str,
    serial: int,
    depth: int,
    mapping: tuple[int, ...],
) -> Program:
    for _ in range(20_000):
        initial = rng.randrange(1, PRIME), rng.randrange(1, PRIME)
        symbols = tuple(rng.randrange(len(OP_NAMES)) for _ in range(depth))
        state = initial
        trace = [state]
        valid = True
        for symbol in symbols:
            if not operation_outputs_are_unique(state):
                valid = False
                break
            state = apply_operation(mapping[symbol], state)
            trace.append(state)
        if valid:
            payload = {
                "episode_id": episode_id,
                "serial": serial,
                "initial": initial,
                "symbols": symbols,
            }
            return Program(
                program_id=commitment("diverge-pl1-program", payload)[:24],
                initial_state=initial,
                symbols=symbols,
                trace=tuple(trace),
            )
    raise RuntimeError("could not generate an operation-separating program")


def build_episode(*, split: str, seed: int, serial: int) -> Episode:
    rng = random.Random(commitment("diverge-pl1-episode-rng", [split, seed, serial]))
    episode_id = commitment("diverge-pl1-episode", [split, seed, serial])[:24]
    mapping_list = list(range(len(OP_NAMES)))
    rng.shuffle(mapping_list)
    mapping = tuple(mapping_list)
    aliases = tuple(_alias(episode_id, symbol) for symbol in range(len(OP_NAMES)))

    acquisition = tuple(
        _program(
            rng=rng,
            episode_id=episode_id,
            serial=index,
            depth=3 + index % 3,
            mapping=mapping,
        )
        for index in range(12)
    )
    transfer = tuple(
        _program(
            rng=rng,
            episode_id=episode_id,
            serial=100 + index,
            depth=12 + index % 9,
            mapping=mapping,
        )
        for index in range(16)
    )
    return Episode(
        episode_id=episode_id,
        split=split,
        seed=seed,
        aliases=aliases,
        symbol_to_operation=mapping,
        acquisition=acquisition,
        transfer=transfer,
    )


def build_split(*, split: str, seed: int, count: int) -> tuple[Episode, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    return tuple(build_episode(split=split, seed=seed, serial=serial) for serial in range(count))


def episode_from_assessor_record(record: dict[str, Any]) -> Episode:
    if record.get("schema") != SCHEMA:
        raise ValueError("PL1 assessor schema differs")

    def parse_program(item: dict[str, Any]) -> Program:
        trace = tuple((int(state[0]), int(state[1])) for state in item["trace"])
        program = Program(
            program_id=str(item["program_id"]),
            initial_state=(int(item["initial_state"][0]), int(item["initial_state"][1])),
            symbols=tuple(int(value) for value in item["symbols"]),
            trace=trace,
        )
        if list(program.terminal_state) != item["terminal_state"]:
            raise ValueError("PL1 assessor terminal differs")
        return program

    episode = Episode(
        episode_id=str(record["episode_id"]),
        split=str(record["split"]),
        seed=int(record["seed"]),
        aliases=tuple(str(value) for value in record["aliases"]),
        symbol_to_operation=tuple(int(value) for value in record["symbol_to_operation"]),
        acquisition=tuple(parse_program(item) for item in record["acquisition"]),
        transfer=tuple(parse_program(item) for item in record["transfer"]),
    )
    if sorted(episode.symbol_to_operation) != list(range(len(OP_NAMES))):
        raise ValueError("PL1 assessor mapping is not a permutation")
    for program in (*episode.acquisition, *episode.transfer):
        if execute_mapping(episode.symbol_to_operation, program) != program.trace:
            raise ValueError("PL1 assessor trace differs from independent execution")
    return episode


def iter_program_identities(episodes: Iterable[Episode]) -> Iterable[str]:
    for episode in episodes:
        for program in (*episode.acquisition, *episode.transfer):
            yield commitment(
                "diverge-pl1-program-identity",
                [
                    program.initial_state,
                    [episode.aliases[symbol] for symbol in program.symbols],
                ],
            )
