#!/usr/bin/env python3
"""Deterministic episode-local affine-law data for DIVERGE-EAL1."""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Mapping, Sequence

from diverge_mze1_runtime import PRIME, ROW_CANDIDATES


TRAIN_SCHEMA = "shohin-diverge-eal1-training-v1"
PUBLIC_SCHEMA = "shohin-diverge-eal1-public-v1"
ASSESSOR_SCHEMA = "shohin-diverge-eal1-assessor-v1"
REPORT_SCHEMA = "shohin-diverge-eal1-data-report-v1"
TRAIN_SEED = 2026080751
DEVELOPMENT_SEED = 2026080752
TRAIN_ROWS = 100_000
DEVELOPMENT_EPISODES = 256
OPERATIONS = 8
DEMONSTRATIONS_PER_OPERATION = 3
TRANSFER_PROGRAMS = 16
TRANSFER_DEPTHS = (12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32)
MAX_SOURCE_BYTES = 320
NUMERIC_ROLES = ("BEFORE_X", "BEFORE_Y", "AFTER_X", "AFTER_Y")

Matrix = tuple[tuple[int, int], tuple[int, int]]
State = tuple[int, int]

_INTEGER = re.compile(r"(?<![A-Za-z0-9])(?:0|[1-9][0-9]?)(?![A-Za-z0-9])")
_BEFORE = (
    "Before {op}, {register} was {value}",
    "Prior to {op}, {register} held {value}",
    "At input to {op}, {register} equaled {value}",
    "Entering {op}, {register} contained {value}",
)
_AFTER = (
    "After {op}, {register} was {value}",
    "Following {op}, {register} held {value}",
    "At output of {op}, {register} equaled {value}",
    "Leaving {op}, {register} contained {value}",
)
_JOINERS = ("; ", ". ", " | ", "; meanwhile, ")


class EAL1DataError(RuntimeError):
    """An EAL1 split or episode violates the frozen contract."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _word(domain: str, seed: int, serial: int, index: int) -> str:
    digest = hashlib.sha256(f"{domain}:{seed}:{serial}:{index}".encode()).digest()
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    return "".join(
        consonants[digest[2 * offset] % len(consonants)]
        + vowels[digest[2 * offset + 1] % len(vowels)]
        for offset in range(6)
    )


def apply_matrix(matrix: Matrix, state: State) -> State:
    return tuple((row[0] * state[0] + row[1] * state[1]) % PRIME for row in matrix)  # type: ignore[return-value]


def _matrix_catalog() -> tuple[Matrix, ...]:
    matrices = []
    for first in ROW_CANDIDATES:
        for second in ROW_CANDIDATES:
            determinant = (first[0] * second[1] - first[1] * second[0]) % PRIME
            if determinant and first != (0, 0) and second != (0, 0):
                matrices.append((first, second))
    return tuple(matrices)


MATRICES = _matrix_catalog()
TRAIN_MATRICES = tuple(
    matrix for matrix in MATRICES if int(canonical_sha256(matrix), 16) % 5 != 4
)
DEVELOPMENT_MATRICES = tuple(
    matrix for matrix in MATRICES if int(canonical_sha256(matrix), 16) % 5 == 4
)
if len(TRAIN_MATRICES) < 8 or len(DEVELOPMENT_MATRICES) < 8:
    raise RuntimeError("EAL1 matrix split is empty")


def scan_integer_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in _INTEGER.finditer(text))


def _role_clause(
    *,
    role: int,
    value: int,
    operation: str,
    registers: tuple[str, str],
    before_renderer: int,
    after_renderer: int,
) -> str:
    register = registers[role % 2]
    template = _BEFORE[before_renderer] if role < 2 else _AFTER[after_renderer]
    return template.format(op=operation, register=register, value=value)


def _render(
    *,
    operation: str,
    registers: tuple[str, str],
    before: State,
    after: State,
    renderer: tuple[int, int, int],
    order: Sequence[int],
    counterfactual: bool,
    scrub: bool,
) -> tuple[str, tuple[int, ...]]:
    values = (*before, *after)
    before_renderer, after_renderer, joiner = renderer
    clauses = []
    roles = []
    for original_role in order:
        role = original_role
        if counterfactual:
            role = original_role + 2 if original_role < 2 else original_role - 2
        if scrub:
            register = registers[original_role % 2]
            clause = f"For {operation}, {register} had {values[original_role]}"
        else:
            clause = _role_clause(
                role=role,
                value=values[original_role],
                operation=operation,
                registers=registers,
                before_renderer=before_renderer,
                after_renderer=after_renderer,
            )
        clauses.append(clause)
        roles.append(role)
    text = _JOINERS[joiner].join(clauses) + "."
    if len(text.encode("ascii")) + 1 > MAX_SOURCE_BYTES:
        raise EAL1DataError("EAL1 source exceeds frozen width")
    if len(scan_integer_spans(text)) != 4:
        raise EAL1DataError("EAL1 source does not expose four integers")
    return text, tuple(roles)


def transition_record(
    *,
    split: str,
    seed: int,
    serial: int,
    operation_index: int,
    operation: str,
    registers: tuple[str, str],
    before: State,
    after: State,
    rng: random.Random,
) -> dict[str, Any]:
    if split not in ("train", "development"):
        raise EAL1DataError("EAL1 transition split differs")
    renderer_bucket = 0 if split == "train" else 1
    pairs = tuple(
        (left, right)
        for left in range(len(_BEFORE))
        for right in range(len(_AFTER))
        if (left + 2 * right) % 2 == renderer_bucket
    )
    pair = pairs[serial % len(pairs)]
    renderer = (pair[0], pair[1], rng.randrange(len(_JOINERS)))
    order = list(range(4))
    rng.shuffle(order)
    source, roles = _render(
        operation=operation,
        registers=registers,
        before=before,
        after=after,
        renderer=renderer,
        order=order,
        counterfactual=False,
        scrub=False,
    )
    counterfactual, counterfactual_roles = _render(
        operation=operation,
        registers=registers,
        before=before,
        after=after,
        renderer=renderer,
        order=order,
        counterfactual=True,
        scrub=False,
    )
    scrubbed, _ = _render(
        operation=operation,
        registers=registers,
        before=before,
        after=after,
        renderer=renderer,
        order=order,
        counterfactual=False,
        scrub=True,
    )
    record = {
        "split": split,
        "seed": seed,
        "serial": serial,
        "renderer": list(renderer),
        "operation_index": operation_index,
        "operation": operation,
        "registers": list(registers),
        "before": list(before),
        "after": list(after),
        "source_text": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "numeric_role_ids": list(roles),
        "counterfactual_text": counterfactual,
        "counterfactual_sha256": hashlib.sha256(counterfactual.encode()).hexdigest(),
        "counterfactual_role_ids": list(counterfactual_roles),
        "scrubbed_text": scrubbed,
        "scrubbed_sha256": hashlib.sha256(scrubbed.encode()).hexdigest(),
    }
    record["identity_sha256"] = canonical_sha256(record)
    validate_transition_record(record)
    return record


def validate_transition_record(record: Mapping[str, Any]) -> None:
    source = str(record["source_text"])
    counterfactual = str(record["counterfactual_text"])
    scrubbed = str(record["scrubbed_text"])
    for text, key in (
        (source, "source_sha256"),
        (counterfactual, "counterfactual_sha256"),
        (scrubbed, "scrubbed_sha256"),
    ):
        if hashlib.sha256(text.encode()).hexdigest() != record[key]:
            raise EAL1DataError("EAL1 source commitment differs")
        if len(scan_integer_spans(text)) != 4:
            raise EAL1DataError("EAL1 numeric geometry differs")
    roles = tuple(int(value) for value in record["numeric_role_ids"])
    counterfactual_roles = tuple(
        int(value) for value in record["counterfactual_role_ids"]
    )
    if sorted(roles) != list(range(4)) or sorted(counterfactual_roles) != list(
        range(4)
    ):
        raise EAL1DataError("EAL1 role assignment is not a permutation")
    if any(
        (left + 2 if left < 2 else left - 2) != right
        for left, right in zip(roles, counterfactual_roles, strict=True)
    ):
        raise EAL1DataError("EAL1 counterfactual role map differs")
    source_values = tuple(
        int(source[start:end]) for start, end in scan_integer_spans(source)
    )
    counterfactual_values = tuple(
        int(counterfactual[start:end])
        for start, end in scan_integer_spans(counterfactual)
    )
    scrubbed_values = tuple(
        int(scrubbed[start:end]) for start, end in scan_integer_spans(scrubbed)
    )
    if source_values != counterfactual_values or source_values != scrubbed_values:
        raise EAL1DataError("EAL1 intervention changed numeric source geometry")
    expected = {
        0: int(record["before"][0]),
        1: int(record["before"][1]),
        2: int(record["after"][0]),
        3: int(record["after"][1]),
    }
    if any(
        value != expected[role]
        for value, role in zip(source_values, roles, strict=True)
    ):
        raise EAL1DataError("EAL1 normal roles disagree with transition")
    if any(
        value != expected[role - 2 if role >= 2 else role + 2]
        for value, role in zip(counterfactual_values, counterfactual_roles, strict=True)
    ):
        raise EAL1DataError("EAL1 counterfactual did not preserve transition values")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise EAL1DataError("EAL1 identity commitment differs")


def build_training_record(serial: int) -> dict[str, Any]:
    rng = random.Random(canonical_sha256(["eal1-train", TRAIN_SEED, serial]))
    operation = _word("operation", TRAIN_SEED, serial, 0)
    registers = (
        _word("register", TRAIN_SEED, serial, 0),
        _word("register", TRAIN_SEED, serial, 1),
    )
    matrix = TRAIN_MATRICES[rng.randrange(len(TRAIN_MATRICES))]
    before = (rng.randrange(PRIME), rng.randrange(PRIME))
    after = apply_matrix(matrix, before)
    record = transition_record(
        split="train",
        seed=TRAIN_SEED,
        serial=serial,
        operation_index=0,
        operation=operation,
        registers=registers,
        before=before,
        after=after,
        rng=rng,
    )
    output = {"schema": TRAIN_SCHEMA, **record}
    output["identity_sha256"] = canonical_sha256(
        {key: value for key, value in output.items() if key != "identity_sha256"}
    )
    validate_training_record(output)
    return output


def validate_training_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != TRAIN_SCHEMA or record.get("split") != "train":
        raise EAL1DataError("EAL1 training schema differs")
    validate_transition_record(record)


def _identifying_states(matrix: Matrix, rng: random.Random) -> tuple[State, ...]:
    del matrix
    left = rng.randrange(1, PRIME)
    right = rng.randrange(1, PRIME)
    both = (rng.randrange(1, PRIME), rng.randrange(1, PRIME))
    # Either axis observation alone leaves five coefficient rows possible. The
    # pair identifies both coefficients; the third statement tests consistency.
    return ((left, 0), (0, right), both)


def build_development_episode(serial: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(
        canonical_sha256(["eal1-development", DEVELOPMENT_SEED, serial])
    )
    episode_id = canonical_sha256(["eal1-episode", DEVELOPMENT_SEED, serial])[:24]
    aliases = tuple(
        _word("alias", DEVELOPMENT_SEED, serial, index) for index in range(OPERATIONS)
    )
    registers = (
        _word("register", DEVELOPMENT_SEED, serial, 0),
        _word("register", DEVELOPMENT_SEED, serial, 1),
    )
    matrices = tuple(rng.sample(DEVELOPMENT_MATRICES, OPERATIONS))
    evidence_public = []
    evidence_hidden = []
    transition_serial = serial * OPERATIONS * DEMONSTRATIONS_PER_OPERATION
    for operation_index, (alias, matrix) in enumerate(
        zip(aliases, matrices, strict=True)
    ):
        states = _identifying_states(matrix, rng)
        for demo_index, before in enumerate(states):
            after = apply_matrix(matrix, before)
            record = transition_record(
                split="development",
                seed=DEVELOPMENT_SEED,
                serial=transition_serial,
                operation_index=operation_index,
                operation=alias,
                registers=registers,
                before=before,
                after=after,
                rng=rng,
            )
            transition_serial += 1
            evidence_public.append(
                {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "numeric_role_ids",
                        "counterfactual_role_ids",
                        "operation_index",
                        "before",
                        "after",
                    }
                }
            )
            evidence_hidden.append(
                {
                    "operation_index": operation_index,
                    "demonstration_index": demo_index,
                    "before": list(before),
                    "after": list(after),
                    "numeric_role_ids": record["numeric_role_ids"],
                    "counterfactual_role_ids": record["counterfactual_role_ids"],
                }
            )
    transfers = []
    hidden_transfers = []
    for index, depth in enumerate(TRANSFER_DEPTHS):
        initial = (rng.randrange(PRIME), rng.randrange(PRIME))
        symbols = tuple(rng.randrange(OPERATIONS) for _ in range(depth))
        state = initial
        for symbol in symbols:
            state = apply_matrix(matrices[symbol], state)
        program_id = canonical_sha256([episode_id, index, initial, symbols])[:24]
        transfers.append(
            {
                "program_id": program_id,
                "initial_state": list(initial),
                "symbols": [aliases[symbol] for symbol in symbols],
                "depth": depth,
            }
        )
        hidden_transfers.append(
            {
                "program_id": program_id,
                "symbol_indices": list(symbols),
                "terminal_state": list(state),
            }
        )
    queries = [
        {"program_id": transfer["program_id"], "register_index": register}
        for transfer in transfers
        for register in range(2)
    ]
    public = {
        "schema": PUBLIC_SCHEMA,
        "episode_id": episode_id,
        "seed": DEVELOPMENT_SEED,
        "serial": serial,
        "aliases": list(aliases),
        "registers": list(registers),
        "evidence": evidence_public,
        "transfer": transfers,
        "queries": queries,
    }
    public["identity_sha256"] = canonical_sha256(public)
    assessor = {
        "schema": ASSESSOR_SCHEMA,
        "public_identity_sha256": public["identity_sha256"],
        "matrices": [[list(row) for row in matrix] for matrix in matrices],
        "evidence": evidence_hidden,
        "transfer": hidden_transfers,
    }
    assessor["identity_sha256"] = canonical_sha256(assessor)
    validate_episode(public, assessor)
    return public, assessor


def validate_episode(public: Mapping[str, Any], assessor: Mapping[str, Any]) -> None:
    if (
        public.get("schema") != PUBLIC_SCHEMA
        or assessor.get("schema") != ASSESSOR_SCHEMA
    ):
        raise EAL1DataError("EAL1 episode schema differs")
    public_payload = dict(public)
    public_identity = str(public_payload.pop("identity_sha256"))
    if canonical_sha256(public_payload) != public_identity:
        raise EAL1DataError("EAL1 public identity differs")
    assessor_payload = dict(assessor)
    assessor_identity = str(assessor_payload.pop("identity_sha256"))
    if canonical_sha256(assessor_payload) != assessor_identity:
        raise EAL1DataError("EAL1 assessor identity differs")
    if assessor["public_identity_sha256"] != public["identity_sha256"]:
        raise EAL1DataError("EAL1 public/assessor binding differs")
    if (
        len(public["aliases"]) != OPERATIONS
        or len(public["evidence"]) != OPERATIONS * DEMONSTRATIONS_PER_OPERATION
        or len(public["transfer"]) != TRANSFER_PROGRAMS
        or len(assessor["matrices"]) != OPERATIONS
    ):
        raise EAL1DataError("EAL1 episode geometry differs")


def overlap_report(
    training: Sequence[Mapping[str, Any]],
    public: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_sources = {str(row["source_sha256"]) for row in training}
    development_sources = {
        str(item["source_sha256"]) for episode in public for item in episode["evidence"]
    }
    train_names = {
        value for row in training for value in (row["operation"], *row["registers"])
    }
    development_names = {
        value
        for episode in public
        for value in (*episode["aliases"], *episode["registers"])
    }
    return {
        "source_overlap": len(train_sources & development_sources),
        "name_overlap": len(train_names & development_names),
        "training_matrix_count": len(TRAIN_MATRICES),
        "development_matrix_count": len(DEVELOPMENT_MATRICES),
        "training_matrix_sha256": canonical_sha256(TRAIN_MATRICES),
        "development_matrix_sha256": canonical_sha256(DEVELOPMENT_MATRICES),
        "matrix_overlap": len(set(TRAIN_MATRICES) & set(DEVELOPMENT_MATRICES)),
        "training_source_count": len(training),
        "training_source_unique": len(train_sources) == len(training),
        "development_source_count": sum(len(episode["evidence"]) for episode in public),
        "development_source_unique": len(development_sources)
        == sum(len(episode["evidence"]) for episode in public),
    }


__all__ = [
    "ASSESSOR_SCHEMA",
    "DEMONSTRATIONS_PER_OPERATION",
    "DEVELOPMENT_EPISODES",
    "DEVELOPMENT_MATRICES",
    "DEVELOPMENT_SEED",
    "EAL1DataError",
    "MAX_SOURCE_BYTES",
    "NUMERIC_ROLES",
    "OPERATIONS",
    "PUBLIC_SCHEMA",
    "REPORT_SCHEMA",
    "TRAIN_MATRICES",
    "TRAIN_ROWS",
    "TRAIN_SCHEMA",
    "TRAIN_SEED",
    "TRANSFER_PROGRAMS",
    "TRANSFER_DEPTHS",
    "apply_matrix",
    "build_development_episode",
    "build_training_record",
    "canonical_sha256",
    "overlap_report",
    "scan_integer_spans",
    "validate_episode",
    "validate_training_record",
    "validate_transition_record",
]
