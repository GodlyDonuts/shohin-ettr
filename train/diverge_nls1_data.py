#!/usr/bin/env python3
"""Deterministic typed law-synthesis supervision for DIVERGE-NLS1."""

from __future__ import annotations

import random
from typing import Any, Mapping

from diverge_eal1_data import TRAIN_MATRICES, apply_matrix, canonical_sha256
from diverge_mze1_runtime import PRIME, ROW_CANDIDATES


TRAIN_SCHEMA = "shohin-diverge-nls1-training-v1"
REPORT_SCHEMA = "shohin-diverge-nls1-data-report-v1"
TRAIN_SEED = 2026080781
DEVELOPMENT_SEED = 2026080782
CONFIRMATION_SEEDS = (
    2026080783,
    2026080784,
    2026080785,
    2026080786,
    2026080787,
)
TRAIN_ROWS = 100_000
DEMONSTRATIONS = 3
VALUES = 4

ROW_TO_INDEX = {row: index for index, row in enumerate(ROW_CANDIDATES)}


class NLS1DataError(RuntimeError):
    """A neural law-synthesis row violates its frozen contract."""


def _identifying_states(rng: random.Random) -> tuple[tuple[int, int], ...]:
    return (
        (rng.randrange(1, PRIME), 0),
        (0, rng.randrange(1, PRIME)),
        (rng.randrange(1, PRIME), rng.randrange(1, PRIME)),
    )


def build_training_record(serial: int) -> dict[str, Any]:
    rng = random.Random(canonical_sha256(["nls1-training", TRAIN_SEED, serial]))
    matrix = TRAIN_MATRICES[rng.randrange(len(TRAIN_MATRICES))]
    demonstrations = [
        [*before, *apply_matrix(matrix, before)] for before in _identifying_states(rng)
    ]
    rng.shuffle(demonstrations)
    record = {
        "schema": TRAIN_SCHEMA,
        "seed": TRAIN_SEED,
        "serial": serial,
        "demonstrations": demonstrations,
        "target_row_ids": [ROW_TO_INDEX[row] for row in matrix],
    }
    record["identity_sha256"] = canonical_sha256(record)
    validate_training_record(record)
    return record


def validate_training_record(record: Mapping[str, Any]) -> None:
    if (
        record.get("schema") != TRAIN_SCHEMA
        or int(record.get("seed", -1)) != TRAIN_SEED
    ):
        raise NLS1DataError("NLS1 training schema or seed differs")
    demonstrations = tuple(
        tuple(int(value) for value in row) for row in record["demonstrations"]
    )
    if (
        len(demonstrations) != DEMONSTRATIONS
        or any(len(row) != VALUES for row in demonstrations)
        or any(value < 0 or value >= PRIME for row in demonstrations for value in row)
    ):
        raise NLS1DataError("NLS1 demonstration geometry differs")
    targets = tuple(int(value) for value in record["target_row_ids"])
    if len(targets) != 2 or any(
        value < 0 or value >= len(ROW_CANDIDATES) for value in targets
    ):
        raise NLS1DataError("NLS1 row target leaves its carrier")
    matrix = tuple(ROW_CANDIDATES[value] for value in targets)
    if any(tuple(apply_matrix(matrix, row[:2])) != row[2:] for row in demonstrations):
        raise NLS1DataError("NLS1 target law does not explain demonstrations")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise NLS1DataError("NLS1 training identity differs")


__all__ = [
    "CONFIRMATION_SEEDS",
    "DEMONSTRATIONS",
    "DEVELOPMENT_SEED",
    "NLS1DataError",
    "REPORT_SCHEMA",
    "TRAIN_ROWS",
    "TRAIN_SCHEMA",
    "TRAIN_SEED",
    "VALUES",
    "build_training_record",
    "validate_training_record",
]
