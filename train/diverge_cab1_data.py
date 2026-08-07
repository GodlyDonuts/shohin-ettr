#!/usr/bin/env python3
"""Deterministic content-addressed register data for DIVERGE-CAB1."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping, Sequence

from diverge_eal1_data import (
    DEVELOPMENT_EPISODES,
    OPERATIONS,
    TRAIN_MATRICES,
    _word as eal_word,
    apply_matrix,
    canonical_sha256,
    scan_integer_spans,
)
from diverge_eal2_data import _transition_record
from diverge_jrb1_data import (
    augment_evaluation_episode as build_jrb1_episode,
    render_initial_state,
    render_query,
    validate_evaluation_episode as validate_jrb1_episode,
)


TRAIN_SCHEMA = "shohin-diverge-cab1-training-v1"
PUBLIC_SCHEMA = "shohin-diverge-cab1-public-v1"
ASSESSOR_SCHEMA = "shohin-diverge-cab1-assessor-v1"
REPORT_SCHEMA = "shohin-diverge-cab1-data-report-v1"
TRAIN_SEED = 2026080821
DEVELOPMENT_SEED = 2026080822
CONFIRMATION_SEEDS = (
    2026080823,
    2026080824,
    2026080825,
    2026080826,
    2026080827,
)
TRAIN_ROWS = 100_000
REGISTERS = 2


class CAB1DataError(RuntimeError):
    """A content-addressed register record violates its frozen contract."""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _word(namespace: str, seed: int, serial: int, index: int) -> str:
    tags = {
        "cab1-train-operation": "c",
        "cab1-train-register": "r",
        "cab1-train-decoy-register": "d",
    }
    if namespace not in tags or serial < 0 or index not in (0, 1):
        raise CAB1DataError("CAB1 opaque-name namespace differs")
    value = serial * REGISTERS + index
    suffix = ""
    for _ in range(4):
        suffix = chr(ord("a") + value % 26) + suffix
        value //= 26
    if value:
        raise CAB1DataError("CAB1 opaque-name serial exceeds frozen carrier")
    return eal_word(namespace, seed, serial, index) + tags[namespace] + suffix


def table_rotation(seed: int, serial: int) -> int:
    return int(canonical_sha256(["cab1-table", seed, serial])[-1], 16) % REGISTERS


def rotate_table(values: Sequence[str], rotation: int) -> tuple[str, str]:
    table = tuple(str(value) for value in values)
    rotation = int(rotation)
    if (
        len(table) != REGISTERS
        or len(set(table)) != REGISTERS
        or rotation not in (0, 1)
    ):
        raise CAB1DataError("CAB1 table geometry differs")
    return (table[rotation], table[1 - rotation])


def canonical_to_position(target: int, rotation: int) -> int:
    target = int(target)
    rotation = int(rotation)
    if target not in (0, 1) or rotation not in (0, 1):
        raise CAB1DataError("CAB1 target carrier differs")
    return target if rotation == 0 else 1 - target


def build_training_record(serial: int) -> dict[str, Any]:
    rng = random.Random(canonical_sha256(["cab1-training", TRAIN_SEED, serial]))
    operation = _word("cab1-train-operation", TRAIN_SEED, serial, 0)
    registers = tuple(
        _word("cab1-train-register", TRAIN_SEED, serial, index)
        for index in range(REGISTERS)
    )
    decoys = tuple(
        _word("cab1-train-decoy-register", TRAIN_SEED, serial, index)
        for index in range(REGISTERS)
    )
    rotation = table_rotation(TRAIN_SEED, serial)
    matrix = TRAIN_MATRICES[rng.randrange(len(TRAIN_MATRICES))]
    before = (rng.randrange(97), rng.randrange(97))
    transition = _transition_record(
        split="train",
        seed=TRAIN_SEED,
        serial=serial,
        operation_index=0,
        operation=operation,
        registers=registers,
        before=before,
        after=apply_matrix(matrix, before),
        rng=rng,
    )
    evidence_canonical = tuple(
        int(value) % REGISTERS for value in transition["numeric_role_ids"]
    )
    initial = (rng.randrange(97), rng.randrange(97))
    initial_order = [0, 1]
    rng.shuffle(initial_order)
    initial_text, initial_canonical, initial_renderer = render_initial_state(
        registers,
        initial,
        split="train",
        serial=serial,
        order=initial_order,
    )
    query_canonical = rng.randrange(REGISTERS)
    query_text, query_renderer = render_query(
        registers, query_canonical, split="train", serial=serial
    )
    record = {
        "schema": TRAIN_SCHEMA,
        "seed": TRAIN_SEED,
        "serial": serial,
        "operation": operation,
        "register_table": list(rotate_table(registers, rotation)),
        "decoy_register_table": list(rotate_table(decoys, rotation)),
        "table_rotation": rotation,
        "evidence_text": transition["source_text"],
        "evidence_sha256": transition["source_sha256"],
        "evidence_position_targets": [
            canonical_to_position(value, rotation) for value in evidence_canonical
        ],
        "initial_text": initial_text,
        "initial_sha256": _hash(initial_text),
        "initial_position_targets": [
            canonical_to_position(value, rotation) for value in initial_canonical
        ],
        "initial_renderer": [list(value) for value in initial_renderer],
        "query_text": query_text,
        "query_sha256": _hash(query_text),
        "query_position_target": canonical_to_position(query_canonical, rotation),
        "query_renderer": list(query_renderer),
    }
    record["identity_sha256"] = canonical_sha256(record)
    validate_training_record(record)
    return record


def validate_training_record(record: Mapping[str, Any]) -> None:
    if (
        record.get("schema") != TRAIN_SCHEMA
        or int(record.get("seed", -1)) != TRAIN_SEED
    ):
        raise CAB1DataError("CAB1 training schema or seed differs")
    table = tuple(str(value) for value in record["register_table"])
    decoys = tuple(str(value) for value in record["decoy_register_table"])
    operation = str(record["operation"])
    evidence = str(record["evidence_text"])
    initial = str(record["initial_text"])
    query = str(record["query_text"])
    evidence_targets = tuple(
        int(value) for value in record["evidence_position_targets"]
    )
    initial_targets = tuple(int(value) for value in record["initial_position_targets"])
    query_target = int(record["query_position_target"])
    if (
        len(table) != REGISTERS
        or len(set(table)) != REGISTERS
        or len(decoys) != REGISTERS
        or len(set(decoys)) != REGISTERS
        or set(table) & set(decoys)
        or not operation.isalpha()
        or not operation.islower()
        or any(not value.isalpha() or not value.islower() for value in table + decoys)
        or operation not in evidence
        or any(value not in evidence or value not in initial for value in table)
        or any(
            value in evidence or value in initial or value in query for value in decoys
        )
        or int(record["table_rotation"]) not in (0, 1)
        or len(scan_integer_spans(str(record["evidence_text"]))) != 4
        or len(scan_integer_spans(str(record["initial_text"]))) != 2
        or sorted(evidence_targets) != [0, 0, 1, 1]
        or sorted(initial_targets) != [0, 1]
        or query_target not in (0, 1)
        or table[query_target] not in query
        or _hash(str(record["evidence_text"])) != record["evidence_sha256"]
        or _hash(str(record["initial_text"])) != record["initial_sha256"]
        or _hash(str(record["query_text"])) != record["query_sha256"]
    ):
        raise CAB1DataError("CAB1 training geometry differs")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise CAB1DataError("CAB1 training identity differs")


def augment_evaluation_episode(
    serial: int, *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    jrb_public, jrb_assessor = build_jrb1_episode(serial, seed=seed)
    validate_jrb1_episode(jrb_public, jrb_assessor)
    registers = tuple(str(value) for value in jrb_public["registers"])
    renamed = tuple(str(value) for value in jrb_public["renamed_registers"])
    rotation = table_rotation(seed, serial)
    evidence = []
    for item in jrb_public["evidence"]:
        visible = dict(item)
        visible.pop("registers", None)
        evidence.append(visible)
    public = {
        **jrb_public,
        "schema": PUBLIC_SCHEMA,
        "register_table": list(rotate_table(registers, rotation)),
        "renamed_register_table": list(rotate_table(renamed, rotation)),
        "table_rotation": rotation,
        "evidence": evidence,
    }
    public.pop("registers")
    public.pop("renamed_registers")
    public.pop("identity_sha256")
    public["identity_sha256"] = canonical_sha256(public)
    assessor = {
        **jrb_assessor,
        "schema": ASSESSOR_SCHEMA,
        "public_identity_sha256": public["identity_sha256"],
        "canonical_registers": list(registers),
        "canonical_renamed_registers": list(renamed),
        "table_rotation": rotation,
    }
    assessor.pop("identity_sha256")
    assessor["identity_sha256"] = canonical_sha256(assessor)
    validate_evaluation_episode(public, assessor)
    return public, assessor


def validate_evaluation_episode(
    public: Mapping[str, Any], assessor: Mapping[str, Any]
) -> None:
    if (
        public.get("schema") != PUBLIC_SCHEMA
        or assessor.get("schema") != ASSESSOR_SCHEMA
    ):
        raise CAB1DataError("CAB1 evaluation schema differs")
    table = tuple(str(value) for value in public["register_table"])
    renamed = tuple(str(value) for value in public["renamed_register_table"])
    canonical = tuple(str(value) for value in assessor["canonical_registers"])
    canonical_renamed = tuple(
        str(value) for value in assessor["canonical_renamed_registers"]
    )
    rotation = int(public["table_rotation"])
    if (
        assessor.get("public_identity_sha256") != public.get("identity_sha256")
        or int(public.get("seed", -1)) != int(assessor.get("seed", -2))
        or int(public.get("serial", -1)) != int(assessor.get("serial", -2))
        or public.get("episode_id") != assessor.get("episode_id")
        or int(assessor.get("table_rotation", -1)) != rotation
        or table != rotate_table(canonical, rotation)
        or renamed != rotate_table(canonical_renamed, rotation)
        or len(public["evidence"]) != OPERATIONS * 3
        or len(public["transfer"]) != len(assessor["transfer"])
        or len(public["queries"]) != len(assessor["query_targets"])
        or "registers" in public
        or "renamed_registers" in public
        or any("registers" in item for item in public["evidence"])
        or any(
            "initial_state" in item or "symbols" in item for item in public["transfer"]
        )
        or any("register_index" in item for item in public["queries"])
    ):
        raise CAB1DataError("CAB1 evaluation geometry differs")
    for episode in (public, assessor):
        payload = dict(episode)
        identity = str(payload.pop("identity_sha256"))
        if canonical_sha256(payload) != identity:
            raise CAB1DataError("CAB1 evaluation identity differs")


__all__ = [
    "ASSESSOR_SCHEMA",
    "CAB1DataError",
    "CONFIRMATION_SEEDS",
    "DEVELOPMENT_EPISODES",
    "DEVELOPMENT_SEED",
    "PUBLIC_SCHEMA",
    "REPORT_SCHEMA",
    "TRAIN_ROWS",
    "TRAIN_SCHEMA",
    "TRAIN_SEED",
    "augment_evaluation_episode",
    "build_training_record",
    "canonical_to_position",
    "rotate_table",
    "table_rotation",
    "validate_evaluation_episode",
    "validate_training_record",
]
