#!/usr/bin/env python3
"""Deterministic learned evidence-operation binding data for DIVERGE-OPB1."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Mapping, Sequence

from diverge_eal1_data import (
    OPERATIONS,
    TRAIN_MATRICES,
    apply_matrix,
    canonical_sha256,
)
from diverge_eal2_data import _transition_record
from diverge_jrb1_data import _replace_names
from diverge_sve1_data import (
    DEVELOPMENT_EPISODES,
    augment_evaluation_episode as build_sve1_episode,
    validate_evaluation_episode as validate_sve1_episode,
)


TRAIN_SCHEMA = "shohin-diverge-opb1-training-v1"
PUBLIC_SCHEMA = "shohin-diverge-opb1-public-v1"
ASSESSOR_SCHEMA = "shohin-diverge-opb1-assessor-v1"
REPORT_SCHEMA = "shohin-diverge-opb1-data-report-v1"
TRAIN_SEED = 2026080861
DEVELOPMENT_SEED = 2026080862
CONFIRMATION_SEEDS = (
    2026080863,
    2026080864,
    2026080865,
    2026080866,
    2026080867,
)
TRAIN_ROWS = 100_000
REGISTERS = 2


class OPB1DataError(RuntimeError):
    """An evidence-operation binding record violates its frozen contract."""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _word(namespace: str, seed: int, serial: int, index: int) -> str:
    if not namespace.startswith("opb1-") or serial < 0 or index < 0:
        raise OPB1DataError("OPB1 opaque-name coordinate differs")
    digest = hashlib.sha256(
        f"{namespace}|{seed}|{serial}|{index}".encode("ascii")
    ).digest()
    return "".join(chr(ord("a") + value % 26) for value in digest[:20])


def _whole_word_count(text: str, word: str) -> int:
    return len(re.findall(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text))


def build_training_record(serial: int) -> dict[str, Any]:
    rng = random.Random(canonical_sha256(["opb1-training", TRAIN_SEED, serial]))
    aliases = [
        _word("opb1-train-alias", TRAIN_SEED, serial, index)
        for index in range(OPERATIONS)
    ]
    decoys = [
        _word("opb1-train-decoy", TRAIN_SEED, serial, index)
        for index in range(OPERATIONS)
    ]
    registers = tuple(
        _word("opb1-train-register", TRAIN_SEED, serial, index)
        for index in range(REGISTERS)
    )
    rng.shuffle(aliases)
    rng.shuffle(decoys)
    target = rng.randrange(OPERATIONS)
    before = (rng.randrange(97), rng.randrange(97))
    matrix = TRAIN_MATRICES[rng.randrange(len(TRAIN_MATRICES))]
    transition = _transition_record(
        split="train",
        seed=TRAIN_SEED,
        serial=serial,
        operation_index=target,
        operation=aliases[target],
        registers=registers,
        before=before,
        after=apply_matrix(matrix, before),
        rng=rng,
    )
    record = {
        "schema": TRAIN_SCHEMA,
        "seed": TRAIN_SEED,
        "serial": serial,
        "source_text": transition["source_text"],
        "source_sha256": transition["source_sha256"],
        "aliases": aliases,
        "decoy_aliases": decoys,
        "registers": list(registers),
        "operation_target": target,
        "renderer": transition["renderer"],
    }
    record["identity_sha256"] = canonical_sha256(record)
    validate_training_record(record)
    return record


def validate_training_record(record: Mapping[str, Any]) -> None:
    aliases = tuple(str(value) for value in record["aliases"])
    decoys = tuple(str(value) for value in record["decoy_aliases"])
    registers = tuple(str(value) for value in record["registers"])
    text = str(record["source_text"])
    target = int(record["operation_target"])
    if (
        record.get("schema") != TRAIN_SCHEMA
        or int(record.get("seed", -1)) != TRAIN_SEED
        or len(aliases) != OPERATIONS
        or len(decoys) != OPERATIONS
        or len(registers) != REGISTERS
        or len(set((*aliases, *decoys, *registers))) != OPERATIONS * 2 + REGISTERS
        or any(
            not value.isalpha() or not value.islower()
            for value in (*aliases, *decoys, *registers)
        )
        or target not in range(OPERATIONS)
        or _whole_word_count(text, aliases[target]) != 4
        or any(
            _whole_word_count(text, alias) != 0
            for index, alias in enumerate(aliases)
            if index != target
        )
        or any(_whole_word_count(text, alias) for alias in decoys)
        or any(_whole_word_count(text, register) != 2 for register in registers)
        or _hash(text) != record["source_sha256"]
    ):
        raise OPB1DataError("OPB1 training geometry differs")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise OPB1DataError("OPB1 training identity differs")


def augment_evaluation_episode(
    serial: int, *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_public, base_assessor = build_sve1_episode(serial, seed=seed)
    validate_sve1_episode(base_public, base_assessor)
    aliases = tuple(str(value) for value in base_public["aliases"])
    renamed_aliases = tuple(str(value) for value in base_public["renamed_aliases"])
    register_table = tuple(str(value) for value in base_public["register_table"])
    renamed_register_table = tuple(
        str(value) for value in base_public["renamed_register_table"]
    )
    scrub_aliases = tuple(
        _word("opb1-eval-scrub", seed, serial, index) for index in range(OPERATIONS)
    )

    evidence = []
    operation_targets = []
    for visible, hidden in zip(
        base_public["evidence"], base_assessor["evidence"], strict=True
    ):
        target = int(hidden["operation_index"])
        if str(visible["operation"]) != aliases[target]:
            raise OPB1DataError("OPB1 parent operation target differs")
        source = str(visible["source_text"])
        renamed = _replace_names(
            source,
            (*aliases, *register_table),
            (*renamed_aliases, *renamed_register_table),
        )
        scrubbed = _replace_names(source, (aliases[target],), (scrub_aliases[target],))
        item = {
            key: value
            for key, value in visible.items()
            if key not in ("operation", "identity_sha256")
        }
        item.update(
            {
                "fully_renamed_source_text": renamed,
                "fully_renamed_source_sha256": _hash(renamed),
                "operation_scrubbed_text": scrubbed,
                "operation_scrubbed_sha256": _hash(scrubbed),
            }
        )
        evidence.append(item)
        operation_targets.append(target)

    public = {
        **base_public,
        "schema": PUBLIC_SCHEMA,
        "evidence": evidence,
    }
    public.pop("identity_sha256")
    public["identity_sha256"] = canonical_sha256(public)
    assessor = {
        **base_assessor,
        "schema": ASSESSOR_SCHEMA,
        "public_identity_sha256": public["identity_sha256"],
        "operation_targets": operation_targets,
        "operation_scrub_aliases": list(scrub_aliases),
    }
    assessor.pop("identity_sha256")
    assessor["identity_sha256"] = canonical_sha256(assessor)
    validate_evaluation_episode(public, assessor)
    return public, assessor


def validate_evaluation_episode(
    public: Mapping[str, Any], assessor: Mapping[str, Any]
) -> None:
    aliases = tuple(str(value) for value in public["aliases"])
    renamed_aliases = tuple(str(value) for value in public["renamed_aliases"])
    scrub_aliases = tuple(str(value) for value in assessor["operation_scrub_aliases"])
    targets = tuple(int(value) for value in assessor["operation_targets"])
    evidence = tuple(public["evidence"])
    if (
        public.get("schema") != PUBLIC_SCHEMA
        or assessor.get("schema") != ASSESSOR_SCHEMA
        or assessor.get("public_identity_sha256") != public.get("identity_sha256")
        or len(aliases) != OPERATIONS
        or len(renamed_aliases) != OPERATIONS
        or len(scrub_aliases) != OPERATIONS
        or len(set((*aliases, *renamed_aliases, *scrub_aliases))) != OPERATIONS * 3
        or len(evidence) != OPERATIONS * 3
        or len(targets) != len(evidence)
        or any(target not in range(OPERATIONS) for target in targets)
        or any("operation" in item or "operation_index" in item for item in evidence)
    ):
        raise OPB1DataError("OPB1 evaluation geometry differs")
    for item, target in zip(evidence, targets, strict=True):
        source = str(item["source_text"])
        renamed = str(item["fully_renamed_source_text"])
        scrubbed = str(item["operation_scrubbed_text"])
        if (
            _whole_word_count(source, aliases[target]) != 4
            or _whole_word_count(renamed, renamed_aliases[target]) != 4
            or _whole_word_count(scrubbed, scrub_aliases[target]) != 4
            or any(
                _whole_word_count(source, alias) != 0
                for index, alias in enumerate(aliases)
                if index != target
            )
            or any(
                _whole_word_count(renamed, alias) != 0
                for index, alias in enumerate(renamed_aliases)
                if index != target
            )
            or any(_whole_word_count(scrubbed, alias) for alias in aliases)
            or _hash(source) != item["source_sha256"]
            or _hash(renamed) != item["fully_renamed_source_sha256"]
            or _hash(scrubbed) != item["operation_scrubbed_sha256"]
        ):
            raise OPB1DataError("OPB1 evidence carrier differs")
    for episode in (public, assessor):
        payload = dict(episode)
        identity = str(payload.pop("identity_sha256"))
        if canonical_sha256(payload) != identity:
            raise OPB1DataError("OPB1 evaluation identity differs")


def rotate_aliases(values: Sequence[str], offset: int = 1) -> tuple[str, ...]:
    table = tuple(str(value) for value in values)
    if len(table) != OPERATIONS or len(set(table)) != OPERATIONS:
        raise OPB1DataError("OPB1 alias rotation geometry differs")
    offset %= OPERATIONS
    return table[offset:] + table[:offset]


__all__ = [
    "ASSESSOR_SCHEMA",
    "CONFIRMATION_SEEDS",
    "DEVELOPMENT_EPISODES",
    "DEVELOPMENT_SEED",
    "OPB1DataError",
    "PUBLIC_SCHEMA",
    "REPORT_SCHEMA",
    "TRAIN_ROWS",
    "TRAIN_SCHEMA",
    "TRAIN_SEED",
    "augment_evaluation_episode",
    "build_training_record",
    "rotate_aliases",
    "validate_evaluation_episode",
    "validate_training_record",
]
