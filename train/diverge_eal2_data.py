#!/usr/bin/env python3
"""Identifiable, composition-held-out transition data for DIVERGE-EAL2."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping

from diverge_eal1_data import (
    DEMONSTRATIONS_PER_OPERATION,
    DEVELOPMENT_EPISODES,
    DEVELOPMENT_MATRICES,
    EAL1DataError,
    OPERATIONS,
    TRAIN_MATRICES,
    TRAIN_ROWS,
    TRANSFER_DEPTHS,
    TRANSFER_PROGRAMS,
    _AFTER,
    _BEFORE,
    _JOINERS,
    _identifying_states,
    _render,
    _word,
    apply_matrix,
    canonical_sha256,
    overlap_report,
    validate_transition_record,
)


TRAIN_SCHEMA = "shohin-diverge-eal2-training-v1"
PUBLIC_SCHEMA = "shohin-diverge-eal2-public-v1"
ASSESSOR_SCHEMA = "shohin-diverge-eal2-assessor-v1"
REPORT_SCHEMA = "shohin-diverge-eal2-data-report-v1"
TRAIN_SEED = 2026080761
DEVELOPMENT_SEED = 2026080762
CONFIRMATION_SEEDS = (2026080763, 2026080764, 2026080765, 2026080766, 2026080767)


def _transition_record(
    *,
    split: str,
    seed: int,
    serial: int,
    operation_index: int,
    operation: str,
    registers: tuple[str, str],
    before: tuple[int, int],
    after: tuple[int, int],
    rng: random.Random,
) -> dict[str, Any]:
    if split not in ("train", "development"):
        raise EAL1DataError("EAL2 transition split differs")
    bucket = 0 if split == "train" else 1
    pairs = tuple(
        (left, right)
        for left in range(len(_BEFORE))
        for right in range(len(_AFTER))
        if (left + right) % 2 == bucket
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


def build_training_record(serial: int) -> dict[str, Any]:
    rng = random.Random(canonical_sha256(["eal2-train", TRAIN_SEED, serial]))
    operation = _word("eal2-operation", TRAIN_SEED, serial, 0)
    registers = (
        _word("eal2-register", TRAIN_SEED, serial, 0),
        _word("eal2-register", TRAIN_SEED, serial, 1),
    )
    matrix = TRAIN_MATRICES[rng.randrange(len(TRAIN_MATRICES))]
    before = (rng.randrange(97), rng.randrange(97))
    record = _transition_record(
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
    output = {"schema": TRAIN_SCHEMA, **record}
    output["identity_sha256"] = canonical_sha256(
        {key: value for key, value in output.items() if key != "identity_sha256"}
    )
    validate_training_record(output)
    return output


def validate_training_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != TRAIN_SCHEMA or record.get("split") != "train":
        raise EAL1DataError("EAL2 training schema differs")
    validate_transition_record(record)


def build_evaluation_episode(
    serial: int, *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(canonical_sha256(["eal2-development", seed, serial]))
    episode_id = canonical_sha256(["eal2-episode", seed, serial])[:24]
    aliases = tuple(
        _word("eal2-alias", seed, serial, index) for index in range(OPERATIONS)
    )
    registers = (
        _word("eal2-register", seed, serial, 0),
        _word("eal2-register", seed, serial, 1),
    )
    matrices = tuple(rng.sample(DEVELOPMENT_MATRICES, OPERATIONS))
    evidence_public = []
    evidence_hidden = []
    transition_serial = serial * OPERATIONS * DEMONSTRATIONS_PER_OPERATION
    for operation_index, (alias, matrix) in enumerate(
        zip(aliases, matrices, strict=True)
    ):
        for demo_index, before in enumerate(_identifying_states(matrix, rng)):
            after = apply_matrix(matrix, before)
            record = _transition_record(
                split="development",
                seed=seed,
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
        initial = (rng.randrange(97), rng.randrange(97))
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
    public = {
        "schema": PUBLIC_SCHEMA,
        "episode_id": episode_id,
        "seed": seed,
        "serial": serial,
        "aliases": list(aliases),
        "registers": list(registers),
        "evidence": evidence_public,
        "transfer": transfers,
        "queries": [
            {"program_id": transfer["program_id"], "register_index": register}
            for transfer in transfers
            for register in range(2)
        ],
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


def build_development_episode(serial: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return build_evaluation_episode(serial, seed=DEVELOPMENT_SEED)


def validate_episode(public: Mapping[str, Any], assessor: Mapping[str, Any]) -> None:
    if (
        public.get("schema") != PUBLIC_SCHEMA
        or assessor.get("schema") != ASSESSOR_SCHEMA
    ):
        raise EAL1DataError("EAL2 episode schema differs")
    public_payload = dict(public)
    public_identity = str(public_payload.pop("identity_sha256"))
    assessor_payload = dict(assessor)
    assessor_identity = str(assessor_payload.pop("identity_sha256"))
    if canonical_sha256(public_payload) != public_identity:
        raise EAL1DataError("EAL2 public identity differs")
    if canonical_sha256(assessor_payload) != assessor_identity:
        raise EAL1DataError("EAL2 assessor identity differs")
    if assessor["public_identity_sha256"] != public["identity_sha256"]:
        raise EAL1DataError("EAL2 public/assessor binding differs")
    if (
        len(public["aliases"]) != OPERATIONS
        or len(public["evidence"]) != OPERATIONS * DEMONSTRATIONS_PER_OPERATION
        or len(public["transfer"]) != TRANSFER_PROGRAMS
        or len(assessor["matrices"]) != OPERATIONS
    ):
        raise EAL1DataError("EAL2 episode geometry differs")


__all__ = [
    "ASSESSOR_SCHEMA",
    "CONFIRMATION_SEEDS",
    "DEVELOPMENT_EPISODES",
    "DEVELOPMENT_SEED",
    "PUBLIC_SCHEMA",
    "REPORT_SCHEMA",
    "TRAIN_ROWS",
    "TRAIN_SCHEMA",
    "TRAIN_SEED",
    "build_development_episode",
    "build_evaluation_episode",
    "build_training_record",
    "overlap_report",
    "validate_episode",
    "validate_training_record",
]
