"""Frozen seven-variant qualification matrix for the ETTR architecture.

This module joins three independently implemented ontology boards without
placing any oracle, family label, alignment, or expected answer on a
candidate-visible surface.  It is offline manifest and audit machinery only.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from functools import lru_cache
import hashlib
import json
from typing import Any

from cross_ontology_horn_board import (
    behavior_signature as horn_behavior_signature,
)
from cross_ontology_horn_variants import (
    HornExpectationRelation,
    materialize_horn_variant_set,
)
from cross_ontology_resource_board import (
    behavior_signature as resource_behavior_signature,
)
from cross_ontology_resource_variants import (
    QUALIFICATION_THEORY_INDICES as RESOURCE_THEORY_INDICES,
    AnswerDirective,
    PairExpectation,
    build_resource_variant_cases,
    build_resource_variants,
)
from cross_ontology_rewrite_board import (
    HELDOUT_THEORY_INDICES as REWRITE_THEORY_INDICES,
    behavior_signature as rewrite_behavior_signature,
)
from cross_ontology_rewrite_variants import (
    QualificationDecision,
    VariantExpectation,
    build_rewrite_variant_family,
)


FOLDS = 3
THEORIES_PER_FOLD = 8
VARIANTS_PER_THEORY = 7
CHALLENGES_PER_VARIANT = 16
PRIMARY_EXECUTIONS = (
    FOLDS
    * THEORIES_PER_FOLD
    * VARIANTS_PER_THEORY
    * CHALLENGES_PER_VARIANT
)
HORN_THEORY_INDICES = (0, 2, 5, 7, 10, 12, 15, 19)


class HeldoutOntology(StrEnum):
    HORN = "fold_0"
    REWRITE = "fold_1"
    RESOURCE = "fold_2"


@dataclass(frozen=True, slots=True)
class QualificationRow:
    fold: int
    heldout_ontology: HeldoutOntology
    theory_index: int
    theory_sha256: str
    variant: str
    challenge_offset: int
    canonical_challenge_index: int
    compiler_source_sha256: str
    challenge_sha256: str
    expected_sha256: str
    directive: str
    expectation: str
    row_sha256: str


@dataclass(frozen=True, slots=True)
class QualificationReceipt:
    fold_count: int
    heldout_theory_count: int
    source_world_count: int
    canonical_challenge_count: int
    primary_execution_count: int
    exact_invariance_execution_count: int
    semantic_separation_execution_count: int
    abstention_execution_count: int
    family_label_leak_count: int
    unique_row_count: int
    unique_theory_hash_count: int
    payload_sha256: str
    all_contracts_pass: bool


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _canonicalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"qualification value is not canonical: {type(value)!r}")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonicalize(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _digest(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _theory_hash(
    ontology: HeldoutOntology,
    theory_index: int,
) -> str:
    if ontology == HeldoutOntology.HORN:
        signature = horn_behavior_signature(theory_index)
    elif ontology == HeldoutOntology.REWRITE:
        signature = rewrite_behavior_signature(theory_index)
    else:
        signature = resource_behavior_signature(theory_index)
    return _digest(
        {
            "ontology_partition": ontology.value,
            "behavior_signature": signature,
        }
    )


def _make_row(
    *,
    fold: int,
    ontology: HeldoutOntology,
    theory_index: int,
    variant: str,
    challenge_offset: int,
    canonical_challenge_index: int,
    compiler_source: bytes,
    challenge: bytes,
    expected: Any,
    directive: str,
    expectation: str,
) -> QualificationRow:
    material = {
        "fold": fold,
        "heldout_ontology": ontology.value,
        "theory_index": theory_index,
        "variant": variant,
        "challenge_offset": challenge_offset,
        "canonical_challenge_index": canonical_challenge_index,
        "compiler_source_sha256": _digest(compiler_source),
        "challenge_sha256": _digest(challenge),
        "expected_sha256": _digest(expected),
        "directive": directive,
        "expectation": expectation,
    }
    return QualificationRow(
        fold=fold,
        heldout_ontology=ontology,
        theory_index=theory_index,
        theory_sha256=_theory_hash(ontology, theory_index),
        variant=variant,
        challenge_offset=challenge_offset,
        canonical_challenge_index=canonical_challenge_index,
        compiler_source_sha256=material["compiler_source_sha256"],
        challenge_sha256=material["challenge_sha256"],
        expected_sha256=material["expected_sha256"],
        directive=directive,
        expectation=expectation,
        row_sha256=_digest(material),
    )


def _horn_rows() -> tuple[QualificationRow, ...]:
    rows = []
    for theory_index in HORN_THEORY_INDICES:
        variants = materialize_horn_variant_set(
            theory_index,
            seed=0xE770 + theory_index,
            challenge_count=CHALLENGES_PER_VARIANT,
        )
        for variant in variants:
            source = variant.compiler_source().encode("ascii")
            for offset, challenge in enumerate(variant.challenges):
                rows.append(
                    _make_row(
                        fold=0,
                        ontology=HeldoutOntology.HORN,
                        theory_index=theory_index,
                        variant=variant.kind.value,
                        challenge_offset=offset,
                        canonical_challenge_index=challenge.challenge_index,
                        compiler_source=source,
                        challenge=challenge.source.encode("ascii"),
                        expected={
                            "possible_terminals": challenge.expected_terminals,
                            "requires_abstention": (
                                challenge.requires_abstention
                            ),
                        },
                        directive=(
                            "abstain"
                            if challenge.requires_abstention
                            else "answer"
                        ),
                        expectation=(
                            variant.expectation.relation_to_base.value
                        ),
                    )
                )
    return tuple(rows)


def _rewrite_rows() -> tuple[QualificationRow, ...]:
    rows = []
    for theory_index in REWRITE_THEORY_INDICES:
        for variant in build_rewrite_variant_family(theory_index):
            source = variant.compiler_source_bytes()
            for offset, canonical_index in enumerate(
                variant.challenge_indices
            ):
                rows.append(
                    _make_row(
                        fold=1,
                        ontology=HeldoutOntology.REWRITE,
                        theory_index=theory_index,
                        variant=variant.kind.value,
                        challenge_offset=offset,
                        canonical_challenge_index=canonical_index,
                        compiler_source=source,
                        challenge=variant.late_challenge_bytes(offset),
                        expected={
                            "possible_outputs": (
                                variant.oracle.possible_outputs[offset]
                            ),
                            "decision": variant.oracle.decision,
                        },
                        directive=(
                            "abstain"
                            if variant.oracle.decision
                            == QualificationDecision.ABSTAIN_AMBIGUOUS
                            else "answer"
                        ),
                        expectation=variant.expectation.value,
                    )
                )
    return tuple(rows)


def _resource_rows() -> tuple[QualificationRow, ...]:
    rows = []
    for theory_index in RESOURCE_THEORY_INDICES:
        sources = {
            variant.name: variant.source.encode("ascii")
            for variant in build_resource_variants(theory_index)
        }
        for case in build_resource_variant_cases(theory_index):
            rows.append(
                _make_row(
                    fold=2,
                    ontology=HeldoutOntology.RESOURCE,
                    theory_index=theory_index,
                    variant=case.variant.value,
                    challenge_offset=(
                        sum(
                            previous.variant == case.variant
                            for previous in rows
                            if previous.theory_index == theory_index
                            and previous.fold == 2
                        )
                    ),
                    canonical_challenge_index=case.challenge_index,
                    compiler_source=sources[case.variant],
                    challenge=case.challenge_source.encode("ascii"),
                    expected={
                        "expected_outcome": case.expected_outcome,
                        "possible_outcomes": case.possible_outcomes,
                        "directive": case.directive,
                    },
                    directive=(
                        "abstain"
                        if case.directive == AnswerDirective.ABSTAIN
                        else "answer"
                    ),
                    expectation=case.pair_expectation.value,
                )
            )
    return tuple(rows)


@lru_cache(maxsize=1)
def build_qualification_matrix() -> tuple[QualificationRow, ...]:
    rows = (*_horn_rows(), *_rewrite_rows(), *_resource_rows())
    if len(rows) != PRIMARY_EXECUTIONS:
        raise ValueError("qualification execution count differs")
    return rows


def audit_qualification_matrix(
    rows: tuple[QualificationRow, ...],
) -> QualificationReceipt:
    if len(rows) != PRIMARY_EXECUTIONS:
        raise ValueError("qualification execution count differs")
    by_world: dict[tuple[int, int, str], list[QualificationRow]] = {}
    for row in rows:
        by_world.setdefault(
            (row.fold, row.theory_index, row.variant),
            [],
        ).append(row)
    if (
        len(by_world)
        != FOLDS * THEORIES_PER_FOLD * VARIANTS_PER_THEORY
        or set(len(group) for group in by_world.values())
        != {CHALLENGES_PER_VARIANT}
    ):
        raise ValueError("qualification world geometry differs")
    if any(
        len({row.compiler_source_sha256 for row in group}) != 1
        or {row.challenge_offset for row in group}
        != set(range(CHALLENGES_PER_VARIANT))
        for group in by_world.values()
    ):
        raise ValueError("qualification world custody differs")

    base_by_challenge = {
        (
            row.fold,
            row.theory_index,
            row.canonical_challenge_index,
        ): row.expected_sha256
        for row in rows
        if row.variant == "base"
    }
    invariant = 0
    separation = 0
    semantic_worlds: dict[tuple[int, int, str], int] = {}
    for row in rows:
        base = base_by_challenge[
            (
                row.fold,
                row.theory_index,
                row.canonical_challenge_index,
            )
        ]
        if row.expectation in {
            HornExpectationRelation.CANONICALLY_INVARIANT.value,
            VariantExpectation.EXACT_INVARIANCE.value,
            PairExpectation.EXACT_INVARIANCE.value,
        }:
            if row.expected_sha256 != base:
                raise ValueError("declared invariant outcome differs")
            invariant += 1
        elif row.expected_sha256 != base:
            separation += 1
            semantic_worlds[
                (row.fold, row.theory_index, row.variant)
            ] = semantic_worlds.get(
                (row.fold, row.theory_index, row.variant),
                0,
            ) + 1

    required_semantic_worlds = {
        (row.fold, row.theory_index, row.variant)
        for row in rows
        if row.expectation
        not in {
            "baseline",
            "reference",
            HornExpectationRelation.CANONICALLY_INVARIANT.value,
            VariantExpectation.EXACT_INVARIANCE.value,
            PairExpectation.EXACT_INVARIANCE.value,
        }
    }
    if not required_semantic_worlds <= set(semantic_worlds):
        raise ValueError("semantic twin lacks a separating challenge")

    forbidden = (
        b"horn",
        b"ontology",
        b"resource",
        b"rewrite",
        b"theory_index",
        b"variant",
    )
    source_payloads: dict[str, bytes] = {}
    for theory_index in HORN_THEORY_INDICES:
        for variant in materialize_horn_variant_set(
            theory_index,
            seed=0xE770 + theory_index,
            challenge_count=CHALLENGES_PER_VARIANT,
        ):
            payload = variant.compiler_source().encode("ascii")
            source_payloads[_digest(payload)] = payload
    for theory_index in REWRITE_THEORY_INDICES:
        for variant in build_rewrite_variant_family(theory_index):
            payload = variant.compiler_source_bytes()
            source_payloads[_digest(payload)] = payload
    for theory_index in RESOURCE_THEORY_INDICES:
        for variant in build_resource_variants(theory_index):
            payload = variant.source.encode("ascii")
            source_payloads[_digest(payload)] = payload
    leak_count = sum(
        any(token in payload.lower() for token in forbidden)
        for payload in source_payloads.values()
    )
    if leak_count:
        raise ValueError("candidate source contains a family label")

    theory_hashes = {row.theory_sha256 for row in rows}
    if len(theory_hashes) != FOLDS * THEORIES_PER_FOLD:
        raise ValueError("heldout theory hashes are not disjoint")
    row_hashes = {row.row_sha256 for row in rows}
    if len(row_hashes) != len(rows):
        raise ValueError("qualification row hashes are not unique")
    payload_sha256 = _digest(
        [
            _canonicalize(row)
            for row in sorted(rows, key=lambda item: item.row_sha256)
        ]
    )
    return QualificationReceipt(
        fold_count=FOLDS,
        heldout_theory_count=len(theory_hashes),
        source_world_count=len(by_world),
        canonical_challenge_count=(
            FOLDS * THEORIES_PER_FOLD * CHALLENGES_PER_VARIANT
        ),
        primary_execution_count=len(rows),
        exact_invariance_execution_count=invariant,
        semantic_separation_execution_count=separation,
        abstention_execution_count=sum(
            row.directive == "abstain"
            for row in rows
        ),
        family_label_leak_count=leak_count,
        unique_row_count=len(row_hashes),
        unique_theory_hash_count=len(theory_hashes),
        payload_sha256=payload_sha256,
        all_contracts_pass=True,
    )


__all__ = [
    "CHALLENGES_PER_VARIANT",
    "FOLDS",
    "HORN_THEORY_INDICES",
    "HeldoutOntology",
    "PRIMARY_EXECUTIONS",
    "QualificationReceipt",
    "QualificationRow",
    "THEORIES_PER_FOLD",
    "VARIANTS_PER_THEORY",
    "audit_qualification_matrix",
    "build_qualification_matrix",
]
