#!/usr/bin/env python3
"""Source-deleted natural transaction runtime for DIVERGE-NPL2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import random

from diverge_npl1_data import parse_program_surface
from diverge_pl1_data import Episode, apply_operation, verify_trace
from diverge_pl1_runtime import (
    Arm,
    BranchReceipt,
    WriteReceipt,
    _canonical_hash,
    _credit_receipts,
    _fast_weight_update,
    _frob,
    _pl1_update,
    _project_write,
    _transient_gradient_update,
    freeze_policy,
    matrix_hash,
    maximum_assignment,
    sample_assignment,
    zero_matrix,
)


@dataclass(frozen=True, slots=True)
class TypedProgram:
    initial_state: tuple[int, int]
    symbols: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TypedEpisode:
    episode_id: str
    branch_names: tuple[str, ...]
    acquisition: tuple[TypedProgram, ...]
    transfer: tuple[TypedProgram, ...]


@dataclass(frozen=True, slots=True)
class DecodedEvidence:
    attempt: int
    target_branch: str
    distractor_branch: str
    certificate_code: int
    commitment: str


@dataclass(frozen=True, slots=True)
class NaturalEpisodeResult:
    arm: str
    episode_id: str
    selected_mapping: tuple[int, ...]
    mapping_exact: bool
    transfer_exact: int
    transfer_total: int
    query_exact: int
    query_total: int
    attempt_passes: tuple[int, ...]
    probe_query_exact: tuple[int, ...]
    policy_hash: str
    policy_state: tuple[tuple[float, ...], ...]
    write_receipts: tuple[WriteReceipt, ...]
    semantic_rejections: int


def typed_episode_from_public(record: Mapping[str, object]) -> TypedEpisode:
    aliases = tuple(str(value) for value in record["aliases"])  # type: ignore[index]
    branches = tuple(str(value) for value in record["branch_names"])  # type: ignore[index]
    registers_raw = tuple(str(value) for value in record["register_names"])  # type: ignore[index]
    if len(aliases) != 8 or len(branches) != 8 or len(registers_raw) != 2:
        raise ValueError("NPL2 typed episode geometry differs")
    registers = (registers_raw[0], registers_raw[1])

    def parse(item: Mapping[str, object]) -> TypedProgram:
        initial, symbols = parse_program_surface(item, aliases, registers)
        return TypedProgram(initial_state=initial, symbols=symbols)

    acquisition = tuple(parse(item) for item in record["acquisition"])  # type: ignore[index]
    transfer = tuple(parse(item) for item in record["transfer"])  # type: ignore[index]
    if len(acquisition) != 12 or len(transfer) != 16:
        raise ValueError("NPL2 typed program geometry differs")
    return TypedEpisode(
        episode_id=str(record["episode_id"]),
        branch_names=branches,
        acquisition=acquisition,
        transfer=transfer,
    )


def execute_typed_mapping(
    mapping: tuple[int, ...], program: TypedProgram
) -> tuple[tuple[int, int], ...]:
    if sorted(mapping) != list(range(8)):
        raise ValueError("NPL2 mapping is not a complete permutation")
    state = program.initial_state
    trace = [state]
    for symbol in program.symbols:
        state = apply_operation(mapping[symbol], state)
        trace.append(state)
    return tuple(trace)


def verification_code(*, passed: bool, first_error: int | None) -> int:
    if passed:
        if first_error is not None:
            raise ValueError("NPL2 passing verification exposes an error")
        return 0
    if first_error is None or first_error <= 0:
        raise ValueError("NPL2 failed verification lacks a legal error position")
    return first_error + 1


def _query_exact(
    typed: TypedEpisode,
    assessor: Episode,
    mapping: tuple[int, ...],
    query_selectors: Sequence[int],
) -> int:
    if len(query_selectors) != 2 * len(typed.transfer):
        raise ValueError("NPL2 query selector geometry differs")
    exact = 0
    for program_index, (public_program, hidden_program) in enumerate(
        zip(typed.transfer, assessor.transfer, strict=True)
    ):
        terminal = execute_typed_mapping(mapping, public_program)[-1]
        for expected_register in range(2):
            selector = int(query_selectors[2 * program_index + expected_register])
            if selector in (0, 1):
                exact += (
                    terminal[selector]
                    == hidden_program.terminal_state[expected_register]
                )
    return exact


def _semantic_receipts(
    *,
    typed: TypedEpisode,
    assessor: Episode,
    attempt: int,
    mappings: tuple[tuple[int, ...], ...],
    evidence: Mapping[tuple[int, int, int], DecodedEvidence],
) -> tuple[tuple[BranchReceipt, ...], int]:
    public_program = typed.acquisition[attempt]
    hidden_program = assessor.acquisition[attempt]
    if (
        public_program.initial_state != hidden_program.initial_state
        or public_program.symbols != hidden_program.symbols
    ):
        raise ValueError("NPL2 public/assessor acquisition differs")
    names = {name: index for index, name in enumerate(typed.branch_names)}
    receipts = []
    rejected = 0
    for branch, mapping in enumerate(mappings):
        trace = execute_typed_mapping(mapping, public_program)
        verified = verify_trace(assessor, hidden_program, trace)
        code = verification_code(
            passed=verified.passed, first_error=verified.first_error
        )
        decoded = evidence[(attempt, branch, code)]
        target = names.get(decoded.target_branch, -1)
        distractor = names.get(decoded.distractor_branch, -1)
        expected_distractor = (branch + (attempt % 7) + 1) % len(typed.branch_names)
        valid = (
            decoded.attempt == attempt
            and target == branch
            and distractor == expected_distractor
            and decoded.certificate_code == code
        )
        prefix = (
            len(public_program.symbols)
            if decoded.certificate_code == 0
            else max(0, decoded.certificate_code - 2)
        )
        if not valid:
            rejected += 1
        receipts.append(
            BranchReceipt(
                branch=branch,
                mapping=mapping,
                passed=decoded.certificate_code == 0,
                correct_prefix=prefix,
                program_depth=len(public_program.symbols),
                receipt=(
                    verified.receipt
                    if valid
                    else f"invalid-semantic:{decoded.commitment}"
                ),
            )
        )
    return tuple(receipts), rejected


def run_natural_episode(
    typed: TypedEpisode,
    assessor: Episode,
    *,
    evidence: Mapping[tuple[int, int, int], DecodedEvidence],
    query_selectors: Sequence[int],
    arm: Arm,
    seed: int,
    candidate_label: str | None = None,
    proposal_arm: Arm | None = None,
    branches: int = 8,
    write_budget: float = 4.0,
    credit_control: str = "normal",
    reset_before_transfer: bool = False,
    homeostatic: bool = True,
    protected_manifest: Mapping[str, str] | None = None,
    inject_protected_mutation: bool = False,
) -> NaturalEpisodeResult:
    """Run unchanged PL1 updates using only precompiled natural receipts."""

    if typed.episode_id != assessor.episode_id or branches <= 0:
        raise ValueError("NPL2 episode or branch contract differs")
    rng_label = proposal_arm or arm
    rng = random.Random(
        _canonical_hash("diverge-pl1-run", [typed.episode_id, rng_label, seed])
    )
    scores = zero_matrix()
    protected = dict(
        protected_manifest
        or {
            "world_owner": "immutable-world-owner-v1",
            "evidence_owner": "immutable-evidence-owner-v1",
            "referent_owner": "immutable-eic1-query-owner-v1",
            "executor": "exact-z97-executor-v1",
        }
    )
    protected_hash = _canonical_hash("diverge-npl2-protected", protected)
    cumulative_write = 0.0
    writes = []
    attempt_passes = []
    probe_query_exact = []
    best_mapping = tuple(range(8))
    best_prefix = -1
    semantic_rejections = 0

    for attempt, program in enumerate(typed.acquisition):
        mappings = tuple(sample_assignment(scores, rng) for _ in range(branches))
        receipts, semantic_rejected = _semantic_receipts(
            typed=typed,
            assessor=assessor,
            attempt=attempt,
            mappings=mappings,
            evidence=evidence,
        )
        semantic_rejections += semantic_rejected
        accepted = tuple(
            receipt
            for receipt in receipts
            if not receipt.receipt.startswith("invalid-")
        )
        attempt_passes.append(sum(receipt.passed for receipt in accepted))
        for receipt in accepted:
            if receipt.correct_prefix > best_prefix:
                best_mapping = receipt.mapping
                best_prefix = receipt.correct_prefix

        if arm in {"STATIC", "DIVERGE_ONLY"}:
            update, rejected = _pl1_update(
                assessor, assessor.acquisition[attempt], receipts, localized=True
            )
            _project_write(update, write_budget)
            semantic_rejections += rejected
            probe_mapping = (
                best_mapping if arm == "DIVERGE_ONLY" else maximum_assignment(scores)
            )
            probe_query_exact.append(
                _query_exact(typed, assessor, probe_mapping, query_selectors)
            )
            continue

        credited = _credit_receipts(receipts, credit_control, rng)  # type: ignore[arg-type]
        if arm in {"CONTEXT_ONLY", "PL1"}:
            update, rejected_credits = _pl1_update(
                assessor,
                assessor.acquisition[attempt],
                credited,
                localized=credit_control != "no_eligibility",
            )
        elif arm == "FAST_WEIGHT":
            update = _fast_weight_update(credited)
            rejected_credits = 0
        elif arm == "TRANSIENT_GRAD":
            update = _transient_gradient_update(credited)
            rejected_credits = 0
        else:
            raise ValueError(f"unknown NPL2 arm {arm}")

        pre_hash = matrix_hash(scores)
        if homeostatic:
            update, update_norm = _project_write(update, write_budget)
        else:
            update_norm = _frob(update)
        cumulative_write += update_norm
        if homeostatic:
            scores = [
                [
                    max(-8.0, min(8.0, scores[row][column] + update[row][column]))
                    for column in range(8)
                ]
                for row in range(8)
            ]
        else:
            scores = [
                [scores[row][column] + update[row][column] for column in range(8)]
                for row in range(8)
            ]
        if inject_protected_mutation and attempt == 0:
            protected["referent_owner"] = "MUTATED"
        if _canonical_hash("diverge-npl2-protected", protected) != protected_hash:
            raise RuntimeError("protected NPL2 owner changed during plastic commit")
        writes.append(
            WriteReceipt(
                attempt=attempt,
                pre_hash=pre_hash,
                post_hash=matrix_hash(scores),
                update_norm=update_norm,
                cumulative_write=cumulative_write,
                protected_hash=protected_hash,
                rejected_credits=rejected_credits,
            )
        )
        probe_query_exact.append(
            _query_exact(typed, assessor, maximum_assignment(scores), query_selectors)
        )

    if arm == "CONTEXT_ONLY" or reset_before_transfer:
        scores = zero_matrix()
    selected = best_mapping if arm == "DIVERGE_ONLY" else maximum_assignment(scores)
    transfer_exact = sum(
        execute_typed_mapping(selected, public)[-1] == hidden.terminal_state
        for public, hidden in zip(typed.transfer, assessor.transfer, strict=True)
    )
    return NaturalEpisodeResult(
        arm=candidate_label or arm,
        episode_id=typed.episode_id,
        selected_mapping=selected,
        mapping_exact=selected == assessor.symbol_to_operation,
        transfer_exact=transfer_exact,
        transfer_total=len(typed.transfer),
        query_exact=_query_exact(typed, assessor, selected, query_selectors),
        query_total=len(query_selectors),
        attempt_passes=tuple(attempt_passes),
        probe_query_exact=tuple(probe_query_exact),
        policy_hash=matrix_hash(scores),
        policy_state=freeze_policy(scores),
        write_receipts=tuple(writes),
        semantic_rejections=semantic_rejections,
    )


__all__ = [
    "DecodedEvidence",
    "NaturalEpisodeResult",
    "TypedEpisode",
    "TypedProgram",
    "execute_typed_mapping",
    "run_natural_episode",
    "typed_episode_from_public",
    "verification_code",
]
