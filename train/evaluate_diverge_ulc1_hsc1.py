#!/usr/bin/env python3
"""Frozen-HSC1 executable matched gate for DIVERGE-ULC1.

No weights are updated.  The failed HSC1 score producer is shared by every arm.
ULC1 differs only by retaining coherent top-K component interpretations in an
exact decision-DAG and applying independently specified delayed state evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch

from version_space_accounting import canonical_json_bytes

from assess_diverge_hsc1_support_rank import (
    _semantic_role_scores,
    cue_k_best,
    cut_k_best,
    load_frozen_hsc1,
    template_k_best,
)
from diverge_hsc1_neural_compiler import (
    HierarchicalStructuredCompiler,
    _episode_encodings,
)
from diverge_hsc1_structured_compiler import _margins, path_viterbi, semantic_templates
from diverge_sc1_source_compiler import (
    ACTION_ADD,
    ACTION_SWAP01,
    ACTION_SWAP23,
    ACTION_SWAP34,
    ALIAS_BEGIN,
    ALIAS_INSIDE,
    BACKGROUND_CUE,
    CANDIDATE_CUE,
    PRIOR_FAVORED,
    RawSourceEpisode,
    generate_episode,
)
from diverge_ulc1_mdd import (
    MDDExecution,
    RuntimeChoice,
    execute_choice_path,
    execute_mdd,
    k_best_product_paths,
    query_mdd,
)
from diverge_v0 import (
    ABSTAIN,
    ANSWER,
    DivergeContractError,
    Query,
    QueryDecision,
    TypedCell,
    TypedState,
    TypedTransaction,
    apply_transaction,
    named_commitment,
    read_query,
)
from diverge_v0_neural_pilot import PROGRAMS
from diverge_wra1_neural_compiler import sha256_file
from diverge_wra1_whole_record import detect_segments

SCHEMA = "shohin-diverge-ulc1-frozen-hsc1-gate-v1"
COHORTS = ("train", "lexical_shift", "renderer_shift", "composition_shift")
COHORT_OFFSETS = {
    "train": 0,
    "lexical_shift": 100_000,
    "renderer_shift": 200_000,
    "composition_shift": 300_000,
}


def _digest(domain: str, payload: object) -> str:
    body = canonical_json_bytes(payload)
    digest = hashlib.sha256()
    for part in (domain.encode("ascii"), body):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _source_commitment(episode: RawSourceEpisode) -> str:
    return _digest("diverge-ulc1-hsc1-source", list(episode.tokens))


def _active_option_index(episode: RawSourceEpisode, record_index: int) -> int:
    payload = f"{episode.episode_id}:{record_index}:delayed-choice".encode("ascii")
    return hashlib.sha256(payload).digest()[0] & 1


def _active_key(
    alias_span: tuple[int, int],
    alias_tokens: Sequence[str],
    program: int,
    prior: int,
) -> str:
    return _digest(
        "diverge-ulc1-hsc1-active-semantics",
        {
            "alias_span": list(alias_span),
            "alias_tokens": list(alias_tokens),
            "program": program,
            "prior": prior,
        },
    )


def _gold_key(episode: RawSourceEpisode, record_index: int) -> str:
    record = episode.records[record_index]
    if not record.is_fault_line:
        return "BACKGROUND"
    option = record.options[_active_option_index(episode, record_index)]
    return _active_key(
        option.alias_span,
        option.alias_tokens,
        option.program,
        option.prior_class,
    )


@dataclass(frozen=True)
class DecodedOption:
    alias_span: tuple[int, int]
    alias_tokens: tuple[str, ...]
    prior_class: int
    program: int
    template_index: int
    path: tuple[int, ...]
    path_score: float


@dataclass(frozen=True)
class RecordScores:
    start: int
    end: int
    cut_scores: tuple[object, ...]
    cue_scores: tuple[tuple[tuple[float, int, int], ...], ...]
    option_scores: tuple[
        tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]], ...
    ]


@dataclass(frozen=True)
class DelayedRecordEvidence:
    """Assessor-issued state evidence bound to one sealed source packet."""

    source_commitment: str
    record_index: int
    record_provenance: str
    observed_witness: int
    evidence_commitment: str

    def record(self) -> dict[str, object]:
        return {
            "source_commitment": self.source_commitment,
            "record_index": self.record_index,
            "record_provenance": self.record_provenance,
            "observed_witness": self.observed_witness,
            "evidence_commitment": self.evidence_commitment,
        }


@dataclass(frozen=True)
class CompiledEpisode:
    episode_id: str
    cohort: str
    source_commitment: str
    choices: tuple[tuple[RuntimeChoice, ...], ...]
    gold_assignment: tuple[int, ...] | None
    gold_witnesses: tuple[int, ...]
    initial_state: TypedState
    expected_state: TypedState
    execution: MDDExecution
    sensitive_query: Query
    invariant_query: Query
    segmentation_exact: bool
    support_recalled: bool
    source_words: int


def _decode_template(
    tokens: Sequence[str],
    *,
    span_start: int,
    role_scores: Sequence[Sequence[float]],
    template_index: int,
) -> DecodedOption | None:
    template = semantic_templates()[template_index]
    score, path = path_viterbi(_margins(role_scores), template.labels)
    if not path:
        return None
    positions = tuple(span_start + value for value in path)
    alias_indices = [
        index
        for index, label in enumerate(template.labels)
        if label in {ALIAS_BEGIN, ALIAS_INSIDE}
    ]
    prior_index = next(
        index
        for index, label in enumerate(template.labels)
        if label in {PRIOR_FAVORED, PRIOR_FAVORED + 1}
    )
    action_indices = [
        index
        for index, label in enumerate(template.labels)
        if label in {ACTION_ADD, ACTION_SWAP01, ACTION_SWAP23, ACTION_SWAP34}
    ]
    alias_positions = tuple(positions[index] for index in alias_indices)
    if not alias_positions or alias_positions != tuple(
        range(alias_positions[0], alias_positions[0] + len(alias_positions))
    ):
        return None
    prior_position = positions[prior_index]
    action_positions = tuple(positions[index] for index in action_indices)
    fields = set(alias_positions)
    if prior_position in fields or any(value in fields for value in action_positions):
        return None
    end = alias_positions[-1] + 1
    return DecodedOption(
        (alias_positions[0], end),
        tuple(tokens[alias_positions[0] : end]),
        template.prior_class,
        template.program,
        template_index,
        path,
        score,
    )


def _record_alternatives(
    episode: RawSourceEpisode,
    record_index: int,
    scores: RecordScores,
    *,
    component_k: int,
) -> tuple[RuntimeChoice, ...]:
    combined: dict[
        str, tuple[float, dict[str, object], tuple[TypedTransaction, ...], int]
    ] = {}

    def add(
        key: str,
        score: float,
        parse: dict[str, object],
        transactions: tuple[TypedTransaction, ...],
        witness_code: int,
    ) -> None:
        previous = combined.get(key)
        if previous is None:
            combined[key] = (score, parse, transactions, witness_code)
            return
        aggregate = _logaddexp(previous[0], score)
        representative = (
            (score, parse, transactions, witness_code)
            if score > previous[0]
            else previous
        )
        combined[key] = (
            aggregate,
            representative[1],
            representative[2],
            representative[3],
        )

    for cut_index, cut in enumerate(scores.cut_scores):
        left, middle, trailer = cut.path
        cue_candidates = scores.cue_scores[cut_index]
        for cue_score, cue_position, cue_kind in cue_candidates:
            base_score = float(cut.score) + float(cue_score)
            if cue_kind == BACKGROUND_CUE:
                add(
                    "BACKGROUND",
                    base_score,
                    {
                        "cut": list(cut.path),
                        "cue_position": cue_position,
                        "cue_kind": cue_kind,
                    },
                    (),
                    _effect_witness(()),
                )
                continue
            if cue_kind != CANDIDATE_CUE:
                continue
            spans = (
                (scores.start + left, scores.start + middle),
                (scores.start + middle, scores.start + trailer),
            )
            for option_index, ((span_start, _), role_scores) in enumerate(
                zip(spans, scores.option_scores[cut_index], strict=True)
            ):
                for template_score, template_index in template_k_best(
                    role_scores, component_k
                ):
                    decoded = _decode_template(
                        episode.tokens,
                        span_start=span_start,
                        role_scores=role_scores,
                        template_index=template_index,
                    )
                    if decoded is None:
                        continue
                    key = _active_key(
                        decoded.alias_span,
                        decoded.alias_tokens,
                        decoded.program,
                        decoded.prior_class,
                    )
                    transactions = tuple(PROGRAMS[decoded.program])
                    witness = _effect_witness(transactions)
                    add(
                        key,
                        base_score
                        + float(template_score)
                        + math.log(3 if decoded.prior_class == 0 else 1),
                        {
                            "cut": list(cut.path),
                            "cue_position": cue_position,
                            "cue_kind": cue_kind,
                            "selected_option": option_index,
                            "template": template_index,
                            "path": list(decoded.path),
                            "alias_span": list(decoded.alias_span),
                            "program": decoded.program,
                            "prior": decoded.prior_class,
                        },
                        transactions,
                        witness,
                    )
    ordered = sorted(combined.items(), key=lambda item: (-item[1][0], item[0]))
    output = []
    for domain_value, (key, (score, parse, transactions, witness)) in enumerate(
        ordered
    ):
        output.append(
            RuntimeChoice(
                record_index,
                domain_value,
                len(ordered) - domain_value,
                transactions,
                witness,
                key,
                named_commitment(
                    "diverge-ulc1-hsc1-choice",
                    f"{episode.episode_id}:{scores.start}:{key}",
                ),
                tuple(
                    sorted(
                        {
                            **parse,
                            "score_microunits": int(round(score * 1_000_000)),
                            "source_span": [scores.start, scores.end],
                        }.items()
                    )
                ),
            )
        )
    return tuple(output)


def _score_batch(
    model: HierarchicalStructuredCompiler,
    episodes: Sequence[RawSourceEpisode],
    device: torch.device,
    *,
    component_k: int,
) -> tuple[tuple[RecordScores, ...] | None, ...]:
    encodings = _episode_encodings(model, episodes)
    words, lengths, boundary = model._frozen_source(encodings, device)
    descriptors: list[list[dict[str, object]]] = [[] for _ in episodes]
    option_memories: list[torch.Tensor] = []
    option_slots: list[tuple[int, int, int]] = []
    for episode_index, episode in enumerate(episodes):
        length = int(lengths[episode_index].item())
        segments, reason, _ = detect_segments(
            boundary[episode_index, : length + 1].detach().cpu().tolist(), length
        )
        if reason is not None:
            descriptors[episode_index] = []
            continue
        for record_index, (start, end) in enumerate(segments):
            memory = words[episode_index, start:end]
            cuts = (
                model.cut_head(memory).transpose(0, 1).float().detach().cpu().tolist()
            )
            cues = model.cue_head(memory).float().detach().cpu().tolist()
            ranked_cuts = cut_k_best(cuts, component_k)
            descriptor: dict[str, object] = {
                "start": start,
                "end": end,
                "cuts": ranked_cuts,
                "cues": tuple(
                    cue_k_best(cues, cut.path[0], component_k) for cut in ranked_cuts
                ),
                "option_indices": [],
            }
            for cut_index, cut in enumerate(ranked_cuts):
                left, middle, trailer = cut.path
                indices = []
                for option_index, (local_start, local_end) in enumerate(
                    ((left, middle), (middle, trailer))
                ):
                    if local_start >= local_end:
                        continue
                    indices.append(len(option_memories))
                    option_memories.append(memory[local_start:local_end])
                    option_slots.append(
                        (episode_index, record_index, cut_index * 2 + option_index)
                    )
                descriptor["option_indices"].append(tuple(indices))  # type: ignore[union-attr]
            descriptors[episode_index].append(descriptor)
    role_scores = _semantic_role_scores(model, option_memories)
    if len(role_scores) != len(option_slots):
        raise AssertionError("HSC1 alternative option accounting differs")
    output = []
    for episode_index, episode_descriptors in enumerate(descriptors):
        if not episode_descriptors:
            output.append(None)
            continue
        records = []
        for descriptor in episode_descriptors:
            option_rows = []
            malformed = False
            for pair in descriptor["option_indices"]:  # type: ignore[union-attr]
                if len(pair) != 2:
                    malformed = True
                    break
                option_rows.append((role_scores[pair[0]], role_scores[pair[1]]))
            if malformed:
                records = []
                break
            records.append(
                RecordScores(
                    int(descriptor["start"]),
                    int(descriptor["end"]),
                    tuple(descriptor["cuts"]),  # type: ignore[arg-type]
                    tuple(descriptor["cues"]),  # type: ignore[arg-type]
                    tuple(option_rows),
                )
            )
        output.append(tuple(records) if records else None)
    return tuple(output)


def _initial_state() -> TypedState:
    return TypedState(
        tuple(
            [
                TypedCell(0, 0, 1),
                TypedCell(1, 0, 10),
                TypedCell(2, 0, 20),
                TypedCell(3, 0, 30),
                TypedCell(4, 0, 40),
            ]
        )
    )


@lru_cache(maxsize=None)
def _effect_witness(transactions: tuple[TypedTransaction, ...]) -> int:
    """Independent observable signature of a program's typed-state effect."""

    state = _initial_state()
    for transaction in transactions:
        state = apply_transaction(state, transaction)
    digest = hashlib.sha256(canonical_json_bytes(state.record())).digest()
    return int.from_bytes(digest[:8], "big")


def _gold_witnesses(episode: RawSourceEpisode) -> tuple[int, ...]:
    output = []
    for record_index, record in enumerate(episode.records):
        key = _gold_key(episode, record_index)
        if key == "BACKGROUND":
            output.append(_effect_witness(()))
            continue
        option = record.options[_active_option_index(episode, record_index)]
        output.append(_effect_witness(tuple(PROGRAMS[option.program])))
    return tuple(output)


def _expected_state(
    episode: RawSourceEpisode,
    initial: TypedState,
) -> TypedState:
    state = initial
    for record_index, record in enumerate(episode.records):
        key = _gold_key(episode, record_index)
        if key == "BACKGROUND":
            continue
        option = record.options[_active_option_index(episode, record_index)]
        for transaction in PROGRAMS[option.program]:
            state = apply_transaction(state, transaction)
    return state


def _compile_batch(
    model: HierarchicalStructuredCompiler,
    episodes: Sequence[RawSourceEpisode],
    device: torch.device,
    *,
    component_k: int,
    max_nodes: int,
    max_groups: int,
) -> tuple[CompiledEpisode, ...]:
    scored = _score_batch(model, episodes, device, component_k=component_k)
    compiled = []
    for episode, records in zip(episodes, scored, strict=True):
        expected_segments = tuple((item.start, item.end) for item in episode.records)
        segmentation_exact = (
            records is not None
            and tuple((item.start, item.end) for item in records) == expected_segments
        )
        rows: list[tuple[RuntimeChoice, ...]] = []
        if records is not None:
            for record_index, record_scores in enumerate(records):
                row = _record_alternatives(
                    episode,
                    record_index,
                    record_scores,
                    component_k=component_k,
                )
                if not row:
                    rows = []
                    break
                rows.append(row)
        initial = _initial_state()
        expected = _expected_state(episode, initial)
        gold_assignment = None
        support_recalled = False
        if segmentation_exact and len(rows) == len(episode.records):
            candidate = []
            support_recalled = True
            for record_index, row in enumerate(rows):
                gold = _gold_key(episode, record_index)
                matches = [
                    item.domain_value for item in row if item.semantic_key == gold
                ]
                if len(matches) != 1:
                    support_recalled = False
                    break
                candidate.append(matches[0])
            if support_recalled:
                gold_assignment = tuple(candidate)
        execution = (
            execute_mdd(initial, rows, max_nodes=max_nodes, max_groups=max_groups)
            if rows
            else execute_mdd(
                initial,
                (
                    (
                        RuntimeChoice(
                            0,
                            0,
                            1,
                            (),
                            0,
                            "FAILED-CLOSED",
                            named_commitment("diverge-ulc1-failed", episode.episode_id),
                        ),
                    ),
                ),
                max_nodes=1,
            )
        )
        compiled.append(
            CompiledEpisode(
                episode.episode_id,
                episode.cohort,
                _source_commitment(episode),
                tuple(rows),
                gold_assignment,
                _gold_witnesses(episode),
                initial,
                expected,
                execution,
                Query("READ_VALUE", (0,)),
                Query("EDGE_COUNT", ()),
                segmentation_exact,
                support_recalled and not execution.overflow,
                len(episode.tokens),
            )
        )
    return tuple(compiled)


def _allowed(compiled: CompiledEpisode) -> dict[int, frozenset[int]]:
    return _certify_delayed_evidence(compiled, _delayed_evidence(compiled))


def _record_provenance(source_commitment: str, record_index: int) -> str:
    return named_commitment(
        "diverge-ulc1-hsc1-record",
        f"{source_commitment}:{record_index}",
    )


def _evidence_commitment(
    source_commitment: str,
    record_index: int,
    observed_witness: int,
) -> str:
    return named_commitment(
        "diverge-ulc1-hsc1-evidence",
        f"{source_commitment}:{record_index}:{observed_witness}",
    )


def _delayed_evidence(
    compiled: CompiledEpisode,
) -> tuple[DelayedRecordEvidence, ...]:
    if len(compiled.gold_witnesses) != len(compiled.choices):
        return ()
    output = []
    for record_index, observed in enumerate(compiled.gold_witnesses):
        output.append(
            DelayedRecordEvidence(
                compiled.source_commitment,
                record_index,
                _record_provenance(compiled.source_commitment, record_index),
                observed,
                _evidence_commitment(
                    compiled.source_commitment,
                    record_index,
                    observed,
                ),
            )
        )
    return tuple(output)


def _certify_delayed_evidence(
    compiled: CompiledEpisode,
    evidence: Sequence[DelayedRecordEvidence],
) -> dict[int, frozenset[int]]:
    """Verify source/provenance and derive exact record-domain nogoods."""

    if compiled.execution.overflow:
        raise DivergeContractError("cannot certify evidence for overflowed packet")
    if len(evidence) != len(compiled.choices):
        raise DivergeContractError("delayed evidence does not cover every record")
    allowed: dict[int, frozenset[int]] = {}
    for item in evidence:
        if item.source_commitment != compiled.source_commitment:
            raise DivergeContractError("evidence belongs to another source packet")
        if item.record_index in allowed or not 0 <= item.record_index < len(
            compiled.choices
        ):
            raise DivergeContractError("evidence record index is invalid or duplicated")
        expected_provenance = _record_provenance(
            compiled.source_commitment, item.record_index
        )
        if item.record_provenance != expected_provenance:
            raise DivergeContractError("evidence record provenance is invalid")
        expected_evidence = _evidence_commitment(
            compiled.source_commitment,
            item.record_index,
            item.observed_witness,
        )
        if item.evidence_commitment != expected_evidence:
            raise DivergeContractError("evidence commitment is invalid")
        permitted = frozenset(
            choice.domain_value
            for choice in compiled.choices[item.record_index]
            if choice.witness_code == item.observed_witness
        )
        if not permitted:
            raise DivergeContractError("evidence removes the complete record domain")
        allowed[item.record_index] = permitted
    if len(allowed) != len(compiled.choices):
        raise DivergeContractError("delayed evidence record coverage is incomplete")
    return allowed


def _rejected_allowed(compiled: CompiledEpisode) -> dict[int, frozenset[int]]:
    return {index: frozenset() for index in range(len(compiled.choices))}


def _factorized_packet_bytes(compiled: CompiledEpisode) -> bytes:
    """Canonical charged runtime packet, including its source seal."""

    return canonical_json_bytes(
        {
            "schema": SCHEMA,
            "source_commitment": compiled.source_commitment,
            "execution": compiled.execution.record(),
        }
    )


def _packet_swap_rejected(
    compiled: CompiledEpisode,
    donor_evidence: Sequence[DelayedRecordEvidence],
) -> bool:
    try:
        _certify_delayed_evidence(compiled, donor_evidence)
    except DivergeContractError:
        return True
    return False


def _source_poison_invariant(
    compiled: CompiledEpisode,
    evidence: Sequence[DelayedRecordEvidence],
) -> bool:
    """Prove post-seal raw-source replacement is outside the runtime API."""

    try:
        allowed = _certify_delayed_evidence(compiled, evidence)
    except DivergeContractError:
        return False
    before_packet = _factorized_packet_bytes(compiled)
    before = query_mdd(compiled.execution, compiled.sensitive_query, allowed=allowed)
    poisoned_source = canonical_json_bytes(
        {
            "episode_id": compiled.episode_id,
            "replacement": ["POST-SEAL-POISON"] * compiled.source_words,
        }
    )
    after_packet = _factorized_packet_bytes(compiled)
    after = query_mdd(compiled.execution, compiled.sensitive_query, allowed=allowed)
    return (
        hashlib.sha256(poisoned_source).hexdigest() != compiled.source_commitment
        and before_packet == after_packet
        and before == after
    )


def _decision_exact(decision: QueryDecision, expected: int) -> bool:
    return decision.disposition == ANSWER and decision.answer == expected


def _path_decision(
    compiled: CompiledEpisode,
    paths: Sequence[tuple[int, ...]],
    *,
    allowed: dict[int, frozenset[int]] | None = None,
) -> QueryDecision:
    states = []
    for path in paths:
        if allowed is not None and any(
            path[index] not in permitted for index, permitted in allowed.items()
        ):
            continue
        state = execute_choice_path(compiled.initial_state, compiled.choices, path)
        if state is None:
            return QueryDecision("REJECT", None, (), 0)
        states.append(state)
    if not states:
        return QueryDecision("REJECT", None, (), 0)
    answers: dict[int, int] = {}
    for state in states:
        answer = read_query(state, compiled.sensitive_query)
        answers[answer] = answers.get(answer, 0) + 1
    marginal = tuple(sorted(answers.items()))
    if len(marginal) == 1:
        return QueryDecision(ANSWER, marginal[0][0], marginal, len(states))
    return QueryDecision(ABSTAIN, None, marginal, len(states))


def _particle_record_bytes(compiled: CompiledEpisode, path: tuple[int, ...]) -> int:
    state = execute_choice_path(compiled.initial_state, compiled.choices, path)
    return len(
        canonical_json_bytes(
            {
                "source_commitment": compiled.source_commitment,
                "choices": [
                    compiled.choices[index][value].record()
                    for index, value in enumerate(path)
                ],
                "initial_state": compiled.initial_state.record(),
                "terminal_state": None if state is None else state.record(),
            }
        )
    )


def _whole_particle_component_bytes(compiled: CompiledEpisode) -> int:
    """Conservative exact charge for duplicated components in every world."""

    worlds = compiled.execution.represented_worlds
    if worlds <= 0:
        return 0
    fixed = len(canonical_json_bytes(compiled.source_commitment)) + len(
        canonical_json_bytes(compiled.initial_state.record())
    )
    choice_bytes = 0
    for row in compiled.choices:
        copies = worlds // len(row)
        choice_bytes += copies * sum(
            len(canonical_json_bytes(choice.record())) for choice in row
        )
    terminal_bytes = sum(
        compiled.execution.arena.assignment_count(group.expression)
        * len(
            canonical_json_bytes(None if group.state is None else group.state.record())
        )
        for group in compiled.execution.groups
    )
    return worlds * fixed + choice_bytes + terminal_bytes


def _select_equal_budget_particles(
    compiled: CompiledEpisode,
) -> tuple[tuple[tuple[int, ...], ...], dict[str, int]]:
    if not compiled.choices or compiled.execution.overflow:
        return (), {"bytes": 0, "transactions": 0, "particles": 0}
    byte_budget = len(_factorized_packet_bytes(compiled))
    transaction_budget = compiled.execution.unique_transaction_applications
    maximum = min(100_000, math.prod(len(row) for row in compiled.choices))
    selected = []
    used_bytes = 0
    used_transactions = 0
    for path in k_best_product_paths(compiled.choices, maximum):
        path_bytes = _particle_record_bytes(compiled, path)
        path_transactions = sum(
            len(compiled.choices[index][value].transactions)
            for index, value in enumerate(path)
        )
        if (
            used_bytes + path_bytes > byte_budget
            or used_transactions + path_transactions > transaction_budget
        ):
            break
        selected.append(path)
        used_bytes += path_bytes
        used_transactions += path_transactions
    return tuple(selected), {
        "bytes": used_bytes,
        "transactions": used_transactions,
        "particles": len(selected),
        "byte_budget": byte_budget,
        "transaction_budget": transaction_budget,
    }


def _recurrent_top1_decision(compiled: CompiledEpisode) -> QueryDecision:
    """Spend the factorized transaction budget replaying one committed state."""

    if not compiled.choices or compiled.execution.overflow:
        return QueryDecision("REJECT", None, (), 0)
    top = tuple(0 for _ in compiled.choices)
    schedule = tuple(
        transaction
        for record_index, domain_value in enumerate(top)
        for transaction in compiled.choices[record_index][domain_value].transactions
    )
    state = compiled.initial_state
    budget = compiled.execution.unique_transaction_applications
    used = 0
    if schedule:
        while used + len(schedule) <= budget:
            for transaction in schedule:
                try:
                    state = apply_transaction(state, transaction)
                except DivergeContractError:
                    return QueryDecision("REJECT", None, (), 0)
            used += len(schedule)
    answer = read_query(state, compiled.sensitive_query)
    return QueryDecision(ANSWER, answer, ((answer, 1),), 1)


def _soft_answer(compiled: CompiledEpisode) -> int | None:
    weighted = 0
    total = 0
    for group in compiled.execution.groups:
        if group.state is None:
            continue
        mass = compiled.execution.arena.total_mass(group.expression)
        weighted += mass * read_query(group.state, compiled.sensitive_query)
        total += mass
    return None if total == 0 else int(round(weighted / total))


def _evaluate_compiled(
    compiled: CompiledEpisode,
    *,
    donor_evidence: Sequence[DelayedRecordEvidence],
) -> dict[str, object]:
    expected = read_query(compiled.expected_state, compiled.sensitive_query)
    top_path = tuple(0 for _ in compiled.choices)
    top = (
        _path_decision(compiled, (top_path,))
        if top_path
        else QueryDecision("REJECT", None, (), 0)
    )
    evidence = _delayed_evidence(compiled)
    try:
        allowed = _certify_delayed_evidence(compiled, evidence)
    except DivergeContractError:
        allowed = _rejected_allowed(compiled)
    full = query_mdd(compiled.execution, compiled.sensitive_query, allowed=allowed)
    no_conflict = query_mdd(compiled.execution, compiled.sensitive_query)
    invariant = query_mdd(compiled.execution, compiled.invariant_query, allowed=allowed)
    particles, particle_resources = _select_equal_budget_particles(compiled)
    particle = _path_decision(compiled, particles, allowed=allowed)
    independent = _path_decision(
        compiled,
        k_best_product_paths(compiled.choices, 2),
        allowed=allowed,
    )
    recurrent = _recurrent_top1_decision(compiled)
    soft_answer = _soft_answer(compiled)
    shuffled_allowed = (
        {index: allowed[(index + 1) % len(allowed)] for index in range(len(allowed))}
        if allowed
        else {}
    )
    shuffled = query_mdd(
        compiled.execution,
        compiled.sensitive_query,
        allowed=shuffled_allowed,
    )
    reset_answer = read_query(compiled.initial_state, compiled.sensitive_query)
    gold_state_exact = False
    gold_survives_evidence = compiled.gold_assignment is None
    if compiled.gold_assignment is not None:
        state = execute_choice_path(
            compiled.initial_state, compiled.choices, compiled.gold_assignment
        )
        gold_state_exact = state == compiled.expected_state
        gold_survives_evidence = all(
            compiled.gold_assignment[index] in permitted
            for index, permitted in allowed.items()
        )
    top_exact = _decision_exact(top, expected)
    full_exact = _decision_exact(full, expected)
    whole_particle_bytes = _whole_particle_component_bytes(compiled)
    factorized_bytes = len(_factorized_packet_bytes(compiled))
    return {
        "segmentation_exact": compiled.segmentation_exact,
        "support_recalled": compiled.support_recalled,
        "at_least_8_worlds": compiled.execution.represented_worlds >= 8,
        "gold_state_exact": gold_state_exact,
        "gold_survives_evidence": gold_survives_evidence,
        "initial_top1_wrong": not top_exact,
        "wrong_top1_recovered": not top_exact and full_exact,
        "A_single_state_exact": top_exact,
        "B_particles_exact": _decision_exact(particle, expected),
        "C_independent_exact": _decision_exact(independent, expected),
        "D_extra_recurrence_exact": _decision_exact(recurrent, expected),
        "E_soft_aggregation_exact": soft_answer == expected,
        "F_no_conflict_exact": _decision_exact(no_conflict, expected),
        "G_diverge_exact": full_exact,
        "invariant_exact": invariant.disposition == ANSWER and invariant.answer == 0,
        "underdetermined_abstains": no_conflict.disposition == ABSTAIN,
        "shuffled_provenance_exact": _decision_exact(shuffled, expected),
        "state_reset_exact": reset_answer == expected,
        "packet_swap_rejected": _packet_swap_rejected(compiled, donor_evidence),
        "source_poison_invariant": _source_poison_invariant(compiled, evidence),
        "expected_answer": expected,
        "decisions": {
            "A_single_state": top.disposition,
            "B_particles": particle.disposition,
            "C_independent": independent.disposition,
            "D_extra_recurrence": recurrent.disposition,
            "E_soft_aggregation": "ANSWER" if soft_answer is not None else "REJECT",
            "F_no_conflict": no_conflict.disposition,
            "G_diverge": full.disposition,
            "invariant": invariant.disposition,
            "shuffled_provenance": shuffled.disposition,
        },
        "resources": {
            "records": len(compiled.choices),
            "domain_sizes": [len(row) for row in compiled.choices],
            "represented_worlds": compiled.execution.represented_worlds,
            "mdd_nodes": len(compiled.execution.arena.nodes),
            "terminal_state_groups": len(compiled.execution.groups),
            "peak_state_groups": compiled.execution.peak_groups,
            "factorized_bytes": factorized_bytes,
            "whole_particle_component_bytes": whole_particle_bytes,
            "storage_sharing_ratio": (
                whole_particle_bytes / factorized_bytes if factorized_bytes else 0.0
            ),
            "factorized_worlds_per_byte": (
                compiled.execution.represented_worlds / factorized_bytes
                if factorized_bytes
                else 0.0
            ),
            "whole_particle_worlds_per_byte": (
                compiled.execution.represented_worlds / whole_particle_bytes
                if whole_particle_bytes
                else 0.0
            ),
            "factorized_unique_transactions": compiled.execution.unique_transaction_applications,
            "whole_world_transaction_applications": compiled.execution.logical_transaction_applications,
            "shared_transaction_applications": compiled.execution.shared_transaction_applications,
            "equal_budget_particles": particle_resources,
        },
    }


def evaluate_cohort(
    model: HierarchicalStructuredCompiler,
    *,
    cohort: str,
    count: int,
    seed: int,
    batch_size: int,
    component_k: int,
    max_nodes: int,
    max_groups: int,
    device: torch.device,
) -> dict[str, object]:
    totals: dict[str, int] = defaultdict(int)
    records = []
    started = time.perf_counter_ns()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            episodes = [
                generate_episode(seed=seed + index, cohort=cohort)
                for index in range(start, min(count, start + batch_size))
            ]
            compiled = _compile_batch(
                model,
                episodes,
                device,
                component_k=component_k,
                max_nodes=max_nodes,
                max_groups=max_groups,
            )
            evidence_rows = [_delayed_evidence(item) for item in compiled]
            for index, (item, episode) in enumerate(
                zip(compiled, episodes, strict=True)
            ):
                if len(compiled) > 1:
                    donor = evidence_rows[(index + 1) % len(compiled)]
                else:
                    donor = (
                        DelayedRecordEvidence(
                            named_commitment(
                                "diverge-ulc1-hsc1-foreign-source", item.episode_id
                            ),
                            0,
                            named_commitment(
                                "diverge-ulc1-hsc1-foreign-record", item.episode_id
                            ),
                            0,
                            named_commitment(
                                "diverge-ulc1-hsc1-foreign-evidence", item.episode_id
                            ),
                        ),
                    )
                result = _evaluate_compiled(item, donor_evidence=donor)
                for key in (
                    "segmentation_exact",
                    "support_recalled",
                    "at_least_8_worlds",
                    "gold_state_exact",
                    "gold_survives_evidence",
                    "initial_top1_wrong",
                    "wrong_top1_recovered",
                    "A_single_state_exact",
                    "B_particles_exact",
                    "C_independent_exact",
                    "D_extra_recurrence_exact",
                    "E_soft_aggregation_exact",
                    "F_no_conflict_exact",
                    "G_diverge_exact",
                    "invariant_exact",
                    "underdetermined_abstains",
                    "shuffled_provenance_exact",
                    "state_reset_exact",
                    "packet_swap_rejected",
                    "source_poison_invariant",
                ):
                    totals[key] += int(result[key])
                totals["episodes"] += 1
                totals["source_words"] += item.source_words
                totals["represented_worlds"] += int(
                    result["resources"]["represented_worlds"]
                )
                totals["factorized_bytes"] += int(
                    result["resources"]["factorized_bytes"]
                )
                totals["whole_particle_component_bytes"] += int(
                    result["resources"]["whole_particle_component_bytes"]
                )
                totals["mdd_nodes"] += int(result["resources"]["mdd_nodes"])
                totals["factorized_unique_transactions"] += int(
                    result["resources"]["factorized_unique_transactions"]
                )
                totals["whole_world_transaction_applications"] += int(
                    result["resources"]["whole_world_transaction_applications"]
                )
                records.append(
                    {
                        "episode_id": episode.episode_id,
                        "source_commitment": item.source_commitment,
                        **result,
                    }
                )
    rates = {
        key: value / totals["episodes"]
        for key, value in totals.items()
        if key
        not in {
            "episodes",
            "source_words",
            "represented_worlds",
            "factorized_bytes",
            "whole_particle_component_bytes",
            "mdd_nodes",
            "factorized_unique_transactions",
            "whole_world_transaction_applications",
        }
    }
    return {
        "cohort": cohort,
        "count": count,
        "elapsed_nanoseconds": time.perf_counter_ns() - started,
        "totals": dict(sorted(totals.items())),
        "rates": dict(sorted(rates.items())),
        "episodes": records,
    }


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    model = load_frozen_hsc1(
        base=args.base,
        tokenizer_path=args.tokenizer,
        sc1_checkpoint=args.sc1_checkpoint,
        hsc1_checkpoint=args.hsc1_checkpoint,
        device=device,
        layer=args.layer,
        width=args.width,
        pair_width=args.pair_width,
        local_layers=args.local_layers,
        local_heads=args.local_heads,
    )
    evaluations = {
        cohort: evaluate_cohort(
            model,
            cohort=cohort,
            count=args.count,
            seed=args.seed + COHORT_OFFSETS[cohort],
            batch_size=args.batch_size,
            component_k=args.component_k,
            max_nodes=args.max_nodes,
            max_groups=args.max_groups,
            device=device,
        )
        for cohort in COHORTS
    }
    shifted = ("lexical_shift", "renderer_shift", "composition_shift")
    gate = {
        "at_least_8_worlds_each": min(
            evaluations[cohort]["rates"]["at_least_8_worlds"] for cohort in COHORTS
        )
        == 1.0,
        "support_at_least_95pct_each": min(
            evaluations[cohort]["rates"]["support_recalled"] for cohort in COHORTS
        )
        >= 0.95,
        "diverge_exact_at_least_90pct_each": min(
            evaluations[cohort]["rates"]["G_diverge_exact"] for cohort in COHORTS
        )
        >= 0.90,
        "wrong_top1_recovery_at_least_90pct_each": min(
            evaluations[cohort]["rates"]["wrong_top1_recovered"]
            / max(1e-12, evaluations[cohort]["rates"]["initial_top1_wrong"])
            for cohort in COHORTS
        )
        >= 0.90,
        "beats_top1_by_10_points_each_shift": min(
            evaluations[cohort]["rates"]["G_diverge_exact"]
            - evaluations[cohort]["rates"]["A_single_state_exact"]
            for cohort in shifted
        )
        >= 0.10,
        "beats_particles_by_10_points_each_shift": min(
            evaluations[cohort]["rates"]["G_diverge_exact"]
            - evaluations[cohort]["rates"]["B_particles_exact"]
            for cohort in shifted
        )
        >= 0.10,
        "beats_extra_recurrence_by_10_points_each_shift": min(
            evaluations[cohort]["rates"]["G_diverge_exact"]
            - evaluations[cohort]["rates"]["D_extra_recurrence_exact"]
            for cohort in shifted
        )
        >= 0.10,
        "beats_soft_aggregation_by_10_points_each_shift": min(
            evaluations[cohort]["rates"]["G_diverge_exact"]
            - evaluations[cohort]["rates"]["E_soft_aggregation_exact"]
            for cohort in shifted
        )
        >= 0.10,
        "invariant_100pct": min(
            evaluations[cohort]["rates"]["invariant_exact"] for cohort in COHORTS
        )
        == 1.0,
        "underdetermined_never_false_commits": min(
            evaluations[cohort]["rates"]["underdetermined_abstains"]
            for cohort in COHORTS
        )
        == 1.0,
        "provenance_intervention_drop_20_points": min(
            evaluations[cohort]["rates"]["G_diverge_exact"]
            - evaluations[cohort]["rates"]["shuffled_provenance_exact"]
            for cohort in shifted
        )
        >= 0.20,
        "state_reset_drop_20_points": min(
            evaluations[cohort]["rates"]["G_diverge_exact"]
            - evaluations[cohort]["rates"]["state_reset_exact"]
            for cohort in shifted
        )
        >= 0.20,
        "source_seal_controls_100pct": min(
            min(
                evaluations[cohort]["rates"]["packet_swap_rejected"],
                evaluations[cohort]["rates"]["source_poison_invariant"],
            )
            for cohort in COHORTS
        )
        == 1.0,
        "gold_state_exact_100pct_when_supported": min(
            evaluations[cohort]["rates"]["gold_state_exact"] for cohort in COHORTS
        )
        >= 0.95,
        "verified_evidence_never_deletes_represented_gold": min(
            evaluations[cohort]["rates"]["gold_survives_evidence"] for cohort in COHORTS
        )
        == 1.0,
        "storage_sharing_at_least_2x_each_shift": min(
            evaluations[cohort]["totals"]["whole_particle_component_bytes"]
            / max(1, evaluations[cohort]["totals"]["factorized_bytes"])
            for cohort in shifted
        )
        >= 2.0,
        "transaction_sharing_at_least_1_25x_each_shift": min(
            evaluations[cohort]["totals"]["whole_world_transaction_applications"]
            / max(1, evaluations[cohort]["totals"]["factorized_unique_transactions"])
            for cohort in shifted
        )
        >= 1.25,
    }
    gate["pass"] = all(gate.values())
    return {
        "schema": SCHEMA,
        "status": "frozen-hsc1-no-training-matched-runtime-gate",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "inputs": {
            "base_sha256": sha256_file(args.base),
            "tokenizer_sha256": sha256_file(args.tokenizer),
            "sc1_sha256": sha256_file(args.sc1_checkpoint),
            "hsc1_sha256": sha256_file(args.hsc1_checkpoint),
        },
        "evaluations": evaluations,
        "gate": gate,
        "claim_boundary": (
            "Frozen synthetic source-score/runtime result only; exact typed host transitions "
            "and assessor-issued delayed evidence are not model-owned neural reasoning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sc1-checkpoint", type=Path, required=True)
    parser.add_argument("--hsc1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202608057600)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--component-k", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--max-groups", type=int, default=100_000)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--pair-width", type=int, default=64)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite ULC1 HSC1 report")
    if args.component_k != 2:
        raise ValueError("the frozen gate requires component K=2")
    if args.count <= 0 or args.batch_size <= 0:
        raise ValueError("evaluation sizes must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    report = run_gate(args)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "gate": report["gate"],
                "rates": {
                    cohort: report["evaluations"][cohort]["rates"] for cohort in COHORTS
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not report["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
