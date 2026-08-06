#!/usr/bin/env python3
"""CPU reference mechanics for the DIVERGE-SC1 raw-source compiler.

The candidate decoder receives only source tokens plus model-like unary,
boundary, and pair scores. Gold record and option objects are confined to the
board generator and assessor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence


SCHEMA = "shohin-diverge-sc1-cpu-nontriviality-v1"

OTHER = 0
CANDIDATE_CUE = 1
BACKGROUND_CUE = 2
ALIAS_BEGIN = 3
ALIAS_INSIDE = 4
PRIOR_FAVORED = 5
PRIOR_RESERVE = 6
ACTION_ADD = 7
ACTION_SWAP01 = 8
ACTION_SWAP23 = 9
ACTION_SWAP34 = 10
ROLE_COUNT = 11

PROGRAM_ROLES = {
    0: (ACTION_ADD, ACTION_SWAP01),
    1: (ACTION_SWAP01, ACTION_ADD),
    2: (ACTION_SWAP23,),
    3: (ACTION_SWAP34,),
}

MAX_ALIAS_TOKENS = 4
MAX_OPTION_WINDOW = 36
MAX_RECORD_WINDOW = 108
MAX_OPTIONS_PER_ALIAS = 2
# CPU calibration found one 452-token / nine-record episode with more than
# 1,024 legal scored proposals. The production cap is frozen at 4,096 before
# any neural result; overflow remains sticky and fail-closed.
MAX_RECORD_CANDIDATES = 4096


@dataclass(frozen=True, slots=True)
class GoldOption:
    alias_span: tuple[int, int]
    alias_tokens: tuple[str, ...]
    prior_position: int
    prior_class: int
    action_positions: tuple[int, ...]
    program: int


@dataclass(frozen=True, slots=True)
class GoldRecord:
    start: int
    end: int
    cue_position: int
    is_fault_line: bool
    options: tuple[GoldOption, GoldOption]


@dataclass(frozen=True, slots=True)
class RawSourceEpisode:
    episode_id: str
    cohort: str
    tokens: tuple[str, ...]
    records: tuple[GoldRecord, ...]
    decoy_roles: tuple[tuple[int, int], ...]
    decoy_boundaries: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CompilerScores:
    role: tuple[tuple[int, ...], ...]
    boundary: tuple[int, ...]
    pair: tuple[tuple[int, int, int], ...]

    def pair_map(self) -> dict[tuple[int, int], int]:
        output: dict[tuple[int, int], int] = {}
        for left, right, score in self.pair:
            output[left, right] = score
            output[right, left] = score
        return output


@dataclass(frozen=True, slots=True)
class OptionCandidate:
    alias_span: tuple[int, int]
    alias_tokens: tuple[str, ...]
    prior_position: int
    prior_class: int
    action_positions: tuple[int, ...]
    program: int
    start: int
    end: int
    score: int

    @property
    def anchor(self) -> int:
        return self.alias_span[0]


@dataclass(frozen=True, slots=True)
class RecordCandidate:
    start: int
    end: int
    cue_position: int
    is_fault_line: bool
    options: tuple[OptionCandidate, OptionCandidate]
    score: int


@dataclass(frozen=True, slots=True)
class SealedOption:
    occurrence_id: int
    nominal_commitment: str
    source_span: tuple[int, int]
    prior_class: int
    program: int


@dataclass(frozen=True, slots=True)
class SealedRecord:
    record_id: int
    source_span: tuple[int, int]
    is_fault_line: bool
    options: tuple[SealedOption, SealedOption]


@dataclass(frozen=True, slots=True)
class SealedSourcePacket:
    records: tuple[SealedRecord, ...]
    source_commitment: str


@dataclass(frozen=True, slots=True)
class DecodeReceipt:
    records: tuple[RecordCandidate, ...]
    score: int
    candidate_options: int
    candidate_records: int
    overflow: bool


class _TokenBuilder:
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.roles: dict[int, int] = {}
        self.decoys: list[tuple[int, int]] = []

    def emit(
        self,
        words: Iterable[str],
        *,
        first_role: int | None = None,
        continuation_role: int | None = None,
        decoy_role: int | None = None,
    ) -> tuple[int, int]:
        words = tuple(words)
        start = len(self.tokens)
        self.tokens.extend(words)
        end = len(self.tokens)
        if words and first_role is not None:
            self.roles[start] = first_role
        if continuation_role is not None:
            for position in range(start + 1, end):
                self.roles[position] = continuation_role
        if words and decoy_role is not None:
            self.decoys.append((start, decoy_role))
        return start, end


def _digest(domain: str, payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256()
    for part in (domain.encode("ascii"), body):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _alias_tokens(
    rng: random.Random,
    cohort: str,
    used: list[tuple[str, ...]],
    *,
    permit_repeat: bool,
) -> tuple[str, ...]:
    if permit_repeat and used and rng.random() < 0.30:
        return rng.choice(used)
    stems = {
        "train": ("amber", "cedar", "flint", "hazel", "indigo", "maple"),
        "lexical_shift": ("birch", "clover", "dahlia", "ginger", "juniper", "lotus"),
        "renderer_shift": ("acorn", "heather", "nectar", "poppy", "spruce", "violet"),
        "composition_shift": ("agate", "basil", "cinder", "elm", "iris", "quartz"),
    }[cohort]
    length = rng.randrange(1, MAX_ALIAS_TOKENS + 1)
    value = tuple(
        [rng.choice(stems)]
        + [f"{rng.choice(stems)}{rng.randrange(10, 99)}" for _ in range(length - 1)]
    )
    used.append(value)
    return value


def _emit_action(builder: _TokenBuilder, role: int) -> int:
    words = {
        ACTION_ADD: ("increase", "slot", "zero", "by", "three"),
        ACTION_SWAP01: ("exchange", "slot", "zero", "with", "slot", "one"),
        ACTION_SWAP23: ("transpose", "slot", "two", "and", "slot", "three"),
        ACTION_SWAP34: ("permute", "slot", "three", "with", "slot", "four"),
    }[role]
    position, _ = builder.emit(words, first_role=role)
    return position


def _emit_option(
    builder: _TokenBuilder,
    *,
    alias: tuple[str, ...],
    program: int,
    prior: int,
    renderer: int,
) -> GoldOption:
    action_positions: list[int] = []
    alias_span: tuple[int, int] | None = None
    prior_position = -1

    def alias_part() -> None:
        nonlocal alias_span
        builder.emit(("key",))
        alias_span = builder.emit(
            alias,
            first_role=ALIAS_BEGIN,
            continuation_role=ALIAS_INSIDE,
        )

    def prior_part() -> None:
        nonlocal prior_position
        word = "favored" if prior == 0 else "reserve"
        prior_position, _ = builder.emit(
            (word,), first_role=PRIOR_FAVORED + prior
        )

    def action_part() -> None:
        for index, role in enumerate(PROGRAM_ROLES[program]):
            if index:
                builder.emit(("then",))
            action_positions.append(_emit_action(builder, role))

    orders = (
        (prior_part, alias_part, action_part),
        (alias_part, action_part, prior_part),
        (action_part, prior_part, alias_part),
        (prior_part, action_part, alias_part),
    )
    builder.emit(("option",))
    for index, component in enumerate(orders[renderer % len(orders)]):
        if index:
            builder.emit(("with",))
        component()
    if alias_span is None or prior_position < 0:
        raise AssertionError("option construction omitted a field")
    return GoldOption(
        alias_span,
        alias,
        prior_position,
        prior,
        tuple(action_positions),
        program,
    )


def _shift_option(option: GoldOption, offset: int) -> GoldOption:
    return GoldOption(
        (option.alias_span[0] + offset, option.alias_span[1] + offset),
        option.alias_tokens,
        option.prior_position + offset,
        option.prior_class,
        tuple(position + offset for position in option.action_positions),
        option.program,
    )


def _build_record(
    rng: random.Random,
    *,
    cohort: str,
    is_fault_line: bool,
    renderer: int,
    used_aliases: list[tuple[str, ...]],
) -> tuple[tuple[str, ...], GoldRecord, tuple[tuple[int, int], ...], int]:
    builder = _TokenBuilder()
    if is_fault_line:
        cue_words = {
            "train": ("candidate", "alternatives"),
            "lexical_shift": ("viable", "possibilities"),
            "renderer_shift": ("competing", "interpretations"),
            "composition_shift": ("live", "hypotheses"),
        }[cohort]
        cue_position, _ = builder.emit(cue_words, first_role=CANDIDATE_CUE)
        opposite = BACKGROUND_CUE
    else:
        cue_words = {
            "train": ("background", "archive"),
            "lexical_shift": ("descriptive", "example"),
            "renderer_shift": ("inactive", "catalog"),
            "composition_shift": ("nonoperative", "note"),
        }[cohort]
        cue_position, _ = builder.emit(cue_words, first_role=BACKGROUND_CUE)
        opposite = CANDIDATE_CUE
    builder.emit(("for", "this", "episode"))
    builder.emit(("aside",), decoy_role=opposite)

    aliases: list[tuple[str, ...]] = []
    while len(aliases) < 2:
        value = _alias_tokens(
            rng,
            cohort,
            used_aliases,
            permit_repeat=True,
        )
        if value not in aliases:
            aliases.append(value)
    programs = [rng.randrange(4), rng.randrange(4)]
    priors = [rng.randrange(2), rng.randrange(2)]
    local_options = [
        _emit_option(
            builder,
            alias=aliases[index],
            program=programs[index],
            prior=priors[index],
            renderer=(renderer + index) % 4,
        )
        for index in range(2)
    ]

    decoy_prior = PRIOR_RESERVE if priors[0] == 0 else PRIOR_FAVORED
    builder.emit(("glossary",))
    builder.emit(("reserve" if decoy_prior == PRIOR_RESERVE else "favored",), decoy_role=decoy_prior)
    decoy_action = rng.choice(tuple(PROGRAM_ROLES.values()))[0]
    decoy_word = {
        ACTION_ADD: "increase",
        ACTION_SWAP01: "exchange",
        ACTION_SWAP23: "transpose",
        ACTION_SWAP34: "permute",
    }[decoy_action]
    builder.emit((decoy_word,), decoy_role=decoy_action)
    builder.emit(("are", "words", "only"))

    ordered = sorted(local_options, key=lambda option: option.alias_span[0])
    record = GoldRecord(
        0,
        len(builder.tokens),
        cue_position,
        is_fault_line,
        (ordered[0], ordered[1]),
    )
    internal_decoy_boundary = max(
        ordered[0].alias_span[1],
        min(ordered[1].alias_span[0], len(builder.tokens) - 1),
    )
    return (
        tuple(builder.tokens),
        record,
        tuple(builder.decoys),
        internal_decoy_boundary,
    )


def generate_episode(*, seed: int, cohort: str) -> RawSourceEpisode:
    if cohort not in {"train", "lexical_shift", "renderer_shift", "composition_shift"}:
        raise ValueError("unknown DIVERGE-SC1 cohort")
    rng = random.Random(seed)
    used_aliases: list[tuple[str, ...]] = []
    record_parts = []
    candidate_count = rng.randrange(2, 7)
    background_count = rng.randrange(1, 4)
    for index in range(candidate_count + background_count):
        is_fault = index < candidate_count
        record_parts.append(
            _build_record(
                rng,
                cohort=cohort,
                is_fault_line=is_fault,
                renderer=(index + seed) % 4,
                used_aliases=used_aliases,
            )
        )
    rng.shuffle(record_parts)

    tokens: list[str] = []
    records: list[GoldRecord] = []
    decoys: list[tuple[int, int]] = []
    decoy_boundaries: list[int] = []
    connectors = (
        ("meanwhile", ","),
        ("in", "another", "entry", ","),
        ("separately", ","),
        ("the", "ledger", "continues", ":"),
    )
    for index, (part_tokens, record, part_decoys, local_boundary) in enumerate(record_parts):
        if index:
            tokens.extend(connectors[(index + seed) % len(connectors)])
        offset = len(tokens)
        tokens.extend(part_tokens)
        shifted_options = tuple(_shift_option(option, offset) for option in record.options)
        records.append(
            GoldRecord(
                offset,
                offset + len(part_tokens),
                record.cue_position + offset,
                record.is_fault_line,
                (shifted_options[0], shifted_options[1]),
            )
        )
        decoys.extend((position + offset, role) for position, role in part_decoys)
        decoy_boundaries.append(local_boundary + offset)
    episode_id = _digest(
        "diverge-sc1-episode",
        {"seed": seed, "cohort": cohort, "tokens": tokens},
    )[:24]
    return RawSourceEpisode(
        episode_id,
        cohort,
        tuple(tokens),
        tuple(records),
        tuple(decoys),
        tuple(decoy_boundaries),
    )


def calibrated_scores(episode: RawSourceEpisode, *, seed: int) -> CompilerScores:
    """Create transparent gold-plus-decoy scores for the CPU mechanics gate."""

    rng = random.Random(seed)
    role = [[-8] * ROLE_COUNT for _ in episode.tokens]
    for row in role:
        row[OTHER] = 4
    boundary = [-8] * (len(episode.tokens) + 1)
    edges: dict[tuple[int, int], int] = {}

    def set_role(position: int, value: int, score: int = 9) -> None:
        role[position][OTHER] = -2
        role[position][value] = score

    def set_pair(left: int, right: int, value: int) -> None:
        key = tuple(sorted((left, right)))
        edges[key] = max(edges.get(key, -10_000), value)

    for record in episode.records:
        set_role(
            record.cue_position,
            CANDIDATE_CUE if record.is_fault_line else BACKGROUND_CUE,
        )
        boundary[record.start] = 10
        boundary[record.end] = 10
        for option in record.options:
            set_role(option.alias_span[0], ALIAS_BEGIN)
            for position in range(option.alias_span[0] + 1, option.alias_span[1]):
                set_role(position, ALIAS_INSIDE)
            set_role(option.prior_position, PRIOR_FAVORED + option.prior_class)
            for position, action_role in zip(
                option.action_positions,
                PROGRAM_ROLES[option.program],
                strict=True,
            ):
                set_role(position, action_role)
            for position in (option.prior_position, *option.action_positions):
                set_pair(option.alias_span[0], position, 12)
            set_pair(record.cue_position, option.alias_span[0], 10)
        set_pair(
            record.options[0].alias_span[0],
            record.options[1].alias_span[0],
            6,
        )

    for position, decoy_role in episode.decoy_roles:
        role[position][OTHER] = 0
        role[position][decoy_role] = 13
        # A few decoys receive plausible but incomplete local associations.
        nearby = [
            option.alias_span[0]
            for record in episode.records
            for option in record.options
            if abs(option.alias_span[0] - position) <= MAX_OPTION_WINDOW
        ]
        if nearby:
            set_pair(rng.choice(nearby), position, 2)
    for gap in episode.decoy_boundaries:
        if boundary[gap] < 10:
            boundary[gap] = 13

    # Sparse low-scoring cross-record links force the parser to score complete
    # objects instead of treating absence of an edge as a hidden hard oracle.
    anchors = [option.alias_span[0] for record in episode.records for option in record.options]
    fields = [
        position
        for record in episode.records
        for option in record.options
        for position in (option.prior_position, *option.action_positions)
    ]
    for _ in range(min(24, len(anchors) * 2)):
        set_pair(rng.choice(anchors), rng.choice(fields), -2)

    return CompilerScores(
        tuple(tuple(row) for row in role),
        tuple(boundary),
        tuple((left, right, value) for (left, right), value in sorted(edges.items())),
    )


def _margin(scores: CompilerScores, position: int, role: int) -> int:
    return scores.role[position][role] - scores.role[position][OTHER]


def enumerate_aliases(
    tokens: Sequence[str], scores: CompilerScores
) -> tuple[tuple[int, int, tuple[str, ...], int], ...]:
    output = []
    for start in range(len(tokens)):
        begin = _margin(scores, start, ALIAS_BEGIN)
        if begin <= 0:
            continue
        output.append((start, start + 1, (tokens[start],), begin))
        score = begin
        for end in range(start + 1, min(len(tokens), start + MAX_ALIAS_TOKENS)):
            inside = _margin(scores, end, ALIAS_INSIDE)
            if inside <= 0:
                break
            score += inside
            output.append((start, end + 1, tuple(tokens[start : end + 1]), score))
    return tuple(output)


def enumerate_options(
    tokens: Sequence[str], scores: CompilerScores
) -> tuple[OptionCandidate, ...]:
    pairs = scores.pair_map()
    aliases = enumerate_aliases(tokens, scores)
    prior_rows = [
        (position, prior, _margin(scores, position, PRIOR_FAVORED + prior))
        for position in range(len(tokens))
        for prior in range(2)
        if _margin(scores, position, PRIOR_FAVORED + prior) > 0
    ]
    action_rows: dict[int, list[tuple[int, int]]] = {}
    for action_role in range(ACTION_ADD, ACTION_SWAP34 + 1):
        action_rows[action_role] = [
            (position, _margin(scores, position, action_role))
            for position in range(len(tokens))
            if _margin(scores, position, action_role) > 0
        ]

    output: list[OptionCandidate] = []
    for alias_start, alias_end, alias_tokens, alias_score in aliases:
        local_priors = sorted(
            (
                (position, prior, unary, unary + pairs.get((alias_start, position), -5))
                for position, prior, unary in prior_rows
                if abs(position - alias_start) <= MAX_OPTION_WINDOW
            ),
            key=lambda row: (-row[3], row[:2]),
        )[:8]
        for program, required_roles in PROGRAM_ROLES.items():
            action_choices: list[tuple[tuple[int, int], ...]] = []
            if len(required_roles) == 1:
                role = required_roles[0]
                action_choices = [((position, unary),) for position, unary in action_rows[role]]
            else:
                first_role, second_role = required_roles
                action_choices = [
                    ((left, left_score), (right, right_score))
                    for left, left_score in action_rows[first_role]
                    for right, right_score in action_rows[second_role]
                    if left < right
                ]
            ranked_actions = sorted(
                (
                    (
                        action,
                        sum(unary + pairs.get((alias_start, position), -5) for position, unary in action),
                    )
                    for action in action_choices
                    if action
                    and max(abs(position - alias_start) for position, _ in action)
                    <= MAX_OPTION_WINDOW
                ),
                key=lambda row: (-row[1], tuple(item[0] for item in row[0])),
            )[:16]
            for prior_position, prior, prior_unary, prior_total in local_priors:
                for action, action_total in ranked_actions:
                    action_positions = tuple(position for position, _ in action)
                    occupied = set(range(alias_start, alias_end))
                    if prior_position in occupied or any(position in occupied for position in action_positions):
                        continue
                    if prior_position in action_positions or len(set(action_positions)) != len(action_positions):
                        continue
                    start = min(alias_start, prior_position, *action_positions)
                    end = max(alias_end, prior_position + 1, *(position + 1 for position in action_positions))
                    if end - start > MAX_OPTION_WINDOW:
                        continue
                    locality = end - start
                    score = alias_score + prior_total + action_total - locality
                    output.append(
                        OptionCandidate(
                            (alias_start, alias_end),
                            alias_tokens,
                            prior_position,
                            prior,
                            action_positions,
                            program,
                            start,
                            end,
                            score,
                        )
                    )
    by_alias: dict[tuple[int, int], list[OptionCandidate]] = {}
    for candidate in output:
        by_alias.setdefault(candidate.alias_span, []).append(candidate)
    kept = []
    for candidates in by_alias.values():
        kept.extend(
            sorted(
                candidates,
                key=lambda item: (-item.score, item.program, item.prior_class, item.action_positions),
            )[:MAX_OPTIONS_PER_ALIAS]
        )
    return tuple(sorted(kept, key=lambda item: (item.start, item.end, -item.score)))


def enumerate_records(
    tokens: Sequence[str], scores: CompilerScores
) -> tuple[RecordCandidate, ...]:
    options = enumerate_options(tokens, scores)
    pair = scores.pair_map()
    cues = [
        (position, kind == CANDIDATE_CUE, _margin(scores, position, kind))
        for position in range(len(tokens))
        for kind in (CANDIDATE_CUE, BACKGROUND_CUE)
        if _margin(scores, position, kind) > 0
    ]
    boundary_positions = [
        position for position, value in enumerate(scores.boundary) if value > 0
    ]
    output: list[RecordCandidate] = []
    for left_index, left in enumerate(options):
        for right in options[left_index + 1 :]:
            if left.end > right.start or right.end - left.start > MAX_RECORD_WINDOW:
                continue
            if left.alias_span == right.alias_span:
                continue
            alias_link = pair.get((left.anchor, right.anchor), -5)
            for cue_position, is_fault, cue_score in cues:
                semantic_start = min(cue_position, left.start)
                semantic_end = max(cue_position + 1, right.end)
                if semantic_end - semantic_start > MAX_RECORD_WINDOW:
                    continue
                cue_link = pair.get((cue_position, left.anchor), -5) + pair.get(
                    (cue_position, right.anchor), -5
                )
                # Proposal pruning is itself score-owned: a complete record
                # must have positive option-option support and strong cue-to-
                # option support.  No gold boundary or record object enters.
                if alias_link < 0 or cue_link < 12:
                    continue
                left_boundaries = sorted(
                    (
                        gap
                        for gap in boundary_positions
                        if gap <= semantic_start and semantic_start - gap <= 12
                    ),
                    key=lambda gap: (-scores.boundary[gap], semantic_start - gap, gap),
                )[:2]
                right_boundaries = sorted(
                    (
                        gap
                        for gap in boundary_positions
                        if gap >= semantic_end and gap - semantic_end <= 20
                    ),
                    key=lambda gap: (-scores.boundary[gap], gap - semantic_end, gap),
                )[:2]
                for start in left_boundaries:
                    for end in right_boundaries:
                        if not start < end or end - start > MAX_RECORD_WINDOW:
                            continue
                        internal = sum(
                            max(0, scores.boundary[gap])
                            for gap in boundary_positions
                            if start < gap < end
                        )
                        score = (
                            left.score
                            + right.score
                            + cue_score
                            + cue_link
                            + alias_link
                            + scores.boundary[start]
                            + scores.boundary[end]
                            - 3 * internal
                            + 24
                        )
                        ordered = tuple(sorted((left, right), key=lambda item: item.alias_span))
                        output.append(
                            RecordCandidate(
                                start,
                                end,
                                cue_position,
                                is_fault,
                                (ordered[0], ordered[1]),
                                score,
                            )
                        )
    unique: dict[tuple[object, ...], RecordCandidate] = {}
    for candidate in output:
        signature = _record_signature(candidate)
        incumbent = unique.get(signature)
        if incumbent is None or candidate.score > incumbent.score:
            unique[signature] = candidate
    output = sorted(
        unique.values(),
        key=lambda item: (
            -item.score,
            item.start,
            item.end,
            item.cue_position,
            tuple(option.alias_span for option in item.options),
        ),
    )[:MAX_RECORD_CANDIDATES]
    return tuple(sorted(output, key=lambda item: (item.start, item.end, -item.score)))


def _record_signature(record: RecordCandidate) -> tuple[object, ...]:
    return (
        record.start,
        record.end,
        record.cue_position,
        record.is_fault_line,
        tuple(
            (
                option.alias_span,
                option.alias_tokens,
                option.prior_position,
                option.prior_class,
                option.action_positions,
                option.program,
            )
            for option in record.options
        ),
    )


def _solution_key(records: Sequence[RecordCandidate], score: int) -> tuple[object, ...]:
    return (-score, tuple(_record_signature(record) for record in records))


def decode_joint(tokens: Sequence[str], scores: CompilerScores) -> DecodeReceipt:
    candidates = enumerate_records(tokens, scores)
    if len(candidates) >= MAX_RECORD_CANDIDATES:
        return DecodeReceipt((), 0, len(enumerate_options(tokens, scores)), len(candidates), True)
    by_start: dict[int, list[RecordCandidate]] = {}
    for candidate in candidates:
        if candidate.score > 0:
            by_start.setdefault(candidate.start, []).append(candidate)
    memo: dict[int, tuple[int, tuple[RecordCandidate, ...]]] = {}

    def solve(cursor: int) -> tuple[int, tuple[RecordCandidate, ...]]:
        if cursor >= len(tokens):
            return 0, ()
        if cursor in memo:
            return memo[cursor]
        choices = [solve(cursor + 1)]
        for candidate in by_start.get(cursor, ()):
            suffix_score, suffix = solve(candidate.end)
            choices.append((candidate.score + suffix_score, (candidate, *suffix)))
        result = min(choices, key=lambda item: _solution_key(item[1], item[0]))
        memo[cursor] = result
        return result

    score, records = solve(0)
    return DecodeReceipt(
        tuple(records),
        score,
        len(enumerate_options(tokens, scores)),
        len(candidates),
        False,
    )


def decode_reference(tokens: Sequence[str], scores: CompilerScores) -> DecodeReceipt:
    """Independent weighted-interval reference over all legal candidates."""

    candidates = tuple(candidate for candidate in enumerate_records(tokens, scores) if candidate.score > 0)
    if len(candidates) >= MAX_RECORD_CANDIDATES:
        return DecodeReceipt((), 0, len(enumerate_options(tokens, scores)), len(candidates), True)
    ordered = tuple(sorted(candidates, key=lambda item: (item.end, item.start, -item.score)))
    compatible = []
    for index, candidate in enumerate(ordered):
        predecessor = -1
        for earlier in range(index - 1, -1, -1):
            if ordered[earlier].end <= candidate.start:
                predecessor = earlier
                break
        compatible.append(predecessor)
    table: list[tuple[int, tuple[RecordCandidate, ...]]] = [(0, ())]
    for index, candidate in enumerate(ordered):
        skip = table[index]
        prior_score, prior_records = table[compatible[index] + 1]
        include_records = tuple(
            sorted((*prior_records, candidate), key=lambda item: (item.start, item.end))
        )
        include = (prior_score + candidate.score, include_records)
        table.append(min((skip, include), key=lambda item: _solution_key(item[1], item[0])))
    score, records = table[-1]
    return DecodeReceipt(
        records,
        score,
        len(enumerate_options(tokens, scores)),
        len(candidates),
        False,
    )


def decode_independent(tokens: Sequence[str], scores: CompilerScores) -> DecodeReceipt:
    """Greedy local baseline using the same candidates and raw boundary scores."""

    options = enumerate_options(tokens, scores)
    boundaries = sorted(
        position for position, value in enumerate(scores.boundary) if value > 0
    )
    records: list[RecordCandidate] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if start == end:
            continue
        local_options = [option for option in options if start <= option.start and option.end <= end]
        local_options.sort(key=lambda item: (-item.score, item.alias_span))
        selected: list[OptionCandidate] = []
        for option in local_options:
            if all(option.end <= old.start or old.end <= option.start for old in selected):
                selected.append(option)
            if len(selected) == 2:
                break
        cues = [
            (scores.role[position][kind] - scores.role[position][OTHER], position, kind)
            for position in range(start, min(end, len(tokens)))
            for kind in (CANDIDATE_CUE, BACKGROUND_CUE)
        ]
        if len(selected) != 2 or not cues:
            continue
        cue_score, cue_position, kind = max(cues)
        if cue_score <= 0:
            continue
        selected.sort(key=lambda item: item.alias_span)
        records.append(
            RecordCandidate(
                start,
                end,
                cue_position,
                kind == CANDIDATE_CUE,
                (selected[0], selected[1]),
                selected[0].score + selected[1].score + cue_score,
            )
        )
    records.sort(key=lambda item: (item.start, item.end))
    return DecodeReceipt(
        tuple(records),
        sum(record.score for record in records),
        len(options),
        0,
        False,
    )


def seal_source(tokens: Sequence[str], receipt: DecodeReceipt) -> SealedSourcePacket:
    if receipt.overflow:
        return SealedSourcePacket((), "overflow")
    records = []
    occurrence_id = 0
    for record_id, record in enumerate(receipt.records):
        options = []
        for option in record.options:
            surface = " ".join(option.alias_tokens).casefold()
            options.append(
                SealedOption(
                    occurrence_id,
                    _digest("diverge-sc1-nominal", surface),
                    option.alias_span,
                    option.prior_class,
                    option.program,
                )
            )
            occurrence_id += 1
        records.append(
            SealedRecord(
                record_id,
                (record.start, record.end),
                record.is_fault_line,
                (options[0], options[1]),
            )
        )
    return SealedSourcePacket(
        tuple(records),
        _digest("diverge-sc1-source", tuple(tokens)),
    )


def _semantic_signature(records: Sequence[RecordCandidate]) -> tuple[object, ...]:
    return tuple(_record_signature(record) for record in records)


def _gold_signature(episode: RawSourceEpisode) -> tuple[object, ...]:
    return tuple(
        (
            record.start,
            record.end,
            record.cue_position,
            record.is_fault_line,
            tuple(
                (
                    option.alias_span,
                    option.alias_tokens,
                    option.prior_position,
                    option.prior_class,
                    option.action_positions,
                    option.program,
                )
                for option in record.options
            ),
        )
        for record in episode.records
    )


def exact(episode: RawSourceEpisode, receipt: DecodeReceipt) -> bool:
    return not receipt.overflow and _semantic_signature(receipt.records) == _gold_signature(episode)


def _zero_pairs(scores: CompilerScores) -> CompilerScores:
    return replace(scores, pair=tuple((left, right, 0) for left, right, _ in scores.pair))


def _shuffle_boundaries(scores: CompilerScores, *, seed: int) -> CompilerScores:
    rng = random.Random(seed)
    values = list(scores.boundary)
    rng.shuffle(values)
    return replace(scores, boundary=tuple(values))


def alpha_rename_episode(episode: RawSourceEpisode) -> RawSourceEpisode:
    """Consistently rename nominal bytes without changing physical addresses."""

    aliases = []
    for record in episode.records:
        for option in record.options:
            if option.alias_tokens not in aliases:
                aliases.append(option.alias_tokens)
    mapping = {
        alias: tuple(f"renamed{index}_{part}" for part in range(len(alias)))
        for index, alias in enumerate(aliases)
    }
    tokens = list(episode.tokens)
    records = []
    for record in episode.records:
        options = []
        for option in record.options:
            renamed = mapping[option.alias_tokens]
            tokens[option.alias_span[0] : option.alias_span[1]] = renamed
            options.append(replace(option, alias_tokens=renamed))
        records.append(replace(record, options=(options[0], options[1])))
    return replace(
        episode,
        episode_id=_digest("diverge-sc1-alpha", episode.episode_id),
        tokens=tuple(tokens),
        records=tuple(records),
    )


def _has_local_rank_trap(episode: RawSourceEpisode, scores: CompilerScores) -> bool:
    for record in episode.records:
        positions = range(record.start, record.end)
        for option in record.options:
            fields = [(option.prior_position, PRIOR_FAVORED + option.prior_class)]
            fields.extend(zip(option.action_positions, PROGRAM_ROLES[option.program], strict=True))
            for gold_position, role in fields:
                gold = _margin(scores, gold_position, role)
                if any(
                    position != gold_position and _margin(scores, position, role) > gold
                    for position in positions
                ):
                    return True
    return False


def _fused_occurrence_exact(packet: SealedSourcePacket) -> bool:
    seen: dict[str, int] = {}
    for record in packet.records:
        for option in record.options:
            prior = seen.setdefault(option.nominal_commitment, option.occurrence_id)
            if prior != option.occurrence_id:
                return False
    return True


def run_gate(*, count: int, seed: int) -> dict[str, object]:
    cohorts = ("train", "lexical_shift", "renderer_shift", "composition_shift")
    totals = {
        "episodes": 0,
        "joint_exact": 0,
        "reference_parity": 0,
        "independent_exact": 0,
        "no_pair_exact": 0,
        "shuffled_boundary_exact": 0,
        "fused_occurrence_exact": 0,
        "local_rank_trap": 0,
        "overflow": 0,
        "post_seal_poison_invariant": 0,
        "alpha_rename_exact": 0,
    }
    by_cohort: dict[str, dict[str, int]] = {
        cohort: {key: 0 for key in totals} for cohort in cohorts
    }
    candidate_options = 0
    candidate_records = 0
    for index in range(count):
        cohort = cohorts[index % len(cohorts)]
        episode = generate_episode(seed=seed + index, cohort=cohort)
        scores = calibrated_scores(episode, seed=seed * 17 + index)
        joint = decode_joint(episode.tokens, scores)
        reference = decode_reference(episode.tokens, scores)
        independent = decode_independent(episode.tokens, scores)
        no_pair = decode_joint(episode.tokens, _zero_pairs(scores))
        shuffled = decode_joint(
            episode.tokens,
            _shuffle_boundaries(scores, seed=seed * 31 + index),
        )
        packet = seal_source(episode.tokens, joint)
        renamed = alpha_rename_episode(episode)
        renamed_joint = decode_joint(renamed.tokens, scores)
        poisoned = tuple("poison" for _ in episode.tokens)
        poison_invariant = seal_source(poisoned, joint).records == packet.records
        values = {
            "episodes": 1,
            "joint_exact": int(exact(episode, joint)),
            "reference_parity": int(
                joint.score == reference.score
                and _semantic_signature(joint.records) == _semantic_signature(reference.records)
            ),
            "independent_exact": int(exact(episode, independent)),
            "no_pair_exact": int(exact(episode, no_pair)),
            "shuffled_boundary_exact": int(exact(episode, shuffled)),
            "fused_occurrence_exact": int(
                exact(episode, joint) and _fused_occurrence_exact(packet)
            ),
            "local_rank_trap": int(_has_local_rank_trap(episode, scores)),
            "overflow": int(joint.overflow),
            "post_seal_poison_invariant": int(poison_invariant),
            "alpha_rename_exact": int(exact(renamed, renamed_joint)),
        }
        for key, value in values.items():
            totals[key] += value
            by_cohort[cohort][key] += value
        candidate_options += joint.candidate_options
        candidate_records += joint.candidate_records

    rates = {
        key: value / totals["episodes"]
        for key, value in totals.items()
        if key != "episodes"
    }
    cohort_rates = {
        cohort: {
            key: values[key] / values["episodes"]
            for key in values
            if key != "episodes"
        }
        for cohort, values in by_cohort.items()
    }
    gates = {
        "reference_parity": rates["reference_parity"] == 1.0,
        "joint_exact": rates["joint_exact"] == 1.0,
        "independent_advantage": (
            rates["joint_exact"] - rates["independent_exact"] >= 0.25
        ),
        "local_rank_traps": rates["local_rank_trap"] >= 0.95,
        "boundary_causal_drop": (
            rates["joint_exact"] - rates["shuffled_boundary_exact"] >= 0.20
        ),
        "occurrence_nominal_separation": (
            rates["joint_exact"] - rates["fused_occurrence_exact"] >= 0.20
        ),
        "post_seal_poison": rates["post_seal_poison_invariant"] == 1.0,
        "alpha_rename": rates["alpha_rename_exact"] == 1.0,
        "no_overflow": rates["overflow"] == 0.0,
    }
    return {
        "schema": SCHEMA,
        "seed": seed,
        "count": count,
        "totals": totals,
        "rates": rates,
        "cohort_rates": cohort_rates,
        "mean_candidate_options": candidate_options / count,
        "mean_candidate_records": candidate_records / count,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=202608056100)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.count <= 0:
        raise ValueError("count must be positive")
    report = run_gate(count=arguments.count, seed=arguments.seed)
    if arguments.output is not None:
        _atomic_json(arguments.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
