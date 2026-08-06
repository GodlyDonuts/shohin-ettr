#!/usr/bin/env python3
"""Exact CPU mechanics for the DIVERGE-WRA1 whole-record compiler.

The runtime receives only source words, model-owned boundary scores, and two
complete slot score bundles per predicted segment. Gold objects are used only
by calibration and assessment helpers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Sequence

from diverge_sc1_source_compiler import (
    DecodeReceipt,
    OptionCandidate,
    PROGRAM_ROLES,
    RawSourceEpisode,
    RecordCandidate,
    SealedSourcePacket,
    generate_episode,
    seal_source,
)

SCHEMA = "shohin-diverge-wra1-whole-record-cpu-v1"
MAX_RECORDS = 9
MAX_RECORD_WIDTH = 108
MAX_ALIAS_LENGTH = 4
HALT = -1


@dataclass(frozen=True, slots=True)
class SlotScores:
    alias_start: tuple[float, ...]
    alias_length: tuple[float, ...]
    prior_class: tuple[float, ...]
    program_class: tuple[float, ...]
    prior_pointer: tuple[float, ...]
    action_1_pointer: tuple[float, ...]
    action_2_pointer_or_halt: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SegmentScores:
    start: int
    end: int
    record_kind: tuple[float, float]
    cue_pointer: tuple[float, ...]
    slots: tuple[SlotScores, SlotScores]


@dataclass(frozen=True, slots=True)
class WholeRecordScores:
    boundary: tuple[float, ...]
    segments: tuple[SegmentScores, ...]


@dataclass(frozen=True, slots=True)
class WholeRecordReceipt:
    records: tuple[RecordCandidate, ...]
    score: float
    option_objects: int
    record_objects: int
    failed: bool
    failure_reason: str | None
    overflow: bool = False

    def as_sc1(self) -> DecodeReceipt:
        return DecodeReceipt(
            self.records,
            int(round(self.score)),
            self.option_objects,
            self.record_objects,
            self.overflow,
        )


def _argmax(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("cannot select from empty scores")
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _peaked(size: int, index: int, *, high: float = 12.0) -> tuple[float, ...]:
    if not 0 <= index < size:
        raise ValueError("calibrated target is outside score domain")
    return tuple(high if position == index else -high for position in range(size))


def detect_segments(
    boundary: Sequence[float], token_count: int
) -> tuple[tuple[tuple[int, int], ...], str | None, bool]:
    if len(boundary) != token_count + 1:
        return (), "boundary-width", False
    positive = tuple(index for index, value in enumerate(boundary) if value > 0)
    if len(positive) % 2:
        return (), "odd-boundary-count", False
    if len(positive) // 2 > MAX_RECORDS:
        return (), "record-overflow", True
    segments = []
    prior_end = -1
    for offset in range(0, len(positive), 2):
        start, end = positive[offset : offset + 2]
        if not 0 <= start < end <= token_count:
            return (), "invalid-segment", False
        if start < prior_end:
            return (), "overlapping-segments", False
        if end - start > MAX_RECORD_WIDTH:
            return (), "segment-overflow", True
        segments.append((start, end))
        prior_end = end
    return tuple(segments), None, False


def _validate_option(
    tokens: Sequence[str], segment: tuple[int, int], scores: SlotScores
) -> tuple[OptionCandidate | None, str | None, float]:
    start, end = segment
    width = end - start
    pointer_fields = (
        scores.alias_start,
        scores.prior_pointer,
        scores.action_1_pointer,
    )
    if any(len(field) != width for field in pointer_fields):
        return None, "pointer-width", 0.0
    if (
        len(scores.alias_length) != MAX_ALIAS_LENGTH
        or len(scores.prior_class) != 2
        or len(scores.program_class) != len(PROGRAM_ROLES)
        or len(scores.action_2_pointer_or_halt) != width + 1
    ):
        return None, "class-width", 0.0

    alias_local = _argmax(scores.alias_start)
    alias_length = _argmax(scores.alias_length) + 1
    alias_start = start + alias_local
    alias_end = alias_start + alias_length
    prior_class = _argmax(scores.prior_class)
    program = _argmax(scores.program_class)
    prior_position = start + _argmax(scores.prior_pointer)
    action_1 = start + _argmax(scores.action_1_pointer)
    action_2_local = _argmax(scores.action_2_pointer_or_halt)
    action_2 = HALT if action_2_local == width else start + action_2_local

    if alias_end > end:
        return None, "alias-outside-segment", 0.0
    required_actions = len(PROGRAM_ROLES[program])
    if required_actions == 1 and action_2 != HALT:
        return None, "unexpected-action-2", 0.0
    if required_actions == 2 and action_2 == HALT:
        return None, "missing-action-2", 0.0
    actions = (action_1,) if action_2 == HALT else (action_1, action_2)
    if len(actions) == 2 and actions[0] >= actions[1]:
        return None, "unordered-actions", 0.0
    occupied = set(range(alias_start, alias_end))
    witnesses = (prior_position, *actions)
    if len(set(witnesses)) != len(witnesses) or any(
        value in occupied for value in witnesses
    ):
        return None, "overlapping-option-fields", 0.0

    selected = (
        scores.alias_start[alias_local]
        + scores.alias_length[alias_length - 1]
        + scores.prior_class[prior_class]
        + scores.program_class[program]
        + scores.prior_pointer[prior_position - start]
        + scores.action_1_pointer[action_1 - start]
        + scores.action_2_pointer_or_halt[action_2_local]
    )
    option_start = min(alias_start, prior_position, *actions)
    option_end = max(alias_end, prior_position + 1, *(value + 1 for value in actions))
    return (
        OptionCandidate(
            (alias_start, alias_end),
            tuple(tokens[alias_start:alias_end]),
            prior_position,
            prior_class,
            tuple(actions),
            program,
            option_start,
            option_end,
            int(round(selected)),
        ),
        None,
        selected,
    )


def _validate_record(
    tokens: Sequence[str], segment: tuple[int, int], scores: SegmentScores
) -> tuple[RecordCandidate | None, str | None, float]:
    start, end = segment
    width = end - start
    if (scores.start, scores.end) != segment:
        return None, "segment-score-mismatch", 0.0
    if len(scores.record_kind) != 2 or len(scores.cue_pointer) != width:
        return None, "record-head-width", 0.0
    cue_position = start + _argmax(scores.cue_pointer)
    options = []
    total = (
        scores.record_kind[_argmax(scores.record_kind)]
        + scores.cue_pointer[cue_position - start]
    )
    for slot in scores.slots:
        option, reason, selected = _validate_option(tokens, segment, slot)
        if reason is not None or option is None:
            return None, reason, 0.0
        options.append(option)
        total += selected

    aliases = [
        set(range(option.alias_span[0], option.alias_span[1])) for option in options
    ]
    witnesses = [
        set((option.prior_position, *option.action_positions)) | aliases[index]
        for index, option in enumerate(options)
    ]
    if aliases[0] & aliases[1]:
        return None, "overlapping-aliases", 0.0
    if witnesses[0] & witnesses[1]:
        return None, "shared-option-field", 0.0
    if cue_position in witnesses[0] or cue_position in witnesses[1]:
        return None, "cue-reused-as-option-field", 0.0

    ordered = tuple(sorted(options, key=lambda option: option.alias_span))
    return (
        RecordCandidate(
            start,
            end,
            cue_position,
            _argmax(scores.record_kind) == 1,
            (ordered[0], ordered[1]),
            int(round(total)),
        ),
        None,
        total,
    )


def decode_whole_records(
    tokens: Sequence[str], scores: WholeRecordScores
) -> WholeRecordReceipt:
    segments, reason, overflow = detect_segments(scores.boundary, len(tokens))
    if reason is not None:
        return WholeRecordReceipt((), 0.0, 0, 0, True, reason, overflow)
    by_span = {(row.start, row.end): row for row in scores.segments}
    if len(by_span) != len(scores.segments) or set(by_span) != set(segments):
        return WholeRecordReceipt((), 0.0, 0, 0, True, "segment-output-mismatch")
    records = []
    total = 0.0
    for segment in segments:
        record, reason, selected = _validate_record(tokens, segment, by_span[segment])
        if reason is not None or record is None:
            return WholeRecordReceipt(
                (), 0.0, 2 * len(segments), len(segments), True, reason
            )
        records.append(record)
        total += selected
    return WholeRecordReceipt(
        tuple(records), total, 2 * len(records), len(records), False, None
    )


def decode_reference(
    tokens: Sequence[str], scores: WholeRecordScores
) -> WholeRecordReceipt:
    """Independent exhaustive reference over the two exchangeable slot orders."""

    segments, reason, overflow = detect_segments(scores.boundary, len(tokens))
    if reason is not None:
        return WholeRecordReceipt((), 0.0, 0, 0, True, reason, overflow)
    rows = {(row.start, row.end): row for row in scores.segments}
    if len(rows) != len(scores.segments) or set(rows) != set(segments):
        return WholeRecordReceipt((), 0.0, 0, 0, True, "segment-output-mismatch")
    records = []
    total = 0.0
    for segment in segments:
        row = rows[segment]
        candidates = []
        for slots in (row.slots, (row.slots[1], row.slots[0])):
            candidate, candidate_reason, selected = _validate_record(
                tokens, segment, replace(row, slots=slots)
            )
            if candidate_reason is None and candidate is not None:
                candidates.append((selected, candidate))
        if not candidates:
            return WholeRecordReceipt(
                (), 0.0, 2 * len(segments), len(segments), True, "no-valid-slot-order"
            )
        selected, record = max(candidates, key=lambda item: (item[0], repr(item[1])))
        records.append(record)
        total += selected
    return WholeRecordReceipt(
        tuple(records), total, 2 * len(records), len(records), False, None
    )


def _semantic_signature(records: Sequence[RecordCandidate]) -> tuple[object, ...]:
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
        for record in records
    )


def _gold_signature(episode: RawSourceEpisode) -> tuple[object, ...]:
    return _semantic_signature(
        tuple(
            RecordCandidate(
                record.start,
                record.end,
                record.cue_position,
                record.is_fault_line,
                tuple(
                    OptionCandidate(
                        option.alias_span,
                        option.alias_tokens,
                        option.prior_position,
                        option.prior_class,
                        option.action_positions,
                        option.program,
                        min(
                            option.alias_span[0],
                            option.prior_position,
                            *option.action_positions,
                        ),
                        max(
                            option.alias_span[1],
                            option.prior_position + 1,
                            *(p + 1 for p in option.action_positions),
                        ),
                        0,
                    )
                    for option in record.options
                ),
                0,
            )
            for record in episode.records
        )
    )


def exact(episode: RawSourceEpisode, receipt: WholeRecordReceipt) -> bool:
    return not receipt.failed and _semantic_signature(
        receipt.records
    ) == _gold_signature(episode)


def calibrated_scores(episode: RawSourceEpisode, *, seed: int) -> WholeRecordScores:
    rng = random.Random(seed)
    boundary = [-12.0] * (len(episode.tokens) + 1)
    segments = []
    for record in episode.records:
        boundary[record.start] = 12.0
        boundary[record.end] = 12.0
        width = record.end - record.start
        ordered = list(record.options)
        if rng.randrange(2):
            ordered.reverse()
        slots = []
        for option in ordered:
            action_2 = (
                option.action_positions[1] - record.start
                if len(option.action_positions) == 2
                else width
            )
            slots.append(
                SlotScores(
                    _peaked(width, option.alias_span[0] - record.start),
                    _peaked(
                        MAX_ALIAS_LENGTH,
                        option.alias_span[1] - option.alias_span[0] - 1,
                    ),
                    _peaked(2, option.prior_class),
                    _peaked(len(PROGRAM_ROLES), option.program),
                    _peaked(width, option.prior_position - record.start),
                    _peaked(width, option.action_positions[0] - record.start),
                    _peaked(width + 1, action_2),
                )
            )
        segments.append(
            SegmentScores(
                record.start,
                record.end,
                _peaked(2, int(record.is_fault_line)),
                _peaked(width, record.cue_position - record.start),
                (slots[0], slots[1]),
            )
        )
    return WholeRecordScores(tuple(boundary), tuple(segments))


def swap_slots(scores: WholeRecordScores) -> WholeRecordScores:
    return replace(
        scores,
        segments=tuple(
            replace(row, slots=(row.slots[1], row.slots[0])) for row in scores.segments
        ),
    )


def shuffle_lineage(scores: WholeRecordScores) -> WholeRecordScores:
    """Swap only alias identity, breaking field lineage without swapping objects."""

    rows = []
    for row in scores.segments:
        left, right = row.slots
        rows.append(
            replace(
                row,
                slots=(
                    replace(
                        left,
                        alias_start=right.alias_start,
                        alias_length=right.alias_length,
                    ),
                    replace(
                        right,
                        alias_start=left.alias_start,
                        alias_length=left.alias_length,
                    ),
                ),
            )
        )
    return replace(scores, segments=tuple(rows))


def duplicate_first_slot(scores: WholeRecordScores) -> WholeRecordScores:
    return replace(
        scores,
        segments=tuple(
            replace(row, slots=(row.slots[0], row.slots[0])) for row in scores.segments
        ),
    )


def seal_source_packet(
    tokens: Sequence[str], receipt: WholeRecordReceipt
) -> SealedSourcePacket:
    return seal_source(tokens, receipt.as_sc1())


def run_gate(*, count: int, seed: int) -> dict[str, object]:
    totals = {
        "episodes": 0,
        "exact": 0,
        "reference_exact": 0,
        "extensional_parity": 0,
        "slot_swap_exact": 0,
        "lineage_shuffle_exact": 0,
        "duplicate_failed_closed": 0,
        "source_poison_invariant": 0,
        "exact_object_accounting": 0,
        "overflow": 0,
    }
    cohorts = ("train", "lexical_shift", "renderer_shift", "composition_shift")
    for index in range(count):
        episode = generate_episode(
            seed=seed + index, cohort=cohorts[index % len(cohorts)]
        )
        scores = calibrated_scores(episode, seed=seed * 31 + index)
        receipt = decode_whole_records(episode.tokens, scores)
        reference = decode_reference(episode.tokens, scores)
        swapped = decode_whole_records(episode.tokens, swap_slots(scores))
        shuffled = decode_whole_records(episode.tokens, shuffle_lineage(scores))
        duplicated = decode_whole_records(episode.tokens, duplicate_first_slot(scores))
        packet = seal_source_packet(episode.tokens, receipt)
        poisoned = seal_source_packet(tuple("poison" for _ in episode.tokens), receipt)
        totals["episodes"] += 1
        totals["exact"] += int(exact(episode, receipt))
        totals["reference_exact"] += int(exact(episode, reference))
        totals["extensional_parity"] += int(receipt.records == reference.records)
        totals["slot_swap_exact"] += int(exact(episode, swapped))
        totals["lineage_shuffle_exact"] += int(exact(episode, shuffled))
        totals["duplicate_failed_closed"] += int(duplicated.failed)
        totals["source_poison_invariant"] += int(packet.records == poisoned.records)
        totals["exact_object_accounting"] += int(
            receipt.option_objects == 2 * len(episode.records)
            and receipt.record_objects == len(episode.records)
        )
        totals["overflow"] += int(receipt.overflow)
    rates = {
        key: value / totals["episodes"]
        for key, value in totals.items()
        if key != "episodes"
    }
    gates = {
        "exact": rates["exact"] == 1.0,
        "reference_exact": rates["reference_exact"] == 1.0,
        "extensional_parity": rates["extensional_parity"] == 1.0,
        "slot_swap_invariance": rates["slot_swap_exact"] == 1.0,
        "lineage_shuffle_drop": rates["exact"] - rates["lineage_shuffle_exact"] >= 0.20,
        "invalid_fail_closed": rates["duplicate_failed_closed"] == 1.0,
        "source_poison": rates["source_poison_invariant"] == 1.0,
        "linear_accounting": rates["exact_object_accounting"] == 1.0,
        "no_overflow": rates["overflow"] == 0.0,
    }
    payload = {
        "schema": SCHEMA,
        "seed": seed,
        "count": count,
        "totals": totals,
        "rates": rates,
        "gates": gates,
        "passed": all(gates.values()),
        "accounting": {"pair_matrix_entries": 0, "objects_per_record": 3},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=202608056500)
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
