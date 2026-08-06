#!/usr/bin/env python3
"""Exact CPU mechanics for DIVERGE-HSC1 hierarchical source compilation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

from diverge_sc1_source_compiler import (
    ACTION_ADD,
    ACTION_SWAP01,
    ACTION_SWAP23,
    ACTION_SWAP34,
    ALIAS_BEGIN,
    ALIAS_INSIDE,
    BACKGROUND_CUE,
    CANDIDATE_CUE,
    OTHER,
    OptionCandidate,
    PRIOR_FAVORED,
    PROGRAM_ROLES,
    ROLE_COUNT,
    RawSourceEpisode,
    RecordCandidate,
    generate_episode,
)
from diverge_wra1_whole_record import (
    WholeRecordReceipt,
    detect_segments,
    exact,
    seal_source_packet,
)

SCHEMA = "shohin-diverge-hsc1-structured-cpu-v1"
COMPONENT_ORDERS = (
    ("prior", "alias", "action"),
    ("alias", "action", "prior"),
    ("action", "prior", "alias"),
    ("prior", "action", "alias"),
)
SEMANTIC_ROLES = frozenset(
    (
        ALIAS_BEGIN,
        ALIAS_INSIDE,
        PRIOR_FAVORED,
        PRIOR_FAVORED + 1,
        ACTION_ADD,
        ACTION_SWAP01,
        ACTION_SWAP23,
        ACTION_SWAP34,
    )
)


@dataclass(frozen=True, slots=True)
class SemanticTemplate:
    prior_class: int
    program: int
    alias_length: int
    component_order: int
    labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OptionRoleScores:
    start: int
    end: int
    role: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class RecordStructuredScores:
    start: int
    end: int
    cuts: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    cue_role: tuple[tuple[float, float, float], ...]
    options: tuple[OptionRoleScores, OptionRoleScores]


@dataclass(frozen=True, slots=True)
class HierarchicalScores:
    boundary: tuple[float, ...]
    records: tuple[RecordStructuredScores, ...]


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _logsumexp(values: Iterable[float]) -> float:
    result = -math.inf
    for value in values:
        result = _logaddexp(result, value)
    return result


@lru_cache(maxsize=1)
def semantic_templates() -> tuple[SemanticTemplate, ...]:
    output = []
    for prior_class in range(2):
        prior = (PRIOR_FAVORED + prior_class,)
        for program, actions in PROGRAM_ROLES.items():
            for alias_length in range(1, 5):
                alias = (ALIAS_BEGIN, *([ALIAS_INSIDE] * (alias_length - 1)))
                components = {"prior": prior, "alias": alias, "action": actions}
                for order_index, order in enumerate(COMPONENT_ORDERS):
                    labels = tuple(
                        label for component in order for label in components[component]
                    )
                    output.append(
                        SemanticTemplate(
                            prior_class,
                            program,
                            alias_length,
                            order_index,
                            labels,
                        )
                    )
    if len(output) != 128 or len({row.labels for row in output}) != 128:
        raise AssertionError("semantic template grammar is not one-to-one")
    return tuple(output)


def _requires_adjacency(previous: int, current: int) -> bool:
    return current == ALIAS_INSIDE and previous in {ALIAS_BEGIN, ALIAS_INSIDE}


def _margins(role: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    output = []
    for row in role:
        if len(row) != ROLE_COUNT:
            raise ValueError("semantic role score width differs")
        output.append(tuple(float(value - row[OTHER]) for value in row))
    return tuple(output)


def path_log_partition(
    margins: Sequence[Sequence[float]], labels: Sequence[int]
) -> float:
    if not margins or not labels or len(labels) > len(margins):
        return -math.inf
    previous = [float(row[labels[0]]) for row in margins]
    for label_index, label in enumerate(labels[1:], start=1):
        current = [-math.inf] * len(margins)
        if _requires_adjacency(labels[label_index - 1], label):
            for position in range(1, len(margins)):
                current[position] = previous[position - 1] + margins[position][label]
        else:
            prefix = -math.inf
            for position in range(len(margins)):
                current[position] = prefix + margins[position][label]
                prefix = _logaddexp(prefix, previous[position])
        previous = current
    return _logsumexp(previous)


def path_viterbi(
    margins: Sequence[Sequence[float]], labels: Sequence[int]
) -> tuple[float, tuple[int, ...]]:
    if not margins or not labels or len(labels) > len(margins):
        return -math.inf, ()
    previous = [
        (float(row[labels[0]]), (position,)) for position, row in enumerate(margins)
    ]
    for label_index, label in enumerate(labels[1:], start=1):
        current = [(-math.inf, ()) for _ in margins]
        if _requires_adjacency(labels[label_index - 1], label):
            for position in range(1, len(margins)):
                score, path = previous[position - 1]
                current[position] = (
                    score + margins[position][label],
                    (*path, position),
                )
        else:
            best = (-math.inf, ())
            for position in range(len(margins)):
                score, path = best
                if path:
                    current[position] = (
                        score + margins[position][label],
                        (*path, position),
                    )
                candidate = previous[position]
                if (candidate[0], tuple(-value for value in candidate[1])) > (
                    best[0],
                    tuple(-value for value in best[1]),
                ):
                    best = candidate
        previous = current
    return max(
        previous,
        key=lambda item: (item[0], tuple(-value for value in item[1])),
    )


def exhaustive_paths(length: int, labels: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    output = []
    for path in itertools.combinations(range(length), len(labels)):
        if any(
            _requires_adjacency(labels[index - 1], labels[index])
            and path[index] != path[index - 1] + 1
            for index in range(1, len(labels))
        ):
            continue
        output.append(path)
    return tuple(output)


def exhaustive_path_partition(
    margins: Sequence[Sequence[float]], labels: Sequence[int]
) -> tuple[float, tuple[float, tuple[int, ...]]]:
    scored = [
        (
            sum(
                margins[position][label]
                for position, label in zip(path, labels, strict=True)
            ),
            path,
        )
        for path in exhaustive_paths(len(margins), labels)
    ]
    if not scored:
        return -math.inf, (-math.inf, ())
    return (
        _logsumexp(score for score, _ in scored),
        max(scored, key=lambda item: (item[0], tuple(-value for value in item[1]))),
    )


def cut_log_partition(cuts: Sequence[Sequence[float]]) -> float:
    if len(cuts) != 3 or not cuts[0] or any(len(row) != len(cuts[0]) for row in cuts):
        return -math.inf
    width = len(cuts[0])
    first = [-math.inf] * width
    for position in range(1, width):
        first[position] = float(cuts[0][position])
    second = [-math.inf] * width
    prefix = -math.inf
    for position in range(1, width):
        second[position] = prefix + cuts[1][position]
        prefix = _logaddexp(prefix, first[position])
    third = [-math.inf] * width
    prefix = -math.inf
    for position in range(1, width):
        third[position] = prefix + cuts[2][position]
        prefix = _logaddexp(prefix, second[position])
    return _logsumexp(third[1:])


def cut_viterbi(cuts: Sequence[Sequence[float]]) -> tuple[float, tuple[int, int, int]]:
    if len(cuts) != 3 or not cuts[0] or any(len(row) != len(cuts[0]) for row in cuts):
        return -math.inf, ()
    width = len(cuts[0])
    first = [(-math.inf, ()) for _ in range(width)]
    for position in range(1, width):
        first[position] = (float(cuts[0][position]), (position,))
    layers = [first]
    for channel in (1, 2):
        prior = layers[-1]
        current = [(-math.inf, ()) for _ in range(width)]
        best = (-math.inf, ())
        for position in range(1, width):
            if best[1]:
                current[position] = (
                    best[0] + cuts[channel][position],
                    (*best[1], position),
                )
            candidate = prior[position]
            if (candidate[0], tuple(-value for value in candidate[1])) > (
                best[0],
                tuple(-value for value in best[1]),
            ):
                best = candidate
        layers.append(current)
    score, path = max(
        layers[-1],
        key=lambda item: (item[0], tuple(-value for value in item[1])),
    )
    return score, path  # type: ignore[return-value]


def exhaustive_cut_partition(
    cuts: Sequence[Sequence[float]],
) -> tuple[float, tuple[float, tuple[int, int, int]]]:
    width = len(cuts[0])
    scored = [
        (cuts[0][left] + cuts[1][middle] + cuts[2][right], (left, middle, right))
        for left, middle, right in itertools.combinations(range(1, width), 3)
    ]
    if not scored:
        return -math.inf, (-math.inf, ())
    return (
        _logsumexp(score for score, _ in scored),
        max(scored, key=lambda item: (item[0], tuple(-value for value in item[1]))),
    )


def option_markers(episode: RawSourceEpisode, record) -> tuple[int, int, int]:
    option_starts = tuple(
        position
        for position in range(record.start, record.end)
        if episode.tokens[position] == "option"
    )
    trailer = tuple(
        position
        for position in range(record.start, record.end)
        if episode.tokens[position] == "glossary"
    )
    if len(option_starts) != 2 or len(trailer) != 1:
        raise ValueError("supervisor cannot identify unique option phases")
    if not record.start < option_starts[0] < option_starts[1] < trailer[0] < record.end:
        raise ValueError("gold option phases are not monotonic")
    return option_starts[0], option_starts[1], trailer[0]


def gold_option_path(
    option, span_start: int
) -> tuple[SemanticTemplate, tuple[int, ...]]:
    positioned = [(option.alias_span[0], ALIAS_BEGIN)]
    positioned.extend(
        (position, ALIAS_INSIDE)
        for position in range(option.alias_span[0] + 1, option.alias_span[1])
    )
    positioned.append((option.prior_position, PRIOR_FAVORED + option.prior_class))
    positioned.extend(
        zip(option.action_positions, PROGRAM_ROLES[option.program], strict=True)
    )
    positioned.sort()
    labels = tuple(label for _, label in positioned)
    matches = [
        template
        for template in semantic_templates()
        if template.prior_class == option.prior_class
        and template.program == option.program
        and template.alias_length == option.alias_span[1] - option.alias_span[0]
        and template.labels == labels
    ]
    if len(matches) != 1:
        raise ValueError("gold option does not map to one semantic template")
    return matches[0], tuple(position - span_start for position, _ in positioned)


def decode_option(
    tokens: Sequence[str], scores: OptionRoleScores
) -> tuple[OptionCandidate | None, str | None, float]:
    if len(scores.role) != scores.end - scores.start or scores.start >= scores.end:
        return None, "option-score-width", 0.0
    margins = _margins(scores.role)
    candidates = []
    for template in semantic_templates():
        score, path = path_viterbi(margins, template.labels)
        if path:
            candidates.append((score, template, path))
    if not candidates:
        return None, "no-semantic-path", 0.0
    score, template, path = max(
        candidates,
        key=lambda item: (
            item[0],
            tuple(-value for value in item[2]),
            -item[1].prior_class,
            -item[1].program,
            -item[1].alias_length,
            -item[1].component_order,
        ),
    )
    positions = tuple(scores.start + value for value in path)
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
    if alias_positions != tuple(
        range(alias_positions[0], alias_positions[0] + len(alias_positions))
    ):
        return None, "noncontiguous-alias", 0.0
    prior_position = positions[prior_index]
    action_positions = tuple(positions[index] for index in action_indices)
    occupied = set(alias_positions)
    if prior_position in occupied or any(
        position in occupied for position in action_positions
    ):
        return None, "overlapping-option-fields", 0.0
    return (
        OptionCandidate(
            (alias_positions[0], alias_positions[-1] + 1),
            tuple(tokens[alias_positions[0] : alias_positions[-1] + 1]),
            prior_position,
            template.prior_class,
            action_positions,
            template.program,
            min(alias_positions[0], prior_position, *action_positions),
            max(
                alias_positions[-1] + 1,
                prior_position + 1,
                *(value + 1 for value in action_positions),
            ),
            int(round(score)),
        ),
        None,
        score,
    )


def decode_hierarchical(
    tokens: Sequence[str], scores: HierarchicalScores
) -> WholeRecordReceipt:
    segments, reason, overflow = detect_segments(scores.boundary, len(tokens))
    if reason is not None:
        return WholeRecordReceipt((), 0.0, 0, 0, True, reason, overflow)
    rows = {(row.start, row.end): row for row in scores.records}
    if len(rows) != len(scores.records) or set(rows) != set(segments):
        return WholeRecordReceipt((), 0.0, 0, 0, True, "record-output-mismatch")
    records = []
    total = 0.0
    for start, end in segments:
        row = rows[start, end]
        width = end - start
        if (
            any(len(channel) != width for channel in row.cuts)
            or len(row.cue_role) != width
        ):
            return WholeRecordReceipt((), 0.0, 0, 0, True, "record-score-width")
        cut_score, cuts = cut_viterbi(row.cuts)
        if len(cuts) != 3:
            return WholeRecordReceipt((), 0.0, 0, 0, True, "no-monotonic-parse")
        left, middle, trailer = (start + value for value in cuts)
        expected_options = ((left, middle), (middle, trailer))
        if (
            tuple((option.start, option.end) for option in row.options)
            != expected_options
        ):
            return WholeRecordReceipt((), 0.0, 0, 0, True, "option-output-mismatch")
        cue_candidates = [
            (
                row.cue_role[position][kind] - row.cue_role[position][OTHER],
                position,
                kind,
            )
            for position in range(cuts[0])
            for kind in (CANDIDATE_CUE, BACKGROUND_CUE)
        ]
        if not cue_candidates:
            return WholeRecordReceipt((), 0.0, 0, 0, True, "empty-header")
        cue_score, cue_local, cue_kind = max(
            cue_candidates, key=lambda item: (item[0], -item[1], -item[2])
        )
        decoded = []
        record_score = cut_score + cue_score
        for option_scores in row.options:
            option, option_reason, option_score = decode_option(tokens, option_scores)
            if option is None or option_reason is not None:
                return WholeRecordReceipt((), 0.0, 0, 0, True, option_reason)
            decoded.append(option)
            record_score += option_score
        records.append(
            RecordCandidate(
                start,
                end,
                start + cue_local,
                cue_kind == CANDIDATE_CUE,
                (decoded[0], decoded[1]),
                int(round(record_score)),
            )
        )
        total += record_score
    return WholeRecordReceipt(
        tuple(records), total, 2 * len(records), len(records), False, None
    )


def _peaked(size: int, index: int, high: float = 12.0) -> tuple[float, ...]:
    return tuple(high if value == index else -high for value in range(size))


def calibrated_scores(episode: RawSourceEpisode) -> HierarchicalScores:
    boundary = [-12.0] * (len(episode.tokens) + 1)
    rows = []
    for record in episode.records:
        boundary[record.start] = 12.0
        boundary[record.end] = 12.0
        width = record.end - record.start
        first, second, trailer = option_markers(episode, record)
        cuts = tuple(
            _peaked(width, position - record.start)
            for position in (first, second, trailer)
        )
        cue_role = []
        for position in range(record.start, record.end):
            target = (
                (CANDIDATE_CUE if record.is_fault_line else BACKGROUND_CUE)
                if position == record.cue_position
                else OTHER
            )
            cue_role.append(_peaked(3, target))
        option_rows = []
        for option, (option_start, option_end) in zip(
            record.options,
            ((first, second), (second, trailer)),
            strict=True,
        ):
            role = [[-12.0] * ROLE_COUNT for _ in range(option_end - option_start)]
            for row in role:
                row[OTHER] = 12.0

            def assign(position: int, semantic_role: int) -> None:
                local = position - option_start
                role[local][OTHER] = -12.0
                role[local][semantic_role] = 12.0

            assign(option.alias_span[0], ALIAS_BEGIN)
            for position in range(option.alias_span[0] + 1, option.alias_span[1]):
                assign(position, ALIAS_INSIDE)
            assign(option.prior_position, PRIOR_FAVORED + option.prior_class)
            for position, semantic_role in zip(
                option.action_positions, PROGRAM_ROLES[option.program], strict=True
            ):
                assign(position, semantic_role)
            option_rows.append(
                OptionRoleScores(
                    option_start,
                    option_end,
                    tuple(tuple(value) for value in role),
                )
            )
        rows.append(
            RecordStructuredScores(
                record.start,
                record.end,
                cuts,  # type: ignore[arg-type]
                tuple(cue_role),
                (option_rows[0], option_rows[1]),
            )
        )
    return HierarchicalScores(tuple(boundary), tuple(rows))


def shuffle_cut_channels(scores: HierarchicalScores) -> HierarchicalScores:
    return replace(
        scores,
        records=tuple(
            replace(row, cuts=(row.cuts[1], row.cuts[0], row.cuts[2]))
            for row in scores.records
        ),
    )


def shuffle_semantic_roles(scores: HierarchicalScores) -> HierarchicalScores:
    permutation = {
        PRIOR_FAVORED: PRIOR_FAVORED + 1,
        PRIOR_FAVORED + 1: PRIOR_FAVORED,
        ACTION_ADD: ACTION_SWAP01,
        ACTION_SWAP01: ACTION_SWAP23,
        ACTION_SWAP23: ACTION_SWAP34,
        ACTION_SWAP34: ACTION_ADD,
    }
    records = []
    for record in scores.records:
        options = []
        for option in record.options:
            rows = []
            for values in option.role:
                changed = list(values)
                for target, source in permutation.items():
                    changed[target] = values[source]
                rows.append(tuple(changed))
            options.append(replace(option, role=tuple(rows)))
        records.append(replace(record, options=(options[0], options[1])))
    return replace(scores, records=tuple(records))


def malformed_option_width(scores: HierarchicalScores) -> HierarchicalScores:
    first = scores.records[0]
    option = first.options[0]
    broken = replace(option, role=option.role[:-1])
    changed = replace(first, options=(broken, first.options[1]))
    return replace(scores, records=(changed, *scores.records[1:]))


def run_gate(*, count: int, seed: int) -> dict[str, object]:
    totals = {
        "episodes": 0,
        "exact": 0,
        "cut_shuffle_exact": 0,
        "semantic_shuffle_exact": 0,
        "malformed_failed_closed": 0,
        "source_poison_invariant": 0,
        "linear_accounting": 0,
        "overflow": 0,
        "source_words": 0,
        "records": 0,
        "option_words": 0,
        "cut_score_cells": 0,
        "cue_score_cells": 0,
        "semantic_score_cells": 0,
        "template_evaluations": 0,
    }
    cohorts = ("train", "lexical_shift", "renderer_shift", "composition_shift")
    for index in range(count):
        episode = generate_episode(seed=seed + index, cohort=cohorts[index % 4])
        scores = calibrated_scores(episode)
        receipt = decode_hierarchical(episode.tokens, scores)
        cut_shuffled = decode_hierarchical(episode.tokens, shuffle_cut_channels(scores))
        semantic_shuffled = decode_hierarchical(
            episode.tokens, shuffle_semantic_roles(scores)
        )
        malformed = decode_hierarchical(episode.tokens, malformed_option_width(scores))
        packet = seal_source_packet(episode.tokens, receipt)
        poison = seal_source_packet(tuple("poison" for _ in episode.tokens), receipt)
        records = len(scores.records)
        record_words = sum(row.end - row.start for row in scores.records)
        option_words = sum(
            option.end - option.start
            for row in scores.records
            for option in row.options
        )
        semantic_cells = sum(
            len(option.role) * ROLE_COUNT
            for row in scores.records
            for option in row.options
        )
        linear = (
            receipt.record_objects == records
            and receipt.option_objects == 2 * records
            and sum(sum(len(channel) for channel in row.cuts) for row in scores.records)
            == 3 * record_words
            and sum(len(row.cue_role) * 3 for row in scores.records) == 3 * record_words
            and semantic_cells == option_words * ROLE_COUNT
        )
        totals["episodes"] += 1
        totals["exact"] += int(exact(episode, receipt))
        totals["cut_shuffle_exact"] += int(exact(episode, cut_shuffled))
        totals["semantic_shuffle_exact"] += int(exact(episode, semantic_shuffled))
        totals["malformed_failed_closed"] += int(malformed.failed)
        totals["source_poison_invariant"] += int(packet.records == poison.records)
        totals["linear_accounting"] += int(linear)
        totals["overflow"] += int(receipt.overflow)
        totals["source_words"] += len(episode.tokens)
        totals["records"] += records
        totals["option_words"] += option_words
        totals["cut_score_cells"] += 3 * record_words
        totals["cue_score_cells"] += 3 * record_words
        totals["semantic_score_cells"] += semantic_cells
        totals["template_evaluations"] += 2 * records * len(semantic_templates())
    rates = {
        key: value / totals["episodes"]
        for key, value in totals.items()
        if key
        not in {
            "episodes",
            "source_words",
            "records",
            "option_words",
            "cut_score_cells",
            "cue_score_cells",
            "semantic_score_cells",
            "template_evaluations",
        }
    }
    gates = {
        "exact": rates["exact"] == 1.0,
        "cut_causality": rates["exact"] - rates["cut_shuffle_exact"] >= 0.20,
        "semantic_causality": rates["exact"] - rates["semantic_shuffle_exact"] >= 0.20,
        "malformed_fail_closed": rates["malformed_failed_closed"] == 1.0,
        "source_poison": rates["source_poison_invariant"] == 1.0,
        "linear_accounting": rates["linear_accounting"] == 1.0,
        "no_overflow": rates["overflow"] == 0.0,
    }
    report = {
        "schema": SCHEMA,
        "seed": seed,
        "count": count,
        "grammar": {
            "templates": len(semantic_templates()),
            "pair_matrix_entries": 0,
            "cut_channels": 3,
            "cue_roles": 3,
            "semantic_roles": ROLE_COUNT,
        },
        "totals": totals,
        "rates": rates,
        "gates": gates,
        "passed": all(gates.values()),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


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
    parser.add_argument("--seed", type=int, default=202608056800)
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
