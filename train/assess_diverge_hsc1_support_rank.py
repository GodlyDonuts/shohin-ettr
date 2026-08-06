#!/usr/bin/env python3
"""Read-only rank audit for the frozen failed DIVERGE-HSC1 compiler.

This assessor does not update or reinterpret HSC1.  It asks whether the gold
shifted interpretation is still present in a compact, factorized k-best
support envelope even when HSC1's single Viterbi parse is wrong.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import torch
from tokenizers import Tokenizer

from diverge_hsc1_neural_compiler import (
    HierarchicalStructuredCompiler,
    _episode_encodings,
)
from diverge_hsc1_structured_compiler import (
    _margins,
    _requires_adjacency,
    gold_option_path,
    option_markers,
    path_log_partition,
    path_viterbi,
    semantic_templates,
)
from diverge_sc1_source_compiler import (
    BACKGROUND_CUE,
    CANDIDATE_CUE,
    OTHER,
    generate_episode,
)
from diverge_wra1_neural_compiler import _state_sha256, load_frozen_sc1, sha256_file
from diverge_wra1_whole_record import detect_segments
from frozen_pointer_backbone import load_frozen_pointer_backbone

SCHEMA = "shohin-diverge-hsc1-support-rank-audit-v1"
EXPECTED_HSC1_SHA256 = (
    "34c7eaee885ba5201e6e07335add1737b7b7d26b2709861b7967e0b97be64a05"
)
EXPECTED_SC1_SHA256 = "7b5348cacb1772bf45e34442e94010db71a6be20bd8d689477d037ac5fee2ffd"
DEFAULT_KS = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True, slots=True)
class RankedPath:
    score: float
    path: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RankedOption:
    score: float
    template_index: int
    path: tuple[int, ...]


def _ranked_path_key(item: RankedPath) -> tuple[object, ...]:
    return (-item.score, item.path)


def _top_k_paths(items: Iterable[RankedPath], k: int) -> list[RankedPath]:
    if k <= 0:
        raise ValueError("k must be positive")
    return sorted(items, key=_ranked_path_key)[:k]


def path_k_best(
    margins: Sequence[Sequence[float]], labels: Sequence[int], k: int
) -> tuple[RankedPath, ...]:
    """Return exact k-best monotonic paths for one semantic template."""

    if not margins or not labels or len(labels) > len(margins):
        return ()
    previous = [
        [RankedPath(float(row[labels[0]]), (position,))]
        for position, row in enumerate(margins)
    ]
    for label_index, label in enumerate(labels[1:], start=1):
        current: list[list[RankedPath]] = [[] for _ in margins]
        for position in range(1, len(margins)):
            if _requires_adjacency(labels[label_index - 1], label):
                predecessors = previous[position - 1]
            else:
                predecessors = [
                    candidate
                    for prior_position in range(position)
                    for candidate in previous[prior_position]
                ]
            current[position] = _top_k_paths(
                (
                    RankedPath(
                        candidate.score + float(margins[position][label]),
                        (*candidate.path, position),
                    )
                    for candidate in predecessors
                ),
                k,
            )
        previous = current
    return tuple(
        _top_k_paths(
            (candidate for ending in previous for candidate in ending),
            k,
        )
    )


def option_k_best(
    role_scores: Sequence[Sequence[float]], k: int
) -> tuple[RankedOption, ...]:
    """Return exact global k-best complete option parses."""

    margins = _margins(role_scores)
    candidates = []
    for template_index, template in enumerate(semantic_templates()):
        candidates.extend(
            RankedOption(path.score, template_index, path.path)
            for path in path_k_best(margins, template.labels, k)
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                -item.score,
                item.path,
                semantic_templates()[item.template_index].prior_class,
                semantic_templates()[item.template_index].program,
                semantic_templates()[item.template_index].alias_length,
                semantic_templates()[item.template_index].component_order,
            ),
        )[:k]
    )


def template_k_best(
    role_scores: Sequence[Sequence[float]], k: int
) -> tuple[tuple[float, int], ...]:
    """Rank templates by exact marginalized alignment mass.

    The proposed support lattice retains every legal alignment inside each
    admitted template. Only template identity is truncated, so this ranking is
    exact and linear in the fixed grammar size.
    """

    margins = _margins(role_scores)
    candidates = [
        (path_log_partition(margins, template.labels), template_index)
        for template_index, template in enumerate(semantic_templates())
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                -item[0],
                semantic_templates()[item[1]].prior_class,
                semantic_templates()[item[1]].program,
                semantic_templates()[item[1]].alias_length,
                semantic_templates()[item[1]].component_order,
            ),
        )[:k]
    )


def cut_k_best(cuts: Sequence[Sequence[float]], k: int) -> tuple[RankedPath, ...]:
    """Return exact k-best legal ``0 < a < b < t < width`` triples."""

    if len(cuts) != 3 or not cuts[0] or any(len(row) != len(cuts[0]) for row in cuts):
        return ()
    width = len(cuts[0])
    return tuple(
        _top_k_paths(
            (
                RankedPath(
                    float(cuts[0][left] + cuts[1][middle] + cuts[2][trailer]),
                    (left, middle, trailer),
                )
                for left, middle, trailer in itertools.combinations(range(1, width), 3)
            ),
            k,
        )
    )


def cue_k_best(
    cue_scores: Sequence[Sequence[float]], header_end: int, k: int
) -> tuple[tuple[float, int, int], ...]:
    """Return exact k-best cue-position/kind assignments inside a header."""

    candidates = [
        (
            float(cue_scores[position][kind] - cue_scores[position][OTHER]),
            position,
            kind,
        )
        for position in range(header_end)
        for kind in (CANDIDATE_CUE, BACKGROUND_CUE)
    ]
    return tuple(sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[:k])


def _rank(target: object, candidates: Sequence[object]) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if candidate == target:
            return index
    return None


def _option_target(option: object, span_start: int) -> tuple[int, tuple[int, ...]]:
    template, path = gold_option_path(option, span_start)
    template_index = semantic_templates().index(template)
    return template_index, path


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_frozen_hsc1(
    *,
    base: Path,
    tokenizer_path: Path,
    sc1_checkpoint: Path,
    hsc1_checkpoint: Path,
    device: torch.device,
    layer: int,
    width: int,
    pair_width: int,
    local_layers: int,
    local_heads: int,
) -> HierarchicalStructuredCompiler:
    if sha256_file(sc1_checkpoint) != EXPECTED_SC1_SHA256:
        raise ValueError("frozen SC1 checkpoint hash differs")
    if sha256_file(hsc1_checkpoint) != EXPECTED_HSC1_SHA256:
        raise ValueError("frozen HSC1 checkpoint hash differs")
    backbone, _, _ = load_frozen_pointer_backbone(base, device=device)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source = load_frozen_sc1(
        backbone,
        tokenizer,
        sc1_checkpoint,
        layer=layer,
        width=width,
        pair_width=pair_width,
    )
    model = HierarchicalStructuredCompiler(
        source,
        width=width,
        local_layers=local_layers,
        local_heads=local_heads,
    ).to(device)
    payload = torch.load(hsc1_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-diverge-hsc1-neural-structured-compiler-v1":
        raise ValueError("unexpected HSC1 checkpoint schema")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("HSC1 checkpoint has no compiler state")
    if _state_sha256(state) != payload.get("model_state_sha256"):
        raise ValueError("HSC1 compiler state digest differs")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("source.") for name in missing):
        raise ValueError("HSC1 checkpoint does not match frozen architecture")
    return model.eval().requires_grad_(False)


def _semantic_role_scores(
    model: HierarchicalStructuredCompiler,
    memories: Sequence[torch.Tensor],
) -> list[tuple[tuple[float, ...], ...]]:
    if not memories:
        return []
    device = memories[0].device
    widths = [int(memory.shape[0]) for memory in memories]
    maximum = max(widths)
    batch = memories[0].new_zeros((len(memories), maximum, model.width))
    valid = (
        torch.arange(maximum, device=device)[None, :]
        < torch.tensor(widths, device=device)[:, None]
    )
    for row, memory in enumerate(memories):
        batch[row, : memory.shape[0]] = memory
    encoded = model.local_encoder(batch, src_key_padding_mask=~valid)
    logits = model.role_head(model.local_norm(encoded)).float()
    return [
        tuple(
            tuple(float(value) for value in values.cpu())
            for values in logits[row, :width]
        )
        for row, width in enumerate(widths)
    ]


def assess_cohort(
    model: HierarchicalStructuredCompiler,
    *,
    cohort: str,
    count: int,
    seed: int,
    batch_size: int,
    ks: Sequence[int],
    device: torch.device,
) -> dict[str, object]:
    maximum_k = max(ks)
    totals: dict[str, object] = {
        "episodes": 0,
        "records": 0,
        "fault_line_records": 0,
        "options": 0,
        "segmentation_exact": 0,
        "phase_rank": {str(k): 0 for k in ks},
        "cue_rank": {str(k): 0 for k in ks},
        "option_template_rank": {str(k): 0 for k in ks},
        "gold_path_viterbi_exact": 0,
        "record_template_support": {str(k): 0 for k in ks},
        "fault_line_record_template_support": {str(k): 0 for k in ks},
        "episode_factorized_support_retained": {str(k): 0 for k in ks},
        "episode_full_template_support_retained": {str(k): 0 for k in ks},
        "episode_viterbi_component_retained": {str(k): 0 for k in ks},
        "rank_overflow": {"phase": 0, "cue": 0, "template": 0},
        "packed_cells": {str(k): 0 for k in ks},
        "materialized_combinations_upper": {str(k): 0 for k in ks},
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            episodes = [
                generate_episode(seed=seed + index, cohort=cohort)
                for index in range(start, min(count, start + batch_size))
            ]
            encodings = _episode_encodings(model, episodes)
            words, lengths, boundary = model._frozen_source(encodings, device)
            option_memories = [
                words[episode_index, span_start:span_end]
                for episode_index, episode in enumerate(episodes)
                for record in episode.records
                for span_start, span_end in (
                    option_markers(episode, record)[:2],
                    option_markers(episode, record)[1:],
                )
            ]
            option_scores = _semantic_role_scores(model, option_memories)
            option_cursor = 0
            for episode_index, episode in enumerate(episodes):
                length = int(lengths[episode_index].item())
                segments, reason, _ = detect_segments(
                    boundary[episode_index, : length + 1].cpu().tolist(), length
                )
                expected_segments = tuple(
                    (record.start, record.end) for record in episode.records
                )
                segmentation_exact = reason is None and segments == expected_segments
                totals["episodes"] = int(totals["episodes"]) + 1
                totals["segmentation_exact"] = int(totals["segmentation_exact"]) + int(
                    segmentation_exact
                )
                episode_support = {k: segmentation_exact for k in ks}
                episode_full = {k: segmentation_exact for k in ks}
                episode_viterbi = {k: segmentation_exact for k in ks}
                for record in episode.records:
                    totals["records"] = int(totals["records"]) + 1
                    totals["fault_line_records"] = int(
                        totals["fault_line_records"]
                    ) + int(record.is_fault_line)
                    memory = words[episode_index, record.start : record.end]
                    cuts = model.cut_head(memory).transpose(0, 1).float().cpu().tolist()
                    cue = model.cue_head(memory).float().cpu().tolist()
                    markers = option_markers(episode, record)
                    gold_cut = tuple(value - record.start for value in markers)
                    cut_rank = _rank(
                        gold_cut,
                        [candidate.path for candidate in cut_k_best(cuts, maximum_k)],
                    )
                    gold_cue = (
                        record.cue_position - record.start,
                        CANDIDATE_CUE if record.is_fault_line else BACKGROUND_CUE,
                    )
                    cue_rank = _rank(
                        gold_cue,
                        [
                            (position, kind)
                            for _, position, kind in cue_k_best(
                                cue, gold_cut[0], maximum_k
                            )
                        ],
                    )
                    template_ranks = []
                    path_counts = []
                    option_widths = []
                    for option, span_start, span_end in zip(
                        record.options,
                        markers[:2],
                        markers[1:],
                        strict=True,
                    ):
                        role_scores = option_scores[option_cursor]
                        option_cursor += 1
                        totals["options"] = int(totals["options"]) + 1
                        target_template, target_path = _option_target(
                            option, span_start
                        )
                        ranked_templates = template_k_best(role_scores, maximum_k)
                        template_ranks.append(
                            _rank(
                                target_template,
                                [
                                    template_index
                                    for _, template_index in ranked_templates
                                ],
                            )
                        )
                        margins = _margins(role_scores)
                        _, viterbi_path = path_viterbi(
                            margins, semantic_templates()[target_template].labels
                        )
                        totals["gold_path_viterbi_exact"] = int(
                            totals["gold_path_viterbi_exact"]
                        ) + int(viterbi_path == target_path)
                        width = span_end - span_start
                        option_widths.append(width)
                        path_counts.append(
                            max(
                                math.comb(width, len(template.labels))
                                for template in semantic_templates()
                                if len(template.labels) <= width
                            )
                        )
                    if cut_rank is None:
                        totals["rank_overflow"]["phase"] += 1  # type: ignore[index]
                    if cue_rank is None:
                        totals["rank_overflow"]["cue"] += 1  # type: ignore[index]
                    totals["rank_overflow"]["template"] += sum(  # type: ignore[index]
                        rank is None for rank in template_ranks
                    )
                    for k in ks:
                        phase_ok = cut_rank is not None and cut_rank <= k
                        cue_ok = cue_rank is not None and cue_rank <= k
                        templates_ok = all(
                            rank is not None and rank <= k for rank in template_ranks
                        )
                        component_ok = phase_ok and cue_ok and templates_ok
                        totals["phase_rank"][str(k)] += int(phase_ok)  # type: ignore[index]
                        totals["cue_rank"][str(k)] += int(cue_ok)  # type: ignore[index]
                        totals["option_template_rank"][str(k)] += sum(  # type: ignore[index]
                            rank is not None and rank <= k for rank in template_ranks
                        )
                        totals["record_template_support"][str(k)] += int(  # type: ignore[index]
                            templates_ok
                        )
                        if record.is_fault_line:
                            totals["fault_line_record_template_support"][str(k)] += int(  # type: ignore[index]
                                templates_ok
                            )
                            episode_support[k] = episode_support[k] and templates_ok
                        episode_full[k] = episode_full[k] and templates_ok
                        episode_viterbi[k] = episode_viterbi[k] and component_ok
                        totals["packed_cells"][str(k)] += (  # type: ignore[index]
                            3 * (record.end - record.start)
                            + 2 * gold_cut[0]
                            + k * sum(option_widths)
                        )
                        totals["materialized_combinations_upper"][str(k)] += (  # type: ignore[index]
                            math.comb(record.end - record.start - 1, 3)
                            * 2
                            * gold_cut[0]
                            * min(k, len(semantic_templates())) ** 2
                            * max(1, path_counts[0])
                            * max(1, path_counts[1])
                        )
                for k in ks:
                    totals["episode_factorized_support_retained"][str(k)] += int(  # type: ignore[index]
                        episode_support[k]
                    )
                    totals["episode_full_template_support_retained"][str(k)] += int(  # type: ignore[index]
                        episode_full[k]
                    )
                    totals["episode_viterbi_component_retained"][str(k)] += int(  # type: ignore[index]
                        episode_viterbi[k]
                    )
            if option_cursor != len(option_scores):
                raise AssertionError("option-score accounting differs")
    episodes = int(totals["episodes"])
    records = int(totals["records"])
    fault_records = int(totals["fault_line_records"])
    options = int(totals["options"])
    rates = {
        "segmentation_exact": int(totals["segmentation_exact"]) / episodes,
        "phase_rank": {
            key: value / records
            for key, value in totals["phase_rank"].items()  # type: ignore[union-attr]
        },
        "cue_rank": {
            key: value / records
            for key, value in totals["cue_rank"].items()  # type: ignore[union-attr]
        },
        "option_template_rank": {
            key: value / options
            for key, value in totals["option_template_rank"].items()  # type: ignore[union-attr]
        },
        "gold_path_viterbi_exact": int(totals["gold_path_viterbi_exact"]) / options,
        "record_template_support": {
            key: value / records
            for key, value in totals["record_template_support"].items()  # type: ignore[union-attr]
        },
        "fault_line_record_template_support": {
            key: value / fault_records
            for key, value in totals[
                "fault_line_record_template_support"
            ].items()  # type: ignore[union-attr]
        },
        "episode_factorized_support_retained": {
            key: value / episodes
            for key, value in totals[
                "episode_factorized_support_retained"
            ].items()  # type: ignore[union-attr]
        },
        "episode_full_template_support_retained": {
            key: value / episodes
            for key, value in totals[
                "episode_full_template_support_retained"
            ].items()  # type: ignore[union-attr]
        },
        "episode_viterbi_component_retained": {
            key: value / episodes
            for key, value in totals[
                "episode_viterbi_component_retained"
            ].items()  # type: ignore[union-attr]
        },
    }
    return {"cohort": cohort, "count": count, "totals": totals, "rates": rates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sc1-checkpoint", type=Path, required=True)
    parser.add_argument("--hsc1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202608057300)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--pair-width", type=int, default=64)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("refusing to overwrite support-rank audit")
    ks = tuple(sorted({int(value) for value in arguments.ks.split(",")}))
    if not ks or ks[0] <= 0 or arguments.count <= 0 or arguments.batch_size <= 0:
        raise ValueError("rank widths and evaluation sizes must be positive")
    torch.set_num_threads(arguments.threads)
    device = torch.device(arguments.device)
    model = load_frozen_hsc1(
        base=arguments.base,
        tokenizer_path=arguments.tokenizer,
        sc1_checkpoint=arguments.sc1_checkpoint,
        hsc1_checkpoint=arguments.hsc1_checkpoint,
        device=device,
        layer=arguments.layer,
        width=arguments.width,
        pair_width=arguments.pair_width,
        local_layers=arguments.local_layers,
        local_heads=arguments.local_heads,
    )
    evaluations = {
        cohort: assess_cohort(
            model,
            cohort=cohort,
            count=arguments.count,
            seed=arguments.seed + offset,
            batch_size=arguments.batch_size,
            ks=ks,
            device=device,
        )
        for cohort, offset in (
            ("train", 0),
            ("lexical_shift", 100_000),
            ("renderer_shift", 200_000),
            ("composition_shift", 300_000),
        )
    }
    shifted = ("lexical_shift", "renderer_shift", "composition_shift")
    min_shifted_support = {
        str(k): min(
            evaluations[cohort]["rates"]["episode_factorized_support_retained"][str(k)]
            for cohort in shifted
        )
        for k in ks
    }
    report = {
        "schema": SCHEMA,
        "status": "read-only-diagnostic-no-capability-standing",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "input_hashes": {
            "base": sha256_file(arguments.base),
            "tokenizer": sha256_file(arguments.tokenizer),
            "sc1_checkpoint": sha256_file(arguments.sc1_checkpoint),
            "hsc1_checkpoint": sha256_file(arguments.hsc1_checkpoint),
        },
        "evaluations": evaluations,
        "minimum_shifted_episode_support_retained": min_shifted_support,
        "decision_rule": {
            "support_lattice_candidate": "K<=64 templates and minimum shifted episode support >=0.95",
            "replace_language_interface": "no K<=64 reaches the support floor",
        },
        "support_lattice_candidate": any(
            k <= 64 and min_shifted_support[str(k)] >= 0.95 for k in ks
        ),
    }
    _atomic_json(arguments.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
