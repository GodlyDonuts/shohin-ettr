#!/usr/bin/env python3
"""Apply a label-free consensus gate to Q36 owner and synthesis trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import extract_short_answer, _normalize_short_answer
import train_apply_q36_mtr_calibration_stack as stack
import train_apply_q36_mtr_sparse_router as sparse

SELECTION_SCHEMA = "shohin-q36-mtr-synthesis-consensus-selection-v1"
REPORT_SCHEMA = "shohin-q36-mtr-synthesis-consensus-report-v1"


class Q36MTRSynthesisConsensusError(RuntimeError):
    """Synthesis-consensus inputs, extraction, or output geometry differs."""


def _answer(task: str, completion: str) -> str | None:
    if task == "mbpp":
        return None
    return _normalize_short_answer(extract_short_answer(completion))


def choose(candidates: list[dict[str, Any]], task: str) -> tuple[int, dict[str, Any]]:
    if len(candidates) != 4 or task not in sparse.TASKS:
        raise Q36MTRSynthesisConsensusError("consensus candidate geometry differs")
    production_index = stack._production_index(candidates[:3], task)
    answers = [_answer(task, candidate["completion"]) for candidate in candidates]
    selected = production_index
    reason = "production_fallback"
    synthesis_answer = answers[3]
    if synthesis_answer is not None and synthesis_answer in answers[:3]:
        selected = 3
        reason = "synthesis_confirmed_by_owner"
    elif task != "mbpp":
        votes = Counter(answer for answer in answers[:3] if answer is not None)
        if votes:
            answer, count = votes.most_common(1)[0]
            tied = sum(value == count for value in votes.values()) > 1
            if count >= 2 and not tied:
                if answers[production_index] == answer:
                    selected = production_index
                else:
                    selected = next(
                        index
                        for index, value in enumerate(answers[:3])
                        if value == answer
                    )
                reason = "owner_consensus"
    return selected, {
        "schema": SELECTION_SCHEMA,
        "reason": reason,
        "selected_index": selected,
        "production_index": production_index,
        "normalized_answers": answers,
        "development_labels_read": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    groups = [
        sparse.load_development_candidates(paths)
        for paths in (
            args.current_candidates,
            args.owner71_candidates,
            args.owner8_candidates,
            args.synthesis_candidates,
        )
    ]
    identities = set(groups[0])
    if (
        any(set(group) != identities for group in groups)
        or len(identities) != sparse.DEVELOPMENT_ROWS
    ):
        raise Q36MTRSynthesisConsensusError("consensus identity coverage differs")
    selected_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    extracted = 0
    for identity in sorted(identities):
        candidates = [group[identity] for group in groups]
        task = candidates[0]["task"]
        if any(candidate["task"] != task for candidate in candidates):
            raise Q36MTRSynthesisConsensusError("consensus task differs")
        selected, metadata = choose(candidates, task)
        chosen = dict(candidates[selected])
        chosen["synthesis_consensus_selection"] = metadata
        selected_rows.append(chosen)
        decisions.append(
            {
                "identity_sha256": identity,
                "task": task,
                **metadata,
            }
        )
        counts[metadata["reason"]] += 1
        counts[f"selected_index:{selected}"] += 1
        extracted += sum(
            answer is not None for answer in metadata["normalized_answers"]
        )
    output_sha256 = sparse._atomic_lines(args.output, selected_rows)
    decisions_sha256 = sparse._atomic_lines(args.decisions, decisions)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "label_free_synthesis_owner_answer_consensus",
        "rows": len(selected_rows),
        "development_labels_read": 0,
        "selection_counts": dict(sorted(counts.items())),
        "normalized_answers_extracted": extracted,
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "decisions": str(args.decisions.resolve()),
        "decisions_sha256": decisions_sha256,
        "candidate_sha256": {
            label: [sparse.sha256_file(path) for path in paths]
            for label, paths in zip(
                (*sparse.LINEAGES, "synthesis"),
                (
                    args.current_candidates,
                    args.owner71_candidates,
                    args.owner8_candidates,
                    args.synthesis_candidates,
                ),
                strict=True,
            )
        },
    }
    sparse._atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("current", "owner71", "owner8", "synthesis"):
        parser.add_argument(
            f"--{name}-candidates", type=Path, action="append", required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
