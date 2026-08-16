#!/usr/bin/env python3
"""Select a Q36 answer by label-free multi-trajectory consensus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import (
    _normalize_math,
    _normalize_short_answer,
    extract_boxed,
    extract_short_answer,
)
from hf_q36_mtr_hierarchical_synthesis import ROWS, load_candidate_group
from select_q36_mtr_interpolation_retention import (
    _atomic_json,
    _atomic_lines,
    sha256_file,
)

SCHEMA = "shohin-q36-mtr-multi-trajectory-consensus-v1"
REPORT_SCHEMA = "shohin-q36-mtr-multi-trajectory-consensus-report-v1"
ARM_ORDER = (
    "hierarchy",
    "interpolation",
    "direct",
    "offset_one",
    "level_two",
    "challenger",
)


class Q36MTRConsensusError(RuntimeError):
    """Raised when the consensus inputs or selection differ."""


def normalized_answer(task: str, completion: str) -> str | None:
    if task == "bbh_logic":
        return _normalize_short_answer(extract_short_answer(completion))
    if task == "math500":
        return _normalize_math(extract_boxed(completion))
    if task == "mbpp":
        return None
    raise Q36MTRConsensusError("consensus task differs")


def choose(rows: dict[str, dict[str, Any]]) -> tuple[str, int]:
    if tuple(rows) != ARM_ORDER:
        raise Q36MTRConsensusError("consensus arm order differs")
    identities = {row.get("identity_sha256") for row in rows.values()}
    tasks = {row.get("task") for row in rows.values()}
    if len(identities) != 1 or len(tasks) != 1:
        raise Q36MTRConsensusError("consensus identity differs")
    task = next(iter(tasks))
    if task == "mbpp":
        return "interpolation", 1
    answers = {
        arm: normalized_answer(task, row["completion"]) for arm, row in rows.items()
    }
    counts = Counter(answer for answer in answers.values() if answer is not None)
    if not counts:
        return "hierarchy", 0
    votes = max(counts.values())
    winners = {answer for answer, count in counts.items() if count == votes}
    selected = next(
        arm for arm in ARM_ORDER if answers[arm] is not None and answers[arm] in winners
    )
    return selected, votes


def run(
    path_groups: dict[str, list[Path]], output: Path, report_path: Path
) -> dict[str, Any]:
    if tuple(path_groups) != ARM_ORDER:
        raise Q36MTRConsensusError("consensus path groups differ")
    groups = {
        arm: load_candidate_group(paths, expected_paths=16)
        for arm, paths in path_groups.items()
    }
    identities = set(groups["hierarchy"])
    if len(identities) != ROWS or any(
        set(group) != identities for group in groups.values()
    ):
        raise Q36MTRConsensusError("consensus coverage differs")
    selections = Counter()
    vote_sizes = Counter()
    outputs = []
    for identity in sorted(identities):
        candidates = {arm: groups[arm][identity] for arm in ARM_ORDER}
        selected, votes = choose(candidates)
        row = dict(candidates[selected])
        row["multi_trajectory_consensus"] = {
            "schema": SCHEMA,
            "selected": selected,
            "votes": votes,
            "arm_order": list(ARM_ORDER),
            "development_labels_read": 0,
        }
        outputs.append(row)
        selections[(row["task"], selected)] += 1
        vote_sizes[(row["task"], votes)] += 1
    output_sha256 = _atomic_lines(output, outputs)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "rows": len(outputs),
        "arm_order": list(ARM_ORDER),
        "selection_counts": {
            f"{task}:{arm}": count for (task, arm), count in sorted(selections.items())
        },
        "vote_size_counts": {
            f"{task}:{votes}": count
            for (task, votes), count in sorted(vote_sizes.items())
        },
        "input_sha256": {
            arm: [sha256_file(path) for path in paths]
            for arm, paths in path_groups.items()
        },
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
        "development_labels_read": 0,
    }
    _atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARM_ORDER:
        parser.add_argument(
            f"--{arm.replace('_', '-')}", action="append", type=Path, required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = {arm: getattr(args, arm) for arm in ARM_ORDER}
    print(json.dumps(run(groups, args.output, args.report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
