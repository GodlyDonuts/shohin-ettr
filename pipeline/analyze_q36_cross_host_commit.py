#!/usr/bin/env python3
"""Replay frozen arm outcomes for a label-free cross-host semantic commit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SELECTION_SCHEMA = "shohin-q36-cross-host-semantic-commit-selection-v1"
REPORT_SCHEMA = "shohin-q36-cross-host-semantic-commit-result-v1"
HOSTS = {
    "gpt_oss_120b_screen": {
        "rows": 256,
        "score_schema": "shohin-gpt-oss-120b-fixed-draft-screen-score-v1",
    },
    "gpt_oss_120b_confirmation": {
        "rows": 256,
        "score_schema": "shohin-gpt-oss-120b-commit-confirmation-score-v1",
    },
    "mixtral_8x22b_validation": {
        "rows": 1_023,
        "score_schema": "shohin-mixtral-8x22b-selective-commit-validation-score-v1",
    },
}
TASKS = ("math500", "bbh_logic", "mbpp", "mmlu_pro")
LINEAGES = ("revision", "unchanged")


class CrossHostAnalysisError(RuntimeError):
    """The frozen selections, outcomes, or paired accounting differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CrossHostAnalysisError(f"missing or linked input: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CrossHostAnalysisError(f"unreadable input: {path}") from error
    if not isinstance(payload, dict):
        raise CrossHostAnalysisError("cross-host JSON differs")
    return payload


def _selections(
    path: Path, host: str, rows: int, revision_margin_threshold: float
) -> dict[str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise CrossHostAnalysisError("cross-host selections are absent or linked")
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        threshold = (
            row.get("revision_margin_threshold", 0.0) if isinstance(row, dict) else None
        )
        if (
            not isinstance(row, dict)
            or row.get("schema") != SELECTION_SCHEMA
            or row.get("host") != host
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in result
            or row.get("task") not in TASKS
            or row.get("selected_index") not in (0, 1)
            or row.get("selected_lineage") != LINEAGES[row["selected_index"]]
            or row.get("order_consistent") is not True
            or isinstance(row.get("margin"), bool)
            or not isinstance(row.get("margin"), (int, float))
            or not math.isfinite(float(row["margin"]))
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) != revision_margin_threshold
        ):
            raise CrossHostAnalysisError("cross-host selection differs")
        result[identity] = row
    if len(result) != rows:
        raise CrossHostAnalysisError("cross-host selection coverage differs")
    return result


def mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    contract = HOSTS[args.host]
    revision_margin_threshold = getattr(args, "expected_revision_margin_threshold", 0.0)
    if args.output.exists() or args.output.is_symlink():
        raise CrossHostAnalysisError("cross-host result already exists")
    if not math.isfinite(revision_margin_threshold) or revision_margin_threshold < 0.0:
        raise CrossHostAnalysisError("cross-host threshold differs")
    selection_rows = _selections(
        args.selections,
        args.host,
        contract["rows"],
        revision_margin_threshold,
    )
    application = _json(args.application_report)
    score = _json(args.score)
    application_threshold = application.get("revision_margin_threshold", 0.0)
    if (
        application.get("schema")
        != "shohin-q36-cross-host-semantic-commit-application-v1"
        or application.get("status") != "complete"
        or application.get("host") != args.host
        or application.get("rows") != contract["rows"]
        or application.get("selections_sha256") != sha256_file(args.selections)
        or application.get("task_correctness_or_host_label_visible") is not False
        or application.get("assessor_access_count") != 0
        or isinstance(application_threshold, bool)
        or not isinstance(application_threshold, (int, float))
        or not math.isfinite(float(application_threshold))
        or float(application_threshold) != revision_margin_threshold
        or score.get("schema") != contract["score_schema"]
        or score.get("status") != "complete"
        or score.get("rows") != contract["rows"]
        or not isinstance(score.get("outcomes"), list)
        or len(score["outcomes"]) != contract["rows"]
    ):
        raise CrossHostAnalysisError("cross-host application or score differs")

    outcomes: dict[str, dict[str, Any]] = {}
    for row in score["outcomes"]:
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        correct = row.get("correct") if isinstance(row, dict) else None
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or identity in outcomes
            or row.get("task") not in TASKS
            or not isinstance(correct, dict)
            or any(correct.get(lineage) not in (True, False) for lineage in LINEAGES)
        ):
            raise CrossHostAnalysisError("cross-host outcome differs")
        outcomes[identity] = row
    if set(outcomes) != set(selection_rows):
        raise CrossHostAnalysisError("cross-host score identities differ")

    selected_correct = unchanged_correct = revision_correct = retained = 0
    selected_counts: Counter[str] = Counter()
    domain: dict[str, Counter[str]] = defaultdict(Counter)
    selected_only_unchanged = unchanged_only_selected = 0
    selected_only_revision = revision_only_selected = 0
    for identity in sorted(outcomes):
        outcome = outcomes[identity]
        selection = selection_rows[identity]
        if outcome["task"] != selection["task"]:
            raise CrossHostAnalysisError("cross-host task binding differs")
        chosen = selection["selected_lineage"]
        selected = bool(outcome["correct"][chosen])
        unchanged = bool(outcome["correct"]["unchanged"])
        revision = bool(outcome["correct"]["revision"])
        selected_correct += int(selected)
        unchanged_correct += int(unchanged)
        revision_correct += int(revision)
        retained += int(selected and unchanged)
        selected_counts[chosen] += 1
        local = domain[outcome["task"]]
        local["rows"] += 1
        local["selected_correct"] += int(selected)
        local["unchanged_correct"] += int(unchanged)
        local["revision_correct"] += int(revision)
        selected_only_unchanged += int(selected and not unchanged)
        unchanged_only_selected += int(unchanged and not selected)
        selected_only_revision += int(selected and not revision)
        revision_only_selected += int(revision and not selected)
    if unchanged_correct <= 0:
        raise CrossHostAnalysisError("cross-host unchanged correctness is empty")

    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "host": args.host,
        "rows": contract["rows"],
        "selected_correct": selected_correct,
        "unchanged_correct": unchanged_correct,
        "revision_correct": revision_correct,
        "selected_gain_over_unchanged": selected_correct - unchanged_correct,
        "selected_gain_over_revision": selected_correct - revision_correct,
        "unchanged_correct_retained": retained,
        "unchanged_correct_retention": retained / unchanged_correct,
        "selected_counts": dict(sorted(selected_counts.items())),
        "domains": {
            task: dict(sorted(values.items()))
            for task, values in sorted(domain.items())
        },
        "selected_vs_unchanged": {
            "selected_only_correct": selected_only_unchanged,
            "unchanged_only_correct": unchanged_only_selected,
            "net_correct": selected_only_unchanged - unchanged_only_selected,
            "mcnemar_exact_two_sided_p": mcnemar_exact(
                selected_only_unchanged, unchanged_only_selected
            ),
        },
        "selected_vs_revision": {
            "selected_only_correct": selected_only_revision,
            "revision_only_correct": revision_only_selected,
            "net_correct": selected_only_revision - revision_only_selected,
            "mcnemar_exact_two_sided_p": mcnemar_exact(
                selected_only_revision, revision_only_selected
            ),
        },
        "retention_at_least_95_percent": retained / unchanged_correct >= 0.95,
        "all_domains_nonnegative_vs_unchanged": all(
            values["selected_correct"] >= values["unchanged_correct"]
            for values in domain.values()
        ),
        "selection_had_score_or_assessor_access": False,
        "revision_margin_threshold": revision_margin_threshold,
        "application_report_sha256": sha256_file(args.application_report),
        "selections_sha256": sha256_file(args.selections),
        "score_sha256": sha256_file(args.score),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=tuple(HOSTS), required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--application-report", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision-margin-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(analyze(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
