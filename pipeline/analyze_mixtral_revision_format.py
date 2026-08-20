#!/usr/bin/env python3
"""Audit the completed Mixtral validation for temporal format collapse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

SCORE_SCHEMA = "shohin-mixtral-8x22b-selective-commit-validation-score-v1"
CANDIDATE_SCHEMA = "shohin-mixtral-8x22b-fixed-draft-candidate-v1"
REPORT_SCHEMA = "shohin-mixtral-revision-format-analysis-v1"
ARMS = ("unchanged", "self_refinement", "revision")
SCORED_ARMS = (*ARMS, "selective_commit")
TASKS = ("bbh_logic", "math500", "mbpp")
EXPECTED_ROWS = 1023
EXPECTED_TASK_COUNTS = {"bbh_logic": 497, "math500": 504, "mbpp": 22}
EXPECTED_SHARDS = 16
EXPECTED_SCORE_SHA256 = (
    "7befd864dd921ec371c175381b4eccec9f0d603bf291eaddf93fc5039043c3d8"
)
EXPECTED_CANDIDATE_RECEIPTS_SHA256 = (
    "2198f2d2e26bb3dad73588916339fb2c06a67eaa3a4ea89c46ac73ef9954c609"
)


class FormatAnalysisError(RuntimeError):
    """The scored validation or candidate projection differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FormatAnalysisError(f"unsafe or missing JSON: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise FormatAnalysisError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FormatAnalysisError(f"unsafe or missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FormatAnalysisError(f"row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def task_summary(
    rows: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    arm: str,
) -> dict[str, Any]:
    tokens = [row["generated_tokens"] for row in rows]
    completions = [row["completion"] for row in rows]
    return {
        "rows": len(rows),
        "correct": sum(
            int(outcomes[row["identity_sha256"]]["correct"][arm]) for row in rows
        ),
        "generated_tokens_sum": sum(tokens),
        "generated_tokens_mean": sum(tokens) / len(tokens),
        "generated_tokens_median": statistics.median(tokens),
        "boxed_completions": sum("\\boxed" in value for value in completions),
        "code_fenced_completions": sum("```" in value for value in completions),
        "function_definition_completions": sum(
            "def " in value for value in completions
        ),
        "return_statement_completions": sum("return" in value for value in completions),
        "unique_completions": len(set(completions)),
    }


def analyze(
    score_path: Path,
    candidates_root: Path,
    *,
    expected_score_sha256: str = EXPECTED_SCORE_SHA256,
    expected_candidate_receipts_sha256: str = EXPECTED_CANDIDATE_RECEIPTS_SHA256,
    expected_rows: int = EXPECTED_ROWS,
    expected_task_counts: dict[str, int] = EXPECTED_TASK_COUNTS,
    expected_shards: int = EXPECTED_SHARDS,
) -> dict[str, Any]:
    if sha256_file(score_path) != expected_score_sha256:
        raise FormatAnalysisError("score SHA-256 differs")
    score = load_json(score_path)
    raw_outcomes = score.get("outcomes")
    if (
        score.get("schema") != SCORE_SCHEMA
        or score.get("status") != "complete"
        or score.get("rows") != expected_rows
        or not isinstance(raw_outcomes, list)
        or len(raw_outcomes) != expected_rows
    ):
        raise FormatAnalysisError("score contract differs")
    outcomes: dict[str, dict[str, Any]] = {}
    ordered_identities: list[str] = []
    observed_task_counts = {task: 0 for task in TASKS}
    for row in raw_outcomes:
        identity = row.get("identity_sha256")
        task = row.get("task")
        correct = row.get("correct")
        selected_arm = row.get("selected_arm")
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or identity in outcomes
            or task not in TASKS
            or not isinstance(correct, dict)
            or set(correct) != set(SCORED_ARMS)
            or any(not isinstance(correct.get(arm), bool) for arm in SCORED_ARMS)
            or selected_arm not in ARMS
            or correct["selective_commit"] != correct[selected_arm]
        ):
            raise FormatAnalysisError("outcome row differs")
        outcomes[identity] = row
        ordered_identities.append(identity)
        observed_task_counts[task] += 1
    if observed_task_counts != expected_task_counts:
        raise FormatAnalysisError("outcome task counts differ")
    observed_selection_counts = {
        arm: sum(row["selected_arm"] == arm for row in raw_outcomes) for arm in ARMS
    }
    if (
        score.get("selection_counts") != observed_selection_counts
        or score.get("selective_commit_score_derived_from_selected_scored_arm")
        is not True
        or score.get("task_label_used_as_commit_feature") is not False
    ):
        raise FormatAnalysisError("selective-commit projection differs")

    if (
        not candidates_root.is_dir()
        or candidates_root.is_symlink()
        or sorted(path.name for path in candidates_root.iterdir()) != sorted(ARMS)
    ):
        raise FormatAnalysisError("candidate root differs")

    arm_rows: dict[str, list[dict[str, Any]]] = {}
    file_receipts: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_root = candidates_root / arm
        shard_names = [f"shard_{index:02d}" for index in range(expected_shards)]
        if (
            not arm_root.is_dir()
            or arm_root.is_symlink()
            or sorted(path.name for path in arm_root.iterdir()) != shard_names
        ):
            raise FormatAnalysisError(f"{arm} shard geometry differs")
        rows: list[dict[str, Any]] = []
        for shard_name in shard_names:
            shard_root = arm_root / shard_name
            if (
                not shard_root.is_dir()
                or shard_root.is_symlink()
                or sorted(path.name for path in shard_root.iterdir())
                != ["candidates.jsonl", "report.json"]
            ):
                raise FormatAnalysisError(f"{arm}/{shard_name} contents differ")
            candidate_path = shard_root / "candidates.jsonl"
            shard_rows = load_jsonl(candidate_path)
            file_receipts.append(
                {
                    "path": f"{arm}/{shard_name}/candidates.jsonl",
                    "rows": len(shard_rows),
                    "sha256": sha256_file(candidate_path),
                }
            )
            rows.extend(shard_rows)
        if len(rows) != expected_rows:
            raise FormatAnalysisError(f"{arm} row count differs")
        seen: set[str] = set()
        for row in rows:
            identity = row.get("identity_sha256")
            generated_tokens = row.get("generated_tokens")
            if (
                row.get("schema") != CANDIDATE_SCHEMA
                or row.get("arm") != arm
                or identity not in outcomes
                or identity in seen
                or row.get("task") != outcomes[identity]["task"]
                or not isinstance(row.get("completion"), str)
                or not isinstance(generated_tokens, int)
                or isinstance(generated_tokens, bool)
                or generated_tokens < 0
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise FormatAnalysisError(f"{arm} candidate row differs")
            seen.add(identity)
        if [row["identity_sha256"] for row in rows] != ordered_identities:
            raise FormatAnalysisError(f"{arm} ordered identity projection differs")
        arm_rows[arm] = rows

    candidate_receipts_sha256 = canonical_sha256(file_receipts)
    if candidate_receipts_sha256 != expected_candidate_receipts_sha256:
        raise FormatAnalysisError("candidate projection SHA-256 differs")

    metrics: dict[str, Any] = {}
    for arm in ARMS:
        metrics[arm] = {
            task: task_summary(
                [row for row in arm_rows[arm] if row["task"] == task], outcomes, arm
            )
            for task in TASKS
        }
        metrics[arm]["all"] = task_summary(arm_rows[arm], outcomes, arm)

    revision = metrics["revision"]
    revision_code = revision["mbpp"]
    if (
        revision["all"]["boxed_completions"] != expected_rows
        or revision_code["correct"] != 0
        or revision_code["code_fenced_completions"] != 0
        or revision_code["function_definition_completions"] != 0
        or revision_code["return_statement_completions"] != 0
    ):
        raise FormatAnalysisError("frozen revision-format observation differs")

    selection_by_task: dict[str, Any] = {}
    revision_transitions: dict[str, Any] = {}
    for task in (*TASKS, "all"):
        rows = (
            raw_outcomes
            if task == "all"
            else [row for row in raw_outcomes if row["task"] == task]
        )
        selection_by_task[task] = {
            arm: sum(row["selected_arm"] == arm for row in rows) for arm in ARMS
        }
        revision_transitions[task] = {
            "rows": len(rows),
            "unchanged_correct": sum(row["correct"]["unchanged"] for row in rows),
            "revision_correct": sum(row["correct"]["revision"] for row in rows),
            "paired_wins": sum(
                not row["correct"]["unchanged"] and row["correct"]["revision"]
                for row in rows
            ),
            "paired_losses": sum(
                row["correct"]["unchanged"] and not row["correct"]["revision"]
                for row in rows
            ),
            "unchanged_correct_retained": sum(
                row["correct"]["unchanged"] and row["correct"]["revision"]
                for row in rows
            ),
        }

    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "date": "2026-08-18",
        "source": {
            "score_sha256": expected_score_sha256,
            "score_schema": SCORE_SCHEMA,
            "rows": expected_rows,
            "task_counts": expected_task_counts,
            "candidate_files": len(file_receipts),
            "candidate_file_receipts_sha256": candidate_receipts_sha256,
            "ordered_identity_replay": "pass",
        },
        "metrics": metrics,
        "selection": {
            "by_task": selection_by_task,
            "task_label_used_as_commit_feature": False,
        },
        "revision_transitions": revision_transitions,
        "finding": {
            "revision_all_rows_boxed": True,
            "revision_mbpp_correct": 0,
            "revision_mbpp_rows": revision_code["rows"],
            "revision_mbpp_median_generated_tokens": revision_code[
                "generated_tokens_median"
            ],
            "revision_mbpp_code_fenced": revision_code["code_fenced_completions"],
            "revision_mbpp_function_definitions": revision_code[
                "function_definition_completions"
            ],
            "unchanged_mbpp_correct": metrics["unchanged"]["mbpp"]["correct"],
            "unchanged_mbpp_code_fenced": metrics["unchanged"]["mbpp"][
                "code_fenced_completions"
            ],
            "unchanged_mbpp_function_definitions": metrics["unchanged"]["mbpp"][
                "function_definition_completions"
            ],
            "commit_selected_unchanged_for_all_mbpp": selection_by_task["mbpp"]
            == {"unchanged": 22, "self_refinement": 0, "revision": 0},
            "commit_math_unchanged_selections": selection_by_task["math500"][
                "unchanged"
            ],
            "interpretation": (
                "The revision surface learned aggressive answer extraction across "
                "all domains. That compression helps logic and mathematics but "
                "systematically violates the executable-code output contract."
            ),
            "architecture_implication": (
                "Large-host revision needs model-owned output-contract preservation; "
                "aggregate capability and conservative modality retention are distinct."
            ),
        },
        "claim_boundary": {
            "new_model_execution": 0,
            "new_candidate_scoring": 0,
            "existing_scored_artifacts_only": True,
            "universal_revision_claim": False,
        },
    }


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_exclusive(args.output, analyze(args.score, args.candidates_root))
    print(args.output)


if __name__ == "__main__":
    main()
