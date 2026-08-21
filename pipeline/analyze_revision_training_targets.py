#!/usr/bin/env python3
"""Audit response-horizon geometry in an immutable revision-training JSONL."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
from typing import Any, Callable

REPORT_SCHEMA = "shohin-revision-training-target-horizon-analysis-v2"
HEX64 = re.compile(r"[0-9a-f]{64}")
EXACT_BOXED = re.compile(r"\\boxed\{.*\}", re.DOTALL)
LENGTH_THRESHOLDS = (20, 80, 100, 256)
DRAFT_MARKER = "Internal draft:\n"
FINAL_PROBLEM_MARKER = "\n\nOriginal problem:"
FORMAT_MARKERS = (
    "\n\nFollow the original problem's requested output format.",
    "\n\nReturn ",
)


class RevisionTrainingTargetError(RuntimeError):
    """The training artifact differs from the target-horizon contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path, expected_schema: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RevisionTrainingTargetError(
                    f"training row {line_number} is not JSON"
                ) from exc
            if not isinstance(row, dict):
                raise RevisionTrainingTargetError(
                    f"training row {line_number} is not an object"
                )
            required_strings = (
                "identity_sha256",
                "source_identity_sha256",
                "target_kind",
                "outcome_class",
                "question",
                "response",
            )
            if (
                row.get("schema") != expected_schema
                or any(not isinstance(row.get(key), str) for key in required_strings)
                or HEX64.fullmatch(row["identity_sha256"]) is None
                or HEX64.fullmatch(row["source_identity_sha256"]) is None
                or not row["target_kind"]
                or not row["outcome_class"]
                or not row["question"]
                or not row["response"]
                or isinstance(row.get("presentation"), bool)
                or not isinstance(row.get("presentation"), int)
                or row["presentation"] < 0
                or row.get("internal_draft_visible") is not True
                or row.get("external_candidate_text_visible") is not False
            ):
                raise RevisionTrainingTargetError(
                    f"training row {line_number} differs from revision schema"
                )
            rows.append(row)
    if not rows:
        raise RevisionTrainingTargetError("training artifact is empty")
    identities = [row["identity_sha256"] for row in rows]
    if len(set(identities)) != len(identities):
        raise RevisionTrainingTargetError("training identity is duplicated")
    return rows


def _length_summary(lengths: list[int]) -> dict[str, Any]:
    return {
        "measurement": "unicode_code_points_before_whitespace_stripping",
        "total": sum(lengths),
        "minimum": min(lengths),
        "maximum": max(lengths),
        "mean": math.fsum(lengths) / len(lengths),
        "median": float(statistics.median(lengths)),
        "below_threshold": {
            str(threshold): sum(length < threshold for length in lengths)
            for threshold in LENGTH_THRESHOLDS
        },
    }


def _internal_draft(question: str) -> str:
    marker_start = question.find(DRAFT_MARKER)
    if marker_start < 0:
        raise RevisionTrainingTargetError("training prompt has no internal draft")
    draft_start = marker_start + len(DRAFT_MARKER)
    final_problem = question.rfind(FINAL_PROBLEM_MARKER)
    if final_problem <= draft_start:
        raise RevisionTrainingTargetError("training prompt lacks repeated problem")
    draft_end = max(
        question.rfind(marker, draft_start, final_problem) for marker in FORMAT_MARKERS
    )
    if draft_end <= draft_start:
        raise RevisionTrainingTargetError("training prompt lacks format instruction")
    return question[draft_start:draft_end]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    response_lengths = [len(row["response"]) for row in rows]
    draft_lengths = [len(_internal_draft(row["question"])) for row in rows]
    ratios = [
        response / max(1, draft)
        for response, draft in zip(response_lengths, draft_lengths, strict=True)
    ]
    responses = [row["response"].strip() for row in rows]
    source_counts = Counter(row["source_identity_sha256"] for row in rows)
    return {
        "rows": len(rows),
        "unique_source_identity_sha256": len(source_counts),
        "presentations_per_source_counts": {
            str(presentations): count
            for presentations, count in sorted(Counter(source_counts.values()).items())
        },
        "response_characters": _length_summary(response_lengths),
        "internal_draft_characters": _length_summary(draft_lengths),
        "response_to_internal_draft_character_ratio": {
            "mean": math.fsum(ratios) / len(ratios),
            "median": float(statistics.median(ratios)),
            "minimum": min(ratios),
            "maximum": max(ratios),
        },
        "contains_think_open_tag": sum("<think>" in response for response in responses),
        "contains_boxed_answer": sum("\\boxed{" in response for response in responses),
        "exact_boxed_response": sum(
            EXACT_BOXED.fullmatch(response) is not None for response in responses
        ),
    }


def analyze(path: Path, expected_sha256: str, expected_schema: str) -> dict[str, Any]:
    if HEX64.fullmatch(expected_sha256) is None:
        raise RevisionTrainingTargetError("expected SHA-256 is not canonical")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RevisionTrainingTargetError("training SHA-256 differs")
    rows = _load_rows(path, expected_schema)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross_tab: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_target[row["target_kind"]].append(row)
        by_outcome[row["outcome_class"]].append(row)
        cross_tab[row["target_kind"]][row["outcome_class"]] += 1
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "input": {
            "path": str(path.resolve()),
            "sha256": actual_sha256,
            "schema": expected_schema,
            "rows": len(rows),
            "unique_identity_sha256": len(rows),
            "unique_source_identity_sha256": len(
                {row["source_identity_sha256"] for row in rows}
            ),
            "presentation_counts": dict(
                sorted(Counter(row["presentation"] for row in rows).items())
            ),
            "internal_draft_visible_rows": len(rows),
            "external_candidate_text_visible_rows": 0,
        },
        "overall": _summary(rows),
        "by_target_kind": {
            name: _summary(group) for name, group in sorted(by_target.items())
        },
        "by_outcome_class": {
            name: _summary(group) for name, group in sorted(by_outcome.items())
        },
        "target_kind_by_outcome_class": {
            target: dict(sorted(counts.items()))
            for target, counts in sorted(cross_tab.items())
        },
    }


def analyze_loss_geometry(
    rows: list[dict[str, Any]],
    *,
    encoded_response_length: Callable[[str], int],
    data_seed: int,
    microsteps: int,
    expected_charged_tokens: int,
) -> dict[str, Any]:
    if microsteps <= 0 or microsteps > len(rows):
        raise RevisionTrainingTargetError("training microstep geometry differs")
    selected = list(rows)
    random.Random(data_seed).shuffle(selected)
    selected = selected[:microsteps]
    lengths = [encoded_response_length(row["response"]) + 1 for row in selected]
    if any(length < 2 for length in lengths) or sum(lengths) != expected_charged_tokens:
        raise RevisionTrainingTargetError("charged response-token geometry differs")

    def summarize(group: list[tuple[dict[str, Any], int]]) -> dict[str, Any]:
        group_lengths = [length for _, length in group]
        return {
            "rows": len(group),
            "row_fraction": len(group) / len(selected),
            "unique_source_identity_sha256": len(
                {row["source_identity_sha256"] for row, _ in group}
            ),
            "response_tokens_including_terminal_eos": {
                "total": sum(group_lengths),
                "fraction_of_charged_tokens": sum(group_lengths) / sum(lengths),
                "mean": math.fsum(group_lengths) / len(group_lengths),
                "median": float(statistics.median(group_lengths)),
                "minimum": min(group_lengths),
                "maximum": max(group_lengths),
            },
            "terminal_eos_tokens": len(group),
            "mean_terminal_eos_fraction_of_row_loss": math.fsum(
                1.0 / length for length in group_lengths
            )
            / len(group_lengths),
        }

    paired = list(zip(selected, lengths, strict=True))
    by_target: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(list)
    by_outcome: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for row, length in paired:
        by_target[row["target_kind"]].append((row, length))
        by_outcome[row["outcome_class"]].append((row, length))
    result = {
        "data_seed": data_seed,
        "microsteps": microsteps,
        "charged_tokens": sum(lengths),
        "terminal_eos_tokens": microsteps,
        "mean_terminal_eos_fraction_of_row_loss": math.fsum(
            1.0 / length for length in lengths
        )
        / len(lengths),
        "loss_reduction": "batch_one_mean_over_response_tokens_including_terminal_eos_then_equal_gradient_accumulation_scaling",
        "by_target_kind": {
            name: summarize(group) for name, group in sorted(by_target.items())
        },
        "by_outcome_class": {
            name: summarize(group) for name, group in sorted(by_outcome.items())
        },
    }
    short_eos = result["by_target_kind"]["source_verified_repair"][
        "mean_terminal_eos_fraction_of_row_loss"
    ]
    full_eos = result["by_target_kind"]["verified_candidate"][
        "mean_terminal_eos_fraction_of_row_loss"
    ]
    result["source_repair_to_verified_candidate_eos_density_ratio"] = (
        short_eos / full_eos
    )
    return result


def _training_loss_geometry(
    args: argparse.Namespace, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    paths = (args.tokenizer_json, args.tokenizer_config, args.training_report)
    hashes = (
        args.expected_tokenizer_sha256,
        args.expected_tokenizer_config_sha256,
        args.expected_training_report_sha256,
    )
    if any(value is None for value in (*paths, *hashes)):
        raise RevisionTrainingTargetError("loss-geometry inputs are incomplete")
    for path, expected in zip(paths, hashes, strict=True):
        if sha256_file(path) != expected:
            raise RevisionTrainingTargetError("loss-geometry input SHA-256 differs")
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    if (
        training.get("schema") != "shohin-hf-product-reasoning-training-v1"
        or training.get("status") != "complete"
        or training.get("data_sha256") != args.expected_sha256
        or training.get("selected_rows") != len(rows)
        or training.get("batch_size") != 1
        or not isinstance(training.get("updates"), int)
        or not isinstance(training.get("gradient_accumulation"), int)
        or not isinstance(training.get("charged_tokens"), int)
        or not isinstance(training.get("data_seed"), int)
    ):
        raise RevisionTrainingTargetError("training report geometry differs")
    tokenizer_config = json.loads(args.tokenizer_config.read_text(encoding="utf-8"))
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    eos_token = tokenizer_config.get("eos_token")
    eos_token_id = (
        tokenizer.token_to_id(eos_token) if isinstance(eos_token, str) else None
    )
    if eos_token_id != args.expected_eos_token_id:
        raise RevisionTrainingTargetError("terminal EOS identity differs")
    geometry = analyze_loss_geometry(
        rows,
        encoded_response_length=lambda response: len(
            tokenizer.encode(response, add_special_tokens=False).ids
        ),
        data_seed=training["data_seed"],
        microsteps=training["updates"] * training["gradient_accumulation"],
        expected_charged_tokens=training["charged_tokens"],
    )
    geometry.update(
        {
            "training_report": str(args.training_report.resolve()),
            "training_report_sha256": args.expected_training_report_sha256,
            "updates": training["updates"],
            "batch_size": training["batch_size"],
            "gradient_accumulation": training["gradient_accumulation"],
            "tokenizer_json": str(args.tokenizer_json.resolve()),
            "tokenizer_sha256": args.expected_tokenizer_sha256,
            "tokenizer_config": str(args.tokenizer_config.resolve()),
            "tokenizer_config_sha256": args.expected_tokenizer_config_sha256,
            "terminal_eos_token": eos_token,
            "terminal_eos_token_id": eos_token_id,
        }
    )
    return geometry


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RevisionTrainingTargetError("refusing to replace target analysis")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-schema", required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-config-sha256", required=True)
    parser.add_argument("--expected-eos-token-id", type=int, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--expected-training-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze(args.input, args.expected_sha256, args.expected_schema)
    rows = _load_rows(args.input, args.expected_schema)
    report["training_loss_geometry"] = _training_loss_geometry(args, rows)
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
