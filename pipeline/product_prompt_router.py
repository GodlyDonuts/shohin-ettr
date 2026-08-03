#!/usr/bin/env python3
"""Train and apply a deterministic prompt-only B1/C2 expert router."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable


MODEL_SCHEMA = "shohin-product-prompt-router-v1"
ROUTED_REPORT_SCHEMA = "shohin-product-routed-report-v1"
WORD_RE = re.compile(r"[a-z]+|\d+|[^\s]", re.IGNORECASE)
TASKS_BY_DOMAIN = {
    "grade_school_math": ("gsm8k",),
    "competition_math": ("math500",),
    "code": ("humaneval", "mbpp"),
    "science": ("gpqa",),
    "logic": ("bbh_logic",),
}
PROCEDURAL_MATH_CUES = (
    "arithmetic problem",
    "calculate",
    "compute",
    "convert the base",
    "exponentiation",
    "factorization",
    "greatest common divisor",
    "how many 1 bits",
    "least common multiple",
    "multiplication",
    "prime numbers",
    "solve this problem",
)


class PromptRouterError(RuntimeError):
    """The prompt router contract cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise PromptRouterError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def prompt_features(text: str) -> set[str]:
    tokens = WORD_RE.findall(text.lower())[:2048]
    features = {f"u:{token}" for token in tokens}
    features.update(f"b:{left}\0{right}" for left, right in zip(tokens, tokens[1:]))
    return features


def training_label(row: dict[str, Any]) -> str | None:
    group = row.get("training_group")
    if group in {"math", "science"}:
        return "dense_residual"
    if group == "code":
        return "baseline"
    if group == "procedural":
        question = str(row.get("question", "")).lower()
        if any(cue in question for cue in PROCEDURAL_MATH_CUES):
            return "dense_residual"
        return "baseline"
    return None


def _read_training_rows(path: Path) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise PromptRouterError(f"training row {line_number} is not an object")
            question = row.get("question")
            if not isinstance(question, str) or not question:
                raise PromptRouterError(f"training row {line_number} lacks a question")
            label = training_label(row)
            if label is not None:
                yield question, label


def train_router(
    rows: Iterable[tuple[str, str]], *, min_feature_count: int, max_features: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    examples = list(rows)
    if not examples:
        raise PromptRouterError("no labeled router examples")
    labels = ("baseline", "dense_residual")
    train_docs = {label: 0 for label in labels}
    feature_docs = {label: collections.Counter() for label in labels}
    validation: list[tuple[str, str]] = []
    label_counts = collections.Counter(label for _, label in examples)
    if set(label_counts) != set(labels):
        raise PromptRouterError(f"both labels are required: {dict(label_counts)}")

    for question, label in examples:
        split = int(hashlib.sha256(question.encode()).hexdigest()[:8], 16) % 10
        if split == 0:
            validation.append((question, label))
            continue
        features = prompt_features(question)
        train_docs[label] += 1
        feature_docs[label].update(features)

    totals = feature_docs["baseline"] + feature_docs["dense_residual"]
    vocabulary = [
        feature
        for feature, count in totals.most_common()
        if count >= min_feature_count
    ][:max_features]
    if not vocabulary:
        raise PromptRouterError("feature pruning removed the vocabulary")
    alpha = 1.0
    weights: dict[str, float] = {}
    for feature in vocabulary:
        baseline_probability = (feature_docs["baseline"][feature] + alpha) / (
            train_docs["baseline"] + 2 * alpha
        )
        dense_probability = (feature_docs["dense_residual"][feature] + alpha) / (
            train_docs["dense_residual"] + 2 * alpha
        )
        weights[feature] = math.log(dense_probability / baseline_probability)

    model = {
        "decision_threshold": 0.0,
        "feature_count": len(weights),
        "feature_weights": weights,
        "label_policy": {
            "baseline": ["code", "non-arithmetic procedural"],
            "dense_residual": ["math", "science", "arithmetic procedural"],
            "excluded": ["teacher"],
        },
        "max_features": max_features,
        "min_feature_count": min_feature_count,
        "schema": MODEL_SCHEMA,
        "train_documents": train_docs,
    }
    metrics = evaluate_router(model, validation)
    report = {
        "input_examples": len(examples),
        "label_counts": dict(label_counts),
        "model": {
            key: value for key, value in model.items() if key != "feature_weights"
        },
        "status": "complete",
        "validation": metrics,
    }
    return model, report


def route_score(model: dict[str, Any], question: str) -> float:
    if model.get("schema") != MODEL_SCHEMA:
        raise PromptRouterError("router model schema differs")
    weights = model.get("feature_weights")
    if not isinstance(weights, dict):
        raise PromptRouterError("router weights are missing")
    return sum(float(weights.get(feature, 0.0)) for feature in prompt_features(question))


def route(model: dict[str, Any], question: str) -> tuple[str, float]:
    score = route_score(model, question)
    threshold = float(model.get("decision_threshold", 0.0))
    return ("dense_residual" if score >= threshold else "baseline"), score


def evaluate_router(
    model: dict[str, Any], examples: Iterable[tuple[str, str]]
) -> dict[str, Any]:
    confusion = {
        gold: {predicted: 0 for predicted in ("baseline", "dense_residual")}
        for gold in ("baseline", "dense_residual")
    }
    total = correct = 0
    for question, gold in examples:
        predicted, _ = route(model, question)
        confusion[gold][predicted] += 1
        total += 1
        correct += int(predicted == gold)
    return {
        "accuracy": correct / total if total else None,
        "confusion": confusion,
        "correct": correct,
        "total": total,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PromptRouterError(f"missing JSON artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PromptRouterError(f"JSON artifact is not an object: {path}")
    return value


def route_reports(
    model: dict[str, Any], baseline: dict[str, Any], dense: dict[str, Any]
) -> dict[str, Any]:
    baseline_results = baseline.get("results")
    dense_results = dense.get("results")
    if not isinstance(baseline_results, list) or not isinstance(dense_results, list):
        raise PromptRouterError("input reports do not contain result lists")
    dense_by_id = {row["identity_sha256"]: row for row in dense_results}
    if len(dense_by_id) != len(dense_results):
        raise PromptRouterError("dense report contains duplicate identities")
    if {row["identity_sha256"] for row in baseline_results} != set(dense_by_id):
        raise PromptRouterError("report identity sets differ")

    selected: list[dict[str, Any]] = []
    route_counts = collections.Counter()
    for baseline_row in baseline_results:
        identity = baseline_row["identity_sha256"]
        dense_row = dense_by_id[identity]
        for field in ("question", "gold"):
            if baseline_row.get(field) != dense_row.get(field):
                raise PromptRouterError(f"report {field} differs for {identity}")
        selected_arm, score = route(model, str(baseline_row["question"]))
        source = dense_row if selected_arm == "dense_residual" else baseline_row
        row = dict(source)
        row["router_score"] = score
        row["selected_arm"] = selected_arm
        selected.append(row)
        route_counts[selected_arm] += 1

    correct = sum(bool(row.get("correct")) for row in selected)
    return {
        "accuracy": correct / len(selected),
        "correct": correct,
        "model_schema": model["schema"],
        "results": selected,
        "route_counts": dict(route_counts),
        "schema": ROUTED_REPORT_SCHEMA,
        "status": "complete",
        "task": baseline.get("task"),
        "total": len(selected),
    }


def _matched_rows(
    baseline: dict[str, Any], dense: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    baseline_results = baseline.get("results")
    dense_results = dense.get("results")
    if not isinstance(baseline_results, list) or not isinstance(dense_results, list):
        raise PromptRouterError("input reports do not contain result lists")
    dense_by_id = {row["identity_sha256"]: row for row in dense_results}
    if len(dense_by_id) != len(dense_results):
        raise PromptRouterError("dense report contains duplicate identities")
    if {row["identity_sha256"] for row in baseline_results} != set(dense_by_id):
        raise PromptRouterError("report identity sets differ")
    result = []
    for baseline_row in baseline_results:
        dense_row = dense_by_id[baseline_row["identity_sha256"]]
        if baseline_row.get("question") != dense_row.get("question"):
            raise PromptRouterError("paired report questions differ")
        result.append((baseline_row, dense_row))
    return result


def calibrate_threshold(
    model: dict[str, Any], reports: dict[str, tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_tasks = {
        task for tasks in TASKS_BY_DOMAIN.values() for task in tasks
    }
    if set(reports) != expected_tasks:
        raise PromptRouterError(
            f"calibration tasks differ: {sorted(reports)} != {sorted(expected_tasks)}"
        )
    examples: dict[str, list[tuple[float, bool, bool]]] = {}
    scores: list[float] = []
    for task, (baseline, dense) in reports.items():
        examples[task] = []
        for baseline_row, dense_row in _matched_rows(baseline, dense):
            score = route_score(model, str(baseline_row["question"]))
            examples[task].append(
                (score, bool(baseline_row["correct"]), bool(dense_row["correct"]))
            )
            scores.append(score)
    if not scores:
        raise PromptRouterError("calibration reports are empty")

    def summary(threshold: float) -> dict[str, Any]:
        tasks: dict[str, dict[str, int]] = {}
        for task, rows in examples.items():
            correct = sum(dense if score >= threshold else baseline for score, baseline, dense in rows)
            tasks[task] = {"correct": correct, "total": len(rows)}
        domains = {}
        for domain, domain_tasks in TASKS_BY_DOMAIN.items():
            correct = sum(tasks[task]["correct"] for task in domain_tasks)
            total = sum(tasks[task]["total"] for task in domain_tasks)
            domains[domain] = {"accuracy": correct / total, "correct": correct, "total": total}
        return {
            "domains": domains,
            "macro_accuracy": sum(value["accuracy"] for value in domains.values()) / len(domains),
            "solved": sum(value["correct"] for value in tasks.values()),
            "tasks": tasks,
            "threshold": threshold,
        }

    baseline_summary = summary(max(scores) + 1.0)
    candidates = sorted(set(scores + [max(scores) + 1.0]))
    eligible = []
    for threshold in candidates:
        candidate = summary(threshold)
        maximum_regression = max(
            baseline_summary["domains"][domain]["accuracy"]
            - candidate["domains"][domain]["accuracy"]
            for domain in TASKS_BY_DOMAIN
        )
        candidate["maximum_domain_regression"] = maximum_regression
        if maximum_regression <= 0.02:
            eligible.append(candidate)
    if not eligible:
        raise PromptRouterError("no calibration threshold satisfies the regression gate")
    selected = max(
        eligible,
        key=lambda value: (
            value["macro_accuracy"],
            value["solved"],
            value["threshold"],
        ),
    )
    calibrated = dict(model)
    calibrated["decision_threshold"] = selected["threshold"]
    calibrated["calibration"] = {
        "objective": "maximum five-domain development macro",
        "regression_gate": "no domain more than 0.02 below all-baseline",
        "selected": selected,
    }
    report = {
        "all_baseline": baseline_summary,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected": selected,
        "status": "complete",
    }
    return calibrated, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--model", required=True, type=Path)
    train.add_argument("--report", required=True, type=Path)
    train.add_argument("--min-feature-count", type=int, default=3)
    train.add_argument("--max-features", type=int, default=50000)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--model", required=True, type=Path)
    calibrate.add_argument("--manifest", required=True, type=Path)
    calibrate.add_argument("--output-model", required=True, type=Path)
    calibrate.add_argument("--report", required=True, type=Path)
    apply = subparsers.add_parser("route-reports")
    apply.add_argument("--model", required=True, type=Path)
    apply.add_argument("--baseline", required=True, type=Path)
    apply.add_argument("--dense", required=True, type=Path)
    apply.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "train":
        if not args.data.is_file():
            raise PromptRouterError(f"missing training data: {args.data}")
        model, report = train_router(
            _read_training_rows(args.data),
            min_feature_count=args.min_feature_count,
            max_features=args.max_features,
        )
        report["data"] = str(args.data.resolve())
        report["data_sha256"] = _sha256(args.data)
        _atomic_json(args.model, model)
        report["model_sha256"] = _sha256(args.model)
        _atomic_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "calibrate":
        model = _load_json(args.model)
        manifest = _load_json(args.manifest)
        report_pairs = {}
        report_files = {}
        for task, paths in manifest.items():
            if not isinstance(paths, dict):
                raise PromptRouterError(f"calibration manifest entry is invalid: {task}")
            baseline_path = Path(paths["baseline"])
            dense_path = Path(paths["dense_residual"])
            report_pairs[task] = (_load_json(baseline_path), _load_json(dense_path))
            report_files[task] = {
                "baseline": {"path": str(baseline_path.resolve()), "sha256": _sha256(baseline_path)},
                "dense_residual": {"path": str(dense_path.resolve()), "sha256": _sha256(dense_path)},
            }
        calibrated, report = calibrate_threshold(model, report_pairs)
        calibrated["parent_model_sha256"] = _sha256(args.model)
        _atomic_json(args.output_model, calibrated)
        report["calibrated_model_sha256"] = _sha256(args.output_model)
        report["manifest_sha256"] = _sha256(args.manifest)
        report["report_files"] = report_files
        _atomic_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    model = _load_json(args.model)
    routed = route_reports(
        model,
        _load_json(args.baseline),
        _load_json(args.dense),
    )
    routed["baseline_report_sha256"] = _sha256(args.baseline)
    routed["dense_report_sha256"] = _sha256(args.dense)
    routed["router_model_sha256"] = _sha256(args.model)
    _atomic_json(args.output, routed)
    print(json.dumps({key: routed[key] for key in ("accuracy", "correct", "route_counts", "task", "total")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
