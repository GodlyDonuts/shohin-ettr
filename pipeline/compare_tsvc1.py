#!/usr/bin/env python3
"""Apply the prospectively frozen TSVC1 semantic-commit gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


SCHEMA = "shohin-tsvc1-comparison-v1"


class TSVC1ComparisonError(RuntimeError):
    """TSVC1 result inputs violate the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    aligned = load(args.aligned)
    shuffled = load(args.shuffled)
    shape = load(args.shape)
    candidates = {}
    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates[(str(row["identity_sha256"]), int(row["sample_index"]))] = row
    selected_rows = aligned.get("results")
    if not isinstance(selected_rows, list) or len(selected_rows) != 125:
        raise TSVC1ComparisonError("aligned selection rows differ")
    family_correct = {"choice_final": 0, "numeric_final": 0}
    member_correct = {"clean": 0, "fault": 0}
    for result in selected_rows:
        key = (str(result["identity_sha256"]), int(result["selected_sample_index"]))
        candidate = candidates.get(key)
        if candidate is None:
            raise TSVC1ComparisonError("selected candidate is absent")
        if bool(result["selected_correct"]) != bool(candidate["correct"]):
            raise TSVC1ComparisonError("selection correctness metadata differs")
        if bool(candidate["correct"]):
            family_correct[str(candidate["corruption_family"])] += 1
            member_correct[str(candidate["pair_member"])] += 1

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    validation = model.get("validation_metrics")
    if not isinstance(validation, dict):
        raise TSVC1ComparisonError("head validation metrics are missing")
    feature_reports = [load(path) for path in args.feature_reports]
    truncation = sum(int(report["prompt_truncated"]) for report in feature_reports)
    aligned_correct = int(aligned["selected_correct"])
    shuffled_correct = int(shuffled["selected_correct"])
    shape_correct = int(shape["selected_correct"])
    overall = 1769 + aligned_correct
    choice = 125 + family_correct["choice_final"]
    numeric = 1644 + family_correct["numeric_final"]
    clean = 900 + member_correct["clean"]
    fault = 869 + member_correct["fault"]
    gate = {
        "zero_feature_truncation": truncation == 0,
        "aligned_disagreement_ge_105": aligned_correct >= 105,
        "overall_ge_1874": overall >= 1874,
        "choice_ge_220": choice >= 220,
        "aligned_minus_shuffled_ge_13": aligned_correct - shuffled_correct >= 13,
        "aligned_minus_shape_ge_13": aligned_correct - shape_correct >= 13,
        "train_validation_ge_0_90": float(validation["selected_accuracy"]) >= 0.90,
    }
    payload = {
        "schema": SCHEMA,
        "status": "pass" if all(gate.values()) else "fail",
        "gate": gate,
        "disagreement_groups": 125,
        "aligned_selected_correct": aligned_correct,
        "shuffled_selected_correct": shuffled_correct,
        "shape_selected_correct": shape_correct,
        "aligned_minus_shuffled": aligned_correct - shuffled_correct,
        "aligned_minus_shape": aligned_correct - shape_correct,
        "overall_correct": overall,
        "overall_total": 1908,
        "choice_correct": choice,
        "choice_total": 256,
        "numeric_correct": numeric,
        "numeric_total": 1652,
        "clean_correct": clean,
        "clean_total": 954,
        "fault_correct": fault,
        "fault_total": 954,
        "train_validation": validation,
        "feature_prompt_truncation": truncation,
        "inputs": {
            "aligned": sha256_file(args.aligned),
            "shuffled": sha256_file(args.shuffled),
            "shape": sha256_file(args.shape),
            "model": sha256_file(args.model),
            "candidates": sha256_file(args.candidates),
            "feature_reports": [sha256_file(path) for path in args.feature_reports],
        },
        "holdout_authorized": all(gate.values()),
        "holdout_used": False,
    }
    if args.output.exists():
        raise TSVC1ComparisonError("refusing existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--shape", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--feature-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
