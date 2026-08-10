#!/usr/bin/env python3
"""Apply the frozen ETV1 whole-trajectory verifier gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-etv1-comparison-v1"


class ETV1ComparisonError(RuntimeError):
    """ETV1 reports or candidate custody differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    fit = load(args.fit)
    aligned = load(args.aligned)
    shuffled = load(args.shuffled)
    shape = load(args.shape)
    candidates = {}
    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                candidates[(str(row["identity_sha256"]), int(row["sample_index"]))] = row
    selections = aligned.get("selections")
    if not isinstance(selections, list) or len(selections) != 125:
        raise ETV1ComparisonError("aligned selections differ")
    family = {"choice_final": 0, "numeric_final": 0}
    member = {"clean": 0, "fault": 0}
    for selection in selections:
        key = (
            str(selection["identity_sha256"]),
            int(selection["process_sample_index"]),
        )
        candidate = candidates.get(key)
        if candidate is None:
            raise ETV1ComparisonError("selected candidate is missing")
        if bool(selection["process_correct"]) != bool(candidate["correct"]):
            raise ETV1ComparisonError("selection correctness differs")
        if bool(candidate["correct"]):
            family[str(candidate["corruption_family"])] += 1
            member[str(candidate["pair_member"])] += 1
    aligned_metrics = aligned["metrics"]
    shuffled_metrics = shuffled["metrics"]
    aligned_correct = int(aligned_metrics["overall"]["process_correct"])
    shuffled_correct = int(shuffled_metrics["overall"]["process_correct"])
    shape_correct = int(shape["selected_correct"])
    internal_final = fit["final_metrics"]["overall"]
    truncation = (
        int(fit["training_prompt_truncated"])
        + int(fit["dev_metrics"]["prompt_truncated"])
        + int(fit["final_metrics"]["prompt_truncated"])
        + int(aligned_metrics["prompt_truncated"])
        + int(shuffled_metrics["prompt_truncated"])
    )
    overall = 1769 + aligned_correct
    choice = 125 + family["choice_final"]
    numeric = 1644 + family["numeric_final"]
    clean = 900 + member["clean"]
    fault = 869 + member["fault"]
    gate = {
        "zero_truncation": truncation == 0,
        "internal_final_ge_0_90": float(internal_final["process_accuracy"]) >= 0.90,
        "aligned_disagreement_ge_105": aligned_correct >= 105,
        "overall_ge_1874": overall >= 1874,
        "choice_ge_220": choice >= 220,
        "aligned_minus_shuffled_ge_13": aligned_correct - shuffled_correct >= 13,
        "aligned_minus_shape_ge_13": aligned_correct - shape_correct >= 13,
    }
    payload = {
        "schema": SCHEMA,
        "status": "pass" if all(gate.values()) else "fail",
        "gate": gate,
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
        "internal_final": internal_final,
        "total_prompt_truncation": truncation,
        "fit_receipt": {
            "updates_completed": fit["updates_completed"],
            "best_update": fit["best_update"],
            "trainable_backbone_parameters": fit["trainable_backbone_parameters"],
            "head_parameters": fit["head_parameters"],
            "elapsed_seconds": fit["elapsed_seconds"],
            "peak_allocated_gpu_bytes": fit["peak_allocated_gpu_bytes"],
        },
        "inputs": {
            "fit": sha256_file(args.fit),
            "aligned": sha256_file(args.aligned),
            "shuffled": sha256_file(args.shuffled),
            "shape": sha256_file(args.shape),
            "candidates": sha256_file(args.candidates),
        },
        "holdout_authorized": all(gate.values()),
        "holdout_used": False,
    }
    if args.output.exists():
        raise ETV1ComparisonError("refusing existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--shape", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
