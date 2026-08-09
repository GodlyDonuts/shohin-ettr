#!/usr/bin/env python3
"""Conservatively separate semantic MoE revisions from output-format repairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "shohin-moe-semantic-repair-attribution-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_label(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    wrapper = re.fullmatch(r"\\(?:text|mathrm|mathbf)\{([^{}]+)\}", normalized)
    if wrapper:
        normalized = wrapper.group(1).strip()
    return normalized.strip(" .,:;()")


def serialization_only(before: dict[str, Any], gold: str) -> tuple[bool, str | None]:
    prediction = str(before.get("prediction") or "").strip()
    normalized_prediction = normalize_label(prediction)
    normalized_gold = normalize_label(gold)
    if not normalized_gold:
        return False, None
    if normalized_prediction == normalized_gold:
        return True, "normalized_prediction_already_gold"
    if len(normalized_gold) == 1 and normalized_gold.isalpha():
        if re.match(
            rf"^\s*{re.escape(normalized_gold)}\s*:", prediction, re.IGNORECASE
        ):
            return True, "gold_label_prefix_with_explanation"
        tail = str(before.get("completion") or "")[-512:]
        if re.search(
            rf"\b(?:option|making\s+option)\s+{re.escape(normalized_gold)}\b",
            tail,
            re.IGNORECASE,
        ):
            return True, "explicit_gold_option_in_final_tail"
    stripped_units = re.sub(r"\\text\{[^{}]*\}", "", prediction)
    if normalize_label(stripped_units) == normalized_gold:
        return True, "gold_value_with_units"
    return False, None


def load(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["identity_sha256"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load(args.unchanged)
    treatment = load(args.treatment)
    data = load(args.development_data)
    if set(baseline) != set(treatment) or set(baseline) != set(data):
        raise RuntimeError("semantic-attribution identities differ")
    transitions = []
    counts = {
        "strict_repairs": 0,
        "certified_serialization_repairs": 0,
        "remaining_possible_semantic_repairs": 0,
        "strict_breaks": 0,
    }
    for identity in sorted(baseline):
        before = baseline[identity]
        after = treatment[identity]
        if not before["correct"] and after["correct"]:
            counts["strict_repairs"] += 1
            gold = data[identity]["assessor"].get("expected_answer_normalized")
            is_serialization, reason = serialization_only(before, str(gold or ""))
            key = (
                "certified_serialization_repairs"
                if is_serialization
                else "remaining_possible_semantic_repairs"
            )
            counts[key] += 1
            transitions.append(
                {
                    "identity_sha256": identity,
                    "task": after["task"],
                    "transition": "repair",
                    "certified_serialization_only": is_serialization,
                    "reason": reason,
                    "before_prediction": before.get("prediction"),
                    "after_prediction": after.get("prediction"),
                }
            )
        elif before["correct"] and not after["correct"]:
            counts["strict_breaks"] += 1
            transitions.append(
                {
                    "identity_sha256": identity,
                    "task": after["task"],
                    "transition": "break",
                    "certified_serialization_only": False,
                    "reason": None,
                    "before_prediction": before.get("prediction"),
                    "after_prediction": after.get("prediction"),
                }
            )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "method": "conservative_rule_based_lower_bound_not_a_rescore",
        "input_sha256": {
            "unchanged": sha256_file(args.unchanged),
            "treatment": sha256_file(args.treatment),
            "development_data": sha256_file(args.development_data),
        },
        "counts": counts,
        "transitions": transitions,
    }
    atomic_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unchanged", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
