#!/usr/bin/env python3
"""Fail-closed matched-arm assessor for DIVERGE-EIC1 development."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "shohin-diverge-eic1-assessment-v1"
TRAIN_SCHEMA = "shohin-diverge-eic1-training-report-v1"
EVAL_SCHEMA = "shohin-diverge-eic1-evaluation-v1"
DEV_SHA256 = "299237068f436ba33a68487b5300fcd724f8c98bd8bfe6b1916a4ebc7541ebf7"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(root: Path, backbone: str, mode: str) -> dict[str, Any]:
    training_path = root / "model" / "report.json"
    development_path = root / "development.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if (
        training.get("schema") != TRAIN_SCHEMA
        or development.get("schema") != EVAL_SCHEMA
        or training.get("backbone_name") != backbone
        or development.get("backbone_name") != backbone
        or training.get("projection_mode") != mode
        or development.get("projection_mode") != mode
        or development.get("board_type") != "development"
        or development.get("data_sha256") != DEV_SHA256
        or development.get("checkpoint_sha256") != training.get("checkpoint_sha256")
        or development.get("adapter_state_sha256") != training.get("adapter_state_sha256")
        or training.get("backbone_forwards_per_source") != 2
    ):
        raise SystemExit(f"EIC1 arm receipt differs: {backbone}/{mode}")
    return {
        "root": str(root),
        "training_report_sha256": sha256_path(training_path),
        "development_report_sha256": sha256_path(development_path),
        "training": training,
        "development": development,
    }


def _matched(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "base_sha256",
        "tokenizer_sha256",
        "public_data_sha256",
        "supervisor_data_sha256",
        "seed",
        "updates",
        "pair_batch_size",
        "learning_rate",
        "consistency_weight",
        "lora_projection_count",
        "trainable_parameters",
        "total_parameters",
        "logical_public_rows",
        "unique_source_rows",
        "backbone_forwards_per_source",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def _exact(report: Mapping[str, Any], condition: str) -> int:
    return int(report[condition]["overall"].get("exact", 0))


def assess(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    st = arms["shohin_involution"]
    sc = arms["shohin_duplicate"]
    mt = arms["smollm2_involution"]
    mc = arms["smollm2_duplicate"]
    st_dev = st["development"]
    sc_dev = sc["development"]
    conditions = {
        "shohin_compute_and_parameter_match": _matched(
            st["training"], sc["training"]
        ),
        "smollm2_compute_and_parameter_match": _matched(
            mt["training"], mc["training"]
        ),
        "shohin_treatment_conjunctive_gate": bool(
            st_dev["promotion_gate"]["passed"]
        ),
        "shohin_projection_identity_exact": float(
            st_dev["projection_identity_max_absolute_error"]
        )
        == 0.0,
        "shohin_swap_gain_at_least_200": _exact(st_dev, "mention_swap")
        - _exact(sc_dev, "mention_swap")
        >= 200,
        "shohin_normal_loss_at_most_3": _exact(st_dev, "normal")
        >= _exact(sc_dev, "normal") - 3,
    }
    passed = all(conditions.values())
    summary = {}
    for name, arm in arms.items():
        development = arm["development"]
        summary[name] = {
            "normal_exact": _exact(development, "normal"),
            "mapped_swap_exact": _exact(development, "mention_swap"),
            "scrub_exact": _exact(development, "scrub_context"),
            "projection_identity_max_absolute_error": development[
                "projection_identity_max_absolute_error"
            ],
            "training_fit_true_exact": arm["training"]["training_fit"][
                "true_exact"
            ],
            "training_report_sha256": arm["training_report_sha256"],
            "development_report_sha256": arm["development_report_sha256"],
        }
    return {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "selected": "shohin_involution" if passed else None,
        "conditions": conditions,
        "summary": summary,
        "confirmation_access_authorized": passed,
        "claim_boundary": (
            "Development-only matched-arm assessment. A pass authorizes one "
            "EIC1 confirmation; it is not itself a reasoning claim."
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shohin-involution", type=Path, required=True)
    parser.add_argument("--shohin-duplicate", type=Path, required=True)
    parser.add_argument("--smollm2-involution", type=Path, required=True)
    parser.add_argument("--smollm2-duplicate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing EIC1 assessment")
    arms = {
        "shohin_involution": _load(
            args.shohin_involution, "shohin", "involution"
        ),
        "shohin_duplicate": _load(args.shohin_duplicate, "shohin", "duplicate"),
        "smollm2_involution": _load(
            args.smollm2_involution, "smollm2", "involution"
        ),
        "smollm2_duplicate": _load(
            args.smollm2_duplicate, "smollm2", "duplicate"
        ),
    }
    report = assess(arms)
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "passed": report["passed"],
                "selected": report["selected"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
