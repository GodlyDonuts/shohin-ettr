#!/usr/bin/env python3
"""Apply the frozen autonomous promotion gate to VCR1 reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DRAFT_SCHEMA = "shohin-diverge-vcr1-autonomous-drafts-v1"
REPORT_SCHEMA = "shohin-diverge-vcr1-autonomous-correction-v1"
GATE_SCHEMA = "shohin-diverge-vcr1-autonomous-gate-v1"
TASKS = ("math500", "preformatted_short_answer")
ARMS = {
    "plain": ("plain", "normal"),
    "treatment": ("vcr1", "normal"),
    "role_blind": ("role_blind", "normal"),
    "reset": ("vcr1", "reset"),
    "swap_roles": ("vcr1", "swap_roles"),
}


class VCR1GateError(RuntimeError):
    """The VCR1 result bundle violates the frozen scoring contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VCR1GateError(f"report is not an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VCR1GateError(f"refusing to replace gate report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _path_matrix(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    return {
        "math500": {
            "drafts": args.math_drafts,
            "plain": args.math_plain,
            "treatment": args.math_treatment,
            "role_blind": args.math_role_blind,
            "reset": args.math_reset,
            "swap_roles": args.math_swap_roles,
        },
        "preformatted_short_answer": {
            "drafts": args.science_drafts,
            "plain": args.science_plain,
            "treatment": args.science_treatment,
            "role_blind": args.science_role_blind,
            "reset": args.science_reset,
            "swap_roles": args.science_swap_roles,
        },
    }


def _validate_bundle(
    task: str, paths: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    draft = _load(paths["drafts"])
    if (
        draft.get("schema") != DRAFT_SCHEMA
        or draft.get("status") != "complete"
        or draft.get("task") != task
        or draft.get("count") != 100
        or len(draft.get("rows", [])) != 100
    ):
        raise VCR1GateError(f"{task} draft bank differs")

    reports: dict[str, dict[str, Any]] = {}
    for name, expected in ARMS.items():
        report = _load(paths[name])
        expected_arm, expected_ablation = expected
        if (
            report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("task") != task
            or report.get("arm") != expected_arm
            or report.get("ablation") != expected_ablation
        ):
            raise VCR1GateError(f"{task} {name} report envelope differs")
        if (
            report.get("drafts_sha256") != hashes["drafts"]
            or report.get("input_rows") != 100
            or report.get("evaluated_rows") != 100
            or report.get("skipped_length") != 0
            or report.get("source_correct") != draft.get("correct")
            or report.get("draft_ordered_identity_sha256")
            != draft.get("ordered_identity_sha256")
            or report.get("evaluated_ordered_identity_sha256")
            != draft.get("ordered_identity_sha256")
        ):
            raise VCR1GateError(f"{task} {name} board accounting differs")
        reports[name] = report
    return draft, reports, hashes


def _domain_summary(
    draft: dict[str, Any], reports: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    treatment = reports["treatment"]
    return {
        "source_correct": int(draft["correct"]),
        "plain_correct": int(reports["plain"]["corrected_correct"]),
        "treatment_correct": int(treatment["corrected_correct"]),
        "role_blind_correct": int(reports["role_blind"]["corrected_correct"]),
        "reset_correct": int(reports["reset"]["corrected_correct"]),
        "swap_roles_correct": int(reports["swap_roles"]["corrected_correct"]),
        "treatment_net_correction": int(treatment["net_correction"]),
        "treatment_transitions": treatment["transitions"],
        "validity_accuracy": treatment["validity_accuracy"],
        "validity_brier": treatment["validity_brier"],
        "generated_tokens": {
            name: int(report["generated_tokens"]) for name, report in reports.items()
        },
        "exhausted": {
            name: int(report["exhausted"]) for name, report in reports.items()
        },
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    domains: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, str]] = {}
    for task, paths in _path_matrix(args).items():
        draft, reports, hashes = _validate_bundle(task, paths)
        domains[task] = _domain_summary(draft, reports)
        receipts[task] = {
            name: f"{paths[name].resolve()}#{hashes[name]}" for name in paths
        }

    aggregate = {
        key: sum(domain[key] for domain in domains.values())
        for key in (
            "source_correct",
            "plain_correct",
            "treatment_correct",
            "role_blind_correct",
            "reset_correct",
            "swap_roles_correct",
            "treatment_net_correction",
        )
    }
    transitions = {
        key: sum(domain["treatment_transitions"][key] for domain in domains.values())
        for key in (
            "wrong_to_right",
            "right_to_right",
            "right_to_wrong",
            "wrong_to_wrong",
        )
    }
    aggregate["treatment_transitions"] = transitions
    checks = {
        "net_corrections_at_least_five": aggregate["treatment_net_correction"] >= 5,
        "beats_role_blind_by_three": (
            aggregate["treatment_correct"] - aggregate["role_blind_correct"] >= 3
        ),
        "positive_net_each_domain": all(
            domain["treatment_net_correction"] > 0 for domain in domains.values()
        ),
        "domain_regression_bounded": all(
            domain["treatment_correct"] >= domain["source_correct"] - 2
            for domain in domains.values()
        ),
        "right_to_wrong_bounded": (
            2 * transitions["right_to_wrong"] <= transitions["wrong_to_right"]
        ),
        "causal_ablation_drop_at_least_two": max(
            aggregate["treatment_correct"] - aggregate["reset_correct"],
            aggregate["treatment_correct"] - aggregate["swap_roles_correct"],
        )
        >= 2,
        "beats_plain_two_pass": (
            aggregate["treatment_correct"] > aggregate["plain_correct"]
        ),
    }
    payload = {
        "schema": GATE_SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "gate_pass": all(checks.values()),
        "checks": checks,
        "domains": domains,
        "aggregate": aggregate,
        "receipts": receipts,
    }
    _atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for domain in ("math", "science"):
        for name in (
            "drafts",
            "plain",
            "treatment",
            "role_blind",
            "reset",
            "swap_roles",
        ):
            parser.add_argument(
                f"--{domain}-{name.replace('_', '-')}", type=Path, required=True
            )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = score(args)
    print(
        f"[vcr1-gate] status={report['status']} "
        f"source={report['aggregate']['source_correct']}/200 "
        f"treatment={report['aggregate']['treatment_correct']}/200",
        flush=True,
    )
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
