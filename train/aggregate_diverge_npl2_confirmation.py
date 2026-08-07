#!/usr/bin/env python3
"""Aggregate five frozen DIVERGE-NPL2 confirmation-seed reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from eval_diverge_pqi1 import sha256_path

SCHEMA = "shohin-diverge-npl2-confirmation-v1"
SEEDS = (2026080911, 2026080912, 2026080913, 2026080914, 2026080915)
ARMS = (
    "STATIC",
    "CONTEXT_ONLY",
    "DIVERGE_ONLY",
    "FAST_WEIGHT",
    "TRANSIENT_GRAD",
)
BOOTSTRAP_SEED = 2026080998
BOOTSTRAP_RESAMPLES = 5_000


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if (
        report.get("schema") != "shohin-diverge-npl2-confirmation-seed-v1"
        or report.get("status") != "complete"
        or report.get("evaluation_split") != "confirmation"
        or not report.get("integrity_gate", {}).get("passed")
    ):
        raise SystemExit(f"invalid NPL2 confirmation seed report: {path}")
    return report


def _combine_summary(
    reports: Sequence[Mapping[str, Any]], section: str, name: str
) -> dict[str, Any]:
    values = [report[section][name] for report in reports]
    query_exact = sum(int(value["query_exact"]) for value in values)
    query_total = sum(int(value["query_total"]) for value in values)
    transfer_exact = sum(int(value["transfer_exact"]) for value in values)
    transfer_total = sum(int(value["transfer_total"]) for value in values)
    mapping_exact = sum(int(value["mapping_exact"]) for value in values)
    mapping_total = sum(int(value["mapping_total"]) for value in values)
    return {
        "query_exact": query_exact,
        "query_total": query_total,
        "query_rate": query_exact / query_total,
        "transfer_exact": transfer_exact,
        "transfer_total": transfer_total,
        "transfer_rate": transfer_exact / transfer_total,
        "mapping_exact": mapping_exact,
        "mapping_total": mapping_total,
        "mapping_rate": mapping_exact / mapping_total,
        "per_episode_query_exact": [
            int(item) for value in values for item in value["per_episode_query_exact"]
        ],
        "probe_query_exact_by_attempt": [
            sum(int(value["probe_query_exact_by_attempt"][index]) for value in values)
            for index in range(12)
        ],
        "probe_query_total_by_attempt": [
            sum(int(value["probe_query_total_by_attempt"][index]) for value in values)
            for index in range(12)
        ],
        "semantic_rejections": sum(
            int(value["semantic_rejections"]) for value in values
        ),
        "protected_hashes_exact": all(
            value["protected_hashes_exact"] for value in values
        ),
        "rejected_credits": sum(int(value["rejected_credits"]) for value in values),
        "elapsed_seconds": sum(float(value["elapsed_seconds"]) for value in values),
    }


def _combine_semantic(
    reports: Sequence[Mapping[str, Any]], owner: str
) -> dict[str, Any]:
    overall = defaultdict(int)
    renderers: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for report in reports:
        score = report["semantic_compilation"][owner]
        for key, value in score["overall"].items():
            overall[key] += int(value)
        for renderer, values in score["by_renderer"].items():
            for key, value in values.items():
                renderers[str(renderer)][key] += int(value)
    return {
        "overall": dict(overall),
        "by_renderer": {key: dict(value) for key, value in sorted(renderers.items())},
    }


def _bootstrap(
    left: Sequence[int], right: Sequence[int], name: str
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("NPL2 bootstrap geometry differs")
    differences = [(a - b) / 32 for a, b in zip(left, right, strict=True)]
    rng = random.Random(f"{BOOTSTRAP_SEED}:{name}")
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draws.append(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    draws.sort()
    return {
        "mean": sum(differences) / len(differences),
        "lower_95": draws[int(0.025 * len(draws))],
        "upper_95": draws[int(0.975 * len(draws))],
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def _semantic_rate(score: Mapping[str, Any]) -> float:
    overall = score["overall"]
    return int(overall["exact"]) / int(overall["total"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--data-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing NPL2 confirmation aggregate")
    if len(args.result) != len(SEEDS):
        raise SystemExit("NPL2 confirmation requires exactly five reports")
    if sha256_path(args.data_report) != args.data_report_sha256:
        raise SystemExit("NPL2 confirmation data report hash differs")
    reports = [_load(path) for path in args.result]
    by_seed = {int(report["episode_seed"]): report for report in reports}
    if tuple(sorted(by_seed)) != SEEDS:
        raise SystemExit("NPL2 confirmation seeds differ")
    ordered = [by_seed[seed] for seed in SEEDS]
    if len({report["episode_ids_sha256"] for report in ordered}) != len(SEEDS):
        raise SystemExit("NPL2 confirmation episode identities overlap")
    custody_keys = (
        "eic_checkpoint_sha256",
        "sti_checkpoint_sha256",
        "eni1_result_sha256",
    )
    if any(
        len({report["custody"][key] for report in ordered}) != 1 for key in custody_keys
    ):
        raise SystemExit("NPL2 confirmation owner custody differs")

    summaries = {
        name: _combine_summary(ordered, "summaries", name)
        for name in (*ARMS, "NPL2", "PL1_ORACLE")
    }
    controls = {
        name: _combine_summary(ordered, "controls", name)
        for name in ("RESET", "SHUFFLED", "WRONG_BRANCH", "NO_ELIGIBILITY")
    }
    evidence = _combine_semantic(ordered, "evidence")
    query = _combine_semantic(ordered, "query")
    per_seed = {
        str(seed): {
            "NPL2": by_seed[seed]["summaries"]["NPL2"]["query_rate"],
            "PL1_ORACLE": by_seed[seed]["summaries"]["PL1_ORACLE"]["query_rate"],
            **{arm: by_seed[seed]["summaries"][arm]["query_rate"] for arm in ARMS},
        }
        for seed in SEEDS
    }
    bootstrap = {
        arm: _bootstrap(
            summaries["NPL2"]["per_episode_query_exact"],
            summaries[arm]["per_episode_query_exact"],
            arm,
        )
        for arm in ARMS
    }
    npl2_rate = summaries["NPL2"]["query_rate"]
    oracle_rate = summaries["PL1_ORACLE"]["query_rate"]
    static_rate = summaries["STATIC"]["query_rate"]
    transplant_rate = sum(
        float(report["transplant_query_rate"]) for report in ordered
    ) / len(ordered)
    rollback = {
        key: sum(int(report["rollback"][key]) for report in ordered)
        for key in ("exact", "changed", "total")
    }
    attempt_gain = (
        summaries["NPL2"]["probe_query_exact_by_attempt"][-1]
        - summaries["NPL2"]["probe_query_exact_by_attempt"][0]
    ) / summaries["NPL2"]["probe_query_total_by_attempt"][0]
    conditions = {
        "semantic_owners_at_least_99_5_percent": _semantic_rate(evidence) >= 0.995
        and _semantic_rate(query) >= 0.995
        and all(report["integrity_gate"]["passed"] for report in ordered),
        "npl2_at_least_80_percent": npl2_rate >= 0.80,
        "npl2_every_seed_at_least_75_percent": all(
            per_seed[str(seed)]["NPL2"] >= 0.75 for seed in SEEDS
        ),
        "npl2_within_5_points_of_oracle_aggregate": npl2_rate >= oracle_rate - 0.05,
        "npl2_within_5_points_of_oracle_4_of_5": sum(
            per_seed[str(seed)]["NPL2"] >= per_seed[str(seed)]["PL1_ORACLE"] - 0.05
            for seed in SEEDS
        )
        >= 4,
        "gain_10_points_over_every_nonoracle": all(
            npl2_rate - summaries[arm]["query_rate"] >= 0.10 for arm in ARMS
        ),
        "gain_5_points_over_every_nonoracle_4_of_5": all(
            sum(
                per_seed[str(seed)]["NPL2"] - per_seed[str(seed)][arm] >= 0.05
                for seed in SEEDS
            )
            >= 4
            for arm in ARMS
        ),
        "bootstrap_lower_bounds_above_zero": all(
            value["lower_95"] > 0.0 for value in bootstrap.values()
        ),
        "attempt_12_gain_at_least_50_points": attempt_gain >= 0.50,
        "reset_loses_25_points_and_returns_to_static": npl2_rate
        - controls["RESET"]["query_rate"]
        >= 0.25
        and controls["RESET"]["query_rate"] <= static_rate + 0.03,
        "shuffled_wrong_and_transplant_at_static": controls["SHUFFLED"]["query_rate"]
        <= static_rate + 0.03
        and controls["WRONG_BRANCH"]["query_rate"] <= static_rate + 0.03
        and transplant_rate <= static_rate + 0.03,
        "eligibility_ablation_loses_5_points": npl2_rate
        - controls["NO_ELIGIBILITY"]["query_rate"]
        >= 0.05,
        "rollback_exact_and_poison_changes_behavior": rollback["exact"]
        == rollback["total"]
        and rollback["changed"] / rollback["total"] >= 0.95,
        "protected_owners_and_source_deletion_exact": all(
            report["custody"]["owner_hashes_exact"]
            and report["gate"]["conditions"]["source_deleted_before_transfer"]
            for report in ordered
        ),
    }
    result = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "seeds": list(SEEDS),
        "seed_result_sha256": {
            str(report["episode_seed"]): sha256_path(path)
            for report, path in zip(reports, args.result, strict=True)
        },
        "data_report": str(args.data_report),
        "data_report_sha256": args.data_report_sha256,
        "semantic_compilation": {"evidence": evidence, "query": query},
        "summaries": summaries,
        "controls": controls,
        "per_seed": per_seed,
        "bootstrap": bootstrap,
        "transplant_query_rate": transplant_rate,
        "rollback": rollback,
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
        "claim_boundary": "controlled source-deleted natural mini-language reasoning; not open-domain reasoning",
    }
    _atomic_json(args.output, result)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "status": result["status"],
                "npl2_query_rate": npl2_rate,
                "oracle_query_rate": oracle_rate,
                "strongest_nonoracle_rate": max(
                    summaries[arm]["query_rate"] for arm in ARMS
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
