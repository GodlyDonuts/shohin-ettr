#!/usr/bin/env python3
"""Evaluate the frozen oracle-typed DIVERGE-PL1 mechanics gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

from diverge_pl1_data import CONFIRMATION_SEEDS, Episode, episode_from_assessor_record
from diverge_pl1_runtime import (
    Arm,
    EpisodeResult,
    evaluate_policy_state,
    poison_and_rollback_probe,
    run_episode,
)


SCHEMA = "shohin-diverge-pl1-evaluation-v1"
ARMS: tuple[Arm, ...] = (
    "STATIC",
    "CONTEXT_ONLY",
    "DIVERGE_ONLY",
    "FAST_WEIGHT",
    "TRANSIENT_GRAD",
    "PL1",
)
RUN_SEED = 2026080799
BOOTSTRAP_SEED = 2026080798
BOOTSTRAP_RESAMPLES = 5_000


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_assessor(path: Path, expected_sha256: str) -> tuple[Episode, ...]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"assessor hash differs: {path}")
    with path.open(encoding="utf-8") as handle:
        return tuple(episode_from_assessor_record(json.loads(line)) for line in handle)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _summarize(episodes: tuple[Episode, ...], results: tuple[EpisodeResult, ...], elapsed: float) -> dict[str, Any]:
    transfer_exact = sum(result.transfer_exact for result in results)
    transfer_total = sum(result.transfer_total for result in results)
    mapping_exact = sum(result.mapping_exact for result in results)
    probe_exact = [
        sum(result.probe_transfer_exact[index] for result in results)
        for index in range(len(episodes[0].acquisition))
    ]
    attempt_passes = [
        sum(result.attempt_passes[index] for result in results)
        for index in range(len(episodes[0].acquisition))
    ]
    by_depth: Counter[int] = Counter()
    exact_by_depth: Counter[int] = Counter()
    for episode, result in zip(episodes, results, strict=True):
        outputs = evaluate_policy_state(episode, result.policy_state)
        for program, output in zip(episode.transfer, outputs, strict=True):
            depth = len(program.symbols)
            by_depth[depth] += 1
            exact_by_depth[depth] += output == program.terminal_state
    write_norms = [
        receipt.update_norm for result in results for receipt in result.write_receipts
    ]
    return {
        "episodes": len(results),
        "transfer_exact": transfer_exact,
        "transfer_total": transfer_total,
        "transfer_rate": transfer_exact / transfer_total,
        "mapping_exact": mapping_exact,
        "mapping_total": len(results),
        "mapping_rate": mapping_exact / len(results),
        "per_episode_transfer_exact": [result.transfer_exact for result in results],
        "probe_transfer_exact_by_attempt": probe_exact,
        "probe_transfer_total_by_attempt": [len(results) * 16] * len(probe_exact),
        "branch_passes_by_attempt": attempt_passes,
        "branch_trials_by_attempt": [len(results) * 8] * len(attempt_passes),
        "by_depth": {
            str(depth): {"exact": exact_by_depth[depth], "total": by_depth[depth]}
            for depth in sorted(by_depth)
        },
        "write_norm_max": max(write_norms, default=0.0),
        "write_norm_mean": sum(write_norms) / max(1, len(write_norms)),
        "protected_hashes_exact": all(
            len({receipt.protected_hash for receipt in result.write_receipts}) <= 1
            for result in results
        ),
        "rejected_credits": sum(
            receipt.rejected_credits
            for result in results
            for receipt in result.write_receipts
        ),
        "elapsed_seconds": elapsed,
    }


def _paired_bootstrap(left: list[float], right: list[float], salt: str) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap inputs differ")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    rng = random.Random(f"{BOOTSTRAP_SEED}:{salt}")
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        samples.append(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    samples.sort()
    return {
        "mean": sum(differences) / len(differences),
        "lower_95": samples[int(0.025 * len(samples))],
        "upper_95": samples[int(0.975 * len(samples))],
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing PL1 result: {args.output}")
    report_path = args.data_dir / "report.json"
    if sha256_path(report_path) != args.data_report_sha256:
        raise SystemExit("PL1 data report hash differs")
    data_report = _load_json(report_path)

    episodes_by_seed: dict[int, tuple[Episode, ...]] = {}
    for seed in CONFIRMATION_SEEDS:
        name = f"confirmation_seed_{seed}"
        split_report = data_report["split_reports"][name]
        assessor_path = args.data_dir / Path(split_report["assessor_path"]).name
        episodes = _load_assessor(assessor_path, split_report["assessor_sha256"])
        if len(episodes) != 256:
            raise SystemExit(f"PL1 confirmation count differs for {seed}")
        episodes_by_seed[seed] = episodes

    summaries: dict[str, dict[str, Any]] = {}
    results_by_arm_seed: dict[str, dict[int, tuple[EpisodeResult, ...]]] = {}
    for arm in ARMS:
        results_by_arm_seed[arm] = {}
        started = time.perf_counter()
        all_episodes = []
        all_results = []
        for seed, episodes in episodes_by_seed.items():
            results = tuple(run_episode(episode, arm=arm, seed=RUN_SEED) for episode in episodes)
            results_by_arm_seed[arm][seed] = results
            all_episodes.extend(episodes)
            all_results.extend(results)
        elapsed = time.perf_counter() - started
        summaries[arm] = _summarize(tuple(all_episodes), tuple(all_results), elapsed)

    control_specs: dict[str, dict[str, Any]] = {
        "RESET": {"reset_before_transfer": True},
        "SHUFFLED": {"credit_control": "shuffled"},
        "WRONG_BRANCH": {"credit_control": "wrong_branch"},
        "NO_ELIGIBILITY": {"credit_control": "no_eligibility"},
        "NO_HOMEOSTASIS": {"homeostatic": False},
    }
    controls: dict[str, dict[str, Any]] = {}
    for name, options in control_specs.items():
        started = time.perf_counter()
        all_episodes = []
        all_results = []
        for episodes in episodes_by_seed.values():
            all_episodes.extend(episodes)
            all_results.extend(
                run_episode(episode, arm="PL1", seed=RUN_SEED, **options)
                for episode in episodes
            )
        controls[name] = _summarize(
            tuple(all_episodes), tuple(all_results), time.perf_counter() - started
        )

    pl1_results = [
        result
        for seed in CONFIRMATION_SEEDS
        for result in results_by_arm_seed["PL1"][seed]
    ]
    all_episodes = [
        episode for seed in CONFIRMATION_SEEDS for episode in episodes_by_seed[seed]
    ]
    transplant_exact = 0
    transplant_total = 0
    rollback_exact = 0
    poison_behavior_changed = 0
    for seed in CONFIRMATION_SEEDS:
        episodes = episodes_by_seed[seed]
        results = results_by_arm_seed["PL1"][seed]
        for index, (episode, result) in enumerate(zip(episodes, results, strict=True)):
            donor = results[index - 1]
            outputs = evaluate_policy_state(episode, donor.policy_state)
            transplant_exact += sum(
                output == program.terminal_state
                for output, program in zip(outputs, episode.transfer, strict=True)
            )
            transplant_total += len(outputs)
            rollback = poison_and_rollback_probe(episode, result)
            rollback_exact += rollback.exact
            poison_behavior_changed += rollback.pre_outputs != rollback.poisoned_outputs

    mutation_rejections = 0
    for seed in CONFIRMATION_SEEDS:
        try:
            run_episode(
                episodes_by_seed[seed][0],
                arm="PL1",
                seed=RUN_SEED,
                inject_protected_mutation=True,
            )
        except RuntimeError:
            mutation_rejections += 1

    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed in CONFIRMATION_SEEDS:
        per_seed[str(seed)] = {}
        for arm in ARMS:
            results = results_by_arm_seed[arm][seed]
            transfer_exact = sum(result.transfer_exact for result in results)
            mapping_exact = sum(result.mapping_exact for result in results)
            per_seed[str(seed)][arm] = {
                "transfer_rate": transfer_exact / (len(results) * 16),
                "mapping_rate": mapping_exact / len(results),
            }

    bootstrap = {}
    for arm in ARMS:
        if arm == "PL1":
            continue
        bootstrap[arm] = _paired_bootstrap(
            [result.transfer_exact / 16 for result in pl1_results],
            [
                result.transfer_exact / 16
                for seed in CONFIRMATION_SEEDS
                for result in results_by_arm_seed[arm][seed]
            ],
            arm,
        )

    static_rate = summaries["STATIC"]["transfer_rate"]
    pl1_rate = summaries["PL1"]["transfer_rate"]
    conditions = {
        "pl1_transfer_at_least_85_percent": pl1_rate >= 0.85,
        "pl1_every_seed_transfer_at_least_80_percent": all(
            per_seed[str(seed)]["PL1"]["transfer_rate"] >= 0.80
            for seed in CONFIRMATION_SEEDS
        ),
        "pl1_mapping_at_least_80_percent": summaries["PL1"]["mapping_rate"] >= 0.80,
        "pl1_every_seed_mapping_at_least_75_percent": all(
            per_seed[str(seed)]["PL1"]["mapping_rate"] >= 0.75
            for seed in CONFIRMATION_SEEDS
        ),
        "aggregate_gain_at_least_10_points_over_every_arm": all(
            pl1_rate - summaries[arm]["transfer_rate"] >= 0.10
            for arm in ARMS
            if arm != "PL1"
        ),
        "every_seed_gain_at_least_5_points_over_every_arm": all(
            per_seed[str(seed)]["PL1"]["transfer_rate"]
            - per_seed[str(seed)][arm]["transfer_rate"]
            >= 0.05
            for seed in CONFIRMATION_SEEDS
            for arm in ARMS
            if arm != "PL1"
        ),
        "bootstrap_lower_bounds_above_zero": all(
            value["lower_95"] > 0.0 for value in bootstrap.values()
        ),
        "attempt_12_gain_at_least_50_points": (
            summaries["PL1"]["probe_transfer_exact_by_attempt"][-1]
            - summaries["PL1"]["probe_transfer_exact_by_attempt"][0]
        )
        / summaries["PL1"]["probe_transfer_total_by_attempt"][0]
        >= 0.50,
        "reset_loses_25_points_and_returns_to_static": pl1_rate
        - controls["RESET"]["transfer_rate"]
        >= 0.25
        and controls["RESET"]["transfer_rate"] <= static_rate + 0.03,
        "shuffled_wrong_and_transplant_at_static": controls["SHUFFLED"]["transfer_rate"]
        <= static_rate + 0.03
        and controls["WRONG_BRANCH"]["transfer_rate"] <= static_rate + 0.03
        and transplant_exact / transplant_total <= static_rate + 0.03,
        "rollback_exact_and_poison_changes_behavior": rollback_exact == len(pl1_results)
        and poison_behavior_changed / len(pl1_results) >= 0.95,
        "eligibility_ablation_loses_5_points": pl1_rate
        - controls["NO_ELIGIBILITY"]["transfer_rate"]
        >= 0.05,
        "protected_mutations_fail_closed": mutation_rejections == len(CONFIRMATION_SEEDS),
        "normal_protected_hashes_and_credits_exact": summaries["PL1"][
            "protected_hashes_exact"
        ]
        and summaries["PL1"]["rejected_credits"] == 0,
    }
    result = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "data_report": str(report_path),
        "data_report_sha256": args.data_report_sha256,
        "run_seed": RUN_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "summaries": summaries,
        "controls": controls,
        "per_seed": per_seed,
        "bootstrap": bootstrap,
        "transplant": {
            "exact": transplant_exact,
            "total": transplant_total,
            "rate": transplant_exact / transplant_total,
        },
        "rollback": {
            "exact": rollback_exact,
            "total": len(pl1_results),
            "poison_behavior_changed": poison_behavior_changed,
        },
        "protected_mutation_rejections": mutation_rejections,
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
        "resource_contract": {
            "branches_per_attempt": 8,
            "attempts_per_episode": 12,
            "verifier_calls_per_episode": 96,
            "transfer_programs_per_episode": 16,
            "mutable_scalars_allowed_per_arm": 64,
            "pl1_actual_mutable_scalars": 64,
            "learned_parameters_in_mechanics_gate": 0,
            "write_norm_cap": 4.0,
            "score_clip": 8.0,
            "durable_consolidation": False,
            "raw_language": False,
            "referent_interface": "REFERENT_ORACLE",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "status": result["status"],
                "pl1_transfer_rate": pl1_rate,
                "strongest_baseline_rate": max(
                    summaries[arm]["transfer_rate"] for arm in ARMS if arm != "PL1"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
