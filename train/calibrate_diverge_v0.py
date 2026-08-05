#!/usr/bin/env python3
"""Run and record the frozen DIVERGE-v0 CPU mechanics calibration gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from version_space_accounting import canonical_json_bytes

from diverge_delayed_board import board_commitment, build_delayed_board
from diverge_v0 import (
    ANSWER,
    ABSTAIN,
    append_verified_nogood,
    account_packet,
    enumerate_assignments,
    execute_packet,
    merge_certified_classes,
    packet_commitment,
    query_execution,
)
from diverge_v0_reference import (
    compare_execution,
    reference_behavioral_classes,
    reference_query,
    verify_nogood,
)


REPORT_SCHEMA = "shohin-diverge-v0-cpu-calibration-v1"


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def calibrate(seed: int) -> dict[str, object]:
    started_ns = time.perf_counter_ns()
    episodes = build_delayed_board(seed)
    episode_reports = []
    split_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    for episode in episodes:
        support = enumerate_assignments(episode.packet)
        candidate = execute_packet(episode.packet)
        parity = compare_execution(candidate, episode.packet)
        invariant = query_execution(candidate, episode.invariant_query)
        uncertain = query_execution(candidate, episode.underdetermined_query)
        verifier = verify_nogood(
            episode.packet,
            guard=episode.evidence.reject_guard,
            evidence_commitment=episode.evidence.evidence_commitment,
            valid_assignments=episode.evidence.valid_assignments,
        )
        if verifier.nogood is None:
            raise RuntimeError(f"valid calibration core rejected: {episode.episode_id}")
        refined = append_verified_nogood(episode.packet, verifier.nogood)
        refined_support = enumerate_assignments(refined)
        refined_receipt = execute_packet(refined)
        refined_parity = compare_execution(refined_receipt, refined)
        sensitive = query_execution(refined_receipt, episode.sensitive_query)
        reference_sensitive = reference_query(refined, episode.sensitive_query)
        structural = reference_behavioral_classes(
            refined,
            after_patches=len(refined.patches),
            queries=(episode.sensitive_query, episode.invariant_query),
        )
        merged = merge_certified_classes(
            refined,
            after_patches=len(refined.patches),
            certified_classes=structural,
        )
        accounting = account_packet(episode.packet, candidate)
        valid_set = set(episode.evidence.valid_assignments)
        refined_set = set(refined_support)
        gold_recalled = episode.gold_assignment in support
        valid_recall_exact = refined_set == valid_set
        false_certificate = not (
            invariant.disposition == ANSWER
            and invariant.answer == 104
            and uncertain.disposition == ABSTAIN
            and sensitive.disposition == ANSWER
            and sensitive.answer == 13
            and sensitive == reference_sensitive
        )
        merge_mass_exact = sum(item.mass for item in merged) == sum(
            world.mass for world in refined_receipt.worlds
        )
        record = {
            **episode.public_record(),
            "packet_commitment": packet_commitment(episode.packet),
            "initial_top1_wrong": episode.initial_top1 != episode.gold_assignment,
            "gold_support_recalled_after_compile": gold_recalled,
            "candidate_reference_parity": parity.exact,
            "refined_candidate_reference_parity": refined_parity.exact,
            "valid_support_recall_after_conflict": valid_recall_exact,
            "nogood_accepted": verifier.accepted,
            "nogood_removed_worlds": verifier.removed_worlds,
            "nogood_deletion_minimal": verifier.deletion_minimal,
            "false_query_certificate": false_certificate,
            "sensitive_disposition": sensitive.disposition,
            "invariant_disposition": invariant.disposition,
            "underdetermined_disposition": uncertain.disposition,
            "merge_mass_exact": merge_mass_exact,
            "merge_classes": len(merged),
            "accounting": {
                **asdict(accounting),
                "worlds_per_packet_byte": _fraction(accounting.worlds_per_packet_byte),
                "worlds_per_materialized_byte": _fraction(
                    accounting.worlds_per_materialized_byte
                ),
            },
        }
        episode_reports.append(record)
        counters = {
            "episodes": 1,
            "worlds": len(support),
            "parity_failures": int(not parity.exact or not refined_parity.exact),
            "compile_gold_support_losses": int(not gold_recalled),
            "conflict_valid_support_losses": int(not valid_recall_exact),
            "false_query_certificates": int(false_certificate),
            "unsafe_merges": int(not merge_mass_exact),
            "nogoods_accepted": int(verifier.accepted),
            "nogood_worlds_removed": verifier.removed_worlds,
            "packet_bytes": accounting.packet_bytes,
            "materialized_world_bytes": accounting.materialized_world_bytes,
            "unique_transactions": accounting.unique_transactions,
            "duplicated_transactions": accounting.duplicated_transactions,
            "shared_transactions": accounting.shared_transactions,
            "verifier_calls": 1,
        }
        for key, value in counters.items():
            totals[key] += value
            split_totals[episode.split][key] += value

    duplicated = totals["duplicated_transactions"]
    unique = totals["unique_transactions"]
    report = {
        "schema": REPORT_SCHEMA,
        "seed": seed,
        "board_commitment": board_commitment(episodes),
        "elapsed_nanoseconds": time.perf_counter_ns() - started_ns,
        "totals": dict(sorted(totals.items())),
        "split_totals": {
            split: dict(sorted(values.items()))
            for split, values in sorted(split_totals.items())
        },
        "sharing": {
            "worlds_per_packet_byte": _fraction(
                Fraction(totals["worlds"], max(1, totals["packet_bytes"]))
            ),
            "worlds_per_materialized_byte": _fraction(
                Fraction(totals["worlds"], max(1, totals["materialized_world_bytes"]))
            ),
            "materialized_to_packet_byte_ratio": _fraction(
                Fraction(
                    totals["materialized_world_bytes"],
                    max(1, totals["packet_bytes"]),
                )
            ),
            "duplicated_to_unique_transaction_ratio": _fraction(
                Fraction(duplicated, max(1, unique))
            ),
            "shared_transaction_fraction": _fraction(
                Fraction(totals["shared_transactions"], max(1, duplicated))
            ),
        },
        "gate": {
            "extensional_parity": totals["parity_failures"] == 0,
            "compile_gold_support_recall": totals["compile_gold_support_losses"] == 0,
            "conflict_valid_support_recall": totals["conflict_valid_support_losses"] == 0,
            "zero_false_query_certificates": totals["false_query_certificates"] == 0,
            "safe_merging": totals["unsafe_merges"] == 0,
            "measured_storage_sharing": (
                totals["materialized_world_bytes"] > totals["packet_bytes"]
            ),
            "measured_transaction_sharing": duplicated > unique,
        },
        "episodes": episode_reports,
    }
    report["gate"]["pass"] = all(report["gate"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reasoning/diverge_v0/cpu_calibration_seed20260805.json"),
    )
    arguments = parser.parse_args()
    report = calibrate(arguments.seed)
    _atomic_write(arguments.output, report)
    print(json.dumps({"output": str(arguments.output), "sha256": _sha256(arguments.output), **report["gate"]}, sort_keys=True))
    if not report["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
