#!/usr/bin/env python3
"""Run the frozen 1,024-episode DIVERGE-ULC1 CPU mechanics gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from version_space_accounting import canonical_json_bytes

from diverge_ulc1 import (
    DelayedObservation,
    DivergeContractError,
    SealedULC1Packet,
    apply_certified_observation,
    execute_ulc1,
    packet_bytes_ulc1,
)
from diverge_ulc1_reference import (
    board_commitment,
    build_ulc1_board,
    certify_observation,
    exact_factorized_parity,
    materialized_particle_bytes,
)
from diverge_v0 import (
    ABSTAIN,
    ANSWER,
    build_packet,
    enumerate_assignments,
    factorized_query_execution,
    named_commitment,
    read_query,
)
from diverge_v0_reference import reference_execute, reference_query

REPORT_SCHEMA = "shohin-diverge-ulc1-cpu-calibration-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _increment(target: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        target[key] += int(value)


def _top1_answer(episode) -> tuple[int, int]:
    worlds = {
        item.assignment: item
        for item in reference_execute(episode.sealed.packet).worlds
    }
    top = worlds[episode.initial_top1]
    gold = worlds[episode.gold_assignment]
    assert top.state is not None and gold.state is not None
    return (
        read_query(top.state, episode.sensitive_query),
        read_query(gold.state, episode.sensitive_query),
    )


def _source_poison_exact(episode) -> bool:
    before = packet_bytes_ulc1(episode.sealed)
    poisoned = replace(episode, source_text=f"POISON::{episode.episode_id}")
    return packet_bytes_ulc1(poisoned.sealed) == before


def _packet_swap_rejected(episode, other) -> bool:
    try:
        certify_observation(episode.sealed, other.observations[0])
    except DivergeContractError:
        return True
    return False


def _shuffled_provenance_rejected(episode) -> bool:
    original = episode.observations[0]
    provenance = (
        episode.observations[1].record_provenance
        if len(episode.observations) > 1
        else named_commitment("diverge-ulc1-unknown-record", episode.episode_id)
    )
    shuffled = DelayedObservation(
        original.source_commitment,
        provenance,
        original.state_slot,
        original.observed_value,
        original.evidence_commitment,
    )
    try:
        certify_observation(episode.sealed, shuffled)
    except DivergeContractError:
        return True
    return False


def _overflow_fails_closed(episode) -> bool:
    caps = replace(episode.sealed.packet.caps, max_worlds=1)
    packet = build_packet(
        source_commitment=episode.sealed.packet.source_commitment,
        shared_state=episode.sealed.packet.shared_state,
        variables=episode.sealed.packet.variables,
        hard_factors=episode.sealed.packet.hard_factors,
        support_factors=episode.sealed.packet.support_factors,
        patches=episode.sealed.packet.patches,
        caps=caps,
    )
    return bool(
        packet.overflow
        and not packet.variables
        and not packet.hard_factors
        and not packet.support_factors
        and not packet.patches
        and execute_ulc1(
            SealedULC1Packet(packet, episode.sealed.records)
        ).receipt.overflow
    )


def calibrate(seed: int, episode_count: int) -> dict[str, object]:
    started_ns = time.perf_counter_ns()
    episodes = build_ulc1_board(seed, episode_count)
    totals: dict[str, int] = defaultdict(int)
    split_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    episode_records = []
    high_ambiguity_ratios: list[Fraction] = []
    for index, episode in enumerate(episodes):
        initial_support = enumerate_assignments(episode.sealed.packet)
        initial = execute_ulc1(episode.sealed)
        initial_parity = exact_factorized_parity(initial)
        top_answer, gold_answer = _top1_answer(episode)
        top1_exact = top_answer == gold_answer
        invariant = factorized_query_execution(
            initial.sealed.packet, initial.receipt, episode.invariant_query
        )
        underdetermined = factorized_query_execution(
            initial.sealed.packet, initial.receipt, episode.underdetermined_query
        )
        conflict_disabled = factorized_query_execution(
            initial.sealed.packet, initial.receipt, episode.sensitive_query
        )

        execution = initial
        parity_failures = int(not initial_parity)
        valid_support_losses = 0
        monotonicity_failures = 0
        verifier_calls = 0
        removed_worlds = 0
        previous_support = set(initial_support)
        for observation in episode.observations:
            while True:
                certificates = certify_observation(execution.sealed, observation)
                if not certificates:
                    break
                certificate = certificates[0]
                verifier_calls += 1
                before = set(enumerate_assignments(execution.sealed.packet))
                execution = apply_certified_observation(execution, certificate)
                after = set(enumerate_assignments(execution.sealed.packet))
                removed_worlds += len(before - after)
                valid_support_losses += int(episode.gold_assignment not in after)
                monotonicity_failures += int(
                    not after < before or not after.issubset(previous_support)
                )
                parity_failures += int(not exact_factorized_parity(execution))
                previous_support = after

        final_support = set(enumerate_assignments(execution.sealed.packet))
        final_sensitive = factorized_query_execution(
            execution.sealed.packet, execution.receipt, episode.sensitive_query
        )
        expected_sensitive = reference_query(
            execution.sealed.packet, episode.sensitive_query
        )
        final_exact = (
            final_sensitive == expected_sensitive
            and final_sensitive.answer == gold_answer
        )
        false_query_certificate = not (
            invariant.disposition == ANSWER
            and invariant.answer == 43
            and underdetermined.disposition == ABSTAIN
            and conflict_disabled.disposition == ABSTAIN
            and final_sensitive.disposition == ANSWER
            and final_exact
        )

        factorized_bytes = (
            len(packet_bytes_ulc1(episode.sealed)) + initial.receipt.peak_group_bytes
        )
        particle_bytes = materialized_particle_bytes(initial)
        storage_ratio = Fraction(particle_bytes, max(1, factorized_bytes))
        if episode.represented_worlds >= 8:
            high_ambiguity_ratios.append(storage_ratio)
        source_poison_exact = _source_poison_exact(episode)
        packet_swap_rejected = _packet_swap_rejected(
            episode, episodes[(index + 1) % len(episodes)]
        )
        shuffled_rejected = _shuffled_provenance_rejected(episode)

        counters = {
            "episodes": 1,
            "represented_worlds": len(initial_support),
            "compile_gold_support_losses": int(
                episode.gold_assignment not in initial_support
            ),
            "initial_top1_wrong": int(episode.initial_top1 != episode.gold_assignment),
            "initial_top1_sensitive_exact": int(top1_exact),
            "final_sensitive_exact": int(final_exact),
            "final_unique_gold_support": int(
                final_support == {episode.gold_assignment}
            ),
            "parity_failures": parity_failures,
            "valid_support_losses": valid_support_losses,
            "monotonicity_failures": monotonicity_failures,
            "false_query_certificates": int(false_query_certificate),
            "verifier_calls": verifier_calls,
            "nogood_worlds_removed": removed_worlds,
            "source_poison_failures": int(not source_poison_exact),
            "packet_swap_acceptances": int(not packet_swap_rejected),
            "shuffled_provenance_acceptances": int(not shuffled_rejected),
            "factorized_bytes": factorized_bytes,
            "materialized_particle_bytes": particle_bytes,
            "factorized_unique_transactions": initial.receipt.unique_transactions,
            "whole_particle_transaction_applications": (
                initial.receipt.logical_transaction_applications
            ),
            "shared_transaction_applications": initial.receipt.shared_transactions,
            "high_ambiguity_episodes": int(episode.represented_worlds >= 8),
            "high_ambiguity_storage_passes": int(
                episode.represented_worlds < 8 or storage_ratio >= 2
            ),
        }
        _increment(totals, counters)
        _increment(split_totals[episode.split], counters)
        episode_records.append(
            {
                **episode.public_record(),
                "initial_top1_wrong": episode.initial_top1 != episode.gold_assignment,
                "initial_top1_sensitive_exact": top1_exact,
                "compile_gold_support_recalled": episode.gold_assignment
                in initial_support,
                "extensional_parity": parity_failures == 0,
                "valid_support_preserved": valid_support_losses == 0,
                "support_monotone": monotonicity_failures == 0,
                "final_unique_gold_support": final_support == {episode.gold_assignment},
                "final_sensitive_exact": final_exact,
                "invariant_answer": invariant.answer,
                "underdetermined_disposition": underdetermined.disposition,
                "conflict_disabled_disposition": conflict_disabled.disposition,
                "verifier_calls": verifier_calls,
                "source_poison_exact": source_poison_exact,
                "packet_swap_rejected": packet_swap_rejected,
                "shuffled_provenance_rejected": shuffled_rejected,
                "resources": {
                    "factorized_bytes": factorized_bytes,
                    "materialized_particle_bytes": particle_bytes,
                    "materialized_to_factorized_ratio": _fraction(storage_ratio),
                    "factorized_unique_transactions": initial.receipt.unique_transactions,
                    "whole_particle_transaction_applications": (
                        initial.receipt.logical_transaction_applications
                    ),
                    "peak_state_groups": initial.receipt.peak_groups,
                    "mask_operations": initial.receipt.mask_operations,
                },
            }
        )

    wrong = totals["initial_top1_wrong"]
    recovered = totals["final_sensitive_exact"] - totals["initial_top1_sensitive_exact"]
    high_ambiguity = totals["high_ambiguity_episodes"]
    overflow_pass = _overflow_fails_closed(episodes[-1])
    gate = {
        "extensional_parity_100pct": totals["parity_failures"] == 0,
        "compile_gold_support_100pct": totals["compile_gold_support_losses"] == 0,
        "zero_valid_world_deletions": totals["valid_support_losses"] == 0,
        "support_monotone_100pct": totals["monotonicity_failures"] == 0,
        "zero_false_query_commitments": totals["false_query_certificates"] == 0,
        "wrong_top1_recovery_at_least_90pct": Fraction(recovered, max(1, wrong))
        >= Fraction(9, 10),
        "sensitive_exact_100pct": totals["final_sensitive_exact"] == totals["episodes"],
        "source_poison_invariance_100pct": totals["source_poison_failures"] == 0,
        "packet_swap_rejection_100pct": totals["packet_swap_acceptances"] == 0,
        "shuffled_provenance_rejection_100pct": totals[
            "shuffled_provenance_acceptances"
        ]
        == 0,
        "high_ambiguity_storage_at_least_2x": (
            high_ambiguity > 0
            and totals["high_ambiguity_storage_passes"] == totals["episodes"]
            and min(high_ambiguity_ratios) >= 2
        ),
        "transaction_sharing_measured": (
            totals["whole_particle_transaction_applications"]
            > totals["factorized_unique_transactions"]
        ),
        "overflow_fails_closed": overflow_pass,
    }
    gate["pass"] = all(gate.values())
    report = {
        "schema": REPORT_SCHEMA,
        "seed": seed,
        "episode_count": episode_count,
        "board_commitment": board_commitment(episodes),
        "elapsed_nanoseconds": time.perf_counter_ns() - started_ns,
        "totals": dict(sorted(totals.items())),
        "split_totals": {
            key: dict(sorted(value.items()))
            for key, value in sorted(split_totals.items())
        },
        "recovery": {
            "wrong_top1_cases": wrong,
            "initial_top1_sensitive_exact": totals["initial_top1_sensitive_exact"],
            "final_sensitive_exact": totals["final_sensitive_exact"],
            "recovered_wrong_top1": recovered,
            "recovery_fraction": _fraction(Fraction(recovered, max(1, wrong))),
        },
        "sharing": {
            "materialized_to_factorized_total_ratio": _fraction(
                Fraction(
                    totals["materialized_particle_bytes"],
                    max(1, totals["factorized_bytes"]),
                )
            ),
            "minimum_high_ambiguity_storage_ratio": _fraction(
                min(high_ambiguity_ratios)
            ),
            "worlds_per_factorized_byte": _fraction(
                Fraction(totals["represented_worlds"], totals["factorized_bytes"])
            ),
            "worlds_per_materialized_byte": _fraction(
                Fraction(
                    totals["represented_worlds"],
                    totals["materialized_particle_bytes"],
                )
            ),
            "worlds_per_factorized_transaction": _fraction(
                Fraction(
                    totals["represented_worlds"],
                    totals["factorized_unique_transactions"],
                )
            ),
            "worlds_per_whole_particle_transaction": _fraction(
                Fraction(
                    totals["represented_worlds"],
                    totals["whole_particle_transaction_applications"],
                )
            ),
            "shared_transaction_fraction": _fraction(
                Fraction(
                    totals["shared_transaction_applications"],
                    totals["whole_particle_transaction_applications"],
                )
            ),
        },
        "gate": gate,
        "episodes": episode_records,
        "claim_boundary": (
            "Exact synthetic source-sealed mechanics and resource result only; "
            "not a learned-language or model-owned reasoning result."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=202608057400)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/reasoning/diverge_ulc1/cpu_calibration_seed202608057400.json"
        ),
    )
    args = parser.parse_args()
    report = calibrate(args.seed, args.episodes)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "gate": report["gate"],
                "recovery": report["recovery"],
                "sharing": report["sharing"],
            },
            sort_keys=True,
        )
    )
    if not report["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
