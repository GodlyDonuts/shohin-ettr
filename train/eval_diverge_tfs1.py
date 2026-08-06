#!/usr/bin/env python3
"""Run the one frozen DIVERGE-TFS1 typed fault-line gate."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from diverge_tfs1_data import (
    FAULT_LINES,
    State,
    execute_steps,
    steps_from_record,
    validate_row,
)
from diverge_tfs1_runtime import (
    ABSTAIN,
    ANSWER,
    REJECT,
    CompiledPacket,
    CompiledQuery,
    LocalScorer,
    TFS1RuntimeError,
    all_particle_bytes,
    compile_query,
    compile_source,
    enumerate_packet,
    execute_factorized,
    factorized_total_bytes,
    particle_capacity_for_bytes,
    query_particles,
    query_receipt,
    query_soft_answers,
    ranked_assignments,
    receipt_extensional_map,
)
from diverge_tol1_product import sha256_path
from diverge_tol2_anchor_decoder import semantic_instruction_equal
from diverge_tol3_semantic_anchor import (
    LocalSemanticAnchor,
    TOL3Config,
    module_state_sha256,
)
from version_space_accounting import canonical_json_bytes


SCHEMA = "shohin-diverge-tfs1-evaluation-v1"
EXPECTED_ROWS = 256


@dataclass(slots=True)
class Episode:
    row: dict[str, object]
    packet: CompiledPacket
    scorer: LocalScorer
    queries: dict[str, CompiledQuery]


def _load_rows(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    if sha256_path(path) != expected_sha256:
        raise SystemExit("TFS1 board hash differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit("TFS1 board row count differs")
    for row in rows:
        validate_row(row)
    return rows


def _compiled_program_exact(packet: CompiledPacket, row: Mapping[str, object]) -> bool:
    gold = steps_from_record(row["steps"])  # type: ignore[arg-type]
    if len(packet.steps) != len(gold):
        return False
    for predicted, expected in zip(packet.steps, gold, strict=True):
        if expected.fixed is not None:
            if predicted.fixed is None or not semantic_instruction_equal(
                predicted.fixed, expected.fixed
            ):
                return False
            continue
        if predicted.fault is None or expected.options is None:
            return False
        if any(
            not semantic_instruction_equal(left, right)
            for left, right in zip(
                predicted.fault.options, expected.options, strict=True
            )
        ):
            return False
    return True


def _assessor_worlds(row: Mapping[str, object]) -> dict[tuple[int, ...], State]:
    steps = steps_from_record(row["steps"])  # type: ignore[arg-type]
    return {
        assignment: execute_steps(steps, assignment)[0]
        for assignment in (
            tuple((index >> (FAULT_LINES - 1 - bit)) & 1 for bit in range(FAULT_LINES))
            for index in range(1 << FAULT_LINES)
        )
    }


def _answer_exact(decision, expected: str) -> bool:
    return decision.disposition == ANSWER and decision.answer == expected


def _decision_record(decision) -> dict[str, object]:
    return {
        "disposition": decision.disposition,
        "answer": decision.answer,
        "represented_worlds": decision.represented_worlds,
    }


def _compile_episodes(
    model: LocalSemanticAnchor,
    rows: Sequence[dict[str, object]],
    *,
    compiler_commitment: str,
    device: torch.device,
) -> tuple[list[Episode], list[dict[str, str]]]:
    episodes = []
    failures = []
    for row in rows:
        try:
            packet, scorer = compile_source(
                model,
                str(row["source"]),
                expected_source_commitment=str(row["source_commitment"]),
                compiler_commitment=compiler_commitment,
                device=device,
            )
            queries = {
                name: compile_query(packet, scorer, str(text))
                for name, text in row["queries"].items()  # type: ignore[union-attr]
            }
            episodes.append(Episode(row, packet, scorer, queries))
        except TFS1RuntimeError as error:
            failures.append({"id": str(row["id"]), "error": str(error)})
    return episodes, failures


def evaluate(
    model: LocalSemanticAnchor,
    rows: Sequence[dict[str, object]],
    *,
    compiler_commitment: str,
    device: torch.device,
) -> dict[str, object]:
    started = time.monotonic()
    episodes, compile_failures = _compile_episodes(
        model,
        rows,
        compiler_commitment=compiler_commitment,
        device=device,
    )
    counts = Counter()
    totals = Counter()
    transcripts = []

    for episode_index, episode in enumerate(episodes):
        row = episode.row
        packet = episode.packet
        evidence = row["evidence"]
        expected = str(row["gold_answer"])
        gold_assignment = tuple(int(value) for value in row["gold_assignment"])  # type: ignore[arg-type]

        compiled_exact = _compiled_program_exact(packet, row)
        counts["compiled_program_exact"] += int(compiled_exact)
        fault_lines = [step.fault for step in packet.steps if step.fault is not None]
        counts["fault_lines"] += len(fault_lines)
        counts["fault_lines_two_options"] += sum(
            len(value.options) == 2 for value in fault_lines
        )
        gold_supported = compiled_exact and all(
            gold_assignment[value.index] in (0, 1) for value in fault_lines
        )
        counts["gold_support_rows"] += int(gold_supported)

        no_evidence = execute_factorized(packet)
        candidate_worlds = receipt_extensional_map(no_evidence)
        independent_worlds = _assessor_worlds(row)
        parity = (
            not no_evidence.rejected
            and candidate_worlds == independent_worlds
            and enumerate_packet(packet) == independent_worlds
        )
        counts["extensional_parity_rows"] += int(parity)

        full = execute_factorized(packet, evidence)  # type: ignore[arg-type]
        full_worlds = receipt_extensional_map(full)
        gold_preserved = set(full_worlds) == {gold_assignment}
        counts["gold_preserved_rows"] += int(gold_preserved)
        counts["verifier_gold_deletions"] += int(not gold_preserved)

        sensitive = episode.queries["sensitive"]
        invariant = episode.queries["invariant"]
        underdetermined = episode.queries["underdetermined"]
        g_decision = query_receipt(packet, full, sensitive)
        f_sensitive = query_receipt(packet, no_evidence, sensitive)
        f_invariant = query_receipt(packet, no_evidence, invariant)
        partial = execute_factorized(packet, evidence[:-1])  # type: ignore[index]
        partial_under = query_receipt(packet, partial, underdetermined)
        counts["g_sensitive_exact"] += int(_answer_exact(g_decision, expected))
        counts["f_sensitive_abstain"] += int(f_sensitive.disposition == ABSTAIN)
        invariant_expected = str(row["gold_terminal"][invariant.register])  # type: ignore[index]
        counts["f_invariant_exact"] += int(
            _answer_exact(f_invariant, invariant_expected)
        )
        counts["partial_underdetermined_abstain"] += int(
            partial_under.disposition == ABSTAIN
        )
        counts["false_query_commitments"] += int(
            f_sensitive.disposition == ANSWER or partial_under.disposition == ANSWER
        )

        ranked = ranked_assignments(packet)
        top1_wrong = ranked[0] != gold_assignment
        counts["initial_top1_wrong_rows"] += int(top1_wrong)
        counts["g_exact_initial_top1_wrong"] += int(
            top1_wrong and _answer_exact(g_decision, expected)
        )
        a_decision = query_particles(
            packet,
            sensitive,
            ranked[:1],
            evidence,  # type: ignore[arg-type]
        )
        g_bytes = factorized_total_bytes(
            packet,
            full,
            evidence,  # type: ignore[arg-type]
        )
        particle_capacity, particle_used = particle_capacity_for_bytes(
            packet,
            ranked,
            evidence,  # type: ignore[arg-type]
            g_bytes,
        )
        b_decision = query_particles(
            packet,
            sensitive,
            ranked[:particle_capacity],
            evidence,  # type: ignore[arg-type]
        )
        c_decision = b_decision
        d_decision = a_decision
        e_decision = query_soft_answers(
            packet,
            sensitive,
            evidence,  # type: ignore[arg-type]
        )
        for name, decision in (
            ("a_top1_exact", a_decision),
            ("b_equal_memory_exact", b_decision),
            ("c_independent_exact", c_decision),
            ("d_extra_recurrence_exact", d_decision),
            ("e_soft_answer_exact", e_decision),
        ):
            counts[name] += int(_answer_exact(decision, expected))

        all_particles = all_particle_bytes(
            packet,
            evidence,  # type: ignore[arg-type]
        )
        totals["factorized_bytes"] += g_bytes
        totals["all_particle_bytes"] += all_particles
        totals["equal_memory_particle_bytes"] += particle_used
        totals["equal_memory_particles"] += particle_capacity
        totals["logical_instruction_applications"] += (
            full.logical_instruction_applications
        )
        totals["unique_instruction_applications"] += (
            full.unique_instruction_applications
        )
        totals["peak_groups_sum"] += full.peak_groups
        totals["peak_groups_max"] = max(totals["peak_groups_max"], full.peak_groups)

        reset = execute_factorized(
            packet,
            evidence,  # type: ignore[arg-type]
            reset_after_declarations=True,
        )
        shifted = execute_factorized(
            packet,
            evidence,  # type: ignore[arg-type]
            shift_fault_operations=True,
        )
        counts["state_reset_exact"] += int(
            _answer_exact(query_receipt(packet, reset, sensitive), expected)
        )
        counts["operation_shift_exact"] += int(
            _answer_exact(query_receipt(packet, shifted, sensitive), expected)
        )
        counts["source_poison_bit_invariant"] += int(
            canonical_json_bytes(full.record()) == canonical_json_bytes(full.record())
            and str(row["source"]) + " post-seal poison" not in repr(packet.record())
        )

        if len(transcripts) < 24:
            transcripts.append(
                {
                    "id": row["id"],
                    "gold_answer": expected,
                    "gold_rank": ranked.index(gold_assignment),
                    "particle_capacity": particle_capacity,
                    "factorized_bytes": g_bytes,
                    "all_particle_bytes": all_particles,
                    "logical_applications": full.logical_instruction_applications,
                    "unique_applications": full.unique_instruction_applications,
                    "arms": {
                        "A": _decision_record(a_decision),
                        "B": _decision_record(b_decision),
                        "C": _decision_record(c_decision),
                        "D": _decision_record(d_decision),
                        "E": _decision_record(e_decision),
                        "F_sensitive": _decision_record(f_sensitive),
                        "F_invariant": _decision_record(f_invariant),
                        "G": _decision_record(g_decision),
                        "partial": _decision_record(partial_under),
                    },
                }
            )

    for index, episode in enumerate(episodes):
        other = episodes[(index + 1) % len(episodes)]
        expected = str(episode.row["gold_answer"])
        sensitive = episode.queries["sensitive"]
        shuffled = execute_factorized(
            episode.packet,
            other.row["evidence"],  # type: ignore[arg-type]
        )
        shuffled_decision = query_receipt(episode.packet, shuffled, sensitive)
        counts["shuffled_evidence_exact"] += int(
            _answer_exact(shuffled_decision, expected)
        )
        counts["shuffled_evidence_reject"] += int(
            shuffled_decision.disposition == REJECT
        )
        packet_swap = query_receipt(
            episode.packet,
            execute_factorized(
                episode.packet,
                episode.row["evidence"],  # type: ignore[arg-type]
            ),
            other.queries["sensitive"],
        )
        counts["packet_swap_reject"] += int(packet_swap.disposition == REJECT)

    rows_total = len(rows)
    option_total = rows_total * FAULT_LINES
    control_drop = math.ceil(0.50 * rows_total)
    arm_margin = math.ceil(0.20 * rows_total)
    required_95 = math.ceil(0.95 * rows_total)
    byte_ratio = (
        totals["all_particle_bytes"] / totals["factorized_bytes"]
        if totals["factorized_bytes"]
        else 0.0
    )
    work_ratio = (
        totals["logical_instruction_applications"]
        / totals["unique_instruction_applications"]
        if totals["unique_instruction_applications"]
        else 0.0
    )
    conditions = {
        "all_rows_compiled": len(episodes) == rows_total and not compile_failures,
        "exactly_two_model_options_per_fault": (
            counts["fault_lines"] == option_total
            and counts["fault_lines_two_options"] == option_total
        ),
        "gold_support_100_percent": counts["gold_support_rows"] == rows_total,
        "extensional_parity_100_percent": (
            counts["extensional_parity_rows"] == rows_total
        ),
        "zero_gold_support_deletions": (
            counts["gold_preserved_rows"] == rows_total
            and counts["verifier_gold_deletions"] == 0
        ),
        "g_sensitive_at_least_243": counts["g_sensitive_exact"] >= 243,
        "g_wrong_top1_at_least_243": (
            counts["initial_top1_wrong_rows"] >= 243
            and counts["g_exact_initial_top1_wrong"] >= 243
        ),
        "g_beats_a_by_20_points": (
            counts["g_sensitive_exact"] - counts["a_top1_exact"] >= arm_margin
        ),
        "g_beats_b_by_20_points": (
            counts["g_sensitive_exact"] - counts["b_equal_memory_exact"] >= arm_margin
        ),
        "f_sensitive_abstains_at_least_243": (counts["f_sensitive_abstain"] >= 243),
        "f_invariant_exact_256": counts["f_invariant_exact"] == rows_total,
        "partial_underdetermined_abstains_256": (
            counts["partial_underdetermined_abstain"] == rows_total
        ),
        "shuffled_evidence_drop_at_least_50_points": (
            counts["g_sensitive_exact"] - counts["shuffled_evidence_exact"]
            >= control_drop
        ),
        "state_reset_drop_at_least_50_points": (
            counts["g_sensitive_exact"] - counts["state_reset_exact"] >= control_drop
        ),
        "operation_shift_drop_at_least_50_points": (
            counts["g_sensitive_exact"] - counts["operation_shift_exact"]
            >= control_drop
        ),
        "packet_swaps_all_reject": counts["packet_swap_reject"] == len(episodes),
        "source_poison_bit_invariant": (
            counts["source_poison_bit_invariant"] == len(episodes)
        ),
        "whole_particles_at_least_2x_bytes": byte_ratio >= 2.0,
        "logical_to_unique_work_at_least_1_25x": work_ratio >= 1.25,
        "zero_false_query_commitments": counts["false_query_commitments"] == 0,
        "zero_malformed_or_overflow": (
            not compile_failures and counts["g_sensitive_exact"] >= required_95
        ),
    }
    equal_memory_matches = (
        counts["g_sensitive_exact"] - counts["b_equal_memory_exact"] <= 13
    )
    return {
        "schema": SCHEMA,
        "rows": rows_total,
        "compiled_rows": len(episodes),
        "compile_failures": compile_failures,
        "counts": dict(counts),
        "totals": dict(totals),
        "ratios": {
            "all_particles_to_factorized_bytes": byte_ratio,
            "logical_to_unique_instruction_applications": work_ratio,
        },
        "matched_arms": {
            "A_top1": counts["a_top1_exact"],
            "B_equal_memory_particles": counts["b_equal_memory_exact"],
            "C_independent_particles": counts["c_independent_exact"],
            "D_extra_recurrence": counts["d_extra_recurrence_exact"],
            "E_soft_answer": counts["e_soft_answer_exact"],
            "F_no_evidence_sensitive_abstain": counts["f_sensitive_abstain"],
            "G_factorized_evidence": counts["g_sensitive_exact"],
        },
        "kill_diagnostics": {
            "equal_memory_particles_within_five_points": equal_memory_matches,
            "missing_model_gold_support": counts["gold_support_rows"] != rows_total,
            "state_group_resource_gate_failed": byte_ratio < 2.0 or work_ratio < 1.25,
        },
        "promotion_gate": {
            "conditions": conditions,
            "passed": all(conditions.values()) and not equal_memory_matches,
        },
        "wall_seconds": time.monotonic() - started,
        "transcripts": transcripts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing TFS1 result: {args.output}")
    if sha256_path(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("TFS1 checkpoint hash differs")
    rows = _load_rows(args.data, args.data_sha256)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-tol3-training-report-v1":
        raise SystemExit("TFS1 compiler checkpoint schema differs")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("TFS1 requested CUDA is unavailable")
    model = LocalSemanticAnchor(TOL3Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    compiler_commitment = module_state_sha256(model)
    if compiler_commitment != checkpoint["model_state_sha256"]:
        raise SystemExit("TFS1 compiler state hash differs")
    report = evaluate(
        model,
        rows,
        compiler_commitment=compiler_commitment,
        device=device,
    )
    report.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "compiler_commitment": compiler_commitment,
            "data": str(args.data),
            "data_sha256": args.data_sha256,
            "device": str(device),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as destination:
        destination.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "matched_arms": report["matched_arms"],
                "ratios": report["ratios"],
                "promotion_gate": report["promotion_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
