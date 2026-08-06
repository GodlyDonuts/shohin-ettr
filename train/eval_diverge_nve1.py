#!/usr/bin/env python3
"""Run the one frozen DIVERGE-NVE1 natural-variable evidence gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from diverge_nve1_data import (
    BOARD_ROWS,
    TRAIN_SEED,
    validate_board_row,
)
from diverge_nve1_runtime import (
    EvidenceCompilation,
    EvidenceCompilerConfig,
    NaturalEvidenceCompiler,
    NaturalEvidenceReceipt,
    compile_evidence_batch,
    execute_natural_evidence,
    module_state_sha256 as evidence_state_sha256,
    mutate_receipt,
    seal_natural_evidence,
)
from diverge_tfs1_data import FAULT_LINES, State, execute_steps, steps_from_record
from diverge_tfs1_runtime import (
    ABSTAIN,
    ANSWER,
    REJECT,
    CompiledPacket,
    CompiledQuery,
    FactorizedReceipt,
    LocalScorer,
    QueryDecision,
    TFS1RuntimeError,
    all_particle_bytes,
    compile_query,
    compile_source,
    execute_factorized,
    factorized_total_bytes,
    particle_capacity_for_bytes,
    query_particles,
    query_receipt,
    query_soft_answers,
    ranked_assignments,
    receipt_extensional_map,
)
from diverge_tol2_anchor_decoder import semantic_instruction_equal
from diverge_tol3_semantic_anchor import (
    LocalSemanticAnchor,
    TOL3Config,
    module_state_sha256 as tol3_state_sha256,
)
from version_space_accounting import canonical_json_bytes


SCHEMA = "shohin-diverge-nve1-evaluation-v1"


class NVE1EvaluationError(RuntimeError):
    """The frozen NVE1 evaluation contract was violated."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise NVE1EvaluationError("NVE1 confirmation board hash differs")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_board_row(row)
            rows.append(row)
    if len(rows) != BOARD_ROWS:
        raise NVE1EvaluationError("NVE1 confirmation row count differs")
    return rows


def _load_evidence_model(
    path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[NaturalEvidenceCompiler, dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise NVE1EvaluationError("NVE1 evidence checkpoint hash differs")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-nve1-training-report-v1":
        raise NVE1EvaluationError("NVE1 evidence checkpoint schema differs")
    if (
        int(checkpoint.get("update", -1)) != 1000
        or int(checkpoint.get("seed", -1)) != TRAIN_SEED
    ):
        raise NVE1EvaluationError("NVE1 evidence training schedule differs")
    model = NaturalEvidenceCompiler(EvidenceCompilerConfig(**checkpoint["config"])).to(
        device
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if evidence_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise NVE1EvaluationError("NVE1 evidence model state differs")
    return model, checkpoint


def _load_tol3(
    path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[LocalSemanticAnchor, dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise NVE1EvaluationError("NVE1 TOL3 checkpoint hash differs")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-tol3-training-report-v1":
        raise NVE1EvaluationError("NVE1 TOL3 checkpoint schema differs")
    model = LocalSemanticAnchor(TOL3Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if tol3_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise NVE1EvaluationError("NVE1 TOL3 model state differs")
    return model, checkpoint


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


def _answer_exact(decision: QueryDecision, expected: str) -> bool:
    return decision.disposition == ANSWER and decision.answer == expected


def _decision_record(decision: QueryDecision) -> dict[str, object]:
    return {
        "disposition": decision.disposition,
        "answer": decision.answer,
        "represented_worlds": decision.represented_worlds,
    }


def _rejected(reason: str) -> FactorizedReceipt:
    return FactorizedReceipt((), 0, 0, 0, 0, 0, 0, True, reason)


def _receipt_tuple(
    results: Sequence[EvidenceCompilation],
) -> tuple[NaturalEvidenceReceipt, ...] | None:
    if any(result.receipt is None for result in results):
        return None
    return tuple(result.receipt for result in results if result.receipt is not None)


def _compile_batches(
    model: NaturalEvidenceCompiler,
    texts: Sequence[str],
    packets: Sequence[CompiledPacket],
    indices: Sequence[int],
    *,
    compiler_commitment: str,
    device: torch.device,
    batch_size: int,
    swap_numeric_roles: bool = False,
    swap_symbol_roles: bool = False,
) -> list[EvidenceCompilation]:
    output = []
    for start in range(0, len(texts), batch_size):
        output.extend(
            compile_evidence_batch(
                model,
                texts[start : start + batch_size],
                packets[start : start + batch_size],
                indices[start : start + batch_size],
                compiler_commitment=compiler_commitment,
                device=device,
                swap_numeric_roles=swap_numeric_roles,
                swap_symbol_roles=swap_symbol_roles,
            )
        )
    return output


def _compile_episodes(
    model: LocalSemanticAnchor,
    rows: Sequence[dict[str, Any]],
    *,
    compiler_commitment: str,
    device: torch.device,
) -> tuple[
    list[CompiledPacket | None],
    list[dict[str, CompiledQuery] | None],
    list[LocalScorer | None],
    list[dict[str, str]],
]:
    packets: list[CompiledPacket | None] = []
    queries: list[dict[str, CompiledQuery] | None] = []
    scorers: list[LocalScorer | None] = []
    failures = []
    for row in rows:
        tfs1 = row["tfs1"]
        try:
            packet, scorer = compile_source(
                model,
                str(tfs1["source"]),
                expected_source_commitment=str(tfs1["source_commitment"]),
                compiler_commitment=compiler_commitment,
                device=device,
            )
            compiled_queries = {
                name: compile_query(packet, scorer, str(text))
                for name, text in tfs1["queries"].items()
            }
            packets.append(packet)
            queries.append(compiled_queries)
            scorers.append(scorer)
        except TFS1RuntimeError as error:
            packets.append(None)
            queries.append(None)
            scorers.append(None)
            failures.append({"id": str(tfs1["id"]), "error": str(error)})
    return packets, queries, scorers, failures


def _natural_receipt_exact(
    result: EvidenceCompilation,
    supervisor: Mapping[str, Any],
) -> bool:
    receipt = result.receipt
    return receipt is not None and (
        receipt.index == int(supervisor["index"])
        and receipt.step_index == int(supervisor["step_index"])
        and receipt.target == str(supervisor["target"])
        and receipt.distractor == str(supervisor["distractor"])
        and receipt.value == str(supervisor["value"])
        and receipt.evidence_source_sha256 == str(supervisor["source_sha256"])
    )


def evaluate(
    rows: list[dict[str, Any]],
    evidence_model: NaturalEvidenceCompiler,
    tol3_model: LocalSemanticAnchor,
    *,
    evidence_commitment: str,
    tol3_commitment: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    started = time.monotonic()
    packets_optional, queries_optional, _, compile_failures = _compile_episodes(
        tol3_model,
        rows,
        compiler_commitment=tol3_commitment,
        device=device,
    )
    if any(packet is None for packet in packets_optional):
        return {
            "schema": SCHEMA,
            "status": "tol3_component_fail",
            "compile_failures": compile_failures,
            "elapsed_seconds": time.monotonic() - started,
        }
    packets = [packet for packet in packets_optional if packet is not None]
    queries = [query for query in queries_optional if query is not None]

    flat_texts = [
        str(item["source_text"]) for row in rows for item in row["natural_evidence"]
    ]
    flat_packets = [packet for packet in packets for _ in range(FAULT_LINES)]
    flat_indices = [index for _ in rows for index in range(FAULT_LINES)]
    learned_flat = _compile_batches(
        evidence_model,
        flat_texts,
        flat_packets,
        flat_indices,
        compiler_commitment=evidence_commitment,
        device=device,
        batch_size=batch_size,
    )
    target_swapped_flat = _compile_batches(
        evidence_model,
        flat_texts,
        flat_packets,
        flat_indices,
        compiler_commitment=evidence_commitment,
        device=device,
        batch_size=batch_size,
        swap_symbol_roles=True,
    )
    numeric_swapped_flat = _compile_batches(
        evidence_model,
        flat_texts,
        flat_packets,
        flat_indices,
        compiler_commitment=evidence_commitment,
        device=device,
        batch_size=batch_size,
        swap_numeric_roles=True,
    )
    learned_sets = [
        learned_flat[start : start + FAULT_LINES]
        for start in range(0, len(learned_flat), FAULT_LINES)
    ]
    target_swapped_sets = [
        target_swapped_flat[start : start + FAULT_LINES]
        for start in range(0, len(target_swapped_flat), FAULT_LINES)
    ]
    numeric_swapped_sets = [
        numeric_swapped_flat[start : start + FAULT_LINES]
        for start in range(0, len(numeric_swapped_flat), FAULT_LINES)
    ]

    component_counts: Counter[str] = Counter()
    renderer_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    component_rows = []
    for row, packet, learned in zip(rows, packets, learned_sets, strict=True):
        tfs1 = row["tfs1"]
        program_exact = _compiled_program_exact(packet, tfs1)
        component_counts["compiled_program_exact"] += program_exact
        fault_lines = [step.fault for step in packet.steps if step.fault is not None]
        component_counts["fault_lines"] += len(fault_lines)
        component_counts["fault_lines_two_options"] += sum(
            fault is not None and len(fault.options) == 2 for fault in fault_lines
        )
        gold_assignment = tuple(int(value) for value in tfs1["gold_assignment"])
        component_counts["gold_support_rows"] += program_exact and all(
            gold_assignment[fault.index] in (0, 1)
            for fault in fault_lines
            if fault is not None
        )
        exact_in_row = 0
        compiled_in_row = 0
        for result, supervisor in zip(learned, row["natural_evidence"], strict=True):
            renderer = str(supervisor["renderer"])
            exact = _natural_receipt_exact(result, supervisor)
            compiled = result.receipt is not None
            exact_in_row += exact
            compiled_in_row += compiled
            component_counts["natural_receipts_compiled"] += compiled
            component_counts["natural_receipts_exact"] += exact
            component_counts["natural_receipts_inexact"] += compiled and not exact
            renderer_counts[renderer]["receipts"] += 1
            renderer_counts[renderer]["compiled"] += compiled
            renderer_counts[renderer]["exact"] += exact
        receipts = _receipt_tuple(learned)
        sealed = False
        if receipts is not None:
            try:
                sealed = (
                    len(
                        seal_natural_evidence(
                            packet,
                            receipts,
                            expected_compiler_commitment=evidence_commitment,
                        )
                    )
                    == FAULT_LINES
                )
            except Exception:
                sealed = False
        component_counts["episodes_fully_sealed"] += sealed
        component_rows.append(
            {
                "identity_sha256": row["identity_sha256"],
                "compiled_program_exact": program_exact,
                "receipts_compiled": compiled_in_row,
                "receipts_exact": exact_in_row,
                "fully_sealed": sealed,
            }
        )

    receipt_total = len(rows) * FAULT_LINES
    tol3_pass = (
        component_counts["compiled_program_exact"] == BOARD_ROWS
        and component_counts["fault_lines"] == receipt_total
        and component_counts["fault_lines_two_options"] == receipt_total
        and component_counts["gold_support_rows"] == BOARD_ROWS
    )
    evidence_pass = (
        component_counts["natural_receipts_exact"] >= 3041
        and component_counts["natural_receipts_inexact"] == 0
    )
    component = {
        "tol3_pass": tol3_pass,
        "natural_evidence_pass": evidence_pass,
        "counts": dict(component_counts),
        "renderer_counts": {
            key: dict(value) for key, value in sorted(renderer_counts.items())
        },
        "rows": component_rows,
    }
    if not (tol3_pass and evidence_pass):
        return {
            "schema": SCHEMA,
            "status": "component_fail",
            "component": component,
            "elapsed_seconds": time.monotonic() - started,
        }

    counts: Counter[str] = Counter()
    accounting: Counter[str] = Counter()
    depth_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    operation_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    episode_records = []
    learned_receipts: list[tuple[NaturalEvidenceReceipt, ...]] = []
    learned_typed: list[tuple[dict[str, object], ...]] = []
    for packet, learned in zip(packets, learned_sets, strict=True):
        receipts = _receipt_tuple(learned)
        assert receipts is not None
        learned_receipts.append(receipts)
        learned_typed.append(
            seal_natural_evidence(
                packet,
                receipts,
                expected_compiler_commitment=evidence_commitment,
            )
        )

    for episode_index, (row, packet, query_set, receipts, typed) in enumerate(
        zip(rows, packets, queries, learned_receipts, learned_typed, strict=True)
    ):
        tfs1 = row["tfs1"]
        oracle = tfs1["evidence"]
        expected = str(tfs1["gold_answer"])
        gold_assignment = tuple(int(value) for value in tfs1["gold_assignment"])
        sensitive = query_set["sensitive"]
        invariant = query_set["invariant"]
        underdetermined = query_set["underdetermined"]
        depth_key = str(len(packet.steps))

        no_evidence = execute_factorized(packet)
        no_evidence_worlds = receipt_extensional_map(no_evidence)
        independent_worlds = _assessor_worlds(tfs1)
        no_evidence_parity = no_evidence_worlds == independent_worlds
        oracle_full = execute_factorized(packet, oracle)
        oracle_decision = query_receipt(packet, oracle_full, sensitive)
        oracle_parity = set(receipt_extensional_map(oracle_full)) == {gold_assignment}
        learned_full = execute_factorized(packet, typed)
        learned_decision = query_receipt(packet, learned_full, sensitive)
        learned_worlds = receipt_extensional_map(learned_full)
        learned_parity = learned_worlds == receipt_extensional_map(oracle_full)
        learned_gold_preserved = set(learned_worlds) == {gold_assignment}
        no_evidence_sensitive = query_receipt(packet, no_evidence, sensitive)
        no_evidence_invariant = query_receipt(packet, no_evidence, invariant)
        partial = execute_factorized(packet, typed[:-1])
        partial_under = query_receipt(packet, partial, underdetermined)

        ranked = ranked_assignments(packet)
        top1_wrong = ranked[0] != gold_assignment
        top1_decision = query_particles(packet, sensitive, ranked[:1], typed)
        soft_decision = query_soft_answers(packet, sensitive, typed)
        factorized_bytes = factorized_total_bytes(packet, learned_full, typed)
        capacity, particle_used = particle_capacity_for_bytes(
            packet, ranked, typed, factorized_bytes
        )
        equal_decision = query_particles(packet, sensitive, ranked[:capacity], typed)
        all_particles = all_particle_bytes(packet, typed)

        reset = execute_natural_evidence(
            packet,
            receipts,
            expected_compiler_commitment=evidence_commitment,
            reset_after_declarations=True,
        )
        shifted = execute_natural_evidence(
            packet,
            receipts,
            expected_compiler_commitment=evidence_commitment,
            shift_fault_operations=True,
        )
        reset_decision = query_receipt(packet, reset, sensitive)
        shifted_decision = query_receipt(packet, shifted, sensitive)

        target_swapped = _receipt_tuple(target_swapped_sets[episode_index])
        target_swapped_receipt = (
            _rejected("target/distractor role compilation failed")
            if target_swapped is None
            else execute_natural_evidence(
                packet,
                target_swapped,
                expected_compiler_commitment=evidence_commitment,
            )
        )
        target_swapped_decision = query_receipt(
            packet, target_swapped_receipt, sensitive
        )
        numeric_swapped = _receipt_tuple(numeric_swapped_sets[episode_index])
        numeric_swapped_receipt = (
            _rejected("step/value role compilation failed")
            if numeric_swapped is None
            else execute_natural_evidence(
                packet,
                numeric_swapped,
                expected_compiler_commitment=evidence_commitment,
            )
        )
        numeric_swapped_decision = query_receipt(
            packet, numeric_swapped_receipt, sensitive
        )

        other_index = (episode_index + 1) % len(rows)
        shuffled = execute_natural_evidence(
            packet,
            learned_receipts[other_index],
            expected_compiler_commitment=evidence_commitment,
        )
        shuffled_decision = query_receipt(packet, shuffled, sensitive)
        packet_swap_decision = query_receipt(
            packet, learned_full, queries[other_index]["sensitive"]
        )

        invalid_by_field = Counter()
        for field in (
            "source",
            "packet",
            "evidence",
            "step",
            "target",
            "distractor",
            "value",
        ):
            mutated = list(receipts)
            mutated[0] = mutate_receipt(mutated[0], field)
            invalid_by_field[field] += not execute_natural_evidence(
                packet,
                mutated,
                expected_compiler_commitment=evidence_commitment,
            ).rejected
            counts[f"invalid_{field}_accepted"] += invalid_by_field[field]

        packet_record = packet.record()
        receipt_records = [receipt.record() for receipt in receipts]
        source_texts = [str(item["source_text"]) for item in row["natural_evidence"]]
        source_absent = all(
            text
            not in json.dumps(
                {"packet": packet_record, "receipts": receipt_records},
                sort_keys=True,
            )
            for text in source_texts
        )
        before_poison = canonical_sha256(
            {
                "packet": packet_record,
                "receipts": receipt_records,
                "decision": _decision_record(learned_decision),
            }
        )
        poisoned = [text + " [post-seal poison]" for text in source_texts]
        after_poison = canonical_sha256(
            {
                "packet": packet_record,
                "receipts": receipt_records,
                "decision": _decision_record(learned_decision),
            }
        )
        poison_invariant = (
            before_poison == after_poison
            and source_absent
            and all(text not in json.dumps(packet_record) for text in poisoned)
        )
        extensional_values = {
            dict(state).get(sensitive.register) for state in learned_worlds.values()
        }
        false_commitment = learned_decision.disposition == ANSWER and (
            len(extensional_values) != 1
            or learned_decision.answer not in extensional_values
        )

        metrics = {
            "oracle_exact": _answer_exact(oracle_decision, expected),
            "oracle_extensional_parity": oracle_parity,
            "learned_exact": _answer_exact(learned_decision, expected),
            "learned_extensional_parity": learned_parity,
            "learned_gold_preserved": learned_gold_preserved,
            "top1_exact": _answer_exact(top1_decision, expected),
            "equal_particle_exact": _answer_exact(equal_decision, expected),
            "soft_exact": _answer_exact(soft_decision, expected),
            "no_evidence_abstain": no_evidence_sensitive.disposition == ABSTAIN,
            "no_evidence_extensional_parity": no_evidence_parity,
            "invariant_exact": _answer_exact(
                no_evidence_invariant,
                str(tfs1["gold_terminal"][invariant.register]),
            ),
            "partial_underdetermined_abstain": (partial_under.disposition == ABSTAIN),
            "shuffled_exact": _answer_exact(shuffled_decision, expected),
            "target_swap_exact": _answer_exact(target_swapped_decision, expected),
            "step_value_swap_exact": _answer_exact(numeric_swapped_decision, expected),
            "state_reset_exact": _answer_exact(reset_decision, expected),
            "operation_shift_exact": _answer_exact(shifted_decision, expected),
            "packet_swap_reject": packet_swap_decision.disposition == REJECT,
            "poison_invariant": poison_invariant,
            "false_commitment": false_commitment,
            "overflow": learned_full.rejection_reason == "overflow",
            "malformed_accepted": learned_full.rejected,
        }
        for name, value in metrics.items():
            counts[name] += int(value)
            depth_counts[depth_key][name] += int(value)
        counts["initial_top1_wrong"] += top1_wrong
        counts["learned_exact_initial_top1_wrong"] += (
            top1_wrong and metrics["learned_exact"]
        )
        counts["invalid_receipts_accepted"] += sum(invalid_by_field.values())
        for operation in {
            option["operation"]
            for step in tfs1["steps"]
            if step["options"] is not None
            for option in step["options"]
        }:
            operation_counts[str(operation)]["episodes"] += 1
            operation_counts[str(operation)]["learned_exact"] += metrics[
                "learned_exact"
            ]

        accounting["factorized_bytes"] += factorized_bytes
        accounting["all_particle_bytes"] += all_particles
        accounting["equal_particle_capacity"] += capacity
        accounting["equal_particle_bytes"] += particle_used
        accounting["logical_instruction_applications"] += (
            learned_full.logical_instruction_applications
        )
        accounting["unique_instruction_applications"] += (
            learned_full.unique_instruction_applications
        )
        accounting["peak_groups_sum"] += learned_full.peak_groups
        accounting["peak_groups_max"] = max(
            accounting["peak_groups_max"], learned_full.peak_groups
        )
        if len(episode_records) < 24:
            episode_records.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "gold_answer": expected,
                    "gold_rank": ranked.index(gold_assignment),
                    "factorized_bytes": factorized_bytes,
                    "all_particle_bytes": all_particles,
                    "equal_particle_capacity": capacity,
                    "arms": {
                        "oracle": _decision_record(oracle_decision),
                        "learned": _decision_record(learned_decision),
                        "top1": _decision_record(top1_decision),
                        "equal_particles": _decision_record(equal_decision),
                        "soft": _decision_record(soft_decision),
                        "no_evidence": _decision_record(no_evidence_sensitive),
                        "shuffled": _decision_record(shuffled_decision),
                        "target_swap": _decision_record(target_swapped_decision),
                        "step_value_swap": _decision_record(numeric_swapped_decision),
                        "state_reset": _decision_record(reset_decision),
                        "operation_shift": _decision_record(shifted_decision),
                    },
                    "metrics": metrics,
                }
            )

    learned_exact = counts["learned_exact"]
    top1_wrong = counts["initial_top1_wrong"]
    control_drop = math.ceil(0.50 * BOARD_ROWS)
    wrong_top1_rate = counts["learned_exact_initial_top1_wrong"] / max(1, top1_wrong)
    conditions = {
        "tol3_compiles_all_256": component_counts["compiled_program_exact"]
        == BOARD_ROWS,
        "tol3_two_options_all_3072": (
            component_counts["fault_lines_two_options"] == receipt_total
        ),
        "tol3_gold_support_100_percent": component_counts["gold_support_rows"]
        == BOARD_ROWS,
        "natural_receipts_at_least_3041": component_counts["natural_receipts_exact"]
        >= 3041,
        "every_compiled_receipt_valid": component_counts["natural_receipts_inexact"]
        == 0,
        "oracle_typed_exact_256": counts["oracle_exact"] == BOARD_ROWS,
        "oracle_extensional_parity_256": counts["oracle_extensional_parity"]
        == BOARD_ROWS,
        "learned_exact_at_least_245": learned_exact >= 245,
        "learned_wrong_top1_conditional_at_least_95_percent": (
            top1_wrong > 0 and wrong_top1_rate >= 0.95
        ),
        "learned_beats_top1_by_50_points": (
            learned_exact - counts["top1_exact"] >= control_drop
        ),
        "learned_beats_equal_particles_by_50_points": (
            learned_exact - counts["equal_particle_exact"] >= control_drop
        ),
        "no_evidence_abstains_at_least_245": counts["no_evidence_abstain"] >= 245,
        "shuffled_evidence_drop_at_least_50_points": (
            learned_exact - counts["shuffled_exact"] >= control_drop
        ),
        "target_distractor_swap_drop_at_least_50_points": (
            learned_exact - counts["target_swap_exact"] >= control_drop
        ),
        "step_value_swap_drop_at_least_50_points": (
            learned_exact - counts["step_value_swap_exact"] >= control_drop
        ),
        "state_reset_drop_at_least_50_points": (
            learned_exact - counts["state_reset_exact"] >= control_drop
        ),
        "operation_shift_drop_at_least_50_points": (
            learned_exact - counts["operation_shift_exact"] >= control_drop
        ),
        "packet_query_swaps_all_reject": counts["packet_swap_reject"] == BOARD_ROWS,
        "source_poison_bit_invariant": counts["poison_invariant"] == BOARD_ROWS,
        "zero_invalid_receipts_accepted": counts["invalid_receipts_accepted"] == 0,
        "zero_false_commitments": counts["false_commitment"] == 0,
        "zero_malformed_accepted": counts["malformed_accepted"] == 0,
        "zero_gold_support_deletions": counts["learned_gold_preserved"] == BOARD_ROWS,
        "zero_overflow": counts["overflow"] == 0,
    }
    byte_ratio = accounting["all_particle_bytes"] / max(
        1, accounting["factorized_bytes"]
    )
    work_ratio = accounting["logical_instruction_applications"] / max(
        1, accounting["unique_instruction_applications"]
    )
    return {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "component": component,
        "counts": dict(counts),
        "matched_arms": {
            "oracle_typed": counts["oracle_exact"],
            "learned_natural": learned_exact,
            "premature_top1": counts["top1_exact"],
            "equal_memory_particles": counts["equal_particle_exact"],
            "soft_after_evidence": counts["soft_exact"],
            "no_evidence_abstain": counts["no_evidence_abstain"],
        },
        "controls": {
            "shuffled_evidence_exact": counts["shuffled_exact"],
            "target_distractor_swap_exact": counts["target_swap_exact"],
            "step_value_swap_exact": counts["step_value_swap_exact"],
            "state_reset_exact": counts["state_reset_exact"],
            "operation_shift_exact": counts["operation_shift_exact"],
            "packet_swap_reject": counts["packet_swap_reject"],
            "poison_invariant": counts["poison_invariant"],
        },
        "accounting": dict(accounting),
        "ratios": {
            "all_particles_to_factorized_bytes": byte_ratio,
            "logical_to_unique_instruction_applications": work_ratio,
            "wrong_top1_conditional_exact_rate": wrong_top1_rate,
        },
        "depth_counts": {
            key: dict(value) for key, value in sorted(depth_counts.items())
        },
        "operation_counts": {
            key: dict(value) for key, value in sorted(operation_counts.items())
        },
        "promotion_gate": {
            "conditions": conditions,
            "passed": all(conditions.values()),
        },
        "elapsed_seconds": time.monotonic() - started,
        "transcripts": episode_records,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-checkpoint-sha256", required=True)
    parser.add_argument("--tol3-checkpoint", type=Path, required=True)
    parser.add_argument("--tol3-checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NVE1 result: {args.output}")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("NVE1 requested CUDA is unavailable")
    rows = _load_rows(args.data, args.data_sha256)
    evidence_model, evidence_checkpoint = _load_evidence_model(
        args.evidence_checkpoint,
        args.evidence_checkpoint_sha256,
        device,
    )
    tol3_model, tol3_checkpoint = _load_tol3(
        args.tol3_checkpoint,
        args.tol3_checkpoint_sha256,
        device,
    )
    evidence_commitment = evidence_state_sha256(evidence_model)
    tol3_commitment = tol3_state_sha256(tol3_model)
    report = evaluate(
        rows,
        evidence_model,
        tol3_model,
        evidence_commitment=evidence_commitment,
        tol3_commitment=tol3_commitment,
        device=device,
        batch_size=args.batch_size,
    )
    report.update(
        {
            "data": str(args.data),
            "data_sha256": args.data_sha256,
            "evidence_checkpoint": str(args.evidence_checkpoint),
            "evidence_checkpoint_sha256": args.evidence_checkpoint_sha256,
            "evidence_model_state_sha256": evidence_commitment,
            "evidence_trainable_parameters": sum(
                parameter.numel() for parameter in evidence_model.parameters()
            ),
            "evidence_data_sha256": evidence_checkpoint["data_sha256"],
            "tol3_checkpoint": str(args.tol3_checkpoint),
            "tol3_checkpoint_sha256": args.tol3_checkpoint_sha256,
            "tol3_model_state_sha256": tol3_commitment,
            "tol3_data_sha256": tol3_checkpoint["data_sha256"],
            "device": str(device),
        }
    )
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "status": report["status"],
                "matched_arms": report.get("matched_arms"),
                "promotion_gate": report.get("promotion_gate"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
