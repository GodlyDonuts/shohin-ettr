#!/usr/bin/env python3
"""Run the one frozen integrated DIVERGE-IEM1 gate."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from diverge_iem1_data import BOARD_ROWS, TRAIN_SEED, validate_board_row
from diverge_iem1_runtime import (
    IEM1Config,
    IEM1LocalView,
    IEM1RuntimeError,
    IntegratedEpistemicMachine,
    QueryCompilation,
    compile_integrated_source,
    compile_query_batch,
    module_state_sha256,
    mutate_query_receipt,
    seal_natural_query,
)
from diverge_nve1_runtime import (
    EvidenceCompilation,
    NaturalEvidenceReceipt,
    execute_natural_evidence,
    mutate_receipt,
    seal_natural_evidence,
)
from diverge_tfs1_data import FAULT_LINES
from diverge_tfs1_runtime import (
    ABSTAIN,
    ANSWER,
    REJECT,
    CompiledPacket,
    CompiledQuery,
    FactorizedReceipt,
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
from diverge_tol1_product import load_rows
from eval_diverge_nve1 import (
    _answer_exact,
    _assessor_worlds,
    _compile_batches,
    _compiled_program_exact,
    _decision_record,
    _load_evidence_model,
    _load_rows as load_nve1_rows,
    _load_tol3,
    _natural_receipt_exact,
    _receipt_tuple,
    evaluate as evaluate_nve1,
)
from eval_diverge_tol3 import evaluate as evaluate_tol3
from version_space_accounting import canonical_json_bytes


SCHEMA = "shohin-diverge-iem1-evaluation-v1"


class IEM1EvaluationError(RuntimeError):
    """The frozen IEM1 evaluation contract was violated."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_board(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise IEM1EvaluationError("IEM1 confirmation board hash differs")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_board_row(row)
            rows.append(row)
    if len(rows) != BOARD_ROWS:
        raise IEM1EvaluationError("IEM1 confirmation row count differs")
    return rows


def _load_iem1(
    path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[IntegratedEpistemicMachine, dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise IEM1EvaluationError("IEM1 checkpoint hash differs")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-iem1-training-report-v1":
        raise IEM1EvaluationError("IEM1 checkpoint schema differs")
    if (
        int(checkpoint.get("update", -1)) != 1000
        or int(checkpoint.get("seed", -1)) != TRAIN_SEED
    ):
        raise IEM1EvaluationError("IEM1 training schedule differs")
    model = IntegratedEpistemicMachine(IEM1Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if module_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise IEM1EvaluationError("IEM1 model state differs")
    return model, checkpoint


def _compile_packets(
    rows: Sequence[dict[str, Any]],
    model,
    *,
    commitment: str,
    device: torch.device,
    integrated: bool,
) -> tuple[list[CompiledPacket | None], list[Any | None], list[dict[str, str]]]:
    packets: list[CompiledPacket | None] = []
    scorers: list[Any | None] = []
    failures = []
    for row in rows:
        tfs1 = row["tfs1"]
        try:
            if integrated:
                packet, scorer = compile_integrated_source(
                    model,
                    str(tfs1["source"]),
                    expected_source_commitment=str(tfs1["source_commitment"]),
                    compiler_commitment=commitment,
                    device=device,
                )
            else:
                packet, scorer = compile_source(
                    model,
                    str(tfs1["source"]),
                    expected_source_commitment=str(tfs1["source_commitment"]),
                    compiler_commitment=commitment,
                    device=device,
                )
            packets.append(packet)
            scorers.append(scorer)
        except (TFS1RuntimeError, IEM1RuntimeError) as error:
            packets.append(None)
            scorers.append(None)
            failures.append({"id": str(tfs1["id"]), "error": str(error)})
    return packets, scorers, failures


def _compile_evidence_for_packets(
    model,
    rows: Sequence[dict[str, Any]],
    packets: Sequence[CompiledPacket | None],
    *,
    commitment: str,
    device: torch.device,
    batch_size: int,
    swap_numeric_roles: bool = False,
    swap_symbol_roles: bool = False,
) -> list[list[EvidenceCompilation] | None]:
    output: list[list[EvidenceCompilation] | None] = [None] * len(rows)
    texts = []
    valid_packets = []
    indices = []
    coordinates = []
    for row_index, (row, packet) in enumerate(zip(rows, packets, strict=True)):
        if packet is None:
            continue
        for evidence_index, item in enumerate(row["natural_evidence"]):
            texts.append(str(item["source_text"]))
            valid_packets.append(packet)
            indices.append(evidence_index)
            coordinates.append(row_index)
    compiled = _compile_batches(
        model,
        texts,
        valid_packets,
        indices,
        compiler_commitment=commitment,
        device=device,
        batch_size=batch_size,
        swap_numeric_roles=swap_numeric_roles,
        swap_symbol_roles=swap_symbol_roles,
    )
    cursor = 0
    for row_index, packet in enumerate(packets):
        if packet is None:
            continue
        assert coordinates[cursor] == row_index
        output[row_index] = compiled[cursor : cursor + FAULT_LINES]
        cursor += FAULT_LINES
    if cursor != len(compiled):
        raise IEM1EvaluationError("IEM1 evidence compilation accounting differs")
    return output


def _compile_natural_queries(
    model: IntegratedEpistemicMachine,
    rows: Sequence[dict[str, Any]],
    packets: Sequence[CompiledPacket | None],
    *,
    commitment: str,
    device: torch.device,
    swap_roles: bool = False,
) -> list[dict[str, QueryCompilation] | None]:
    names = ("sensitive", "invariant", "underdetermined")
    output: list[dict[str, QueryCompilation] | None] = [None] * len(rows)
    texts = []
    valid_packets = []
    coordinates = []
    for row_index, (row, packet) in enumerate(zip(rows, packets, strict=True)):
        if packet is None:
            continue
        for name in names:
            texts.append(str(row["natural_queries"][name]["source_text"]))
            valid_packets.append(packet)
            coordinates.append(row_index)
    compiled = compile_query_batch(
        model,
        texts,
        valid_packets,
        compiler_commitment=commitment,
        device=device,
        swap_roles=swap_roles,
    )
    cursor = 0
    for row_index, packet in enumerate(packets):
        if packet is None:
            continue
        assert coordinates[cursor] == row_index
        output[row_index] = {
            name: compiled[cursor + offset] for offset, name in enumerate(names)
        }
        cursor += len(names)
    if cursor != len(compiled):
        raise IEM1EvaluationError("IEM1 query compilation accounting differs")
    return output


def _compile_typed_queries(
    rows: Sequence[dict[str, Any]],
    packets: Sequence[CompiledPacket | None],
    scorers: Sequence[Any | None],
) -> list[dict[str, CompiledQuery] | None]:
    output = []
    for row, packet, scorer in zip(rows, packets, scorers, strict=True):
        if packet is None or scorer is None:
            output.append(None)
            continue
        try:
            output.append(
                {
                    name: compile_query(packet, scorer, str(text))
                    for name, text in row["tfs1"]["queries"].items()
                }
            )
        except TFS1RuntimeError:
            output.append(None)
    return output


def _natural_query_exact(
    compilation: QueryCompilation,
    supervisor: Mapping[str, Any],
) -> bool:
    receipt = compilation.receipt
    query = compilation.query
    return (
        receipt is not None
        and query is not None
        and (
            receipt.target == str(supervisor["target"])
            and receipt.distractor == str(supervisor["distractor"])
            and receipt.query_source_sha256 == str(supervisor["source_sha256"])
            and query.register == str(supervisor["target"])
        )
    )


def _rejected(reason: str) -> FactorizedReceipt:
    return FactorizedReceipt((), 0, 0, 0, 0, 0, 0, True, reason)


def _query_from_compilation(
    compilation: QueryCompilation | None,
) -> CompiledQuery | None:
    return None if compilation is None else compilation.query


def _safe_query(
    packet: CompiledPacket,
    receipt: FactorizedReceipt,
    query: CompiledQuery | None,
) -> QueryDecision:
    if query is None:
        return QueryDecision(REJECT, None, 0)
    return query_receipt(packet, receipt, query)


def _malformed_packet_accepted(packet: CompiledPacket) -> bool:
    try:
        replace(packet, compiler_commitment="malformed")
    except TFS1RuntimeError:
        return False
    return True


def _protected_regressions(
    model: IntegratedEpistemicMachine,
    *,
    commitment: str,
    nve1_rows: list[dict[str, Any]],
    tol3_rows: list[dict[str, object]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    tol3 = evaluate_tol3(
        IEM1LocalView(model),  # type: ignore[arg-type]
        tol3_rows,
        device=device,
        batch_size=batch_size,
    )
    nve1 = evaluate_nve1(
        nve1_rows,
        model,  # type: ignore[arg-type]
        IEM1LocalView(model),  # type: ignore[arg-type]
        evidence_commitment=commitment,
        tol3_commitment=commitment,
        device=device,
        batch_size=batch_size,
    )
    return {
        "tol3_semantic_program_exact": tol3["counts"]["semantic_program_exact"],
        "tol3_rows": tol3["rows"],
        "tol3": tol3,
        "nve1_learned_exact": nve1.get("counts", {}).get("learned_exact", 0),
        "nve1_rows": len(nve1_rows),
        "nve1": nve1,
    }


def evaluate(
    rows: list[dict[str, Any]],
    model: IntegratedEpistemicMachine,
    ceiling_evidence,
    ceiling_tol3,
    *,
    commitment: str,
    ceiling_evidence_commitment: str,
    ceiling_tol3_commitment: str,
    protected: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    started = time.monotonic()
    packets, _, source_failures = _compile_packets(
        rows,
        model,
        commitment=commitment,
        device=device,
        integrated=True,
    )
    ceiling_packets, ceiling_scorers, ceiling_failures = _compile_packets(
        rows,
        ceiling_tol3,
        commitment=ceiling_tol3_commitment,
        device=device,
        integrated=False,
    )
    shifted_model = copy.deepcopy(model)
    with torch.no_grad():
        shifted_model.operation_transport_logits.copy_(
            torch.roll(shifted_model.operation_transport_logits, shifts=1, dims=-1)
        )
        shifted_model.comparator_transport_logits.copy_(
            torch.roll(shifted_model.comparator_transport_logits, shifts=1, dims=-1)
        )
    shifted_commitment = module_state_sha256(shifted_model)
    shifted_packets, _, shifted_failures = _compile_packets(
        rows,
        shifted_model,
        commitment=shifted_commitment,
        device=device,
        integrated=True,
    )

    learned_evidence = _compile_evidence_for_packets(
        model,
        rows,
        packets,
        commitment=commitment,
        device=device,
        batch_size=batch_size,
    )
    evidence_swapped = _compile_evidence_for_packets(
        model,
        rows,
        packets,
        commitment=commitment,
        device=device,
        batch_size=batch_size,
        swap_symbol_roles=True,
    )
    shifted_evidence = _compile_evidence_for_packets(
        shifted_model,
        rows,
        shifted_packets,
        commitment=shifted_commitment,
        device=device,
        batch_size=batch_size,
    )
    ceiling_evidence_sets = _compile_evidence_for_packets(
        ceiling_evidence,
        rows,
        ceiling_packets,
        commitment=ceiling_evidence_commitment,
        device=device,
        batch_size=batch_size,
    )
    queries = _compile_natural_queries(
        model,
        rows,
        packets,
        commitment=commitment,
        device=device,
    )
    query_swapped = _compile_natural_queries(
        model,
        rows,
        packets,
        commitment=commitment,
        device=device,
        swap_roles=True,
    )
    shifted_queries = _compile_natural_queries(
        shifted_model,
        rows,
        shifted_packets,
        commitment=shifted_commitment,
        device=device,
    )
    ceiling_queries = _compile_typed_queries(rows, ceiling_packets, ceiling_scorers)

    component = Counter()
    renderer_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    query_renderer_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for index, row in enumerate(rows):
        packet = packets[index]
        ceiling_packet = ceiling_packets[index]
        program_exact = packet is not None and _compiled_program_exact(
            packet, row["tfs1"]
        )
        ceiling_program_exact = ceiling_packet is not None and _compiled_program_exact(
            ceiling_packet, row["tfs1"]
        )
        component["source_program_exact"] += program_exact
        component["ceiling_source_program_exact"] += ceiling_program_exact
        if packet is not None:
            faults = [step.fault for step in packet.steps if step.fault is not None]
            component["fault_lines"] += len(faults)
            component["fault_lines_two_options"] += sum(
                fault is not None and len(fault.options) == 2 for fault in faults
            )
            component["gold_support_rows"] += program_exact
        evidence_set = learned_evidence[index]
        if evidence_set is not None:
            for result, supervisor in zip(
                evidence_set, row["natural_evidence"], strict=True
            ):
                exact = _natural_receipt_exact(result, supervisor)
                compiled = result.receipt is not None
                component["evidence_compiled"] += compiled
                component["evidence_exact"] += exact
                component["evidence_inexact"] += compiled and not exact
                local = renderer_counts[str(supervisor["renderer"])]
                local["total"] += 1
                local["compiled"] += compiled
                local["exact"] += exact
        query_set = queries[index]
        if query_set is not None:
            for name, result in query_set.items():
                supervisor = row["natural_queries"][name]
                exact = _natural_query_exact(result, supervisor)
                compiled = result.query is not None and result.receipt is not None
                component["query_compiled"] += compiled
                component["query_exact"] += exact
                component["query_inexact"] += compiled and not exact
                local = query_renderer_counts[str(supervisor["renderer"])]
                local["total"] += 1
                local["compiled"] += compiled
                local["exact"] += exact

    counts = Counter()
    accounting = Counter()
    transcripts = []
    learned_receipts: list[tuple[NaturalEvidenceReceipt, ...] | None] = []
    learned_typed: list[tuple[dict[str, object], ...] | None] = []
    for packet, evidence_set in zip(packets, learned_evidence, strict=True):
        receipts = None if evidence_set is None else _receipt_tuple(evidence_set)
        learned_receipts.append(receipts)
        if packet is None or receipts is None:
            learned_typed.append(None)
            continue
        try:
            learned_typed.append(
                seal_natural_evidence(
                    packet,
                    receipts,
                    expected_compiler_commitment=commitment,
                )
            )
            component["episodes_fully_sealed"] += 1
        except Exception:
            learned_typed.append(None)

    for episode_index, row in enumerate(rows):
        packet = packets[episode_index]
        query_set = queries[episode_index]
        receipts = learned_receipts[episode_index]
        typed = learned_typed[episode_index]
        if packet is None or query_set is None or receipts is None or typed is None:
            continue
        tfs1 = row["tfs1"]
        gold_assignment = tuple(int(value) for value in tfs1["gold_assignment"])
        expected = str(tfs1["gold_answer"])
        sensitive = _query_from_compilation(query_set["sensitive"])
        invariant = _query_from_compilation(query_set["invariant"])
        underdetermined = _query_from_compilation(query_set["underdetermined"])
        if sensitive is None or invariant is None or underdetermined is None:
            continue

        counts["episodes_evaluated"] += 1

        independent = _assessor_worlds(tfs1)
        no_evidence = execute_factorized(packet)
        oracle = execute_factorized(packet, tfs1["evidence"])
        learned = execute_factorized(packet, typed)
        oracle_decision = query_receipt(packet, oracle, sensitive)
        learned_decision = query_receipt(packet, learned, sensitive)
        no_evidence_sensitive = query_receipt(packet, no_evidence, sensitive)
        no_evidence_invariant = query_receipt(packet, no_evidence, invariant)
        partial = execute_factorized(packet, typed[:-1])
        partial_under = query_receipt(packet, partial, underdetermined)
        ranked = ranked_assignments(packet)
        top1_wrong = ranked[0] != gold_assignment
        top1 = query_particles(packet, sensitive, ranked[:1], typed)
        soft = query_soft_answers(packet, sensitive, typed)
        factorized_bytes = factorized_total_bytes(packet, learned, typed)
        capacity, particle_bytes = particle_capacity_for_bytes(
            packet, ranked, typed, factorized_bytes
        )
        equal = query_particles(packet, sensitive, ranked[:capacity], typed)

        reset = execute_natural_evidence(
            packet,
            receipts,
            expected_compiler_commitment=commitment,
            reset_after_declarations=True,
        )
        reset_decision = query_receipt(packet, reset, sensitive)
        swapped_receipts = (
            None
            if evidence_swapped[episode_index] is None
            else _receipt_tuple(evidence_swapped[episode_index])
        )
        evidence_swap_execution = (
            _rejected("evidence role swap did not compile")
            if swapped_receipts is None
            else execute_natural_evidence(
                packet,
                swapped_receipts,
                expected_compiler_commitment=commitment,
            )
        )
        evidence_swap_decision = query_receipt(
            packet, evidence_swap_execution, sensitive
        )
        query_swap = (
            None
            if query_swapped[episode_index] is None
            else _query_from_compilation(query_swapped[episode_index]["sensitive"])
        )
        query_swap_decision = _safe_query(packet, learned, query_swap)
        other_index = (episode_index + 1) % len(rows)
        shuffled_receipts = learned_receipts[other_index]
        shuffled = (
            _rejected("shuffled evidence absent")
            if shuffled_receipts is None
            else execute_natural_evidence(
                packet,
                shuffled_receipts,
                expected_compiler_commitment=commitment,
            )
        )
        shuffled_decision = query_receipt(packet, shuffled, sensitive)
        other_query = (
            None
            if queries[other_index] is None
            else _query_from_compilation(queries[other_index]["sensitive"])
        )
        packet_swap_decision = _safe_query(packet, learned, other_query)

        shifted_decision = QueryDecision(REJECT, None, 0)
        shifted_packet = shifted_packets[episode_index]
        shifted_set = shifted_evidence[episode_index]
        shifted_query_set = shifted_queries[episode_index]
        if (
            shifted_packet is not None
            and shifted_set is not None
            and shifted_query_set is not None
        ):
            shifted_receipts = _receipt_tuple(shifted_set)
            shifted_query = _query_from_compilation(shifted_query_set["sensitive"])
            if shifted_receipts is not None and shifted_query is not None:
                shifted_execution = execute_natural_evidence(
                    shifted_packet,
                    shifted_receipts,
                    expected_compiler_commitment=shifted_commitment,
                )
                shifted_decision = query_receipt(
                    shifted_packet, shifted_execution, shifted_query
                )

        ceiling_decision = QueryDecision(REJECT, None, 0)
        ceiling_packet = ceiling_packets[episode_index]
        ceiling_evidence_set = ceiling_evidence_sets[episode_index]
        ceiling_query_set = ceiling_queries[episode_index]
        if (
            ceiling_packet is not None
            and ceiling_evidence_set is not None
            and ceiling_query_set is not None
        ):
            ceiling_receipts = _receipt_tuple(ceiling_evidence_set)
            if ceiling_receipts is not None:
                ceiling_typed = seal_natural_evidence(
                    ceiling_packet,
                    ceiling_receipts,
                    expected_compiler_commitment=ceiling_evidence_commitment,
                )
                ceiling_execution = execute_factorized(ceiling_packet, ceiling_typed)
                ceiling_decision = query_receipt(
                    ceiling_packet,
                    ceiling_execution,
                    ceiling_query_set["sensitive"],
                )

        invalid_evidence = 0
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
            invalid_evidence += not execute_natural_evidence(
                packet,
                mutated,
                expected_compiler_commitment=commitment,
            ).rejected
        counts["invalid_evidence_accepted"] += invalid_evidence
        invalid_queries = 0
        natural_query_receipt = query_set["sensitive"].receipt
        assert natural_query_receipt is not None
        for field in (
            "packet",
            "source",
            "compiler",
            "target",
            "distractor",
            "commitment",
        ):
            try:
                seal_natural_query(
                    packet,
                    mutate_query_receipt(natural_query_receipt, field),
                    expected_compiler_commitment=commitment,
                )
                invalid_queries += 1
            except IEM1RuntimeError:
                pass
        counts["invalid_query_accepted"] += invalid_queries

        learned_worlds = receipt_extensional_map(learned)
        extensional_values = {
            str(value)
            for state in learned_worlds.values()
            if (value := dict(state).get(sensitive.register)) is not None
        }
        false_commitment = learned_decision.disposition == ANSWER and (
            len(extensional_values) != 1
            or learned_decision.answer not in extensional_values
        )
        packet_record = packet.record()
        evidence_records = [value.record() for value in receipts]
        query_record = natural_query_receipt.record()
        sealed_hash = canonical_sha256(
            {
                "packet": packet_record,
                "evidence": evidence_records,
                "query": query_record,
                "decision": _decision_record(learned_decision),
            }
        )
        poisoned_sources = [
            str(item["source_text"]) + " [post-seal poison]"
            for item in row["natural_evidence"]
        ]
        poisoned_sources.extend(
            str(item["source_text"]) + " [post-seal poison]"
            for item in row["natural_queries"].values()
        )
        poisoned_sources.append(str(row["tfs1"]["source"]) + " [post-seal poison]")
        poisoned_execution = execute_natural_evidence(
            packet,
            receipts,
            expected_compiler_commitment=commitment,
        )
        poisoned_decision = query_receipt(packet, poisoned_execution, sensitive)
        poisoned_hash = canonical_sha256(
            {
                "packet": packet_record,
                "evidence": evidence_records,
                "query": query_record,
                "decision": _decision_record(poisoned_decision),
            }
        )
        source_texts = [str(item["source_text"]) for item in row["natural_evidence"]]
        source_texts.extend(
            str(item["source_text"]) for item in row["natural_queries"].values()
        )
        source_texts.append(str(row["tfs1"]["source"]))
        sealed_record = {
            "packet": packet_record,
            "evidence": evidence_records,
            "query": query_record,
        }
        source_absent = all(
            text not in json.dumps(sealed_record, sort_keys=True)
            for text in source_texts
        )
        poison_invariant = (
            sealed_hash == poisoned_hash
            and source_absent
            and all(text not in json.dumps(sealed_record) for text in poisoned_sources)
        )

        metrics = {
            "ceiling_exact": _answer_exact(ceiling_decision, expected),
            "oracle_exact": _answer_exact(oracle_decision, expected),
            "learned_exact": _answer_exact(learned_decision, expected),
            "learned_extensional_parity": learned_worlds
            == {gold_assignment: independent[gold_assignment]},
            "learned_gold_preserved": set(learned_worlds) == {gold_assignment},
            "top1_exact": _answer_exact(top1, expected),
            "equal_particle_exact": _answer_exact(equal, expected),
            "soft_exact": _answer_exact(soft, expected),
            "no_evidence_abstain": no_evidence_sensitive.disposition == ABSTAIN,
            "invariant_exact": _answer_exact(
                no_evidence_invariant,
                str(row["tfs1"]["gold_terminal"][invariant.register]),
            ),
            "partial_underdetermined_abstain": partial_under.disposition == ABSTAIN,
            "semantic_transport_shift_exact": _answer_exact(shifted_decision, expected),
            "evidence_role_swap_exact": _answer_exact(evidence_swap_decision, expected),
            "query_role_swap_exact": _answer_exact(query_swap_decision, expected),
            "shuffled_evidence_exact": _answer_exact(shuffled_decision, expected),
            "state_reset_exact": _answer_exact(reset_decision, expected),
            "packet_query_swap_reject": packet_swap_decision.disposition == REJECT,
            "poison_invariant": poison_invariant,
            "false_commitment": false_commitment,
            "overflow": learned.rejection_reason == "overflow",
            "valid_execution_rejected": learned.rejected,
            "malformed_packet_accepted": _malformed_packet_accepted(packet),
        }
        for name, value in metrics.items():
            counts[name] += int(value)
        counts["initial_top1_wrong"] += top1_wrong
        counts["learned_exact_initial_top1_wrong"] += int(
            top1_wrong and metrics["learned_exact"]
        )
        accounting["factorized_bytes"] += factorized_bytes
        accounting["all_particle_bytes"] += all_particle_bytes(packet, typed)
        accounting["equal_particle_capacity"] += capacity
        accounting["equal_particle_bytes"] += particle_bytes
        accounting["logical_instruction_applications"] += (
            learned.logical_instruction_applications
        )
        accounting["unique_instruction_applications"] += (
            learned.unique_instruction_applications
        )
        accounting["peak_groups_max"] = max(
            accounting["peak_groups_max"], learned.peak_groups
        )
        if len(transcripts) < 24:
            transcripts.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "expected": expected,
                    "arms": {
                        "ceiling": _decision_record(ceiling_decision),
                        "learned": _decision_record(learned_decision),
                        "top1": _decision_record(top1),
                        "equal_particles": _decision_record(equal),
                        "no_evidence": _decision_record(no_evidence_sensitive),
                        "transport_shift": _decision_record(shifted_decision),
                        "evidence_swap": _decision_record(evidence_swap_decision),
                        "query_swap": _decision_record(query_swap_decision),
                    },
                    "metrics": metrics,
                }
            )

    learned_exact = counts["learned_exact"]
    evaluated = counts["episodes_evaluated"]
    top1_wrong = counts["initial_top1_wrong"]
    control_drop = math.ceil(0.50 * BOARD_ROWS)
    wrong_top1_rate = counts["learned_exact_initial_top1_wrong"] / max(1, top1_wrong)
    conditions = {
        "protected_tol3_at_least_1000": protected["tol3_semantic_program_exact"]
        >= 1000,
        "protected_nve1_at_least_250": protected["nve1_learned_exact"] >= 250,
        "fresh_source_at_least_250": component["source_program_exact"] >= 250,
        "fresh_fault_lines_all_two_options": component["fault_lines_two_options"]
        == component["fault_lines"],
        "fresh_gold_support_no_deletion": component["gold_support_rows"]
        == component["source_program_exact"],
        "fresh_evidence_at_least_3041": component["evidence_exact"] >= 3041,
        "fresh_query_at_least_752": component["query_exact"] >= 752,
        "accepted_evidence_all_exact": component["evidence_inexact"] == 0,
        "accepted_queries_all_exact": component["query_inexact"] == 0,
        "ceiling_exact_256": counts["ceiling_exact"] == BOARD_ROWS,
        "learned_exact_at_least_245": learned_exact >= 245,
        "wrong_top1_conditional_at_least_95_percent": top1_wrong > 0
        and wrong_top1_rate >= 0.95,
        "learned_extensional_parity_on_answers": counts["learned_extensional_parity"]
        == learned_exact,
        "no_evidence_abstains_at_least_245": counts["no_evidence_abstain"] >= 245,
        "invariant_all_accepted": counts["invariant_exact"] == evaluated,
        "partial_underdetermined_all_accepted": counts[
            "partial_underdetermined_abstain"
        ]
        == evaluated,
        "beats_top1_by_50_points": learned_exact - counts["top1_exact"] >= control_drop,
        "beats_equal_particles_by_50_points": learned_exact
        - counts["equal_particle_exact"]
        >= control_drop,
        "transport_shift_drop_50_points": learned_exact
        - counts["semantic_transport_shift_exact"]
        >= control_drop,
        "evidence_swap_drop_50_points": learned_exact
        - counts["evidence_role_swap_exact"]
        >= control_drop,
        "query_swap_drop_50_points": learned_exact - counts["query_role_swap_exact"]
        >= control_drop,
        "shuffled_evidence_drop_50_points": learned_exact
        - counts["shuffled_evidence_exact"]
        >= control_drop,
        "state_reset_drop_50_points": learned_exact - counts["state_reset_exact"]
        >= control_drop,
        "packet_query_swaps_all_reject": counts["packet_query_swap_reject"]
        == evaluated,
        "post_seal_poison_invariant": counts["poison_invariant"] == evaluated,
        "zero_invalid_evidence": counts["invalid_evidence_accepted"] == 0,
        "zero_invalid_query": counts["invalid_query_accepted"] == 0,
        "zero_false_commitments": counts["false_commitment"] == 0,
        "zero_gold_deletions": counts["learned_gold_preserved"] == evaluated,
        "zero_valid_execution_rejections": counts["valid_execution_rejected"] == 0,
        "zero_malformed_packet_accepted": counts["malformed_packet_accepted"] == 0,
        "zero_overflow": counts["overflow"] == 0,
    }
    return {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "protected_regressions": protected,
        "component": {
            "counts": dict(component),
            "evidence_renderer_counts": {
                key: dict(value) for key, value in sorted(renderer_counts.items())
            },
            "query_renderer_counts": {
                key: dict(value) for key, value in sorted(query_renderer_counts.items())
            },
            "source_failures": source_failures,
            "ceiling_failures": ceiling_failures,
            "shifted_failures": shifted_failures,
        },
        "counts": dict(counts),
        "matched_arms": {
            "separate_ceiling": counts["ceiling_exact"],
            "integrated_iem1": learned_exact,
            "premature_top1": counts["top1_exact"],
            "equal_memory_particles": counts["equal_particle_exact"],
            "no_evidence_abstain": counts["no_evidence_abstain"],
        },
        "controls": {
            "semantic_transport_shift_exact": counts["semantic_transport_shift_exact"],
            "evidence_role_swap_exact": counts["evidence_role_swap_exact"],
            "query_role_swap_exact": counts["query_role_swap_exact"],
            "shuffled_evidence_exact": counts["shuffled_evidence_exact"],
            "state_reset_exact": counts["state_reset_exact"],
            "packet_query_swap_reject": counts["packet_query_swap_reject"],
            "poison_invariant": counts["poison_invariant"],
        },
        "accounting": dict(accounting),
        "ratios": {
            "all_particles_to_factorized_bytes": accounting["all_particle_bytes"]
            / max(1, accounting["factorized_bytes"]),
            "logical_to_unique_instruction_applications": accounting[
                "logical_instruction_applications"
            ]
            / max(1, accounting["unique_instruction_applications"]),
            "wrong_top1_conditional_exact_rate": wrong_top1_rate,
        },
        "promotion_gate": {
            "conditions": conditions,
            "passed": all(conditions.values()),
        },
        "elapsed_seconds": time.monotonic() - started,
        "transcripts": transcripts,
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
    parser.add_argument("--iem1-checkpoint", type=Path, required=True)
    parser.add_argument("--iem1-checkpoint-sha256", required=True)
    parser.add_argument("--ceiling-evidence-checkpoint", type=Path, required=True)
    parser.add_argument("--ceiling-evidence-checkpoint-sha256", required=True)
    parser.add_argument("--ceiling-tol3-checkpoint", type=Path, required=True)
    parser.add_argument("--ceiling-tol3-checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--protected-nve1-data", type=Path, required=True)
    parser.add_argument("--protected-nve1-data-sha256", required=True)
    parser.add_argument("--protected-tol3-data", type=Path, required=True)
    parser.add_argument("--protected-tol3-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing IEM1 result: {args.output}")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("IEM1 requested CUDA is unavailable")

    rows = _load_board(args.data, args.data_sha256)
    model, checkpoint = _load_iem1(
        args.iem1_checkpoint,
        args.iem1_checkpoint_sha256,
        device,
    )
    ceiling_evidence, ceiling_evidence_checkpoint = _load_evidence_model(
        args.ceiling_evidence_checkpoint,
        args.ceiling_evidence_checkpoint_sha256,
        device,
    )
    ceiling_tol3, ceiling_tol3_checkpoint = _load_tol3(
        args.ceiling_tol3_checkpoint,
        args.ceiling_tol3_checkpoint_sha256,
        device,
    )
    protected_nve1 = load_nve1_rows(
        args.protected_nve1_data,
        args.protected_nve1_data_sha256,
    )
    protected_tol3 = load_rows(
        args.protected_tol3_data,
        args.protected_tol3_data_sha256,
        "ood",
    )
    commitment = module_state_sha256(model)
    protected = _protected_regressions(
        model,
        commitment=commitment,
        nve1_rows=protected_nve1,
        tol3_rows=protected_tol3,
        device=device,
        batch_size=args.batch_size,
    )
    report = evaluate(
        rows,
        model,
        ceiling_evidence,
        ceiling_tol3,
        commitment=commitment,
        ceiling_evidence_commitment=ceiling_evidence_checkpoint["model_state_sha256"],
        ceiling_tol3_commitment=ceiling_tol3_checkpoint["model_state_sha256"],
        protected=protected,
        device=device,
        batch_size=args.batch_size,
    )
    report.update(
        {
            "data": str(args.data),
            "data_sha256": args.data_sha256,
            "iem1_checkpoint": str(args.iem1_checkpoint),
            "iem1_checkpoint_sha256": args.iem1_checkpoint_sha256,
            "iem1_model_state_sha256": commitment,
            "iem1_trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "iem1_training_data": {
                "tol1": checkpoint["tol1_data_sha256"],
                "evidence": checkpoint["evidence_data_sha256"],
                "query": checkpoint["query_data_sha256"],
            },
            "ceiling_evidence_checkpoint_sha256": args.ceiling_evidence_checkpoint_sha256,
            "ceiling_tol3_checkpoint_sha256": args.ceiling_tol3_checkpoint_sha256,
            "protected_nve1_data_sha256": args.protected_nve1_data_sha256,
            "protected_tol3_data_sha256": args.protected_tol3_data_sha256,
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
