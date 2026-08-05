#!/usr/bin/env python3
"""Complete source-compiler-backed matched A--G gate for DIVERGE-v0.

The learned source compiler is shared by every arm. The arms differ only in
how they retain coherent hypotheses, incorporate delayed evidence, and answer
late queries. Exact assessors construct targets and validate nogoods, but they
never repair candidate packets or arm outputs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from tokenizers import Tokenizer

import diverge_v0_neural_pilot as pilot
from diverge_v0 import (
    ABSTAIN,
    ANSWER,
    REJECT,
    ExecutionReceipt,
    FactorizedExecutionReceipt,
    Guard,
    Literal,
    Query,
    QueryDecision,
    WorldResult,
    account_packet,
    append_verified_nogood,
    certify_binary_option_evidence,
    enumerate_assignments,
    execute_packet,
    execute_packet_factorized,
    factorized_query_execution,
    materialized_world_bytes,
    named_commitment,
    packet_bytes,
    query_execution,
    refine_factorized_receipt,
)
from diverge_v0_reference import verify_nogood
from diverge_v0_role_copy_pilot import (
    SmolDivergeRoleCopyCompiler,
    predict_source_fields,
)
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-v0-matched-a-g-gate-v2"
ARM_NAMES = (
    "A_single",
    "B_full_particles",
    "C_independent",
    "D_recurrent_single",
    "E_soft",
    "F_factorized_no_conflict",
    "G_diverge",
)
QUERY_KINDS = ("sensitive", "invariant", "underdetermined")
ONTOLOGIES = ("register-workshop", "parcel-relation", "signal-routing")
EVIDENCE_PATTERN = re.compile(
    r"^delayed diagnostic confirms active key ([a-z]+-[0-9]+)\.$"
)


@dataclass(frozen=True)
class GateEpisode:
    source: pilot.PilotEpisode
    family: str
    sensitive_query: Query
    invariant_query: Query
    underdetermined_query: Query


@dataclass(frozen=True)
class CompiledEpisode:
    gate: GateEpisode
    packet: object | None
    refined_packet: object | None
    prediction: pilot.CompilerPrediction
    packet_exact: bool
    gold_support_recalled: bool
    primary_variable: int | None
    evidence_variable: int | None
    evidence_option: int | None
    primary_gold_option: int
    initial: ExecutionReceipt | None
    refined: ExecutionReceipt | None
    factorized_initial: FactorizedExecutionReceipt | None
    factorized_refined: FactorizedExecutionReceipt | None
    expected: dict[str, QueryDecision]
    verifier_calls: int
    valid_support_preserved: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _replace_primary_semantics(
    episode: pilot.PilotEpisode,
    *,
    gold_program: int,
) -> pilot.PilotEpisode:
    """Balance which noncommuting primary program is initially disfavored."""

    if gold_program not in (0, 1):
        raise ValueError("primary gold program must be zero or one")
    records = []
    evidence_alias = ""
    for record in episode.records:
        if record.record_id != episode.primary_record_id:
            records.append(record)
            continue
        options = tuple(
            dataclasses.replace(
                option,
                prior_class=int(option.program == gold_program),
                text=pilot._render_option(
                    option.alias,
                    option.program,
                    int(option.program == gold_program),
                    episode.renderer,
                ),
            )
            for option in record.options
        )
        gold_option = next(
            index for index, option in enumerate(options) if option.program == gold_program
        )
        evidence_alias = options[gold_option].alias
        records.append(
            dataclasses.replace(
                record,
                options=options,
                gold_option=gold_option,
                text=pilot._render_record(
                    episode.ontology,
                    options[0],
                    options[1],
                    is_fault_line=True,
                    renderer=episode.renderer,
                ),
            )
        )
    if not evidence_alias:
        raise AssertionError("primary record disappeared")
    records = tuple(records)
    return dataclasses.replace(
        episode,
        records=records,
        source_text="\n".join(record.text for record in records),
        evidence_alias=evidence_alias,
        evidence_text=f"delayed diagnostic confirms active key {evidence_alias}.",
    )


def _truth_prediction(episode: pilot.PilotEpisode) -> pilot.CompilerPrediction:
    selected = tuple(record.is_fault_line for record in episode.records)
    programs = tuple(
        tuple(option.program for option in record.options) for record in episode.records
    )
    priors = tuple(
        tuple(option.prior_class for option in record.options) for record in episode.records
    )
    for record_index, record in enumerate(episode.records):
        for option_index, option in enumerate(record.options):
            if option.alias == episode.evidence_alias:
                return pilot.CompilerPrediction(
                    selected,
                    programs,
                    priors,
                    record_index,
                    option_index,
                )
    raise AssertionError("gold evidence alias is absent")


def _primary_record(episode: pilot.PilotEpisode) -> pilot.RecordExample:
    return next(
        record for record in episode.records if record.record_id == episode.primary_record_id
    )


def _refine_packet(
    packet,
    *,
    variable: int,
    confirmed: int,
    valid_assignments: Sequence[tuple[int, ...]],
    evidence_text: str,
):
    verification = verify_nogood(
        packet,
        guard=Guard((Literal(variable, 1 - confirmed),)),
        evidence_commitment=pilot._digest("diverge-neural-evidence", evidence_text),
        valid_assignments=valid_assignments,
    )
    if not verification.accepted or verification.nogood is None:
        return packet, verification
    return append_verified_nogood(packet, verification.nogood), verification


def _bind_delayed_evidence(packet, evidence_text: str):
    """Produce a nogood from only the sealed packet and delayed evidence."""

    match = EVIDENCE_PATTERN.fullmatch(evidence_text.lower())
    if match is None:
        return None
    option_commitment = named_commitment("diverge-neural-option", match.group(1))
    evidence_commitment = pilot._digest("diverge-neural-evidence", evidence_text)
    return certify_binary_option_evidence(
        packet,
        option_commitment=option_commitment,
        evidence_commitment=evidence_commitment,
    )


def _find_underdetermined_query(receipt: ExecutionReceipt) -> Query:
    for slot in (2, 3, 4, 0, 1):
        query = Query("READ_VALUE", (slot,))
        if query_execution(receipt, query).disposition == ABSTAIN:
            return query
    raise AssertionError("board lacks an underdetermined late query")


def build_gate_board(seed: int, repetitions: int) -> tuple[GateEpisode, ...]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    episodes = []
    serial = 0
    for family_index, ontology in enumerate(ONTOLOGIES):
        for renderer in (2, 3):
            for width in range(1, 7):
                for repetition in range(repetitions):
                    item_seed = seed + serial * 1009 + family_index * 100_003
                    split = "development" if renderer == 2 else "confirmation"
                    source = pilot.generate_episode(
                        seed=item_seed,
                        split=split,
                        width=width,
                        renderer=renderer,
                        ontology=ontology,
                    )
                    source = _replace_primary_semantics(
                        source,
                        gold_program=repetition % 2,
                    )
                    truth = _truth_prediction(source)
                    packet, canonical, _ = pilot._build_predicted_packet(source, truth)
                    assert packet is not None and not packet.overflow
                    initial = execute_packet(packet)
                    primary = _primary_record(source)
                    variable = canonical[source.primary_record_id]
                    valid = tuple(
                        assignment
                        for assignment in enumerate_assignments(packet)
                        if assignment[variable] == primary.gold_option
                    )
                    refined_packet, verification = _refine_packet(
                        packet,
                        variable=variable,
                        confirmed=primary.gold_option,
                        valid_assignments=valid,
                        evidence_text=source.evidence_text,
                    )
                    if not verification.accepted:
                        raise AssertionError("truth-board evidence failed verification")
                    refined = execute_packet(refined_packet)
                    sensitive = Query("READ_VALUE", (0,))
                    invariant = Query("SUM_VALUES", (0, 1, 2, 3, 4))
                    underdetermined = _find_underdetermined_query(initial)
                    if query_execution(refined, sensitive).disposition != ANSWER:
                        raise AssertionError("truth board did not resolve sensitive query")
                    if query_execution(initial, invariant).disposition != ANSWER:
                        raise AssertionError("truth board lacks invariant query")
                    episodes.append(
                        GateEpisode(
                            source,
                            ontology,
                            sensitive,
                            invariant,
                            underdetermined,
                        )
                    )
                    serial += 1
    return tuple(episodes)


def _compile_episode(
    model: SmolDivergeRoleCopyCompiler,
    gate: GateEpisode,
    device: torch.device,
) -> CompiledEpisode:
    episode = gate.source
    truth = _truth_prediction(episode)
    true_packet, true_canonical, _ = pilot._build_predicted_packet(episode, truth)
    assert true_packet is not None and not true_packet.overflow
    true_primary = _primary_record(episode)
    true_variable = true_canonical[episode.primary_record_id]
    true_valid = tuple(
        assignment
        for assignment in enumerate_assignments(true_packet)
        if assignment[true_variable] == true_primary.gold_option
    )
    true_refined_packet, true_verification = _refine_packet(
        true_packet,
        variable=true_variable,
        confirmed=true_primary.gold_option,
        valid_assignments=true_valid,
        evidence_text=episode.evidence_text,
    )
    if not true_verification.accepted:
        raise AssertionError("assessor truth refinement failed")
    true_initial = execute_packet(true_packet)
    true_refined = execute_packet(true_refined_packet)
    expected = {
        "sensitive": query_execution(true_refined, gate.sensitive_query),
        "invariant": query_execution(true_initial, gate.invariant_query),
        "underdetermined": query_execution(true_initial, gate.underdetermined_query),
    }

    prediction = predict_source_fields(model, episode, device)
    packet, canonical, _ = pilot._build_predicted_packet(episode, prediction)
    packet_exact = (
        packet is not None
        and not packet.overflow
        and packet_bytes(packet) == packet_bytes(true_packet)
    )
    true_faults = {record.record_id for record in episode.records if record.is_fault_line}
    predicted_faults = {
        record.record_id
        for record, selected in zip(episode.records, prediction.selected, strict=True)
        if selected
    }
    gold_support_recalled = true_faults.issubset(predicted_faults)
    if packet is None or packet.overflow or episode.primary_record_id not in canonical:
        return CompiledEpisode(
            gate=gate,
            packet=packet,
            refined_packet=None,
            prediction=prediction,
            packet_exact=packet_exact,
            gold_support_recalled=gold_support_recalled,
            primary_variable=None,
            evidence_variable=None,
            evidence_option=None,
            primary_gold_option=true_primary.gold_option,
            initial=None,
            refined=None,
            factorized_initial=None,
            factorized_refined=None,
            expected=expected,
            verifier_calls=0,
            valid_support_preserved=False,
        )
    certificate = _bind_delayed_evidence(packet, episode.evidence_text)
    if certificate is None:
        evidence_variable = None
        evidence_option = None
    else:
        evidence_variable = certificate.variable_id
        evidence_option = certificate.confirmed_option
    initial = execute_packet(packet, commute_disjoint=True)
    factorized_initial = execute_packet_factorized(packet)
    refined = None
    factorized_refined = None
    refined_packet = None
    verifier_calls = 0
    valid_support_preserved = False
    if certificate is not None:
        candidate_primary = canonical[episode.primary_record_id]
        valid = tuple(
            assignment
            for assignment in enumerate_assignments(packet)
            if assignment[candidate_primary] == true_primary.gold_option
        )
        verification = verify_nogood(
            packet,
            guard=certificate.nogood.guard,
            evidence_commitment=certificate.nogood.evidence_commitment,
            valid_assignments=valid,
        )
        verifier_calls = 1
        refined_packet = append_verified_nogood(packet, certificate.nogood)
        refined = execute_packet(refined_packet, commute_disjoint=True)
        factorized_refined = refine_factorized_receipt(
            refined_packet,
            factorized_initial,
        )
        valid_support_preserved = (
            verification.accepted
            and set(enumerate_assignments(refined_packet)) == set(valid)
        )
    return CompiledEpisode(
        gate=gate,
        packet=packet,
        refined_packet=refined_packet,
        prediction=prediction,
        packet_exact=packet_exact,
        gold_support_recalled=gold_support_recalled,
        primary_variable=canonical[episode.primary_record_id],
        evidence_variable=evidence_variable,
        evidence_option=evidence_option,
        primary_gold_option=true_primary.gold_option,
        initial=initial,
        refined=refined,
        factorized_initial=factorized_initial,
        factorized_refined=factorized_refined,
        expected=expected,
        verifier_calls=verifier_calls,
        valid_support_preserved=valid_support_preserved,
    )


def _receipt(worlds: Iterable[WorldResult]) -> ExecutionReceipt:
    return ExecutionReceipt(tuple(worlds), 0, 0, False)


def _single_world_decision(world: WorldResult | None, query: Query) -> QueryDecision:
    return query_execution(_receipt(() if world is None else (world,)), query)


def _filter_evidence(
    worlds: Iterable[WorldResult],
    variable: int | None,
    option: int | None,
) -> tuple[WorldResult, ...]:
    if variable is None or option is None:
        return ()
    return tuple(
        world
        for world in worlds
        if variable < len(world.assignment) and world.assignment[variable] == option
    )


def _world_transaction_count(packet, world: WorldResult) -> int:
    return sum(patch.guard.matches(world.assignment) for patch in packet.patches)


def _select_full_particles(compiled: CompiledEpisode) -> tuple[tuple[WorldResult, ...], dict[str, int]]:
    if (
        compiled.packet is None
        or compiled.initial is None
        or compiled.factorized_initial is None
    ):
        return (), {"particles": 0, "bytes": 0, "transactions": 0}
    accounting = account_packet(compiled.packet, compiled.initial)
    factorized = compiled.factorized_initial
    byte_budget = accounting.packet_bytes + factorized.peak_group_bytes
    transaction_budget = factorized.unique_transactions
    if compiled.refined_packet is not None and compiled.factorized_refined is not None:
        byte_budget = max(
            byte_budget,
            len(packet_bytes(compiled.refined_packet))
            + compiled.factorized_refined.peak_group_bytes,
        )
        transaction_budget = max(
            transaction_budget,
            compiled.factorized_refined.unique_transactions,
        )
    ordered = sorted(compiled.initial.worlds, key=lambda world: (-world.mass, world.assignment))
    selected = []
    used_bytes = 0
    used_transactions = 0
    for world in ordered:
        byte_cost = materialized_world_bytes(compiled.packet, world)
        transaction_cost = _world_transaction_count(compiled.packet, world)
        if (
            used_bytes + byte_cost <= byte_budget
            and used_transactions + transaction_cost <= transaction_budget
        ):
            selected.append(world)
            used_bytes += byte_cost
            used_transactions += transaction_cost
    return tuple(selected), {
        "particles": len(selected),
        "bytes": used_bytes,
        "transactions": used_transactions,
    }


def _sample_independent_particles(
    compiled: CompiledEpisode,
    count: int,
    seed: int,
) -> tuple[WorldResult, ...]:
    if compiled.initial is None or count <= 0:
        return ()
    worlds = compiled.initial.worlds
    masses = [world.mass for world in worlds]
    return tuple(random.Random(seed).choices(worlds, weights=masses, k=count))


def _mean_field_probabilities(packet, evidence: tuple[int, int] | None) -> list[list[float]]:
    probabilities = [
        [1.0 / len(variable.options)] * len(variable.options) for variable in packet.variables
    ]
    for factor in packet.support_factors:
        if len(factor.scope) != 1:
            raise ValueError("soft control supports unary factors only")
        variable = factor.scope[0]
        masses = [0.0] * len(packet.variables[variable].options)
        for row, mass in factor.masses:
            masses[row[0]] += float(mass)
        total = sum(masses)
        probabilities[variable] = [mass / total for mass in masses]
    if evidence is not None:
        variable, option = evidence
        probabilities[variable] = [
            float(index == option) for index in range(len(probabilities[variable]))
        ]
    return probabilities


def _mean_field_state(packet, evidence: tuple[int, int] | None) -> list[float]:
    probabilities = _mean_field_probabilities(packet, evidence)
    state = [float(cell.value) for cell in packet.shared_state.cells]
    for patch in packet.patches:
        probability = 1.0
        for literal in patch.guard.literals:
            probability *= probabilities[literal.variable_id][literal.option]
        updated = list(state)
        opcode = patch.transaction.opcode
        args = patch.transaction.arguments
        if opcode == "SET_VALUE":
            updated[args[0]] = float(args[1])
        elif opcode == "ADD_VALUE":
            updated[args[0]] += float(args[1])
        elif opcode == "COPY_VALUE":
            updated[args[1]] = state[args[0]]
        elif opcode == "SWAP_VALUE":
            updated[args[0]], updated[args[1]] = state[args[1]], state[args[0]]
        else:
            raise ValueError(f"soft control does not implement {opcode}")
        state = [
            probability * new + (1.0 - probability) * old
            for old, new in zip(state, updated, strict=True)
        ]
    return state


def _mean_field_decision(packet, query: Query, evidence: tuple[int, int] | None) -> QueryDecision:
    state = _mean_field_state(packet, evidence)
    if query.opcode == "READ_VALUE":
        answer = round(state[query.arguments[0]])
    elif query.opcode == "SUM_VALUES":
        answer = round(sum(state[slot] for slot in query.arguments))
    else:
        return QueryDecision(REJECT, None, (), 0)
    return QueryDecision(ANSWER, answer, ((answer, 1),), 1)


def _decisions_for_arm(
    arm: str,
    compiled: CompiledEpisode,
    *,
    particle_seed: int,
) -> tuple[dict[str, QueryDecision], dict[str, int]]:
    gate = compiled.gate
    queries = {
        "sensitive": gate.sensitive_query,
        "invariant": gate.invariant_query,
        "underdetermined": gate.underdetermined_query,
    }
    empty = {kind: QueryDecision(REJECT, None, (), 0) for kind in QUERY_KINDS}
    if compiled.packet is None or compiled.initial is None or not compiled.initial.worlds:
        return empty, {"particles": 0, "bytes": 0, "transactions": 0}
    top = min(compiled.initial.worlds, key=lambda world: (-world.mass, world.assignment))
    evidence = (compiled.evidence_variable, compiled.evidence_option)
    if arm in {"A_single", "D_recurrent_single"}:
        sensitive_worlds = _filter_evidence((top,), *evidence)
        decisions = {
            "sensitive": query_execution(_receipt(sensitive_worlds), queries["sensitive"]),
            "invariant": _single_world_decision(top, queries["invariant"]),
            "underdetermined": _single_world_decision(top, queries["underdetermined"]),
        }
        accounting = account_packet(compiled.packet, compiled.initial)
        transactions = (
            _world_transaction_count(compiled.packet, top)
            if arm == "A_single"
            else accounting.unique_transactions
        )
        return decisions, {
            "particles": 1,
            "bytes": materialized_world_bytes(compiled.packet, top),
            "transactions": transactions,
        }
    if arm == "B_full_particles":
        worlds, resources = _select_full_particles(compiled)
        return {
            "sensitive": query_execution(
                _receipt(_filter_evidence(worlds, *evidence)), queries["sensitive"]
            ),
            "invariant": query_execution(_receipt(worlds), queries["invariant"]),
            "underdetermined": query_execution(
                _receipt(worlds), queries["underdetermined"]
            ),
        }, resources
    if arm == "C_independent":
        _, full_resources = _select_full_particles(compiled)
        worlds = _sample_independent_particles(
            compiled, full_resources["particles"], particle_seed
        )
        resources = {
            "particles": len(worlds),
            "bytes": sum(
                materialized_world_bytes(compiled.packet, world) for world in worlds
            ),
            "transactions": sum(
                _world_transaction_count(compiled.packet, world) for world in worlds
            ),
        }
        return {
            "sensitive": query_execution(
                _receipt(_filter_evidence(worlds, *evidence)), queries["sensitive"]
            ),
            "invariant": query_execution(_receipt(worlds), queries["invariant"]),
            "underdetermined": query_execution(
                _receipt(worlds), queries["underdetermined"]
            ),
        }, resources
    if arm == "E_soft":
        return {
            "sensitive": _mean_field_decision(compiled.packet, queries["sensitive"], evidence),
            "invariant": _mean_field_decision(compiled.packet, queries["invariant"], None),
            "underdetermined": _mean_field_decision(
                compiled.packet, queries["underdetermined"], None
            ),
        }, {
            "particles": 0,
            "bytes": len(compiled.packet.shared_state.cells) * 8,
            "transactions": len(compiled.packet.patches),
        }
    if arm == "F_factorized_no_conflict":
        if compiled.factorized_initial is None:
            return empty, {"particles": 0, "bytes": 0, "transactions": 0}
        factorized = compiled.factorized_initial
        return {
            "sensitive": factorized_query_execution(
                compiled.packet, factorized, queries["sensitive"]
            ),
            "invariant": factorized_query_execution(
                compiled.packet, factorized, queries["invariant"]
            ),
            "underdetermined": factorized_query_execution(
                compiled.packet, factorized, queries["underdetermined"]
            ),
        }, {
            "particles": 0,
            "state_groups": len(factorized.groups),
            "bytes": len(packet_bytes(compiled.packet)) + factorized.peak_group_bytes,
            "transactions": factorized.unique_transactions,
            "logical_transaction_applications": factorized.logical_transaction_applications,
            "mask_operations": factorized.mask_operations,
        }
    if arm == "G_diverge":
        if compiled.factorized_initial is None:
            return empty, {"particles": 0, "bytes": 0, "transactions": 0}
        factorized = compiled.factorized_initial
        sensitive = (
            QueryDecision(REJECT, None, (), 0)
            if compiled.factorized_refined is None
            else factorized_query_execution(
                compiled.packet,
                compiled.factorized_refined,
                queries["sensitive"],
            )
        )
        runtime_bytes = len(packet_bytes(compiled.packet)) + factorized.peak_group_bytes
        if compiled.refined_packet is not None and compiled.factorized_refined is not None:
            runtime_bytes = max(
                runtime_bytes,
                len(packet_bytes(compiled.refined_packet))
                + compiled.factorized_refined.peak_group_bytes,
            )
        return {
            "sensitive": sensitive,
            "invariant": factorized_query_execution(
                compiled.packet, factorized, queries["invariant"]
            ),
            "underdetermined": factorized_query_execution(
                compiled.packet, factorized, queries["underdetermined"]
            ),
        }, {
            "particles": 0,
            "state_groups": len(factorized.groups),
            "bytes": runtime_bytes,
            "transactions": factorized.unique_transactions,
            "logical_transaction_applications": factorized.logical_transaction_applications,
            "mask_operations": factorized.mask_operations,
        }
    raise ValueError(f"unknown arm: {arm}")


def _decision_correct(predicted: QueryDecision, expected: QueryDecision) -> bool:
    return predicted.disposition == expected.disposition and predicted.answer == expected.answer


def _intervention_decisions(
    compiled: CompiledEpisode,
    name: str,
    *,
    swapped: CompiledEpisode | None = None,
) -> dict[str, QueryDecision]:
    gate = compiled.gate
    if compiled.packet is None or compiled.initial is None:
        return {kind: QueryDecision(REJECT, None, (), 0) for kind in QUERY_KINDS}
    if name == "conflict_disabled":
        if compiled.factorized_initial is None:
            return {kind: QueryDecision(REJECT, None, (), 0) for kind in QUERY_KINDS}
        return {
            "sensitive": factorized_query_execution(
                compiled.packet, compiled.factorized_initial, gate.sensitive_query
            ),
            "invariant": factorized_query_execution(
                compiled.packet, compiled.factorized_initial, gate.invariant_query
            ),
            "underdetermined": factorized_query_execution(
                compiled.packet, compiled.factorized_initial, gate.underdetermined_query
            ),
        }
    if name == "shuffled_guard_provenance":
        if compiled.evidence_variable is not None and compiled.evidence_option is not None:
            if compiled.primary_variable is None:
                raise AssertionError("shuffled-guard control lacks primary variable")
            valid = tuple(
                assignment
                for assignment in enumerate_assignments(compiled.packet)
                if assignment[compiled.primary_variable] == compiled.primary_gold_option
            )
            invalid = verify_nogood(
                compiled.packet,
                guard=Guard((Literal(compiled.evidence_variable, compiled.evidence_option),)),
                evidence_commitment=pilot._digest(
                    "diverge-neural-shuffled-evidence",
                    compiled.gate.source.evidence_text,
                ),
                valid_assignments=valid,
            )
            if invalid.accepted:
                raise AssertionError("shuffled provenance removed valid support")
        if compiled.factorized_initial is None:
            return {kind: QueryDecision(REJECT, None, (), 0) for kind in QUERY_KINDS}
        return {
            "sensitive": factorized_query_execution(
                compiled.packet, compiled.factorized_initial, gate.sensitive_query
            ),
            "invariant": factorized_query_execution(
                compiled.packet, compiled.factorized_initial, gate.invariant_query
            ),
            "underdetermined": factorized_query_execution(
                compiled.packet, compiled.factorized_initial, gate.underdetermined_query
            ),
        }
    if name == "packet_swap":
        if (
            swapped is None
            or swapped.factorized_initial is None
            or swapped.packet is None
        ):
            return {kind: QueryDecision(REJECT, None, (), 0) for kind in QUERY_KINDS}
        if swapped.packet.source_commitment == compiled.packet.source_commitment:
            raise AssertionError("packet-swap control reused the same source packet")
        return {
            "sensitive": factorized_query_execution(
                swapped.packet, swapped.factorized_initial, gate.sensitive_query
            ),
            "invariant": factorized_query_execution(
                swapped.packet, swapped.factorized_initial, gate.invariant_query
            ),
            "underdetermined": factorized_query_execution(
                swapped.packet, swapped.factorized_initial, gate.underdetermined_query
            ),
        }
    if name == "forced_premature_top1":
        return _decisions_for_arm("A_single", compiled, particle_seed=0)[0]
    if name == "state_reset":
        reset_worlds = tuple(
            dataclasses.replace(world, state=compiled.packet.shared_state, contradiction=False)
            for world in compiled.initial.worlds
        )
        sensitive_worlds = _filter_evidence(
            reset_worlds, compiled.evidence_variable, compiled.evidence_option
        )
        return {
            "sensitive": query_execution(_receipt(sensitive_worlds), gate.sensitive_query),
            "invariant": query_execution(_receipt(reset_worlds), gate.invariant_query),
            "underdetermined": query_execution(
                _receipt(reset_worlds), gate.underdetermined_query
            ),
        }
    raise ValueError(name)


def _query_only_decisions() -> dict[str, QueryDecision]:
    return {
        "sensitive": QueryDecision(ANSWER, 10, ((10, 1),), 1),
        "invariant": QueryDecision(ANSWER, 104, ((104, 1),), 1),
        "underdetermined": QueryDecision(ABSTAIN, None, (), 1),
    }


def _shuffled_label_expected(expected: dict[str, QueryDecision]) -> dict[str, QueryDecision]:
    sensitive = expected["sensitive"]
    if sensitive.disposition != ANSWER or sensitive.answer not in {10, 13}:
        raise AssertionError("sensitive label is outside the frozen pair")
    invariant = expected["invariant"]
    if invariant.disposition != ANSWER or invariant.answer is None:
        raise AssertionError("invariant label is not exact")
    return {
        "sensitive": QueryDecision(
            ANSWER,
            13 if sensitive.answer == 10 else 10,
            (),
            sensitive.total_mass,
        ),
        "invariant": QueryDecision(
            ANSWER,
            invariant.answer + 1,
            (),
            invariant.total_mass,
        ),
        "underdetermined": QueryDecision(ANSWER, 0, ((0, 1),), 1),
    }


def _reconstruct_training_receipt(
    model: SmolDivergeRoleCopyCompiler,
    source_report: dict[str, object],
) -> dict[str, int]:
    arguments = source_report["arguments"]
    updates = int(arguments["updates"])
    batch_size = int(arguments["batch_size"])
    data_seed = int(arguments["data_seed"])
    smol_token_positions = 0
    alias_character_positions = 0
    record_sequences = 0
    option_sequences = 0
    for update in range(1, updates + 1):
        for index in range(batch_size):
            episode = pilot.generate_episode(
                seed=data_seed + update * batch_size + index,
                split="train",
                width=1 + ((update + index) % 4),
                renderer=(update + index) % 2,
                ontology="register-workshop",
            )
            aliases = [
                option.alias for record in episode.records for option in record.options
            ]
            alias_character_positions += sum(len(pilot.char_ids(alias)) for alias in aliases)
            alias_character_positions += len(pilot.char_ids(episode.evidence_alias))
            for record in episode.records:
                record_aliases = tuple(option.alias for option in record.options)
                texts = (
                    record.text,
                    pilot._render_record(
                        episode.ontology,
                        record.options[0],
                        record.options[1],
                        is_fault_line=not record.is_fault_line,
                        renderer=episode.renderer,
                    ),
                )
                for text in texts:
                    smol_token_positions += len(model.encode(text, record_aliases).ids)
                    record_sequences += 1
                if record.is_fault_line:
                    for option in record.options:
                        smol_token_positions += len(
                            model.encode(option.text, (option.alias,)).ids
                        )
                        option_sequences += 1
    return {
        "updates": updates,
        "charged_episodes": updates * batch_size,
        "smol_token_positions": smol_token_positions,
        "alias_character_positions": alias_character_positions,
        "record_sequences": record_sequences,
        "option_sequences": option_sequences,
    }


def _profile_training_step(
    model: SmolDivergeRoleCopyCompiler,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> dict[str, object]:
    """Profile a representative source-compiler forward/backward step."""

    if device.type != "cuda":
        return {"available": False, "reason": "CUDA profiler required"}
    from diverge_v0_role_copy_pilot import training_batch

    episodes = [
        pilot.generate_episode(
            seed=seed + index,
            split="train",
            width=1 + (index % 4),
            renderer=index % 2,
            ontology="register-workshop",
        )
        for index in range(batch_size)
    ]
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as profile:
        loss, _ = training_batch(episodes, model, device)
        loss.backward()
        torch.cuda.synchronize()
    flops = sum(int(event.flops or 0) for event in profile.key_averages())
    peak = torch.cuda.max_memory_allocated(device)
    model.zero_grad(set_to_none=True)
    return {
        "available": True,
        "profiled_batch_size": batch_size,
        "forward_backward_flops": flops,
        "peak_device_memory_bytes": peak,
        "method": "torch.profiler operator FLOP estimates; unsupported operators are not counted",
    }


def _load_model(args: argparse.Namespace, device: torch.device):
    backbone, _, receipt = load_frozen_pointer_backbone(args.base, device=device)
    model = SmolDivergeRoleCopyCompiler(
        backbone,
        Tokenizer.from_file(str(args.tokenizer)),
        layer=args.layer,
        width=args.width,
        char_width=args.char_width,
    ).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith("backbone.") for key in incompatible.missing_keys
    ):
        raise RuntimeError(f"incompatible role-copy checkpoint: {incompatible}")
    model.eval()
    return model, receipt


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, base_receipt = _load_model(args, device)
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source_report.get("checkpoint_sha256") != _sha256(args.checkpoint):
        raise RuntimeError("source report does not bind the supplied compiler checkpoint")
    if int(source_report["arguments"]["seed"]) != args.seed:
        raise RuntimeError("source report seed differs from gate seed")
    training_receipt = _reconstruct_training_receipt(model, source_report)
    profile = _profile_training_step(model, device, args.board_seed, args.profile_batch_size)
    model.eval()
    board = build_gate_board(args.board_seed, args.repetitions)
    started = time.perf_counter()
    compiled = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for index, gate in enumerate(board):
            compiled.append(_compile_episode(model, gate, device))
            if (index + 1) % 32 == 0:
                print(json.dumps({"compiled": index + 1, "total": len(board)}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        inference_peak = torch.cuda.max_memory_allocated(device)
    else:
        inference_peak = 0
    compile_seconds = time.perf_counter() - started

    arm_totals = {arm: Counter() for arm in ARM_NAMES}
    family_totals = {
        arm: {family: Counter() for family in ONTOLOGIES} for arm in ARM_NAMES
    }
    query_totals = {arm: {kind: Counter() for kind in QUERY_KINDS} for arm in ARM_NAMES}
    resources = {arm: Counter() for arm in ARM_NAMES}
    resource_peaks = {arm: Counter() for arm in ARM_NAMES}
    intervention_names = (
        "conflict_disabled",
        "forced_premature_top1",
        "shuffled_guard_provenance",
        "packet_swap",
        "state_reset",
        "query_only",
        "shuffled_labels",
    )
    interventions = {name: Counter() for name in intervention_names}
    packet_accounts = []
    source_deletion_identical = True
    for index, item in enumerate(compiled):
        for arm in ARM_NAMES:
            arm_started = time.perf_counter_ns()
            decisions, receipt = _decisions_for_arm(
                arm,
                item,
                particle_seed=args.seed + index * 7919,
            )
            receipt["wall_nanoseconds"] = time.perf_counter_ns() - arm_started
            for key, value in receipt.items():
                resources[arm][key] += value
                resource_peaks[arm][key] = max(resource_peaks[arm][key], value)
            for kind in QUERY_KINDS:
                correct = int(_decision_correct(decisions[kind], item.expected[kind]))
                arm_totals[arm]["correct"] += correct
                arm_totals[arm]["queries"] += 1
                family_totals[arm][item.gate.family]["correct"] += correct
                family_totals[arm][item.gate.family]["queries"] += 1
                query_totals[arm][kind]["correct"] += correct
                query_totals[arm][kind]["queries"] += 1
                if item.expected[kind].disposition == ABSTAIN:
                    arm_totals[arm]["false_certificates"] += int(
                        decisions[kind].disposition == ANSWER
                    )
        g_before = _decisions_for_arm(
            "G_diverge", item, particle_seed=args.seed + index
        )[0]
        g_after_source_delete = _decisions_for_arm(
            "G_diverge", dataclasses.replace(item, gate=dataclasses.replace(
                item.gate,
                source=dataclasses.replace(item.gate.source, source_text=""),
            )), particle_seed=args.seed + index
        )[0]
        source_deletion_identical &= g_before == g_after_source_delete
        swapped = compiled[(index + 1) % len(compiled)]
        for name in intervention_names:
            if name == "query_only":
                decisions = _query_only_decisions()
                expected = item.expected
            elif name == "shuffled_labels":
                decisions = g_before
                expected = _shuffled_label_expected(item.expected)
            else:
                decisions = _intervention_decisions(item, name, swapped=swapped)
                expected = item.expected
            for kind in QUERY_KINDS:
                interventions[name]["correct"] += int(
                    _decision_correct(decisions[kind], expected[kind])
                )
                interventions[name]["queries"] += 1
        if (
            item.packet is not None
            and item.initial is not None
            and item.factorized_initial is not None
        ):
            packet_accounts.append(
                (account_packet(item.packet, item.initial), item.factorized_initial)
            )

    episodes = len(compiled)
    compiler = {
        "episodes": episodes,
        "packet_exact": sum(item.packet_exact for item in compiled) / episodes,
        "gold_support_recall": sum(item.gold_support_recalled for item in compiled) / episodes,
        "valid_support_preserved": sum(item.valid_support_preserved for item in compiled) / episodes,
        "verifier_calls": sum(item.verifier_calls for item in compiled),
    }
    arm_reports = {}
    for arm in ARM_NAMES:
        total = arm_totals[arm]
        arm_reports[arm] = {
            "exact": total["correct"] / max(1, total["queries"]),
            "correct": total["correct"],
            "queries": total["queries"],
            "false_certificates": total["false_certificates"],
            "by_family": {
                family: values["correct"] / max(1, values["queries"])
                for family, values in family_totals[arm].items()
            },
            "by_query": {
                kind: values["correct"] / max(1, values["queries"])
                for kind, values in query_totals[arm].items()
            },
            "resource_totals": dict(resources[arm]),
            "resource_peaks": dict(resource_peaks[arm]),
        }
    g_exact = arm_reports["G_diverge"]["exact"]
    intervention_reports = {
        name: {
            "exact": values["correct"] / max(1, values["queries"]),
            "drop_points_from_G": 100.0 * (
                g_exact - values["correct"] / max(1, values["queries"])
            ),
        }
        for name, values in interventions.items()
    }
    majority_by_query = {}
    for kind in QUERY_KINDS:
        labels = Counter(
            (item.expected[kind].disposition, item.expected[kind].answer)
            for item in compiled
        )
        majority_by_query[kind] = max(labels.values()) / episodes
    empirical_majority = sum(majority_by_query.values()) / len(QUERY_KINDS)
    strongest_control = max(arm_reports[arm]["exact"] for arm in ARM_NAMES[:-1])
    width_accounts = defaultdict(list)
    for item, accounting in zip(
        (
            item
            for item in compiled
            if item.packet is not None
            and item.initial is not None
            and item.factorized_initial is not None
        ),
        packet_accounts,
        strict=True,
    ):
        width_accounts[len(item.packet.variables)].append(accounting)
    sharing = {
        str(2 ** width): {
            "mean_packet_bytes": sum(row[0].packet_bytes for row in rows) / len(rows),
            "mean_peak_group_bytes": sum(row[1].peak_group_bytes for row in rows)
            / len(rows),
            "mean_effective_factorized_bytes": sum(
                row[0].packet_bytes + row[1].peak_group_bytes for row in rows
            )
            / len(rows),
            "mean_materialized_bytes": sum(
                row[0].materialized_world_bytes for row in rows
            )
            / len(rows),
            "storage_ratio": sum(row[0].materialized_world_bytes for row in rows)
            / sum(row[0].packet_bytes + row[1].peak_group_bytes for row in rows),
            "static_packet_storage_ratio": sum(
                row[0].materialized_world_bytes for row in rows
            )
            / sum(row[0].packet_bytes for row in rows),
            "transaction_ratio": sum(row[0].duplicated_transactions for row in rows)
            / max(1, sum(row[1].unique_transactions for row in rows)),
            "mean_peak_state_groups": sum(row[1].peak_groups for row in rows)
            / len(rows),
            "mean_mask_operations": sum(row[1].mask_operations for row in rows)
            / len(rows),
        }
        for width, rows in sorted(width_accounts.items())
    }
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.adapter_parameters())
    return {
        "schema": SCHEMA,
        "status": "complete_matched_gate",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "inputs": {
            "base_sha256": _sha256(args.base),
            "tokenizer_sha256": _sha256(args.tokenizer),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "source_report_sha256": _sha256(args.source_report),
            "base_import": base_receipt.base_import,
        },
        "board": {
            "episodes": episodes,
            "queries": episodes * len(QUERY_KINDS),
            "families": list(ONTOLOGIES),
            "widths": [2, 4, 8, 16, 32, 64],
            "renderers": [2, 3],
            "repetitions": args.repetitions,
        },
        "compiler": compiler,
        "arms": arm_reports,
        "interventions": intervention_reports,
        "sharing": sharing,
        "resources": {
            "same_parameters_all_arms": True,
            "same_training_data_and_tokens_all_arms": True,
            "same_source_compiler_all_arms": True,
            "exact_training_charge": training_receipt,
            "total_model_parameters": total_parameters,
            "source_compiler_trainable_parameters": trainable_parameters,
            "downstream_trainable_parameters_each_arm": 0,
            "profiled_training_step": profile,
            "compiler_inference_peak_device_memory_bytes": inference_peak,
            "compiler_inference_wall_seconds": compile_seconds,
            "compiler_inference_episodes_per_second": episodes / compile_seconds,
            "executor_flop_proxy": "exact guarded transaction applications",
            "activation_memory_proxy": (
                "canonical packet plus measured peak factorized state-group bytes "
                "or complete whole-particle bytes"
            ),
        },
        "gate": {
            "compile_gold_support_recall": compiler["gold_support_recall"] == 1.0,
            "packet_exact": compiler["packet_exact"] == 1.0,
            "valid_support_recall": compiler["valid_support_preserved"] == 1.0,
            "zero_G_false_certificates": arm_reports["G_diverge"]["false_certificates"] == 0,
            "G_wrong_top1_recovery_at_least_90": (
                arm_reports["G_diverge"]["by_query"]["sensitive"] >= 0.90
            ),
            "G_gain_over_strongest_control_at_least_10_points": (
                100.0 * (g_exact - strongest_control) >= 10.0
            ),
            "G_gain_each_family_at_least_5_points": all(
                100.0 * (
                    arm_reports["G_diverge"]["by_family"][family]
                    - max(
                        arm_reports[arm]["by_family"][family]
                        for arm in ARM_NAMES[:-1]
                    )
                )
                >= 5.0
                for family in ONTOLOGIES
            ),
            "packet_64_world_bytes_at_most_8192": sharing.get("64", {}).get(
                "mean_packet_bytes", float("inf")
            )
            <= 8192,
            "storage_advantage_4_worlds_at_least_2": sharing.get("4", {}).get(
                "storage_ratio", 0
            )
            >= 2.0,
            "storage_advantage_32_worlds_at_least_10": sharing.get("32", {}).get(
                "storage_ratio", 0
            )
            >= 10.0,
            "transaction_advantage_8_worlds_at_least_2": sharing.get("8", {}).get(
                "transaction_ratio", 0
            )
            >= 2.0,
            "transaction_advantage_64_worlds_at_least_5": sharing.get("64", {}).get(
                "transaction_ratio", 0
            )
            >= 5.0,
            "conflict_disabled_drop_at_least_5": intervention_reports[
                "conflict_disabled"
            ]["drop_points_from_G"]
            >= 5.0,
            "forced_top1_drop_at_least_20": intervention_reports[
                "forced_premature_top1"
            ]["drop_points_from_G"]
            >= 20.0,
            "shuffled_guard_drop_at_least_20": intervention_reports[
                "shuffled_guard_provenance"
            ]["drop_points_from_G"]
            >= 20.0,
            "packet_swap_drop_at_least_20": intervention_reports["packet_swap"][
                "drop_points_from_G"
            ]
            >= 20.0,
            "state_reset_drop_at_least_20": intervention_reports["state_reset"][
                "drop_points_from_G"
            ]
            >= 20.0,
            "query_only_at_most_5_above_empirical_majority": (
                100.0
                * (intervention_reports["query_only"]["exact"] - empirical_majority)
                <= 5.0
            ),
            "shuffled_labels_at_most_5_above_empirical_majority": (
                100.0
                * (intervention_reports["shuffled_labels"]["exact"] - empirical_majority)
                <= 5.0
            ),
            "source_deletion_bit_identical": source_deletion_identical,
        },
        "empirical_majority": {
            "by_query": majority_by_query,
            "aggregate": empirical_majority,
        },
        "claim_boundary": (
            "Synthetic delayed-disambiguation mechanism gate with a learned finite source compiler. "
            "Not unrestricted language reasoning, public benchmark capability, or long-run scaling evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--board-seed", type=int, default=202608056000)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--char-width", type=int, default=48)
    parser.add_argument("--profile-batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_gate(args)
    report["gate"]["pass_single_seed"] = all(report["gate"].values())
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "gate": report["gate"],
                "arms": {
                    arm: values["exact"] for arm, values in report["arms"].items()
                },
            },
            sort_keys=True,
        )
    )
    if not report["gate"]["pass_single_seed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
