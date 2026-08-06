#!/usr/bin/env python3
"""Exact source-free runtime checks for DIVERGE-TFS1."""

from __future__ import annotations

import hashlib
import random

from diverge_tfs1_data import generate_row, steps_from_record
from diverge_tfs1_runtime import (
    ABSTAIN,
    ANSWER,
    REJECT,
    AnchorProvenance,
    CompiledPacket,
    CompiledQuery,
    FaultLine,
    PacketStep,
    all_particle_bytes,
    enumerate_packet,
    execute_factorized,
    factorized_total_bytes,
    particle_capacity_for_bytes,
    query_particles,
    query_receipt,
    ranked_assignments,
    receipt_extensional_map,
)


def _packet(row: dict[str, object]) -> CompiledPacket:
    output = []
    for step in steps_from_record(row["steps"]):  # type: ignore[arg-type]
        clause_sha256 = hashlib.sha256(step.text.encode("ascii")).hexdigest()
        if step.fixed is not None:
            output.append(PacketStep(clause_sha256, fixed=step.fixed))
            continue
        assert step.options is not None
        assert step.fault_index is not None
        output.append(
            PacketStep(
                clause_sha256,
                fault=FaultLine(
                    step.fault_index,
                    step.options,
                    (
                        AnchorProvenance(clause_sha256, 0, 1, 2.0),
                        AnchorProvenance(clause_sha256, 2, 3, 1.0),
                    ),
                ),
            )
        )
    return CompiledPacket(
        str(row["source_commitment"]),
        "0" * 64,
        tuple(row["symbols"]),  # type: ignore[arg-type]
        tuple(output),
    )


def _query(packet: CompiledPacket, row: dict[str, object], name: str) -> CompiledQuery:
    text = str(row["queries"][name])  # type: ignore[index]
    register = str(row["query_registers"][name])  # type: ignore[index]
    digest = hashlib.sha256(text.encode("ascii")).hexdigest()
    return CompiledQuery(
        packet.commitment,
        digest,
        register,
        AnchorProvenance(digest, 0, 1, 1.0),
    )


def main() -> None:
    row = generate_row(random.Random(2026080607), index=0)
    packet = _packet(row)
    sensitive = _query(packet, row, "sensitive")
    invariant = _query(packet, row, "invariant")
    underdetermined = _query(packet, row, "underdetermined")

    no_evidence = execute_factorized(packet)
    assert not no_evidence.rejected
    assert enumerate_packet(packet) == receipt_extensional_map(no_evidence)
    assert query_receipt(packet, no_evidence, sensitive).disposition == ABSTAIN
    assert query_receipt(packet, no_evidence, invariant).disposition == ANSWER

    full = execute_factorized(packet, row["evidence"])  # type: ignore[arg-type]
    full_answer = query_receipt(packet, full, sensitive)
    assert full_answer.disposition == ANSWER
    assert full_answer.answer == row["gold_answer"]
    assert full.represented_worlds == 1
    assert full.logical_instruction_applications > full.unique_instruction_applications

    partial = execute_factorized(packet, row["evidence"][:-1])  # type: ignore[index]
    assert query_receipt(packet, partial, underdetermined).disposition == ABSTAIN

    shifted = execute_factorized(
        packet,
        row["evidence"],  # type: ignore[arg-type]
        shift_fault_operations=True,
    )
    reset = execute_factorized(
        packet,
        row["evidence"],  # type: ignore[arg-type]
        reset_after_declarations=True,
    )
    assert query_receipt(packet, shifted, sensitive).disposition == REJECT
    assert query_receipt(packet, reset, sensitive).disposition == REJECT

    ranked = ranked_assignments(packet)
    top1 = query_particles(
        packet,
        sensitive,
        ranked[:1],
        row["evidence"],  # type: ignore[arg-type]
    )
    assert top1.disposition == REJECT
    factorized_bytes = factorized_total_bytes(
        packet,
        full,
        row["evidence"],  # type: ignore[arg-type]
    )
    particle_bytes = all_particle_bytes(
        packet,
        row["evidence"],  # type: ignore[arg-type]
    )
    capacity, used = particle_capacity_for_bytes(
        packet,
        ranked,
        row["evidence"],  # type: ignore[arg-type]
        factorized_bytes,
    )
    assert particle_bytes >= 2 * factorized_bytes
    assert capacity >= 1 and used <= factorized_bytes

    poisoned = row["source"] + " poisoned"
    assert poisoned not in repr(packet.record())
    assert query_receipt(packet, full, sensitive) == full_answer
    print("diverge TFS1 runtime tests passed")


if __name__ == "__main__":
    main()
