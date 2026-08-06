#!/usr/bin/env python3
"""Focused learned-interface and sealing tests for DIVERGE-NVE1."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib

import torch

from diverge_nve1_data import generate_training_records
from diverge_nve1_runtime import (
    EvidenceCompilerConfig,
    NaturalEvidenceCompiler,
    NaturalEvidenceReceipt,
    canonical_sha256,
    execute_natural_evidence,
    mutate_receipt,
    seal_natural_evidence,
    tensorize_sources,
)
from diverge_tfs1_runtime import (
    AnchorProvenance,
    CompiledPacket,
    FaultLine,
    PacketStep,
)
from diverge_tol1_ir import Action, Atom, Instruction


def _packet() -> CompiledPacket:
    symbols = ("apricot", "beacon", "canvas", "dahlia", "equinox")
    steps = []
    for symbol in symbols:
        instruction = Instruction(
            "SET", action=Action("SET", symbol, Atom("CONST", "1"))
        )
        steps.append(PacketStep("1" * 64, fixed=instruction))
    provenance = (
        AnchorProvenance("2" * 64, 0, 3, 1.0),
        AnchorProvenance("2" * 64, 4, 7, 1.0),
    )
    for index in range(12):
        options = (
            Instruction("ADD", action=Action("ADD", "apricot", Atom("CONST", "1"))),
            Instruction(
                "SUBTRACT",
                action=Action("SUBTRACT", "apricot", Atom("CONST", "1")),
            ),
        )
        steps.append(PacketStep("3" * 64, fault=FaultLine(index, options, provenance)))
    return CompiledPacket("4" * 64, "5" * 64, symbols, tuple(steps))


def _receipt(
    packet: CompiledPacket,
    index: int,
    compiler_commitment: str,
) -> NaturalEvidenceReceipt:
    provisional = NaturalEvidenceReceipt(
        index=index,
        packet_commitment=packet.commitment,
        source_commitment=packet.source_commitment,
        evidence_source_sha256=hashlib.sha256(f"evidence-{index}".encode()).hexdigest(),
        compiler_commitment=compiler_commitment,
        step_index=5 + index,
        target="apricot",
        distractor="beacon",
        value=str(index + 2),
        numeric_provenance=(
            ("STEP", 0, 1, "6" * 64),
            ("VALUE", 2, 3, "7" * 64),
        ),
        symbol_provenance=(
            ("TARGET", "apricot", ((4, 11),)),
            ("DISTRACTOR", "beacon", ((12, 18),)),
        ),
        commitment="",
    )
    return replace(provisional, commitment=canonical_sha256(provisional.payload()))


def main() -> None:
    device = torch.device("cpu")
    model = NaturalEvidenceCompiler(EvidenceCompilerConfig())
    rows = generate_training_records()[:2]
    tensors = tensorize_sources(rows, device)
    numeric, symbols = model(*tensors[:4])
    assert numeric.shape == (2, 2, 2)
    assert symbols.shape == (2, 2, 2)

    packet = _packet()
    compiler_commitment = "8" * 64
    receipts = tuple(
        _receipt(packet, index, compiler_commitment) for index in range(12)
    )
    typed = seal_natural_evidence(
        packet,
        receipts,
        expected_compiler_commitment=compiler_commitment,
    )
    assert len(typed) == 12
    execution = execute_natural_evidence(
        packet,
        receipts,
        expected_compiler_commitment=compiler_commitment,
    )
    assert not execution.rejected and execution.represented_worlds == 1
    assert dict(execution.groups[0].state)["apricot"] == Fraction(13)
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
        assert execute_natural_evidence(
            packet,
            mutated,
            expected_compiler_commitment=compiler_commitment,
        ).rejected
    print("diverge NVE1 runtime tests passed")


if __name__ == "__main__":
    main()
