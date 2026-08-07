#!/usr/bin/env python3
"""Focused learned-reader, sealing, and execution tests for DIVERGE-EAL1."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from diverge_eal1_data import build_development_episode, build_training_record
from diverge_eal1_runtime import (
    NaturalTransitionReader,
    TransitionReaderConfig,
    compile_episode_laws,
    execute_program,
    hard_role_permutation,
    rebind_packet,
    tensorize_sources,
)


def main() -> None:
    device = torch.device("cpu")
    model = NaturalTransitionReader(TransitionReaderConfig())
    rows = [build_training_record(index) for index in range(2)]
    tensors = tensorize_sources(rows, device)
    logits = model(*tensors[:3])
    assert logits.shape == (2, 4, 4)
    forced = torch.full((4, 4), -10.0)
    for mention, role in enumerate((2, 0, 3, 1)):
        forced[mention, role] = 10.0
    assert hard_role_permutation(forced) == (2, 0, 3, 1)

    public, assessor = build_development_episode(0)
    roles = [item["numeric_role_ids"] for item in assessor["evidence"]]
    compilation = compile_episode_laws(
        public,
        roles,
        reader_state_sha256="1" * 64,
    )
    assert compilation.error is None and compilation.packet is not None
    assert all(size == 1 for pair in compilation.support_sizes for size in pair)
    for program, hidden in zip(public["transfer"], assessor["transfer"], strict=True):
        assert (
            list(execute_program(compilation.packet, program))
            == hidden["terminal_state"]
        )
    packet_record = json.dumps(compilation.packet.record(), sort_keys=True)
    for forbidden in ("source_text", "before", "after", "terminal_state"):
        assert forbidden not in packet_record

    one_example = compile_episode_laws(
        public,
        roles,
        reader_state_sha256="1" * 64,
        evidence_limit_per_operation=1,
    )
    assert one_example.packet is None and one_example.error == "underdetermined"
    assert all(size == 5 for pair in one_example.support_sizes for size in pair)

    other_public, _ = build_development_episode(1)
    rebound = rebind_packet(compilation.packet, other_public["aliases"])
    assert rebound.aliases == tuple(other_public["aliases"])
    assert rebound.rows == compilation.packet.rows
    assert rebound.commitment != compilation.packet.commitment

    runtime_source = Path(__file__).with_name("diverge_eal1_runtime.py").read_text()
    assert "from diverge_eal1_data" not in runtime_source
    assert "diverge_pl1_data" not in runtime_source
    assert "oracle_transition" not in runtime_source
    print("diverge EAL1 runtime tests passed")


if __name__ == "__main__":
    main()
