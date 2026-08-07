#!/usr/bin/env python3
"""Focused identifiable-interface tests for DIVERGE-EAL2."""

from __future__ import annotations

from pathlib import Path

import torch

from diverge_eal1_runtime import execute_program
from diverge_eal2_data import build_development_episode, build_training_record
from diverge_eal2_runtime import (
    NaturalTemporalReader,
    compile_episode_laws,
    compose_complete_roles,
    hard_temporal_assignment,
    scan_register_ids,
    tensorize_temporal_sources,
)


def main() -> None:
    rows = [build_training_record(index) for index in range(2)]
    device = torch.device("cpu")
    model = NaturalTemporalReader()
    tensors = tensorize_temporal_sources(rows, device)
    logits = model(*tensors[:3])
    assert logits.shape == (2, 4, 2)
    for row in rows:
        register_ids = scan_register_ids(row["source_text"], row["registers"])
        assert sorted(register_ids) == [0, 0, 1, 1]

    forced = torch.full((4, 2), -10.0)
    registers = (1, 0, 1, 0)
    target = (0, 1, 1, 0)
    for mention, temporal in enumerate(target):
        forced[mention, temporal] = 10.0
    assert hard_temporal_assignment(forced, registers) == target

    public, assessor = build_development_episode(0)
    normal_temporal = [
        tuple(int(value) // 2 for value in item["numeric_role_ids"])
        for item in assessor["evidence"]
    ]
    counterfactual_temporal = [
        tuple(int(value) // 2 for value in item["counterfactual_role_ids"])
        for item in assessor["evidence"]
    ]
    normal_complete = compose_complete_roles(
        public["evidence"], normal_temporal, text_key="source_text"
    )
    counterfactual_complete = compose_complete_roles(
        public["evidence"],
        counterfactual_temporal,
        text_key="counterfactual_text",
    )
    assert normal_complete == [
        tuple(item["numeric_role_ids"]) for item in assessor["evidence"]
    ]
    assert counterfactual_complete == [
        tuple(item["counterfactual_role_ids"]) for item in assessor["evidence"]
    ]

    compilation = compile_episode_laws(
        public,
        normal_temporal,
        reader_state_sha256="2" * 64,
    )
    assert compilation.error is None and compilation.packet is not None
    for program, target_state in zip(
        public["transfer"], assessor["transfer"], strict=True
    ):
        assert (
            list(execute_program(compilation.packet, program))
            == target_state["terminal_state"]
        )
    one_example = compile_episode_laws(
        public,
        normal_temporal,
        reader_state_sha256="2" * 64,
        evidence_limit_per_operation=1,
    )
    assert one_example.packet is None and one_example.error == "underdetermined"

    source = Path(__file__).with_name("diverge_eal2_runtime.py").read_text()
    assert "from diverge_eal1_data" not in source
    assert "diverge_pl1_data" not in source
    assert "oracle_transition" not in source
    assert "verifier" not in source.lower()
    print("diverge EAL2 runtime tests passed")


if __name__ == "__main__":
    main()
