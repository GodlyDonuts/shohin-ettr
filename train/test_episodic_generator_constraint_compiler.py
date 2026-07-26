from __future__ import annotations

import pytest
import torch

from episodic_generator_constraint_compiler import (
    EpisodicGeneratorConstraintCompiler,
    execute_episodic_generator_query,
    scan_episodic_generator_source,
    seal_episodic_generator_packet,
)
from run_episodic_generator_constraint import _delete_target_witness
from sparse_latent_law_compiler import (
    MAX_ACTIONS,
    MAX_CARDINALITY,
    SparseLawCompilerError,
    collate_sparse_sources,
    scan_sparse_query,
    scan_sparse_source,
)
from source_deleted_episodic_generator_law_board import (
    HELD_OUT_FAMILY,
    generate_episode,
)


def test_closure_receipt_and_forward_geometry() -> None:
    actions = (
        "h00000000000000000001",
        "h00000000000000000002",
        "h00000000000000000003",
        "h00000000000000000004",
    )
    source = "\n".join(
        [
            "domain-size=8",
            *[
                f"origin={state}; operation={actions[0]}; "
                f"destination={(state + 1) % 8}"
                for state in range(8)
            ],
            *[
                f"origin={state}; operation={actions[1]}; "
                f"destination={(-state) % 8}"
                for state in range(8)
            ],
            f"origin=0; operation={actions[2]}; destination=2",
            f"origin=1; operation={actions[2]}; destination=3",
            f"origin=0; operation={actions[3]}; destination=7",
            f"origin=2; operation={actions[3]}; destination=5",
        ]
    )
    batch = collate_sparse_sources(
        (scan_sparse_source(source.encode("ascii")),)
    )
    model = EpisodicGeneratorConstraintCompiler(
        width=32,
        layers=1,
        heads=4,
        maximum_depth=4,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.direction_head[-1].bias.fill_(20.0)
    output = model(batch)
    assert output.direction_logits.shape[0] == 1
    assert output.transition_logits.shape == (
        1,
        MAX_ACTIONS,
        MAX_CARDINALITY,
        MAX_CARDINALITY,
    )
    assert model.closure_receipt.syntactic_programs == 31
    assert model.parameter_receipt().complete_system < 200_000_000


def test_episode_local_closure_solves_held_out_generator_family() -> None:
    row = generate_episode(
        seed=1234,
        split="development",
        family=HELD_OUT_FAMILY,
        renderer=0,
        cell="law",
        cardinality=8,
    )
    batch = collate_sparse_sources(
        (
            scan_episodic_generator_source(
                row.candidate.source.encode("ascii")
            ),
        )
    )
    model = EpisodicGeneratorConstraintCompiler(
        width=32,
        layers=1,
        heads=4,
        maximum_depth=6,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.direction_head[-1].bias.fill_(20.0)
    output = model(batch)
    packet = seal_episodic_generator_packet(
        batch,
        output,
        row=0,
    )
    query = scan_sparse_query(
        row.candidate.query.encode("ascii")
    )
    assert (
        execute_episodic_generator_query(packet, query)
        == row.supervisor.answer
    )
    source_bytes = row.candidate.source.encode("ascii")
    assert source_bytes not in packet.deployed_wire()
    assert (
        packet.from_deployed_wire(packet.deployed_wire())
        == packet
    )


def test_missing_target_witness_fails_closed() -> None:
    row = generate_episode(
        seed=4321,
        split="development",
        family=HELD_OUT_FAMILY,
        renderer=0,
        cell="law",
        cardinality=8,
    )
    batch = collate_sparse_sources(
        (
            scan_episodic_generator_source(
                row.candidate.source.encode("ascii")
            ),
        )
    )
    incomplete = _delete_target_witness(batch)
    model = EpisodicGeneratorConstraintCompiler(
        width=32,
        layers=1,
        heads=4,
        maximum_depth=6,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.direction_head[-1].bias.fill_(20.0)
        output = model(incomplete)
    with pytest.raises(
        SparseLawCompilerError,
        match="target map is not a permutation",
    ):
        seal_episodic_generator_packet(incomplete, output, row=0)
