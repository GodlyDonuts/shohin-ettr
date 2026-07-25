from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from source_deleted_multifamily_machine_board import (  # noqa: E402
    FAMILIES,
    generate_episode,
)
from multifamily_raw_machine_compiler import (  # noqa: E402
    CompilerOutput,
    MultiFamilyCompilerError,
    QUERY_ACTION,
    QUERY_START,
    QueryOutput,
    ROLE_ACTION,
    ROLE_SOURCE,
    ROLE_TARGET,
    SealedAnonymousMachine,
    SharedRawMachineCompiler,
    collate_queries,
    collate_sources,
    execute_query,
    project_byte_features_to_units,
    scan_query,
    scan_source,
    seal_machine,
)


def _source_labels(renderer: int) -> tuple[int, int, int]:
    if renderer == 0:
        return (ROLE_SOURCE, ROLE_ACTION, ROLE_TARGET)
    if renderer in {1, 2}:
        return (ROLE_ACTION, ROLE_SOURCE, ROLE_TARGET)
    if renderer == 3:
        return (ROLE_TARGET, ROLE_ACTION, ROLE_SOURCE)
    raise AssertionError


def _query_labels(renderer: int, count: int) -> tuple[int, ...]:
    if renderer in {0, 1, 2}:
        return (QUERY_START,) + (QUERY_ACTION,) * (count - 1)
    if renderer == 3:
        return (QUERY_ACTION,) * (count - 1) + (QUERY_START,)
    raise AssertionError


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("renderer", range(4))
def test_oracle_roles_seal_and_execute_without_source(
    family: str,
    renderer: int,
) -> None:
    episode = generate_episode(
        seed=9001,
        split="development",
        family=family,
        renderer=renderer,
        cell="renderer",
    )
    source = collate_sources(
        (scan_source(episode.candidate.source.encode("ascii")),)
    )
    query = collate_queries(
        (scan_query(episode.candidate.query.encode("ascii")),)
    )
    source_logits = torch.full((1, 48, 3, 3), -20.0)
    labels = _source_labels(renderer)
    record_count = int(source.record_valid[0].sum())
    for record in range(record_count):
        for occurrence, role in enumerate(labels):
            source_logits[0, record, occurrence, role] = 20.0
    machine = seal_machine(
        source,
        CompilerOutput(source_role_logits=source_logits),
        row=0,
    )

    query_logits = torch.full((1, 9, 2), -20.0)
    query_count = int(query.occurrence_valid[0].sum())
    for occurrence, role in enumerate(_query_labels(renderer, query_count)):
        query_logits[0, occurrence, role] = 20.0
    answer = execute_query(
        machine,
        query,
        QueryOutput(query_role_logits=query_logits),
        row=0,
    )
    assert answer.decode("ascii") == episode.supervisor.answer

    del episode
    assert len(machine.packet_sha256) == 64
    assert SealedAnonymousMachine.from_deployed_wire(
        machine.deployed_wire()
    ) == machine


def test_role_free_scan_masks_every_opaque_key() -> None:
    episode = generate_episode(
        seed=42,
        split="train",
        family="affine_modular",
        renderer=1,
        cell="fit",
    )
    scanned = scan_source(episode.candidate.source.encode("ascii"))
    assert len(scanned.unique_keys) == 11
    assert all(len(record.occurrence_keys) == 3 for record in scanned.records)
    assert all(
        all(key not in bytes(unit for unit in record.units if unit < 256) for key in record.occurrence_keys)
        for record in scanned.records
    )


def test_shared_compiler_parameter_ledger_and_forward() -> None:
    episodes = [
        generate_episode(
            seed=seed,
            split="train",
            family=family,
            renderer=renderer,
            cell="fit",
        )
        for seed, family, renderer in (
            (1, "affine_modular", 0),
            (2, "bitwise_rotate_xor", 1),
            (3, "permutation", 2),
        )
    ]
    source = collate_sources(
        tuple(
            scan_source(episode.candidate.source.encode("ascii"))
            for episode in episodes
        )
    )
    query = collate_queries(
        tuple(
            scan_query(episode.candidate.query.encode("ascii"))
            for episode in episodes
        )
    )
    model = SharedRawMachineCompiler(width=64, layers=1)
    source_output = model.compile_source(source)
    query_output = model.parse_query(query)
    assert source_output.source_role_logits.shape == (3, 48, 3, 3)
    assert query_output.query_role_logits.shape == (3, 9, 2)
    receipt = model.parameter_receipt()
    assert receipt.learned_compiler == model.parameter_count()
    assert receipt.complete_system < receipt.global_limit

    connected = SharedRawMachineCompiler(
        width=64,
        layers=1,
        external_width=12,
    )
    source_features = torch.randn((*source.unit_ids.shape, 12))
    query_features = torch.randn((*query.unit_ids.shape, 12))
    assert connected.compile_source(
        source,
        external_unit_features=source_features,
    ).source_role_logits.shape == (3, 48, 3, 3)
    assert connected.parse_query(
        query,
        external_unit_features=query_features,
    ).query_role_logits.shape == (3, 9, 2)


def test_frozen_byte_features_project_to_masked_units() -> None:
    episode = generate_episode(
        seed=66,
        split="train",
        family="bitwise_rotate_xor",
        renderer=2,
        cell="fit",
    )
    scanned = scan_source(episode.candidate.source.encode("ascii"))
    record = scanned.records[0]
    byte_features = torch.arange(
        len(record.payload) * 4,
        dtype=torch.float32,
    ).reshape(len(record.payload), 4)
    projected = project_byte_features_to_units(
        unit_byte_bounds=record.unit_byte_bounds,
        byte_features=byte_features,
    )
    assert projected.shape == (len(record.units), 4)
    for position in record.occurrence_positions:
        start, end = record.unit_byte_bounds[position]
        assert torch.equal(
            projected[position],
            byte_features[start:end].mean(0),
        )


def test_bad_predicted_partition_fails_closed() -> None:
    episode = generate_episode(
        seed=77,
        split="train",
        family="permutation",
        renderer=1,
        cell="fit",
    )
    source = collate_sources(
        (scan_source(episode.candidate.source.encode("ascii")),)
    )
    logits = torch.zeros((1, 48, 3, 3))
    with pytest.raises(MultiFamilyCompilerError, match="partition"):
        seal_machine(
            source,
            CompilerOutput(source_role_logits=logits),
            row=0,
        )
