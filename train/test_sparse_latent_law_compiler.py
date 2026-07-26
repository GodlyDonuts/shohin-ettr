from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from source_deleted_sparse_latent_law_board import (  # noqa: E402
    FAMILIES,
    compile_source,
    generate_episode,
)
from sparse_latent_law_compiler import (  # noqa: E402
    MAX_ACTIONS,
    MAX_CARDINALITY,
    MAX_RECORDS,
    FactorizedSparseLatentLawCompiler,
    MicrocodedSparseLatentLawCompiler,
    SealedLearnedSparseMachine,
    SparseCompilerOutput,
    SparseLatentLawCompiler,
    SparseLawCompilerError,
    collate_sparse_sources,
    execute_sparse_query,
    scan_sparse_query,
    scan_sparse_source,
    seal_sparse_machine,
)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("renderer", range(6))
def test_candidate_scanner_and_oracle_logits_execute(
    family: str,
    renderer: int,
) -> None:
    row = generate_episode(
        seed=901,
        split="development",
        family=family,
        renderer=renderer,
        cell="joint",
        cardinality=16,
        action_count=4,
    )
    scanned = scan_sparse_source(row.candidate.source.encode("ascii"))
    expected_answer = row.supervisor.answer
    query_payload = row.candidate.query.encode("ascii")
    batch = collate_sparse_sources((scanned,))
    exact = compile_source(row.candidate.source)
    logits = torch.full(
        (1, MAX_ACTIONS, MAX_CARDINALITY, MAX_CARDINALITY),
        -20.0,
    )
    for action, transition in enumerate(exact.transition):
        for source, target in enumerate(transition):
            logits[0, action, source, target] = 20.0
    direction = torch.zeros((1, MAX_RECORDS))
    direction[:, : len(scanned.records)] = (
        20.0 if renderer in {0, 1, 2, 3} else -20.0
    )
    machine = seal_sparse_machine(
        batch,
        SparseCompilerOutput(
            direction_logits=direction,
            transition_logits=logits,
        ),
        row=0,
    )
    del row
    query = scan_sparse_query(query_payload)
    assert execute_sparse_query(machine, query) == expected_answer
    assert SealedLearnedSparseMachine.from_deployed_wire(
        machine.deployed_wire()
    ) == machine


def test_model_forward_controls_and_parameter_receipt() -> None:
    rows = [
        generate_episode(
            seed=seed,
            split="train",
            family=family,
            renderer=renderer,
            cell="fit",
            cardinality=cardinality,
            action_count=action_count,
        )
        for seed, family, renderer, cardinality, action_count in (
            (1, "affine_modular", 0, 8, 2),
            (2, "bitwise_rotate_xor", 3, 16, 3),
            (3, "gray_conjugate_affine", 4, 16, 4),
        )
    ]
    batch = collate_sparse_sources(
        tuple(
            scan_sparse_source(row.candidate.source.encode("ascii"))
            for row in rows
        )
    )
    model = SparseLatentLawCompiler(width=64, layers=1, heads=4)
    output = model(batch)
    direction = model(batch, direction_sign=-1.0)
    shifted = model(batch, observation_target_shift=1)
    zeroed = model(batch, observations_zeroed=True)
    assert output.direction_logits.shape == (3, MAX_RECORDS)
    assert output.transition_logits.shape == (
        3,
        MAX_ACTIONS,
        MAX_CARDINALITY,
        MAX_CARDINALITY,
    )
    assert bool(torch.isfinite(output.transition_logits).all())
    assert torch.equal(
        direction.direction_logits,
        -output.direction_logits,
    )
    assert not torch.equal(
        shifted.transition_logits,
        output.transition_logits,
    )
    assert not torch.equal(
        zeroed.transition_logits,
        output.transition_logits,
    )
    receipt = model.parameter_receipt()
    assert receipt.learned_compiler == model.parameter_count()
    assert receipt.complete_system < receipt.global_limit

    factorized = FactorizedSparseLatentLawCompiler(
        width=64,
        layers=1,
        heads=4,
        generators=12,
        composition_depth=3,
    )
    factorized_output = factorized(batch)
    assert factorized_output.transition_logits.shape == (
        3,
        MAX_ACTIONS,
        MAX_CARDINALITY,
        MAX_CARDINALITY,
    )
    assert bool(torch.isfinite(factorized_output.transition_logits).all())
    factorized_receipt = factorized.parameter_receipt()
    assert (
        factorized_receipt.learned_compiler
        == factorized.parameter_count()
    )

    microcoded = MicrocodedSparseLatentLawCompiler(
        width=64,
        layers=1,
        heads=4,
    )
    microcoded_output = microcoded(batch)
    assert microcoded_output.transition_logits.shape == (
        3,
        MAX_ACTIONS,
        MAX_CARDINALITY,
        MAX_CARDINALITY,
    )
    assert microcoded_output.microcode is not None
    assert microcoded_output.microcode.family_logits.shape == (
        3,
        MAX_ACTIONS,
        3,
    )
    assert microcoded_output.microcode.multiplier_logits.shape == (
        3,
        MAX_ACTIONS,
        MAX_CARDINALITY,
    )
    assert microcoded_output.microcode.offset_logits.shape == (
        3,
        MAX_ACTIONS,
        MAX_CARDINALITY,
    )
    assert microcoded_output.microcode.shift_logits.shape == (
        3,
        MAX_ACTIONS,
        4,
    )
    assert microcoded_output.microcode.mask_logits.shape == (
        3,
        MAX_ACTIONS,
        MAX_CARDINALITY,
    )
    assert bool(torch.isfinite(microcoded_output.transition_logits).all())
    assert (
        microcoded.parameter_receipt().learned_compiler
        == microcoded.parameter_count()
    )


def test_sparse_query_scanner_rejects_ambiguous_numbers() -> None:
    with pytest.raises(SparseLawCompilerError):
        scan_sparse_query(b"origin=1; program=h00000000000000000000; extra=2")
