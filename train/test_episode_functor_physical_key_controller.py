from __future__ import annotations

from dataclasses import replace

import torch

from episode_functor_conflict_compiler import DirectEvidenceProjector
from episode_functor_physical_key_controller import (
    PhysicalKeyPathController,
)
from episode_functor_witness_compiler import (
    ProofCarryingWitnessCompiler,
    collate_witness_sources,
    scan_witness_source,
)
from pipeline.episode_functor_identifiable_board import (
    GrammarFactors,
    encode_source,
    generate_machine,
    hide_one_cell_per_relation,
)


def _fixture():
    machine = generate_machine(
        seed="physical-key-controller-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="physical-key-controller-test-v1",
        split="mechanics",
        index=0,
    )
    source = encode_source(evidence, GrammarFactors(0, 0, 0))
    batch = collate_witness_sources((scan_witness_source(source),))
    torch.manual_seed(20260724)
    witness = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
        projector=DirectEvidenceProjector(),
    )(batch)
    torch.manual_seed(20260725)
    controller = PhysicalKeyPathController()
    return batch, witness, controller


def test_controller_exact_parameter_receipt() -> None:
    controller = PhysicalKeyPathController()
    assert controller.parameter_count() == 3_978_602


def test_controller_is_physical_key_permutation_equivariant() -> None:
    batch, witness, controller = _fixture()
    baseline = controller(witness, batch.record_valid)
    generator = torch.Generator().manual_seed(7)
    key_order = torch.randperm(
        witness.unique_key_valid.shape[1],
        generator=generator,
    )
    recoded_evidence = replace(
        witness.relation_evidence,
        record_role_unique=(
            witness.relation_evidence.record_role_unique[
                ...,
                key_order,
            ]
        ),
    )
    recoded = replace(
        witness,
        relation_evidence=recoded_evidence,
        key_assignment_logits=witness.key_assignment_logits[
            ...,
            key_order,
        ],
        raw_key_assignment_logits=witness.raw_key_assignment_logits[
            ...,
            key_order,
        ],
        unique_key_bytes=witness.unique_key_bytes[:, key_order],
        unique_key_valid=witness.unique_key_valid[:, key_order],
    )
    treatment = controller(recoded, batch.record_valid)
    assert torch.allclose(
        baseline.correction[..., key_order],
        treatment.correction,
        atol=2e-5,
        rtol=0.0,
    )


def test_controller_is_slot_recoding_equivariant() -> None:
    batch, witness, controller = _fixture()
    baseline = controller(witness, batch.record_valid)
    generator = torch.Generator().manual_seed(19)
    slot_order = torch.arange(
        witness.key_assignment_logits.shape[1]
    )
    slot_order[:8] = torch.randperm(8, generator=generator)
    slot_order[8:11] = 8 + torch.randperm(3, generator=generator)
    slot_order[12:14] = 12 + torch.randperm(2, generator=generator)
    recoded = replace(
        witness,
        key_assignment_logits=witness.key_assignment_logits[
            :,
            slot_order,
        ],
        raw_key_assignment_logits=witness.raw_key_assignment_logits[
            :,
            slot_order,
        ],
    )
    treatment = controller(recoded, batch.record_valid)
    assert torch.allclose(
        baseline.correction[:, slot_order],
        treatment.correction,
        atol=2e-5,
        rtol=0.0,
    )


def test_controller_is_record_invariant_and_answer_invariant() -> None:
    batch, witness, controller = _fixture()
    baseline = controller(witness, batch.record_valid)
    record_order = torch.arange(
        witness.record_type_logits.shape[1]
    ).roll(11)
    reordered_evidence = replace(
        witness.relation_evidence,
        record_role_unique=(
            witness.relation_evidence.record_role_unique[
                :,
                record_order,
            ]
        ),
    )
    reordered = replace(
        witness,
        relation_evidence=reordered_evidence,
        record_type_logits=witness.record_type_logits[:, record_order],
        answer_logits=witness.answer_logits[:, record_order],
    )
    record_result = controller(
        reordered,
        batch.record_valid[:, record_order],
    )
    assert torch.allclose(
        baseline.correction,
        record_result.correction,
        atol=2e-5,
        rtol=0.0,
    )
    answer_order = torch.tensor((2, 0, 3, 1))
    answer_recoded = replace(
        witness,
        answer_logits=witness.answer_logits[..., answer_order],
    )
    answer_result = controller(answer_recoded, batch.record_valid)
    assert torch.allclose(
        baseline.correction,
        answer_result.correction,
        atol=2e-5,
        rtol=0.0,
    )


def test_controller_controls_share_weights_and_change_only_path_signal() -> None:
    batch, witness, controller = _fixture()
    causal = controller(witness, batch.record_valid, mode="causal")
    broken = controller(
        witness,
        batch.record_valid,
        mode="broken-glue",
    )
    one_step = controller(
        witness,
        batch.record_valid,
        mode="one-step-only",
    )
    assert controller.parameter_count() == 3_978_602
    assert torch.equal(
        causal.nerve.transition_relation,
        broken.nerve.transition_relation,
    )
    assert torch.equal(
        causal.nerve.observation_relation,
        one_step.nerve.observation_relation,
    )
    assert not torch.equal(causal.correction, broken.correction)
    assert not torch.equal(causal.correction, one_step.correction)
    assert bool(causal.correction.abs().amax().le(2.0))
    assert bool(broken.correction.abs().amax().le(2.0))
    assert bool(one_step.correction.abs().amax().le(2.0))


def test_zero_path_mass_has_bit_exact_zero_correction() -> None:
    batch, witness, controller = _fixture()
    result = controller(
        witness,
        torch.zeros_like(batch.record_valid),
    )
    assert torch.equal(
        result.correction,
        torch.zeros_like(result.correction),
    )


def test_open_loop_is_graph_connected_zero_for_every_parameter() -> None:
    batch, witness, controller = _fixture()
    result = controller(
        witness,
        batch.record_valid,
        mode="open-loop",
    )
    assert torch.equal(
        result.correction,
        torch.zeros_like(result.correction),
    )
    result.correction.sum().backward()
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(parameter.grad.eq(0).all())
        for parameter in controller.parameters()
    )


def test_reduced_controller_is_finite_under_bf16_autocast() -> None:
    batch, witness, _ = _fixture()
    controller = PhysicalKeyPathController(
        cell_width=64,
        context_width=128,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = controller(witness, batch.record_valid)
        loss = result.correction.square().mean()
    loss.backward()
    assert bool(torch.isfinite(result.correction).all())
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in controller.parameters()
    )
