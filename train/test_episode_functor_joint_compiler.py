from __future__ import annotations

import copy
from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from episode_functor_joint_compiler import (
    JointCompilerError,
    JointProofCarryingCompiler,
    SealedJointMachine,
)
from episode_functor_witness_compiler import (
    WitnessCompilerError,
    canonicalize_witness_batch,
    collate_witness_sources,
    scan_witness_source,
)
from pipeline.episode_functor_identifiable_board import (
    GrammarFactors,
    encode_source,
    generate_machine,
    hide_one_cell_per_relation,
)


PROTECTED_PARAMETERS = 125_081_664
DETACHED_PARSER_PARAMETERS = 748_033


def _source() -> bytes:
    machine = generate_machine(
        seed="joint-compiler-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="joint-compiler-test-v1",
        split="mechanics",
        index=0,
    )
    return encode_source(evidence, GrammarFactors(0, 0, 0))


def _batch(source: bytes | None = None):
    payload = _source() if source is None else source
    return collate_witness_sources((scan_witness_source(payload),))


def _rename_all_keys(source: bytes) -> bytes:
    scanned = scan_witness_source(source)
    replacements = tuple(
        f"h{0xf000000000000000 + index:016x}".encode("ascii")
        for index in range(len(scanned.pointer.unique_keys))
    )
    output = bytearray()
    cursor = 0
    for (start, end), unique in zip(
        scanned.pointer.spans,
        scanned.pointer.occurrence_to_unique,
        strict=True,
    ):
        output.extend(source[cursor:start])
        replacement = replacements[unique]
        assert len(replacement) == end - start
        output.extend(replacement)
        cursor = end
    output.extend(source[cursor:])
    return bytes(output)


def _rename_all_keys_variable_width(source: bytes) -> bytes:
    scanned = scan_witness_source(source)
    replacements = tuple(
        f"d{index + 1}{'0' * index}".encode("ascii")
        for index in range(len(scanned.pointer.unique_keys))
    )
    output = bytearray()
    cursor = 0
    for (start, end), unique in zip(
        scanned.pointer.spans,
        scanned.pointer.occurrence_to_unique,
        strict=True,
    ):
        output.extend(source[cursor:start])
        output.extend(replacements[unique])
        cursor = end
    output.extend(source[cursor:])
    return bytes(output)


def _retie_one_occurrence(source: bytes) -> bytes:
    scanned = scan_witness_source(source)
    target_occurrences = [
        index
        for index, unique in enumerate(
            scanned.pointer.occurrence_to_unique
        )
        if unique == 0
    ]
    assert len(target_occurrences) >= 2
    replacement_occurrence = next(
        index
        for index, unique in enumerate(
            scanned.pointer.occurrence_to_unique
        )
        if unique == 1
    )
    target = target_occurrences[-1]
    target_start, target_end = scanned.pointer.spans[target]
    source_start, source_end = scanned.pointer.spans[
        replacement_occurrence
    ]
    replacement = source[source_start:source_end]
    assert len(replacement) == target_end - target_start
    return (
        source[:target_start]
        + replacement
        + source[target_end:]
    )


def _small() -> JointProofCarryingCompiler:
    return JointProofCarryingCompiler(
        external_feature_width=0,
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        assignment_width=48,
        assignment_context_width=96,
        machine_width=64,
        machine_context_width=128,
        cycles=2,
        sinkhorn_iterations=16,
    )


def test_joint_compiler_forward_backward_and_seal() -> None:
    torch.manual_seed(20260724)
    compiler = _small()
    output = compiler(_batch())
    loss = (
        output.witness.projection.transition_transport.square().sum()
        + output.witness.projection.observer_transport.square().sum()
        + output.equilibrium.key_assignment_logits.exp().square().sum()
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in compiler.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    sealed = compiler.seal(output)
    assert sealed.machine.batch_size == 1
    assert sealed.keys.batch_size == 1
    assert not hasattr(sealed, "compilation_receipt")
    assert not hasattr(sealed, "source_sha256")
    assert compiler.verify_sealed(sealed) == (
        sealed.deployed_wire(0),
    )


def test_joint_compiler_rejects_cross_instance_output() -> None:
    torch.manual_seed(20260724)
    compiler_a = _small()
    compiler_b = _small()
    output = compiler_a(_batch())
    with pytest.raises(
        JointCompilerError,
        match="provenance",
    ):
        compiler_b.seal(output)


def test_joint_compiler_rejects_deep_copied_instance() -> None:
    torch.manual_seed(20260724)
    compiler = _small()
    output = compiler(_batch())
    copied = copy.deepcopy(compiler)
    with pytest.raises(
        JointCompilerError,
        match="provenance",
    ):
        copied.seal(output)


def test_joint_compiler_rejects_mutated_output_and_state() -> None:
    torch.manual_seed(20260724)
    compiler = _small()
    output = compiler(_batch())
    revised_logits = output.equilibrium.key_assignment_logits.clone()
    revised_logits[0, 0, 0] += 1.0
    mutated = replace(
        output,
        equilibrium=replace(
            output.equilibrium,
            key_assignment_logits=revised_logits,
        ),
    )
    with pytest.raises(
        JointCompilerError,
        match="provenance",
    ):
        compiler.seal(mutated)
    output = compiler(_batch())
    with torch.no_grad():
        next(compiler.parameters()).add_(1.0)
    with pytest.raises(
        JointCompilerError,
        match="provenance",
    ):
        compiler.seal(output)


def test_joint_compiler_rejects_mutated_execution_configuration() -> None:
    torch.manual_seed(20260724)
    compiler = _small()
    output = compiler(_batch())
    compiler.equilibrium.cycles += 1
    with pytest.raises(
        JointCompilerError,
        match="provenance",
    ):
        compiler.seal(output)


def test_joint_compiler_rejects_forged_raw_source_custody() -> None:
    batch = _batch()
    forged_hash = replace(batch, source_sha256=("0" * 64,))
    with pytest.raises(
        WitnessCompilerError,
        match="source hash differs",
    ):
        _small()(forged_hash)

    key_bytes = batch.pointer.unique_key_bytes.clone()
    key_bytes[0, 0, 0] ^= 1
    forged_keys = replace(
        batch,
        pointer=replace(
            batch.pointer,
            unique_key_bytes=key_bytes,
        ),
    )
    with pytest.raises(
        WitnessCompilerError,
        match="key custody differs",
    ):
        _small()(forged_keys)


def test_joint_compiler_is_exactly_invariant_to_literal_key_renaming() -> None:
    source = _source()
    renamed_source = _rename_all_keys(source)
    original_batch = _batch(source)
    renamed_batch = _batch(renamed_source)
    original_model_view = canonicalize_witness_batch(original_batch)
    renamed_model_view = canonicalize_witness_batch(renamed_batch)
    assert original_batch.source_sha256 != renamed_batch.source_sha256
    assert torch.equal(
        original_model_view.pointer.byte_ids,
        renamed_model_view.pointer.byte_ids,
    )
    assert torch.equal(
        original_model_view.record_bounds,
        renamed_model_view.record_bounds,
    )
    assert torch.equal(
        original_model_view.pointer.occurrence_to_unique,
        renamed_model_view.pointer.occurrence_to_unique,
    )
    assert not torch.equal(
        original_model_view.pointer.unique_key_bytes,
        renamed_model_view.pointer.unique_key_bytes,
    )

    torch.manual_seed(20260724)
    compiler = _small()
    original = compiler(original_batch)
    renamed = compiler(renamed_batch)
    for left, right in (
        (
            original.initial_witness.raw_key_assignment_logits,
            renamed.initial_witness.raw_key_assignment_logits,
        ),
        (
            original.initial_witness.record_type_logits,
            renamed.initial_witness.record_type_logits,
        ),
        (
            original.initial_witness.occurrence_role_logits,
            renamed.initial_witness.occurrence_role_logits,
        ),
        (
            original.initial_witness.answer_logits,
            renamed.initial_witness.answer_logits,
        ),
        (
            original.equilibrium.key_assignment_logits,
            renamed.equilibrium.key_assignment_logits,
        ),
        (
            original.equilibrium.transition_probabilities,
            renamed.equilibrium.transition_probabilities,
        ),
        (
            original.equilibrium.observer_probabilities,
            renamed.equilibrium.observer_probabilities,
        ),
    ):
        assert torch.equal(left, right)
    for left, right in zip(
        original.equilibrium.cycle_transition_probabilities,
        renamed.equilibrium.cycle_transition_probabilities,
        strict=True,
    ):
        assert torch.equal(left, right)
    for left, right in zip(
        original.equilibrium.cycle_observer_probabilities,
        renamed.equilibrium.cycle_observer_probabilities,
        strict=True,
    ):
        assert torch.equal(left, right)
    parameters = tuple(compiler.parameters())
    original_loss = (
        original.equilibrium.transition_probabilities.square().sum()
        + original.equilibrium.observer_probabilities.square().sum()
        + original.equilibrium.key_assignment_logits.exp().square().sum()
    )
    renamed_loss = (
        renamed.equilibrium.transition_probabilities.square().sum()
        + renamed.equilibrium.observer_probabilities.square().sum()
        + renamed.equilibrium.key_assignment_logits.exp().square().sum()
    )
    original_gradients = torch.autograd.grad(
        original_loss,
        parameters,
    )
    renamed_gradients = torch.autograd.grad(
        renamed_loss,
        parameters,
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            original_gradients,
            renamed_gradients,
            strict=True,
        )
    )
    original_seal = compiler.seal(original)
    renamed_seal = compiler.seal(renamed)
    assert torch.equal(
        original_seal.machine.action_next,
        renamed_seal.machine.action_next,
    )
    assert torch.equal(
        original_seal.machine.observer_answer,
        renamed_seal.machine.observer_answer,
    )
    assert not torch.equal(
        original_seal.keys.state_keys,
        renamed_seal.keys.state_keys,
    )


def test_joint_compiler_is_invariant_to_variable_width_decimal_keys() -> None:
    source = _source()
    renamed_source = _rename_all_keys_variable_width(source)
    original_batch = _batch(source)
    renamed_batch = _batch(renamed_source)
    original_view = canonicalize_witness_batch(original_batch)
    renamed_view = canonicalize_witness_batch(renamed_batch)
    assert len(source) != len(renamed_source)
    assert torch.equal(
        original_view.pointer.byte_ids,
        renamed_view.pointer.byte_ids,
    )
    assert torch.equal(
        original_view.record_bounds,
        renamed_view.record_bounds,
    )
    assert torch.equal(
        original_view.pointer.occurrence_to_unique,
        renamed_view.pointer.occurrence_to_unique,
    )

    torch.manual_seed(20260724)
    compiler = _small()
    original = compiler(original_batch)
    renamed = compiler(renamed_batch)
    for left, right in (
        (
            original.equilibrium.key_assignment_logits,
            renamed.equilibrium.key_assignment_logits,
        ),
        (
            original.equilibrium.transition_probabilities,
            renamed.equilibrium.transition_probabilities,
        ),
        (
            original.equilibrium.observer_probabilities,
            renamed.equilibrium.observer_probabilities,
        ),
    ):
        assert torch.equal(left, right)
    original_seal = compiler.seal(original)
    renamed_seal = compiler.seal(renamed)
    assert torch.equal(
        original_seal.machine.action_next,
        renamed_seal.machine.action_next,
    )
    assert torch.equal(
        original_seal.machine.observer_answer,
        renamed_seal.machine.observer_answer,
    )
    assert not torch.equal(
        original_seal.keys.state_keys,
        renamed_seal.keys.state_keys,
    )


def test_joint_compiler_uses_equality_partition_not_literal_spelling() -> None:
    source = _source()
    retied_source = _retie_one_occurrence(source)
    original_batch = _batch(source)
    retied_batch = _batch(retied_source)
    original_view = canonicalize_witness_batch(original_batch)
    retied_view = canonicalize_witness_batch(retied_batch)
    assert torch.equal(
        original_view.pointer.byte_ids,
        retied_view.pointer.byte_ids,
    )
    assert not torch.equal(
        original_view.pointer.occurrence_to_unique,
        retied_view.pointer.occurrence_to_unique,
    )
    torch.manual_seed(20260724)
    compiler = _small()
    original = compiler(original_batch)
    retied = compiler(retied_batch)
    assert not torch.equal(
        original.initial_witness.raw_key_assignment_logits,
        retied.initial_witness.raw_key_assignment_logits,
    )


def test_joint_compiler_rejects_mutated_seal_receipt() -> None:
    torch.manual_seed(20260724)
    compiler = _small()
    sealed = compiler.seal(compiler(_batch()))
    forged = SealedJointMachine(
        machine=sealed.machine,
        keys=sealed.keys,
        wire_sha256=sealed.wire_sha256,
        seal_receipt_sha256="0" * 64,
        seal_capability=sealed.seal_capability,
    )
    with pytest.raises(
        JointCompilerError,
        match="receipt",
    ):
        compiler.verify_sealed(forged)


def test_joint_compiler_rejects_rehashed_forged_machine() -> None:
    torch.manual_seed(20260724)
    compiler = _small()
    sealed = compiler.seal(compiler(_batch()))
    action_next = sealed.machine.action_next.clone()
    action_next[0, 0, 0] = (
        int(action_next[0, 0, 0]) + 1
    ) % 8
    forged_machine = replace(
        sealed.machine,
        action_next=action_next,
    )
    forged_wire = forged_machine.deployed_wire(sealed.keys, 0)
    forged = SealedJointMachine(
        machine=forged_machine,
        keys=sealed.keys,
        wire_sha256=(sha256(forged_wire).hexdigest(),),
        seal_receipt_sha256=sha256(forged_wire + b"seal").hexdigest(),
        seal_capability=sealed.seal_capability,
    )
    with pytest.raises(
        JointCompilerError,
        match="receipt",
    ):
        compiler.verify_sealed(forged)


def test_joint_compiler_rejects_straight_through_training() -> None:
    with pytest.raises(
        JointCompilerError,
        match="solver-backed",
    ):
        _small()(_batch(), straight_through=True)


def test_default_joint_compiler_parameter_receipt() -> None:
    compiler = JointProofCarryingCompiler()
    assert compiler.witness.parameter_count() == 44_658_064
    assert compiler.equilibrium.parameter_count() == 19_013_524
    assert compiler.parameter_count() == 63_671_588
    complete = (
        PROTECTED_PARAMETERS
        + compiler.parameter_count()
        + DETACHED_PARSER_PARAMETERS
    )
    assert complete == 189_501_285
    assert 200_000_000 - complete == 10_498_715
