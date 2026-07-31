from __future__ import annotations

from dataclasses import replace
import pickle

import pytest
import torch

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_data_contract import (
    ETTR_CONTINUATION_SCHEMA,
    ETTRContinuationBatch,
    ETTRContinuationManifest,
    ETTRPacketSufficiencyIndex,
)
from ettr_episode import ETTREpisodeSegment
from ettr_objectives import ETTRObjectiveConfig
from ettr_optimization import (
    ETTROptimizerBundle,
    ETTROptimizerConfig,
)
from ettr_train_step import ETTRTrainStep, ETTRTrainStepConfig
from test_ettr_data_contract import (
    _alignment,
    _packet,
    _rectangle_episodes,
    _rectangles,
    _terminal_packet,
    _transactions,
)
from test_ettr_episode import _runner


MANIFEST_SHA256 = "a" * 64
DATASET_SHA256 = "b" * 64


def _trainer(
    *,
    accumulation: int,
    compile_backend: str | None = None,
    compile_mode: str | None = None,
    gradient_clip_mode: str = "global",
    warmup_updates: int = 1,
    train_base: bool = False,
) -> tuple[ETTRTrainStep, ETTRContinuationBatch]:
    model = _runner().model
    optimizer = ETTROptimizerBundle(
        model,
        ETTROptimizerConfig(
            train_base=train_base,
            warmup_updates=warmup_updates,
            total_updates=10,
        ),
    )
    objective = ETTRObjectiveConfig(
        vocab_size=64,
        num_slots=6,
        num_types=3,
        num_relations=3,
        num_value_codes=64,
        active_slot_budget=6,
        relation_edge_budget=96,
    )
    batch = ETTRContinuationBatch(
        manifest_sha256=MANIFEST_SHA256,
        dataset_sha256=DATASET_SHA256,
        episodes=_rectangle_episodes(),
        packet_targets=_packet(4),
        terminal_packet_targets=_terminal_packet(4),
        causal_rectangles=_rectangles(),
        transaction_targets=_transactions(4),
        initial_committed=torch.zeros(4, dtype=torch.bool),
        initial_halted=torch.zeros(4, dtype=torch.bool),
        equivariance=_alignment(),
    )
    validation_query = batch.episodes.query.tokens.clone()
    validation_query[:, 0] = 9
    validation = replace(
        batch,
        episodes=replace(
            batch.episodes,
            query=ETTREpisodeSegment.from_tokens(validation_query),
        ),
    )
    packet_sufficiency = ETTRPacketSufficiencyIndex.from_splits(
        (batch,),
        (validation,),
    )
    receipt = packet_sufficiency.receipt
    manifest = ETTRContinuationManifest(
        schema=ETTR_CONTINUATION_SCHEMA,
        protected_checkpoint_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        qualification_payload_sha256="3" * 64,
        hybrid_payload_sha256="4" * 64,
        train_rows=4,
        validation_rows=4,
        train_payload_sha256=packet_sufficiency.train_payload_sha256,
        validation_payload_sha256=(
            packet_sufficiency.validation_payload_sha256
        ),
        dataset_sha256=ETTRContinuationManifest.combined_dataset_sha256(
            packet_sufficiency.train_payload_sha256,
            packet_sufficiency.validation_payload_sha256,
        ),
        packet_sufficiency_train_batches=1,
        packet_sufficiency_validation_batches=1,
        packet_sufficiency_rows=receipt.rows,
        packet_sufficiency_unique_contexts=receipt.unique_contexts,
        packet_sufficiency_train_contexts=(
            packet_sufficiency.train_contexts
        ),
        packet_sufficiency_validation_contexts=(
            packet_sufficiency.validation_contexts
        ),
        packet_sufficiency_context_sha256=receipt.context_sha256,
        packet_sufficiency_target_bound_sha256=(
            receipt.target_bound_sha256
        ),
        source_deleted=True,
        immutable_snapshot=True,
        live_writer_input=False,
        family_label_fields=(),
    )
    batch = replace(
        batch,
        manifest_sha256=manifest.sha256(),
        dataset_sha256=manifest.dataset_sha256,
    )
    trainer = ETTRTrainStep(
        model,
        optimizer,
        objective,
        manifest=manifest,
        packet_sufficiency=packet_sufficiency,
        manifest_sha256=manifest.sha256(),
        step_config=ETTRTrainStepConfig(
            gradient_accumulation_steps=accumulation,
            gradient_clip_mode=gradient_clip_mode,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
        ),
    )
    return trainer, batch


def test_compile_configuration_rejects_mode_without_backend() -> None:
    with pytest.raises(TheoryReactorError, match="configuration"):
        ETTRTrainStepConfig(compile_mode="default").validate()
    with pytest.raises(TheoryReactorError, match="configuration"):
        ETTRTrainStepConfig(gradient_clip_mode="combined").validate()


def test_owner_gradient_clipping_reports_both_parameter_owners() -> None:
    trainer, batch = _trainer(
        accumulation=1,
        gradient_clip_mode="owner",
        train_base=True,
        warmup_updates=0,
    )
    receipt = trainer.update((batch,))
    assert receipt.base_gradient_norm > 0
    assert receipt.architecture_gradient_norm > 0
    torch.testing.assert_close(
        receipt.gradient_norm.square(),
        receipt.base_gradient_norm.float().square()
        + receipt.architecture_gradient_norm.float().square(),
    )


def test_component_gradient_clipping_reports_every_parameter_owner() -> None:
    trainer, batch = _trainer(
        accumulation=1,
        gradient_clip_mode="component",
        train_base=True,
        warmup_updates=0,
    )
    receipt = trainer.update((batch,))
    assert receipt.base_gradient_norm > 0
    assert receipt.compiler_gradient_norm > 0
    assert receipt.reactor_gradient_norm > 0
    assert receipt.query_reader_gradient_norm > 0
    torch.testing.assert_close(
        receipt.architecture_gradient_norm.square(),
        receipt.compiler_gradient_norm.float().square()
        + receipt.reactor_gradient_norm.float().square()
        + receipt.query_reader_gradient_norm.float().square(),
    )
    torch.testing.assert_close(
        receipt.gradient_norm.square(),
        receipt.base_gradient_norm.float().square()
        + receipt.architecture_gradient_norm.float().square(),
    )


@pytest.mark.filterwarnings(
    "ignore:.*should not be instantiated.*:DeprecationWarning"
)
@pytest.mark.filterwarnings(
    "ignore:The .grad attribute of a Tensor that is not a leaf Tensor.*:"
    "UserWarning"
)
def test_compiled_subject_matches_eager_update_on_cpu() -> None:
    eager, eager_batch = _trainer(
        accumulation=1,
        warmup_updates=0,
    )
    compiled, compiled_batch = _trainer(
        accumulation=1,
        compile_backend="eager",
        warmup_updates=0,
    )
    compiled.model.load_state_dict(eager.model.state_dict())
    eager_receipt = eager.update((eager_batch,))
    compiled_receipt = compiled.update((compiled_batch,))
    for name in (
        "total_loss",
        "token_lm_loss",
        "packet_loss",
        "world_intervention_loss",
        "command_intervention_loss",
        "world_query_binding_loss",
        "command_query_binding_loss",
        "transaction_loss",
        "equivariance_loss",
        "commit_halt_loss",
        "sparsity_loss",
        "anti_bypass_loss",
        "gradient_norm",
        "base_gradient_norm",
        "architecture_gradient_norm",
        "compiler_gradient_norm",
        "reactor_gradient_norm",
        "query_reader_gradient_norm",
    ):
        torch.testing.assert_close(
            getattr(compiled_receipt, name),
            getattr(eager_receipt, name),
            rtol=1e-5,
            atol=1e-6,
        )
    for name in (
        "supervised_token_count",
        "supervised_world_query_pairs",
        "supervised_command_query_pairs",
        "world_query_contrast_pairs",
        "command_query_contrast_pairs",
        "world_query_invariance_pairs",
        "command_query_invariance_pairs",
        "world_query_margin_satisfied",
        "command_query_margin_satisfied",
    ):
        assert torch.equal(
            getattr(compiled_receipt, name),
            getattr(eager_receipt, name),
        )
    for name, value in eager.model.state_dict().items():
        torch.testing.assert_close(
            compiled.model.state_dict()[name],
            value,
            rtol=1e-5,
            atol=1e-6,
        )


def test_training_subject_does_not_duplicate_model_registration() -> None:
    trainer, _ = _trainer(
        accumulation=1,
        compile_backend="eager",
        warmup_updates=0,
    )
    assert set(trainer._modules) == {"model"}
    assert len(tuple(trainer.named_parameters())) == len(
        tuple(trainer.model.named_parameters())
    )
    assert set(trainer.state_dict()) == {
        f"model.{name}" for name in trainer.model.state_dict()
    }


def test_update_runs_complete_native_objective_and_advances_cursor() -> None:
    trainer, batch = _trainer(accumulation=2)
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.compiler.named_parameters()
    }
    receipt = trainer.update((batch, batch))
    assert receipt.optimizer_step == 1
    assert receipt.learning_rate_scale == 0
    assert receipt.supervised_token_count > 0
    for name in (
        "total_loss",
        "token_lm_loss",
        "packet_loss",
        "world_intervention_loss",
        "command_intervention_loss",
        "world_query_binding_loss",
        "command_query_binding_loss",
        "transaction_loss",
        "equivariance_loss",
        "commit_halt_loss",
        "sparsity_loss",
        "anti_bypass_loss",
        "gradient_norm",
        "base_gradient_norm",
        "architecture_gradient_norm",
        "compiler_gradient_norm",
        "reactor_gradient_norm",
        "query_reader_gradient_norm",
    ):
        value = getattr(receipt, name)
        assert value.shape == ()
        assert torch.isfinite(value)
    for name in (
        "supervised_world_query_pairs",
        "supervised_command_query_pairs",
        "world_query_contrast_pairs",
        "command_query_contrast_pairs",
        "world_query_invariance_pairs",
        "command_query_invariance_pairs",
        "world_query_margin_satisfied",
        "command_query_margin_satisfied",
    ):
        value = getattr(receipt, name)
        assert value.shape == ()
        assert value.dtype == torch.int64
        assert value >= 0
    torch.testing.assert_close(
        receipt.supervised_world_query_pairs,
        torch.tensor(8),
    )
    torch.testing.assert_close(
        receipt.supervised_command_query_pairs,
        torch.tensor(8),
    )
    # Warmup update zero is intentional; the next scheduled update moves.
    second = trainer.update((batch, batch))
    assert second.optimizer_step == 2
    assert second.learning_rate_scale == 1
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in trainer.model.compiler.named_parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in trainer.model.base.parameters()
    )


def test_gradient_synchronizer_runs_after_backward_before_step() -> None:
    trainer, batch = _trainer(accumulation=1)
    calls: list[tuple[int, int]] = []

    def synchronize(parameters: object) -> None:
        values = tuple(parameters)
        calls.append(
            (
                len(values),
                sum(value.grad is not None for value in values),
            )
        )

    trainer.gradient_synchronizer = synchronize
    trainer.update((batch,))
    assert len(calls) == 1
    assert calls[0][0] > 0
    assert calls[0][1] > 0


def test_gradient_synchronizer_failure_poisoning_is_fail_stop() -> None:
    trainer, batch = _trainer(accumulation=1)

    def fail(_parameters: object) -> None:
        raise RuntimeError("collective failed")

    trainer.gradient_synchronizer = fail
    with pytest.raises(TheoryReactorError, match="optimizer update failed"):
        trainer.update((batch,))
    with pytest.raises(TheoryReactorError, match="fail-stop"):
        trainer.update((batch,))


def test_wrong_accumulation_window_fails_before_mutation() -> None:
    trainer, batch = _trainer(accumulation=2)
    before = {
        name: tensor.detach().clone()
        for name, tensor in trainer.model.state_dict().items()
    }
    with pytest.raises(TheoryReactorError, match="accumulation"):
        trainer.update((batch,))
    assert trainer.optimizer.next_update == 0
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(before[name], tensor)


def test_invalid_batch_fails_before_optimizer_mutation() -> None:
    trainer, batch = _trainer(accumulation=1)
    before_lrs = tuple(
        group["lr"]
        for optimizer in (trainer.optimizer.muon, trainer.optimizer.adam)
        if optimizer is not None
        for group in optimizer.param_groups
    )
    invalid = replace(
        batch,
        initial_committed=torch.zeros(1, dtype=torch.bool),
    )
    with pytest.raises(TheoryReactorError, match="geometry"):
        trainer.update((invalid,))
    after_lrs = tuple(
        group["lr"]
        for optimizer in (trainer.optimizer.muon, trainer.optimizer.adam)
        if optimizer is not None
        for group in optimizer.param_groups
    )
    assert after_lrs == before_lrs
    assert trainer.optimizer.next_update == 0


def test_update_rejects_cross_batch_terminal_packet_collision() -> None:
    trainer, batch = _trainer(accumulation=2)
    query = batch.episodes.query.tokens.clone()
    query[0, 2] = 24
    collision = replace(
        batch,
        episodes=replace(
            batch.episodes,
            query=ETTREpisodeSegment.from_tokens(query),
        ),
    )
    batch.validate(trainer.model.config, trainer.objective_config)
    collision.validate(trainer.model.config, trainer.objective_config)
    with pytest.raises(
        TheoryReactorError,
        match="absent from the frozen train payload index",
    ):
        trainer.update((batch, collision))
    assert trainer.optimizer.next_update == 0


def test_immutable_index_rejects_collision_after_a_prior_update() -> None:
    trainer, batch = _trainer(accumulation=1)
    trainer.update((batch,))
    query = batch.episodes.query.tokens.clone()
    query[0, 2] = 24
    collision = replace(
        batch,
        episodes=replace(
            batch.episodes,
            query=ETTREpisodeSegment.from_tokens(query),
        ),
    )
    collision.validate(trainer.model.config, trainer.objective_config)
    with pytest.raises(
        TheoryReactorError,
        match="absent from the frozen train payload index",
    ):
        trainer.update((collision,))
    assert trainer.optimizer.next_update == 1
    resumed, _ = _trainer(accumulation=1)
    with pytest.raises(
        TheoryReactorError,
        match="absent from the frozen train payload index",
    ):
        resumed.update((collision,))
    assert resumed.optimizer.next_update == 0


def test_update_rejects_validation_only_contexts() -> None:
    trainer, batch = _trainer(accumulation=1)
    validation_tokens = batch.episodes.query.tokens.clone()
    validation_tokens[:, 0] = 9
    validation = replace(
        batch,
        episodes=replace(
            batch.episodes,
            query=ETTREpisodeSegment.from_tokens(validation_tokens),
        ),
    )
    trainer.packet_sufficiency.verify_validation((validation,))
    with pytest.raises(
        TheoryReactorError,
        match="absent from the frozen train payload index",
    ):
        trainer.update((validation,))
    assert trainer.optimizer.next_update == 0


def test_train_step_rejects_optimizer_from_equal_shape_model() -> None:
    reference, _ = _trainer(accumulation=1)
    first = _runner().model
    second = _runner().model
    optimizer = ETTROptimizerBundle(
        first,
        ETTROptimizerConfig(
            train_base=False,
            warmup_updates=1,
            total_updates=10,
        ),
    )
    objective = ETTRObjectiveConfig(
        vocab_size=64,
        num_slots=6,
        num_types=3,
        num_relations=3,
        num_value_codes=64,
        active_slot_budget=6,
        relation_edge_budget=96,
    )
    with pytest.raises(TheoryReactorError, match="not bound"):
        ETTRTrainStep(
            second,
            optimizer,
            objective,
            manifest=reference.manifest,
            packet_sufficiency=reference.packet_sufficiency,
            manifest_sha256=reference.manifest_sha256,
        )


def test_train_step_pickle_rebinds_optimizer_to_reconstructed_model() -> None:
    trainer, batch = _trainer(accumulation=1)
    restored = pickle.loads(pickle.dumps(trainer))
    restored.optimizer.assert_bound_to(restored.model)
    restored.packet_sufficiency.verify_train((batch,))
    assert torch.isfinite(restored.forward_loss(batch).total)


def test_train_step_rejects_batch_from_another_snapshot() -> None:
    trainer, batch = _trainer(accumulation=1)
    wrong = replace(batch, dataset_sha256="c" * 64)
    with pytest.raises(TheoryReactorError, match="snapshot differs"):
        trainer.update((wrong,))


def test_train_step_binds_manifest_hash_and_global_sufficiency_receipt() -> None:
    trainer, _ = _trainer(accumulation=1)
    with pytest.raises(TheoryReactorError, match="manifest hash"):
        ETTRTrainStep(
            trainer.model,
            trainer.optimizer,
            trainer.objective_config,
            manifest=trainer.manifest,
            packet_sufficiency=trainer.packet_sufficiency,
            manifest_sha256="0" * 64,
        )
    wrong_manifest = replace(
        trainer.manifest,
        packet_sufficiency_target_bound_sha256="f" * 64,
    )
    with pytest.raises(TheoryReactorError, match="sufficiency receipt"):
        ETTRTrainStep(
            trainer.model,
            trainer.optimizer,
            trainer.objective_config,
            manifest=wrong_manifest,
            packet_sufficiency=trainer.packet_sufficiency,
            manifest_sha256=wrong_manifest.sha256(),
        )


def test_optimizer_failure_poison_requires_checkpoint_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, batch = _trainer(accumulation=1)
    parameter = next(
        parameter
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )

    def partial_failure() -> None:
        with torch.no_grad():
            parameter.flatten()[0].add_(1)
        raise RuntimeError("injected partial optimizer failure")

    monkeypatch.setattr(trainer.optimizer, "step", partial_failure)
    with pytest.raises(
        TheoryReactorError,
        match="restart from the last verified checkpoint",
    ):
        trainer.update((batch,))
    assert trainer.optimizer.next_update == 0
    assert all(
        parameter.grad is None
        for parameter in trainer.model.parameters()
    )
    with pytest.raises(TheoryReactorError, match="fail-stop"):
        trainer.update((batch,))
    with pytest.raises(TheoryReactorError, match="fail-stop"):
        trainer.forward_loss(batch)
    with pytest.raises(TheoryReactorError, match="fail-stop"):
        ETTRTrainStep(
            trainer.model,
            trainer.optimizer,
            trainer.objective_config,
            manifest=trainer.manifest,
            packet_sufficiency=trainer.packet_sufficiency,
            manifest_sha256=trainer.manifest_sha256,
        )
