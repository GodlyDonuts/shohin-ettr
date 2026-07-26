from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from ettr_checkpoint import (
    BaseProvenance,
    DataStreamState,
    ETTRCheckpointError,
    EpisodeLifecycleState,
    TrainingProgress,
    _integrity_payload,
    _load_ettr_checkpoint_for_test,
    _save_ettr_checkpoint_for_test,
    file_sha256,
    load_ettr_checkpoint,
    load_protected_base_provenance,
    runtime_source_manifest,
    save_ettr_checkpoint,
    tree_sha256,
)
from ettr_optimization import (
    ETTROptimizerBundle,
    ETTROptimizerConfig,
)
from model import GPT, GPTConfig


def _small_model(seed: int) -> EndogenousTypedTheoryReactorGPT:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = GPT(
            GPTConfig(
                vocab_size=64,
                n_layer=4,
                n_head=4,
                n_kv_head=2,
                d_model=24,
                d_ff=48,
                seq_len=32,
                zloss=0.0,
            )
        )
        config = TheoryReactorConfig(
            d_model=24,
            state_width=16,
            num_slots=4,
            num_types=4,
            num_relations=3,
            num_value_codes=8,
            max_edges=12,
            num_heads=4,
            compiler_layers=1,
            reactor_layers=1,
            query_layers=1,
            ff_multiplier=2,
            max_steps=4,
            stage_after_block=1,
        )
        return EndogenousTypedTheoryReactorGPT(base, config)


def _optimizer_and_scheduler(
    model: EndogenousTypedTheoryReactorGPT,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.base.parameters()),
                "lr": 1e-4,
                "weight_decay": 0.01,
            },
            {
                "params": [
                    parameter
                    for module in (
                        model.compiler,
                        model.reactor,
                        model.query_reader,
                    )
                    for parameter in module.parameters()
                ],
                "lr": 3e-4,
                "weight_decay": 0.0,
            },
        ],
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5,
    )
    return optimizer, scheduler


def _advance_optimizer(
    model: EndogenousTypedTheoryReactorGPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
) -> None:
    loss = sum(parameter.float().square().mean() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


def _base_provenance(path: Path) -> BaseProvenance:
    config = {"synthetic": True, "d_model": 24}
    payload = b"immutable synthetic protected checkpoint"
    path.write_bytes(payload)
    return BaseProvenance(
        checkpoint_path=str(path.resolve()),
        checkpoint_bytes=len(payload),
        checkpoint_sha256=file_sha256(path),
        step=300_000,
        data_seed=777,
        data_stream_generation=1,
        data_stream_seed=1_000_780,
        base_config=config,
        config_sha256=_json_digest(config),
        base_state_sha256="1" * 64,
        state_key_sha256="2" * 64,
        state_key_count=10,
        base_parameters=10_000,
    )


def _json_digest(payload: object) -> str:
    import hashlib
    import json

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _progress() -> TrainingProgress:
    return TrainingProgress(
        global_step=300_123,
        optimizer_step=123,
        micro_step=0,
        gradient_accumulation_steps=8,
        tokens_seen=64_487_424,
    )


def _initial_progress() -> TrainingProgress:
    return TrainingProgress(
        global_step=300_000,
        optimizer_step=0,
        micro_step=0,
        gradient_accumulation_steps=8,
        tokens_seen=0,
    )


def _data_stream() -> DataStreamState:
    return DataStreamState(
        manifest_sha256="3" * 64,
        dataset_sha256="4" * 64,
        generation=2,
        seed=1_000_781,
        epoch=3,
        shard_index=7,
        sample_index=41,
        token_offset=1024,
        sampler_state={
            "permutation_cursor": 11,
            "buffer": torch.tensor([7, 2, 5], dtype=torch.int64),
        },
    )


def _episode() -> EpisodeLifecycleState:
    return EpisodeLifecycleState(
        episode_index=99,
        phase="between_episodes",
        episode_sha256=None,
        token_offset=0,
        reactor_step=0,
        source_deleted=False,
        committed=False,
        halted=False,
    )


def _save_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    str,
    EndogenousTypedTheoryReactorGPT,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.StepLR,
    BaseProvenance,
]:
    base = _base_provenance(tmp_path / "protected.pt")
    model = _small_model(2026072501)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    _advance_optimizer(model, optimizer, scheduler)
    checkpoint = tmp_path / "ettr.pt"
    digest = _save_ettr_checkpoint_for_test(
        checkpoint,
        model=model,
        protected_base=base,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        data_stream=_data_stream(),
        episode_lifecycle=_episode(),
    )
    return checkpoint, digest, model, optimizer, scheduler, base


def _load_fixture(
    checkpoint: Path,
    digest: str,
    base: BaseProvenance,
) -> tuple[
    EndogenousTypedTheoryReactorGPT,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.StepLR,
    object,
]:
    target = _small_model(2026072502)
    optimizer, scheduler = _optimizer_and_scheduler(target)
    resume = _load_ettr_checkpoint_for_test(
        checkpoint,
        expected_sha256=digest,
        model=target,
        protected_base=base,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return target, optimizer, scheduler, resume


def _rewrite(
    source: Path,
    destination: Path,
    mutation: object,
) -> tuple[Path, str]:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    mutation(payload)
    torch.save(payload, destination)
    return destination, file_sha256(destination)


def _model_snapshot(
    model: EndogenousTypedTheoryReactorGPT,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def _assert_model_unchanged(
    model: EndogenousTypedTheoryReactorGPT,
    before: dict[str, torch.Tensor],
) -> None:
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def _refresh_integrity(payload: dict[str, object]) -> None:
    payload["integrity"] = _integrity_payload(
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer"]["state"],
        scheduler_state=payload["scheduler"]["state"],
        rng_state=payload["rng_state"],
        data_stream_state=payload["data_stream_state"],
        episode_lifecycle_state=payload["episode_lifecycle_state"],
    )


def test_exact_resume_round_trip_restores_every_state_and_rng(
    tmp_path: Path,
) -> None:
    random.seed(2026072503)
    np.random.seed(2026072503)
    torch.manual_seed(2026072503)
    checkpoint, digest, source, source_optimizer, source_scheduler, base = (
        _save_fixture(tmp_path)
    )
    protected_bytes = Path(base.checkpoint_path).read_bytes()
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)

    target, optimizer, scheduler, resume = _load_fixture(
        checkpoint,
        digest,
        base,
    )

    assert resume.checkpoint_sha256 == digest
    assert resume.progress == _progress()
    assert resume.episode_lifecycle == _episode()
    assert tree_sha256(asdict(resume.data_stream)) == tree_sha256(
        asdict(_data_stream())
    )
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, target.state_dict()[name]), name
    assert tree_sha256(optimizer.state_dict()) == tree_sha256(
        source_optimizer.state_dict()
    )
    assert scheduler.state_dict() == source_scheduler.state_dict()
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)
    _advance_optimizer(source, source_optimizer, source_scheduler)
    _advance_optimizer(target, optimizer, scheduler)
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, target.state_dict()[name]), name
    assert tree_sha256(optimizer.state_dict()) == tree_sha256(
        source_optimizer.state_dict()
    )
    assert scheduler.state_dict() == source_scheduler.state_dict()
    assert Path(base.checkpoint_path).read_bytes() == protected_bytes
    assert checkpoint.read_bytes()


def test_native_muon_adam_bundle_and_embedded_schedule_resume(
    tmp_path: Path,
) -> None:
    provenance = _base_provenance(tmp_path / "protected.pt")
    config = ETTROptimizerConfig(
        warmup_updates=10,
        total_updates=100,
    )
    source = _small_model(2026072510)
    source_bundle = ETTROptimizerBundle(source, config)
    source_bundle.apply_schedule()
    checkpoint = tmp_path / "ettr-bundle.pt"
    digest = _save_ettr_checkpoint_for_test(
        checkpoint,
        model=source,
        protected_base=provenance,
        optimizer=source_bundle,
        scheduler=None,
        progress=_initial_progress(),
        data_stream=_data_stream(),
        episode_lifecycle=EpisodeLifecycleState(
            episode_index=0,
            phase="between_episodes",
            episode_sha256=None,
            token_offset=0,
            reactor_step=0,
            source_deleted=False,
            committed=False,
            halted=False,
        ),
    )

    target = _small_model(2026072511)
    target_bundle = ETTROptimizerBundle(target, config)
    resume = _load_ettr_checkpoint_for_test(
        checkpoint,
        expected_sha256=digest,
        model=target,
        protected_base=provenance,
        optimizer=target_bundle,
        scheduler=None,
    )
    assert resume.progress == _initial_progress()
    assert target_bundle.next_update == 0
    assert tree_sha256(target_bundle.state_dict()) == tree_sha256(
        source_bundle.state_dict()
    )
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, target.state_dict()[name]), name


def test_atomic_save_refuses_overwrite_and_protected_aliases(
    tmp_path: Path,
) -> None:
    checkpoint, _, model, optimizer, scheduler, base = _save_fixture(tmp_path)
    protected = Path(base.checkpoint_path)
    protected_bytes = protected.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        _save_ettr_checkpoint_for_test(
            checkpoint,
            model=model,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=_progress(),
            data_stream=_data_stream(),
            episode_lifecycle=_episode(),
        )
    with pytest.raises(ETTRCheckpointError, match="protected base"):
        _save_ettr_checkpoint_for_test(
            protected,
            model=model,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=_progress(),
            data_stream=_data_stream(),
            episode_lifecycle=_episode(),
        )
    alias = tmp_path / "protected-alias.pt"
    alias.hardlink_to(protected)
    with pytest.raises(ETTRCheckpointError, match="aliases"):
        _save_ettr_checkpoint_for_test(
            alias,
            model=model,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=_progress(),
            data_stream=_data_stream(),
            episode_lifecycle=_episode(),
        )
    assert protected.read_bytes() == protected_bytes
    assert not list(tmp_path.glob(".ettr.pt.*"))


def test_hash_is_checked_before_deserialization_or_mutation(
    tmp_path: Path,
) -> None:
    checkpoint, _, _, _, _, base = _save_fixture(tmp_path)
    target = _small_model(2026072504)
    optimizer, scheduler = _optimizer_and_scheduler(target)
    before = _model_snapshot(target)
    with pytest.raises(ETTRCheckpointError, match="file hash mismatch"):
        _load_ettr_checkpoint_for_test(
            checkpoint,
            expected_sha256="0" * 64,
            model=target,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    _assert_model_unchanged(target, before)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("schema"), "keys differ"),
        (
            lambda payload: payload.__setitem__("schema", "future"),
            "schema is invalid",
        ),
        (
            lambda payload: payload.__setitem__("future", True),
            "keys differ",
        ),
        (
            lambda payload: payload["integrity"].__setitem__(
                "future",
                True,
            ),
            "integrity keys differ",
        ),
        (
            lambda payload: payload["data_stream_state"].__setitem__(
                "schema",
                "future",
            ),
            "data stream schema",
        ),
        (
            lambda payload: payload["episode_lifecycle_state"].__setitem__(
                "schema",
                "future",
            ),
            "episode lifecycle schema",
        ),
    ],
)
def test_schema_and_state_key_drift_fail_closed(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    checkpoint, _, _, _, _, base = _save_fixture(tmp_path)
    malformed, digest = _rewrite(
        checkpoint,
        tmp_path / "malformed.pt",
        mutation,
    )
    target = _small_model(2026072505)
    optimizer, scheduler = _optimizer_and_scheduler(target)
    before = _model_snapshot(target)
    with pytest.raises(ETTRCheckpointError, match=message):
        _load_ettr_checkpoint_for_test(
            malformed,
            expected_sha256=digest,
            model=target,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    _assert_model_unchanged(target, before)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["model_state"].pop(
                next(iter(payload["model_state"]))
            ),
            "model state keys differ",
        ),
        (
            lambda payload: payload["model_state"].__setitem__(
                "unexpected.weight",
                torch.zeros(1),
            ),
            "model state keys differ",
        ),
        (
            lambda payload: (
                next(
                    tensor
                    for tensor in payload["model_state"].values()
                    if tensor.is_floating_point()
                )
                .view(-1)
                .__setitem__(0, float("nan"))
            ),
            "nonfinite",
        ),
        (
            lambda payload: payload["model_state"].__setitem__(
                next(iter(payload["model_state"])),
                payload["model_state"][next(iter(payload["model_state"]))].reshape(-1)[
                    :-1
                ],
            ),
            "shape differs",
        ),
        (
            lambda payload: payload["model_state"].__setitem__(
                next(iter(payload["model_state"])),
                payload["model_state"][next(iter(payload["model_state"]))].double(),
            ),
            "dtype differs",
        ),
    ],
)
def test_model_state_corruption_is_rejected_without_target_mutation(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    checkpoint, _, _, _, _, base = _save_fixture(tmp_path)
    malformed, digest = _rewrite(
        checkpoint,
        tmp_path / "bad-model.pt",
        mutation,
    )
    target = _small_model(2026072506)
    optimizer, scheduler = _optimizer_and_scheduler(target)
    before = _model_snapshot(target)
    with pytest.raises(ETTRCheckpointError, match=message):
        _load_ettr_checkpoint_for_test(
            malformed,
            expected_sha256=digest,
            model=target,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    _assert_model_unchanged(target, before)


def test_finite_tensor_tampering_is_caught_by_section_digest(
    tmp_path: Path,
) -> None:
    checkpoint, _, _, _, _, base = _save_fixture(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensor = next(
            tensor
            for tensor in payload["model_state"].values()
            if tensor.is_floating_point()
        )
        tensor.view(-1)[0].add_(0.25)

    malformed, digest = _rewrite(
        checkpoint,
        tmp_path / "tampered.pt",
        mutate,
    )
    target = _small_model(2026072507)
    optimizer, scheduler = _optimizer_and_scheduler(target)
    before = _model_snapshot(target)
    with pytest.raises(ETTRCheckpointError, match="section integrity"):
        _load_ettr_checkpoint_for_test(
            malformed,
            expected_sha256=digest,
            model=target,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    _assert_model_unchanged(target, before)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["optimizer"]["contract"].__setitem__(
                "class_qualname",
                "OtherOptimizer",
            ),
            "optimizer contract differs",
        ),
        (
            lambda payload: payload["optimizer"]["state"].__setitem__(
                "future",
                {},
            ),
            "optimizer state keys differ",
        ),
        (
            lambda payload: (
                next(
                    tensor
                    for slot in payload["optimizer"]["state"]["state"].values()
                    for tensor in slot.values()
                    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
                )
                .view(-1)
                .__setitem__(0, float("inf"))
            ),
            "optimizer state contains nonfinite",
        ),
        (
            lambda payload: payload["scheduler"]["contract"].__setitem__(
                "class_qualname",
                "OtherScheduler",
            ),
            "scheduler contract differs",
        ),
        (
            lambda payload: payload["scheduler"]["state"].__setitem__(
                "future",
                1,
            ),
            "scheduler state keys differ",
        ),
    ],
)
def test_optimizer_and_scheduler_corruption_fail_closed(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    checkpoint, _, _, _, _, base = _save_fixture(tmp_path)
    malformed, digest = _rewrite(
        checkpoint,
        tmp_path / "bad-training-state.pt",
        mutation,
    )
    target = _small_model(2026072508)
    optimizer, scheduler = _optimizer_and_scheduler(target)
    before = _model_snapshot(target)
    optimizer_before = tree_sha256(optimizer.state_dict())
    scheduler_before = dict(scheduler.state_dict())
    with pytest.raises(ETTRCheckpointError, match=message):
        _load_ettr_checkpoint_for_test(
            malformed,
            expected_sha256=digest,
            model=target,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    _assert_model_unchanged(target, before)
    assert tree_sha256(optimizer.state_dict()) == optimizer_before
    assert scheduler.state_dict() == scheduler_before


@pytest.mark.parametrize(
    ("section", "updates", "message"),
    [
        (
            "training_progress",
            {"micro_step": 3},
            "training progress is invalid",
        ),
        (
            "data_stream_state",
            {"sample_index": -1},
            "data-stream counters",
        ),
        (
            "episode_lifecycle_state",
            {
                "phase": "reactor",
                "episode_sha256": "5" * 64,
                "source_deleted": True,
            },
            "between-episodes boundary",
        ),
        (
            "episode_lifecycle_state",
            {"reactor_step": 5},
            "between-episode lifecycle state is inconsistent",
        ),
        (
            "episode_lifecycle_state",
            {"episode_sha256": "5" * 64},
            "between-episode lifecycle state is inconsistent",
        ),
    ],
)
def test_progress_stream_and_episode_contracts_reject_invalid_resume(
    tmp_path: Path,
    section: str,
    updates: dict[str, object],
    message: str,
) -> None:
    checkpoint, _, _, _, _, base = _save_fixture(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload[section].update(updates)
        _refresh_integrity(payload)

    malformed, digest = _rewrite(
        checkpoint,
        tmp_path / "bad-cursor.pt",
        mutate,
    )
    target = _small_model(2026072509)
    optimizer, scheduler = _optimizer_and_scheduler(target)
    before = _model_snapshot(target)
    with pytest.raises(ETTRCheckpointError, match=message):
        _load_ettr_checkpoint_for_test(
            malformed,
            expected_sha256=digest,
            model=target,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    _assert_model_unchanged(target, before)


def test_config_source_and_base_provenance_are_exact(
    tmp_path: Path,
) -> None:
    checkpoint, _, _, _, _, base = _save_fixture(tmp_path)

    mutations = (
        (
            lambda payload: payload["ettr_config"].__setitem__(
                "max_steps",
                5,
            ),
            "configuration differs",
        ),
        (
            lambda payload: payload["runtime_source_manifest"].__setitem__(
                "model.py", "0" * 64
            ),
            "runtime implementation differs",
        ),
        (
            lambda payload: payload["protected_base"].__setitem__(
                "checkpoint_sha256",
                "0" * 64,
            ),
            "another protected base",
        ),
    )
    for index, (mutation, message) in enumerate(mutations):
        malformed, digest = _rewrite(
            checkpoint,
            tmp_path / f"wrong-binding-{index}.pt",
            mutation,
        )
        target = _small_model(2026072510 + index)
        optimizer, scheduler = _optimizer_and_scheduler(target)
        before = _model_snapshot(target)
        with pytest.raises(ETTRCheckpointError, match=message):
            _load_ettr_checkpoint_for_test(
                malformed,
                expected_sha256=digest,
                model=target,
                protected_base=base,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        _assert_model_unchanged(target, before)


def test_relocated_base_path_is_portable_but_file_is_reverified(
    tmp_path: Path,
) -> None:
    checkpoint, digest, _, _, _, base = _save_fixture(tmp_path)
    relocated_path = tmp_path / "relocated-protected.pt"
    relocated_path.write_bytes(Path(base.checkpoint_path).read_bytes())
    relocated = replace(base, checkpoint_path=str(relocated_path.resolve()))
    target, _, _, _ = _load_fixture(checkpoint, digest, relocated)
    assert target is not None
    relocated_path.write_bytes(b"mutated")
    target = _small_model(2026072513)
    optimizer, scheduler = _optimizer_and_scheduler(target)
    with pytest.raises(ETTRCheckpointError, match="size changed"):
        _load_ettr_checkpoint_for_test(
            checkpoint,
            expected_sha256=digest,
            model=target,
            protected_base=relocated,
            optimizer=optimizer,
            scheduler=scheduler,
        )


def test_production_apis_reject_a_synthetic_trust_root(
    tmp_path: Path,
) -> None:
    base = _base_provenance(tmp_path / "protected.pt")
    model = _small_model(2026072514)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    with pytest.raises(ETTRCheckpointError, match="not the protected"):
        save_ettr_checkpoint(
            tmp_path / "production.pt",
            model=model,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=_progress(),
            data_stream=_data_stream(),
            episode_lifecycle=_episode(),
        )
    with pytest.raises(ETTRCheckpointError, match="byte count differs"):
        load_protected_base_provenance(Path(base.checkpoint_path))
    with pytest.raises(ETTRCheckpointError, match="not the protected"):
        load_ettr_checkpoint(
            tmp_path / "missing.pt",
            expected_sha256="0" * 64,
            model=model,
            protected_base=base,
            optimizer=optimizer,
            scheduler=scheduler,
        )


def test_nonaliasing_snapshot_and_runtime_manifest(tmp_path: Path) -> None:
    checkpoint, _, model, optimizer, scheduler, _ = _save_fixture(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["runtime_source_manifest"] == runtime_source_manifest()
    name = next(iter(payload["model_state"]))
    saved = payload["model_state"][name].clone()
    with torch.no_grad():
        model.state_dict()[name].add_(1)
    assert torch.equal(payload["model_state"][name], saved)
    optimizer_name = next(iter(payload["optimizer"]["state"]["state"]))
    optimizer_tensor = next(
        tensor
        for tensor in payload["optimizer"]["state"]["state"][optimizer_name].values()
        if isinstance(tensor, torch.Tensor)
    )
    assert optimizer_tensor.device.type == "cpu"
    assert scheduler.state_dict()


def test_resume_accepts_scheduler_mutated_current_learning_rate(
    tmp_path: Path,
) -> None:
    base = _base_provenance(tmp_path / "protected.pt")
    model = _small_model(2026072515)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    _advance_optimizer(model, optimizer, scheduler)
    for _ in range(9):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    checkpoint = tmp_path / "decayed.pt"
    digest = _save_ettr_checkpoint_for_test(
        checkpoint,
        model=model,
        protected_base=base,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        data_stream=_data_stream(),
        episode_lifecycle=_episode(),
    )
    target, target_optimizer, _, _ = _load_fixture(
        checkpoint,
        digest,
        base,
    )
    assert target is not None
    assert target_optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
