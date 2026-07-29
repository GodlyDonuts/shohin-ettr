from __future__ import annotations

from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path
import stat

import pytest
import torch

import profile_ettr_h100 as profile
from model import GPT, GPTConfig


def _settings(mode: str) -> profile.ProfileSettings:
    return profile.ProfileSettings(
        mode=mode,
        batch_size=4,
        microsteps=1,
        warmup_updates=0,
        measured_updates=1,
        world_tokens=8,
        command_tokens=4,
        query_tokens=5,
        reactor_steps=2,
        learning_rate=1e-3,
        seed=1234,
        train_scope="architecture",
        compile_mode="default",
    )


def _checkpoint(path: Path, *, step: int = 7) -> str:
    torch.manual_seed(17)
    config = GPTConfig(
        vocab_size=64,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        d_model=32,
        d_ff=64,
        seq_len=32,
        zloss=0.0,
    )
    model = GPT(config)
    torch.save(
        {
            "cfg": vars(config),
            "model": model.state_dict(),
            "step": step,
        },
        path,
    )
    return profile.sha256_file(path)


def test_synthetic_batches_are_deterministic_factorial_continuations() -> None:
    settings = _settings("cpu-validation")
    model = profile._tiny_model(settings.seed)
    objective_config = profile._objective_config(model)
    first, first_hash = profile.synthetic_batches(
        settings,
        reactor_config=model.config,
        objective_config=objective_config,
    )
    second, second_hash = profile.synthetic_batches(
        settings,
        reactor_config=model.config,
        objective_config=objective_config,
    )
    assert first_hash == second_hash
    for left, right in zip(first, second, strict=True):
        assert left.manifest_sha256 == right.manifest_sha256
        assert left.dataset_sha256 == right.dataset_sha256
        assert left.episodes.episode_ids == right.episodes.episode_ids
        assert torch.equal(
            left.episodes.reset_mask,
            right.episodes.reset_mask,
        )
        assert torch.equal(
            left.episodes.query_read_index,
            right.episodes.query_read_index,
        )
        for name in ("world", "command", "query"):
            left_segment = getattr(left.episodes, name)
            right_segment = getattr(right.episodes, name)
            for field in ("tokens", "targets", "attention_mask"):
                left_tensor = getattr(left_segment, field)
                right_tensor = getattr(right_segment, field)
                assert left_tensor.device.type == "cpu"
                assert torch.equal(left_tensor, right_tensor)
            assert left_segment.tokens.dtype == torch.long
            assert left_segment.targets.dtype == torch.long
            assert left_segment.attention_mask.dtype == torch.bool
        assert torch.equal(
            left.causal_rectangles.rows,
            torch.tensor([[[0, 1], [2, 3]]]),
        )
        assert left.episodes.query_read_index.tolist() == [0, 0, 0, 0]
        assert torch.equal(
            left.episodes.query.tokens[:, :1],
            left.episodes.query.tokens[:1, :1].expand(4, -1),
        )
        assert left.episodes.query.targets[:, 0].tolist() == [4, 5, 6, 7]
        assert not torch.equal(
            left.episodes.world.tokens[0],
            left.episodes.world.tokens[1],
        )
        assert not torch.equal(
            left.episodes.world.tokens[2],
            left.episodes.world.tokens[3],
        )
        assert not torch.equal(
            left.episodes.command.tokens[0],
            left.episodes.command.tokens[2],
        )
        assert not torch.equal(
            left.episodes.command.tokens[1],
            left.episodes.command.tokens[3],
        )
        assert not torch.equal(
            left.episodes.world.tokens[0],
            left.episodes.world.tokens[2],
        )
        assert not torch.equal(
            left.episodes.command.tokens[0],
            left.episodes.command.tokens[1],
        )
        assert torch.equal(
            left.packet_targets.value_code[0],
            left.packet_targets.value_code[1],
        )
        assert torch.equal(
            left.packet_targets.value_code[2],
            left.packet_targets.value_code[3],
        )
        for target_name in (
            "packet_targets",
            "terminal_packet_targets",
            "transaction_targets",
        ):
            left_target = getattr(left, target_name)
            right_target = getattr(right, target_name)
            for field in fields(left_target):
                assert torch.equal(
                    getattr(left_target, field.name),
                    getattr(right_target, field.name),
                )
        (
            _world_packet,
            _world_command,
            world_target,
            _command_packet,
            _command_command,
            command_target,
        ) = left.causal_rectangles.intervention_indices()
        assert torch.equal(
            left.terminal_packet_targets.value_code.index_select(
                0,
                world_target,
            ),
            left.terminal_packet_targets.value_code[
                torch.tensor([2, 3, 0, 1])
            ],
        )
        assert torch.equal(
            left.terminal_packet_targets.value_code.index_select(
                0,
                command_target,
            ),
            left.terminal_packet_targets.value_code[
                torch.tensor([1, 0, 3, 2])
            ],
        )
        left.validate(model.config, objective_config)

    changed = profile.ProfileSettings(
        **{**profile.asdict(settings), "seed": settings.seed + 1}
    )
    _, changed_hash = profile.synthetic_batches(
        changed,
        reactor_config=model.config,
        objective_config=objective_config,
    )
    assert changed_hash != first_hash


def test_profile_geometry_requires_complete_factorial_rectangles() -> None:
    with pytest.raises(profile.ETTRProfileError, match="divisible by four"):
        profile.ProfileSettings(
            **{**profile.asdict(_settings("dry-run")), "batch_size": 6}
        ).validate()


def test_encoded_token_accounting_matches_actual_encoder_calls() -> None:
    settings = _settings("cpu-validation")
    assert profile._encoded_tokens_per_update(settings) == (
        settings.batch_size
        * settings.microsteps
        * (
            settings.world_tokens
            + 2 * settings.command_tokens
            + 3 * settings.query_tokens
        )
    )


def test_parameter_sampling_detects_a_later_tensor_update() -> None:
    first = torch.nn.Parameter(torch.arange(10_000, dtype=torch.float32))
    later = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    before = profile._sample_parameters((first, later), maximum=8)
    with torch.no_grad():
        later[0].add_(3.0)
    after = profile._sample_parameters((first, later), maximum=8)
    assert torch.equal(before[:4], after[:4])
    assert float((after - before).abs().sum()) == 3.0
    torch.testing.assert_close(
        before,
        profile._sample_parameters(
            (
                torch.nn.Parameter(torch.arange(10_000, dtype=torch.float32)),
                torch.nn.Parameter(torch.arange(8, dtype=torch.float32)),
            ),
            maximum=8,
        ),
    )


def test_parameter_sampling_fails_when_budget_cannot_cover_every_tensor() -> None:
    parameters = tuple(
        torch.nn.Parameter(torch.zeros(1))
        for _ in range(5)
    )
    with pytest.raises(profile.ETTRProfileError, match="every trainable tensor"):
        profile._sample_parameters(parameters, maximum=4)


def test_parameter_sampling_indices_are_exact_above_float32_boundary() -> None:
    length = 2**24 + 1
    indices = profile._evenly_spaced_indices(
        length,
        4096,
        device=torch.device("cpu"),
    )
    assert indices.dtype == torch.long
    assert indices[0] == 0
    assert indices[-1] == length - 1
    assert bool(torch.all(indices[1:] > indices[:-1]))
    assert int(indices.min()) >= 0
    assert int(indices.max()) < length


def test_reactor_core_gradient_component_excludes_command_projection() -> None:
    model = profile._tiny_model(7)
    components = profile._component_parameters(model)
    command_ids = {
        id(parameter) for parameter in components["command_projection"]
    }
    reactor_core_ids = {
        id(parameter) for parameter in components["reactor_core"]
    }
    reactor_ids = {
        id(parameter) for parameter in components["reactor"]
    }
    assert command_ids
    assert reactor_core_ids
    assert command_ids.isdisjoint(reactor_core_ids)
    assert command_ids | reactor_core_ids == reactor_ids


def test_output_custody_rejects_existing_aliases_and_symlinks(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(profile.ETTRProfileError, match="existing"):
        profile.validate_output_directory(
            existing,
            protected_paths=(protected,),
        )
    with pytest.raises(profile.ETTRProfileError, match="protected"):
        profile.validate_output_directory(
            protected / "profile",
            protected_paths=(protected,),
        )
    with pytest.raises(profile.ETTRProfileError, match="protected"):
        profile.validate_output_directory(
            tmp_path / "parent",
            protected_paths=(tmp_path / "parent" / "protected-child",),
        )
    physical = tmp_path / "physical"
    physical.mkdir()
    link = tmp_path / "link"
    link.symlink_to(physical, target_is_directory=True)
    with pytest.raises(profile.ETTRProfileError, match="symlink"):
        profile.validate_output_directory(
            link / "profile",
            protected_paths=(protected,),
        )


def test_dry_run_is_explicit_sealed_and_writes_no_model_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dry"
    report = profile.run(
        settings=_settings("dry-run"),
        output_dir=output,
        protected_paths=(),
    )
    assert report["execution"] == {
        "executed": False,
        "validation_only": True,
    }
    assert report["schema"] == "shohin-ettr-h100-profile-v5"
    assert report["sync_points"] == list(profile.SYNC_POINTS)
    assert report["custody"]["pretraining_started"] is False
    assert report["custody"]["model_or_optimizer_state_written"] is False
    assert {path.name for path in output.iterdir()} == {"report.json"}
    mode = stat.S_IMODE((output / "report.json").stat().st_mode)
    assert mode == 0o400
    assert stat.S_IMODE(output.stat().st_mode) == 0o500


def test_cpu_validation_runs_bf16_microstep_and_receipts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cpu"
    report = profile.run(
        settings=_settings("cpu-validation"),
        output_dir=output,
        protected_paths=(),
    )
    assert report["mode"] == "cpu-validation"
    assert report["comparison"]["available"] is True
    assert report["comparison"]["matched_batch_sha256"] is True
    assert report["comparison"]["matched_parameter_receipt"] is True
    assert report["gates"] == {
        "compiled_arm_completed": True,
        "eager_arm_completed": True,
        "matched_batch_sha256": True,
        "matched_initial_parameter_sha256": True,
        "matched_parameter_receipt": True,
    }
    for arm_name in ("eager", "compiled"):
        arm = report["arms"][arm_name]
        assert arm["status"] == "completed"
        assert arm["execution"]["executed"] is True
        assert arm["execution"]["subject"] == (
            "ETTRCompositeFactorialInterventionSubject"
        )
        assert arm["execution"]["full_composite_objective"] is True
        assert arm["execution"]["hard_transactions"] is True
        assert arm["execution"]["factual_episode_path"] is True
        assert arm["execution"]["intervention_arms"] == [
            "WORLD",
            "COMMAND",
        ]
        assert arm["execution"]["factual_validate_batch_in_hot_path"] is False
        assert (
            arm["execution"]["intervention_validate_batch_in_hot_path"]
            is True
        )
        assert arm["execution"]["forward_backward_optimizer"] is True
        assert arm["execution"]["last_logits_dtype"] == "torch.bfloat16"
        assert arm["execution"]["optimizer"] == ("ettr_muon_plus_adamw")
        assert arm["gates"] == {
            "bf16_autocast_exercised": True,
            "gradient_receipt_pass": True,
            "isolated_query_binding_eager_bf16_gradient_receipt_pass": True,
            "loss_finite": True,
            "parameter_cap_pass": True,
        }
        assert arm["parameters"]["architecture_parameters"] > 0
        assert (
            arm["parameters"]["complete_system_parameters"]
            == arm["parameters"]["base_parameters"]
            + arm["parameters"]["architecture_parameters"]
        )
        assert (
            arm["parameters"]["optimizer_receipt"]["unique_trainable_parameters"]
            == arm["parameters"]["architecture_parameters"]
        )
        assert len(arm["parameters"]["initial_parameter_sha256"]) == 64
        assert arm["batch"]["source"] == (
            "validated_deterministic_synthetic_ettr_continuation_rectangles"
        )
        assert arm["batch"]["causal_rectangles_per_microstep"] == 1
        assert arm["batch"]["factual_rows_per_microstep"] == 4
        assert arm["batch"]["intervention_rows_per_arm_per_microstep"] == 4
        assert arm["batch"]["objective_targets"] == (
            "factual_rows_gathered_by_rectangle_indices"
        )
        assert arm["batch"]["episode_segments"] == [
            "WORLD",
            "COMMAND",
            "QUERY",
        ]
        assert arm["batch"]["encoded_tokens_per_update"] == 124
        assert set(arm["execution"]["objective_losses"]) == set(
            profile.OBJECTIVE_LOSS_NAMES
        )
        for receipt in arm["execution"]["objective_losses"].values():
            assert torch.isfinite(torch.tensor(receipt["last"]))
            assert torch.isfinite(torch.tensor(receipt["mean"]))
        for name in ("compiler", "reactor", "query_reader"):
            receipt = arm["gradients"][name]
            assert receipt["gradient_nonzero_elements"] > 0
            assert receipt["gradient_nonfinite_elements"] == 0
            assert receipt["sampled_parameter_abs_delta"] > 0
        assert arm["gradients"]["base"]["gradient_tensors"] == 0
        isolated = arm["isolated_query_binding_gradients"]
        expected_positive = {
            "world": ("compiler", "reactor_core", "query_reader"),
            "command": (
                "command_projection",
                "reactor_core",
                "query_reader",
            ),
        }
        for causal_arm, components in expected_positive.items():
            treatment = isolated[causal_arm]["treatment"]
            detached = isolated[causal_arm]["detached_state"]
            for component in components:
                assert treatment[component]["gradient_nonzero_elements"] > 0
                assert treatment[component]["gradient_nonfinite_elements"] == 0
            assert detached["query_reader"]["gradient_nonzero_elements"] > 0
            assert detached["compiler"]["gradient_nonzero_elements"] == 0
            assert detached["reactor_core"]["gradient_nonzero_elements"] == 0
            assert treatment["base"]["gradient_nonzero_elements"] == 0
            assert detached["base"]["gradient_nonzero_elements"] == 0
        assert (
            isolated["command"]["detached_state"]["command_projection"][
                "gradient_nonzero_elements"
            ]
            == 0
        )
        isolated_execution = arm["isolated_query_binding_execution"]
        assert isolated_execution["autocast_dtype"] == "torch.bfloat16"
        assert isolated_execution["compiled"] is False
        assert isolated_execution["elapsed_ms"] > 0
        assert isolated_execution["peak_allocated_bytes"] == 0
        assert isolated_execution["peak_reserved_bytes"] == 0
        assert isolated_execution["purpose"] == (
            "causal_path_attribution_not_throughput"
        )
    persisted = json.loads((output / "report.json").read_text())
    assert persisted == report


def test_checkpoint_loader_is_read_only_hash_and_step_bound(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "base.pt"
    expected_hash = _checkpoint(checkpoint)
    before = checkpoint.stat()
    payload, opened = profile.load_checkpoint_read_only(
        checkpoint,
        expected_sha256=expected_hash,
        expected_step=7,
    )
    after = checkpoint.stat()
    assert payload["step"] == 7
    assert opened.st_ino == before.st_ino == after.st_ino
    assert opened.st_mtime_ns == before.st_mtime_ns == after.st_mtime_ns
    assert profile.sha256_file(checkpoint) == expected_hash
    with pytest.raises(profile.ETTRProfileError, match="hash"):
        profile.load_checkpoint_read_only(
            checkpoint,
            expected_sha256="0" * 64,
            expected_step=7,
        )
    with pytest.raises(profile.ETTRProfileError, match="step"):
        profile.load_checkpoint_read_only(
            checkpoint,
            expected_sha256=expected_hash,
            expected_step=8,
        )


def test_non_h100_modes_reject_checkpoint_and_live_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "base.pt"
    expected_hash = _checkpoint(checkpoint)
    with pytest.raises(profile.ETTRProfileError, match="may not read"):
        profile.run(
            settings=_settings("dry-run"),
            output_dir=tmp_path / "out",
            checkpoint_path=checkpoint,
            checkpoint_sha256=expected_hash,
            expected_step=7,
            protected_paths=(),
        )
    monkeypatch.setenv("SHARDS", "/live/data")
    with pytest.raises(profile.ETTRProfileError, match="SHARDS"):
        profile.run(
            settings=_settings("dry-run"),
            output_dir=tmp_path / "shards-out",
            protected_paths=(),
        )
    assert not (tmp_path / "shards-out").exists()


def test_h100_mode_requires_complete_checkpoint_binding(
    tmp_path: Path,
) -> None:
    with pytest.raises(profile.ETTRProfileError, match="requires checkpoint"):
        profile.run(
            settings=_settings("h100"),
            output_dir=tmp_path / "profile",
            protected_paths=(),
        )
    assert not (tmp_path / "profile").exists()


def test_h100_admission_is_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _device: "NVIDIA A100-SXM4-80GB",
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (8, 0),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    with pytest.raises(profile.ETTRProfileError, match="H100"):
        profile.require_h100(torch.device("cuda"))


def test_compile_failure_is_reported_without_disguised_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("synthetic compiler unavailable")

    monkeypatch.setattr(profile, "_compile_subject", unavailable)
    result = profile.execute_profile_arms(
        lambda: profile._tiny_model(1234),
        _settings("cpu-validation"),
        device=torch.device("cpu"),
        device_receipt={
            "name": "cpu-validation",
            "validation_only": True,
        },
    )
    assert result["arms"]["eager"]["status"] == "completed"
    assert result["arms"]["compiled"] == {
        "status": "unavailable",
        "error_type": "RuntimeError",
        "error": "synthetic compiler unavailable",
    }
    assert result["comparison"] == {
        "available": False,
        "compiled_attempted": True,
    }
    assert result["gates"]["compiled_arm_completed"] is False


def test_profiler_has_declared_sync_points_and_no_hot_loop_item() -> None:
    source = Path(profile.__file__).read_text()
    assert "Tensor.item()" in source
    hot_loop = inspect.getsource(profile._run_update)
    assert ".item(" not in hot_loop
    assert ".cpu(" not in hot_loop
    assert "CausalETTREpisodeRunner" in source
    assert "ETTRCompositeObjective" in source
    assert "forward_staged" not in source
    assert "output.losses.token_lm" not in hot_loop
    subject = inspect.getsource(
        profile.ETTRCompositeFactorialInterventionSubject
    )
    assert "validate_batch=False" in subject
    assert subject.count("hard=True") == 2
    assert "batch.objective_batch(output, interventions)" in subject
    assert "self.objective" in subject
    assert profile.SYNC_POINTS == (
        "cuda_after_warmup_before_measurement",
        "cuda_after_all_measured_updates_before_receipt",
        "cuda_between_arms_before_allocator_cleanup",
    )


def test_slurm_wrapper_is_profile_only_and_never_submits() -> None:
    wrapper = Path(profile.__file__).with_name("profile_ettr_h100.sbatch").read_text()
    executable = "\n".join(
        line
        for line in wrapper.splitlines()
        if line and not line.lstrip().startswith("#")
    )
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in wrapper
    assert "#SBATCH --cpus-per-task=4" in wrapper
    assert "--mode h100" in wrapper
    assert "profile_ettr_h100.py" in wrapper
    assert "unset SHARDS" in executable
    assert "train.py" not in executable
    assert "ShardLoader" not in executable
    assert "sbatch " not in executable.lower()
    assert "--checkpoint-sha256" in executable
    assert "--expected-step" in executable
    assert "--output-dir" in executable
    assert "--compile-mode" in executable
    assert '--batch-size "${BATCH_SIZE:-4}"' in executable


def test_report_bytes_are_canonical() -> None:
    value = {"z": 1, "a": [True, None]}
    payload = profile.canonical_json_bytes(value)
    assert hashlib.sha256(payload).hexdigest()
    assert payload == b'{"a":[true,null],"z":1}\n'
