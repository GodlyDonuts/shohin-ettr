from __future__ import annotations

import ast
from dataclasses import asdict
import inspect
import json
from pathlib import Path

import pytest
import torch

import ssqac_on_policy_search_dagger_pilot as pilot


@pytest.fixture(scope="module")
def search_config() -> pilot.base.SearchPreparationConfig:
    return pilot.base.SearchPreparationConfig(
        max_nodes_expanded=512,
        max_edges_considered=10_000,
        max_depth=24,
        max_frontier=32,
        beam_width=32,
        max_program_instructions=96,
        policy_noise_scale=4.0,
    )


@pytest.fixture(scope="module")
def off_policy_batch(
    tmp_path_factory: pytest.TempPathFactory,
    search_config: pilot.base.SearchPreparationConfig,
) -> pilot.OffPolicyBatch:
    matrices = pilot.base.generate_matrix_cases(
        seed=73,
        count=4,
        minimum_rows=2,
        maximum_rows=3,
        minimum_columns=2,
        maximum_columns=4,
    )
    return pilot.prepare_off_policy_batch(
        matrices,
        seed=73,
        states_per_case=4,
        search_config=search_config,
        scratch_root=tmp_path_factory.mktemp("off-policy-search"),
    )


def test_off_policy_preparation_deletes_search_traces_and_matches_controls(
    off_policy_batch: pilot.OffPolicyBatch,
) -> None:
    labels = off_policy_batch.labels
    assert labels.trace_directory_deleted
    assert labels.retained_trace_files == 0
    assert int(labels.resources["search_calls"]) == 4
    assert int(labels.resources["search_failures"]) == 0
    assert len(labels.search_states) > 0
    assert (
        len(labels.search_states)
        == len(labels.ordinary_states)
        == len(labels.random_states)
    )
    observation_manifests = {
        pilot._observation_manifest(states)
        for states in (
            labels.search_states,
            labels.ordinary_states,
            labels.random_states,
        )
    }
    assert len(observation_manifests) == 1
    for search, ordinary, randomized in zip(
        labels.search_states,
        labels.ordinary_states,
        labels.random_states,
        strict=True,
    ):
        assert search.rows == ordinary.rows == randomized.rows
        legal = pilot.base.enumerate_policy_actions(search.rows)
        assert search.target in legal
        assert ordinary.target in legal
        assert randomized.target in legal
        if len(legal) > 1:
            assert randomized.target != search.target


def test_on_policy_labeling_deletes_raw_programs_and_receipts(
    tmp_path: Path,
    off_policy_batch: pilot.OffPolicyBatch,
    search_config: pilot.base.SearchPreparationConfig,
) -> None:
    observations = tuple(
        pilot.CollectedObservation(rows=state.rows, pre_error=True)
        for state in off_policy_batch.labels.search_states[:3]
    )
    labels = pilot.label_observations_with_search(
        observations,
        seed=79,
        search_config=search_config,
        scratch_root=tmp_path,
    )
    assert labels.trace_directory_deleted
    assert labels.retained_trace_files == 0
    assert int(labels.resources["trace_files_written_then_deleted"]) == len(
        observations
    )
    assert not tuple(tmp_path.glob("ssqac-op-dagger-traces-*"))


def test_model_only_collector_has_no_search_or_assessor_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = pilot.base.PolicyConfig(
        maximum_rows=4,
        maximum_columns=5,
        width=16,
        blocks=1,
        feedforward=32,
    )

    class HaltOnlyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config

        def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
            action_kind = inputs["action_kind"]
            halt_index = pilot.base.ACTION_TYPES.index(pilot.base.ACTION_HALT)
            return torch.where(
                action_kind == halt_index,
                torch.ones_like(action_kind, dtype=torch.float32),
                torch.full_like(action_kind, -1.0, dtype=torch.float32),
            )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("preparation or assessor dependency was called")

    monkeypatch.setattr(pilot, "label_observations_with_search", forbidden)
    monkeypatch.setattr(pilot, "prepare_off_policy_batch", forbidden)
    monkeypatch.setattr(
        pilot.base,
        "evaluate_autonomous_policy",
        forbidden,
    )
    observations, receipt = pilot.collect_on_policy_observations(
        HaltOnlyPolicy(),
        (((1, 0), (0, 1)),),
        device=torch.device("cpu"),
        amp_bfloat16=False,
        maximum_macros=4,
        maximum_instructions=8,
        states_per_case=2,
        state_budget=2,
    )
    assert len(observations) == 1
    assert receipt.halted == 1
    assert receipt.model_decisions == 1
    assert receipt.final_search_calls == 0
    assert receipt.final_oracle_calls == 0
    assert receipt.final_verifier_calls == 0

    tree = ast.parse(inspect.getsource(pilot.collect_on_policy_observations))
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    for forbidden_name in (
        "bounded_beam_candidate_search",
        "structural_potential",
        "compile_reference_program",
        "verify_reduction_program",
        "evaluate_autonomous_policy",
    ):
        assert forbidden_name not in called


def test_geometry_holdout_must_be_strict_on_both_axes() -> None:
    args = pilot._parse_args(["--smoke"])
    args.evaluation_minimum_columns = args.fit_maximum_columns
    with pytest.raises(pilot.OnPolicySearchDAggerError):
        pilot.run_pilot(args)


def test_h100_cli_exposes_escalation_and_matched_budget_controls() -> None:
    args = pilot._parse_args(
        [
            "--device",
            "cuda",
            "--initial-cases",
            "384",
            "--dagger-rounds",
            "3",
            "--collection-cases",
            "128",
            "--on-policy-state-budget",
            "768",
            "--evaluation-cases",
            "512",
            "--width",
            "384",
            "--blocks",
            "6",
            "--feedforward",
            "1536",
            "--collector-epochs",
            "30",
            "--final-epochs",
            "60",
            "--batch-size",
            "512",
        ]
    )
    assert args.device == "cuda"
    assert args.initial_cases == 384
    assert args.dagger_rounds == 3
    assert args.collection_cases == 128
    assert args.on_policy_state_budget == 768
    assert args.evaluation_cases == 512
    assert (args.width, args.blocks, args.feedforward) == (384, 6, 1536)
    assert args.collector_epochs == 30
    assert args.final_epochs == 60
    assert args.batch_size == 512


def test_bounded_cpu_smoke_is_source_sealed_and_fully_matched(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    models = tmp_path / "models"
    assert (
        pilot.main(
            [
                "--smoke",
                "--scratch-root",
                str(tmp_path),
                "--model-output-directory",
                str(models),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="ascii"))
    assert report["schema"] == pilot.SCHEMA
    assert report["status"] == pilot.STATUS
    assert report["reasoning_claim_authorized"] is False
    assert report["reference_baseline_certified"] == 28
    assert report["reference_baseline_total"] == 768
    assert report["reference_baseline_evaluation_seed_xor"] == 0xE7A1
    assert report["paired_reference_evaluation_cases"] is True
    assert report["strict_geometry_disjoint"] is True
    assert report["candidate_runtime"] == (
        "unmasked_greedy_model_plus_primitive_vm_only"
    )
    assert report["matched_examples"] is True
    assert report["matched_parameters"] is True
    assert report["matched_updates"] is True
    assert report["matched_batch_schedule"] is True
    assert report["identical_initial_weights"] is True
    assert report["final_search_calls"] == 0
    assert report["final_oracle_calls"] == 0
    assert report["inference_verifier_calls"] == 0

    preparation = report["preparation"]
    assert preparation["source_directory_deleted"] is True
    assert preparation["retained_source_files"] == 0
    assert preparation["all_search_trace_directories_deleted"] is True
    assert preparation["retained_search_trace_files"] == 0
    assert preparation["total_search_calls"] > 0
    assert preparation["total_ordinary_oracle_calls"] > 0
    assert not tuple(tmp_path.glob("ssqac-op-dagger-source-*"))
    assert not tuple(tmp_path.glob("ssqac-*-traces-*"))

    assert len(report["rounds"]) == 1
    round_report = report["rounds"][0]
    assert (
        round_report["on_policy_search_labeled_states"]
        == (round_report["no_dagger_off_policy_states"])
    )
    assert round_report["all_search_traces_deleted"] is True
    assert round_report["collection"]["final_search_calls"] == 0
    assert round_report["collection"]["final_oracle_calls"] == 0
    assert round_report["collection"]["final_verifier_calls"] == 0

    arms = report["arms"]
    assert {arm["name"] for arm in arms} == set(pilot.ARMS)
    assert len({arm["examples"] for arm in arms}) == 1
    assert len({arm["parameters"] for arm in arms}) == 1
    assert len({arm["optimizer_updates"] for arm in arms}) == 1
    assert len({arm["batch_schedule_sha256"] for arm in arms}) == 1
    assert len({arm["initial_model_sha256"] for arm in arms}) == 1
    assert all(arm["final_search_calls"] == 0 for arm in arms)
    assert all(arm["final_oracle_calls"] == 0 for arm in arms)
    assert all(arm["inference_verifier_calls"] == 0 for arm in arms)
    assert all(arm["assessor_verifier_calls"] == 8 for arm in arms)
    assert all(arm["artifact"]["bytes"] > 0 for arm in arms)
    assert all(len(arm["artifact"]["sha256"]) == 64 for arm in arms)
    assert {path.name for path in models.iterdir()} == {
        arm["artifact"]["filename"] for arm in arms
    }

    canonical = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert output.read_text(encoding="ascii") == canonical + "\n"
    assert all(
        len(value) == 64
        for value in (
            report["initial_source_manifest_sha256"],
            report["collection_source_manifest_sha256"],
            report["evaluation_manifest_sha256"],
            preparation["source_manifest_sha256"],
            preparation["search_receipt_manifest_sha256"],
            preparation["search_trace_manifest_sha256"],
        )
    )


def test_report_schema_contains_exact_resource_fields() -> None:
    required = {
        "slurm_job_id",
        "node",
        "device",
        "gpu_name",
        "slurm_cpus",
        "peak_cuda_memory_bytes",
        "training_wall_seconds",
        "evaluation_wall_seconds",
        "total_wall_seconds",
    }
    assert required <= set(pilot.EscalationReport.__dataclass_fields__)
    assert "assessor_verifier_calls" in {
        field.name for field in pilot.FinalArmReport.__dataclass_fields__.values()
    }
    assert asdict(pilot.SearchResources(search_calls=1))["search_calls"] == 1
