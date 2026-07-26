from __future__ import annotations

# ruff: noqa: E402

import ast
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from ettr_il_v2_custody import cj1_dumps, cj1_loads, sha256_bytes
from ettr_il_v2_source_deletion import (
    ANSWER_SCHEMA,
    COMMAND_SOURCE_SCHEMA,
    PROTOCOL,
    QUERY_SOURCE_SCHEMA,
    REHEARSAL_SCHEMA,
    SEALED_PACKET_SCHEMA,
    SourceDeletionError,
    TERMINAL_PACKET_SCHEMA,
    WORLD_SOURCE_SCHEMA,
    _invoke_stage_worker,
    _prepare_stage_input,
    _source_packages,
    run_source_deletion_rehearsal,
)


def _read(path: Path) -> object:
    return cj1_loads(path.read_bytes())


def _walk_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            *(str(key).lower() for key in value),
            *(
                child_key
                for child in value.values()
                for child_key in _walk_keys(child)
            ),
        ]
    if isinstance(value, list):
        return [key for child in value for key in _walk_keys(child)]
    return []


def test_clean_and_poison_lanes_use_hard_process_boundaries(
    tmp_path: Path,
) -> None:
    run = run_source_deletion_rehearsal(tmp_path / "rehearsal")
    report = run.receipt

    assert report["schema"] == REHEARSAL_SCHEMA
    assert report["protocol"] == PROTOCOL
    assert report["status"] == "pass"
    assert report["mode"] == "cpu_only_no_model_no_fit"
    assert report["artifact_invariance_under_poison_replacement"] == {
        "WORLD": True,
        "COMMAND": True,
        "QUERY": True,
    }
    assert report["hard_process_boundaries"] == {
        "all_workers_differ_from_supervisor": True,
        "process_launch_receipts_unique": True,
        "spawned_stage_processes": 6,
    }
    assert run.receipt_sha256 == sha256_bytes(run.receipt_path.read_bytes())
    assert run.receipt_path.stat().st_mode & 0o222 == 0

    launches: list[str] = []
    for lane_name in ("clean_lane", "poisoned_lane"):
        lane = report[lane_name]
        assert lane["denied_upstream_source_probes"] == {
            "WORLD": 0,
            "COMMAND": 1,
            "QUERY": 2,
        }
        assert lane["package_removed_before_successor"] == {
            "WORLD": True,
            "COMMAND": True,
            "QUERY": True,
        }
        assert set(lane["worker_pids"]) == {"WORLD", "COMMAND", "QUERY"}
        assert all(pid != os.getpid() for pid in lane["worker_pids"].values())
        assert all(
            parent == os.getpid()
            for parent in lane["worker_parent_pids"].values()
        )
        launches.extend(lane["process_launch_receipts"])
    assert len(launches) == len(set(launches)) == 6

    assert report["clean_lane"]["poison_replacement_sha256s"] == {
        "WORLD": None,
        "COMMAND": None,
        "QUERY": None,
    }
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in report["poisoned_lane"][
            "poison_replacement_sha256s"
        ].values()
    )


def test_only_fixed_packet_and_answer_artifacts_cross_boundaries(
    tmp_path: Path,
) -> None:
    run = run_source_deletion_rehearsal(tmp_path / "rehearsal")
    forbidden = (
        "residual",
        "hidden_state",
        "kv_cache",
        "key_cache",
        "value_cache",
        "source_payload",
        "source_bytes",
    )
    expected_schemas = {
        "world": SEALED_PACKET_SCHEMA,
        "command": TERMINAL_PACKET_SCHEMA,
        "query": ANSWER_SCHEMA,
    }
    for lane in ("clean", "poisoned"):
        for stage, expected_schema in expected_schemas.items():
            output = run.root / lane / "outputs" / stage
            assert {path.name for path in output.iterdir()} == {
                (
                    "sealed_packet.json"
                    if stage == "world"
                    else (
                        "terminal_packet.json"
                        if stage == "command"
                        else "answer.json"
                    )
                ),
                "receipt.json",
            }
            artifact_name = next(
                name
                for name in {
                    "sealed_packet.json",
                    "terminal_packet.json",
                    "answer.json",
                }
                if (output / name).exists()
            )
            artifact = _read(output / artifact_name)
            assert artifact["schema"] == expected_schema
            if stage == "query":
                assert artifact["answer"] is True
            assert not any(
                marker in key
                for key in _walk_keys(artifact)
                for marker in forbidden
            )
        for stage in ("world", "command", "query"):
            assert not (run.root / lane / "inputs" / stage).exists()
    assert run.receipt["transfer_closure"] == {
        "digest_only_source_binding": True,
        "kv_cache_transfer_forbidden": True,
        "raw_upstream_source_transfer_forbidden": True,
        "residual_state_transfer_forbidden": True,
        "sealed_packet_is_only_interstage_semantic_artifact": True,
    }
    assert run.receipt["clean_lane"]["transfer_scan"] == {
        "all_primary_source_payloads_absent": True,
        "all_primary_source_sentinels_absent": True,
        "all_replacement_payloads_absent": True,
        "all_replacement_sentinels_absent": True,
    }
    assert (
        run.receipt["poisoned_lane"]["transfer_scan"]
        == run.receipt["clean_lane"]["transfer_scan"]
    )


@pytest.mark.parametrize(
    "extra_name",
    ("residual.json", "kv_cache.json", "copied_source.json"),
)
def test_worker_rejects_any_extra_transfer_file(
    tmp_path: Path,
    extra_name: str,
) -> None:
    source = _source_packages(poison=False)["WORLD"]
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    _prepare_stage_input(
        input_directory,
        stage="WORLD",
        payloads={"source.json": source},
    )
    input_directory.chmod(0o755)
    path = input_directory / extra_name
    path.write_bytes(cj1_dumps({"forbidden": True}))
    path.chmod(0o444)
    input_directory.chmod(0o555)

    with pytest.raises(SourceDeletionError, match="worker failed closed"):
        _invoke_stage_worker(
            stage="WORLD",
            input_directory=input_directory,
            output_directory=output_directory,
            spawn_nonce="0" * 32,
        )


def test_worker_rejects_mutable_source_package(tmp_path: Path) -> None:
    source = _source_packages(poison=False)["WORLD"]
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    _prepare_stage_input(
        input_directory,
        stage="WORLD",
        payloads={"source.json": source},
    )
    input_directory.chmod(0o755)
    (input_directory / "source.json").chmod(0o644)
    input_directory.chmod(0o555)

    with pytest.raises(SourceDeletionError, match="worker failed closed"):
        _invoke_stage_worker(
            stage="WORLD",
            input_directory=input_directory,
            output_directory=output_directory,
            spawn_nonce="1" * 32,
        )


def test_parent_receipt_tamper_fails_closed_in_fresh_command_process(
    tmp_path: Path,
) -> None:
    run = run_source_deletion_rehearsal(tmp_path / "rehearsal")
    world_output = run.root / "clean" / "outputs" / "world"
    world_packet = (world_output / "sealed_packet.json").read_bytes()
    world_receipt = cj1_loads((world_output / "receipt.json").read_bytes())
    world_receipt["output_artifact_sha256"] = "f" * 64
    tampered_receipt = cj1_dumps(world_receipt)
    deletion_receipt = (
        run.root / "clean" / "deletions" / "world.json"
    ).read_bytes()
    command_source = _source_packages(poison=False)["COMMAND"]
    input_directory = tmp_path / "tampered_command_input"
    output_directory = tmp_path / "tampered_command_output"
    _prepare_stage_input(
        input_directory,
        stage="COMMAND",
        payloads={
            "source.json": command_source,
            "upstream_packet.json": world_packet,
            "upstream_receipt.json": tampered_receipt,
            "upstream_deletion_receipt.json": deletion_receipt,
        },
        parent_receipt_sha256s=(
            sha256_bytes(tampered_receipt),
            sha256_bytes(deletion_receipt),
        ),
    )

    with pytest.raises(SourceDeletionError, match="worker failed closed"):
        _invoke_stage_worker(
            stage="COMMAND",
            input_directory=input_directory,
            output_directory=output_directory,
            spawn_nonce="2" * 32,
        )


def test_poison_packages_are_valid_and_contradict_primary_semantics() -> None:
    primary = {
        stage: cj1_loads(payload)
        for stage, payload in _source_packages(poison=False).items()
    }
    poison = {
        stage: cj1_loads(payload)
        for stage, payload in _source_packages(poison=True).items()
    }
    assert primary["WORLD"]["schema"] == WORLD_SOURCE_SCHEMA
    assert primary["COMMAND"]["schema"] == COMMAND_SOURCE_SCHEMA
    assert primary["QUERY"]["schema"] == QUERY_SOURCE_SCHEMA
    assert primary["WORLD"]["cells"] != poison["WORLD"]["cells"]
    assert primary["COMMAND"]["operations"] != poison["COMMAND"]["operations"]
    assert primary["QUERY"]["predicate"] != poison["QUERY"]["predicate"]
    assert all(
        _source_packages(poison=False)[stage]
        != _source_packages(poison=True)[stage]
        for stage in ("WORLD", "COMMAND", "QUERY")
    )


def test_module_has_no_training_checkpoint_or_job_surface() -> None:
    source_path = Path(__file__).with_name("ettr_il_v2_source_deletion.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    called_attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ):
            called_attributes.add(node.func.attr)
    assert "torch" not in imported_roots
    assert imported_roots.isdisjoint(
        {
            "ettr_checkpoint",
            "ettr_optimization",
            "ettr_train_step",
            "torch",
        }
    )
    assert called_attributes.isdisjoint(
        {
            "backward",
            "load_state_dict",
            "save",
            "save_file",
            "step",
            "zero_grad",
        }
    )
