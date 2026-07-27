from __future__ import annotations

import copy
import collections
import base64
import hashlib
import inspect
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

import causal_carry_motor as upstream
import causal_carry_motor_recovery as recovery


NEW_COMMIT = "b" * 40
EXPECTED_SHARD_RECEIPTS = (
    "4affa12434513ebe9587464ff38656abaaf7e47904d9db6ced252c3adea52a96",
    "4731c1644703e26c1978ca1ec1ba80af7c173c5d9676ae68fbd04368f3b54c2c",
    "e81639e68a838bfa6695be92f7c1333d100b2317c48fb2cf0d995f22a6e50a43",
    "ae86ec1b70dca21d67849fc4be17ffec682472851735c3b9523292836a74e70f",
    "ce5a151f89e20e774c7d37afc446ea026ec14a587c70fa614414f060f10a2144",
    "f02d8221bf3a393566c279e27bf888fcbd1ef9ea17bdd33262472c898950ea83",
    "009b83f0c2a70362654e3e3e4cad27d30f79f93f3bdd32d6ce3064695dd2b9db",
    "8214d356288c56a116a3de753a8948a35f731d52c520fa906f4e31c1b0f14fb4",
)
ISOLATED_TRANSITIVE_MUTATIONS = (
    ("torch.tensor", "recovery.torch", "tensor"),
    ("torch.stack", "recovery.torch", "stack"),
    ("torch.nextafter", "recovery.torch", "nextafter"),
    ("torch.optim.AdamW", "recovery.torch.optim", "AdamW"),
    ("torch.optim.adamw.adamw", "recovery.torch_adamw_module", "adamw"),
    ("torch.optim.adam.adam", "recovery.torch_adam_module", "adam"),
    ("torch.optim.AdamW.step", "recovery.torch.optim.AdamW", "step"),
    (
        "torch.optim.Optimizer.zero_grad",
        "recovery.torch.optim.Optimizer",
        "zero_grad",
    ),
    ("torch.nn.Module.__call__", "recovery.torch.nn.Module", "__call__"),
    ("torch.nn.Module.to", "recovery.torch.nn.Module", "to"),
    ("torch.nn.Module.cpu", "recovery.torch.nn.Module", "cpu"),
    ("torch.nn.Module.eval", "recovery.torch.nn.Module", "eval"),
    ("torch.nn.Module.parameters", "recovery.torch.nn.Module", "parameters"),
    ("torch.nn.Module.load_state_dict", "recovery.torch.nn.Module", "load_state_dict"),
    ("torch.nn.Module.state_dict", "recovery.torch.nn.Module", "state_dict"),
    ("torch.Tensor.backward", "recovery.torch.Tensor", "backward"),
    ("torch.load", "recovery.torch", "load"),
    ("torch.save", "recovery.torch", "save"),
    ("torch._C", "recovery.torch", "_C"),
    ("tokenizers.tokenizers", "recovery.tokenizers_module", "tokenizers"),
    ("upstream.rollout_episode", "recovery.upstream", "rollout_episode"),
    (
        "digitwise_controller.rollout_episode",
        "recovery.digitwise_controller_module",
        "rollout_episode",
    ),
    (
        "rollout_episode.__globals__.parse_state",
        "recovery.digitwise_controller_module",
        "parse_state",
    ),
    *tuple(
        (
            f"torch.serialization.{name}",
            "recovery.torch_serialization_module",
            name,
        )
        for name in recovery._SERIALIZATION_HELPER_NAMES
    ),
)


@pytest.fixture(autouse=True)
def _restore_process_umask():
    inherited = os.umask(recovery.SECURE_CREATION_UMASK)
    os.umask(inherited)
    try:
        yield
    finally:
        os.umask(inherited)


def _raw_board(rows):
    return {
        "attempts": 100,
        "forbidden_prompt_count": 7,
        "position_counts": {"add|4|0|core|0|1|0": 1},
        "prefix_order_sha256": "1" * 64,
        "prompt_length_histogram": {97: 3, 99: 4, 103: 5, 105: 6},
        "quota": 1,
        "rows": len(rows),
        "seed": 20260717,
        "strata": {"add|4|0|core|0|1": 1},
        "token_length_histogram": {114: 3, 116: 4, 120: 5, 122: 6},
    }


def _normalization_fixture(monkeypatch):
    rows = [{"prefix_sha256": "a" * 64, "target": 1, "target_id": 29}]
    raw = _raw_board(rows)
    raw_with_hash = copy.deepcopy(raw)
    raw_with_hash["rows_sha256"] = upstream.stable_json_sha256(rows)
    sealed = recovery.normalize_generated_board_document(raw_with_hash)
    monkeypatch.setattr(
        recovery, "UPSTREAM_BOARD_ROWS_SHA256", upstream.stable_json_sha256(rows)
    )
    monkeypatch.setattr(
        recovery, "UPSTREAM_CANONICAL_BOARD_SHA256", recovery.stable_json_sha256(sealed)
    )
    return rows, raw, sealed


def _minimal_source_contract():
    return {
        "schema": recovery.RECOVERY_EXECUTOR_SOURCE_SCHEMA,
        "git_commit": NEW_COMMIT,
        "sources": {},
        "manifest_sha256": "1" * 64,
    }


def _minimal_recovery_plan(monkeypatch, tmp_path):
    upstream_root = tmp_path / "upstream"
    recovery_parent = tmp_path / "recoveries"
    monkeypatch.setattr(recovery, "UPSTREAM_ROOT", upstream_root)
    monkeypatch.setattr(recovery, "RECOVERY_PARENT", recovery_parent)
    root = recovery.recovery_root(NEW_COMMIT)
    source_contract = _minimal_source_contract()
    parent_binding = recovery.ensure_recovery_parent(
        source_contract, allow_install=True
    )
    layout_binding = recovery.reserve_recovery_layout(
        root, source_contract, parent_binding
    )
    document = {
        "audit": recovery.RECOVERY_PLAN_AUDIT,
        "recovery": True,
        "recovery_plan_path": str(root / "recovery_plan.json"),
        "recovery_executor_source_contract": source_contract,
        "executor_runtime_contract": {
            "schema": recovery.RECOVERY_EXECUTOR_RUNTIME_SCHEMA,
            "source_root": "/reviewed/recovery",
        },
        "recovery_parent_binding": parent_binding,
        "hostile_review_binding": {
            "path": "/review.json",
            "sha256": "2" * 64,
            "document": {},
        },
        "upstream_protocol": {
            "source_contract": {"git_commit": recovery.UPSTREAM_SOURCE_COMMIT},
            "plan_binding": {"sha256": recovery.UPSTREAM_PLAN_SHA256},
            "shard_receipts": [],
        },
        "normalization_proof": {"mismatch_count": 2},
        "allowed_transformation": recovery.ALLOWED_TRANSFORMATION,
        "downstream_evaluation_contract": recovery.DOWNSTREAM_EVALUATION_CONTRACT,
        "fit_contract": {
            "fit_budget": {
                "seed": upstream.FIT_SEED,
                "rank": upstream.RANK,
                "quota": upstream.FIT_QUOTA,
                "updates": upstream.CANONICAL_UPDATES,
                "batch_size": upstream.CANONICAL_BATCH,
                "lr": upstream.CANONICAL_LR,
                "weight_decay": upstream.CANONICAL_WEIGHT_DECAY,
            }
        },
        "output_contract": {
            "root": str(root),
            "fit_artifact": str(root / "fit" / "motor.pt"),
            "development_eval_artifact": str(
                root / "development_eval" / "evaluation.json"
            ),
            "confirmation_eval_artifact": str(
                root / "confirmation_eval" / "evaluation.json"
            ),
            "upstream_root_must_remain_untouched": str(upstream_root),
            "layout_binding": layout_binding,
            "slurm_h100_contract": recovery.EXPECTED_SLURM_REQUEST,
        },
        "deserialization_contract": recovery.DESERIALIZATION_CONTRACT,
        "claim_boundary": recovery.RECOVERY_PLAN_CLAIM_BOUNDARY,
    }
    return root, document


def _git(repo, *arguments):
    return subprocess.check_output(
        [str(recovery.PINNED_GIT), "-C", str(repo), *arguments], text=True
    ).strip()


def _isolated_python_command(repo, source):
    bootstrap = """
import sys, sysconfig
from pathlib import Path
root = Path(sys.argv[1])
source = sys.argv[2]
candidates = (
    root / "train",
    Path(sysconfig.get_path("stdlib")),
    Path(sysconfig.get_config_var("DESTSHARED")),
    Path(sysconfig.get_path("purelib")),
    Path(sysconfig.get_path("platlib")),
)
paths = []
for candidate in candidates:
    value = str(candidate.resolve(strict=True))
    if value not in paths:
        paths.append(value)
sys.path[:] = paths
exec(compile(source, "<isolated-recovery-test>", "exec"), {"__name__": "__main__"})
"""
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        bootstrap,
        str(repo),
        source,
    ]


def _source_contract_repo(monkeypatch, tmp_path, *, extra_paths=()):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    subprocess.run([str(recovery.PINNED_GIT), "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            str(recovery.PINNED_GIT),
            "-C",
            str(repo),
            "config",
            "user.email",
            "test@example.invalid",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(recovery.PINNED_GIT),
            "-C",
            str(repo),
            "config",
            "user.name",
            "Recovery Test",
        ],
        check=True,
    )
    (repo / "baseline.txt").write_text("sealed upstream\n")
    subprocess.run(
        [str(recovery.PINNED_GIT), "-C", str(repo), "add", "baseline.txt"],
        check=True,
    )
    subprocess.run(
        [str(recovery.PINNED_GIT), "-C", str(repo), "commit", "-qm", "upstream"],
        check=True,
    )
    upstream_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(recovery, "UPSTREAM_SOURCE_COMMIT", upstream_commit)
    for index, name in enumerate((*recovery.RECOVERY_SOURCE_PATHS, *extra_paths)):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"recovery source {index}\n")
    subprocess.run(
        [str(recovery.PINNED_GIT), "-C", str(repo), "add", "--all"], check=True
    )
    subprocess.run(
        [str(recovery.PINNED_GIT), "-C", str(repo), "commit", "-qm", "recovery"],
        check=True,
    )
    recovery_commit = _git(repo, "rev-parse", "HEAD")
    sources = {
        name: recovery.sha256_file(repo / name)
        for name in recovery.RECOVERY_SOURCE_PATHS
    }
    monkeypatch.setattr(
        recovery,
        "validate_loaded_module_paths",
        lambda _root: {
            "recovery": str(repo / "train" / "causal_carry_motor_recovery.py"),
            "upstream": str(repo / "train" / "causal_carry_motor.py"),
            "model": str(repo / "train" / "model.py"),
            "digitwise_controller": str(repo / "train" / "digitwise_controller.py"),
            "digitwise_protocol": str(repo / "train" / "digitwise_protocol.py"),
            "eval_suite": str(repo / "train" / "eval_suite.py"),
            "probe_digitwise_workspace": str(
                repo / "train" / "probe_digitwise_workspace.py"
            ),
        },
    )
    return repo, upstream_commit, recovery_commit, recovery.stable_json_sha256(sources)


def test_exact_upstream_receipts_are_complete_and_cross_file_bound():
    root = Path(__file__).resolve().parents[1]
    prereg = (root / "R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md").read_text()
    wrapper = (root / "train/jobs/causal_carry_motor_recovery.sbatch").read_text()
    assert recovery.UPSTREAM_SOURCE_COMMIT == (
        "a0c258e6709766c643cf127a429a7d6ef4a4211b"
    )
    assert recovery.UPSTREAM_PLAN_SHA256 == (
        "1b845d47f6875df571169efb5adb0716dfbc5d266a2499e4a92451351a262b6d"
    )
    assert recovery.UPSTREAM_SHARD_SHA256 == EXPECTED_SHARD_RECEIPTS
    assert len(set(recovery.UPSTREAM_SHARD_SHA256)) == 8
    for receipt in (
        recovery.UPSTREAM_SOURCE_COMMIT,
        recovery.UPSTREAM_SOURCE_MANIFEST_SHA256,
        recovery.UPSTREAM_PLAN_SHA256,
        recovery.UPSTREAM_CONFIRMATION_COMMITMENT_SHA256,
        recovery.UPSTREAM_BOARD_ROWS_SHA256,
        recovery.UPSTREAM_CANONICAL_BOARD_SHA256,
        recovery.NORMALIZATION_MISMATCH_LEDGER_SHA256,
        *EXPECTED_SHARD_RECEIPTS,
    ):
        assert prereg.count(receipt) >= 1
    assert recovery.UPSTREAM_SOURCE_COMMIT in wrapper
    assert recovery.UPSTREAM_PLAN_SHA256 in wrapper
    for audit in (
        recovery.RECOVERY_PLAN_AUDIT,
        recovery.RECOVERY_FIT_AUDIT,
        recovery.RECOVERY_REVIEW_AUDIT,
    ):
        assert audit in prereg


def test_normalization_proof_has_exactly_two_key_type_differences(monkeypatch):
    rows, raw, sealed = _normalization_fixture(monkeypatch)
    proof, normalized = recovery.build_normalization_proof(raw, sealed, rows)
    assert proof["mismatch_count"] == 2
    assert proof["mismatches"] == list(recovery.EXPECTED_NORMALIZATION_MISMATCHES)
    assert proof["canonical_board_equal"] is True
    assert recovery.type_strict_equal(normalized, sealed)
    assert proof["allowed_transformation"] == recovery.ALLOWED_TRANSFORMATION


def test_normalization_rejects_non_histogram_difference(monkeypatch):
    rows, raw, sealed = _normalization_fixture(monkeypatch)
    sealed["attempts"] += 1
    monkeypatch.setattr(
        recovery, "UPSTREAM_CANONICAL_BOARD_SHA256", recovery.stable_json_sha256(sealed)
    )
    with pytest.raises(ValueError, match="non-histogram board difference"):
        recovery.build_normalization_proof(raw, sealed, rows)


def test_normalization_rejects_histogram_value_or_extra_key_change(monkeypatch):
    rows, raw, sealed = _normalization_fixture(monkeypatch)
    sealed["prompt_length_histogram"]["97"] += 1
    monkeypatch.setattr(
        recovery, "UPSTREAM_CANONICAL_BOARD_SHA256", recovery.stable_json_sha256(sealed)
    )
    with pytest.raises(ValueError, match="differs beyond JSON key typing"):
        recovery.build_normalization_proof(raw, sealed, rows)

    rows, raw, sealed = _normalization_fixture(monkeypatch)
    raw["prompt_length_histogram"]["97"] = raw["prompt_length_histogram"][97]
    with pytest.raises(ValueError, match="generated keys are not all integers"):
        recovery.build_normalization_proof(raw, sealed, rows)


def test_strict_json_and_equality_reject_python_type_aliases():
    assert not recovery.type_strict_equal(True, 1)
    assert not recovery.type_strict_equal(1, 1.0)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        recovery.load_exact_json('{"x":1,"x":2}', "hostile")
    with pytest.raises(ValueError, match="non-finite"):
        recovery.canonical_json_payload({"x": float("nan")})
    assert not recovery.type_strict_equal({1: {"x": 1}}, {True: {"x": 1}})
    assert not recovery.type_strict_equal(
        {"nested": {1: "value"}}, {"nested": {True: "value"}}
    )
    with pytest.raises(ValueError, match="non-string JSON key"):
        recovery.canonical_json_payload({"nested": [{1: "value"}]})
    with pytest.raises(ValueError, match="non-JSON type"):
        recovery.canonical_json_payload({"nested": (1, 2)})
    recursive = []
    recursive.append(recursive)
    with pytest.raises(ValueError, match="recursive JSON container"):
        recovery.canonical_json_payload(recursive)


@pytest.mark.parametrize("literal", ("1e999", "-1e999"))
def test_strict_json_rejects_exponent_overflow(literal):
    assert math.isinf(json.loads(literal))
    with pytest.raises(ValueError, match=rf"non-finite JSON float: {literal}"):
        recovery.load_exact_json(f'{{"nested":[{literal}]}}', "hostile overflow")


def test_strict_json_accepts_largest_finite_exponent():
    observed = recovery.load_exact_json('{"value":1e308}', "finite JSON")
    assert type(observed["value"]) is float
    assert math.isfinite(observed["value"])


def test_bound_file_rejects_alias_and_receipt_substitution(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"sealed")
    digest = hashlib.sha256(b"sealed").hexdigest()
    alias = tmp_path / "alias.bin"
    alias.symlink_to(artifact)
    with pytest.raises(ValueError, match="aliases or differs"):
        recovery.BoundFile(str(alias), artifact, digest, "artifact")
    with pytest.raises(ValueError, match="hash mismatch"):
        recovery.BoundFile(str(artifact), artifact, "0" * 64, "artifact")
    bound = recovery.BoundFile(str(artifact), artifact, digest, "artifact")
    try:
        assert bound.bytes() == b"sealed"
    finally:
        bound.close()


def test_bound_file_rejects_symlinked_ancestor_even_with_exact_leaf_bytes(tmp_path):
    physical = tmp_path / "physical"
    physical.mkdir()
    artifact = physical / "artifact.bin"
    artifact.write_bytes(b"sealed")
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    aliased_artifact = alias / artifact.name
    with pytest.raises(ValueError, match="symlinked or non-directory ancestor"):
        recovery.BoundFile(
            str(aliased_artifact),
            aliased_artifact,
            recovery.sha256_file(artifact),
            "ancestor alias attack",
        )


def _upstream_custody_fixture(monkeypatch, tmp_path):
    root = tmp_path / "canonical"
    root.mkdir()
    plan_path = root / "plan.json"
    plan_path.write_bytes(b"sealed plan\n")
    os.chmod(plan_path, 0o444)
    for name in ("fit", "development_eval", "confirmation_eval"):
        (root / name).mkdir(mode=0o700)
    shard_hashes = []
    for index in range(upstream.CANONICAL_FEATURE_SHARDS):
        directory = root / f"shard_{index:02d}"
        directory.mkdir()
        artifact = directory / "features.pt"
        artifact.write_bytes(f"sealed shard {index}\n".encode())
        shard_hashes.append(recovery.sha256_file(artifact))
        os.chmod(artifact, 0o444)
        os.chmod(directory, 0o555)
    commitment_dir = tmp_path / "commitment"
    commitment_dir.mkdir()
    commitment_path = commitment_dir / "commitment.json"
    commitment_path.write_bytes(b"sealed commitment\n")
    os.chmod(commitment_path, 0o444)
    os.chmod(commitment_dir, 0o555)
    os.chmod(root, 0o555)
    monkeypatch.setattr(recovery, "UPSTREAM_ROOT", root)
    monkeypatch.setattr(recovery, "UPSTREAM_PLAN_PATH", plan_path)
    monkeypatch.setattr(recovery, "UPSTREAM_CONFIRMATION_PATH", commitment_path)
    monkeypatch.setattr(
        recovery, "UPSTREAM_PLAN_SHA256", recovery.sha256_file(plan_path)
    )
    monkeypatch.setattr(
        recovery,
        "UPSTREAM_CONFIRMATION_COMMITMENT_SHA256",
        recovery.sha256_file(commitment_path),
    )
    monkeypatch.setattr(recovery, "UPSTREAM_SHARD_SHA256", tuple(shard_hashes))
    return root, commitment_path


def test_upstream_custody_snapshot_covers_every_directory_file_and_mode(
    monkeypatch, tmp_path
):
    root, commitment_path = _upstream_custody_fixture(monkeypatch, tmp_path)
    snapshot = recovery.capture_upstream_custody_snapshot()
    entries = {entry["path"]: entry for entry in snapshot["entries"]}
    assert snapshot["schema"] == recovery.UPSTREAM_CUSTODY_SCHEMA
    assert len(entries) == 23
    assert entries[str(root)]["identity"]["mode"] == 0o555
    for name in ("fit", "development_eval", "confirmation_eval"):
        entry = entries[str(root / name)]
        assert entry["identity"]["mode"] == 0o700
        assert entry["children"] == []
    for index in range(upstream.CANONICAL_FEATURE_SHARDS):
        directory = root / f"shard_{index:02d}"
        assert entries[str(directory)]["children"] == ["features.pt"]
        assert entries[str(directory / "features.pt")]["identity"]["mode"] == 0o444
    assert entries[str(commitment_path)]["sha256"] == recovery.sha256_file(
        commitment_path
    )
    assert recovery.assert_upstream_custody_unchanged(snapshot, "during test")


def test_upstream_custody_revalidation_rejects_empty_dir_shard_and_inode_attacks(
    monkeypatch, tmp_path
):
    root, _commitment = _upstream_custody_fixture(monkeypatch, tmp_path)
    snapshot = recovery.capture_upstream_custody_snapshot()
    injected = root / "fit" / "foreign.pt"
    injected.write_bytes(b"attack")
    with pytest.raises(ValueError, match="not closed-world"):
        recovery.assert_upstream_custody_unchanged(snapshot, "after publication")
    injected.unlink()
    snapshot = recovery.capture_upstream_custody_snapshot()

    shard_dir = root / "shard_00"
    os.chmod(shard_dir, 0o755)
    with pytest.raises(ValueError, match="mode mismatch"):
        recovery.assert_upstream_custody_unchanged(snapshot, "after publication")
    os.chmod(shard_dir, 0o555)
    snapshot = recovery.capture_upstream_custody_snapshot()

    artifact = shard_dir / "features.pt"
    original = artifact.read_bytes()
    os.chmod(shard_dir, 0o755)
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(original)
    os.chmod(replacement, 0o444)
    os.replace(replacement, artifact)
    os.chmod(shard_dir, 0o555)
    with pytest.raises(RuntimeError, match="custody changed"):
        recovery.assert_upstream_custody_unchanged(snapshot, "after publication")


def test_fit_revalidates_full_custody_immediately_around_publish_and_seal():
    source = inspect.getsource(recovery._fit)
    before_publish = source.index("immediately before recovery artifact publication")
    publish = source.index("publish_recovery_torch(")
    after_publish = source.index("immediately after recovery artifact publication")
    before_seal = source.index("immediately before recovery artifact sealing")
    seal = source.index("seal_recovery_fit(out, signed_layout_binding)", before_seal)
    after_seal = source.index("immediately after recovery artifact sealing")
    assert before_publish < publish < after_publish
    assert before_seal < seal < after_seal


def test_safe_torch_load_is_weights_only_allowlisted_and_rejects_env(
    monkeypatch, tmp_path
):
    artifact = tmp_path / "payload.pt"
    torch.save({"runtime": torch.torch_version.TorchVersion("2.6.0+cu124")}, artifact)
    digest = recovery.sha256_file(artifact)
    bound = recovery.BoundFile(str(artifact), artifact, digest, "torch payload")
    try:
        loaded = recovery.safe_torch_load(bound)
        assert str(loaded["runtime"]) == "2.6.0+cu124"
        monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        with pytest.raises(RuntimeError, match="override is forbidden"):
            recovery.safe_torch_load(bound)
    finally:
        bound.close()


def test_recovery_runtime_must_equal_frozen_upstream_runtime(monkeypatch):
    monkeypatch.setattr(recovery, "assert_reviewed_callable_exports", lambda: {})
    monkeypatch.setattr(
        recovery, "capture_slurm_h100_attestation", lambda: {"ok": True}
    )
    monkeypatch.setattr(recovery, "REVIEWED_REQUIRE_CANONICAL_CUDA", lambda: "cuda")
    monkeypatch.setattr(
        recovery,
        "REVIEWED_CUDA_GET_DEVICE_NAME",
        lambda _index: "NVIDIA H100 PCIe",
    )
    expected = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": "NVIDIA H100 PCIe",
    }
    assert recovery.require_recovery_cuda_runtime(
        {"runtime_contract": {"artifact_runtime": expected}}
    ) == ("cuda", {"ok": True})
    hostile = copy.deepcopy(expected)
    hostile["torch"] = "different"
    with pytest.raises(RuntimeError, match="differs from frozen"):
        recovery.require_recovery_cuda_runtime(
            {"runtime_contract": {"artifact_runtime": hostile}}
        )


def test_board_reconstruction_freezes_row_order_control_schedule_and_init(monkeypatch):
    rows, raw, sealed = _normalization_fixture(monkeypatch)

    class FrozenText:
        def __init__(self, value):
            self.value = value
            self.verified = False

        def text(self):
            return self.value

        def verify(self):
            self.verified = True

    tokenizer_bound = FrozenText("tokenizer")
    episodes_bound = FrozenText("episodes")
    frozen = {"tokenizer": tokenizer_bound, "episodes": episodes_bound}
    tokenizer = object()
    control_labels = torch.tensor([1])
    control = {"permutation_sha256": "4" * 64}
    schedule_sha256 = "5" * 64
    initial_sha256 = "6" * 64
    plan = {
        "board": sealed,
        "extraction_order_sha256": upstream.stable_json_sha256(
            [row["prefix_sha256"] for row in rows]
        ),
        "fit_budget": {
            "control": control,
            "batch_size": 512,
            "updates": 2000,
            "seed": 20260717,
            "schedule_sha256": schedule_sha256,
            "initial_state_sha256": initial_sha256,
        },
        "d_model": 8,
    }
    monkeypatch.setattr(
        recovery,
        "REVIEWED_TOKENIZER_FROM_STR",
        lambda value: tokenizer if value == "tokenizer" else None,
    )
    monkeypatch.setattr(recovery, "assert_reviewed_callable_exports", lambda: {})
    monkeypatch.setattr(
        recovery,
        "REVIEWED_GENERATE_FIT_ROWS",
        lambda observed_tokenizer, observed_episodes, *_: (
            (
                rows,
                raw,
            )
            if observed_tokenizer is tokenizer and observed_episodes == "episodes"
            else None
        ),
    )
    monkeypatch.setattr(
        recovery,
        "REVIEWED_PERMUTED_CONTROL_LABELS",
        lambda _rows: (control_labels, control),
    )
    monkeypatch.setattr(
        recovery,
        "REVIEWED_BATCH_SCHEDULE",
        lambda *_: (None, schedule_sha256),
    )
    monkeypatch.setattr(
        recovery,
        "REVIEWED_INITIAL_MOTOR_STATE",
        lambda _d_model: ({"state": "frozen"}, initial_sha256),
    )

    result = recovery.reconstruct_board(plan, frozen)
    assert result["control_labels"] is control_labels
    assert all(bound.verified for bound in frozen.values())

    hostile = copy.deepcopy(plan)
    hostile["extraction_order_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="row order changed"):
        recovery.reconstruct_board(hostile, frozen)

    hostile = copy.deepcopy(plan)
    hostile["fit_budget"]["control"] = {"permutation_sha256": "0" * 64}
    with pytest.raises(ValueError, match="control changed"):
        recovery.reconstruct_board(hostile, frozen)

    hostile = copy.deepcopy(plan)
    hostile["fit_budget"]["schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schedule changed"):
        recovery.reconstruct_board(hostile, frozen)

    hostile = copy.deepcopy(plan)
    hostile["fit_budget"]["initial_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="initial motor state changed"):
        recovery.reconstruct_board(hostile, frozen)


def test_executor_source_contract_is_distinct_clean_and_commit_bound(
    monkeypatch, tmp_path
):
    repo, upstream_commit, commit, manifest = _source_contract_repo(
        monkeypatch, tmp_path
    )
    contract = recovery.build_recovery_executor_source_contract(
        commit, manifest, repo_root=repo
    )
    assert contract["git_commit"] == commit
    assert contract["parent_commit"] == upstream_commit
    assert contract["name_status_diff"] == list(recovery.RECOVERY_NAME_STATUS_DIFF)
    assert contract["manifest_sha256"] == manifest
    (repo / recovery.RECOVERY_SOURCE_PATHS[1]).write_text("dirty attack\n")
    with pytest.raises(ValueError, match="not clean"):
        recovery.build_recovery_executor_source_contract(
            commit, manifest, repo_root=repo
        )
    with pytest.raises(ValueError, match="may not alias"):
        recovery.build_recovery_executor_source_contract(
            upstream_commit,
            manifest,
            repo_root=repo,
        )


def test_recovery_source_topology_is_exactly_the_four_non_v9_paths():
    expected = (
        "R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md",
        "train/causal_carry_motor_recovery.py",
        "train/test_causal_carry_motor_recovery.py",
        "train/jobs/causal_carry_motor_recovery.sbatch",
    )
    assert recovery.RECOVERY_SOURCE_PATHS == expected
    assert recovery._PREIMPORT_RECOVERY_SOURCE_PATHS == expected
    assert all("_v9" not in path for path in expected)
    assert recovery.RECOVERY_NAME_STATUS_DIFF == tuple(
        f"A\t{path}" for path in sorted(expected)
    )


def test_executor_source_contract_rejects_extra_file_and_grandchild(
    monkeypatch, tmp_path
):
    extra_repo, _upstream, extra_commit, manifest = _source_contract_repo(
        monkeypatch, tmp_path / "extra", extra_paths=("train/model.py",)
    )
    with pytest.raises(ValueError, match="exactly four added"):
        recovery.build_recovery_executor_source_contract(
            extra_commit,
            manifest,
            repo_root=extra_repo,
        )

    child_repo, _upstream, recovery_commit, manifest = _source_contract_repo(
        monkeypatch, tmp_path / "grandchild"
    )
    subprocess.run(
        [
            str(recovery.PINNED_GIT),
            "-C",
            str(child_repo),
            "commit",
            "--allow-empty",
            "-qm",
            "unreviewed grandchild",
        ],
        check=True,
    )
    grandchild = _git(child_repo, "rev-parse", "HEAD")
    assert grandchild != recovery_commit
    with pytest.raises(ValueError, match="sole direct parent"):
        recovery.build_recovery_executor_source_contract(
            grandchild,
            manifest,
            repo_root=child_repo,
        )


def test_executor_source_contract_rejects_ignored_shadow_file(monkeypatch, tmp_path):
    repo, _upstream, commit, manifest = _source_contract_repo(monkeypatch, tmp_path)
    exclude = repo / ".git" / "info" / "exclude"
    with exclude.open("a") as sink:
        sink.write("train/sitecustomize.py\n")
    shadow = repo / "train" / "sitecustomize.py"
    shadow.write_text("raise RuntimeError('shadow executed')\n")
    with pytest.raises(ValueError, match="not closed-world"):
        recovery.build_recovery_executor_source_contract(
            commit,
            manifest,
            repo_root=repo,
        )


def test_executor_source_contract_rejects_hard_link_alias(monkeypatch, tmp_path):
    repo, _upstream, commit, manifest = _source_contract_repo(monkeypatch, tmp_path)
    source = repo / recovery.RECOVERY_SOURCE_PATHS[0]
    alias = tmp_path / "source-alias.md"
    os.link(source, alias)
    with pytest.raises(ValueError, match="checkout file identity mismatch"):
        recovery.build_recovery_executor_source_contract(
            commit,
            manifest,
            repo_root=repo,
        )


def test_loaded_module_shadow_is_rejected(monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    recovery.validate_loaded_module_paths(repo)
    shadow = tmp_path / "model.py"
    shadow.write_text("# hostile shadow\n")
    monkeypatch.setattr(recovery.model_module, "__file__", str(shadow))
    with pytest.raises(ValueError, match="shadowed or aliased"):
        recovery.validate_loaded_module_paths(repo)


def test_runtime_contract_uses_fixed_launcher_and_rejects_environment_aliases():
    assert (
        "launcher_path"
        not in inspect.signature(recovery.capture_executor_runtime_contract).parameters
    )
    repo = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
from pathlib import Path
import causal_carry_motor_recovery as recovery
recovery.PINNED_PYTHON_LAUNCHER = Path(sys.executable)
recovery.enforce_secure_creation_umask()
print(json.dumps(recovery.capture_executor_runtime_contract(), sort_keys=True))
"""
    environment = os.environ.copy()
    for name in tuple(environment):
        if name in recovery.FORBIDDEN_EXECUTOR_ENVIRONMENT or name in (
            recovery.forbidden_executor_environment_names(environment)
        ):
            environment.pop(name, None)
    environment.update(recovery.EXECUTOR_ENVIRONMENT)
    clean = subprocess.run(
        _isolated_python_command(repo, script),
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(clean.stdout)
    assert contract["schema"] == recovery.RECOVERY_EXECUTOR_RUNTIME_SCHEMA
    assert contract["source_root"] == str(repo)
    assert contract["python"]["flags"]["no_user_site"] == 1
    assert contract["python"]["flags"]["no_site"] == 1
    assert contract["python"]["flags"]["isolated"] == 1
    assert contract["python"]["flags"]["ignore_environment"] == 1
    dependencies = contract["dependency_manifest"]
    assert set(dependencies["files"]) == {
        "fcntl",
        "torch",
        "torch._C",
        "torch._weights_only_unpickler",
        "torch.nn.functional",
        "torch.nn.modules.linear",
        "torch.nn.modules.module",
        "torch.optim.adam",
        "torch.optim.adamw",
        "torch.optim.optimizer",
        "torch.serialization",
        "tokenizers",
        "tokenizers.tokenizers",
    }
    assert dependencies["manifest_sha256"] == recovery.stable_json_sha256(
        dependencies["portable_receipts"]
    )
    assert dependencies["identity_manifest_sha256"] == recovery.stable_json_sha256(
        dependencies["files"]
    )
    assert Path(dependencies["files"]["torch._C"]["path"]).suffix in {
        ".so",
        ".pyd",
        ".dylib",
    }
    assert Path(dependencies["files"]["tokenizers.tokenizers"]["path"]).suffix in {
        ".so",
        ".pyd",
        ".dylib",
    }
    assert dependencies["files"]["torch.optim.adamw"]["sha256"]
    assert contract["numerical_runtime"]["creation_umask"] == "0077"
    assert contract["callable_contract"]["exports"]["torch.optim.AdamW"]
    assert contract["callable_contract"]["exports"]["torch.optim.adamw.adamw"]
    assert contract["callable_contract"]["exports"]["torch.nn.Module.load_state_dict"]
    assert contract["callable_contract"]["semantic_dependencies"]
    assert contract["git_repository_contract"]["configuration_exclusion"]

    hostile = dict(environment)
    hostile["OMP_NUM_THREADS"] = "8"
    rejected = subprocess.run(
        _isolated_python_command(repo, script),
        cwd=repo,
        env=hostile,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "environment mismatch: OMP_NUM_THREADS" in rejected.stderr

    hostile = dict(environment)
    hostile["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = ""
    rejected = subprocess.run(
        _isolated_python_command(repo, script),
        cwd=repo,
        env=hostile,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in rejected.stderr


@pytest.mark.parametrize(
    "name",
    (
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "TORCH_BLAS_PREFER_CUBLASLT",
        "NVIDIA_TF32_OVERRIDE",
        "CUBLAS_FORCE_TF32",
        "CUDNN_DETERMINISTIC",
        "MKL_CBWR",
        "OPENBLAS_CORETYPE",
        "LD_LIBRARY_PATH",
        "GIT_DIR",
        "GIT_CONFIG_COUNT",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCHINDUCTOR_MAX_AUTOTUNE",
        "NCCL_ALGO",
    ),
)
def test_numerical_and_git_control_environment_is_presence_strict(name):
    assert name in recovery.forbidden_executor_environment_names({name: ""})


def test_fixed_numerical_environment_is_sanitized_not_forbidden():
    assert (
        recovery.forbidden_executor_environment_names(recovery.EXECUTOR_ENVIRONMENT)
        == []
    )


def test_dependency_manifest_rejects_native_module_file_substitution(
    monkeypatch, tmp_path
):
    substitute = tmp_path / "_C.so"
    substitute.write_bytes(b"not the loaded extension")
    monkeypatch.setattr(recovery.torch_c_module, "__file__", str(substitute))
    with pytest.raises(ValueError, match="module identity mismatch: torch._C"):
        recovery.capture_dependency_manifest()


def test_callable_binding_rejects_live_monkeypatch_with_unchanged_file_receipts(
    monkeypatch,
):
    before = recovery.capture_dependency_manifest()["manifest_sha256"]
    original = torch.optim.AdamW

    class ForgedAdamW:
        pass

    monkeypatch.setattr(torch.optim, "AdamW", ForgedAdamW)
    assert recovery.capture_dependency_manifest()["manifest_sha256"] == before
    assert recovery.REVIEWED_ADAMW is original
    with pytest.raises(RuntimeError, match="torch.optim.AdamW"):
        recovery.assert_reviewed_callable_exports()


@pytest.mark.parametrize(
    ("owner", "name", "expected"),
    (
        (torch, "load", "torch.load"),
        (torch, "save", "torch.save"),
        (
            upstream,
            "_validate_motor_bundle_against_replayed_features",
            "upstream._validate_motor_bundle_against_replayed_features",
        ),
        (torch, "equal", "torch.equal"),
        (torch, "no_grad", "torch.no_grad"),
        (upstream, "stable_json_sha256", "upstream.stable_json_sha256"),
        (torch.cuda, "get_device_name", "torch.cuda.get_device_name"),
    ),
)
def test_callable_binding_rejects_deserializer_serializer_and_validator_mutation(
    monkeypatch, owner, name, expected
):
    monkeypatch.setattr(owner, name, lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        recovery.assert_reviewed_callable_exports()


@pytest.mark.parametrize(
    ("dependency", "owner_expression", "attribute"),
    ISOLATED_TRANSITIVE_MUTATIONS,
)
def test_each_transitive_semantic_mutation_fails_before_construction_and_replay(
    dependency,
    owner_expression,
    attribute,
):
    repo = Path(__file__).resolve().parents[1]
    source = f"""
import causal_carry_motor_recovery as recovery
owner = eval({owner_expression!r}, {{"recovery": recovery}})
setattr(owner, {attribute!r}, object())
calls = (
    lambda: recovery._recovery_fit_motor(None, None, None, "cpu", 1, 1, 1.0, 0.0, 1),
    lambda: recovery.prove_fit_payload_trajectory(
        {{}}, {{}}, {{}}, {{}}, None, "cpu"
    ),
)
for call in calls:
    try:
        call()
    except RuntimeError as exc:
        if "monkeypatched" not in str(exc):
            raise
    else:
        raise AssertionError("semantic mutation reached construction or replay")
"""
    command = _isolated_python_command(repo, source)
    assert command[1:4] == ["-I", "-S", "-B"]
    observed = subprocess.run(
        command,
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert observed.returncode == 0, dependency + "\n" + observed.stderr


def test_every_recursive_semantic_binding_fails_before_construction_and_replay():
    repo = Path(__file__).resolve().parents[1]
    source = """
import causal_carry_motor_recovery as recovery

bindings = recovery.REVIEWED_SEMANTIC_BINDINGS
if set(bindings) != set(recovery.REVIEWED_SEMANTIC_DESCRIPTORS):
    raise AssertionError("semantic binding inventory is incomplete")
for dependency in tuple(bindings):
    owner, key, captured = bindings[dependency]
    current = owner.cell_contents if key is None else owner[key]
    if current is not captured:
        raise AssertionError("semantic binding changed before mutation")
    prioritized = {dependency: bindings[dependency]}
    prioritized.update(
        (name, binding) for name, binding in bindings.items() if name != dependency
    )
    recovery.REVIEWED_SEMANTIC_BINDINGS = prioritized
    try:
        if key is None:
            owner.cell_contents = object()
        else:
            owner[key] = object()
        calls = (
            lambda: recovery._recovery_fit_motor(
                None, None, None, "cpu", 1, 1, 1.0, 0.0, 1
            ),
            lambda: recovery.prove_fit_payload_trajectory(
                {}, {}, {}, {}, None, "cpu"
            ),
        )
        for call in calls:
            try:
                call()
            except RuntimeError as exc:
                if "monkeypatched" not in str(exc):
                    raise
            else:
                raise AssertionError(
                    f"semantic mutation reached construction or replay: {dependency}"
                )
    finally:
        if key is None:
            owner.cell_contents = current
        else:
            owner[key] = current
        recovery.REVIEWED_SEMANTIC_BINDINGS = bindings
print(len(bindings))
"""
    command = _isolated_python_command(repo, source)
    assert command[1:4] == ["-I", "-S", "-B"]
    observed = subprocess.run(
        command,
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert observed.returncode == 0, observed.stderr
    assert int(observed.stdout.strip()) == len(recovery.REVIEWED_SEMANTIC_BINDINGS)


def test_semantic_inventory_includes_constants_registries_tls_and_closures():
    names = tuple(recovery.REVIEWED_SEMANTIC_BINDINGS)
    for suffix in (
        ".__globals__.RANK",
        ".__globals__.STATE_RE",
        ".__globals__._global_forward_hooks",
        ".__globals__._package_registry",
        ".__globals__._serialization_tls",
        ".__closure__.recursive",
    ):
        assert any(name.endswith(suffix) for name in names), suffix
    assert not any(
        name.endswith(".__globals__._initialized")
        or name.endswith(".__globals__._cached_device_count")
        for name in names
    )


def test_mutable_semantic_state_fails_before_construction_and_replay():
    repo = Path(__file__).resolve().parents[1]
    source = """
import causal_carry_motor_recovery as recovery

cases = (
    (
        "module forward hook registry",
        lambda: recovery.torch_module_module._global_forward_hooks.__setitem__(
            7, recovery.REVIEWED_MODULE_CALL
        ),
        lambda: recovery.torch_module_module._global_forward_hooks.pop(7),
    ),
    (
        "serialization package registry",
        lambda: recovery.torch_serialization_module._package_registry.append(
            recovery.torch_serialization_module._package_registry[0]
        ),
        lambda: recovery.torch_serialization_module._package_registry.pop(),
    ),
    (
        "serialization TLS",
        lambda: setattr(
            recovery.torch_serialization_module._serialization_tls, "skip_data", True
        ),
        lambda: setattr(
            recovery.torch_serialization_module._serialization_tls, "skip_data", False
        ),
    ),
)
for dependency, mutate, restore in cases:
    mutate()
    try:
        calls = (
            lambda: recovery._recovery_fit_motor(
                None, None, None, "cpu", 1, 1, 1.0, 0.0, 1
            ),
            lambda: recovery.prove_fit_payload_trajectory(
                {}, {}, {}, {}, None, "cpu"
            ),
        )
        for call in calls:
            try:
                call()
            except RuntimeError as exc:
                if "semantic dependency descriptor changed" not in str(exc):
                    raise
            else:
                raise AssertionError(
                    f"mutable semantic state reached construction or replay: {dependency}"
                )
    finally:
        restore()
    recovery.assert_reviewed_callable_exports()
"""
    command = _isolated_python_command(repo, source)
    assert command[1:4] == ["-I", "-S", "-B"]
    observed = subprocess.run(
        command,
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert observed.returncode == 0, observed.stderr


def test_isolated_no_site_startup_blocks_sitecustomize_and_pth(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    hostile = tmp_path / "hostile-site"
    hostile.mkdir()
    marker = tmp_path / "startup-executed"
    (hostile / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('sitecustomize')\n"
    )
    (hostile / "payload.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('pth')\n"
    )
    (hostile / "hostile.pth").write_text("import payload\n")
    source = """
import json, sys
import causal_carry_motor_recovery as recovery
print(json.dumps({
    "isolated": sys.flags.isolated,
    "no_site": sys.flags.no_site,
    "site_loaded": "site" in sys.modules,
    "sitecustomize_loaded": "sitecustomize" in sys.modules,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(hostile)
    observed = subprocess.run(
        _isolated_python_command(repo, source),
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(observed.stdout)
    assert report == {
        "isolated": 1,
        "no_site": 1,
        "site_loaded": False,
        "sitecustomize_loaded": False,
    }
    assert not marker.exists()


def test_config_excluded_git_never_executes_include_clean_process_or_attributes(
    monkeypatch, tmp_path
):
    repo, _upstream, commit, manifest = _source_contract_repo(monkeypatch, tmp_path)
    markers = {
        name: tmp_path / f"{name}-executed" for name in ("include", "clean", "process")
    }
    scripts = {}
    for name, marker in markers.items():
        script = tmp_path / f"malicious-{name}.sh"
        script.write_text(
            f"#!/bin/sh\nprintf executed > {shlex.quote(str(marker))}\nexit 1\n"
        )
        os.chmod(script, 0o700)
        scripts[name] = script
    included = tmp_path / "included-filter.conf"
    included.write_text(
        f'[filter "included"]\n\tclean = {scripts["include"]}\n\trequired = true\n'
    )
    subprocess.run(
        [
            str(recovery.PINNED_GIT),
            "-C",
            str(repo),
            "config",
            "include.path",
            str(included),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(recovery.PINNED_GIT),
            "-C",
            str(repo),
            "config",
            "filter.localclean.clean",
            str(scripts["clean"]),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(recovery.PINNED_GIT),
            "-C",
            str(repo),
            "config",
            "filter.localprocess.process",
            str(scripts["process"]),
        ],
        check=True,
    )
    info_attributes = repo / ".git" / "info" / "attributes"
    info_attributes.write_text(
        "R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md filter=included\n"
        "train/causal_carry_motor_recovery.py filter=localclean\n"
        "train/test_causal_carry_motor_recovery.py filter=localprocess\n"
    )
    contract = recovery.build_recovery_executor_source_contract(
        commit, manifest, repo_root=repo
    )
    git_contract = contract["git_repository_contract"]
    assert git_contract["configuration_exclusion"]["local"] == (
        "synthetic bare GIT_DIR with no config file"
    )
    controls = git_contract["control_files_observed_but_never_interpreted"]
    assert controls["common_config"]
    assert controls["common_info_attributes"]
    assert not any(marker.exists() for marker in markers.values())


def test_recovery_plan_rejects_budget_change_old_root_and_extra_transform(
    monkeypatch, tmp_path
):
    _root, expected = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery.validate_recovery_plan_document(expected, expected, NEW_COMMIT)

    changed = copy.deepcopy(expected)
    changed["fit_contract"]["fit_budget"]["updates"] += 1
    with pytest.raises(ValueError, match="differs from independently reconstructed"):
        recovery.validate_recovery_plan_document(changed, expected, NEW_COMMIT)

    old_output = copy.deepcopy(expected)
    old_output["output_contract"]["fit_artifact"] = str(
        Path(expected["output_contract"]["upstream_root_must_remain_untouched"])
        / "fit"
        / "motor.pt"
    )
    with pytest.raises(ValueError, match="aliases the old canonical root"):
        recovery.validate_recovery_plan_document(old_output, old_output, NEW_COMMIT)

    expanded = copy.deepcopy(expected)
    expanded["allowed_transformation"] = {
        **expanded["allowed_transformation"],
        "permitted_additional_transformations": 1,
    }
    with pytest.raises(ValueError, match="extra transformations"):
        recovery.validate_recovery_plan_document(expanded, expanded, NEW_COMMIT)

    scalar_alias = copy.deepcopy(expected)
    scalar_alias["allowed_transformation"]["permitted_semantic_changes"] = False
    with pytest.raises(ValueError, match="extra transformations"):
        recovery.validate_recovery_plan_document(scalar_alias, scalar_alias, NEW_COMMIT)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("seed", upstream.FIT_SEED + 1),
        ("rank", upstream.RANK + 1),
        ("quota", upstream.FIT_QUOTA + 1),
        ("updates", upstream.CANONICAL_UPDATES + 1),
        ("batch_size", upstream.CANONICAL_BATCH + 1),
        ("lr", upstream.CANONICAL_LR * 2),
        ("weight_decay", upstream.CANONICAL_WEIGHT_DECAY * 2),
    ),
)
def test_recovery_plan_rejects_every_frozen_fit_scalar(
    monkeypatch, tmp_path, field, replacement
):
    _root, expected = _minimal_recovery_plan(monkeypatch, tmp_path)
    hostile = copy.deepcopy(expected)
    hostile["fit_contract"]["fit_budget"][field] = replacement
    with pytest.raises(ValueError, match="differs from independently reconstructed"):
        recovery.validate_recovery_plan_document(hostile, expected, NEW_COMMIT)


def test_downstream_scientific_boundary_is_frozen_before_fit(monkeypatch, tmp_path):
    contract = recovery.DOWNSTREAM_EVALUATION_CONTRACT
    assert contract["fit_metrics_are_capability_evidence"] is False
    assert contract["development"] == {
        "episodes": 300,
        "boundary_cycle_cases": 50,
        "researcher_direct_interactions": 12,
        "selection": "frozen a0c258e source-order development selection",
    }
    assert contract["confirmation"]["episodes"] == 256
    assert contract["confirmation"]["reveals"] == 1
    rescue = contract["carry_commit_rescue_set"]
    assert rescue["source_replay_sha256"] == (
        "756911f568c12093f3a303a42525a2519c38187c8eac71f5da3ca06ac1ce3b20"
    )
    assert [
        (case["id"], case["branch"], case["expected_answer"])
        for case in rescue["cases"]
    ] == [
        ("width_ood_w8-00175", "counterfactual", 177453123),
        ("width_ood_w8-00207", "normal", 176477219),
        ("width_ood_w8-00209", "normal", 169264069),
        ("width_ood_w8-00219", "normal", 187969887),
        ("width_ood_w8-00219", "counterfactual", 187969888),
        ("width_ood_w8-00242", "normal", 164377525),
    ]
    assert rescue["required_treatment_rescues"] == 6
    assert rescue["required_treatment_gain_over_constant_bias_cases"] == 2
    assert rescue["required_treatment_gain_over_nuisance_only_cases"] == 2
    assert rescue["allowed_shuffled_rescues"] == 0
    assert rescue["allowed_new_prefix_divergences"] == 0
    sweep = contract["terminal_width_sweep_oracle"]
    assert sweep["artifact_sha256"] == (
        "db6056e66310ed7d56509403d40f7549d016294a014c0c4527173b4005210520"
    )
    assert sweep["superseded_diagnostics"] == [
        {
            "artifact": (
                "artifacts/eval_history/"
                "drs_terminal_width_sweep_w2_w10_20260718_mps.json"
            ),
            "artifact_sha256": (
                "c9670853040349cce4eb4f89c5d5d8381d7b25494ff4428fd873fc2b7be6098d"
            ),
            "reason": "strict-parser nulls overcounted field errors",
            "decision_authority": False,
        },
        {
            "artifact": (
                "artifacts/eval_history/"
                "manual_drs_carry_serializer_probe_v2_20260718_mps.json"
            ),
            "artifact_sha256": (
                "b1cafe345bad726517e4c426596c691bf3ae1133d93619af581927ca7a336806"
            ),
            "reason": "narrow historical probe superseded by matched widths",
            "decision_authority": False,
        },
    ]
    assert sweep["construction"] == {
        "widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "lower_digit_history_matched_within_width": True,
        "only_terminal_operand_digits_change_within_width": True,
        "positive_terminal_digits_lsf": ["9", "8"],
        "negative_terminal_digits_lsf": ["2", "3"],
    }
    assert [
        (
            case["id"],
            case["arm"],
            case["width"],
            case["carry_class"],
            case["expected_answer"],
        )
        for case in sweep["cases"]
    ] == [
        ("w2_positive", "positive", 2, "11", 183),
        ("w2_negative", "negative", 2, "10", 63),
        ("w3_positive", "positive", 3, "11", 1823),
        ("w3_negative", "negative", 3, "10", 623),
        ("w4_positive", "positive", 4, "11", 18123),
        ("w4_negative", "negative", 4, "10", 6123),
        ("w5_positive", "positive", 5, "01", 173123),
        ("w5_negative", "negative", 5, "00", 53123),
        ("w6_positive", "positive", 6, "11", 1853123),
        ("w6_negative", "negative", 6, "10", 653123),
        ("w7_positive", "positive", 7, "01", 17453123),
        ("w7_negative", "negative", 7, "00", 5453123),
        ("w8_positive", "positive", 8, "01", 177453123),
        ("w8_negative", "negative", 8, "00", 57453123),
        ("w9_positive", "positive", 9, "11", 1827453123),
        ("w9_negative", "negative", 9, "10", 627453123),
        ("w10_positive", "positive", 10, "11", 18027453123),
        ("w10_negative", "negative", 10, "10", 6027453123),
    ]
    assert sweep["frozen_parent_observation"] == {
        "positive_transition_exact": 0,
        "positive_transition_denominator": 9,
        "positive_serializer_exact": 0,
        "positive_serializer_denominator": 9,
        "positive_raw_carry_exact_widths": [2, 3],
        "positive_raw_carry_failed_widths": [4, 5, 6, 7, 8, 9, 10],
        "positive_failure_classes": {
            "z_only": [2, 3],
            "c_only": [4, 5, 7],
            "c_and_r": [6, 8, 9],
            "broad_p_c_r_z": [10],
        },
        "common_positive_answer_error": "leading-1 omission",
        "negative_raw_carry_exact_widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "negative_serializer_exact_widths": [2, 3, 4, 5, 6],
        "negative_serializer_failed_widths": [7, 8, 9, 10],
        "negative_transition_exact_widths": [2, 3, 4, 5, 7],
        "negative_transition_failed_widths": [6, 8, 9, 10],
    }
    assert sweep["residual_swap_diagnostic"] == {
        "artifact": (
            "artifacts/eval_history/"
            "drs_terminal_carry_residual_swap_w2_w10_20260718_mps.json"
        ),
        "artifact_sha256": (
            "4183b8c381e559b23c41b88c8c8cc3b3d0e0b41c03b3dea4786df98a7676590f"
        ),
        "layer": 29,
        "teacher_forced_class_separation_widths": [2, 3, 4, 5, 7, 8, 9, 10],
        "teacher_forced_inverted_widths": [6],
        "ordinary_positive_c1_logit_over_c0_widths": [2, 3],
        "ordinary_positive_c1_logit_not_over_c0_widths": [4, 5, 6, 7, 8, 9, 10],
        "evidence_boundary": "teacher-forced calibration hypothesis only",
        "autonomous_motor_or_reasoning_claim": False,
    }
    assert sweep["carry_motor_decision"] == {
        "positive_widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "required_positive_transition_exact": 9,
        "positive_transition_denominator": 9,
        "required_positive_failure_class_exact": {
            "z_only": {"widths": [2, 3], "fields": ["z"], "required": 2},
            "c_only": {"widths": [4, 5, 7], "fields": ["c"], "required": 3},
            "c_and_r": {
                "widths": [6, 8, 9],
                "fields": ["c", "r"],
                "required": 3,
            },
            "broad_p_c_r_z": {
                "widths": [10],
                "fields": ["p", "c", "r", "z"],
                "required": 1,
            },
        },
        "negative_carry_preservation_widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "required_negative_carry_preservation_exact": 9,
        "negative_carry_preservation_denominator": 9,
        "negative_serializer_preservation_widths": [2, 3, 4, 5, 6],
        "required_negative_serializer_preservation_exact": 5,
        "negative_serializer_preservation_denominator": 5,
        "allowed_shuffled_positive_transition_rescues": 0,
        "required_treatment_gain_over_constant_bias_transitions": 2,
        "required_treatment_gain_over_nuisance_only_transitions": 2,
        "requires_no_new_matched_negative_divergence": True,
        "serializer_transfer_metrics_may_satisfy_carry_gate": False,
    }
    assert sweep["heldout_motor_calibration"] == {
        "widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "oracle_exclusion_proof": (
            "derive zero source-input and prompt-receipt overlap between the "
            "canonical fit split and every frozen matched oracle"
        ),
        "report_fields_separately": ["p", "c", "r", "z"],
        "report_each_width_separately": True,
        "pooled_field_score_may_satisfy_gate": False,
        "teacher_forced_swap_may_satisfy_gate": False,
        "fit_metric_may_satisfy_gate": False,
        "width_6_inversion_may_be_pooled_away": False,
        "recompute_constant_and_nuisance_on_all_18_terminal_cases": True,
        "one_off_terminal_pairwise_values_may_be_embedded_as_gate": False,
        "full_vocabulary_per_width_evaluation_required": True,
        "public_width_ood_is_unopened": False,
        "confirmation_width_ood_must_remain_unopened_until_reveal": True,
        "requires_development_and_confirmation": True,
    }
    assert sweep["serializer_decision"] == {
        "positive_readout_widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "negative_preservation_widths": [2, 3, 4, 5, 6],
        "negative_transfer_readout_widths": [7, 8, 9, 10],
        "report_each_width_separately": True,
        "required_negative_preservation_exact": 5,
        "negative_preservation_denominator": 5,
        "length_transfer_required_positive_exact": 9,
        "length_transfer_positive_denominator": 9,
        "length_transfer_required_negative_exact": 9,
        "length_transfer_negative_denominator": 9,
        "carry_transition_metrics_may_satisfy_serializer_gate": False,
    }
    assert contract["decision_labels"]["serializer_only"] == (
        "serializer-length-transfer-only"
    )
    assert contract["preservation"]["matched_arms"] == [
        "base",
        "treatment",
        "constant_bias",
        "nuisance_only",
        "shuffled",
    ]
    assert contract["preservation"]["allowed_false_fires"] == 0
    assert (
        contract["preservation"]["required_gate_off_token_and_logit_identity"]
        == "exact"
    )
    constant = contract["constant_bias_control"]
    assert constant["control_id"] == recovery.CONSTANT_BIAS_CONTROL_ID
    assert constant["trainable_parameters"] == 1
    assert constant["optimization"]["schema"] == recovery.NULL_OPTIMIZATION_SCHEMA
    assert constant["optimization"]["nonconverged_policy"] == (
        "fail_closed_without_final_iterate_fallback"
    )
    nuisance = contract["nuisance_only_control"]
    assert nuisance["control_id"] == recovery.NUISANCE_ONLY_CONTROL_ID
    assert nuisance["trainable_parameters"] == 20
    assert nuisance["capacity_ledger"] == recovery.nuisance_capacity_ledger()
    assert nuisance["capacity_ledger"]["fit_metadata_cells"] == 20
    assert nuisance["capacity_ledger"]["fit_design_rank"] == 20
    assert nuisance["capacity_ledger"]["width_extrapolation_parameters"] == 0
    assert nuisance["optimization"]["schema"] == recovery.NULL_OPTIMIZATION_SCHEMA
    assert nuisance["optimization"]["nonconverged_policy"] == (
        "fail_closed_without_final_iterate_fallback"
    )
    assert nuisance["width8_selection_access"] is False
    assert nuisance["confirmation_fit_access"] is False
    assert contract["selection_boundary"] == {
        "model_family": "singleton saturated_fit_cell_v1",
        "model_family_frozen_by": "reviewed recovery source",
        "checkpoint_objective": "fit-board full-vocabulary cross entropy",
        "fit_width_development_role": "audit_only_no_reselection",
        "width8_role": "post-freeze evaluation_only",
        "confirmation_role": "post-freeze evaluation_only",
        "fallback_after_nonconvergence": False,
    }
    assert contract["required_case_coverage"] == {
        "matched_positive_cases": 9,
        "matched_negative_cases": 9,
        "carry_commit_rescue_cases": 6,
        "development_episodes_per_arm": 300,
        "development_boundary_cycle_cases_per_arm": 50,
        "development_direct_interactions_per_arm": 12,
        "confirmation_episodes_per_arm": 256,
        "missing_arm_or_case_policy": "fail_closed",
    }
    assert contract["required_scoring"] == {
        "matched_oracles": "full-vocabulary greedy decode with exact grammar gate",
        "development": "full-vocabulary autonomous greedy decode",
        "confirmation": "full-vocabulary autonomous greedy decode",
        "report_per_regime_and_width": True,
        "confirmation_transition_strata": ["operation", "width", "position"],
        "confirmation_episode_strata": ["regime", "operation", "width"],
        "all_manifest_strata_required": True,
        "all_nonempty_fit_development_confirmation_strata_required": True,
        "treatment_must_beat_both_calibration_nulls_in_every_stratum": True,
        "pairwise_c0_c1_probe_may_satisfy_gate": False,
        "teacher_forced_metric_may_satisfy_gate": False,
    }
    assert (
        contract["raw_margin_diagnostic"][
            "all_positive_and_negative_feasibility_condition"
        ]
        == "-min(m_positive) < delta < -max(m_negative)"
    )
    v4_boundary = contract["recovery_v4_calibration_boundary"]
    assert v4_boundary["artifact_sha256"] == (
        "94bf0b4b61b239601a7677f7badca03ac9b507c3aad6616b80d37f11072c7f68"
    )
    assert v4_boundary["operation_width_fit_optimal_value_ood_sensitivity"] == {
        "before_correct": 11,
        "after_correct": 16,
        "denominator": 16,
    }
    assert v4_boundary["binary_margin_cross_entropy"] == {
        "correct": 14,
        "denominator": 16,
    }
    assert v4_boundary["production_full_vocabulary_objective"] is False
    assert v4_boundary["candidate_fit_or_selection_authority"] is False
    assert v4_boundary["downstream_gate_authority"] is False
    pairwise = contract["pre_fit_pairwise_constant_bias_audit"]
    assert pairwise["artifact_sha256"] == (
        "7f2eef8843eb686c2b63683ab7f11a248b5e1b8c8a4358c936a6c2d49326b7b3"
    )
    assert pairwise["source_probe_sha256"] == (
        "c3c2d0b037852cb57d54e1f147d445d27093a8548b965c41466e81bcc1a27778"
    )
    assert pairwise["raw_pairwise"] == {
        "correct": 32,
        "total": 40,
        "target_0_correct": 13,
        "target_0_total": 20,
        "target_1_correct": 19,
        "target_1_total": 20,
    }
    assert pairwise["favorable_constant"] == {
        "representative_delta": -0.7841806411743164,
        "correct": 35,
        "total": 40,
        "target_0_correct": 18,
        "target_0_total": 20,
        "target_1_correct": 17,
        "target_1_total": 20,
    }
    assert pairwise["perfect_constant_feasibility"] == {
        "feasible": False,
        "lower_open": 0.6561751365661621,
        "upper_open": -1.1492173671722412,
    }
    assert pairwise["bind_probe_code_bytes"] is False
    assert pairwise["sufficient_autonomous_gate"] is False
    nuisance_audit = contract["unreviewed_width_calibration_diagnostic"]
    assert nuisance_audit["decision_authority"] is False
    assert nuisance_audit["oracle_per_width"]["admissible_control"] is False
    assert nuisance_audit["oracle_per_width"]["correct"] == 38
    assert nuisance_audit["fresh_fit_only_nuisance_audit"]["op_only"] == {
        "add_delta": -0.7778145075,
        "sub_delta": -0.4535870552,
        "add_fit_correct": 10,
        "add_fit_total": 10,
        "sub_fit_correct": 5,
        "sub_fit_total": 6,
        "width_ood_w8_correct": 7,
        "width_ood_w8_total": 8,
        "correct": 35,
        "total": 40,
    }
    assert nuisance_audit["public_width_ood_is_now_unopened"] is False
    assert nuisance_audit["may_satisfy_confirmation_gate"] is False
    mechanism = contract["mechanism_go"]
    assert mechanism["next_carry_gain_over_constant_bias_points_min"] == 15
    assert mechanism["next_carry_gain_over_nuisance_only_points_min"] == 15
    assert mechanism["one_step_state_gain_over_constant_bias_points_min"] == 15
    assert mechanism["one_step_state_gain_over_nuisance_only_points_min"] == 15
    assert mechanism["autonomous_episode_gain_over_constant_bias_points_min"] == 20
    assert mechanism["autonomous_episode_gain_over_nuisance_only_points_min"] == 20
    assert mechanism["unseen_width_gain_over_constant_bias_points_min"] == 15
    assert mechanism["unseen_width_gain_over_nuisance_only_points_min"] == 15
    assert mechanism["treatment_constant_bias_comparisons_are_noncompensatory"]
    assert mechanism["treatment_nuisance_only_comparisons_are_noncompensatory"]

    _root, expected = _minimal_recovery_plan(monkeypatch, tmp_path)
    hostile = copy.deepcopy(expected)
    hostile["downstream_evaluation_contract"]["terminal_width_sweep_oracle"][
        "carry_motor_decision"
    ]["required_positive_transition_exact"] = 8
    with pytest.raises(ValueError, match="independently reconstructed contract"):
        recovery.validate_recovery_plan_document(hostile, expected, NEW_COMMIT)


def test_recovery_parent_is_durably_installed_under_hostile_umask(
    monkeypatch, tmp_path
):
    parent = tmp_path / "recoveries"
    monkeypatch.setattr(recovery, "RECOVERY_PARENT", parent)
    os.umask(0o777)
    binding = recovery.ensure_recovery_parent(
        _minimal_source_contract(), allow_install=True
    )
    receipt = parent / recovery.RECOVERY_PARENT_RECEIPT_NAME
    assert recovery.require_secure_creation_umask() == "0077"
    assert stat.S_IMODE(os.lstat(parent).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(receipt).st_mode) == 0o444
    assert os.lstat(receipt).st_nlink == 1
    assert binding["receipt"]["document"]["required_umask"] == "0077"
    assert binding["receipt"]["document"]["directory_identity"] == binding["identity"]
    assert (
        recovery.ensure_recovery_parent(_minimal_source_contract(), allow_install=False)
        == binding
    )


def test_recovery_parent_never_adopts_or_deletes_a_partial_canonical_receipt(
    monkeypatch, tmp_path
):
    parent = tmp_path / "recoveries"
    parent.mkdir(mode=0o700)
    receipt = parent / recovery.RECOVERY_PARENT_RECEIPT_NAME
    receipt.write_bytes(b"partial receipt")
    os.chmod(receipt, 0o600)
    monkeypatch.setattr(recovery, "RECOVERY_PARENT", parent)
    with pytest.raises(ValueError, match="identity or mode mismatch"):
        recovery.ensure_recovery_parent(_minimal_source_contract(), allow_install=False)
    with pytest.raises(ValueError, match="identity or mode mismatch"):
        recovery.ensure_recovery_parent(_minimal_source_contract(), allow_install=True)
    assert receipt.read_bytes() == b"partial receipt"
    assert stat.S_IMODE(os.lstat(receipt).st_mode) == 0o600


def _leave_parent_publisher_stage(monkeypatch, tmp_path):
    parent = tmp_path / "recoveries"
    parent.mkdir(mode=0o700)
    monkeypatch.setattr(recovery, "RECOVERY_PARENT", parent)
    directory_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    ownership = recovery.PublisherOwnership(
        directory_fd,
        parent,
        lock_name=recovery.RECOVERY_PARENT_OWNER_NAME,
        purpose="recovery-parent-receipt",
        target_name=recovery.RECOVERY_PARENT_RECEIPT_NAME,
        stage_name=recovery.RECOVERY_PARENT_STAGE_NAME,
    )
    os.close(directory_fd)
    stage_fd = os.open(
        recovery.RECOVERY_PARENT_STAGE_NAME,
        os.O_RDWR | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=ownership.directory_fd,
    )
    ownership.bind_stage(stage_fd)
    os.write(stage_fd, b"interrupted receipt stage")
    os.fsync(stage_fd)
    os.close(stage_fd)
    return parent, ownership


def _mark_publisher_journal_stale(path):
    document = json.loads(path.read_text())
    document["owner"]["process_start"] = "stale-" + document["owner"]["process_start"]
    path.write_bytes(recovery.canonical_json_receipt_bytes(document))
    os.chmod(path, 0o600)


def test_recovery_parent_live_publisher_is_never_cleaned(monkeypatch, tmp_path):
    parent, ownership = _leave_parent_publisher_stage(monkeypatch, tmp_path)
    stage = parent / recovery.RECOVERY_PARENT_STAGE_NAME
    before = recovery._publisher_file_identity(os.lstat(stage))
    try:
        with pytest.raises(RuntimeError, match="another live publisher"):
            recovery.ensure_recovery_parent(
                _minimal_source_contract(), allow_install=True
            )
        assert recovery._publisher_file_identity(os.lstat(stage)) == before
        assert not (parent / recovery.RECOVERY_PARENT_RECEIPT_NAME).exists()
    finally:
        os.unlink(recovery.RECOVERY_PARENT_STAGE_NAME, dir_fd=ownership.directory_fd)
        ownership.clear_stage_binding()
        ownership.close()


def test_recovery_parent_recovers_only_a_stale_bound_stage(monkeypatch, tmp_path):
    parent, ownership = _leave_parent_publisher_stage(monkeypatch, tmp_path)
    ownership.close(preserve=True)
    _mark_publisher_journal_stale(parent / recovery.RECOVERY_PARENT_OWNER_NAME)
    binding = recovery.ensure_recovery_parent(
        _minimal_source_contract(), allow_install=True
    )
    receipt = parent / recovery.RECOVERY_PARENT_RECEIPT_NAME
    assert binding["receipt"]["sha256"] == recovery.sha256_file(receipt)
    assert stat.S_IMODE(os.lstat(receipt).st_mode) == 0o444
    assert sorted(item.name for item in parent.iterdir()) == [
        recovery.RECOVERY_PARENT_RECEIPT_NAME
    ]


def test_recovery_parent_refuses_foreign_stage_inode(monkeypatch, tmp_path):
    parent, ownership = _leave_parent_publisher_stage(monkeypatch, tmp_path)
    ownership.close(preserve=True)
    stage = parent / recovery.RECOVERY_PARENT_STAGE_NAME
    stage.unlink()
    stage.write_bytes(b"foreign receipt stage")
    os.chmod(stage, 0o600)
    foreign_identity = recovery._publisher_file_identity(os.lstat(stage))
    _mark_publisher_journal_stale(parent / recovery.RECOVERY_PARENT_OWNER_NAME)
    with pytest.raises(ValueError, match="foreign or substituted"):
        recovery.ensure_recovery_parent(_minimal_source_contract(), allow_install=True)
    assert stage.read_bytes() == b"foreign receipt stage"
    assert recovery._publisher_file_identity(os.lstat(stage)) == foreign_identity
    assert (parent / recovery.RECOVERY_PARENT_OWNER_NAME).exists()


def test_recovery_parent_rejects_foreign_unreceipted_content(monkeypatch, tmp_path):
    parent = tmp_path / "recoveries"
    parent.mkdir(mode=0o700)
    (parent / "foreign").write_bytes(b"must not be adopted")
    monkeypatch.setattr(recovery, "RECOVERY_PARENT", parent)
    with pytest.raises(ValueError, match="unreceipted recovery parent is not empty"):
        recovery.ensure_recovery_parent(_minimal_source_contract(), allow_install=True)


def test_recovery_plan_rejects_same_bytes_parent_identity_substitution(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    parent = root.parent
    displaced = tmp_path / "displaced_recoveries"
    parent.rename(displaced)
    replacement = recovery.ensure_recovery_parent(
        document["recovery_executor_source_contract"], allow_install=True
    )
    assert (
        replacement["receipt"]["document"]["installer_source_binding"]
        == (
            document["recovery_parent_binding"]["receipt"]["document"][
                "installer_source_binding"
            ]
        )
    )
    assert (
        replacement["identity"]["inode"]
        != document["recovery_parent_binding"]["identity"]["inode"]
    )
    with pytest.raises(RuntimeError, match="parent or durable receipt changed"):
        recovery._publish_recovery_plan(root, document)


def test_recovery_parent_receipt_rejects_hardlink_and_same_bytes_substitution(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    receipt = root.parent / recovery.RECOVERY_PARENT_RECEIPT_NAME
    alias = tmp_path / "receipt_alias.json"
    os.link(receipt, alias)
    with pytest.raises(ValueError, match="receipt file identity or mode mismatch"):
        recovery.verify_recovery_parent_binding(
            document["recovery_parent_binding"],
            document["recovery_executor_source_contract"],
        )
    alias.unlink()

    payload = receipt.read_bytes()
    displaced = tmp_path / "displaced_parent_receipt.json"
    receipt.rename(displaced)
    receipt.write_bytes(payload)
    os.chmod(receipt, 0o444)
    with pytest.raises(RuntimeError, match="parent or durable receipt changed"):
        recovery.verify_recovery_parent_binding(
            document["recovery_parent_binding"],
            document["recovery_executor_source_contract"],
        )


def test_recovery_layout_rejects_same_byte_root_and_subdirectory_replacement(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    displaced = tmp_path / "displaced-root"
    root.rename(displaced)
    root.mkdir(mode=0o700)
    for name in ("fit", "development_eval", "confirmation_eval"):
        (root / name).mkdir(mode=0o700)
    receipt = root / recovery.RECOVERY_LAYOUT_RECEIPT_NAME
    receipt.write_bytes(
        (displaced / recovery.RECOVERY_LAYOUT_RECEIPT_NAME).read_bytes()
    )
    os.chmod(receipt, 0o444)
    assert recovery.sha256_file(receipt) == recovery.sha256_file(
        displaced / recovery.RECOVERY_LAYOUT_RECEIPT_NAME
    )
    with pytest.raises(ValueError, match="layout receipt identity mismatch"):
        recovery._publish_recovery_plan(root, document)
    assert not (root / "recovery_plan.json").exists()


def test_recovery_plan_publication_is_immutable_and_closed_world(monkeypatch, tmp_path):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    os.umask(0o777)
    recovery._publish_recovery_plan(root, document)
    assert recovery.require_secure_creation_umask() == "0077"
    recovery.validate_recovery_layout(root, fit_state="empty")
    parent = root.parent
    receipt = parent / recovery.RECOVERY_PARENT_RECEIPT_NAME
    assert stat.S_IMODE(os.lstat(parent).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(receipt).st_mode) == 0o444
    assert os.lstat(receipt).st_nlink == 1
    assert (
        recovery.verify_recovery_parent_binding(
            document["recovery_parent_binding"],
            document["recovery_executor_source_contract"],
        )
        == document["recovery_parent_binding"]
    )
    assert (root / "recovery_plan.json").stat().st_mode & 0o777 == 0o444
    assert root.stat().st_mode & 0o777 == 0o555
    for name in ("fit", "development_eval", "confirmation_eval"):
        assert stat.S_IMODE(os.lstat(root / name).st_mode) == 0o700
    with pytest.raises((FileNotFoundError, ValueError)):
        recovery._publish_recovery_plan(root, document)


def test_descriptor_publication_is_no_replace_one_link_and_crash_recoverable(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    assert recovery.recovery_fit_state(root) == "empty"
    recovery.validate_recovery_layout(root, fit_state="empty")

    artifact = root / "fit" / "motor.pt"
    layout = document["output_contract"]["layout_binding"]
    payload = {"tensor": torch.tensor([1, 2, 3]), "audit": "v9-test"}
    os.umask(0o777)
    digest = recovery.publish_recovery_torch(artifact, payload, layout)
    assert recovery.require_secure_creation_umask() == "0077"
    assert digest == recovery.sha256_file(artifact)
    assert recovery.recovery_fit_state(root) == "recoverable"
    recovery.validate_recovery_layout(root, fit_state="recoverable")
    artifact_stat = os.lstat(artifact)
    assert stat.S_IMODE(artifact_stat.st_mode) == 0o444
    assert artifact_stat.st_nlink == 1
    assert [item.name for item in artifact.parent.iterdir()] == ["motor.pt"]
    assert not any("stage" in item.name for item in artifact.parent.iterdir())
    bound = recovery.BoundFile(
        str(artifact),
        artifact,
        digest,
        "recoverable publication",
        required_mode=0o444,
        required_parent_mode=0o700,
    )
    try:
        loaded = recovery.safe_torch_load(bound)
        assert torch.equal(loaded["tensor"], payload["tensor"])
    finally:
        bound.close()

    with pytest.raises(FileExistsError, match="not empty"):
        recovery.publish_recovery_torch(artifact, payload, layout)

    # A crash immediately after publish leaves this exact recoverable state.
    recovery.seal_recovery_fit(artifact, layout)
    assert recovery.recovery_fit_state(root) == "sealed"
    recovery.validate_recovery_layout(root, fit_state="sealed")


def _leave_owned_fit_stage(monkeypatch, artifact, layout):
    original_save = recovery.REVIEWED_TORCH_SAVE

    def crash_during_save(_value, sink):
        sink.write(b"partial archive")
        sink.flush()
        raise RuntimeError("injected publication crash")

    monkeypatch.setattr(recovery, "assert_reviewed_callable_exports", lambda: {})
    monkeypatch.setattr(recovery, "REVIEWED_TORCH_SAVE", crash_during_save)
    with pytest.raises(RuntimeError, match="injected publication crash"):
        recovery.publish_recovery_torch(artifact, {"audit": "recovery-test"}, layout)
    return original_save


def test_stale_owned_descriptor_stage_is_recovered_by_exact_inode(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    layout = document["output_contract"]["layout_binding"]
    os.umask(0o777)
    original_save = _leave_owned_fit_stage(monkeypatch, artifact, layout)
    assert recovery.require_secure_creation_umask() == "0077"
    assert recovery.recovery_fit_state(root) == "publisher-residue"
    stage = artifact.parent / recovery.RECOVERY_FIT_STAGE_NAME
    owner = artifact.parent / recovery.RECOVERY_FIT_OWNER_NAME
    stage_identity = recovery._publisher_file_identity(os.lstat(stage))
    assert stat.S_IMODE(os.lstat(stage).st_mode) == 0o600
    assert stat.S_IMODE(os.lstat(owner).st_mode) == 0o600
    assert not artifact.exists()

    monkeypatch.setattr(recovery, "REVIEWED_TORCH_SAVE", original_save)
    _mark_publisher_journal_stale(owner)
    assert recovery.prepare_recovery_fit_publication(artifact, layout) == "empty"
    assert not stage.exists()
    assert not owner.exists()
    recovery.publish_recovery_torch(artifact, {"audit": "recovery-test"}, layout)
    assert recovery.recovery_fit_state(root) == "recoverable"
    assert recovery._publisher_file_identity(os.lstat(artifact)) != stage_identity


def test_live_descriptor_publisher_excludes_a_concurrent_writer(monkeypatch, tmp_path):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    layout = document["output_contract"]["layout_binding"]
    entered = threading.Event()
    release = threading.Event()
    errors = []
    original_save = recovery.REVIEWED_TORCH_SAVE

    def blocking_save(value, sink):
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("live-writer test timed out")
        original_save(value, sink)

    monkeypatch.setattr(recovery, "assert_reviewed_callable_exports", lambda: {})
    monkeypatch.setattr(recovery, "REVIEWED_TORCH_SAVE", blocking_save)

    def writer():
        try:
            recovery.publish_recovery_torch(artifact, {"audit": "live-writer"}, layout)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    assert entered.wait(timeout=10)
    stage = artifact.parent / recovery.RECOVERY_FIT_STAGE_NAME
    before = recovery._publisher_file_identity(os.lstat(stage))
    with pytest.raises(RuntimeError, match="another live publisher"):
        recovery.publish_recovery_torch(artifact, {"audit": "competing-writer"}, layout)
    assert recovery._publisher_file_identity(os.lstat(stage)) == before
    release.set()
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert errors == []
    assert recovery.recovery_fit_state(root) == "recoverable"


def test_stale_descriptor_recovery_refuses_a_foreign_stage_inode(monkeypatch, tmp_path):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    layout = document["output_contract"]["layout_binding"]
    _leave_owned_fit_stage(monkeypatch, artifact, layout)
    stage = artifact.parent / recovery.RECOVERY_FIT_STAGE_NAME
    owner = artifact.parent / recovery.RECOVERY_FIT_OWNER_NAME
    stage.unlink()
    stage.write_bytes(b"foreign replacement")
    os.chmod(stage, 0o600)
    foreign_identity = recovery._publisher_file_identity(os.lstat(stage))
    _mark_publisher_journal_stale(owner)
    with pytest.raises(ValueError, match="foreign or substituted"):
        recovery.prepare_recovery_fit_publication(artifact, layout)
    assert stage.read_bytes() == b"foreign replacement"
    assert recovery._publisher_file_identity(os.lstat(stage)) == foreign_identity
    assert owner.exists()


def test_stale_hardlink_commit_recovers_to_one_immutable_final(monkeypatch, tmp_path):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    layout = document["output_contract"]["layout_binding"]
    stage = artifact.parent / recovery.RECOVERY_FIT_STAGE_NAME
    owner = artifact.parent / recovery.RECOVERY_FIT_OWNER_NAME
    original_unlink = recovery.os.unlink
    injected = False

    def fail_after_commit(path, *args, **kwargs):
        nonlocal injected
        if (
            not injected
            and os.fspath(path) == recovery.RECOVERY_FIT_STAGE_NAME
            and kwargs.get("dir_fd") is not None
        ):
            injected = True
            raise RuntimeError("injected post-commit crash")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(recovery.os, "unlink", fail_after_commit)
    with pytest.raises(RuntimeError, match="injected post-commit crash"):
        recovery.publish_recovery_torch(
            artifact, {"audit": "post-commit-crash"}, layout
        )
    assert injected
    assert artifact.exists() and stage.exists() and owner.exists()
    assert os.path.samestat(os.lstat(artifact), os.lstat(stage))
    assert os.lstat(artifact).st_nlink == 2
    assert stat.S_IMODE(os.lstat(artifact).st_mode) == 0o444
    digest = recovery.sha256_file(artifact)

    monkeypatch.setattr(recovery.os, "unlink", original_unlink)
    _mark_publisher_journal_stale(owner)
    assert recovery.prepare_recovery_fit_publication(artifact, layout) == "recoverable"
    assert recovery.sha256_file(artifact) == digest
    assert os.lstat(artifact).st_nlink == 1
    assert stat.S_IMODE(os.lstat(artifact).st_mode) == 0o444
    assert not stage.exists() and not owner.exists()


def test_signed_fit_directory_rejects_link_and_mode_identity_attacks(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    layout = document["output_contract"]["layout_binding"]

    hostile_layout = copy.deepcopy(layout)
    hostile_layout["directory_identities"]["fit"]["links"] += 1
    hostile_layout["phase_independent_receipt"]["document"]["directories"]["fit"][
        "identity"
    ]["links"] += 1
    with pytest.raises(RuntimeError, match="signed reservation"):
        recovery.publish_recovery_torch(
            artifact, {"audit": "link-identity-attack"}, hostile_layout
        )
    assert not artifact.exists()

    os.chmod(artifact.parent, 0o755)
    with pytest.raises(RuntimeError, match="signed reservation"):
        recovery.publish_recovery_torch(
            artifact, {"audit": "mode-identity-attack"}, layout
        )
    assert not artifact.exists()


def test_signed_fit_directory_rejects_path_retarget_before_create(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    layout = document["output_contract"]["layout_binding"]
    fit = artifact.parent
    displaced = tmp_path / "signed-fit"
    original_verify = recovery.BoundDir.verify
    calls = 0

    def retarget_on_second_verify(bound, *, required_mode, children=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            os.chmod(root, 0o755)
            fit.rename(displaced)
            fit.mkdir(mode=0o700)
        return original_verify(bound, required_mode=required_mode, children=children)

    monkeypatch.setattr(recovery.BoundDir, "verify", retarget_on_second_verify)
    with pytest.raises(RuntimeError, match="ancestor chain changed"):
        recovery.publish_recovery_torch(
            artifact, {"audit": "path-retarget-attack"}, layout
        )
    assert not (fit / "motor.pt").exists()
    assert not (displaced / "motor.pt").exists()


def test_signed_fit_directory_rejects_mode_race_before_create(monkeypatch, tmp_path):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    layout = document["output_contract"]["layout_binding"]
    original_verify = recovery.BoundDir.verify
    calls = 0

    def chmod_on_second_verify(bound, *, required_mode, children=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            os.chmod(artifact.parent, 0o755)
        return original_verify(bound, required_mode=required_mode, children=children)

    monkeypatch.setattr(recovery.BoundDir, "verify", chmod_on_second_verify)
    with pytest.raises(RuntimeError, match="ancestor chain changed"):
        recovery.publish_recovery_torch(artifact, {"audit": "mode-race"}, layout)
    assert not artifact.exists()


def test_signed_fit_directory_rejects_path_retarget_before_directory_chmod(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    layout = document["output_contract"]["layout_binding"]
    recovery.publish_recovery_torch(artifact, {"audit": "v9-test"}, layout)
    fit = artifact.parent
    displaced = tmp_path / "seal-fit"
    original_verify = recovery.BoundDir.verify
    calls = 0

    def retarget_on_second_verify(bound, *, required_mode, children=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            os.chmod(root, 0o755)
            fit.rename(displaced)
            fit.mkdir(mode=0o700)
        return original_verify(bound, required_mode=required_mode, children=children)

    monkeypatch.setattr(recovery.BoundDir, "verify", retarget_on_second_verify)
    with pytest.raises(RuntimeError, match="ancestor chain changed"):
        recovery.seal_recovery_fit(artifact, layout)
    assert not artifact.exists()
    assert stat.S_IMODE(os.lstat(displaced).st_mode) == 0o700
    assert (displaced / "motor.pt").exists()


def test_recovery_fit_state_rejects_wrong_mode_extra_child_or_link(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    artifact.write_bytes(b"mutable")
    os.chmod(artifact, 0o640)
    with pytest.raises(ValueError, match="artifact mode mismatch"):
        recovery.recovery_fit_state(root)

    os.chmod(artifact, 0o444)
    sibling = tmp_path / "linked.pt"
    os.link(artifact, sibling)
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        recovery.recovery_fit_state(root)
    sibling.unlink()
    hostile = artifact.parent / "extra.pt"
    hostile.write_bytes(b"substitution")
    with pytest.raises(ValueError, match="not closed-world"):
        recovery.recovery_fit_state(root)


def test_recovery_plan_publication_rejects_symlink_parent(monkeypatch, tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setattr(recovery, "RECOVERY_PARENT", alias_parent)
    with pytest.raises(ValueError, match="identity, owner, or mode"):
        recovery.ensure_recovery_parent(_minimal_source_contract(), allow_install=True)


def test_recovery_plan_publication_never_replaces_raced_target(monkeypatch, tmp_path):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    marker = root / "recovery_plan.json"
    original_verify = recovery.verify_recovery_parent_binding

    def race_after_parent_verification(expected, source_contract):
        observed = original_verify(expected, source_contract)
        marker.write_bytes(b"must-survive")
        return observed

    monkeypatch.setattr(
        recovery, "verify_recovery_parent_binding", race_after_parent_verification
    )
    with pytest.raises(ValueError, match="closed-world"):
        recovery._publish_recovery_plan(root, document)
    assert marker.read_bytes() == b"must-survive"


def test_hostile_review_binds_exact_executor_and_normalization(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, "REVIEW_PARENT", tmp_path / "reviews")
    contract = {
        "schema": recovery.RECOVERY_EXECUTOR_SOURCE_SCHEMA,
        "git_commit": NEW_COMMIT,
        "sources": {"executor.py": "1" * 64},
        "manifest_sha256": "2" * 64,
    }
    runtime = {
        "schema": recovery.RECOVERY_EXECUTOR_RUNTIME_SCHEMA,
        "source_root": "/reviewed/recovery",
    }
    layout = {
        "schema": recovery.RECOVERY_LAYOUT_BINDING_SCHEMA,
        "root_identity": {"device": 1, "inode": 2, "uid": 3, "gid": 4},
    }
    statement = recovery.expected_review_statement(
        NEW_COMMIT, contract, runtime, layout
    )
    assert statement["signer"] == {
        "algorithm": "Ed25519",
        "key_id": recovery.PRODUCTION_REVIEW_KEY_ID,
        "sequence": 1,
    }
    assert statement["output_contract"]["layout_binding"] == layout
    assert statement["fit_claim_boundary"] == recovery.RECOVERY_FIT_CLAIM_BOUNDARY
    review = {
        "audit": recovery.RECOVERY_REVIEW_AUDIT,
        "algorithm": "Ed25519",
        "key_id": recovery.PRODUCTION_REVIEW_KEY_ID,
        "signed_payload": statement,
        "signature_base64": base64.b64encode(b"\0" * 64).decode("ascii"),
    }
    path = recovery.recovery_review_path(NEW_COMMIT)
    path.parent.mkdir(parents=True)
    path.write_bytes(recovery.canonical_json_receipt_bytes(review))
    os.chmod(path, 0o444)
    os.chmod(path.parent, 0o555)
    digest = recovery.sha256_file(path)
    verified = []
    monkeypatch.setattr(
        recovery,
        "verify_ed25519_signature",
        lambda key, message, signature: (
            verified.append((key, message, signature)) or True
        ),
    )
    bound, loaded = recovery.load_hostile_review(
        NEW_COMMIT, contract, runtime, layout, str(path), digest
    )
    try:
        assert loaded["signed_payload"]["decision"] == "GO"
        assert len(verified) == 1
    finally:
        bound.close()

    os.chmod(path.parent, 0o755)
    os.chmod(path, 0o644)
    path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    os.chmod(path, 0o444)
    os.chmod(path.parent, 0o555)
    with pytest.raises(ValueError, match="outer receipt encoding is not canonical"):
        recovery.load_hostile_review(
            NEW_COMMIT,
            contract,
            runtime,
            layout,
            str(path),
            recovery.sha256_file(path),
        )

    os.chmod(path.parent, 0o755)
    os.chmod(path, 0o644)
    canonical_signature = review["signature_base64"]
    assert canonical_signature.endswith("AA==")
    review["signature_base64"] = canonical_signature[:-3] + "B=="
    assert base64.b64decode(review["signature_base64"], validate=True) == b"\0" * 64
    path.write_bytes(recovery.canonical_json_receipt_bytes(review))
    os.chmod(path, 0o444)
    os.chmod(path.parent, 0o555)
    with pytest.raises(ValueError, match="non-canonical padding"):
        recovery.load_hostile_review(
            NEW_COMMIT,
            contract,
            runtime,
            layout,
            str(path),
            recovery.sha256_file(path),
        )
    review["signature_base64"] = canonical_signature

    os.chmod(path.parent, 0o755)
    os.chmod(path, 0o644)
    review["signed_payload"]["decision"] = "NO-GO"
    path.write_text(json.dumps(review, sort_keys=True) + "\n")
    os.chmod(path, 0o444)
    os.chmod(path.parent, 0o555)
    with pytest.raises(ValueError, match="does not authorize"):
        recovery.load_hostile_review(
            NEW_COMMIT,
            contract,
            runtime,
            layout,
            str(path),
            recovery.sha256_file(path),
        )

    os.chmod(path.parent, 0o755)
    os.chmod(path, 0o644)
    review["signed_payload"] = copy.deepcopy(statement)
    review["signed_payload"]["signer"]["sequence"] = True
    path.write_text(json.dumps(review, sort_keys=True) + "\n")
    os.chmod(path, 0o444)
    os.chmod(path.parent, 0o555)
    with pytest.raises(ValueError, match="does not authorize"):
        recovery.load_hostile_review(
            NEW_COMMIT,
            contract,
            runtime,
            layout,
            str(path),
            recovery.sha256_file(path),
        )


def test_ed25519_test_vector_verifies_but_test_key_has_no_production_authority():
    public = bytes.fromhex(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    )
    signature = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    )
    assert recovery.verify_ed25519_signature(public, b"", signature)
    assert public.hex() != recovery.PRODUCTION_REVIEW_PUBLIC_KEY_HEX
    with pytest.raises(ValueError, match="signature is invalid"):
        recovery.verify_ed25519_signature(
            bytes.fromhex(recovery.PRODUCTION_REVIEW_PUBLIC_KEY_HEX),
            b"",
            signature,
        )


def test_confirmation_generator_substitution_is_rejected():
    source_hashes = {
        name: f"{index + 1:064x}"
        for index, name in enumerate(upstream.SCIENTIFIC_SOURCE_PATHS)
    }
    source_contract = {
        "git_commit": recovery.UPSTREAM_SOURCE_COMMIT,
        "manifest_sha256": recovery.UPSTREAM_SOURCE_MANIFEST_SHA256,
    }
    generator_sources = {
        name: source_hashes[name]
        for name in upstream.CANONICAL_CONFIRMATION_GENERATOR_SOURCES
    }
    plan = {
        "confirmation_commitment": {
            "document": {
                "source_contract": source_contract,
                "generator_source_contract": {
                    "schema": upstream.CANONICAL_CONFIRMATION_GENERATOR_SCHEMA,
                    "entrypoint": upstream.CANONICAL_CONFIRMATION_GENERATOR_ENTRYPOINT,
                    "sources": generator_sources,
                    "manifest_sha256": upstream.stable_json_sha256(generator_sources),
                },
            }
        }
    }
    recovery.validate_confirmation_generator_contract(
        plan, source_contract, source_hashes
    )
    hostile = copy.deepcopy(plan)
    hostile["confirmation_commitment"]["document"]["generator_source_contract"][
        "entrypoint"
    ] = "recovery:generate_confirmation_board"
    with pytest.raises(ValueError, match="substitution"):
        recovery.validate_confirmation_generator_contract(
            hostile, source_contract, source_hashes
        )


def _legacy_payload_fixture(monkeypatch):
    fit_budget = {
        "rank": 2,
        "updates": 4,
        "batch_size": 3,
        "lr": 0.003,
        "weight_decay": 0.0001,
        "seed": 7,
        "schedule_sha256": "a" * 64,
        "initial_state_sha256": "b" * 64,
    }
    rows = [
        {
            "operation": operation,
            "width": width,
            "position": position,
            "target": target,
            "prompt_ids": [index, target],
            "prompt_sha256": recovery.sha256_bytes(
                f"prompt-{index}-{target}".encode()
            ),
            "prefix_ids": [index, target, 99],
            "prefix_sha256": recovery.sha256_bytes(
                f"prefix-{index}-{target}".encode()
            ),
        }
        for index, (operation, width, position) in enumerate(
            recovery.NUISANCE_FIT_CELLS
        )
        for target in (0, 1)
    ]
    labels = torch.tensor(
        [target for _cell in recovery.NUISANCE_FIT_CELLS for target in (0, 1)]
    )
    row_count = len(rows)
    board = {"rows": row_count, "canonical": True}
    control = {"seed": 7, "derangement": True}
    feature_merge = {
        "rows": row_count,
        "deployment_logit_dtype": "torch.bfloat16",
        "teacher_metric_feature_payload_sha256": "f" * 64,
    }
    source_hashes = {"train/model.py": "c" * 64}
    source_contract = {
        "git_commit": recovery.UPSTREAM_SOURCE_COMMIT,
        "manifest_sha256": "d" * 64,
    }
    expected_bindings = {
        "base_checkpoint_sha256": upstream.EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": upstream.EXPECTED_TOKENIZER_SHA256,
        "episodes_sha256": "3" * 64,
        "cycle_sha256": "4" * 64,
        "confirmation_commitment_sha256": "5" * 64,
    }
    deployment_vocabulary = {
        "schema": recovery.DEPLOYMENT_VOCABULARY_SCHEMA,
        "base_checkpoint_sha256": upstream.EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": upstream.EXPECTED_TOKENIZER_SHA256,
        "output_vocab_width": 13,
        "token_id_order": list(range(13)),
        "token_id_order_sha256": recovery.stable_json_sha256(list(range(13))),
        "tokenizer_id_to_token_sha256": recovery.stable_json_sha256(
            [f"token-{index}" for index in range(13)]
        ),
        "zero_id": 11,
        "one_id": 12,
        "deployment_logit_dtype": "torch.bfloat16",
    }
    context = {
        "upstream_plan": {
            "fit_budget": fit_budget,
            "checkpoint_step": 280000,
            "d_model": 4,
            "zero_id": 11,
            "one_id": 12,
            "runtime_contract": {"extract_batch": 32},
        },
        "expected_bindings": expected_bindings,
        "upstream_source_hashes": source_hashes,
        "upstream_source_contract": source_contract,
        "shard_receipts": [{"shard_index": index} for index in range(8)],
        "feature_merge": feature_merge,
        "features": {
            "deployment_logit_dtype": "torch.bfloat16",
            "hidden": torch.zeros(row_count, 4),
            "base01": torch.stack(
                (
                    torch.linspace(-0.5, 0.5, row_count),
                    torch.linspace(0.5, -0.5, row_count),
                ),
                dim=1,
            ).to(torch.bfloat16),
            "other_lse": torch.zeros(row_count),
            "labels": labels,
            "other_token_ids": torch.arange(11, dtype=torch.long),
        },
        "board_context": {
            "normalized_board": board,
            "control": control,
            "rows": rows,
            "control_labels": labels.roll(1),
        },
        "expected_recovery_plan": {
            "fit_contract": {"deployment_vocabulary": deployment_vocabulary}
        },
    }
    state = collections.OrderedDict(
        (
            ("down.weight", torch.zeros(2, 4)),
            ("down.bias", torch.zeros(2)),
            ("up.weight", torch.zeros(2, 2)),
            ("up.bias", torch.zeros(2)),
        )
    )
    fit_report = {
        "updates": 4,
        "batch_size": 3,
        "lr": 0.003,
        "weight_decay": 0.0001,
        "schedule_sha256": "a" * 64,
        "first_loss": 1.25,
        "final_loss": 0.75,
        "min_loss": 0.5,
    }
    linear = {
        "train_rows": 2,
        "test_rows": 1,
        "test_correct": 1,
        "test_accuracy": 1.0,
        "schedule_sha256": "e" * 64,
        "claim_boundary": "diagnostic only",
    }
    evidence = {"accuracy": 1.0, "correct": 3, "nested": {"passed": True}}
    monkeypatch.setattr(recovery, "_expected_teacher_evidence", lambda *_args: evidence)
    payload = {
        "plan_sha256": recovery.UPSTREAM_PLAN_SHA256,
        **expected_bindings,
        "scientific_source_sha256": source_hashes,
        "source_contract": source_contract,
        "checkpoint_step": 280000,
        "d_model": 4,
        "rank": 2,
        "parameter_count": 16,
        "extract_batch": 32,
        "feature_shard_merge": feature_merge,
        "deployment_logit_dtype": "torch.bfloat16",
        "zero_id": 11,
        "one_id": 12,
        "initial_state_sha256": "b" * 64,
        "treatment": copy.deepcopy(state),
        "shuffled": copy.deepcopy(state),
        "treatment_state_sha256": "6" * 64,
        "shuffled_state_sha256": "7" * 64,
        "board": board,
        "control": control,
        "treatment_fit": copy.deepcopy(fit_report),
        "shuffled_fit": copy.deepcopy(fit_report),
        "linear_diagnostic": linear,
        "fit_feature_metrics": evidence,
        "claim_boundary": upstream.CANONICAL_FIT_CLAIM_BOUNDARY,
    }
    assert set(payload) == recovery.LEGACY_PAYLOAD_KEYS
    return payload, context


def _constant_bias_payload_fixture(fit_payload, context):
    initial_state, initial_sha256 = recovery.initial_constant_bias_state()
    assert initial_state["delta"].item() == 0.0
    budget = context["upstream_plan"]["fit_budget"]
    motor, fit_report = recovery._recovery_fit_constant_bias(
        context["features"],
        context["features"]["labels"],
        initial_state,
        "cpu",
        budget["updates"],
        budget["batch_size"],
        budget["lr"],
        budget["weight_decay"],
        budget["seed"],
        fit_rows=recovery.canonical_fit_selection_rows(context),
        fit_model_binding=recovery._context_fit_model_binding(context),
    )
    state = collections.OrderedDict(
        (name, value.detach().clone()) for name, value in motor.state_dict().items()
    )
    delta = float(state["delta"].item())
    return {
        "schema": recovery.CONSTANT_BIAS_PAYLOAD_SCHEMA,
        "control_id": recovery.CONSTANT_BIAS_CONTROL_ID,
        "parameterization": recovery.constant_bias_parameterization(),
        "training_rows": len(context["features"]["labels"]),
        "training_feature_payload_sha256": context["feature_merge"][
            "teacher_metric_feature_payload_sha256"
        ],
        "label_source": "features.labels_true",
        "initial_state_sha256": initial_sha256,
        "state": state,
        "state_sha256": recovery.REVIEWED_TENSOR_STATE_SHA256(state),
        "fit": fit_report,
        "raw_margin_diagnostic": recovery.raw_carry_margin_diagnostic(
            context["features"]["base01"],
            context["features"]["labels"],
            float(delta),
        ),
        "claim_boundary": recovery.CONSTANT_BIAS_CLAIM_BOUNDARY,
    }


def _nuisance_only_payload_fixture(fit_payload, context):
    initial_state, initial_sha256 = recovery.initial_nuisance_only_state()
    budget = context["upstream_plan"]["fit_budget"]
    motor, fit_report = recovery._recovery_fit_nuisance_only(
        context["features"],
        recovery.canonical_fit_selection_rows(context),
        context["features"]["labels"],
        initial_state,
        "cpu",
        budget["updates"],
        budget["batch_size"],
        budget["lr"],
        budget["weight_decay"],
        budget["seed"],
        fit_model_binding=recovery._context_fit_model_binding(context),
    )
    state = collections.OrderedDict(
        (name, value.detach().clone()) for name, value in motor.state_dict().items()
    )
    return {
        "schema": recovery.NUISANCE_ONLY_PAYLOAD_SCHEMA,
        "control_id": recovery.NUISANCE_ONLY_CONTROL_ID,
        "parameterization": recovery.nuisance_only_parameterization(),
        "training_rows": len(context["features"]["labels"]),
        "training_feature_payload_sha256": context["feature_merge"][
            "teacher_metric_feature_payload_sha256"
        ],
        "training_metadata_receipt": recovery.nuisance_metadata_receipt(
            context["board_context"]["rows"]
        ),
        "capacity_ledger": recovery.nuisance_capacity_ledger(),
        "label_source": "features.labels_true",
        "initial_state_sha256": initial_sha256,
        "state": state,
        "state_sha256": recovery.REVIEWED_TENSOR_STATE_SHA256(state),
        "fit": fit_report,
        "width_ood_policy": recovery.nuisance_width_ood_policy(),
        "claim_boundary": recovery.NUISANCE_ONLY_CLAIM_BOUNDARY,
    }


def test_forged_self_consistent_0444_motor_requires_complete_trajectory_replay(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    payload, context = _legacy_payload_fixture(monkeypatch)
    constant_bias = _constant_bias_payload_fixture(payload, context)
    nuisance_only = _nuisance_only_payload_fixture(payload, context)
    forged = {
        "fit_payload": payload,
        "constant_bias_payload": constant_bias,
        "nuisance_only_payload": nuisance_only,
        "trajectory_replay_proof": recovery.build_trajectory_replay_proof(
            payload, constant_bias, nuisance_only
        ),
    }
    artifact = root / "fit" / "motor.pt"
    torch.save(forged, artifact)
    os.chmod(artifact, 0o444)
    assert stat.S_IMODE(os.lstat(artifact.parent).st_mode) == 0o700
    bound = recovery.BoundFile(
        str(artifact),
        artifact,
        recovery.sha256_file(artifact),
        "forged recovery motor",
        required_mode=0o444,
        required_parent_mode=0o700,
    )
    try:
        loaded = recovery.safe_torch_load(bound)
    finally:
        bound.close()
    independently_replayed = copy.deepcopy(payload)
    independently_replayed["checkpoint_step"] += 1
    monkeypatch.setattr(recovery, "assert_reviewed_callable_exports", lambda: {})
    monkeypatch.setattr(recovery, "validate_recovery_fit_bundle", lambda *_: None)
    monkeypatch.setattr(
        recovery, "_build_fit_payload", lambda *_: independently_replayed
    )
    with pytest.raises(RuntimeError, match="complete trajectory replay"):
        recovery.validate_and_replay_recovery_fit(
            loaded,
            {},
            "1" * 64,
            context,
            {},
            "cpu",
        )
    assert stat.S_IMODE(os.lstat(artifact.parent).st_mode) == 0o700


def test_real_cpu_fit_and_replay_callables_match_for_all_fitted_arms():
    assert upstream.CANONICAL_UPDATES == 2000
    d_model = 6
    rows = 40
    updates = 4
    batch_size = 4
    seed = 20260718
    generator = torch.Generator().manual_seed(seed)
    features = {
        "hidden": torch.randn(rows, d_model, generator=generator),
        "base01": torch.randn(rows, 2, generator=generator).to(torch.bfloat16),
        "other_lse": torch.randn(rows, generator=generator),
    }
    treatment_labels = torch.tensor([0, 1] * 20)
    control_labels = treatment_labels.roll(1)
    torch.manual_seed(seed)
    initial_state, initial_sha256 = upstream.initial_motor_state(d_model)
    initial_snapshot = copy.deepcopy(initial_state)
    arms = {}
    for name, labels in (
        ("treatment", treatment_labels),
        ("control", control_labels),
    ):
        fitted, fit_report = upstream.fit_motor(
            features,
            labels,
            initial_state,
            "cpu",
            updates=updates,
            batch_size=batch_size,
            lr=upstream.CANONICAL_LR,
            weight_decay=upstream.CANONICAL_WEIGHT_DECAY,
            seed=seed,
        )
        replayed, replay_report = recovery._recovery_fit_motor(
            features,
            labels,
            initial_state,
            "cpu",
            updates,
            batch_size,
            upstream.CANONICAL_LR,
            upstream.CANONICAL_WEIGHT_DECAY,
            seed,
        )
        assert recovery.type_strict_equal(fit_report, replay_report)
        for key, value in fitted.state_dict().items():
            assert torch.equal(value, replayed.state_dict()[key])
        arms[name] = fit_report
    assert arms["treatment"]["schedule_sha256"] == arms["control"]["schedule_sha256"]
    assert upstream.tensor_state_sha256(initial_state) == initial_sha256
    for key, value in initial_snapshot.items():
        assert torch.equal(value, initial_state[key])

    constant_initial, constant_initial_sha256 = recovery.initial_constant_bias_state()
    constant_a, report_a = recovery._recovery_fit_constant_bias(
        features,
        treatment_labels,
        constant_initial,
        "cpu",
        updates,
        batch_size,
        upstream.CANONICAL_LR,
        upstream.CANONICAL_WEIGHT_DECAY,
        seed,
    )
    constant_b, report_b = recovery._recovery_fit_constant_bias(
        features,
        treatment_labels,
        constant_initial,
        "cpu",
        updates,
        batch_size,
        upstream.CANONICAL_LR,
        upstream.CANONICAL_WEIGHT_DECAY,
        seed,
    )
    assert recovery.type_strict_equal(report_a, report_b)
    assert report_a["converged"] is True
    assert report_a["optimizer_final_iterate_used"] is False
    assert torch.equal(
        constant_a.state_dict()["delta"], constant_b.state_dict()["delta"]
    )
    assert recovery.REVIEWED_TENSOR_STATE_SHA256(constant_initial) == (
        constant_initial_sha256
    )
    with torch.no_grad():
        constant_full_board = upstream.full_vocab_motor_loss(
            features["hidden"],
            features["base01"],
            features["other_lse"],
            treatment_labels,
            constant_a,
        )
    assert report_a["full_board_cross_entropy"] == float(constant_full_board)

    nuisance_rows = [
        {"operation": operation, "width": width, "position": position}
        for operation, width, position in recovery.NUISANCE_FIT_CELLS
        for _target in (0, 1)
    ]
    nuisance_initial, nuisance_initial_sha256 = recovery.initial_nuisance_only_state()
    nuisance_a, nuisance_report_a = recovery._recovery_fit_nuisance_only(
        features,
        nuisance_rows,
        treatment_labels,
        nuisance_initial,
        "cpu",
        updates,
        batch_size,
        upstream.CANONICAL_LR,
        upstream.CANONICAL_WEIGHT_DECAY,
        seed,
    )
    nuisance_b, nuisance_report_b = recovery._recovery_fit_nuisance_only(
        features,
        nuisance_rows,
        treatment_labels,
        nuisance_initial,
        "cpu",
        updates,
        batch_size,
        upstream.CANONICAL_LR,
        upstream.CANONICAL_WEIGHT_DECAY,
        seed,
    )
    assert recovery.type_strict_equal(nuisance_report_a, nuisance_report_b)
    assert nuisance_report_a["converged"] is True
    assert nuisance_report_a["optimizer_final_iterate_used"] is False
    assert recovery.type_strict_equal(nuisance_a.state_dict(), nuisance_b.state_dict())
    assert recovery.REVIEWED_TENSOR_STATE_SHA256(nuisance_initial) == (
        nuisance_initial_sha256
    )
    nuisance_metadata = recovery.nuisance_metadata_from_rows(nuisance_rows)
    with torch.no_grad():
        nuisance_full_board = upstream.full_vocab_motor_loss(
            nuisance_metadata,
            features["base01"],
            features["other_lse"],
            treatment_labels,
            nuisance_a,
        )
    assert nuisance_report_a["full_board_cross_entropy"] == float(nuisance_full_board)


def test_constant_bias_is_feature_blind_and_uses_exact_treatment_gate():
    motor = recovery.ConstantBiasMotor()
    with torch.no_grad():
        motor.delta.fill_(4.0)
    hidden_a = torch.randn(5, 7)
    hidden_b = torch.randn(5, 7) * 1000
    assert torch.equal(motor(hidden_a), motor(hidden_b))
    assert torch.equal(motor(hidden_a), torch.tensor([[-2.0, 2.0]]).expand(5, 2))
    assert tuple(motor.state_dict()) == ("delta",)
    parameterization = recovery.constant_bias_parameterization()
    for key in (
        "hidden_value_dependence",
        "width_dependence",
        "prompt_dependence",
        "token_history_dependence",
    ):
        assert parameterization[key] is False

    logits = torch.randn(5, 19)
    gate_off = upstream.apply_motor_logits(logits, hidden_a, motor, 3, 11, False)
    assert torch.equal(gate_off, logits)
    gate_on = upstream.apply_motor_logits(logits, hidden_a, motor, 3, 11, True)
    unchanged = [index for index in range(19) if index not in {3, 11}]
    assert torch.equal(gate_on[:, unchanged], logits[:, unchanged])
    assert torch.equal(gate_on[:, 3], logits[:, 3] - 2.0)
    assert torch.equal(gate_on[:, 11], logits[:, 11] + 2.0)

    state = upstream.initial_state("add", 96, 87, 2)
    prompt = upstream.microstep_prompt(state, style="core")
    router = upstream.CarryRouter(prompt, motor_present=True)
    assert router.observe("dws:op=add;w=2;p=1;c=") is True
    assert router.observe("dws:op=add;w=2;p=1;c=") is False
    hostile = upstream.CarryRouter(prompt + "extra", motor_present=True)
    assert hostile.observe("dws:op=add;w=2;p=1;c=") is False


def test_nuisance_only_reads_exact_metadata_and_never_hidden_or_prompt_content():
    rows_a = [
        {
            "operation": "add",
            "width": 4,
            "position": 3,
            "prompt_ids": [1, 2, 3],
            "style": "core",
            "current_carry": 0,
            "target": 1,
            "hidden": [999.0],
        }
    ]
    rows_b = [
        {
            "operation": "add",
            "width": 4,
            "position": 3,
            "prompt_ids": [88, 77],
            "style": "contrast",
            "current_carry": 1,
            "target": 0,
            "hidden": [-999.0],
        }
    ]
    metadata_a = recovery.nuisance_metadata_from_rows(rows_a)
    metadata_b = recovery.nuisance_metadata_from_rows(rows_b)
    assert torch.equal(metadata_a, metadata_b)
    assert (
        recovery.nuisance_metadata_receipt(rows_a)["metadata_sha256"]
        == (recovery.nuisance_metadata_receipt(rows_b)["metadata_sha256"])
    )

    motor = recovery.NuisanceOnlyMotor()
    with torch.no_grad():
        motor.cell_delta.copy_(
            torch.arange(recovery.NUISANCE_PARAMETER_COUNT, dtype=torch.float32)
        )
    assert torch.equal(motor(metadata_a), motor(metadata_b))
    changed_width = recovery.nuisance_metadata_from_rows(
        [{"operation": "add", "width": 8, "position": 7}]
    )
    assert not torch.equal(motor(metadata_a), motor(changed_width))
    assert tuple(motor.state_dict()) == ("cell_delta",)
    assert motor.parameter_count() == recovery.NUISANCE_PARAMETER_COUNT == 20

    parameterization = recovery.nuisance_only_parameterization()
    for key in (
        "hidden_residual_dependence",
        "prompt_text_dependence",
        "token_history_dependence",
        "style_dependence",
        "current_carry_dependence",
        "operand_digit_dependence",
    ):
        assert parameterization[key] is False
    assert parameterization["fit_widths"] == [4, 6]
    assert parameterization["fit_design_rank"] == 20
    fit_design = recovery.nuisance_metadata_from_rows(
        [
            {"operation": operation, "width": width, "position": position}
            for operation, width, position in recovery.NUISANCE_FIT_CELLS
        ]
    )
    assert int(torch.linalg.matrix_rank(fit_design)) == 20
    assert torch.equal(fit_design, torch.eye(20, dtype=torch.float32))
    width8_terminal = recovery.nuisance_metadata_from_rows(
        [{"operation": "add", "width": 8, "position": 7}]
    )[0]
    add_w4_terminal = recovery.NUISANCE_FIT_CELLS.index(("add", 4, 3))
    add_w6_terminal = recovery.NUISANCE_FIT_CELLS.index(("add", 6, 5))
    expected_width8 = torch.zeros(20, dtype=torch.float32)
    expected_width8[add_w4_terminal] = -1.0
    expected_width8[add_w6_terminal] = 2.0
    assert torch.equal(width8_terminal, expected_width8)

    logits = torch.randn(1, 19)
    gate_off = upstream.apply_motor_logits(logits, metadata_a, motor, 3, 11, False)
    assert torch.equal(gate_off, logits)
    gate_on = upstream.apply_motor_logits(logits, metadata_a, motor, 3, 11, True)
    unchanged = [index for index in range(19) if index not in {3, 11}]
    assert torch.equal(gate_on[:, unchanged], logits[:, unchanged])
    assert not torch.equal(gate_on[:, [3, 11]], logits[:, [3, 11]])


def test_exact_combined_parameter_ledger_is_strictly_below_150m():
    cfg = dict(recovery.EXPECTED_BASE_PARAMETER_CONFIG)
    ledger = recovery.deployment_parameter_ledger(cfg, upstream.RANK)
    assert ledger["base"]["unique_trainable_parameters"] == 125_081_664
    assert ledger["treatment"]["trainable_parameters"] == 4_634
    assert ledger["null_heads"]["trainable_parameters"] == 21
    assert ledger["combined_unique_trainable_parameters"] == 125_086_319
    assert ledger["combined_unique_trainable_parameters"] < 150_000_000
    assert recovery.validate_deployment_parameter_ledger(
        ledger, cfg, upstream.RANK
    ) == ledger

    over_cap = copy.deepcopy(ledger)
    over_cap["combined_unique_trainable_parameters"] = 150_000_000
    over_cap["remaining_headroom"] = 0
    over_cap["strictly_below_cap"] = False
    with pytest.raises(ValueError, match="exceeds the cap"):
        recovery.validate_deployment_parameter_ledger(
            over_cap, cfg, upstream.RANK
        )


def test_nuisance_only_fit_rejects_ood_width_access_before_optimization():
    features = {
        "base01": torch.zeros(2, 2, dtype=torch.bfloat16),
        "other_lse": torch.zeros(2),
    }
    initial, _receipt = recovery.initial_nuisance_only_state()
    with pytest.raises(ValueError, match="only frozen training widths"):
        recovery._recovery_fit_nuisance_only(
            features,
            [
                {"operation": "add", "width": 4, "position": 0},
                {"operation": "add", "width": 8, "position": 0},
            ],
            torch.tensor([0, 1]),
            initial,
            "cpu",
            1,
            2,
            upstream.CANONICAL_LR,
            upstream.CANONICAL_WEIGHT_DECAY,
            upstream.FIT_SEED,
        )


def test_nuisance_only_fit_rejects_missing_or_one_class_fit_cells():
    initial, _receipt = recovery.initial_nuisance_only_state()
    missing_rows = [
        {"operation": "add", "width": 4, "position": 0},
        {"operation": "add", "width": 4, "position": 0},
        {"operation": "add", "width": 6, "position": 0},
        {"operation": "add", "width": 6, "position": 0},
    ]
    missing_features = {
        "base01": torch.zeros(4, 2, dtype=torch.bfloat16),
        "other_lse": torch.zeros(4),
    }
    with pytest.raises(ValueError, match="missing metadata cell"):
        recovery._recovery_fit_nuisance_only(
            missing_features,
            missing_rows,
            torch.tensor([0, 1, 0, 1]),
            initial,
            "cpu",
            1,
            4,
            upstream.CANONICAL_LR,
            upstream.CANONICAL_WEIGHT_DECAY,
            upstream.FIT_SEED,
        )

    one_class_rows = [
        {"operation": operation, "width": width, "position": position}
        for operation, width, position in recovery.NUISANCE_FIT_CELLS
    ]
    one_class_features = {
        "base01": torch.zeros(20, 2, dtype=torch.bfloat16),
        "other_lse": torch.zeros(20),
    }
    with pytest.raises(ValueError, match="both targets in every cell"):
        recovery._recovery_fit_nuisance_only(
            one_class_features,
            one_class_rows,
            torch.zeros(20, dtype=torch.long),
            initial,
            "cpu",
            1,
            20,
            upstream.CANONICAL_LR,
            upstream.CANONICAL_WEIGHT_DECAY,
            upstream.FIT_SEED,
        )


def test_fit_board_and_shuffled_control_are_exactly_balanced_within_nuisance():
    rows = [
        {
            "operation": operation,
            "width": width,
            "position": position,
            "style": style,
            "current_carry": current,
            "target": target,
        }
        for operation, width, position, style, current, target in sorted(
            upstream._all_fit_keys()
        )
    ]
    nuisance_keys = (
        "operation",
        "width",
        "position",
        "style",
        "current_carry",
    )

    def class_counts(labels):
        counts = collections.defaultdict(collections.Counter)
        for row, label in zip(rows, labels):
            counts[tuple(row[key] for key in nuisance_keys)][int(label)] += 1
        return counts

    true_labels = [row["target"] for row in rows]
    shuffled, receipt = upstream.permuted_control_labels(rows)
    before = class_counts(true_labels)
    after = class_counts(shuffled)
    assert before == after
    assert all(
        counts == collections.Counter({0: 1, 1: 1}) for counts in before.values()
    )
    assert receipt["changed"] >= len(rows) // 3


def test_raw_margin_diagnostic_reports_feasible_interval_and_optimal_threshold():
    margins = torch.tensor([-2.0, -1.0, -3.0, -4.0])
    base01 = torch.stack((torch.zeros_like(margins), margins), dim=1)
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    report = recovery.raw_carry_margin_diagnostic(base01, labels, 2.5)
    assert report["feasibility_condition"] == (
        "-min(m_positive) < delta < -max(m_negative)"
    )
    assert report["feasible_delta_interval"] == {
        "lower_exclusive": 2.0,
        "upper_exclusive": 3.0,
        "nonempty": True,
    }
    assert report["binary_accuracy_optimum"]["correct"] == 4
    assert report["binary_accuracy_optimum"]["denominator"] == 4
    assert report["fitted_delta_in_feasible_interval"] is True

    impossible = recovery.raw_carry_margin_diagnostic(
        torch.tensor([[0.0, -2.0], [0.0, -1.0]]),
        torch.tensor([1, 0], dtype=torch.long),
        1.5,
    )
    assert impossible["feasible_delta_interval"]["nonempty"] is False


_FEATURE_READING_AUTHORITIES = {}
_FEATURE_READING_TEMPLATE = None
_FEATURE_READING_TEMPLATE_AUTHORITY = None


def _feature_reading_evaluation():
    global _FEATURE_READING_TEMPLATE, _FEATURE_READING_TEMPLATE_AUTHORITY
    if _FEATURE_READING_TEMPLATE is not None:
        evaluation = copy.deepcopy(_FEATURE_READING_TEMPLATE)
        _FEATURE_READING_AUTHORITIES[id(evaluation)] = (
            _FEATURE_READING_TEMPLATE_AUTHORITY
        )
        return evaluation

    def make_case(
        case_id,
        phase,
        suite,
        regime,
        *,
        operation=None,
        width=None,
        position=None,
        target=None,
        transition=None,
        serializer=None,
        episode=None,
    ):
        case = {
            "schema": recovery.FEATURE_READING_CASE_SCHEMA,
            "case_id": case_id,
            "phase": phase,
            "suite": suite,
            "regime": regime,
            "operation": operation,
            "width": width,
            "position": position,
            "target": target,
            "expected_token_ids": [1],
            "expected_transition": transition,
            "expected_serializer_token_ids": serializer,
            "expected_episode_output": episode,
        }
        source_input = {
            "case_id": case_id,
            "phase": phase,
            "suite": suite,
            "regime": regime,
            "operation": operation,
            "width": width,
            "position": position,
            "target": target,
        }
        prompt = f"canonical prompt for {case_id}"
        generator_binding = {
            "schema": recovery.CASE_GENERATOR_BINDING_SCHEMA,
            "generator_id": recovery._expected_case_generator_id(phase),
            "source_contract_sha256": "d" * 64,
        }
        source_sha256 = recovery.stable_json_sha256(source_input)
        prompt_sha256 = recovery.sha256_bytes(prompt.encode())
        generator_receipt = recovery.stable_json_sha256(
            {
                "generator_binding": generator_binding,
                "source_input_sha256": source_sha256,
                "prompt_sha256": prompt_sha256,
            }
        )
        split = recovery._expected_case_split(case)
        membership = {
            "case_id": case_id,
            "split": split,
            "source_input_sha256": source_sha256,
            "prompt_sha256": prompt_sha256,
            "generator_receipt_sha256": generator_receipt,
        }
        case.update(
            {
                "source_input": source_input,
                "source_input_sha256": source_sha256,
                "prompt": prompt,
                "prompt_sha256": prompt_sha256,
                "generator_binding": generator_binding,
                "generator_receipt_sha256": generator_receipt,
                "split_membership": split,
                "split_membership_receipt_sha256": (
                    recovery.stable_json_sha256(membership)
                ),
            }
        )
        return case

    cases = []
    for item in recovery.DOWNSTREAM_EVALUATION_CONTRACT["carry_commit_rescue_set"][
        "cases"
    ]:
        cases.append(
            make_case(
                f"rescue:{item['id']}:{item['branch']}",
                "matched",
                "rescue",
                item["branch"],
                operation=item["operation"],
                episode=item["expected_answer"],
            )
        )
    for item in recovery.DOWNSTREAM_EVALUATION_CONTRACT["terminal_width_sweep_oracle"][
        "cases"
    ]:
        cases.append(
            make_case(
                f"terminal:{item['id']}",
                "matched",
                "terminal",
                item["arm"],
                operation="add",
                width=item["width"],
                position=item["width"] - 1,
                target=1 if item["arm"] == "positive" else 0,
                transition=recovery._terminal_transition_from_state(
                    item["expected_state"]
                ),
                serializer=[1],
                episode=item["expected_answer"],
            )
        )

    def generated_identity(index):
        cell = (index // 2) % 8
        operation = "add" if cell < 4 else "sub"
        within_operation = cell % 4
        width = 8 if within_operation < 2 else 9
        position = 0 if within_operation % 2 == 0 else width - 1
        return operation, width, position, index % 2

    for index in range(300):
        if index < 40:
            operation, width, position = recovery.NUISANCE_FIT_CELLS[index // 2]
            target = index % 2
            regime = "fit_width"
        else:
            operation, width, position, target = generated_identity(index - 40)
            regime = "width_ood"
        cases.append(
            make_case(
                f"development-episode-{index:03d}",
                "development",
                "episode",
                regime,
                operation=operation,
                width=width,
                position=position,
                target=target,
                transition={
                    "p": position,
                    "c": target,
                    "r": 1000 + index,
                    "z": 1,
                },
                serializer=[1],
                episode=1000 + index,
            )
        )
    for suite, count in (("boundary", 50), ("direct", 12)):
        for index in range(count):
            cases.append(
                make_case(
                    f"development-{suite}-{index:03d}",
                    "development",
                    suite,
                    suite,
                    episode=2000 + index,
                )
            )
    for index in range(256):
        operation, width, position, target = generated_identity(index)
        cases.append(
            make_case(
                f"confirmation-episode-{index:03d}",
                "confirmation",
                "episode",
                "width_ood",
                operation=operation,
                width=width,
                position=position,
                target=target,
                transition={
                    "p": position,
                    "c": target,
                    "r": 3000 + index,
                    "z": 1,
                },
                serializer=[1],
                episode=3000 + index,
            )
        )
    for phase in ("matched", "development", "confirmation"):
        cases.append(
            make_case(
                f"{phase}-preservation-000",
                phase,
                "preservation",
                "non_dws",
            )
        )

    deployment_vocabulary = {
        "schema": recovery.DEPLOYMENT_VOCABULARY_SCHEMA,
        "base_checkpoint_sha256": upstream.EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": upstream.EXPECTED_TOKENIZER_SHA256,
        "output_vocab_width": 3,
        "token_id_order": [0, 1, 2],
        "token_id_order_sha256": recovery.stable_json_sha256([0, 1, 2]),
        "tokenizer_id_to_token_sha256": recovery.stable_json_sha256(
            ["zero", "one", "other"]
        ),
        "zero_id": 0,
        "one_id": 1,
        "deployment_logit_dtype": "torch.bfloat16",
    }
    vocabulary_sha256 = recovery.stable_json_sha256(deployment_vocabulary)
    success_logits = [[0.0, 1.0, -1.0]]
    failure_logits = [[1.0, 0.0, -1.0]]
    gate_off_logits = [[0.0, 1.0, -1.0]]
    arms = recovery.DOWNSTREAM_EVALUATION_CONTRACT["arms"]
    records = []
    for case in cases:
        for arm in arms:
            preservation = case["suite"] == "preservation"
            treatment = arm == "treatment"
            full_vocab_success = preservation or treatment
            transition = None
            if case["expected_transition"] is not None:
                transition = copy.deepcopy(case["expected_transition"])
                matched_negative = (
                    case["phase"] == "matched"
                    and case["suite"] == "terminal"
                    and case["target"] == 0
                )
                if not treatment:
                    for field in ("p", "c", "r", "z"):
                        if not (matched_negative and field == "c"):
                            transition[field] += 1
            serializer = None
            if case["expected_serializer_token_ids"] is not None:
                matched_negative_preservation = (
                    case["phase"] == "matched"
                    and case["suite"] == "terminal"
                    and case["target"] == 0
                    and case["width"] <= 6
                )
                serializer = (
                    copy.deepcopy(case["expected_serializer_token_ids"])
                    if treatment or matched_negative_preservation
                    else [0]
                )
            episode = case["expected_episode_output"]
            if episode is not None and not treatment:
                episode += 1
            records.append(
                {
                    "schema": recovery.FEATURE_READING_RECORD_SCHEMA,
                    "case_id": case["case_id"],
                    "arm": arm,
                    "actual_token_ids": [1] if full_vocab_success else [0],
                    "full_vocab_logits": (
                        success_logits if full_vocab_success else failure_logits
                    ),
                    "actual_transition": transition,
                    "actual_serializer_token_ids": serializer,
                    "actual_episode_output": episode,
                    "motor_gate_trace": [False] if preservation else [arm != "base"],
                    "gate_off_base_logits": gate_off_logits,
                    "gate_off_arm_logits": gate_off_logits,
                    "deployment_vocabulary_sha256": vocabulary_sha256,
                }
            )

    fit_rows = []
    for index, (operation, width, position) in enumerate(
        recovery.NUISANCE_FIT_CELLS
    ):
        for target in (0, 1):
            case_id = f"fit-row-{index:03d}-{target}"
            source_input = {
                "case_id": case_id,
                "operation": operation,
                "width": width,
                "position": position,
                "target": target,
            }
            prompt = f"fit prompt for {case_id}"
            generator_binding = {
                "schema": recovery.CASE_GENERATOR_BINDING_SCHEMA,
                "generator_id": "a0c258e_fit_generator_v1",
                "source_contract_sha256": "e" * 64,
            }
            source_sha256 = recovery.stable_json_sha256(source_input)
            prompt_sha256 = recovery.sha256_bytes(prompt.encode())
            generator_receipt = recovery.stable_json_sha256(
                {
                    "generator_binding": generator_binding,
                    "source_input_sha256": source_sha256,
                    "prompt_sha256": prompt_sha256,
                }
            )
            membership = {
                "case_id": case_id,
                "split": "fit",
                "source_input_sha256": source_sha256,
                "prompt_sha256": prompt_sha256,
                "generator_receipt_sha256": generator_receipt,
            }
            fit_rows.append(
                {
                    "case_id": case_id,
                    "operation": operation,
                    "width": width,
                    "position": position,
                    "target": target,
                    "source_input": source_input,
                    "source_input_sha256": source_sha256,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                    "generator_binding": generator_binding,
                    "generator_receipt_sha256": generator_receipt,
                    "split_membership": "fit",
                    "split_membership_receipt_sha256": (
                        recovery.stable_json_sha256(membership)
                    ),
                }
            )
    labels = torch.tensor([row["target"] for row in fit_rows], dtype=torch.long)
    base01 = torch.zeros(len(fit_rows), 2, dtype=torch.bfloat16)
    other_lse = torch.zeros(len(fit_rows), dtype=torch.float32)
    other_token_ids = torch.tensor([2], dtype=torch.int64)
    fit_model_binding = {
        "schema": recovery.FIT_MODEL_BINDING_SCHEMA,
        "base_checkpoint_sha256": upstream.EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": upstream.EXPECTED_TOKENIZER_SHA256,
        "upstream_plan_sha256": recovery.UPSTREAM_PLAN_SHA256,
        "upstream_source_contract_sha256": "1" * 64,
        "upstream_shard_receipts_sha256": "2" * 64,
        "feature_merge_sha256": "3" * 64,
        "fit_rows_sha256": recovery.stable_json_sha256(fit_rows),
        "raw_feature_payload_sha256": recovery.scientific_tree_sha256(
            {
                "base01": base01,
                "other_lse": other_lse,
                "labels": labels,
                "other_token_ids": other_token_ids,
            }
        ),
        "deployment_vocabulary_sha256": vocabulary_sha256,
        "deployment_logit_dtype": "torch.bfloat16",
        "parameter_ledger": recovery.deployment_parameter_ledger(
            dict(recovery.EXPECTED_BASE_PARAMETER_CONFIG), upstream.RANK
        ),
    }
    groups = []
    for operation, width, position in recovery.NUISANCE_FIT_CELLS:
        indices = [
            index
            for index, row in enumerate(fit_rows)
            if (row["operation"], row["width"], row["position"])
            == (operation, width, position)
        ]
        groups.append((f"{operation}-w{width}-p{position}", indices))
    selected, fit_evidence_core = recovery._solve_full_board_scalar_groups(
        {"base01": base01, "other_lse": other_lse},
        labels,
        groups,
        "nuisance_only",
        fit_rows,
        fit_model_binding,
    )
    deployed_state = collections.OrderedDict(
        (("cell_delta", torch.tensor(selected, dtype=torch.float32)),)
    )
    fit_evidence = recovery._finalize_null_optimization_evidence(
        fit_evidence_core, deployed_state
    )
    raw_fit_evidence = {
        "schema": recovery.FIT_SELECTION_RAW_EVIDENCE_SCHEMA,
        "fit_model_binding": fit_model_binding,
        "fit_rows": fit_rows,
        "base01": base01,
        "other_lse": other_lse,
        "labels": labels,
        "other_token_ids": other_token_ids,
        "deployed_state": deployed_state,
    }
    derived_split = recovery._derive_split_receipt(cases, fit_rows)
    split_receipt = {
        "schema": recovery.CASE_SPLIT_RECEIPT_SCHEMA,
        "memberships": derived_split["memberships"],
        "memberships_sha256": derived_split["memberships_sha256"],
        "disjointness_sha256": derived_split["disjointness_sha256"],
    }
    evaluation = {
        "schema": recovery.FEATURE_READING_EVALUATION_SCHEMA,
        "deployment_vocabulary": deployment_vocabulary,
        "canonical_cases": cases,
        "records": records,
        "fit_selection_raw_evidence": raw_fit_evidence,
        "fit_selection_evidence": fit_evidence,
        "split_receipt": split_receipt,
    }
    _FEATURE_READING_AUTHORITIES[id(evaluation)] = (
        vocabulary_sha256,
        recovery.stable_json_sha256(fit_model_binding),
    )
    _FEATURE_READING_TEMPLATE = copy.deepcopy(evaluation)
    _FEATURE_READING_TEMPLATE_AUTHORITY = _FEATURE_READING_AUTHORITIES[id(evaluation)]
    return evaluation


def _evaluate_feature_reading(evaluation):
    vocabulary_sha256, model_binding_sha256 = _FEATURE_READING_AUTHORITIES.get(
        id(evaluation), ("0" * 64, "0" * 64)
    )
    return recovery.evaluate_feature_reading_decision(
        evaluation,
        expected_deployment_vocabulary_sha256=vocabulary_sha256,
        expected_fit_model_binding_sha256=model_binding_sha256,
    )


def _record(evaluation, case_id, arm):
    return next(
        record
        for record in evaluation["records"]
        if record["case_id"] == case_id and record["arm"] == arm
    )


def _reseal_feature_case(case):
    for name in (
        "case_id",
        "phase",
        "suite",
        "regime",
        "operation",
        "width",
        "position",
        "target",
    ):
        case["source_input"][name] = case[name]

    _reseal_feature_receipts(case)


def _reseal_feature_receipts(case):
    case["source_input_sha256"] = recovery.stable_json_sha256(case["source_input"])
    case["prompt_sha256"] = recovery.sha256_bytes(case["prompt"].encode())
    case["generator_receipt_sha256"] = recovery.stable_json_sha256(
        {
            "generator_binding": case["generator_binding"],
            "source_input_sha256": case["source_input_sha256"],
            "prompt_sha256": case["prompt_sha256"],
        }
    )
    membership = {
        "case_id": case["case_id"],
        "split": case["split_membership"],
        "source_input_sha256": case["source_input_sha256"],
        "prompt_sha256": case["prompt_sha256"],
        "generator_receipt_sha256": case["generator_receipt_sha256"],
    }
    case["split_membership_receipt_sha256"] = recovery.stable_json_sha256(
        membership
    )


def _refresh_feature_split_receipt(evaluation):
    derived = recovery._derive_split_receipt(
        evaluation["canonical_cases"],
        evaluation["fit_selection_raw_evidence"]["fit_rows"],
    )
    evaluation["split_receipt"] = {
        "schema": recovery.CASE_SPLIT_RECEIPT_SCHEMA,
        "memberships": derived["memberships"],
        "memberships_sha256": derived["memberships_sha256"],
        "disjointness_sha256": derived["disjointness_sha256"],
    }


def test_feature_reading_decision_derives_complete_case_matrix_and_strata():
    evaluation = _feature_reading_evaluation()
    decision = _evaluate_feature_reading(evaluation)
    assert decision["passed"] is True
    evidence = decision["evidence"]
    assert evidence["coverage"]["record_count"] == (
        evidence["coverage"]["case_count"] * 5
    )
    canonical_cases = sorted(
        evaluation["canonical_cases"], key=lambda case: case["case_id"]
    )
    assert evidence["canonical_case_manifest_sha256"] == recovery.stable_json_sha256(
        canonical_cases
    )
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", item["gate_off_arm_logits_sha256"])
        for item in evidence["per_case_evidence"]
    )
    assert evidence["model_selection"]["frozen_before_width8_and_confirmation"]
    assert evidence["model_selection"]["width8_selection_access"] is False
    assert any("_transition_" in name for name in evidence["strata"])
    assert any("_serializer_" in name for name in evidence["strata"])
    assert any("_full_vocab_" in name for name in evidence["strata"])


def test_feature_reading_rejects_receipt_count_and_case_matrix_forgeries():
    forged_summary = {
        "schema": "carry_motor_feature_reading_evaluation_v2",
        "coverage": {"confirmation_episodes_per_arm": 10**9},
        "generator_case_receipt_sha256": "a" * 64,
        "reported_case_receipt_sha256": "a" * 64,
        "confirmation_strata": {"transition": [{}], "episode": [{}]},
    }
    with pytest.raises(ValueError, match="schema mismatch"):
        _evaluate_feature_reading(forged_summary)

    missing = _feature_reading_evaluation()
    missing["records"].pop()
    with pytest.raises(ValueError, match="missing canonical case result"):
        _evaluate_feature_reading(missing)

    duplicate = _feature_reading_evaluation()
    duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
    with pytest.raises(ValueError, match="duplicate canonical case result"):
        _evaluate_feature_reading(duplicate)

    duplicate_case = _feature_reading_evaluation()
    duplicate_case["canonical_cases"].append(
        copy.deepcopy(duplicate_case["canonical_cases"][0])
    )
    with pytest.raises(ValueError, match="malformed or duplicated"):
        _evaluate_feature_reading(duplicate_case)

    extra = _feature_reading_evaluation()
    extra["records"][0]["case_id"] = "not-in-manifest"
    with pytest.raises(ValueError, match="extra or unknown"):
        _evaluate_feature_reading(extra)


def test_feature_reading_rejects_one_stratum_missing_cell_and_width8_leakage():
    one_stratum = _feature_reading_evaluation()
    for case in one_stratum["canonical_cases"]:
        if case["phase"] == "confirmation" and case["suite"] == "episode":
            case["operation"] = "add"
            case["width"] = 8
            case["position"] = 0
            _reseal_feature_case(case)
    _refresh_feature_split_receipt(one_stratum)
    with pytest.raises(ValueError, match="transition strata incomplete"):
        _evaluate_feature_reading(one_stratum)

    missing_cell = _feature_reading_evaluation()
    victim = next(
        case
        for case in missing_cell["canonical_cases"]
        if case["split_membership"] == "fit_width_audit"
    )
    victim["operation"] = "sub" if victim["operation"] == "add" else "add"
    _reseal_feature_case(victim)
    _refresh_feature_split_receipt(missing_cell)
    with pytest.raises(ValueError, match="missing fit metadata cell"):
        _evaluate_feature_reading(missing_cell)

    leakage = _feature_reading_evaluation()
    victim = next(
        case
        for case in leakage["canonical_cases"]
        if case["phase"] == "development" and case["width"] == 8
    )
    victim["split_membership"] = "fit_width_audit"
    with pytest.raises(ValueError, match="source or split receipt is forged"):
        _evaluate_feature_reading(leakage)


def test_feature_reading_rejects_nonconverged_final_iterate_and_under_capacity():
    nonconverged = _feature_reading_evaluation()
    nonconverged["fit_selection_evidence"]["converged"] = False
    nonconverged["fit_selection_evidence"]["selected_is_final_iterate"] = True
    with pytest.raises(ValueError, match="raw-evidence replay"):
        _evaluate_feature_reading(nonconverged)

    under_capacity = _feature_reading_evaluation()
    under_capacity["fit_selection_evidence"]["parameters"] = 9
    with pytest.raises(ValueError, match="raw-evidence replay"):
        _evaluate_feature_reading(under_capacity)


def test_fit_selection_replays_ce_candidates_hashes_and_deployed_state():
    forged_ce = _feature_reading_evaluation()
    evidence = forged_ce["fit_selection_evidence"]
    evidence["full_board_cross_entropy"] = 999.0
    evidence["full_board_cross_entropy_sha256"] = recovery.stable_json_sha256(
        {"value": 999.0}
    )
    core = {
        name: value
        for name, value in evidence.items()
        if name
        not in {
            "selected_state_sha256",
            "selected_state_scientific_sha256",
            "selected_checkpoint_evidence_sha256",
        }
    }
    evidence["selected_checkpoint_evidence_sha256"] = recovery.stable_json_sha256(
        core
    )
    with pytest.raises(ValueError, match="raw-evidence replay"):
        _evaluate_feature_reading(forged_ce)

    fabricated_hash = _feature_reading_evaluation()
    evidence = fabricated_hash["fit_selection_evidence"]
    evidence["fit_case_payload_sha256"] = "f" * 64
    core = {
        name: value
        for name, value in evidence.items()
        if name
        not in {
            "selected_state_sha256",
            "selected_state_scientific_sha256",
            "selected_checkpoint_evidence_sha256",
        }
    }
    evidence["selected_checkpoint_evidence_sha256"] = recovery.stable_json_sha256(
        core
    )
    with pytest.raises(ValueError, match="raw-evidence replay"):
        _evaluate_feature_reading(fabricated_hash)

    fabricated_state = _feature_reading_evaluation()
    raw = fabricated_state["fit_selection_raw_evidence"]
    raw["deployed_state"]["cell_delta"].fill_(999.0)
    evidence = fabricated_state["fit_selection_evidence"]
    evidence["selected_state_sha256"] = recovery.REVIEWED_TENSOR_STATE_SHA256(
        raw["deployed_state"]
    )
    evidence["selected_state_scientific_sha256"] = (
        recovery.scientific_tree_sha256(raw["deployed_state"])
    )
    with pytest.raises(ValueError, match="did not deploy"):
        _evaluate_feature_reading(fabricated_state)

    rebound_raw = _feature_reading_evaluation()
    raw = rebound_raw["fit_selection_raw_evidence"]
    raw["base01"][0, 0] = 4.0
    groups = []
    for operation, width, position in recovery.NUISANCE_FIT_CELLS:
        indices = [
            index
            for index, row in enumerate(raw["fit_rows"])
            if (row["operation"], row["width"], row["position"])
            == (operation, width, position)
        ]
        groups.append((f"{operation}-w{width}-p{position}", indices))
    selected, core = recovery._solve_full_board_scalar_groups(
        {"base01": raw["base01"], "other_lse": raw["other_lse"]},
        raw["labels"],
        groups,
        "nuisance_only",
        raw["fit_rows"],
        raw["fit_model_binding"],
    )
    raw["deployed_state"] = collections.OrderedDict(
        (("cell_delta", torch.tensor(selected, dtype=torch.float32)),)
    )
    rebound_raw["fit_selection_evidence"] = (
        recovery._finalize_null_optimization_evidence(core, raw["deployed_state"])
    )
    with pytest.raises(ValueError, match="sealed raw evidence"):
        _evaluate_feature_reading(rebound_raw)


def test_fit_selection_rejects_float32_and_noncanonical_vocabulary():
    float32 = _feature_reading_evaluation()
    float32["fit_selection_raw_evidence"]["base01"] = float32[
        "fit_selection_raw_evidence"
    ]["base01"].float()
    with pytest.raises(ValueError, match="bfloat16"):
        _evaluate_feature_reading(float32)

    narrower = _feature_reading_evaluation()
    for record in narrower["records"]:
        for name in (
            "full_vocab_logits",
            "gate_off_base_logits",
            "gate_off_arm_logits",
        ):
            record[name] = [row[:2] for row in record[name]]
    with pytest.raises(ValueError, match="deployed vocabulary width"):
        _evaluate_feature_reading(narrower)

    reordered = _feature_reading_evaluation()
    vocabulary = reordered["deployment_vocabulary"]
    vocabulary["token_id_order"] = [1, 0, 2]
    vocabulary["token_id_order_sha256"] = recovery.stable_json_sha256([1, 0, 2])
    with pytest.raises(ValueError, match="ordered"):
        _evaluate_feature_reading(reordered)


def test_feature_reading_rejects_source_split_and_oracle_forgeries():
    forged_source = _feature_reading_evaluation()
    forged_source["canonical_cases"][0]["source_input_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source or split receipt is forged"):
        _evaluate_feature_reading(forged_source)

    missing_source = _feature_reading_evaluation()
    del missing_source["canonical_cases"][0]["prompt_sha256"]
    with pytest.raises(ValueError, match="case schema mismatch"):
        _evaluate_feature_reading(missing_source)

    rehashed_inconsistent_source = _feature_reading_evaluation()
    case = rehashed_inconsistent_source["canonical_cases"][0]
    case["source_input"]["operation"] = (
        "sub" if case["operation"] == "add" else "add"
    )
    _reseal_feature_receipts(case)
    with pytest.raises(ValueError, match="differs from its bound source input"):
        _evaluate_feature_reading(rehashed_inconsistent_source)

    rehashed_incomplete_source = _feature_reading_evaluation()
    case = rehashed_incomplete_source["canonical_cases"][0]
    del case["source_input"]["phase"]
    _reseal_feature_receipts(case)
    with pytest.raises(ValueError, match="missing required identity fields"):
        _evaluate_feature_reading(rehashed_incomplete_source)

    forged_split = _feature_reading_evaluation()
    forged_split["split_receipt"]["disjointness_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="split receipt is missing or forged"):
        _evaluate_feature_reading(forged_split)

    oracle_declaration = _feature_reading_evaluation()
    oracle_declaration["split_receipt"]["oracle_cases_excluded_from_fit"] = True
    with pytest.raises(ValueError, match="split receipt is missing or forged"):
        _evaluate_feature_reading(oracle_declaration)

    oracle_overlap = _feature_reading_evaluation()
    oracle = next(
        case
        for case in oracle_overlap["canonical_cases"]
        if case["phase"] == "matched" and case["suite"] == "terminal"
    )
    fit_row = oracle_overlap["fit_selection_raw_evidence"]["fit_rows"][0]
    oracle["prompt"] = fit_row["prompt"]
    oracle["prompt_sha256"] = fit_row["prompt_sha256"]
    oracle["generator_receipt_sha256"] = recovery.stable_json_sha256(
        {
            "generator_binding": oracle["generator_binding"],
            "source_input_sha256": oracle["source_input_sha256"],
            "prompt_sha256": oracle["prompt_sha256"],
        }
    )
    membership = {
        "case_id": oracle["case_id"],
        "split": oracle["split_membership"],
        "source_input_sha256": oracle["source_input_sha256"],
        "prompt_sha256": oracle["prompt_sha256"],
        "generator_receipt_sha256": oracle["generator_receipt_sha256"],
    }
    oracle["split_membership_receipt_sha256"] = recovery.stable_json_sha256(
        membership
    )
    with pytest.raises(ValueError, match="crosses sealed splits|entered fit"):
        _evaluate_feature_reading(oracle_overlap)


def test_feature_reading_rejects_incomplete_output_and_gate_evidence():
    missing_gate = _feature_reading_evaluation()
    del missing_gate["records"][0]["gate_off_arm_logits"]
    with pytest.raises(ValueError, match="record schema mismatch"):
        _evaluate_feature_reading(missing_gate)

    missing_serializer = _feature_reading_evaluation()
    victim = next(
        record
        for record in missing_serializer["records"]
        if record["actual_serializer_token_ids"] is not None
    )
    victim["actual_serializer_token_ids"] = None
    with pytest.raises(ValueError, match="serializer output is missing"):
        _evaluate_feature_reading(missing_serializer)

    gate_mismatch = _feature_reading_evaluation()
    gate_mismatch["records"][0]["gate_off_arm_logits"] = [[1.0, 0.0, -1.0]]
    decision = _evaluate_feature_reading(gate_mismatch)
    assert decision["passed"] is False
    assert any(
        name.startswith("all_case_gate_off_full_vocab_identity") and not passed
        for name, passed in decision["checks"].items()
    )


def test_feature_reading_is_noncompensatory_in_every_required_stratum():
    matched = _feature_reading_evaluation()
    treatment_records = {
        record["case_id"]: record
        for record in matched["records"]
        if record["arm"] == "treatment"
    }
    for index, record in enumerate(matched["records"]):
        if record["arm"] != "constant_bias":
            continue
        replacement = copy.deepcopy(treatment_records[record["case_id"]])
        replacement["arm"] = "constant_bias"
        matched["records"][index] = replacement
    decision = _evaluate_feature_reading(matched)
    assert decision["passed"] is False
    assert any(
        name.endswith("constant_bias_noncompensatory") and not passed
        for name, passed in decision["checks"].items()
    )

    stratum = _feature_reading_evaluation()
    anchor = next(
        case
        for case in stratum["canonical_cases"]
        if case["phase"] == "confirmation" and case["suite"] == "episode"
    )
    identity = (
        anchor["operation"],
        anchor["width"],
        anchor["position"],
        anchor["target"],
    )
    for case in stratum["canonical_cases"]:
        if (
            case["phase"] == "confirmation"
            and case["suite"] == "episode"
            and (case["operation"], case["width"], case["position"], case["target"])
            == identity
        ):
            record = _record(stratum, case["case_id"], "nuisance_only")
            record["actual_transition"] = copy.deepcopy(case["expected_transition"])
    decision = _evaluate_feature_reading(stratum)
    assert decision["passed"] is False
    assert any(
        name.startswith("confirmation_transition_")
        and name.endswith("nuisance_only_noncompensatory")
        and not passed
        for name, passed in decision["checks"].items()
    )


def test_constant_bias_payload_is_type_strict_and_recomputed(monkeypatch):
    fit_payload, context = _legacy_payload_fixture(monkeypatch)
    payload = _constant_bias_payload_fixture(fit_payload, context)
    recovery.validate_constant_bias_payload(payload, fit_payload, context)

    attacks = []
    hostile = copy.deepcopy(payload)
    hostile["state"] = dict(hostile["state"])
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["state"]["hidden_weight"] = torch.zeros(4)
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["fit"]["converged"] = False
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["raw_margin_diagnostic"]["fitted_delta"] = 0.0
    attacks.append(hostile)
    for hostile in attacks:
        with pytest.raises(ValueError, match="constant[-_]bias"):
            recovery.validate_constant_bias_payload(hostile, fit_payload, context)


def test_nuisance_only_payload_is_type_strict_and_metadata_receipted(monkeypatch):
    fit_payload, context = _legacy_payload_fixture(monkeypatch)
    payload = _nuisance_only_payload_fixture(fit_payload, context)
    recovery.validate_nuisance_only_payload(payload, fit_payload, context)

    attacks = []
    hostile = copy.deepcopy(payload)
    hostile["state"] = dict(hostile["state"])
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["state"]["hidden_weight"] = torch.zeros(4)
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["fit"]["converged"] = False
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["training_metadata_receipt"]["source_fields"].append("hidden")
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["width_ood_policy"]["fit_may_read_public_or_confirmation_ood_rows"] = True
    attacks.append(hostile)
    for hostile in attacks:
        with pytest.raises(ValueError, match="nuisance[-_]only"):
            recovery.validate_nuisance_only_payload(hostile, fit_payload, context)


def test_fit_cli_places_complete_replay_gate_before_every_seal():
    source = inspect.getsource(recovery._fit)
    assert source.index("_build_constant_bias_payload(") < source.index("bundle = {")
    assert source.index("_build_nuisance_only_payload(") < source.index("bundle = {")
    assert '"constant_bias_payload": constant_bias_payload' in source
    assert '"nuisance_only_payload": nuisance_only_payload' in source
    assert source.count("validate_and_replay_recovery_fit(") == 2
    assert source.index("validate_and_replay_recovery_fit(") < source.index(
        "seal_recovery_fit(out, signed_layout_binding)"
    )
    second_gate = source.rindex("validate_and_replay_recovery_fit(")
    second_seal = source.rindex("seal_recovery_fit(out, signed_layout_binding)")
    assert second_gate < second_seal
    publication = source.rindex("published_sha256 = publish_recovery_torch(")
    post_publish_binding = source.index(
        "verify_context_bindings(context, recovery_plan_bound)", publication
    )
    assert publication < post_publish_binding < second_gate
    pre_seal_binding = source.rindex(
        "verify_context_bindings(context, recovery_plan_bound)",
        second_gate,
        second_seal,
    )
    post_seal_binding = source.index(
        "verify_context_bindings(context, recovery_plan_bound)", second_seal
    )
    assert second_gate < pre_seal_binding < second_seal < post_seal_binding


def test_wrapper_and_fit_bind_both_exact_calibration_controls(monkeypatch):
    wrapper = (
        Path(__file__).resolve().parent / "jobs" / "causal_carry_motor_recovery.sbatch"
    ).read_text()
    assert "readonly CONSTANT_BIAS_CONTROL=zero_sum_delta_v1" in wrapper
    assert "readonly NUISANCE_ONLY_CONTROL=saturated_op_width_position_v1" in wrapper
    assert '--constant-bias-control "$CONSTANT_BIAS_CONTROL"' in wrapper
    assert '--nuisance-only-control "$NUISANCE_ONLY_CONTROL"' in wrapper
    assert '--null-optimization-control "$NULL_OPTIMIZATION_CONTROL"' in wrapper
    assert '--selection-widths "$SELECTION_WIDTHS"' in wrapper
    reached = []
    monkeypatch.setattr(recovery, "prepare_context", lambda _args: reached.append(True))
    with pytest.raises(ValueError, match="reviewed null"):
        recovery._fit(
            recovery.argparse.Namespace(
                constant_bias_control="hostile_control",
                nuisance_only_control=recovery.NUISANCE_ONLY_CONTROL_ID,
                null_optimization_control=recovery.NULL_OPTIMIZATION_SCHEMA,
                selection_widths="4,6",
            )
        )
    with pytest.raises(ValueError, match="nuisance-only"):
        recovery._fit(
            recovery.argparse.Namespace(
                constant_bias_control=recovery.CONSTANT_BIAS_CONTROL_ID,
                nuisance_only_control="hostile_control",
                null_optimization_control=recovery.NULL_OPTIMIZATION_SCHEMA,
                selection_widths="4,6",
            )
        )
    with pytest.raises(ValueError, match="full-board"):
        recovery._fit(
            recovery.argparse.Namespace(
                constant_bias_control=recovery.CONSTANT_BIAS_CONTROL_ID,
                nuisance_only_control=recovery.NUISANCE_ONLY_CONTROL_ID,
                null_optimization_control="final_adamw_iterate",
                selection_widths="4,6",
            )
        )
    with pytest.raises(ValueError, match="selection widths"):
        recovery._fit(
            recovery.argparse.Namespace(
                constant_bias_control=recovery.CONSTANT_BIAS_CONTROL_ID,
                nuisance_only_control=recovery.NUISANCE_ONLY_CONTROL_ID,
                null_optimization_control=recovery.NULL_OPTIMIZATION_SCHEMA,
                selection_widths="4,6,8",
            )
        )
    assert reached == []


def test_legacy_payload_validator_is_complete_and_type_strict(monkeypatch):
    payload, context = _legacy_payload_fixture(monkeypatch)
    recovery.validate_legacy_payload_type_strict(payload, context, "cpu")

    attacks = []
    hostile = copy.deepcopy(payload)
    hostile["checkpoint_step"] = 280000.0
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["d_model"] = True
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["rank"] = 2.0
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["zero_id"] = False
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["board"]["rows"] = 3.0
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["control"]["seed"] = 7.0
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["treatment"] = dict(hostile["treatment"])
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["treatment_fit"]["updates"] = 4.0
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["shuffled_fit"]["first_loss"] = 1
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["linear_diagnostic"]["test_correct"] = True
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["linear_diagnostic"]["test_accuracy"] = 1
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["fit_feature_metrics"]["accuracy"] = 1
    attacks.append(hostile)

    for hostile in attacks:
        with pytest.raises(ValueError, match="legacy payload"):
            recovery.validate_legacy_payload_type_strict(hostile, context, "cpu")


def test_v11_output_has_dual_provenance_and_cannot_publish_as_v8(monkeypatch):
    executor = {"git_commit": NEW_COMMIT}
    runtime = {"schema": recovery.RECOVERY_EXECUTOR_RUNTIME_SCHEMA}
    parent_binding = {"schema": recovery.RECOVERY_PARENT_BINDING_SCHEMA}
    upstream_contract = {"git_commit": recovery.UPSTREAM_SOURCE_COMMIT}
    plan_binding = {"path": "/upstream/plan.json", "sha256": "1" * 64}
    receipts = [{"shard_index": index} for index in range(8)]
    proof = {"mismatch_count": 2}
    layout = {"schema": recovery.RECOVERY_LAYOUT_BINDING_SCHEMA}
    checkpoint = {"cfg": dict(recovery.EXPECTED_BASE_PARAMETER_CONFIG)}
    parameter_ledger = recovery.deployment_parameter_ledger(
        checkpoint["cfg"], upstream.RANK
    )
    recovery_plan = {
        "recovery_executor_source_contract": executor,
        "executor_runtime_contract": runtime,
        "recovery_parent_binding": parent_binding,
        "output_contract": {"layout_binding": layout},
        "upstream_protocol": {
            "source_contract": upstream_contract,
            "plan_binding": plan_binding,
            "shard_receipts": receipts,
        },
        "normalization_proof": proof,
        "fit_contract": {"parameter_ledger": parameter_ledger},
    }
    fit_payload = {key: None for key in recovery.LEGACY_PAYLOAD_KEYS}
    constant_bias_payload = {key: None for key in recovery.CONSTANT_BIAS_PAYLOAD_KEYS}
    nuisance_only_payload = {key: None for key in recovery.NUISANCE_ONLY_PAYLOAD_KEYS}
    bundle = {
        "audit": recovery.RECOVERY_FIT_AUDIT,
        "recovery": True,
        "recovery_plan_sha256": "2" * 64,
        "recovery_executor_source_contract": executor,
        "executor_runtime_contract": runtime,
        "recovery_parent_binding": parent_binding,
        "recovery_layout_binding": layout,
        "upstream_protocol_source_contract": upstream_contract,
        "upstream_plan_binding": plan_binding,
        "upstream_shard_receipts": receipts,
        "normalization_proof": proof,
        "allowed_transformation": recovery.ALLOWED_TRANSFORMATION,
        "deserialization_contract": recovery.DESERIALIZATION_CONTRACT,
        "slurm_h100_attestation": {"schema": recovery.RECOVERY_SLURM_CONTRACT_SCHEMA},
        "parameter_ledger": parameter_ledger,
        "trajectory_replay_proof": {
            "schema": recovery.RECOVERY_TRAJECTORY_PROOF_SCHEMA
        },
        "fit_payload": fit_payload,
        "constant_bias_payload": constant_bias_payload,
        "nuisance_only_payload": nuisance_only_payload,
        "claim_boundary": recovery.RECOVERY_FIT_CLAIM_BOUNDARY,
    }
    monkeypatch.setattr(recovery, "assert_reviewed_callable_exports", lambda: {})
    monkeypatch.setattr(recovery, "REVIEWED_VALIDATE_MOTOR_BUNDLE", lambda *_: None)
    monkeypatch.setattr(
        recovery, "validate_legacy_payload_type_strict", lambda *_: None
    )
    monkeypatch.setattr(recovery, "validate_constant_bias_payload", lambda *_: None)
    monkeypatch.setattr(recovery, "validate_nuisance_only_payload", lambda *_: None)
    monkeypatch.setattr(recovery, "validate_slurm_h100_attestation", lambda *_: True)
    monkeypatch.setattr(
        recovery,
        "build_trajectory_replay_proof",
        lambda *_: {"schema": recovery.RECOVERY_TRAJECTORY_PROOF_SCHEMA},
    )
    context = {
        "expected_bindings": {},
        "upstream_source_hashes": {},
        "upstream_source_contract": upstream_contract,
        "upstream_plan": {"fit_budget": {"rank": upstream.RANK}},
        "features": {},
        "feature_merge": {},
    }
    recovery.validate_recovery_fit_bundle(
        bundle, recovery_plan, "2" * 64, context, checkpoint, "cpu"
    )
    assert bundle["audit"] == recovery.RECOVERY_FIT_AUDIT
    assert "canonical" not in bundle
    hostile = copy.deepcopy(bundle)
    hostile["audit"] = upstream.CANONICAL_FIT_AUDIT
    with pytest.raises(ValueError, match="not a v11"):
        recovery.validate_recovery_fit_bundle(
            hostile, recovery_plan, "2" * 64, context, checkpoint, "cpu"
        )


def _valid_slurm_records():
    tres = "billing=4,cpu=4,mem=32G,node=1,gres/gpu=1,gres/gpu:nvidia_h100_pcie=1"
    job = {
        "JobId": "12345",
        "Account": "skattel",
        "Partition": "normal",
        "NumNodes": "1",
        "NumTasks": "1",
        "CPUs/Task": "4",
        "MinMemoryNode": "32G",
        "TimeLimit": "06:00:00",
        "Requeue": "0",
        "ReqTRES": tres,
        "AllocTRES": tres,
        "TresPerNode": "gres/gpu:nvidia_h100_pcie:1",
        "JobState": "RUNNING",
        "NodeList": "evc50",
        "ExcNodeList": "evc[22,26,31-32,36-37,40,43-44]",
    }
    accounting = {
        "JobIDRaw": "12345",
        "Account": "skattel",
        "Partition": "normal",
        "AllocCPUS": "4",
        "NNodes": "1",
        "NTasks": "1",
        "ReqMem": "32Gn",
        "ReqTRES": tres,
        "AllocTRES": tres,
        "State": "RUNNING",
        "TimelimitRaw": "21600",
    }
    return job, accounting


def _valid_slurm_attestation():
    job, accounting = _valid_slurm_records()
    uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    return {
        "schema": recovery.RECOVERY_SLURM_CONTRACT_SCHEMA,
        "job_id": 12345,
        "restart_count": 0,
        "cuda_visible_devices": uuid,
        "slurm_gres": {
            "account": "skattel",
            "job_id": "12345",
            "alloc_tres": job["AllocTRES"],
            "tres_per_node": job["TresPerNode"],
            "slurm_job_gpus": "0",
            "slurm_step_gpus": "0",
        },
        "slurm_node_binding": {
            "node_list_expression": job["NodeList"],
            "allocated_nodes": ["evc50"],
            "allocated_nodes_sha256": recovery.stable_json_sha256(["evc50"]),
            "excluded_node_list_expression": job["ExcNodeList"],
            "excluded_nodes": list(recovery.EXPECTED_EXCLUDED_NODES),
            "excluded_nodes_sha256": recovery.stable_json_sha256(
                list(recovery.EXPECTED_EXCLUDED_NODES)
            ),
        },
        "scontrol": job,
        "sacct": accounting,
        "gpu": {
            "index": "0",
            "uuid": uuid,
            "pci.bus_id": "00000000:65:00.0",
            "name": "NVIDIA H100 PCIe",
            "memory.total": 81559,
            "compute_cap": "9.0",
            "mig.mode.current": "Disabled",
        },
        "cgroup_device_authorization": {
            "membership": {
                "hierarchy": "0",
                "controllers": [],
                "path": "/slurm/uid_501/job_12345/step_batch",
            },
            "authorization_method": "effective_character_device_open",
            "authorized_gpu_device_nodes": ["/dev/nvidia0"],
            "selected_device": {
                "path": "/dev/nvidia0",
                "major": 195,
                "minor": 0,
                "mode": 0o666,
                "device": 9,
                "inode": 10,
                "links": 1,
            },
            "sysfs_device_path": "/sys/devices/pci0000:64/0000:65:00.0",
            "pci_bus_id": "0000:65:00.0",
            "effective_authorization": True,
        },
        "mig_identity": {
            "mode": "Disabled",
            "gpu_uuid": uuid,
            "gpu_instance_id": None,
            "compute_instance_id": None,
            "mig_device_uuid": None,
            "nvidia_smi_list": [f"GPU 0: NVIDIA H100 PCIe (UUID: {uuid})"],
        },
        "contract": recovery.EXPECTED_SLURM_REQUEST,
    }


def test_slurm_parsers_and_full_h100_attestation_are_type_strict():
    job, accounting = _valid_slurm_records()
    assert recovery.validate_slurm_records(job, accounting)
    assert recovery.validate_slurm_h100_attestation(_valid_slurm_attestation())
    parsed = recovery.parse_slurm_key_value_record(
        "JobId=12345 Partition=normal Requeue=0"
    )
    assert parsed == {"JobId": "12345", "Partition": "normal", "Requeue": "0"}
    with pytest.raises(ValueError, match="duplicate"):
        recovery.parse_slurm_key_value_record("JobId=1 JobId=2")
    with pytest.raises(ValueError, match="field count"):
        recovery.parse_sacct_pipe_records("1|normal|", ("JobIDRaw",))


def test_production_h100_capture_accepts_one_consistent_pci_selector(
    monkeypatch, tmp_path
):
    job, accounting = _valid_slurm_records()
    uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    pci_long = "00000000:65:00.0"
    pci_normalized = "0000:65:00.0"
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    binaries = {}
    for name in ("scontrol", "sacct", "nvidia-smi"):
        path = binary_dir / name
        path.write_text("fixture\n")
        binaries[name] = path
    monkeypatch.setattr(recovery, "PINNED_SCONTROL", binaries["scontrol"])
    monkeypatch.setattr(recovery, "PINNED_SACCT", binaries["sacct"])
    monkeypatch.setattr(recovery, "PINNED_NVIDIA_SMI", binaries["nvidia-smi"])
    monkeypatch.setattr(recovery, "REVIEWED_CUDA_DEVICE_COUNT", lambda: 1)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", pci_normalized)
    monkeypatch.setenv("SLURM_JOB_GPUS", pci_long)
    monkeypatch.setenv("SLURM_STEP_GPUS", pci_normalized)

    fields = (
        "JobIDRaw",
        "Account",
        "Partition",
        "AllocCPUS",
        "NNodes",
        "NTasks",
        "ReqMem",
        "ReqTRES",
        "AllocTRES",
        "State",
        "TimelimitRaw",
    )
    scontrol_text = " ".join(f"{key}={value}" for key, value in job.items()) + "\n"
    sacct_text = "|".join(accounting[key] for key in fields) + "|\n"
    gpu_text = f"0, {uuid}, {pci_long}, NVIDIA H100 PCIe, 81559, 9.0, Disabled\n"
    list_text = f"GPU 0: NVIDIA H100 PCIe (UUID: {uuid})\n"

    def fake_check_output(arguments, *, text):
        assert text is True
        if arguments[0] == str(binaries["scontrol"]):
            if arguments[1:3] == ["show", "hostnames"]:
                if arguments[3] == job["NodeList"]:
                    return "evc50\n"
                if arguments[3] == job["ExcNodeList"]:
                    return "\n".join(recovery.EXPECTED_EXCLUDED_NODES) + "\n"
                raise AssertionError(f"unexpected host list: {arguments!r}")
            return scontrol_text
        if arguments[0] == str(binaries["sacct"]):
            return sacct_text
        if arguments[-1] == "-L":
            return list_text
        if arguments[0] == str(binaries["nvidia-smi"]):
            return gpu_text
        raise AssertionError(f"unexpected capture command: {arguments!r}")

    monkeypatch.setattr(recovery.subprocess, "check_output", fake_check_output)

    device_fixture = tmp_path / "nvidia0"
    device_fixture.write_bytes(b"device fixture")
    sysfs_fixture = tmp_path / "sysfs" / "0000:65:00.0"
    sysfs_fixture.mkdir(parents=True)
    cgroup_fixture = tmp_path / "proc-self-cgroup"
    cgroup_fixture.write_text("0::/slurm/uid_501/job_12345/step_batch\n")

    class DeviceStat:
        st_mode = stat.S_IFCHR | 0o666
        st_dev = 9
        st_ino = 10
        st_nlink = 1
        st_uid = os.getuid()
        st_gid = os.getgid()
        st_size = 0
        st_mtime_ns = 1
        st_ctime_ns = 1
        st_rdev = 1

    class DeviceEntry:
        name = "nvidia0"
        path = "/dev/nvidia0"

    original_stat = recovery.os.stat
    original_scandir = recovery.os.scandir
    original_open = recovery.os.open
    original_fstat = recovery.os.fstat
    original_close = recovery.os.close
    original_resolve = recovery.Path.resolve
    original_read_kernel_text = recovery._read_kernel_text
    device_fds = set()

    def fake_stat(path, *args, **kwargs):
        if os.fspath(path) == "/dev/nvidia0":
            return DeviceStat()
        return original_stat(path, *args, **kwargs)

    def fake_scandir(path):
        if os.fspath(path) == "/dev":
            return [DeviceEntry()]
        return original_scandir(path)

    def fake_open(path, flags, mode=0o777, *, dir_fd=None):
        if os.fspath(path) == "/dev/nvidia0" and dir_fd is None:
            descriptor = original_open(device_fixture, os.O_RDONLY)
            device_fds.add(descriptor)
            return descriptor
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def fake_fstat(descriptor):
        if descriptor in device_fds:
            return DeviceStat()
        return original_fstat(descriptor)

    def fake_close(descriptor):
        device_fds.discard(descriptor)
        return original_close(descriptor)

    def fake_resolve(path, strict=False):
        if str(path) == "/sys/dev/char/195:0/device":
            return sysfs_fixture
        return original_resolve(path, strict=strict)

    def fake_read_kernel_text(path, label):
        if os.fspath(path) == "/proc/self/cgroup":
            assert label == "process cgroup membership"
            return cgroup_fixture.read_text()
        return original_read_kernel_text(path, label)

    monkeypatch.setattr(recovery.os, "stat", fake_stat)
    monkeypatch.setattr(recovery.os, "scandir", fake_scandir)
    monkeypatch.setattr(recovery.os, "open", fake_open)
    monkeypatch.setattr(recovery.os, "fstat", fake_fstat)
    monkeypatch.setattr(recovery.os, "close", fake_close)
    monkeypatch.setattr(recovery.os, "major", lambda value: 195 if value == 1 else 0)
    monkeypatch.setattr(recovery.os, "minor", lambda value: 0)
    monkeypatch.setattr(recovery.Path, "resolve", fake_resolve)
    monkeypatch.setattr(recovery, "_read_kernel_text", fake_read_kernel_text)

    captured = recovery.capture_slurm_h100_attestation()
    assert captured["cuda_visible_devices"] == pci_normalized
    assert captured["gpu"]["pci.bus_id"] == pci_long
    assert captured["slurm_gres"]["slurm_job_gpus"] == pci_long
    assert captured["slurm_gres"]["slurm_step_gpus"] == pci_normalized
    assert captured["cgroup_device_authorization"]["pci_bus_id"] == pci_normalized
    assert captured["cgroup_device_authorization"]["sysfs_device_path"] == str(
        sysfs_fixture
    )
    assert recovery.validate_slurm_h100_attestation(captured)


def test_slurm_attestation_rejects_every_resource_and_gpu_substitution():
    attacks = []
    for key, value in (
        ("Account", "other"),
        ("Partition", "short"),
        ("NumNodes", "2"),
        ("NumTasks", "2"),
        ("CPUs/Task", "8"),
        ("MinMemoryNode", "16G"),
        ("TimeLimit", "05:59:59"),
        ("Requeue", "1"),
        ("TresPerNode", "gres/gpu:1"),
    ):
        hostile = _valid_slurm_attestation()
        hostile["scontrol"][key] = value
        attacks.append(hostile)
    for key in ("ReqTRES", "AllocTRES"):
        hostile = _valid_slurm_attestation()
        hostile["scontrol"][key] = hostile["scontrol"][key].replace(
            ",gres/gpu:nvidia_h100_pcie=1", ""
        )
        hostile["sacct"][key] = hostile["scontrol"][key]
        attacks.append(hostile)
        hostile = _valid_slurm_attestation()
        hostile["scontrol"][key] = hostile["scontrol"][key].replace(
            "mem=32G", "mem=16G"
        )
        hostile["sacct"][key] = hostile["scontrol"][key]
        attacks.append(hostile)
        hostile = _valid_slurm_attestation()
        hostile["scontrol"][key] += ",gres/gpu:nvidia_a100_80gb_pcie=1"
        hostile["sacct"][key] = hostile["scontrol"][key]
        attacks.append(hostile)
    for key, value in (
        ("name", "NVIDIA A100-SXM4-80GB"),
        ("memory.total", 79999),
        ("memory.total", 82001),
        ("compute_cap", "8.0"),
        ("mig.mode.current", "Enabled"),
        ("uuid", "not-a-gpu-uuid"),
        ("pci.bus_id", "00000000:66:00.0"),
    ):
        hostile = _valid_slurm_attestation()
        hostile["gpu"][key] = value
        attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["cuda_visible_devices"] = "1"
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["job_id"] = 12345.0
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["restart_count"] = False
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["slurm_gres"]["account"] = "other"
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["slurm_gres"]["alloc_tres"] = "gres/gpu=2"
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["slurm_gres"]["slurm_job_gpus"] = "1"
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["cgroup_device_authorization"]["authorized_gpu_device_nodes"] = [
        "/dev/nvidia0",
        "/dev/nvidia1",
    ]
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["cgroup_device_authorization"]["membership"]["path"] = (
        "/slurm/job_99999/step_batch"
    )
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["cgroup_device_authorization"]["pci_bus_id"] = "0000:66:00.0"
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["mig_identity"]["gpu_instance_id"] = 1
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["mig_identity"]["nvidia_smi_list"].append(
        "  MIG 1g.10gb Device 0: (UUID: MIG-fake)"
    )
    attacks.append(hostile)
    hostile = _valid_slurm_attestation()
    hostile["scontrol"]["NodeList"] = "evc43"
    hostile["slurm_node_binding"]["node_list_expression"] = "evc43"
    hostile["slurm_node_binding"]["allocated_nodes"] = ["evc43"]
    hostile["slurm_node_binding"]["allocated_nodes_sha256"] = (
        recovery.stable_json_sha256(["evc43"])
    )
    attacks.append(hostile)
    for hostile in attacks:
        with pytest.raises(ValueError):
            recovery.validate_slurm_h100_attestation(hostile)


def test_slurm_wrapper_is_no_requeue_and_never_opens_confirmation_secret():
    wrapper = (
        Path(__file__).resolve().parent / "jobs" / "causal_carry_motor_recovery.sbatch"
    )
    text = wrapper.read_text()
    assert text.startswith("#!/bin/bash -p\n")
    builtin_proof = text.index("$(builtin type -t compgen)")
    first_enumeration = text.index("builtin compgen -A function")
    assert builtin_proof < first_enumeration
    assert "#SBATCH --no-requeue" in text
    assert "#SBATCH --account=skattel" in text
    assert "#SBATCH --export=NONE" in text
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in text
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=32G" in text
    assert "#SBATCH --time=06:00:00" in text
    assert (
        "#SBATCH --exclude=evc22,evc26,evc31,evc32,evc36,evc37,evc40,evc43,evc44"
        in text
    )
    assert "umask 077" in text
    assert '"$(umask)" != 0077' in text
    assert "if [[ -n ${!name+x} ]]" in text
    assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in text
    assert "NVIDIA_TF32_OVERRIDE" in text
    assert "BLIS_*|CUBLAS_*" in text
    assert "BASH_FUNC_*" in text
    assert "BASH_ENV is forbidden before recovery startup" in text
    assert "SHELLOPTS=" in text
    assert "exported Bash functions are forbidden" in text
    assert "GIT_*" in text
    assert "export GIT_ATTR_NOSYSTEM=1" in text
    assert "export GIT_CONFIG_GLOBAL=/dev/null" in text
    assert "export GIT_CONFIG_NOSYSTEM=1" in text
    assert "export GIT_CONFIG_SYSTEM=/dev/null" in text
    assert "export GIT_NO_REPLACE_OBJECTS=1" in text
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in text
    assert "TORCH_FORCE_WEIGHTS_ONLY_LOAD" in text
    assert "readonly PY=$DATA_ROOT/miniforge3/bin/python" in text
    assert "${PY:-" not in text
    assert "readonly GIT=/usr/bin/git" not in text
    assert "rev-list --parents" not in text
    assert "diff --name-status" not in text
    assert "status --porcelain" not in text
    assert "git_safe" not in text
    assert "six reviewed positional controls" in text
    assert "hostile_review" in text
    assert "RECOVERY_PARENT_RECEIPT" in text
    assert "recovery parent or durable receipt identity failed" in text
    assert "${CONFIRMATION_SECRET_FILE:?" not in text
    assert 'cat "$CONFIRMATION_SECRET_FILE"' not in text
    assert 'source "$CONFIRMATION_SECRET_FILE"' not in text
    assert '"$PY" train/causal_carry_motor.py' not in text
    assert (
        '"$PY" -I -S -B "$SOURCE_ROOT/train/causal_carry_motor_recovery.py" fit'
    ) in text
    assert "SBATCH_*" in text
    assert "readonly SCONTROL=/usr/bin/scontrol" in text
    assert "readonly SACCT=/usr/bin/sacct" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "Account=skattel" in text
    assert "Partition=normal" in text
    assert '"$SCONTROL" show hostnames "$NODE_LIST"' in text
    assert '"$SCONTROL" show hostnames "$EXCLUDED_NODE_LIST"' in text
    assert "recovery allocation landed on an excluded node" in text
    assert "atomic_torch" not in text
    assert "ln " not in text


def _run_production_wrapper(environment):
    wrapper = (
        Path(__file__).resolve().parent / "jobs" / "causal_carry_motor_recovery.sbatch"
    )
    return subprocess.run(
        ["/bin/bash", "-p", str(wrapper)],
        env=environment,
        capture_output=True,
        text=True,
    )


def test_preimport_source_gate_precedes_repository_sys_path_and_imports():
    source = Path(recovery.__file__).read_text()
    gate = source.index('if __name__ == "__main__":')
    release = source.index("_preimport_validate_and_release_sys_path()", gate)
    torch_import = source.index("import torch", release)
    assert gate < release < torch_import
    assert "repository path entered sys.path before closed-world validation" in source
    assert '"local": "synthetic bare GIT_DIR with no config file"' in source
    for porcelain in ('"diff"', '"show"', '"status"', '"checkout"', '"add"'):
        assert porcelain in source
    assert 'repository.run("diff"' not in source
    assert 'repository.run("show"' not in source
    assert 'repository.run("status"' not in source


@pytest.mark.parametrize(
    "name",
    (
        "BASH_ENV",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "GIT_CONFIG_COUNT",
        "SBATCH_PARTITION",
    ),
)
def test_slurm_wrapper_rejects_empty_control_variables_before_inputs(name):
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        name: "",
    }
    observed = _run_production_wrapper(environment)
    assert observed.returncode == 2
    assert name in observed.stderr
    assert "set SOURCE_ROOT" not in observed.stderr


def test_slurm_wrapper_rejects_imported_exported_function_before_inputs():
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "BASH_FUNC_carry_recovery_attack%%": "() {  :; }",
    }
    observed = _run_production_wrapper(environment)
    assert observed.returncode == 2
    assert "BASH_FUNC_carry_recovery_attack%%" in observed.stderr
    assert "six reviewed positional controls" not in observed.stderr


def test_slurm_wrapper_rejects_exported_shellopts_before_inputs():
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "SHELLOPTS": "xtrace",
    }
    observed = _run_production_wrapper(environment)
    assert observed.returncode == 2
    assert "exported Bash startup controls are forbidden" in observed.stderr
    assert "six reviewed positional controls" not in observed.stderr


def test_slurm_wrapper_privileged_boundary_blocks_bash_env_before_line_one(tmp_path):
    marker = tmp_path / "bash-env-executed"
    startup = tmp_path / "hostile-bash-env"
    startup.write_text(f"printf owned > {shlex.quote(str(marker))}\n")
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "BASH_ENV": str(startup),
    }
    observed = _run_production_wrapper(environment)
    assert observed.returncode == 2
    assert "BASH_ENV is forbidden before recovery startup" in observed.stderr
    assert not marker.exists()
    assert "six reviewed positional controls" not in observed.stderr


def test_slurm_wrapper_privileged_boundary_blocks_exported_compgen(tmp_path):
    marker = tmp_path / "exported-compgen-executed"
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "BASH_FUNC_compgen%%": (f"() {{ printf owned > {shlex.quote(str(marker))}; }}"),
    }
    observed = _run_production_wrapper(environment)
    assert observed.returncode == 2
    assert "BASH_FUNC_compgen%%" in observed.stderr
    assert not marker.exists()
    assert "six reviewed positional controls" not in observed.stderr
