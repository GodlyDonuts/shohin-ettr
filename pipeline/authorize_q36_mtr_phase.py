#!/usr/bin/env python3
"""Mint the single Q36 phase authorization from read-only admission evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

from build_q36_mtr_custody import _manifest_tree
from capture_q36_mtr_environment import (
    BNB_MANIFEST_SHA256,
    FAST_KERNEL_MANIFEST_SHA256,
)
from capture_q36_mtr_cluster_preflight import SCHEMA as CLUSTER_SCHEMA
from compile_q36_mtr_plan import validate_plan
from pcf1_code_sandbox import validate_sandbox_receipt_payload
from q36_mtr_contract import (
    MIN_FREE_BYTES,
    MIN_FREE_INODES,
    MODEL_REVISION,
    SOURCE_SHA256,
    validate_graph,
)
from q36_mtr_roles import MODEL_CONFIG_SHA256, MODEL_MANIFEST_SHA256

SCHEMA = "shohin-q36-mtr-phase-authorization-v1"
BRANCH = "codex/q36-moe-temporal-revision"
PRIVATE_REMOTE = "https://github.com/GodlyDonuts/shohin-ettr.git"
PUBLIC_REMOTE = "https://github.com/GodlyDonuts/shohin.git"
PUBLIC_PUSH_DISABLED = "DISABLED_PUBLIC_REPO_DO_NOT_PUSH"
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class Q36MTRPhaseAuthorizationError(RuntimeError):
    """The read-only Q36 admission evidence cannot authorize this one phase."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRPhaseAuthorizationError("Q36 admission input differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise Q36MTRPhaseAuthorizationError("Q36 admission schema differs")
    return value


def _run(
    runner: Callable[..., subprocess.CompletedProcess[str]], command: list[str]
) -> str:
    return runner(command, check=True, text=True, capture_output=True).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRPhaseAuthorizationError("Q36 phase authorization exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def authorize(
    args: argparse.Namespace,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    resolved_run_root = args.run_root.resolve(strict=False)
    if (
        not RUN_ID_PATTERN.fullmatch(args.run_id)
        or not args.run_root.is_absolute()
        or args.run_root.exists()
        or args.run_root.is_symlink()
        or resolved_run_root in {Path("/"), Path.home().resolve()}
    ):
        raise Q36MTRPhaseAuthorizationError("Q36 run root already exists")
    if (
        not args.output.is_absolute()
        or args.output.parent.is_symlink()
        or not args.output.parent.is_dir()
        or args.output.resolve(strict=False) == resolved_run_root
    ):
        raise Q36MTRPhaseAuthorizationError("Q36 authorization output differs")
    try:
        args.output.resolve(strict=False).relative_to(resolved_run_root)
    except ValueError:
        pass
    else:
        raise Q36MTRPhaseAuthorizationError("Q36 authorization cannot create run root")
    graph = _load(args.graph_contract, "shohin-q36-mtr-graph-v1")
    plan = _load(args.plan, "shohin-q36-mtr-dry-run-plan-v1")
    validate_graph(graph)
    validate_plan(plan)
    source_commit = graph["source_commit"]
    graph_sha256 = sha256_file(args.graph_contract)
    if (
        plan.get("source_commit") != source_commit
        or plan.get("graph_sha256") != graph_sha256
    ):
        raise Q36MTRPhaseAuthorizationError("Q36 plan/graph differs")
    if args.repository.is_symlink() or not args.repository.is_dir():
        raise Q36MTRPhaseAuthorizationError("Q36 repository admission differs")
    repository = args.repository.resolve(strict=True)
    git = ["git", "-C", str(repository)]
    private_ref = _run(runner, [*git, "ls-remote", "--heads", "origin", BRANCH])
    public_ref = _run(runner, [*git, "ls-remote", "--heads", "public", BRANCH])
    if (
        _run(runner, [*git, "rev-parse", "HEAD"]) != source_commit
        or _run(runner, [*git, "branch", "--show-current"]) != BRANCH
        or _run(runner, [*git, "status", "--porcelain=v1"])
        or _run(runner, [*git, "config", "--get", "remote.origin.url"])
        != PRIVATE_REMOTE
        or _run(runner, [*git, "config", "--get", "remote.public.url"]) != PUBLIC_REMOTE
        or _run(runner, [*git, "config", "--get", "remote.public.pushurl"])
        != PUBLIC_PUSH_DISABLED
        or private_ref.split() != [source_commit, f"refs/heads/{BRANCH}"]
        or public_ref
    ):
        raise Q36MTRPhaseAuthorizationError("Q36 repository admission differs")
    runtime = _manifest_tree(args.runtime_root, args.runtime_manifest)
    model = _manifest_tree(args.model_root, args.model_manifest)
    runtime_receipt = _load(
        args.runtime_root / "runtime.json", "shohin-q36-mtr-runtime-v1"
    )
    environment = _load(args.environment_receipt, "shohin-q36-mtr-environment-v1")
    cluster = _load(args.cluster_preflight, CLUSTER_SCHEMA)
    sandbox = json.loads(args.sandbox_receipt.read_text(encoding="utf-8"))
    try:
        validate_sandbox_receipt_payload(sandbox)
    except Exception as error:
        raise Q36MTRPhaseAuthorizationError("Q36 sandbox admission differs") from error
    source_paths = {
        "pairs": args.pairs,
        "math": args.math,
        "logic_science": args.logic_science,
        "code": args.code,
        "b1": args.b1,
    }
    if any(path.is_symlink() or not path.is_file() for path in source_paths.values()):
        raise Q36MTRPhaseAuthorizationError("Q36 source admission path differs")
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    model_config = args.model_root / "config.json"
    model_revision = args.model_root / "SOURCE_REVISION"
    if (
        model_config.is_symlink()
        or not model_config.is_file()
        or model_revision.is_symlink()
        or not model_revision.is_file()
    ):
        raise Q36MTRPhaseAuthorizationError("Q36 model identity differs")
    if (
        source_hashes != SOURCE_SHA256
        or model.get("manifest_sha256") != MODEL_MANIFEST_SHA256
        or sha256_file(model_config) != MODEL_CONFIG_SHA256
        or model_revision.read_text(encoding="utf-8") != MODEL_REVISION + "\n"
        or runtime_receipt.get("status") != "complete"
        or runtime_receipt.get("source_commit") != source_commit
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("model_config_sha256") != MODEL_CONFIG_SHA256
        or environment.get("runtime_manifest_sha256") != runtime["manifest_sha256"]
        or not _hex(environment.get("environment_tree_sha256"))
        or environment.get("bitsandbytes_overlay", {}).get("manifest_sha256")
        != BNB_MANIFEST_SHA256
        or environment.get("fast_kernel_overlay", {}).get("manifest_sha256")
        != FAST_KERNEL_MANIFEST_SHA256
        or environment.get("scientific_rows_read") != 0
        or cluster.get("status") != "pass"
        or cluster.get("source_commit") != source_commit
        or cluster.get("graph_contract_sha256") != graph_sha256
        or cluster.get("plan_sha256") != sha256_file(args.plan)
        or cluster.get("queue_empty") is not True
        or cluster.get("scientific_rows_read") != 0
        or cluster.get("scientific_jobs_submitted") != 0
        or cluster.get("eligible_h100_node_count", 0) <= 0
        or cluster.get("quota", {}).get("free_bytes", -1) < MIN_FREE_BYTES
        or cluster.get("quota", {}).get("free_inodes", -1) < MIN_FREE_INODES
        or cluster.get("h100_hours_remaining_before_plan", -1)
        < graph["expected_h100_hours"]
        or cluster.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise Q36MTRPhaseAuthorizationError("Q36 phase admission differs")
    payload = {
        "schema": SCHEMA,
        "status": "authorized",
        "run_id": args.run_id,
        "source_commit": source_commit,
        "branch": BRANCH,
        "private_remote": PRIVATE_REMOTE,
        "private_remote_commit": source_commit,
        "public_remote_branch_present": False,
        "public_push_disabled": True,
        "repository_clean": True,
        "graph_contract_sha256": graph_sha256,
        "plan_sha256": sha256_file(args.plan),
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": model["manifest_sha256"],
        "runtime_manifest_sha256": runtime["manifest_sha256"],
        "environment_receipt_sha256": sha256_file(args.environment_receipt),
        "sandbox_receipt_sha256": sha256_file(args.sandbox_receipt),
        "cluster_preflight_sha256": sha256_file(args.cluster_preflight),
        "source_sha256": source_hashes,
        "run_root": str(resolved_run_root),
        "gate": "one_source_disjoint_development_gate",
        "scientific_submit_authorized": True,
        "data_materialization_authorized": True,
        "model_acquisition_authorized": False,
        "automatic_retry": False,
        "automatic_successor": False,
        "automatic_confirmation": False,
        "holdout_access_authorized": False,
        "product_access_authorized": False,
        "public_access_authorized": False,
        "one_output_per_identity": True,
        "stop_after_gate": True,
        "authorization_scope": "exact_frozen_q36_graph_only",
    }
    _atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--graph-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--sandbox-receipt", type=Path, required=True)
    parser.add_argument("--cluster-preflight", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--math", type=Path, required=True)
    parser.add_argument("--logic-science", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = authorize(parser.parse_args())
    print(json.dumps({"status": result["status"], "run_id": result["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
