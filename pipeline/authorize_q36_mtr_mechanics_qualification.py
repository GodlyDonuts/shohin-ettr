#!/usr/bin/env python3
"""Build or verify one no-score Q36 mechanics qualification authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from build_q36_mtr_custody import Q36MTRCustodyError, _manifest_tree
from q36_mtr_roles import (
    MODEL_CONFIG_SHA256,
    MODEL_MANIFEST_SHA256,
    MODEL_REVISION,
)

SCHEMA = "shohin-q36-mtr-mechanics-qualification-authorization-v1"
TERMINAL_SCHEMA = "shohin-q36-mtr-execution-terminal-infrastructure-v1"
B1_SHA256 = "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549"
PRIOR_TERMINAL_SHA256S = {
    "c8a46b44817c8d22ed0cc404f84161ef08795741cf07abb8963e4f841bd3636a",
    "dd5dc465e80635d88afde12b293607ee4ddcc210f271e4f9a8f7fb6836cbc16a",
}
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class Q36MTRMechanicsQualificationError(RuntimeError):
    """The mechanics-only authorization or its immutable inputs differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRMechanicsQualificationError("Q36 qualification input differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q36MTRMechanicsQualificationError(
            "Q36 qualification input is unreadable"
        ) from error
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise Q36MTRMechanicsQualificationError("Q36 qualification schema differs")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRMechanicsQualificationError("Q36 qualification exists")
    if not path.is_absolute() or path.parent.is_symlink() or not path.parent.is_dir():
        raise Q36MTRMechanicsQualificationError("Q36 qualification destination differs")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _safe_fresh_output(path: Path) -> Path:
    if (
        not path.is_absolute()
        or path.exists()
        or path.is_symlink()
        or path.resolve(strict=False) in {Path("/"), Path.home().resolve()}
    ):
        raise Q36MTRMechanicsQualificationError("Q36 qualification output root differs")
    return path.resolve(strict=False)


def _validate_terminal(value: dict[str, Any]) -> None:
    if (
        value.get("status") != "terminal_infrastructure_failure"
        or value.get("formal_scientific_result") is not None
        or value.get("capability_rows_scored") != 0
        or value.get("development_assessor_reads") != 0
        or value.get("sealed_holdout_accesses") != 0
        or value.get("protected_product_accesses") != 0
        or value.get("public_accesses") != 0
        or value.get("automatic_retry_authorized") is not False
        or value.get("automatic_successor_authorized") is not False
        or value.get("stop_after_terminal") is not True
        or value.get("infrastructure_diagnosis", {}).get("scientific_gate_entered")
        is not False
    ):
        raise Q36MTRMechanicsQualificationError("Q36 prior terminal boundary differs")


def authorize(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _safe_fresh_output(args.output_root)
    if (
        not RUN_ID_PATTERN.fullmatch(args.run_id)
        or len(args.source_commit) != 40
        or any(character not in "0123456789abcdef" for character in args.source_commit)
    ):
        raise Q36MTRMechanicsQualificationError("Q36 qualification identity differs")
    try:
        runtime = _manifest_tree(args.runtime_root, args.runtime_manifest)
        model = _manifest_tree(args.model_root, args.model_manifest)
    except Q36MTRCustodyError as error:
        raise Q36MTRMechanicsQualificationError(str(error)) from error
    runtime_receipt = _load(
        args.runtime_root / "runtime.json", "shohin-q36-mtr-runtime-v1"
    )
    environment = _load(args.environment_receipt, "shohin-q36-mtr-environment-v1")
    terminal_receipts = []
    for path in args.prior_terminal:
        value = _load(path, TERMINAL_SCHEMA)
        _validate_terminal(value)
        terminal_receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "run_id": value.get("run_id"),
                "source_commit": value.get("source_commit"),
                "diagnosis": value.get("infrastructure_diagnosis", {}).get("class"),
            }
        )
    terminal_hashes = {receipt["sha256"] for receipt in terminal_receipts}
    config = args.model_root / "config.json"
    revision = args.model_root / "SOURCE_REVISION"
    mechanics_source = args.runtime_root / "train/hf_q36_mtr_mechanics.py"
    wrapper_source = (
        args.runtime_root / "train/jobs/q36_mtr_mechanics_qualification.sbatch"
    )
    if (
        terminal_hashes != PRIOR_TERMINAL_SHA256S
        or len(terminal_receipts) != 2
        or args.b1.is_symlink()
        or not args.b1.is_file()
        or sha256_file(args.b1) != B1_SHA256
        or config.is_symlink()
        or not config.is_file()
        or revision.is_symlink()
        or not revision.is_file()
        or mechanics_source.is_symlink()
        or not mechanics_source.is_file()
        or wrapper_source.is_symlink()
        or not wrapper_source.is_file()
        or model.get("manifest_sha256") != MODEL_MANIFEST_SHA256
        or sha256_file(config) != MODEL_CONFIG_SHA256
        or revision.read_text(encoding="utf-8") != MODEL_REVISION + "\n"
        or runtime_receipt.get("status") != "complete"
        or runtime_receipt.get("source_commit") != args.source_commit
        or runtime_receipt.get("model_acquisition_capability") is not False
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("model_config_sha256") != MODEL_CONFIG_SHA256
        or environment.get("runtime_manifest_sha256") != runtime["manifest_sha256"]
        or environment.get("scientific_rows_read") != 0
    ):
        raise Q36MTRMechanicsQualificationError("Q36 qualification custody differs")
    payload = {
        "schema": SCHEMA,
        "status": "authorized",
        "run_id": args.run_id,
        "source_commit": args.source_commit,
        "runtime_root": str(args.runtime_root.resolve()),
        "runtime_manifest_sha256": runtime["manifest_sha256"],
        "mechanics_source_sha256": sha256_file(mechanics_source),
        "wrapper_source_sha256": sha256_file(wrapper_source),
        "model_root": str(args.model_root.resolve()),
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": model["manifest_sha256"],
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "environment_receipt": str(args.environment_receipt.resolve()),
        "environment_receipt_sha256": sha256_file(args.environment_receipt),
        "environment_tree_sha256": environment.get("environment_tree_sha256"),
        "b1": str(args.b1.resolve()),
        "b1_sha256": B1_SHA256,
        "prior_terminal_receipts": sorted(
            terminal_receipts, key=lambda receipt: receipt["sha256"]
        ),
        "output_root": str(output_root),
        "partition": "normal",
        "excluded_nodes": [
            "evc26",
            "evc29",
            "evc31",
            "evc32",
            "evc33",
            "evc37",
            "evc38",
            "evc46",
        ],
        "h100_allocations_authorized": 1,
        "mechanics_rows": 24,
        "mechanics_seed": 2026080825,
        "mechanics_data_seed": 2026080824,
        "no_requeue": True,
        "one_shot": True,
        "mechanics_qualification_authorized": True,
        "scientific_graph_authorized": False,
        "capability_scoring_authorized": False,
        "assessor_access_authorized": False,
        "holdout_access_authorized": False,
        "product_access_authorized": False,
        "public_access_authorized": False,
        "automatic_retry_authorized": False,
        "automatic_successor_authorized": False,
        "submission_capability": False,
        "next_action_on_pass": "stop_and_preserve_qualification",
        "next_action_on_failure": "stop_and_preserve_terminal_infrastructure_evidence",
    }
    _atomic_json(args.output, payload)
    return payload


def verify_authorization(
    path: Path,
    expected_sha256: str,
    source_commit: str,
    run_id: str,
    output_root: Path,
    runtime_manifest: Path,
    model_manifest: Path,
    environment_receipt: Path,
    b1: Path,
    prior_terminals: list[Path],
) -> dict[str, Any]:
    value = _load(path, SCHEMA)
    expected_output = _safe_fresh_output(output_root)
    custody_files = (
        runtime_manifest,
        model_manifest,
        environment_receipt,
        b1,
        *prior_terminals,
    )
    if any(
        candidate.is_symlink() or not candidate.is_file() for candidate in custody_files
    ):
        raise Q36MTRMechanicsQualificationError(
            "Q36 mechanics qualification custody input differs"
        )
    terminal_receipts = []
    for terminal in prior_terminals:
        terminal_value = _load(terminal, TERMINAL_SCHEMA)
        _validate_terminal(terminal_value)
        terminal_receipts.append(
            {
                "path": str(terminal.resolve()),
                "sha256": sha256_file(terminal),
                "run_id": terminal_value.get("run_id"),
                "source_commit": terminal_value.get("source_commit"),
                "diagnosis": terminal_value.get("infrastructure_diagnosis", {}).get(
                    "class"
                ),
            }
        )
    terminal_receipts.sort(key=lambda receipt: receipt["sha256"])
    runtime_root = Path(str(value.get("runtime_root", "")))
    model_root = Path(str(value.get("model_root", "")))
    if (
        sha256_file(path) != expected_sha256
        or path.stat().st_mode & 0o222
        or value.get("status") != "authorized"
        or value.get("source_commit") != source_commit
        or value.get("run_id") != run_id
        or value.get("output_root") != str(expected_output)
        or value.get("h100_allocations_authorized") != 1
        or value.get("mechanics_rows") != 24
        or value.get("mechanics_seed") != 2026080825
        or value.get("mechanics_data_seed") != 2026080824
        or value.get("no_requeue") is not True
        or value.get("one_shot") is not True
        or value.get("mechanics_qualification_authorized") is not True
        or value.get("scientific_graph_authorized") is not False
        or value.get("capability_scoring_authorized") is not False
        or value.get("assessor_access_authorized") is not False
        or value.get("holdout_access_authorized") is not False
        or value.get("product_access_authorized") is not False
        or value.get("public_access_authorized") is not False
        or value.get("automatic_retry_authorized") is not False
        or value.get("automatic_successor_authorized") is not False
        or value.get("submission_capability") is not False
        or len(prior_terminals) != 2
        or {receipt["sha256"] for receipt in terminal_receipts}
        != PRIOR_TERMINAL_SHA256S
        or terminal_receipts != value.get("prior_terminal_receipts")
        or runtime_manifest.resolve() != (runtime_root / "SHA256SUMS").resolve()
        or sha256_file(runtime_manifest) != value.get("runtime_manifest_sha256")
        or model_manifest.resolve() != (model_root / "SHA256SUMS").resolve()
        or sha256_file(model_manifest) != value.get("model_manifest_sha256")
        or environment_receipt.resolve()
        != Path(str(value.get("environment_receipt", ""))).resolve()
        or sha256_file(environment_receipt) != value.get("environment_receipt_sha256")
        or b1.resolve() != Path(str(value.get("b1", ""))).resolve()
        or sha256_file(b1) != value.get("b1_sha256")
    ):
        raise Q36MTRMechanicsQualificationError(
            "Q36 mechanics qualification authorization differs"
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--run-id", required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--runtime-root", type=Path, required=True)
    build.add_argument("--runtime-manifest", type=Path, required=True)
    build.add_argument("--model-root", type=Path, required=True)
    build.add_argument("--model-manifest", type=Path, required=True)
    build.add_argument("--environment-receipt", type=Path, required=True)
    build.add_argument("--b1", type=Path, required=True)
    build.add_argument("--prior-terminal", type=Path, action="append", required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--authorization", type=Path, required=True)
    verify.add_argument("--authorization-sha256", required=True)
    verify.add_argument("--source-commit", required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument("--runtime-manifest", type=Path, required=True)
    verify.add_argument("--model-manifest", type=Path, required=True)
    verify.add_argument("--environment-receipt", type=Path, required=True)
    verify.add_argument("--b1", type=Path, required=True)
    verify.add_argument("--prior-terminal", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = authorize(args)
    else:
        result = verify_authorization(
            args.authorization,
            args.authorization_sha256,
            args.source_commit,
            args.run_id,
            args.output_root,
            args.runtime_manifest,
            args.model_manifest,
            args.environment_receipt,
            args.b1,
            args.prior_terminal,
        )
    print(json.dumps({"run_id": result["run_id"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
