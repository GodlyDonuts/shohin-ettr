#!/usr/bin/env python3
"""Atomically mirror the exact irreplaceable Q36 evidence into a fresh root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from build_q36_mtr_custody import (
    ACCOUNTING_SCHEMA,
    EVIDENCE_PRECOMPUTE_ARTIFACTS,
    PRECOMPUTE_SCHEMA,
)
from compare_q36_mtr import ARM_SCHEMA
from q36_mtr_contract import MODEL_REVISION, validate_graph
from q36_mtr_evidence import verify_evidence_snapshot
from score_q36_mtr import AUTHORIZATION_SCHEMA, CONSUMPTION_SCHEMA, SCORE_SCHEMA

SCHEMA = "shohin-q36-mtr-evidence-mirror-v1"
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class Q36MTREvidenceError(RuntimeError):
    """Q36 evidence is incomplete, unsafe, or differs at its durable mirror."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTREvidenceError(f"Q36 evidence input is absent or symbolic: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("status")
        not in {"complete", "consumed", "authorized_single_execution"}
    ):
        raise Q36MTREvidenceError(f"Q36 evidence schema/status differs: {path}")
    return value


def _bindings(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, rendered = value.partition("=")
        path = Path(rendered)
        if (
            not separator
            or not NAME_PATTERN.fullmatch(name)
            or name in result
            or not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
        ):
            raise Q36MTREvidenceError("Q36 evidence binding differs")
        result[name] = path.resolve(strict=True)
    return result


def _safe_output_root(authorized_root: Path, output_root: Path) -> tuple[Path, Path]:
    if (
        not authorized_root.is_absolute()
        or authorized_root.is_symlink()
        or not authorized_root.is_dir()
        or not output_root.is_absolute()
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise Q36MTREvidenceError("Q36 authorized evidence root differs")
    resolved_authorized = authorized_root.resolve(strict=True)
    resolved_output = output_root.resolve(strict=False)
    if resolved_authorized in {Path("/"), Path.home().resolve()}:
        raise Q36MTREvidenceError("Q36 authorized evidence root is too broad")
    try:
        relative = resolved_output.relative_to(resolved_authorized)
    except ValueError as error:
        raise Q36MTREvidenceError("Q36 mirror escapes its authorized root") from error
    if not relative.parts:
        raise Q36MTREvidenceError("Q36 mirror equals its authorized root")
    return resolved_authorized, resolved_output


def mirror(args: argparse.Namespace) -> dict[str, Any]:
    _, output = _safe_output_root(args.authorized_root, args.output_root)
    graph = _load(args.graph_contract, "shohin-q36-mtr-graph-v1")
    validate_graph(graph)
    precompute = _load(args.precompute_custody, PRECOMPUTE_SCHEMA)
    prescore = _load(args.prescore_accounting, ACCOUNTING_SCHEMA)
    accounting = _load(args.scheduler_accounting, ACCOUNTING_SCHEMA)
    authorization = _load(args.score_authorization, AUTHORIZATION_SCHEMA)
    consumption = _load(args.score_consumption, CONSUMPTION_SCHEMA)
    score = _load(args.score_report, SCORE_SCHEMA)
    precompute_artifacts = _bindings(args.precompute_artifact)
    arm_reports = _bindings(args.arm_report)
    expected_arms = {
        "learned_commit",
        "trained_revision",
        "unchanged",
        "self_refinement",
        "draft_hidden",
    }
    if (
        set(precompute_artifacts) != EVIDENCE_PRECOMPUTE_ARTIFACTS
        or set(arm_reports) != expected_arms
    ):
        raise Q36MTREvidenceError("Q36 mirrored evidence set differs")
    graph_sha256 = sha256_file(args.graph_contract)
    precompute_sha256 = sha256_file(args.precompute_custody)
    score_sha256 = sha256_file(args.score_report)
    consumption_sha256 = sha256_file(args.score_consumption)
    for name, path in arm_reports.items():
        report = _load(path, ARM_SCHEMA)
        if (
            report.get("arm") != name
            or report.get("run_id") != args.run_id
            or report.get("score_report_sha256") != score_sha256
            or report.get("precompute_custody_sha256") != precompute_sha256
        ):
            raise Q36MTREvidenceError("Q36 normalized arm evidence differs")
    for name, path in precompute_artifacts.items():
        if precompute.get("artifact_sha256s", {}).get(name) != sha256_file(path):
            raise Q36MTREvidenceError(f"Q36 precompute mirror hash differs: {name}")
    if (
        precompute.get("run_id") != args.run_id
        or precompute.get("source_commit") != args.source_commit
        or precompute.get("graph_contract_sha256") != graph_sha256
        or precompute.get("model_revision") != MODEL_REVISION
        or prescore.get("phase") != "prescore"
        or prescore.get("run_id") != args.run_id
        or accounting.get("phase") != "final"
        or accounting.get("run_id") != args.run_id
        or accounting.get("graph_contract_sha256") != graph_sha256
        or accounting.get("plan_sha256") != sha256_file(args.plan)
        or accounting.get("dispatch_receipt_sha256")
        != sha256_file(args.dispatch_receipt)
        or authorization.get("run_id") != args.run_id
        or score.get("run_id") != args.run_id
        or score.get("score_authorization_sha256")
        != sha256_file(args.score_authorization)
        or score.get("score_consumption_sha256") != consumption_sha256
        or score.get("outcomes_sha256") != sha256_file(args.score_outcomes)
        or score.get("sandbox_receipt_sha256")
        != sha256_file(args.score_sandbox_receipt)
        or score.get("input_hashes", {}).get("prescore_accounting_sha256")
        != sha256_file(args.prescore_accounting)
        or consumption.get("run_id") != args.run_id
        or consumption.get("authorization_sha256")
        != sha256_file(args.score_authorization)
        or sha256_file(args.model_manifest) != precompute.get("model_manifest_sha256")
        or sha256_file(args.runtime_manifest)
        != precompute.get("runtime_manifest_sha256")
    ):
        raise Q36MTREvidenceError("Q36 evidence lineage differs")
    primary = {
        "graph_contract": args.graph_contract,
        "precompute_custody": args.precompute_custody,
        "prescore_accounting": args.prescore_accounting,
        "score_authorization": args.score_authorization,
        "score_consumption": args.score_consumption,
        "score_report": args.score_report,
        "score_outcomes": args.score_outcomes,
        "score_sandbox_receipt": args.score_sandbox_receipt,
        "scheduler_accounting": args.scheduler_accounting,
        "plan": args.plan,
        "dispatch_receipt": args.dispatch_receipt,
        "model_manifest": args.model_manifest,
        "runtime_manifest": args.runtime_manifest,
        **{f"arm_{name}": path for name, path in arm_reports.items()},
        **{f"precompute_{name}": path for name, path in precompute_artifacts.items()},
    }
    artifact_sha256s = {
        name: sha256_file(path) for name, path in sorted(primary.items())
    }
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.mkdir(parents=True, mode=0o700)
    records = []
    try:
        artifact_root = temporary / "artifacts"
        artifact_root.mkdir(mode=0o700)
        for name, source in sorted(primary.items()):
            suffix = "".join(source.suffixes)
            relative = Path("artifacts") / f"{name}{suffix}"
            destination = temporary / relative
            with source.open("rb") as input_handle, destination.open(
                "xb"
            ) as output_handle:
                shutil.copyfileobj(input_handle, output_handle, 1 << 20)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            digest = sha256_file(destination)
            if digest != artifact_sha256s[name]:
                raise Q36MTREvidenceError("Q36 mirror byte verification differs")
            os.chmod(destination, 0o444)
            records.append(
                {
                    "name": name,
                    "primary": str(source.resolve()),
                    "mirror": str((output / relative).resolve()),
                    "sha256": digest,
                    "bytes": destination.stat().st_size,
                }
            )
        payload = {
            "schema": SCHEMA,
            "status": "complete",
            "verified": True,
            "run_id": args.run_id,
            "source_commit": args.source_commit,
            "graph_contract_sha256": graph_sha256,
            "model_revision": MODEL_REVISION,
            "artifact_sha256s": artifact_sha256s,
            "artifact_count": len(records),
            "records": records,
            "primary_mirror_hashes_exact": True,
            "write_once_snapshot": True,
            "assessor_board_copied_or_opened": False,
            "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        }
        tree_rows = [
            {"name": row["name"], "sha256": row["sha256"], "bytes": row["bytes"]}
            for row in sorted(records, key=lambda value: value["name"])
        ]
        payload["artifact_tree_sha256"] = hashlib.sha256(
            b"".join(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for row in tree_rows
            )
        ).hexdigest()
        manifest = temporary / "manifest.json"
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with manifest.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(manifest, 0o444)
        os.chmod(artifact_root, 0o555)
        os.chmod(temporary, 0o555)
        for directory in (artifact_root, temporary):
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        os.replace(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verification = verify_evidence_snapshot(output / "manifest.json", payload)
    if verification["artifact_tree_sha256"] != payload["artifact_tree_sha256"]:
        raise Q36MTREvidenceError("Q36 durable mirror verification differs")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--graph-contract", type=Path, required=True)
    parser.add_argument("--precompute-custody", type=Path, required=True)
    parser.add_argument("--prescore-accounting", type=Path, required=True)
    parser.add_argument("--score-authorization", type=Path, required=True)
    parser.add_argument("--score-consumption", type=Path, required=True)
    parser.add_argument("--score-report", type=Path, required=True)
    parser.add_argument("--score-outcomes", type=Path, required=True)
    parser.add_argument("--score-sandbox-receipt", type=Path, required=True)
    parser.add_argument("--scheduler-accounting", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dispatch-receipt", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--arm-report", action="append", default=[])
    parser.add_argument("--precompute-artifact", action="append", default=[])
    parser.add_argument("--authorized-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    result = mirror(parser.parse_args())
    print(
        json.dumps({"status": result["status"], "artifacts": result["artifact_count"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
