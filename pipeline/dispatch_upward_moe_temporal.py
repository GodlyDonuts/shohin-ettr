#!/usr/bin/env python3
"""Prestage one exact host-owned upward-MoE temporal measurement graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

SCHEMA = "shohin-upward-moe-temporal-dispatch-v1"
PATH_VALUE = "/apps/slurm/current/bin:/usr/bin:/bin"
ARMS = ("unchanged", "self_refinement", "owner", "aligned_revision", "temporal_gate")


class UpwardMoETemporalDispatchError(RuntimeError):
    """The upward-MoE temporal dependency graph could not be staged exactly."""


def _atom(name: str, value: str) -> None:
    if (
        not re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
        or not value
        or any(character in value for character in ",\n\r\t")
    ):
        raise UpwardMoETemporalDispatchError("upward Slurm export differs")


def _exports(values: dict[str, str]) -> str:
    for name, value in values.items():
        _atom(name, value)
    return ",".join(f"{name}={value}" for name, value in sorted(values.items()))


def _require_input(path: Path, *, directory: bool = False) -> None:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or (not path.is_dir() if directory else not path.is_file())
    ):
        raise UpwardMoETemporalDispatchError(f"upward input differs: {path}")


def _require_python(path: Path) -> None:
    if not path.is_absolute() or not path.exists() or not path.resolve().is_file():
        raise UpwardMoETemporalDispatchError("upward Python entrypoint differs")


def _host_exports(args: argparse.Namespace) -> dict[str, str]:
    if args.host == "nemotron-super":
        for path in (args.overlay_root, args.causal_conv_root):
            if not isinstance(path, Path):
                raise UpwardMoETemporalDispatchError("Nemotron host input is absent")
            _require_input(path, directory=True)
        if not isinstance(args.overlay_manifest, Path):
            raise UpwardMoETemporalDispatchError("Nemotron overlay manifest is absent")
        _require_input(args.overlay_manifest)
        return {
            "OVERLAY_ROOT": str(args.overlay_root),
            "OVERLAY_MANIFEST": str(args.overlay_manifest),
            "CAUSAL_CONV_ROOT": str(args.causal_conv_root),
        }
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_model_manifest_sha256 or ""):
        raise UpwardMoETemporalDispatchError("Mixtral model manifest hash is absent")
    return {"EXPECTED_MODEL_MANIFEST_SHA256": args.expected_model_manifest_sha256}


def _common(args: argparse.Namespace) -> dict[str, str]:
    return {
        "PATH": PATH_VALUE,
        "HOST": args.host,
        "PYTHON": str(args.python),
        "RUNTIME": str(args.runtime),
        "RUNTIME_MANIFEST_SHA256": args.runtime_manifest_sha256,
        "MODEL_ROOT": str(args.model_root),
        "MODEL_MANIFEST": str(args.model_manifest),
        "MECHANICS_REPORT": str(args.mechanics_report),
        **_host_exports(args),
    }


def build_graph(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.run_root
    owner = root / "owner"
    drafts = root / "drafts"
    merged = root / "merged"
    data = root / "data"
    aligned = root / "aligned"
    temporal = root / "temporal_gate"
    evaluations = root / "evaluations"
    common = _common(args)
    stages: list[dict[str, Any]] = [
        {
            "name": "owner",
            "script": "train/jobs/upward_moe_train_owner.sbatch",
            "dependencies": [],
            "exports": {
                **common,
                "DATA": str(args.b1),
                "OUTPUT": str(owner),
            },
        },
        {
            "name": "drafts",
            "script": "train/jobs/upward_moe_generate_drafts.sbatch",
            "dependencies": ["owner"],
            "array": "0-15%16",
            "exports": {
                **common,
                "TRAIN_SOURCE": str(args.train_source),
                "DEVELOPMENT_SOURCE": str(args.development_source),
                "FREEZE_REPORT": str(args.freeze_report),
                "OWNER_CHECKPOINT": str(owner / "checkpoint_0000256.pt"),
                "SHARD_ROOT": str(drafts),
            },
        },
        {
            "name": "merge",
            "script": "pipeline/jobs/merge_upward_moe_drafts.sbatch",
            "dependencies": ["drafts"],
            "exports": {
                **common,
                "TRAIN_SOURCE": str(args.train_source),
                "DEVELOPMENT_SOURCE": str(args.development_source),
                "FREEZE_REPORT": str(args.freeze_report),
                "SHARD_ROOT": str(drafts),
                "OUTPUT": str(merged / "drafts.jsonl"),
                "REPORT": str(merged / "report.json"),
            },
        },
        {
            "name": "materialize",
            "script": "pipeline/jobs/materialize_upward_moe_data.sbatch",
            "dependencies": ["merge"],
            "exports": {
                **common,
                "SOURCE_ROOT": str(args.source_root),
                "DRAFTS": str(merged / "drafts.jsonl"),
                "DRAFT_REPORT": str(merged / "report.json"),
                "ASSESSOR_RECEIPT": str(args.assessor_receipt),
                "OUTPUT": str(data),
            },
        },
        {
            "name": "aligned",
            "script": "train/jobs/upward_moe_train_aligned.sbatch",
            "dependencies": ["materialize"],
            "exports": {
                **common,
                "DATA": str(data / "revision_train.jsonl"),
                "DATA_REPORT": str(data / "report.json"),
                "OWNER_CHECKPOINT": str(owner / "checkpoint_0000256.pt"),
                "OUTPUT": str(aligned),
            },
        },
        {
            "name": "temporal",
            "script": "train/jobs/upward_moe_train_temporal_gate.sbatch",
            "dependencies": ["aligned"],
            "exports": {
                **common,
                "DATA": str(data / "revision_train.jsonl"),
                "DATA_REPORT": str(data / "report.json"),
                "OWNER_CHECKPOINT": str(owner / "checkpoint_0000256.pt"),
                "REVISION_CHECKPOINT": str(aligned / "checkpoint_0000256.pt"),
                "OUTPUT": str(temporal),
            },
        },
    ]
    for arm in ARMS:
        stages.append(
            {
                "name": f"evaluate_{arm}",
                "script": "train/jobs/upward_moe_evaluate_temporal_gate.sbatch",
                "dependencies": ["temporal" if arm == "temporal_gate" else "aligned"],
                "array": "0-15%16",
                "exports": {
                    **common,
                    "ARM": arm,
                    "DATA": str(data / "development_eval.jsonl"),
                    "DATA_REPORT": str(data / "report.json"),
                    "OWNER_CHECKPOINT": str(owner / "checkpoint_0000256.pt"),
                    "REVISION_CHECKPOINT": str(aligned / "checkpoint_0000256.pt"),
                    "GATE_CHECKPOINT": str(temporal / "checkpoint_0000256.pt"),
                    "OUTPUT_ROOT": str(evaluations),
                },
            }
        )
    stages.append(
        {
            "name": "score",
            "script": "pipeline/jobs/score_upward_moe_temporal_gate.sbatch",
            "dependencies": [f"evaluate_{arm}" for arm in ARMS],
            "exports": {
                **common,
                "ASSESSORS": str(args.assessors),
                "EVALUATION_ROOT": str(evaluations),
                "OUTPUT": str(root / "score.json"),
                "SANDBOX_RECEIPT": str(root / "score_sandbox.json"),
            },
        }
    )
    return stages


def validate(args: argparse.Namespace, stages: list[dict[str, Any]]) -> None:
    if (
        args.run_root.exists()
        or args.run_root.is_symlink()
        or args.receipt.exists()
        or args.receipt.is_symlink()
    ):
        raise UpwardMoETemporalDispatchError("upward run root exists")
    if not args.run_root.is_absolute() or not args.receipt.is_absolute():
        raise UpwardMoETemporalDispatchError("upward output path differs")
    _require_input(args.runtime, directory=True)
    _require_input(args.model_root, directory=True)
    _require_input(args.source_root, directory=True)
    _require_python(args.python)
    for path in (
        args.model_manifest,
        args.mechanics_report,
        args.b1,
        args.train_source,
        args.development_source,
        args.freeze_report,
        args.assessor_receipt,
        args.assessors,
    ):
        _require_input(path)
    if not os.access(args.python, os.X_OK) or not re.fullmatch(
        r"[0-9a-f]{64}", args.runtime_manifest_sha256
    ):
        raise UpwardMoETemporalDispatchError("upward runtime identity differs")
    names = [stage["name"] for stage in stages]
    if len(stages) != 12 or len(names) != len(set(names)):
        raise UpwardMoETemporalDispatchError("upward graph cardinality differs")
    seen: set[str] = set()
    for stage in stages:
        if any(dependency not in seen for dependency in stage["dependencies"]):
            raise UpwardMoETemporalDispatchError("upward dependency order differs")
        script = args.runtime / stage["script"]
        _require_input(script)
        _exports(stage["exports"])
        seen.add(stage["name"])
    if sum(16 if "array" in stage else 1 for stage in stages) != 102:
        raise UpwardMoETemporalDispatchError("upward allocation geometry differs")


def submit(args: argparse.Namespace, stages: list[dict[str, Any]]) -> dict[str, Any]:
    job_ids: dict[str, str] = {}
    commands = []
    clean_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "SBATCH_"))
    }
    for index, stage in enumerate(stages, start=1):
        command = ["sbatch", "--parsable"]
        if stage.get("array"):
            command.append(f"--array={stage['array']}")
        if stage["dependencies"]:
            dependency_ids = [job_ids[name] for name in stage["dependencies"]]
            command.append("--dependency=afterok:" + ":".join(dependency_ids))
        command.append("--export=" + _exports(stage["exports"]))
        command.append(str(args.runtime / stage["script"]))
        commands.append(command)
        if args.submit:
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    text=True,
                    capture_output=True,
                    env=clean_environment,
                )
            except subprocess.CalledProcessError as error:
                if job_ids:
                    subprocess.run(
                        ["scancel", *job_ids.values()],
                        check=False,
                        text=True,
                        capture_output=True,
                        env=clean_environment,
                    )
                raise UpwardMoETemporalDispatchError(
                    f"upward submission failed at {stage['name']}; predecessors cancelled"
                ) from error
            job_id = result.stdout.strip().split(";", 1)[0]
            if not re.fullmatch(r"[0-9]+", job_id):
                raise UpwardMoETemporalDispatchError("upward sbatch receipt differs")
        else:
            job_id = f"DRY{index:03d}"
        job_ids[stage["name"]] = job_id
    return {
        "schema": SCHEMA,
        "status": "submitted" if args.submit else "dry_run",
        "host": args.host,
        "run_root": str(args.run_root),
        "submission_roots": len(stages),
        "allocation_tasks": 102,
        "two_h100_tasks": 99,
        "cpu_tasks": 3,
        "draft_shards": 16,
        "evaluation_arms": list(ARMS),
        "evaluation_shards_per_arm": 16,
        "job_ids": job_ids,
        "commands": commands,
        "no_requeue": True,
        "duplicate_outputs_authorized": False,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", choices=("nemotron-super", "mixtral-8x22b"), required=True
    )
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256")
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--overlay-manifest", type=Path)
    parser.add_argument("--causal-conv-root", type=Path)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--assessor-receipt", type=Path, required=True)
    parser.add_argument("--assessors", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    graph = build_graph(parsed)
    validate(parsed, graph)
    receipt = submit(parsed, graph)
    if parsed.submit:
        _atomic_json(parsed.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
