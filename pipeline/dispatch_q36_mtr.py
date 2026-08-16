#!/usr/bin/env python3
"""Preflight or submit exactly one frozen Q36-MTR Slurm dependency graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from compile_q36_mtr_plan import validate_plan
from q36_mtr_contract import (
    EXCLUDED_NODES,
    SOURCE_FREEZE_SHA256,
    STAGES,
    validate_graph,
)

SCHEMA = "shohin-q36-mtr-dispatch-v1"
ACK = "ONE_FROZEN_DEVELOPMENT_GATE_ONLY"
PATH_VALUE = "/apps/slurm/current/bin:/usr/bin:/bin"
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}")

SCRIPTS = {
    "preflight_cpu": "pipeline/jobs/q36_mtr_live_preflight.sbatch",
    "mechanics": "train/jobs/q36_mtr_mechanics.sbatch",
    "owner_fit": "train/jobs/q36_mtr_train_role.sbatch",
    "draft_generate": "train/jobs/q36_mtr_generate_drafts.sbatch",
    "draft_merge": "pipeline/jobs/q36_mtr_merge_drafts.sbatch",
    "materialize": "pipeline/jobs/q36_mtr_materialize.sbatch",
    "aligned_fit": "train/jobs/q36_mtr_train_role.sbatch",
    "draft_hidden_fit": "train/jobs/q36_mtr_train_role.sbatch",
    "calibration_revision": "train/jobs/q36_mtr_evaluate.sbatch",
    "calibration_unchanged": "train/jobs/q36_mtr_evaluate.sbatch",
    "calibration_revision_merge": "pipeline/jobs/q36_mtr_merge_evaluation.sbatch",
    "calibration_unchanged_merge": "pipeline/jobs/q36_mtr_merge_evaluation.sbatch",
    "commit_pairs": "pipeline/jobs/q36_mtr_commit_pairs.sbatch",
    "development_revision": "train/jobs/q36_mtr_evaluate.sbatch",
    "development_unchanged": "train/jobs/q36_mtr_evaluate.sbatch",
    "development_self_refinement": "train/jobs/q36_mtr_evaluate.sbatch",
    "development_draft_hidden": "train/jobs/q36_mtr_evaluate.sbatch",
    "development_revision_merge": "pipeline/jobs/q36_mtr_merge_evaluation.sbatch",
    "development_unchanged_merge": "pipeline/jobs/q36_mtr_merge_evaluation.sbatch",
    "development_self_refinement_merge": "pipeline/jobs/q36_mtr_merge_evaluation.sbatch",
    "development_draft_hidden_merge": "pipeline/jobs/q36_mtr_merge_evaluation.sbatch",
    "development_commit_pairs": "pipeline/jobs/q36_mtr_commit_pairs.sbatch",
    "commit_fit": "train/jobs/q36_mtr_train_commit.sbatch",
    "commit_apply": "pipeline/jobs/q36_mtr_validate_commit_application.sbatch",
    "precompute_custody": "pipeline/jobs/q36_mtr_build_precompute_custody.sbatch",
    "prescore_accounting": "pipeline/jobs/q36_mtr_capture_accounting.sbatch",
    "authorize_score": "pipeline/jobs/q36_mtr_authorize_score.sbatch",
    "score_once": "pipeline/jobs/q36_mtr_score.sbatch",
    "normalize": "pipeline/jobs/q36_mtr_normalize.sbatch",
    "final_accounting": "pipeline/jobs/q36_mtr_capture_accounting.sbatch",
    "evidence_mirror": "pipeline/jobs/q36_mtr_mirror_evidence.sbatch",
    "compute_custody": "pipeline/jobs/q36_mtr_build_final_custody.sbatch",
    "final_compare": "pipeline/jobs/q36_mtr_compare.sbatch",
}


class Q36MTRDispatchError(RuntimeError):
    """The exactly-once Q36 graph cannot be safely dispatched."""


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _load(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise Q36MTRDispatchError(f"Q36 dispatch input differs: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q36MTRDispatchError(
            f"Q36 dispatch input is unreadable: {path}"
        ) from error
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise Q36MTRDispatchError(f"Q36 dispatch schema differs: {path}")
    return value


def _validate_atom(name: str, value: str) -> None:
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or not value:
        raise Q36MTRDispatchError("Q36 Slurm export atom differs")
    if any(character in value for character in ",\n\r\t"):
        raise Q36MTRDispatchError(f"Q36 Slurm export delimiter differs: {name}")


def _render_exports(values: dict[str, str]) -> str:
    for name, value in values.items():
        _validate_atom(name, value)
    return ",".join(f"{name}={value}" for name, value in sorted(values.items()))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRDispatchError("Q36 dispatch receipt exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o444)
    path.parent.chmod(0o555)


def _common(args: argparse.Namespace, environment: dict[str, Any]) -> dict[str, str]:
    return {
        "PATH": PATH_VALUE,
        "RUNTIME": str(args.runtime),
        "RUNTIME_MANIFEST_SHA256": args.runtime_manifest_sha256,
        "SOURCE_COMMIT": args.source_commit,
        "PYTHON": str(args.python),
        "RUN_ID": args.run_id,
        "PHASE_AUTHORIZATION": str(args.phase_authorization),
        "PHASE_AUTHORIZATION_SHA256": args.phase_authorization_sha256,
        "MODEL_ROOT": str(args.model_root),
        "MODEL_MANIFEST": str(args.model_manifest),
        "MODEL_MANIFEST_SHA256": args.model_manifest_sha256,
        "MODEL_REVISION": args.model_revision,
        "MODEL_CONFIG_SHA256": args.model_config_sha256,
        "ENVIRONMENT_RECEIPT": str(args.environment_receipt),
        "ENVIRONMENT_RECEIPT_SHA256": sha256_file(args.environment_receipt),
        "ENVIRONMENT_TREE_SHA256": str(environment["environment_tree_sha256"]),
    }


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.run_root
    return {
        "live": root / "preflight/live.json",
        "mechanics": root / "mechanics",
        "owner": root / "roles/owner",
        "aligned": root / "roles/aligned",
        "hidden": root / "roles/draft_hidden",
        "drafts": root / "drafts",
        "merged": root / "merged",
        "data": root / "data",
        "evaluations": root / "evaluations",
        "pairs": root / "pairs",
        "commit": root / "commit",
        "application": root / "application/validation.json",
        "precompute": root / "custody/precompute.json",
        "prescore_accounting": root / "accounting/prescore.json",
        "authorization": root / "score_authorization.json",
        "score": root / "score",
        "score_sandbox": root / "score_sandbox.json",
        "normalize": root / "normalized",
        "final_accounting": root / "accounting/final.json",
        "final_custody": root / "custody/final.json",
        "final_result": root / "final_comparison.json",
        "dispatch": root / "dispatch/dispatch.json",
    }


def _artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    p = _paths(args)
    merged = p["merged"]
    return {
        "ALIGNED_CHECKPOINT": p["aligned"] / "checkpoint_0000256.pt",
        "ALIGNED_REPORT": p["aligned"] / "report.json",
        "APPLICATION_REPORT": p["commit"] / "application_report.json",
        "APPLICATION_VALIDATION": p["application"],
        "ASSESSOR_RECEIPT": args.assessor_receipt,
        "CALIBRATION_DATA": p["data"] / "calibration_eval.jsonl",
        "CALIBRATION_PAIRS": p["pairs"] / "calibration.jsonl",
        "CALIBRATION_PAIRS_REPORT": p["pairs"] / "calibration_report.json",
        "CALIBRATION_REVISION_CANDIDATES": merged
        / "calibration_revision_candidates.jsonl",
        "CALIBRATION_REVISION_REPORT": merged / "calibration_revision_report.json",
        "CALIBRATION_UNCHANGED_CANDIDATES": merged
        / "calibration_unchanged_candidates.jsonl",
        "CALIBRATION_UNCHANGED_REPORT": merged / "calibration_unchanged_report.json",
        "COMMIT_CHECKPOINT": p["commit"] / "commit.pt",
        "COMMIT_TRAINING_REPORT": p["commit"] / "report.json",
        "DATA_REPORT": p["data"] / "report.json",
        "DEVELOPMENT_DATA": p["data"] / "development_eval.jsonl",
        "DEVELOPMENT_PAIRS": p["pairs"] / "development.jsonl",
        "DEVELOPMENT_PAIRS_REPORT": p["pairs"] / "development_report.json",
        "DEVELOPMENT_SOURCES": args.development_sources,
        "DRAFT_HIDDEN_CANDIDATES": merged / "development_draft_hidden_candidates.jsonl",
        "DRAFT_HIDDEN_CHECKPOINT": p["hidden"] / "checkpoint_0000256.pt",
        "DRAFT_HIDDEN_EVALUATION_REPORT": merged
        / "development_draft_hidden_report.json",
        "DRAFT_HIDDEN_REPORT": p["hidden"] / "report.json",
        "DRAFT_REPORT": merged / "drafts_report.json",
        "DRAFTS": merged / "drafts.jsonl",
        "ENVIRONMENT_RECEIPT": args.environment_receipt,
        "FREEZE_REPORT": args.freeze_report,
        "LIVE_PREFLIGHT": p["live"],
        "MECHANICS_REPORT": p["mechanics"] / "report.json",
        "OWNER_CHECKPOINT": p["owner"] / "checkpoint_0000256.pt",
        "OWNER_DATA": args.b1,
        "OWNER_REPORT": p["owner"] / "report.json",
        "REVISION_CANDIDATES": merged / "development_revision_candidates.jsonl",
        "REVISION_REPORT": merged / "development_revision_report.json",
        "REVISION_TRAINING_DATA": p["data"] / "revision_train.jsonl",
        "SELECTIONS": p["commit"] / "development_selections.jsonl",
        "SELF_REFINEMENT_CANDIDATES": merged
        / "development_self_refinement_candidates.jsonl",
        "SELF_REFINEMENT_REPORT": merged / "development_self_refinement_report.json",
        "TRAIN_SOURCES": args.train_sources,
        "UNCHANGED_CANDIDATES": merged / "development_unchanged_candidates.jsonl",
        "UNCHANGED_REPORT": merged / "development_unchanged_report.json",
    }


def _stage_exports(
    stage: str, args: argparse.Namespace, environment: dict[str, Any]
) -> dict[str, str]:
    p = _paths(args)
    a = _artifact_paths(args)
    values = _common(args, environment)
    merged, data, evaluations, pairs = (
        p["merged"],
        p["data"],
        p["evaluations"],
        p["pairs"],
    )
    checkpoint = {
        "revision": a["ALIGNED_CHECKPOINT"],
        "unchanged": a["OWNER_CHECKPOINT"],
        "self_refinement": a["OWNER_CHECKPOINT"],
        "draft_hidden": a["DRAFT_HIDDEN_CHECKPOINT"],
    }
    if stage == "preflight_cpu":
        values.update(
            OUTPUT=str(p["live"]),
            GRAPH_CONTRACT=str(args.graph_contract),
            PLAN=str(args.plan),
            SANDBOX_RECEIPT=str(args.sandbox_receipt),
            CLUSTER_PREFLIGHT=str(args.cluster_preflight),
        )
    elif stage == "mechanics":
        values.update(
            DATA=str(args.b1), DATA_SHA256=args.b1_sha256, OUTPUT=str(p["mechanics"])
        )
    elif stage in {"owner_fit", "aligned_fit", "draft_hidden_fit"}:
        role = {
            "owner_fit": "owner",
            "aligned_fit": "aligned",
            "draft_hidden_fit": "draft_hidden",
        }[stage]
        values.update(
            ROLE=role,
            DATA=str(args.b1 if role == "owner" else data / "revision_train.jsonl"),
            OUTPUT=str(
                p[
                    {"owner": "owner", "aligned": "aligned", "draft_hidden": "hidden"}[
                        role
                    ]
                ]
            ),
        )
        if role != "owner":
            values["WARM_START_CHECKPOINT"] = str(a["OWNER_CHECKPOINT"])
    elif stage == "draft_generate":
        values.update(
            TRAIN_SOURCE=str(args.train_sources),
            TRAIN_SOURCE_SHA256=SOURCE_FREEZE_SHA256["train_sources"],
            DEVELOPMENT_SOURCE=str(args.development_sources),
            DEVELOPMENT_SOURCE_SHA256=SOURCE_FREEZE_SHA256["development_sources"],
            FREEZE_REPORT=str(args.freeze_report),
            FREEZE_REPORT_SHA256=SOURCE_FREEZE_SHA256["freeze_report"],
            OWNER_CHECKPOINT=str(a["OWNER_CHECKPOINT"]),
            SHARD_ROOT=str(p["drafts"]),
            SHARD_COUNT="16",
        )
    elif stage == "draft_merge":
        values.update(
            TRAIN_SOURCE=str(args.train_sources),
            TRAIN_SOURCE_SHA256=SOURCE_FREEZE_SHA256["train_sources"],
            DEVELOPMENT_SOURCE=str(args.development_sources),
            DEVELOPMENT_SOURCE_SHA256=SOURCE_FREEZE_SHA256["development_sources"],
            FREEZE_REPORT=str(args.freeze_report),
            FREEZE_REPORT_SHA256=SOURCE_FREEZE_SHA256["freeze_report"],
            SHARD_ROOT=str(p["drafts"]),
            OUTPUT=str(merged / "drafts.jsonl"),
            REPORT=str(merged / "drafts_report.json"),
        )
    elif stage == "materialize":
        values.update(
            SOURCE_ROOT=str(args.train_sources.parent),
            DRAFTS=str(a["DRAFTS"]),
            DRAFT_REPORT=str(a["DRAFT_REPORT"]),
            ASSESSOR_RECEIPT=str(args.assessor_receipt),
            OUTPUT=str(data),
        )
    elif stage in {
        "calibration_revision",
        "calibration_unchanged",
        "development_revision",
        "development_unchanged",
        "development_self_refinement",
        "development_draft_hidden",
    }:
        split, arm = stage.split("_", 1)
        values.update(
            ARM=arm,
            SPLIT=split,
            CHECKPOINT=str(checkpoint[arm]),
            DATA=str(
                data
                / (
                    "calibration_eval.jsonl"
                    if split == "calibration"
                    else "development_eval.jsonl"
                )
            ),
            DATA_REPORT=str(data / "report.json"),
            SHARD_ROOT=str(evaluations / stage),
            SHARD_COUNT="4" if split == "calibration" else "8",
        )
    elif stage.endswith("_merge"):
        base = stage.removesuffix("_merge")
        split, arm = base.split("_", 1)
        values.update(
            ARM=arm,
            SPLIT=split,
            DATA=str(
                data
                / (
                    "calibration_eval.jsonl"
                    if split == "calibration"
                    else "development_eval.jsonl"
                )
            ),
            DATA_REPORT=str(data / "report.json"),
            SHARD_ROOT=str(evaluations / base),
            OUTPUT=str(merged / f"{base}_candidates.jsonl"),
            REPORT=str(merged / f"{base}_report.json"),
        )
    elif stage in {"commit_pairs", "development_commit_pairs"}:
        split = "calibration" if stage == "commit_pairs" else "development"
        values.update(
            SPLIT=split,
            DATA=str(data / f"{split}_eval.jsonl"),
            REVISION_REPORT=str(merged / f"{split}_revision_report.json"),
            REVISION_CANDIDATES=str(merged / f"{split}_revision_candidates.jsonl"),
            UNCHANGED_REPORT=str(merged / f"{split}_unchanged_report.json"),
            UNCHANGED_CANDIDATES=str(merged / f"{split}_unchanged_candidates.jsonl"),
            CANDIDATES_ROOT=str(merged),
            OUTPUT=str(pairs / f"{split}.jsonl"),
            REPORT=str(pairs / f"{split}_report.json"),
        )
    elif stage == "commit_fit":
        values.update(
            ALIGNED_CHECKPOINT=str(a["ALIGNED_CHECKPOINT"]),
            PAIRS=str(a["CALIBRATION_PAIRS"]),
            PAIRS_REPORT=str(a["CALIBRATION_PAIRS_REPORT"]),
            DEVELOPMENT_PAIRS=str(a["DEVELOPMENT_PAIRS"]),
            DEVELOPMENT_PAIRS_REPORT=str(a["DEVELOPMENT_PAIRS_REPORT"]),
            OUTPUT=str(p["commit"]),
        )
    elif stage == "commit_apply":
        values.update(
            COMMIT_CHECKPOINT=str(a["COMMIT_CHECKPOINT"]),
            COMMIT_TRAINING_REPORT=str(a["COMMIT_TRAINING_REPORT"]),
            DEVELOPMENT_PAIRS=str(a["DEVELOPMENT_PAIRS"]),
            DEVELOPMENT_PAIRS_REPORT=str(a["DEVELOPMENT_PAIRS_REPORT"]),
            APPLICATION_REPORT=str(a["APPLICATION_REPORT"]),
            SELECTIONS=str(a["SELECTIONS"]),
            OUTPUT=str(p["application"]),
        )
    elif stage == "precompute_custody":
        values.update(
            RUN_ID=args.run_id,
            GRAPH_CONTRACT=str(args.graph_contract),
            OUTPUT=str(p["precompute"]),
        )
        values.update({name: str(path) for name, path in a.items()})
    elif stage in {"prescore_accounting", "final_accounting"}:
        values.update(
            PHASE="prescore" if stage == "prescore_accounting" else "final",
            RUN_ID=args.run_id,
            GRAPH_CONTRACT=str(args.graph_contract),
            PLAN=str(args.plan),
            DISPATCH_RECEIPT=str(p["dispatch"]),
            OUTPUT=str(
                p["prescore_accounting"]
                if stage == "prescore_accounting"
                else p["final_accounting"]
            ),
        )
    elif stage == "authorize_score":
        names = (
            "APPLICATION_REPORT",
            "ASSESSOR_RECEIPT",
            "COMMIT_TRAINING_REPORT",
            "DATA_REPORT",
            "DEVELOPMENT_DATA",
            "DRAFT_HIDDEN_CANDIDATES",
            "DRAFT_HIDDEN_EVALUATION_REPORT",
            "ENVIRONMENT_RECEIPT",
            "REVISION_CANDIDATES",
            "REVISION_REPORT",
            "SELECTIONS",
            "SELF_REFINEMENT_CANDIDATES",
            "SELF_REFINEMENT_REPORT",
            "UNCHANGED_CANDIDATES",
            "UNCHANGED_REPORT",
        )
        values.update({name: str(a[name]) for name in names})
        values.update(
            PRESCORE_ACCOUNTING=str(p["prescore_accounting"]),
            PRECOMPUTE_CUSTODY=str(p["precompute"]),
            GRAPH_CONTRACT=str(args.graph_contract),
            SCORE_OUTPUT_ROOT=str(p["score"]),
            OUTPUT=str(p["authorization"]),
        )
    elif stage == "score_once":
        values.update(
            DEVELOPMENT_DATA=str(a["DEVELOPMENT_DATA"]),
            DATA_REPORT=str(a["DATA_REPORT"]),
            ASSESSOR_BOARD=str(args.assessor_board),
            ASSESSOR_RECEIPT=str(args.assessor_receipt),
            CANDIDATES_ROOT=str(merged),
            REVISION_REPORT=str(a["REVISION_REPORT"]),
            REVISION_CANDIDATES=str(a["REVISION_CANDIDATES"]),
            UNCHANGED_REPORT=str(a["UNCHANGED_REPORT"]),
            UNCHANGED_CANDIDATES=str(a["UNCHANGED_CANDIDATES"]),
            SELF_REFINEMENT_REPORT=str(a["SELF_REFINEMENT_REPORT"]),
            SELF_REFINEMENT_CANDIDATES=str(a["SELF_REFINEMENT_CANDIDATES"]),
            DRAFT_HIDDEN_REPORT=str(a["DRAFT_HIDDEN_EVALUATION_REPORT"]),
            DRAFT_HIDDEN_CANDIDATES=str(a["DRAFT_HIDDEN_CANDIDATES"]),
            APPLICATION_REPORT=str(a["APPLICATION_REPORT"]),
            SELECTIONS=str(a["SELECTIONS"]),
            COMMIT_TRAINING_REPORT=str(a["COMMIT_TRAINING_REPORT"]),
            PRECOMPUTE_CUSTODY=str(p["precompute"]),
            PRESCORE_ACCOUNTING=str(p["prescore_accounting"]),
            GRAPH_CONTRACT=str(args.graph_contract),
            SCORE_AUTHORIZATION=str(p["authorization"]),
            SANDBOX_RECEIPT_OUTPUT=str(p["score_sandbox"]),
            OUTPUT=str(p["score"]),
        )
    elif stage == "normalize":
        values.update(
            SCORE_REPORT=str(p["score"] / "report.json"),
            PRECOMPUTE_CUSTODY=str(p["precompute"]),
            OUTPUT=str(p["normalize"]),
        )
    elif stage == "evidence_mirror":
        values.update(
            RUN_ID=args.run_id,
            GRAPH_CONTRACT=str(args.graph_contract),
            PRECOMPUTE_CUSTODY=str(p["precompute"]),
            PRESCORE_ACCOUNTING=str(p["prescore_accounting"]),
            SCORE_AUTHORIZATION=str(p["authorization"]),
            SCORE_CONSUMPTION=str(
                args.run_root / "score.score-authorization-consumed.json"
            ),
            SCORE_REPORT=str(p["score"] / "report.json"),
            SCORE_OUTCOMES=str(p["score"] / "outcomes.jsonl"),
            SCORE_SANDBOX_RECEIPT=str(p["score_sandbox"]),
            SCHEDULER_ACCOUNTING=str(p["final_accounting"]),
            PLAN=str(args.plan),
            DISPATCH_RECEIPT=str(p["dispatch"]),
            MODEL_MANIFEST=str(args.model_manifest),
            AUTHORIZED_EVIDENCE_ROOT=str(args.evidence_root),
            OUTPUT_ROOT=str(args.evidence_root / f"{args.run_id}-preterminal"),
            LEARNED_COMMIT_ARM_REPORT=str(p["normalize"] / "learned_commit.json"),
            TRAINED_REVISION_ARM_REPORT=str(p["normalize"] / "trained_revision.json"),
            UNCHANGED_ARM_REPORT=str(p["normalize"] / "unchanged.json"),
            SELF_REFINEMENT_ARM_REPORT=str(p["normalize"] / "self_refinement.json"),
            DRAFT_HIDDEN_ARM_REPORT=str(p["normalize"] / "draft_hidden.json"),
        )
        for name in (
            "ALIGNED_CHECKPOINT",
            "ALIGNED_REPORT",
            "APPLICATION_REPORT",
            "APPLICATION_VALIDATION",
            "ASSESSOR_RECEIPT",
            "CALIBRATION_PAIRS",
            "CALIBRATION_PAIRS_REPORT",
            "CALIBRATION_REVISION_CANDIDATES",
            "CALIBRATION_REVISION_REPORT",
            "CALIBRATION_UNCHANGED_CANDIDATES",
            "CALIBRATION_UNCHANGED_REPORT",
            "COMMIT_CHECKPOINT",
            "COMMIT_TRAINING_REPORT",
            "DATA_REPORT",
            "DEVELOPMENT_PAIRS",
            "DEVELOPMENT_PAIRS_REPORT",
            "DRAFT_HIDDEN_CANDIDATES",
            "DRAFT_HIDDEN_CHECKPOINT",
            "DRAFT_HIDDEN_EVALUATION_REPORT",
            "DRAFT_HIDDEN_REPORT",
            "DRAFT_REPORT",
            "DRAFTS",
            "ENVIRONMENT_RECEIPT",
            "FREEZE_REPORT",
            "LIVE_PREFLIGHT",
            "MECHANICS_REPORT",
            "OWNER_CHECKPOINT",
            "OWNER_REPORT",
            "REVISION_CANDIDATES",
            "REVISION_REPORT",
            "SELECTIONS",
            "SELF_REFINEMENT_CANDIDATES",
            "SELF_REFINEMENT_REPORT",
            "UNCHANGED_CANDIDATES",
            "UNCHANGED_REPORT",
        ):
            values[name] = str(a[name])
    elif stage == "compute_custody":
        values.update(
            PRECOMPUTE_CUSTODY=str(p["precompute"]),
            SCORE_REPORT=str(p["score"] / "report.json"),
            SCORE_CONSUMPTION=str(
                args.run_root / "score.score-authorization-consumed.json"
            ),
            SCHEDULER_ACCOUNTING=str(p["final_accounting"]),
            EVIDENCE_MIRROR=str(
                args.evidence_root / f"{args.run_id}-preterminal/manifest.json"
            ),
            GRAPH_CONTRACT=str(args.graph_contract),
            LEARNED_COMMIT_REPORT=str(p["normalize"] / "learned_commit.json"),
            TRAINED_REVISION_REPORT=str(p["normalize"] / "trained_revision.json"),
            UNCHANGED_REPORT=str(p["normalize"] / "unchanged.json"),
            SELF_REFINEMENT_REPORT=str(p["normalize"] / "self_refinement.json"),
            DRAFT_HIDDEN_REPORT=str(p["normalize"] / "draft_hidden.json"),
            OUTPUT=str(p["final_custody"]),
        )
    elif stage == "final_compare":
        values.update(
            GRAPH_CONTRACT=str(args.graph_contract),
            FINAL_CUSTODY=str(p["final_custody"]),
            PRETERMINAL_EVIDENCE_MIRROR=str(
                args.evidence_root / f"{args.run_id}-preterminal/manifest.json"
            ),
            AUTHORIZED_EVIDENCE_ROOT=str(args.evidence_root),
            TERMINAL_EVIDENCE_ROOT=str(args.evidence_root / f"{args.run_id}-terminal"),
            LEARNED_COMMIT_REPORT=str(p["normalize"] / "learned_commit.json"),
            TRAINED_REVISION_REPORT=str(p["normalize"] / "trained_revision.json"),
            UNCHANGED_REPORT=str(p["normalize"] / "unchanged.json"),
            SELF_REFINEMENT_REPORT=str(p["normalize"] / "self_refinement.json"),
            DRAFT_HIDDEN_REPORT=str(p["normalize"] / "draft_hidden.json"),
            OUTPUT=str(p["final_result"]),
        )
    else:
        raise Q36MTRDispatchError(f"unknown Q36 stage: {stage}")
    return {name: str(value) for name, value in values.items()}


def preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if any(name.startswith(("SLURM_", "SBATCH_")) for name in os.environ):
        raise Q36MTRDispatchError("ambient Slurm control is not admissible")
    if (
        not RUN_ID_PATTERN.fullmatch(args.run_id)
        or args.run_root.exists()
        or args.run_root.is_symlink()
    ):
        raise Q36MTRDispatchError("Q36 run identity/root differs")
    if (
        not args.run_root.is_absolute()
        or args.run_root.parent.is_symlink()
        or not args.run_root.parent.is_dir()
    ):
        raise Q36MTRDispatchError("Q36 run-root parent differs")
    if (
        not args.evidence_root.is_absolute()
        or args.evidence_root.is_symlink()
        or not args.evidence_root.is_dir()
    ):
        raise Q36MTRDispatchError("Q36 evidence root differs")
    graph = _load(args.graph_contract, "shohin-q36-mtr-graph-v1")
    plan = _load(args.plan, "shohin-q36-mtr-execution-plan-v1")
    authorization = _load(
        args.phase_authorization, "shohin-q36-mtr-phase-authorization-v1"
    )
    environment = _load(args.environment_receipt, "shohin-q36-mtr-environment-v1")
    validate_graph(graph)
    validate_plan(plan)
    if (
        graph.get("source_commit") != args.source_commit
        or plan.get("source_commit") != args.source_commit
        or plan.get("graph_sha256") != sha256_file(args.graph_contract)
        or authorization.get("status") != "authorized"
        or authorization.get("source_commit") != args.source_commit
        or authorization.get("run_id") != args.run_id
        or authorization.get("run_root") != str(args.run_root.resolve(strict=False))
        or authorization.get("graph_contract_sha256")
        != sha256_file(args.graph_contract)
        or authorization.get("plan_sha256") != sha256_file(args.plan)
        or authorization.get("scientific_submit_authorized") is not True
        or authorization.get("automatic_retry") is not False
        or authorization.get("automatic_successor") is not False
        or authorization.get("automatic_confirmation") is not False
        or authorization.get("source_freeze_sha256") != SOURCE_FREEZE_SHA256
        or sha256_file(args.phase_authorization) != args.phase_authorization_sha256
        or sha256_file(args.runtime / "SHA256SUMS") != args.runtime_manifest_sha256
        or json.loads((args.runtime / "runtime.json").read_text(encoding="utf-8")).get(
            "source_commit"
        )
        != args.source_commit
        or environment.get("status") != "pass"
        or not isinstance(environment.get("environment_tree_sha256"), str)
        or len(environment["environment_tree_sha256"]) != 64
        or authorization.get("runtime_manifest_sha256") != args.runtime_manifest_sha256
        or authorization.get("model_revision") != args.model_revision
        or authorization.get("model_manifest_sha256") != args.model_manifest_sha256
        or sha256_file(args.b1) != args.b1_sha256
        or sha256_file(args.model_manifest) != args.model_manifest_sha256
    ):
        raise Q36MTRDispatchError("Q36 execution binding differs")
    for path in (
        args.train_sources,
        args.development_sources,
        args.freeze_report,
        args.assessor_receipt,
        args.assessor_board,
        args.b1,
        args.model_manifest,
        args.sandbox_receipt,
        args.cluster_preflight,
    ):
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise Q36MTRDispatchError(f"Q36 execution input differs: {path}")
    if {
        "train_sources": sha256_file(args.train_sources),
        "development_sources": sha256_file(args.development_sources),
        "freeze_report": sha256_file(args.freeze_report),
        "assessor_receipt": sha256_file(args.assessor_receipt),
    } != {
        name: SOURCE_FREEZE_SHA256[name]
        for name in (
            "train_sources",
            "development_sources",
            "freeze_report",
            "assessor_receipt",
        )
    }:
        raise Q36MTRDispatchError("Q36 source-freeze hash differs")
    completed = subprocess.run(
        ["squeue", "-h", "-u", args.user, "-o", "%i|%T|%P|%R"],
        check=True,
        text=True,
        capture_output=True,
        env={"PATH": PATH_VALUE},
    )
    if completed.stdout.strip():
        raise Q36MTRDispatchError("Q36 account queue is not empty")
    return graph, plan, environment


def _dependency(stage: Any, job_ids: dict[str, str]) -> str | None:
    if not stage.dependencies:
        return None
    return "afterok:" + ":".join(job_ids[name] for name in stage.dependencies)


def submit(args: argparse.Namespace) -> dict[str, Any]:
    if args.submit_ack != ACK:
        raise Q36MTRDispatchError(f"set --submit-ack {ACK} to submit")
    graph, plan, environment = preflight(args)
    args.run_root.mkdir(mode=0o700)
    (args.run_root / "logs").mkdir(mode=0o700)
    job_ids: dict[str, str] = {}
    submitted: list[str] = []
    try:
        for stage in STAGES:
            command = [
                "sbatch",
                "--parsable",
                "--no-requeue",
                "--partition=normal",
                "--exclude=" + ",".join(EXCLUDED_NODES),
                "--nodes=1",
                "--ntasks=1",
            ]
            if not stage.dependencies:
                command.append("--hold")
            dependency = _dependency(stage, job_ids)
            if dependency:
                command.append("--dependency=" + dependency)
            if stage.tasks > 1:
                command.append(f"--array=0-{stage.tasks - 1}")
            command.append(
                "--export="
                + _render_exports(_stage_exports(stage.name, args, environment))
            )
            command.append(str(args.runtime / SCRIPTS[stage.name]))
            result = (
                subprocess.run(
                    command,
                    check=True,
                    text=True,
                    capture_output=True,
                    cwd=args.run_root,
                    env={"PATH": PATH_VALUE},
                )
                .stdout.strip()
                .split(";", 1)[0]
            )
            if not result.isdigit() or result in job_ids.values():
                raise Q36MTRDispatchError(
                    f"Q36 Slurm job identity differs: {stage.name}"
                )
            job_ids[stage.name] = result
            submitted.append(result)
        receipt = {
            "schema": SCHEMA,
            "status": "submitted",
            "run_id": args.run_id,
            "source_commit": args.source_commit,
            "graph_contract_sha256": sha256_file(args.graph_contract),
            "plan_sha256": sha256_file(args.plan),
            "partition": "normal",
            "excluded_nodes": list(EXCLUDED_NODES),
            "requeue": False,
            "retry_authorized": False,
            "successor_authorized": False,
            "preflight_queue_empty": True,
            "job_ids": job_ids,
            "stage_resources": {
                stage.name: {
                    "tasks": stage.tasks,
                    "h100s_per_task": stage.h100_per_task,
                    "is_array": stage.tasks > 1,
                }
                for stage in STAGES
            },
            "submission_count": 1,
            "h100_requests": 61,
            "expected_h100_hours": 58.9,
            "maximum_concurrent_single_h100_requests": 32,
            "stop_after_gate": True,
            "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        }
        _atomic_json(_paths(args)["dispatch"], receipt)
        subprocess.run(
            ["scontrol", "release", job_ids["preflight_cpu"]],
            check=True,
            text=True,
            capture_output=True,
            env={"PATH": PATH_VALUE},
        )
        return receipt
    except BaseException:
        if submitted:
            subprocess.run(
                ["scancel", *submitted],
                check=False,
                text=True,
                capture_output=True,
                env={"PATH": PATH_VALUE},
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "submit"), required=True)
    parser.add_argument("--submit-ack")
    parser.add_argument("--user", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--graph-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase-authorization", type=Path, required=True)
    parser.add_argument("--phase-authorization-sha256", required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--sandbox-receipt", type=Path, required=True)
    parser.add_argument("--cluster-preflight", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--train-sources", type=Path, required=True)
    parser.add_argument("--development-sources", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--assessor-receipt", type=Path, required=True)
    parser.add_argument("--assessor-board", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--b1-sha256", required=True)
    args = parser.parse_args()
    graph, plan, _ = preflight(args)
    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "status": "pass",
                    "source_commit": graph["source_commit"],
                    "h100_requests": plan["h100_requests"],
                },
                sort_keys=True,
            )
        )
        return 0
    result = submit(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "root_job": result["job_ids"]["preflight_cpu"],
                "terminal_job": result["job_ids"]["final_compare"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
