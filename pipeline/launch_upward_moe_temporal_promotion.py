#!/usr/bin/env python3
"""Replay one larger-MoE promotion receipt and launch its exact temporal graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dispatch_upward_moe_temporal as dispatcher
import select_upward_moe_temporal_promotion as promotion_selector
from analyze_upward_moe_scaling import sha256_file

SCHEMA = "shohin-upward-moe-temporal-auto-launch-v1"
CLAIM_SCHEMA = "shohin-upward-moe-temporal-launch-claim-v1"
FAILURE_SCHEMA = "shohin-upward-moe-temporal-launch-failure-v1"

COMMON_PATHS = {
    "b1": Path(
        "/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/data/"
        "v10_tokenbalanced_35m20c10s10p25t_4m_verified_r1.jsonl"
    ),
    "source_root": Path(
        "/lustre/fs1/home/sa305415/shohin/artifacts/"
        "pcf17_ministral_037f122_r1/prepared/sources"
    ),
    "train_source": Path(
        "/lustre/fs1/home/sa305415/shohin/artifacts/"
        "pcf17_ministral_037f122_r1/prepared/sources/train_sources.jsonl"
    ),
    "development_source": Path(
        "/lustre/fs1/home/sa305415/shohin/artifacts/"
        "pcf17_ministral_037f122_r1/prepared/sources/development_sources.jsonl"
    ),
    "freeze_report": Path(
        "/lustre/fs1/home/sa305415/shohin/artifacts/"
        "pcf17_ministral_037f122_r1/prepared/sources/report.json"
    ),
    "assessor_receipt": Path(
        "/lustre/fs1/home/sa305415/shohin/artifacts/"
        "pcf17_ministral_037f122_r1/custodian/confirmation_assessor_receipt.json"
    ),
    "assessors": Path(
        "/lustre/fs1/home/sa305415/shohin/artifacts/"
        "pcf17_ministral_037f122_r1/custodian/confirmation_assessors.jsonl"
    ),
}

HOST_PATHS: dict[str, dict[str, Any]] = {
    "nemotron-super": {
        "model_root": Path(
            "/lustre/fs1/home/sa305415/shohin/artifacts/external/"
            "nemotron-3-super-120b-a12b-fp8-7d7e5797"
        ),
        "model_manifest": Path(
            "/lustre/fs1/home/sa305415/shohin/artifacts/external/"
            "nemotron-3-super-120b-a12b-fp8-7d7e5797/SHA256SUMS"
        ),
        "expected_model_manifest_sha256": None,
        "mechanics_report": Path(
            "/lustre/fs1/home/sa305415/shohin/artifacts/"
            "nemotron_super_mechanics_glibc228_a14b1dff_r1/report.json"
        ),
        "overlay_root": Path(
            "/lustre/fs1/home/sa305415/shohin/artifacts/"
            "nemotron_super_overlay_glibc228_a14b1dff_r1"
        ),
        "overlay_manifest": Path(
            "/lustre/fs1/home/sa305415/shohin/artifacts/"
            "nemotron_super_overlay_glibc228_a14b1dff_r1/SHA256SUMS"
        ),
        "causal_conv_root": Path(
            "/lustre/fs1/home/sa305415/shohin/env_targets/"
            "qwen36-fastkernels-0.4.2-r5"
        ),
    },
    "mixtral-8x22b": {
        "model_root": Path(
            "/lustre/fs1/home/sa305415/shohin/artifacts/external/"
            "mixtral-8x22b-instruct-cc88a6c"
        ),
        "model_manifest": Path(
            "/lustre/fs1/home/sa305415/shohin/artifacts/external/"
            "mixtral-8x22b-instruct-cc88a6c/SHA256SUMS"
        ),
        "expected_model_manifest_sha256": (
            "46b8475d98e2a49f9a81329287beb9d450dfd4d7a74886e8780708764a8f3fe7"
        ),
        "mechanics_report": Path(
            "/lustre/fs1/home/sa305415/shohin/artifacts/"
            "mixtral_8x22b_upward_trained_revision_cc88a6c_r1/mechanics/report.json"
        ),
        "overlay_root": None,
        "overlay_manifest": None,
        "causal_conv_root": None,
    },
}


class UpwardMoETemporalLaunchError(RuntimeError):
    """The evidence-bound larger-MoE temporal launch differed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise UpwardMoETemporalLaunchError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def replay_promotion(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise UpwardMoETemporalLaunchError("promotion receipt is absent")
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UpwardMoETemporalLaunchError("promotion receipt is unreadable") from error
    candidates = observed.get("candidates") if isinstance(observed, dict) else None
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise UpwardMoETemporalLaunchError("promotion candidate receipts differ")
    score_paths = []
    for candidate in candidates:
        source = candidate.get("source_path") if isinstance(candidate, dict) else None
        if not isinstance(source, str):
            raise UpwardMoETemporalLaunchError("promotion score path differs")
        score_paths.append(Path(source))
    try:
        replayed = promotion_selector.select(score_paths)
    except Exception as error:
        raise UpwardMoETemporalLaunchError("promotion replay failed") from error
    if observed != replayed:
        raise UpwardMoETemporalLaunchError("promotion receipt differs from replay")
    return replayed


def _dispatch_args(args: argparse.Namespace, host: str) -> SimpleNamespace:
    if host not in HOST_PATHS:
        raise UpwardMoETemporalLaunchError("selected host differs")
    host_paths = HOST_PATHS[host]
    return SimpleNamespace(
        host=host,
        runtime=args.runtime,
        runtime_manifest_sha256=args.runtime_manifest_sha256,
        python=args.python,
        model_root=host_paths["model_root"],
        model_manifest=host_paths["model_manifest"],
        mechanics_report=host_paths["mechanics_report"],
        expected_model_manifest_sha256=host_paths["expected_model_manifest_sha256"],
        overlay_root=host_paths["overlay_root"],
        overlay_manifest=host_paths["overlay_manifest"],
        causal_conv_root=host_paths["causal_conv_root"],
        b1=COMMON_PATHS["b1"],
        source_root=COMMON_PATHS["source_root"],
        train_source=COMMON_PATHS["train_source"],
        development_source=COMMON_PATHS["development_source"],
        freeze_report=COMMON_PATHS["freeze_report"],
        assessor_receipt=COMMON_PATHS["assessor_receipt"],
        assessors=COMMON_PATHS["assessors"],
        run_root=args.run_root,
        receipt=args.dispatch_receipt,
        submit=args.submit,
    )


def _input_receipts(dispatch_args: SimpleNamespace) -> dict[str, Any]:
    files = {
        "runtime_manifest": dispatch_args.runtime / "SHA256SUMS",
        "model_manifest": dispatch_args.model_manifest,
        "mechanics_report": dispatch_args.mechanics_report,
        "b1": dispatch_args.b1,
        "train_source": dispatch_args.train_source,
        "development_source": dispatch_args.development_source,
        "freeze_report": dispatch_args.freeze_report,
        "assessor_receipt": dispatch_args.assessor_receipt,
        "assessors": dispatch_args.assessors,
    }
    if dispatch_args.overlay_manifest is not None:
        files["overlay_manifest"] = dispatch_args.overlay_manifest
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in files.items()
    }


def launch(args: argparse.Namespace) -> dict[str, Any]:
    if any(
        path.exists() or path.is_symlink()
        for path in (
            args.launch_receipt,
            args.claim,
            args.dispatch_receipt,
            args.failure_receipt,
        )
    ):
        raise UpwardMoETemporalLaunchError("launch output already exists")
    decision = replay_promotion(args.promotion)
    selected = decision.get("selected_dispatcher_host")
    if decision.get("status") == "no_qualifying_larger_host":
        result = {
            "schema": SCHEMA,
            "status": "no_launch",
            "promotion_sha256": sha256_file(args.promotion),
            "selected_host": None,
            "scientific_jobs_submitted": 0,
            "reason": "no_larger_host_satisfied_the_frozen_promotion_contract",
        }
        if args.submit:
            _atomic_json(args.launch_receipt, result)
        return result
    if decision.get("status") != "promote" or selected not in HOST_PATHS:
        raise UpwardMoETemporalLaunchError("promotion decision differs")

    dispatch_args = _dispatch_args(args, selected)
    stages = dispatcher.build_graph(dispatch_args)
    dispatcher.validate(dispatch_args, stages)
    receipts = _input_receipts(dispatch_args)
    plan = dispatcher.submit(dispatch_args, stages) if not args.submit else None
    if not args.submit:
        return {
            "schema": SCHEMA,
            "status": "dry_run",
            "promotion_sha256": sha256_file(args.promotion),
            "selected_host": selected,
            "input_receipts": receipts,
            "dispatch": plan,
        }

    claim = {
        "schema": CLAIM_SCHEMA,
        "status": "claimed",
        "promotion_sha256": sha256_file(args.promotion),
        "selected_host": selected,
        "run_root": str(args.run_root.resolve()),
        "dispatch_receipt": str(args.dispatch_receipt.resolve()),
        "duplicate_launch_authorized": False,
    }
    _atomic_json(args.claim, claim)
    try:
        dispatch = dispatcher.submit(dispatch_args, stages)
        dispatcher._atomic_json(args.dispatch_receipt, dispatch)
        result = {
            "schema": SCHEMA,
            "status": "submitted",
            "promotion_sha256": sha256_file(args.promotion),
            "claim_sha256": sha256_file(args.claim),
            "selected_host": selected,
            "run_root": str(args.run_root.resolve()),
            "input_receipts": receipts,
            "dispatch_receipt": str(args.dispatch_receipt.resolve()),
            "dispatch_receipt_sha256": sha256_file(args.dispatch_receipt),
            "job_ids": dispatch["job_ids"],
            "allocation_tasks": dispatch["allocation_tasks"],
            "scientific_jobs_submitted": dispatch["allocation_tasks"],
            "duplicate_launch_authorized": False,
        }
        _atomic_json(args.launch_receipt, result)
        return result
    except BaseException as error:
        failure = {
            "schema": FAILURE_SCHEMA,
            "status": "terminal_infrastructure_failure_after_claim",
            "promotion_sha256": sha256_file(args.promotion),
            "claim_sha256": sha256_file(args.claim),
            "selected_host": selected,
            "error_type": type(error).__name__,
            "error": str(error),
            "automatic_retry": False,
        }
        if not args.failure_receipt.exists():
            _atomic_json(args.failure_receipt, failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--dispatch-receipt", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--failure-receipt", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = launch(parse_args())
    print(
        json.dumps(
            {"status": result["status"], "selected_host": result["selected_host"]},
            sort_keys=True,
        )
    )
