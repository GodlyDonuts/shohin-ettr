#!/usr/bin/env python3
"""Package the qualified draft/revision/commit reasoner as an immutable delta."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


SCHEMA = "shohin-idr-aqc-release-v1"
COMMIT_SCHEMA = "shohin-aqc1-commit-report-v1"
PRODUCT_SCHEMA = "shohin-aqc1-product-application-v1"


class IDRAQCReleaseError(RuntimeError):
    """The requested release violates the qualified lineage contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise IDRAQCReleaseError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise IDRAQCReleaseError(f"{label} SHA-256 differs")
    return observed


def load_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require_file(path, expected_sha256, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IDRAQCReleaseError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise IDRAQCReleaseError(f"{label} is not an object")
    return payload


def validate_lineage(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    draft_sha = require_file(
        args.draft_checkpoint,
        args.expected_draft_checkpoint_sha256,
        "draft checkpoint",
    )
    revision_sha = require_file(
        args.revision_checkpoint,
        args.expected_revision_checkpoint_sha256,
        "revision checkpoint",
    )
    commit_sha = require_file(
        args.commit_checkpoint,
        args.expected_commit_checkpoint_sha256,
        "commit checkpoint",
    )
    draft_report = load_json(
        args.draft_report, args.expected_draft_report_sha256, "draft report"
    )
    revision_report = load_json(
        args.revision_report,
        args.expected_revision_report_sha256,
        "revision report",
    )
    commit_report = load_json(
        args.commit_report, args.expected_commit_report_sha256, "commit report"
    )
    product_report = load_json(
        args.product_report, args.expected_product_report_sha256, "product report"
    )

    for label, report in (
        ("draft report", draft_report),
        ("revision report", revision_report),
    ):
        if report.get("status") != "complete":
            raise IDRAQCReleaseError(f"{label} is incomplete")
        if report.get("model_revision") != args.model_revision:
            raise IDRAQCReleaseError(f"{label} model revision differs")
    if revision_report.get("warm_start_sha256") != draft_sha:
        raise IDRAQCReleaseError("revision report is not bound to the draft checkpoint")
    if commit_report.get("schema") != COMMIT_SCHEMA:
        raise IDRAQCReleaseError("commit report schema differs")
    if commit_report.get("status") != "complete":
        raise IDRAQCReleaseError("commit report is incomplete")
    if commit_report.get("arm") != "antisymmetric":
        raise IDRAQCReleaseError("commit report is not the qualified treatment")
    if commit_report.get("model_revision") != args.model_revision:
        raise IDRAQCReleaseError("commit report model revision differs")
    if commit_report.get("adapter_checkpoint_sha256") != draft_sha:
        raise IDRAQCReleaseError("commit report is not bound to the draft checkpoint")
    if commit_report.get("checkpoint_sha256") != commit_sha:
        raise IDRAQCReleaseError("commit report checkpoint binding differs")
    if commit_report.get("holdout_gate_pass") is not True:
        raise IDRAQCReleaseError("commit holdout gate did not pass")
    if commit_report.get("protected_adapter_unchanged") is not True:
        raise IDRAQCReleaseError("commit training changed the protected draft adapter")

    if product_report.get("schema") != PRODUCT_SCHEMA:
        raise IDRAQCReleaseError("product report schema differs")
    if product_report.get("status") != "complete":
        raise IDRAQCReleaseError("product report is incomplete")
    if product_report.get("arm") != "antisymmetric":
        raise IDRAQCReleaseError("product report arm differs")
    if product_report.get("gate_pass") is not True:
        raise IDRAQCReleaseError("product gate did not pass")
    if product_report.get("commit_sha256") != commit_sha:
        raise IDRAQCReleaseError("product report commit binding differs")
    if (
        product_report.get("commit_report_sha256")
        != args.expected_commit_report_sha256
    ):
        raise IDRAQCReleaseError("product report commit-report binding differs")
    gates = product_report.get("gates")
    if not isinstance(gates, dict) or not gates or not all(gates.values()):
        raise IDRAQCReleaseError("product report gates are incomplete")
    selected = product_report.get("arms", {}).get("selected")
    if not isinstance(selected, dict):
        raise IDRAQCReleaseError("product report lacks selected summary")

    return {
        "draft": draft_report,
        "revision": revision_report,
        "commit": commit_report,
        "product": product_report,
        "hashes": {
            "draft_checkpoint": draft_sha,
            "revision_checkpoint": revision_sha,
            "commit_checkpoint": commit_sha,
        },
    }


def model_card(model_name: str, model_revision: str, product: dict[str, Any]) -> str:
    selected = product["arms"]["selected"]
    revision = product["arms"]["idr1"]
    control = product["arms"]["control"]
    return f"""# Shohin Draft/Revise/Commit Delta

This immutable delta runs one pinned `{model_name}` model family at revision
`{model_revision}` through four model-owned stages:

1. produce a complete internal draft with `draft_adapter.pt`;
2. produce a trained correction with `revision_adapter.pt`;
3. produce a matched unchanged continuation with `draft_adapter.pt`; and
4. use `commit.pt` to select one complete trajectory without averaging fields.

No external proposal model, verifier, answer label, correctness bit, benchmark
router, or tool is used at inference. The exact base model is not included.

## Qualified product result

- Learned commit: {selected['solved']}/{selected['total']} solved;
  {100 * selected['macro_accuracy']:.3f}% five-domain macro.
- Trained revision alone: {revision['solved']}/{revision['total']} solved;
  {100 * revision['macro_accuracy']:.3f}% five-domain macro.
- Matched unchanged continuation: {control['solved']}/{control['total']} solved;
  {100 * control['macro_accuracy']:.3f}% five-domain macro.

The release preserves a measured architecture, not a claim that its
antisymmetric head is uniquely responsible: the matched independent commit
was one answer behind on the protected board.
"""


def package(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise IDRAQCReleaseError(f"refusing existing output: {args.output}")
    config = args.model_root / "config.json"
    config_sha = require_file(
        config, args.expected_model_config_sha256, "model config"
    )
    lineage = validate_lineage(args)
    prompt_sha: str | None = None
    if args.interaction_prompts is not None:
        if args.expected_interaction_prompts_sha256 is None:
            raise IDRAQCReleaseError("interaction prompt SHA-256 is required")
        prompt_sha = require_file(
            args.interaction_prompts,
            args.expected_interaction_prompts_sha256,
            "interaction prompts",
        )

    temporary = args.output.with_name(f".{args.output.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise IDRAQCReleaseError(f"refusing existing temporary output: {temporary}")
    temporary.mkdir(parents=True)
    try:
        payloads = {
            "config.json": config,
            "draft_adapter.pt": args.draft_checkpoint,
            "draft_report.json": args.draft_report,
            "revision_adapter.pt": args.revision_checkpoint,
            "revision_report.json": args.revision_report,
            "commit.pt": args.commit_checkpoint,
            "commit_report.json": args.commit_report,
            "product_report.json": args.product_report,
        }
        if args.interaction_prompts is not None:
            payloads["interaction_prompts.jsonl"] = args.interaction_prompts
        for name, source in payloads.items():
            shutil.copy2(source, temporary / name)

        file_hashes = {
            name: sha256_file(temporary / name) for name in sorted(payloads)
        }
        manifest = {
            "schema": SCHEMA,
            "status": "qualified",
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "base_model_included": False,
            "model_config_sha256": config_sha,
            "inference_path": [
                "draft_adapter.pt",
                "revision_adapter.pt",
                "draft_adapter.pt",
                "commit.pt",
            ],
            "inference_stages": [
                "internal_draft",
                "trained_revision",
                "unchanged_continuation",
                "whole_trajectory_commit",
            ],
            "files": file_hashes,
            "interaction_prompts_sha256": prompt_sha,
            "product_summary": lineage["product"]["arms"]["selected"],
            "revision_summary": lineage["product"]["arms"]["idr1"],
            "control_summary": lineage["product"]["arms"]["control"],
            "product_gates": lineage["product"]["gates"],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "MODEL_CARD.md").write_text(
            model_card(args.model_name, args.model_revision, lineage["product"]),
            encoding="utf-8",
        )
        names = sorted(path.name for path in temporary.iterdir() if path.is_file())
        sums = "".join(
            f"{sha256_file(temporary / name)}  {name}\n" for name in names
        )
        (temporary / "SHA256SUMS").write_text(sums, encoding="utf-8")
        os.replace(temporary, args.output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    for path in args.output.iterdir():
        if path.is_file():
            path.chmod(0o444)
    args.output.chmod(0o555)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--expected-model-config-sha256", required=True)
    for name in ("draft", "revision"):
        parser.add_argument(f"--{name}-checkpoint", type=Path, required=True)
        parser.add_argument(
            f"--expected-{name}-checkpoint-sha256", required=True
        )
        parser.add_argument(f"--{name}-report", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-report-sha256", required=True)
    parser.add_argument("--commit-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-commit-checkpoint-sha256", required=True)
    parser.add_argument("--commit-report", type=Path, required=True)
    parser.add_argument("--expected-commit-report-sha256", required=True)
    parser.add_argument("--product-report", type=Path, required=True)
    parser.add_argument("--expected-product-report-sha256", required=True)
    parser.add_argument("--interaction-prompts", type=Path)
    parser.add_argument("--expected-interaction-prompts-sha256")
    parser.add_argument("--output", type=Path, required=True)
    manifest = package(parser.parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
