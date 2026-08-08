#!/usr/bin/env python3
"""Package a qualified internal-draft/revision system as an immutable delta release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


class IDRReleaseError(RuntimeError):
    """The requested release violates its qualified lineage contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise IDRReleaseError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise IDRReleaseError(f"{label} SHA-256 differs")
    return observed


def load_product_report(path: Path, expected_sha256: str) -> dict[str, Any]:
    require_file(path, expected_sha256, "product report")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IDRReleaseError("product report is invalid JSON") from exc
    if report.get("schema") != "shohin-idr-product-comparison-v1":
        raise IDRReleaseError("product report schema differs")
    if report.get("status") != "complete":
        raise IDRReleaseError("product report is incomplete")
    if report.get("gate_pass") is not True:
        raise IDRReleaseError("product gate did not pass")
    required_gates = {
        "at_least_27_additional_main_answers",
        "at_least_0_05_macro_gain",
        "all_five_domain_deltas_nonnegative",
    }
    gates = report.get("gates")
    if not isinstance(gates, dict) or set(gates) != required_gates:
        raise IDRReleaseError("product gate set differs")
    if not all(gates.values()):
        raise IDRReleaseError("one or more product gates are false")
    return report


def model_card(model_name: str, model_revision: str, report: dict[str, Any]) -> str:
    treatment = report["treatment_summary"]
    control = report["control_summary"]
    delta = report["deltas"]
    domains = treatment["domains"]
    rows = "\n".join(
        f"| {name} | {values['correct']}/{values['total']} |"
        for name, values in sorted(domains.items())
    )
    return f"""# Shohin Internal Draft/Revision Delta

This package contains two adapter states for `{model_name}` at immutable
revision `{model_revision}`. Inference first produces one complete internal
draft with `draft_adapter.pt`, then rereads the original request plus that
draft and produces one corrected response with `revision_adapter.pt`.

The package is a delta release: obtain the exact base model separately and
verify its `config.json` against `manifest.json`. No external solver, answer
router, verifier, or proposal model is required at inference.

## Qualified product result

- Treatment: {treatment['solved']}/{treatment['total']} solved;
  {100 * treatment['macro_accuracy']:.3f}% five-domain macro.
- Matched unchanged second pass: {control['solved']}/{control['total']} solved;
  {100 * control['macro_accuracy']:.3f}% five-domain macro.
- Delta: {delta['solved']:+d} solved;
  {100 * delta['macro_accuracy']:+.3f} macro points.
- AIME is reported separately: {treatment['aime']['correct']}/{treatment['aime']['total']}.

| Domain | Exact |
|---|---:|
{rows}

See `product_report.json` for the complete matched comparison and gate.
"""


def package(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise IDRReleaseError(f"refusing existing output: {args.output}")
    config = args.model_root / "config.json"
    config_sha = require_file(
        config, args.expected_model_config_sha256, "model config"
    )
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
    product = load_product_report(
        args.product_report, args.expected_product_report_sha256
    )
    prompt_sha: str | None = None
    if args.interaction_prompts is not None:
        if args.expected_interaction_prompts_sha256 is None:
            raise IDRReleaseError("interaction prompt SHA-256 is required")
        prompt_sha = require_file(
            args.interaction_prompts,
            args.expected_interaction_prompts_sha256,
            "interaction prompts",
        )

    temporary = args.output.with_name(f".{args.output.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise IDRReleaseError(f"refusing existing temporary output: {temporary}")
    temporary.mkdir(parents=True)
    try:
        payloads = {
            "config.json": config,
            "draft_adapter.pt": args.draft_checkpoint,
            "revision_adapter.pt": args.revision_checkpoint,
            "product_report.json": args.product_report,
        }
        if args.interaction_prompts is not None:
            payloads["interaction_prompts.jsonl"] = args.interaction_prompts
        for name, source in payloads.items():
            shutil.copy2(source, temporary / name)

        manifest = {
            "schema": "shohin-idr-release-v1",
            "status": "qualified",
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "inference_path": ["draft_adapter.pt", "revision_adapter.pt"],
            "base_model_included": False,
            "source": {
                "model_root": str(args.model_root.resolve()),
                "model_config_sha256": config_sha,
                "draft_checkpoint": str(args.draft_checkpoint.resolve()),
                "draft_checkpoint_sha256": draft_sha,
                "revision_checkpoint": str(args.revision_checkpoint.resolve()),
                "revision_checkpoint_sha256": revision_sha,
                "product_report": str(args.product_report.resolve()),
                "product_report_sha256": args.expected_product_report_sha256,
                "interaction_prompts_sha256": prompt_sha,
            },
            "product_summary": product["treatment_summary"],
            "matched_control_summary": product["control_summary"],
            "deltas": product["deltas"],
            "gates": product["gates"],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "MODEL_CARD.md").write_text(
            model_card(args.model_name, args.model_revision, product), encoding="utf-8"
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
    parser.add_argument("--draft-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-draft-checkpoint-sha256", required=True)
    parser.add_argument("--revision-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-revision-checkpoint-sha256", required=True)
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
