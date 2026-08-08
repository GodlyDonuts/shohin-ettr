#!/usr/bin/env python3
"""Audit quality admission for every corpus in a Phase-2 training contract."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = _ROOT / "train"
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from data_contract import resolve_training_data_contract  # noqa: E402
from pipeline.tokenize_shards import (  # noqa: E402
    canonical_payload_sha256,
    file_receipt,
)


ADMISSION_SCHEMA = "shohin-phase2-corpus-admission-v1"
BUNDLE_SCHEMA = "shohin-phase2-admission-bundle-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
LEVELS = {"canary": 0, "production": 1}
REQUIRED_CHECKS = {
    "cross_source_residualization_complete",
    "document_holdout_frozen",
    "domain_holdout_frozen",
    "eval_contamination_scan_complete",
    "eval_exact_overlap_removed",
    "exact_dedup_complete",
    "license_review_complete",
    "near_dedup_complete",
    "privacy_scan_complete",
    "provenance_complete",
    "redistribution_terms_recorded",
    "retained_sample_review_complete",
    "severe_privacy_findings_zero",
    "training_permitted",
}
REQUIRED_UTILITY = {
    "aggregate_utility_nonnegative",
    "equal_token_gate_complete",
    "protected_regression_within_floor",
}


class Phase2AdmissionError(RuntimeError):
    """A physical corpus lacks the evidence required for optimizer use."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase2AdmissionError(f"admission JSON is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise Phase2AdmissionError("admission report is not an object")
    return value


def _parse_admission(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    path = Path(raw_path)
    if not separator or not name or not path.is_absolute():
        raise argparse.ArgumentTypeError("expected corpus_name=/absolute/admission.json")
    return name, path


def validate_admission(
    path: Path,
    corpus: dict[str, Any],
    *,
    required_level: str,
) -> dict[str, Any]:
    report = _load_json(path)
    claimed = report.get("payload_sha256")
    unsigned = dict(report)
    unsigned.pop("payload_sha256", None)
    if (
        report.get("schema") != ADMISSION_SCHEMA
        or report.get("status") != "admitted"
        or not isinstance(claimed, str)
        or HEX64.fullmatch(claimed) is None
        or canonical_payload_sha256(unsigned) != claimed
    ):
        raise Phase2AdmissionError("corpus admission payload differs")
    level = report.get("admission_level")
    if level not in LEVELS or LEVELS[level] < LEVELS[required_level]:
        raise Phase2AdmissionError("corpus admission level is insufficient")
    if (
        report.get("corpus_name") != corpus["name"]
        or report.get("manifest_payload_sha256")
        != corpus["manifest_payload_sha256"]
    ):
        raise Phase2AdmissionError("corpus admission identity differs")
    manifest_path = Path(corpus["path"]) / "manifest.json"
    manifest = _load_json(manifest_path)
    if (
        report.get("unique_tokens") != manifest.get("tokens")
        or report.get("documents") != manifest.get("kept")
        or not isinstance(report.get("fresh_source"), bool)
    ):
        raise Phase2AdmissionError("corpus admission size/freshness differs")
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CHECKS:
        raise Phase2AdmissionError("corpus admission check inventory differs")
    if not all(value is True for value in checks.values()):
        raise Phase2AdmissionError("one or more corpus admission checks failed")
    utility = report.get("utility")
    if not isinstance(utility, dict) or set(utility) != REQUIRED_UTILITY:
        raise Phase2AdmissionError("corpus utility check inventory differs")
    if required_level == "production" and not all(
        value is True for value in utility.values()
    ):
        raise Phase2AdmissionError("production corpus utility is unqualified")

    evidence = report.get("evidence_receipts")
    if not isinstance(evidence, list) or len(evidence) < 6:
        raise Phase2AdmissionError("corpus admission evidence is incomplete")
    labels = set()
    evidence_bindings = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"label", "path", "sha256"}:
            raise Phase2AdmissionError("corpus admission evidence fields differ")
        label = item["label"]
        evidence_path = Path(str(item["path"]))
        if (
            not isinstance(label, str)
            or not label
            or label in labels
            or not evidence_path.is_absolute()
            or not isinstance(item["sha256"], str)
            or HEX64.fullmatch(item["sha256"]) is None
        ):
            raise Phase2AdmissionError("corpus admission evidence identity differs")
        observed = file_receipt(evidence_path)
        if observed["sha256"] != item["sha256"]:
            raise Phase2AdmissionError("corpus admission evidence SHA-256 differs")
        labels.add(label)
        evidence_bindings.append(observed)
    return {
        "path": str(path.resolve()),
        "sha256": file_receipt(path)["sha256"],
        "payload_sha256": claimed,
        "admission_level": level,
        "fresh_source": report["fresh_source"],
        "unique_tokens": report["unique_tokens"],
        "documents": report["documents"],
        "utility": utility,
        "evidence": evidence_bindings,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise Phase2AdmissionError(f"refusing existing output: {args.output}")
    if args.level not in LEVELS:
        raise Phase2AdmissionError("requested admission level differs")
    if args.minimum_unique_tokens <= 0:
        raise Phase2AdmissionError("minimum unique-token floor must be positive")
    resolution = resolve_training_data_contract(
        args.contract,
        expected_sha256=args.contract_sha256,
        deep_verify=args.deep_verify,
    )
    supplied = dict(args.admission)
    if len(supplied) != len(args.admission):
        raise Phase2AdmissionError("duplicate corpus admission name")
    expected_names = {corpus["name"] for corpus in resolution["corpora"]}
    if set(supplied) != expected_names:
        raise Phase2AdmissionError("admission reports do not cover the contract exactly")

    admitted = {}
    unique_tokens = 0
    fresh_weight = 0.0
    for corpus, normalized_weight in zip(
        resolution["corpora"], resolution["domain_weights"], strict=True
    ):
        evidence = validate_admission(
            supplied[corpus["name"]], corpus, required_level=args.level
        )
        admitted[corpus["name"]] = evidence
        unique_tokens += evidence["unique_tokens"]
        fresh_weight += normalized_weight * float(evidence["fresh_source"])
    gates = {
        "minimum_unique_tokens_met": unique_tokens >= args.minimum_unique_tokens,
        "mostly_fresh_sampling_weight": fresh_weight >= 0.70,
        "all_corpora_admitted": len(admitted) == len(resolution["corpora"]),
        "production_utility_complete": args.level != "production"
        or all(all(item["utility"].values()) for item in admitted.values()),
    }
    report = {
        "schema": BUNDLE_SCHEMA,
        "status": "admitted" if all(gates.values()) else "rejected",
        "requested_level": args.level,
        "contract": resolution["contract"],
        "contract_payload_sha256": resolution["contract_payload_sha256"],
        "deep_verified": resolution["deep_verified"],
        "tokenizer_sha256": resolution["tokenizer_sha256"],
        "tokenizer_vocab_size": resolution["tokenizer_vocab_size"],
        "unique_tokens": unique_tokens,
        "minimum_unique_tokens": args.minimum_unique_tokens,
        "fresh_sampling_weight": fresh_weight,
        "normalized_domain_weights": resolution["domain_weights"],
        "corpora": admitted,
        "gates": gates,
        "training_eligible": all(gates.values()),
    }
    if not math.isclose(sum(resolution["domain_weights"]), 1.0):
        raise Phase2AdmissionError("normalized corpus weights do not sum to one")
    report["payload_sha256"] = canonical_payload_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--admission", action="append", type=_parse_admission, required=True)
    parser.add_argument("--level", choices=sorted(LEVELS), required=True)
    parser.add_argument("--minimum-unique-tokens", type=int, required=True)
    parser.add_argument("--deep-verify", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    report = audit(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
