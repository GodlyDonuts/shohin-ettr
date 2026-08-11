#!/usr/bin/env python3
"""Qualify every nonsealed frozen MBPP reference without publishing data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_pcf1_data import assigned_split
from pcf1_code_sandbox import (
    atomic_json,
    mbpp_allocation_setup_receipts_sha256,
    preflight_mbpp_reference,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
)

SOURCE_SHA256 = "0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398"
SPLIT_SEED = 2026080811


class PCF3ReferenceError(RuntimeError):
    """The isolated reference-admission canary differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qualify(source_bank: Path, output_root: Path) -> dict[str, Any]:
    if (
        not source_bank.is_file()
        or source_bank.is_symlink()
        or sha256_file(source_bank) != SOURCE_SHA256
    ):
        raise PCF3ReferenceError("PCF3 frozen MBPP source differs")
    if output_root.exists() or output_root.is_symlink():
        raise PCF3ReferenceError("PCF3 reference output must be fresh")

    rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    identities: set[str] = set()
    with source_bank.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            identity = row.get("identity_sha256")
            if (
                not isinstance(identity, str)
                or len(identity) != 64
                or any(character not in "0123456789abcdef" for character in identity)
                or identity in identities
                or row.get("task") != "mbpp"
            ):
                raise PCF3ReferenceError("PCF3 frozen MBPP row differs")
            identities.add(identity)
            split = assigned_split(identity, SPLIT_SEED)
            split_counts[split] += 1
            if split == "holdout":
                continue
            rows.append(row)

    output_root.mkdir(parents=True)
    sandbox_receipt = qualify_allocation()
    sandbox_path = output_root / "sandbox_receipt.json"
    sandbox_sha256 = atomic_json(sandbox_path, sandbox_receipt)
    setup_receipts = qualify_mbpp_assessor_setups(rows)
    setup_by_sha256 = {
        str(receipt["setup_source_sha256"]): receipt for receipt in setup_receipts
    }

    reference_receipts: list[dict[str, Any]] = []
    for row in rows:
        setup_source = row.get("test_setup_code", "")
        if not isinstance(setup_source, str):
            raise PCF3ReferenceError("PCF3 frozen setup differs")
        setup_sha256 = hashlib.sha256(setup_source.encode()).hexdigest()
        reference_receipts.append(
            preflight_mbpp_reference(
                row,
                split=assigned_split(str(row["identity_sha256"]), SPLIT_SEED),
                setup_qualification=setup_by_sha256[setup_sha256],
            )
        )
    reference_receipts.sort(key=lambda receipt: str(receipt["identity_sha256"]))
    receipt_digest = hashlib.sha256()
    identity_digest = hashlib.sha256()
    for receipt in reference_receipts:
        receipt_digest.update(
            (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        identity_digest.update(str(receipt["identity_sha256"]).encode())
        identity_digest.update(b"\n")

    report = {
        "schema": "shohin-pcf3-reference-canary-v1",
        "status": "pass",
        "source_bank_sha256": SOURCE_SHA256,
        "split_seed": SPLIT_SEED,
        "source_rows": len(identities),
        "split_counts": dict(sorted(split_counts.items())),
        "nonsealed_reference_rows": len(reference_receipts),
        "ordered_nonsealed_identity_sha256": identity_digest.hexdigest(),
        "reference_receipts_sha256": receipt_digest.hexdigest(),
        "unique_setups": len(setup_receipts),
        "setup_receipts_sha256": mbpp_allocation_setup_receipts_sha256(setup_receipts),
        "sandbox_receipt_sha256": sandbox_sha256,
        "reference_assessment_mode": "trusted_reference",
        "generated_candidate_policy_applied": False,
        "all_references_passed": len(reference_receipts) == len(rows),
        "all_sandbox_passed": True,
        "holdout_reference_content_accesses": 0,
        "model_opened": False,
        "assessor_published": False,
        "scientific_score_emitted": False,
    }
    atomic_json(output_root / "report.json", report)
    directory = os.open(output_root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    report = qualify(**vars(parser.parse_args()))
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
