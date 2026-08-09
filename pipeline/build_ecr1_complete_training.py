#!/usr/bin/env python3
"""Build the exact ECR1 training subset admitted by sequence custody."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from hf_product_reasoning_train import reservoir_rows_with_sha256


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--custody", type=Path, required=True)
    parser.add_argument("--custody-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=9655)
    parser.add_argument("--data-seed", type=int, default=2026080814)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("output already exists")
    custody_bytes = args.custody.read_bytes()
    if hashlib.sha256(custody_bytes).hexdigest() != args.custody_sha256:
        parser.error("custody hash differs")
    custody = json.loads(custody_bytes)
    sequence = custody.get("sequence_custody", {})
    overflow = sequence.get("overflow_receipts")
    if custody.get("status") != "incompatible" or not isinstance(overflow, list):
        parser.error("custody report does not describe overflows")
    excluded = [str(row["identity_sha256"]) for row in overflow]
    rows, observed_sha256 = reservoir_rows_with_sha256(args.data, args.rows, args.data_seed)
    if observed_sha256 != args.data_sha256 or len(rows) != args.rows:
        parser.error("source data differs")
    kept = []
    observed_excluded = []
    for row in rows:
        identity = hashlib.sha256(row["question"].encode("utf-8")).hexdigest()
        if identity in excluded:
            observed_excluded.append(identity)
        else:
            kept.append(row)
    if sorted(observed_excluded) != sorted(excluded):
        parser.error("overflow identities do not match source occurrences")
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in kept
    )
    output_sha256 = hashlib.sha256(payload).hexdigest()
    report = {
        "schema": "shohin-ecr1-complete-training-v1",
        "status": "complete",
        "source_data": str(args.data.resolve()),
        "source_data_sha256": observed_sha256,
        "custody": str(args.custody.resolve()),
        "custody_sha256": args.custody_sha256,
        "source_rows": len(rows),
        "excluded_rows": len(observed_excluded),
        "excluded_unique_identities": len(set(observed_excluded)),
        "excluded_identity_sha256": sorted(set(observed_excluded)),
        "output_rows": len(kept),
        "output_sha256": output_sha256,
    }
    _atomic_bytes(args.output, payload)
    _atomic_bytes(
        args.report,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        f"[ecr1-data] kept={len(kept)} excluded={len(observed_excluded)} sha={output_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
