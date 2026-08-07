#!/usr/bin/env python3
"""Audit all frozen CWC1/NPL2 wrapper splits and clean regenerations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from diverge_cwc1_npl2_data import audit_wrapper_records


SCHEMA = "shohin-diverge-cwc1-npl2-wrapper-aggregate-audit-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--repro", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing CWC1/NPL2 aggregate audit")
    if len(args.data) != len(args.repro) or len(args.data) != 6:
        raise SystemExit("CWC1/NPL2 aggregate audit requires six paired splits")
    all_sources = set()
    all_identities = set()
    all_labels = set()
    reports = {}
    byte_identical = True
    for data, repro in zip(args.data, args.repro, strict=True):
        rows = _load(data)
        audit = audit_wrapper_records(rows)
        split = str(rows[0]["split"])
        if split in reports:
            raise SystemExit("CWC1/NPL2 split repeats")
        data_sha256 = sha256_path(data)
        repro_sha256 = sha256_path(repro)
        byte_identical &= data_sha256 == repro_sha256
        for row in rows:
            source = str(row["source_sha256"])
            identity = str(row["identity_sha256"])
            labels = {str(value) for value in row["candidate_labels"]}
            if (
                source in all_sources
                or identity in all_identities
                or labels & all_labels
            ):
                raise SystemExit("CWC1/NPL2 cross-split identity overlap")
            all_sources.add(source)
            all_identities.add(identity)
            all_labels.update(labels)
        reports[split] = {
            "data": str(data),
            "data_sha256": data_sha256,
            "repro": str(repro),
            "repro_sha256": repro_sha256,
            "byte_identical": data_sha256 == repro_sha256,
            "audit": audit,
        }
    result = {
        "schema": SCHEMA,
        "splits": reports,
        "cross_split": {
            "unique_sources": len(all_sources),
            "unique_identities": len(all_identities),
            "unique_candidate_labels": len(all_labels),
            "source_overlap": 0,
            "identity_overlap": 0,
            "candidate_label_overlap": 0,
        },
        "clean_regeneration_byte_identical": byte_identical,
        "all_conditions_passed": byte_identical
        and all(value["audit"]["all_conditions_passed"] for value in reports.values()),
    }
    if not result["all_conditions_passed"]:
        raise SystemExit("CWC1/NPL2 aggregate audit failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    os.chmod(args.output, 0o444)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
