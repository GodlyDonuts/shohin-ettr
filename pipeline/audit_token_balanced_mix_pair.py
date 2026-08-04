#!/usr/bin/env python3
"""Fail closed unless two reasoning mixes form an interpretable data-quality pair."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "shohin-token-balanced-mix-pair-audit-v1"
MIX_SCHEMA = "shohin-token-balanced-reasoning-mix-v1"
SHARED_REPORT_FIELDS = (
    "model_revision",
    "tokenizer_name_or_path",
    "max_sequence_length",
    "workspace_slots",
    "weights",
    "requested_total_target_tokens",
    "seed",
)


class MixPairAuditError(RuntimeError):
    """The proposed control/treatment pair is not causally interpretable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_question(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    if not normalized:
        raise MixPairAuditError("row has an empty question")
    return normalized


def _read_report(path: Path, data_path: Path) -> dict[str, Any]:
    if not path.is_file() or not data_path.is_file():
        raise MixPairAuditError(f"missing data/report pair: {data_path}, {path}")
    try:
        report = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise MixPairAuditError(f"invalid report: {path}") from exc
    if report.get("schema") != MIX_SCHEMA or report.get("status") != "complete":
        raise MixPairAuditError(f"incomplete or incompatible report: {path}")
    observed = sha256_file(data_path)
    if report.get("output_sha256") != observed:
        raise MixPairAuditError(f"data hash differs from report: {data_path}")
    return report


def _read_rows(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    by_identity: dict[str, dict[str, Any]] = {}
    groups: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MixPairAuditError(
                    f"malformed row at {path}:{line_number}"
                ) from exc
            question = row.get("question") or row.get("prompt")
            group = row.get("training_group") or row.get("domain")
            if not isinstance(question, str) or not isinstance(group, str):
                raise MixPairAuditError(
                    f"row lacks question/group at {path}:{line_number}"
                )
            identity = hashlib.sha256(
                normalized_question(question).encode()
            ).hexdigest()
            if identity in by_identity:
                raise MixPairAuditError(f"duplicate normalized question in {path}")
            by_identity[identity] = row
            groups[group] += 1
    return by_identity, groups


def _validate_report_counts(
    report: dict[str, Any], rows: dict[str, dict[str, Any]], groups: Counter[str]
) -> None:
    if report.get("selected_rows") != len(rows):
        raise MixPairAuditError("selected row count differs from report")
    reported_groups = report.get("selected_groups")
    if not isinstance(reported_groups, dict):
        raise MixPairAuditError("selected group report is missing")
    expected = {group: count for group, count in sorted(groups.items())}
    observed = {
        group: metrics.get("rows")
        for group, metrics in sorted(reported_groups.items())
        if isinstance(metrics, dict)
    }
    if observed != expected:
        raise MixPairAuditError("selected group row counts differ from report")


def audit_pair(
    control_data: Path,
    control_report_path: Path,
    treatment_data: Path,
    treatment_report_path: Path,
    *,
    treatment_group: str,
    treatment_subtype: str,
    minimum_subtype_fraction: float,
) -> dict[str, Any]:
    if not 0.0 <= minimum_subtype_fraction <= 1.0:
        raise MixPairAuditError("minimum subtype fraction must be in [0, 1]")
    control_report = _read_report(control_report_path, control_data)
    treatment_report = _read_report(treatment_report_path, treatment_data)
    mismatches = {
        field: {
            "control": control_report.get(field),
            "treatment": treatment_report.get(field),
        }
        for field in SHARED_REPORT_FIELDS
        if control_report.get(field) != treatment_report.get(field)
    }
    if mismatches:
        raise MixPairAuditError(
            f"shared mix contract differs: {json.dumps(mismatches, sort_keys=True)}"
        )

    control_rows, control_groups = _read_rows(control_data)
    treatment_rows, treatment_groups = _read_rows(treatment_data)
    _validate_report_counts(control_report, control_rows, control_groups)
    _validate_report_counts(treatment_report, treatment_rows, treatment_groups)

    control_non_treatment = {
        identity: row
        for identity, row in control_rows.items()
        if (row.get("training_group") or row.get("domain")) != treatment_group
    }
    treatment_non_treatment = {
        identity: row
        for identity, row in treatment_rows.items()
        if (row.get("training_group") or row.get("domain")) != treatment_group
    }
    if control_non_treatment != treatment_non_treatment:
        missing = sorted(set(control_non_treatment) - set(treatment_non_treatment))
        added = sorted(set(treatment_non_treatment) - set(control_non_treatment))
        changed = sorted(
            identity
            for identity in set(control_non_treatment) & set(treatment_non_treatment)
            if control_non_treatment[identity] != treatment_non_treatment[identity]
        )
        raise MixPairAuditError(
            "non-treatment selections differ: "
            f"missing={missing[:4]} added={added[:4]} changed={changed[:4]}"
        )

    treatment_group_rows = [
        row
        for row in treatment_rows.values()
        if (row.get("training_group") or row.get("domain")) == treatment_group
    ]
    if not treatment_group_rows:
        raise MixPairAuditError("treatment has no rows in the treatment group")
    subtype_rows = [
        row
        for row in treatment_group_rows
        if row.get("reasoning_subtype") == treatment_subtype
        and row.get("verification") == "execution_verified_source_tests"
    ]
    subtype_fraction = len(subtype_rows) / len(treatment_group_rows)
    if subtype_fraction < minimum_subtype_fraction:
        raise MixPairAuditError(
            f"treatment subtype fraction {subtype_fraction:.6f} is below "
            f"required {minimum_subtype_fraction:.6f}"
        )

    control_group_ids = {
        identity
        for identity, row in control_rows.items()
        if (row.get("training_group") or row.get("domain")) == treatment_group
    }
    treatment_group_ids = {
        identity
        for identity, row in treatment_rows.items()
        if (row.get("training_group") or row.get("domain")) == treatment_group
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "control": {
            "data": str(control_data.resolve()),
            "data_sha256": control_report["output_sha256"],
            "report": str(control_report_path.resolve()),
            "report_sha256": sha256_file(control_report_path),
            "rows": len(control_rows),
        },
        "treatment": {
            "data": str(treatment_data.resolve()),
            "data_sha256": treatment_report["output_sha256"],
            "report": str(treatment_report_path.resolve()),
            "report_sha256": sha256_file(treatment_report_path),
            "rows": len(treatment_rows),
        },
        "shared_non_treatment_rows": len(control_non_treatment),
        "treatment_group": treatment_group,
        "control_treatment_group_rows": len(control_group_ids),
        "treatment_group_rows": len(treatment_group_ids),
        "shared_treatment_group_questions": len(
            control_group_ids & treatment_group_ids
        ),
        "required_treatment_subtype": treatment_subtype,
        "required_treatment_verification": "execution_verified_source_tests",
        "treatment_subtype_rows": len(subtype_rows),
        "treatment_subtype_fraction": subtype_fraction,
        "minimum_treatment_subtype_fraction": minimum_subtype_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-data", required=True, type=Path)
    parser.add_argument("--control-report", required=True, type=Path)
    parser.add_argument("--treatment-data", required=True, type=Path)
    parser.add_argument("--treatment-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--treatment-group", default="code")
    parser.add_argument("--treatment-subtype", default="ocr2_execution_verified")
    parser.add_argument("--minimum-subtype-fraction", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise MixPairAuditError(f"refusing to replace output: {args.output}")
    report = audit_pair(
        args.control_data,
        args.control_report,
        args.treatment_data,
        args.treatment_report,
        treatment_group=args.treatment_group,
        treatment_subtype=args.treatment_subtype,
        minimum_subtype_fraction=args.minimum_subtype_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
