#!/usr/bin/env python3
"""Resolve one exact upward-MoE materialized data hash for dependent jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_q36_mtr_data import REPORT_SCHEMA
from hf_upward_moe_generate_drafts import host_spec
from upward_moe_role_lineage import sha256_file

KINDS = {"revision_train", "development"}


class UpwardMoEDataReceiptError(RuntimeError):
    """The upward-MoE materialized data receipt differed."""


def resolve_hash(host: str, report_path: Path, kind: str, data: Path) -> str:
    spec = host_spec(host)
    if kind not in KINDS or report_path.is_symlink() or not report_path.is_file():
        raise UpwardMoEDataReceiptError("upward data receipt settings differ")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output = report.get("outputs", {}).get(kind)
    expected_rows = 9_655 if kind == "revision_train" else 1_289
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("model_revision") != spec.model_revision
        or report.get("draft_host") != spec.host
        or report.get("source_disjoint") is not True
        or report.get("model_owned_drafts") is not True
        or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or not isinstance(output, dict)
        or output.get("rows") != expected_rows
        or Path(str(output.get("path", ""))).resolve() != data.resolve()
        or output.get("sha256") != sha256_file(data)
    ):
        raise UpwardMoEDataReceiptError("upward data receipt differs")
    return str(output["sha256"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", choices=("nemotron-super", "mixtral-8x22b"), required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(resolve_hash(args.host, args.report, args.kind, args.data))
