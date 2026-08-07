#!/usr/bin/env python3
"""Validate the sole pass-only EIC1 confirmation authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ASSESSMENT_SCHEMA = "shohin-diverge-eic1-assessment-v1"
BOARD_REPORT_SCHEMA = "shohin-diverge-eic1-confirmation-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authorize(
    assessment: dict,
    board_report: dict,
    *,
    board_sha256: str,
) -> None:
    if (
        assessment.get("schema") != ASSESSMENT_SCHEMA
        or assessment.get("passed") is not True
        or assessment.get("selected") != "shohin_involution"
        or assessment.get("confirmation_access_authorized") is not True
    ):
        raise RuntimeError("EIC1 development did not authorize confirmation")
    if (
        board_report.get("schema") != BOARD_REPORT_SCHEMA
        or board_report.get("board_sha256") != board_sha256
        or board_report.get("model_score_used_for_selection") is not False
        or board_report.get("generated_before_eic1_development_result") is not True
        or any(board_report.get("overlap", {}).values())
    ):
        raise RuntimeError("EIC1 confirmation board receipt differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--assessment-sha256", required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--board-sha256", required=True)
    parser.add_argument("--board-report", type=Path, required=True)
    parser.add_argument("--board-report-sha256", required=True)
    args = parser.parse_args()
    for path, expected, label in (
        (args.assessment, args.assessment_sha256, "assessment"),
        (args.board, args.board_sha256, "board"),
        (args.board_report, args.board_report_sha256, "board report"),
    ):
        if sha256_path(path) != expected:
            raise SystemExit(f"EIC1 {label} hash differs")
    assessment = json.loads(args.assessment.read_text(encoding="utf-8"))
    board_report = json.loads(args.board_report.read_text(encoding="utf-8"))
    authorize(assessment, board_report, board_sha256=args.board_sha256)
    print(
        json.dumps(
            {
                "authorized": True,
                "assessment_sha256": args.assessment_sha256,
                "board_sha256": args.board_sha256,
                "board_report_sha256": args.board_report_sha256,
                "selected": "shohin_involution",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
