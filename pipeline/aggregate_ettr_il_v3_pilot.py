#!/usr/bin/env python3
"""Independently aggregate the complete ETTR-IL-v3 Stokes pilot matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

from ettr_il_v3_pilot import SCHEMA as CELL_SCHEMA, pilot_cells
from ettr_il_v3_protocol import (
    CURRICULUM_STAGES,
    FAMILIES,
    PROTOCOL,
    SPLITS,
    candidate_floor,
    canonical_json_bytes,
    split_stage_family_allocation,
)


SCHEMA = "r12-ettr-il-v3-pilot-aggregate-v1"


class PilotAggregateError(ValueError):
    """The Stokes pilot matrix is incomplete, malformed, or inconsistent."""


def _load_canonical(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise PilotAggregateError(f"pilot report is a symlink: {path.name}")
    status = path.stat()
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or status.st_size > 2 * 1024 * 1024
    ):
        raise PilotAggregateError(f"pilot report is not a safe file: {path.name}")
    payload = path.read_bytes()
    try:
        value = json.loads(
            payload,
            parse_constant=lambda item: (_ for _ in ()).throw(
                PilotAggregateError(f"non-finite pilot value: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotAggregateError(
            f"pilot report is not canonical JSON: {path.name}"
        ) from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise PilotAggregateError(
            f"pilot report is not canonical JSON: {path.name}"
        )
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotAggregateError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise PilotAggregateError(f"{name} is not finite and positive")
    return result


def _verify_report(
    report: Mapping[str, object],
    *,
    expected_index: int,
    expected_source_commit: str | None,
    expected_freeze: str | None,
) -> tuple[str, str]:
    if report.get("schema") != CELL_SCHEMA or report.get("protocol") != PROTOCOL:
        raise PilotAggregateError("pilot report schema or protocol differs")
    if report.get("status") != "pass":
        raise PilotAggregateError("pilot report did not pass")
    if report.get("primary_replay_mismatches") != 0:
        raise PilotAggregateError("pilot report contains semantic mismatches")
    cell = report.get("cell")
    if cell != pilot_cells()[expected_index].to_value():
        raise PilotAggregateError("pilot report cell identity differs")
    report_sha256 = report.get("report_sha256")
    if (
        not isinstance(report_sha256, str)
        or len(report_sha256) != 64
        or any(character not in "0123456789abcdef" for character in report_sha256)
    ):
        raise PilotAggregateError("pilot report SHA-256 differs")
    unhashed = dict(report)
    del unhashed["report_sha256"]
    if hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest() != report_sha256:
        raise PilotAggregateError("pilot report self-hash differs")
    source_commit = report.get("source_commit")
    freeze = report.get("protocol_freeze_sha256")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise PilotAggregateError("pilot source commit differs")
    if not isinstance(freeze, str) or len(freeze) != 64:
        raise PilotAggregateError("pilot protocol freeze differs")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise PilotAggregateError("pilot source commits are inconsistent")
    if expected_freeze is not None and freeze != expected_freeze:
        raise PilotAggregateError("pilot freezes are inconsistent")
    cores = report.get("cores")
    if type(cores) is not int or cores < 1:
        raise PilotAggregateError("pilot core count differs")
    _positive_number(report.get("cpu_seconds"), "pilot CPU seconds")
    _positive_number(
        report.get("compressed_bytes_per_core"),
        "pilot compressed bytes per core",
    )
    _positive_number(report.get("peak_rss_kib"), "pilot peak RSS")
    return source_commit, freeze


def aggregate_reports(report_directory: Path) -> dict[str, object]:
    cells = pilot_cells()
    expected_names = {f"cell-{index}.json" for index in range(len(cells))}
    observed_names = {
        path.name for path in report_directory.iterdir() if path.is_file()
    }
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise PilotAggregateError(
            f"pilot report inventory differs; missing={missing}, "
            f"unexpected={unexpected}"
        )

    reports: list[dict[str, object]] = []
    source_commit: str | None = None
    freeze: str | None = None
    for index in range(len(cells)):
        report = _load_canonical(report_directory / f"cell-{index}.json")
        source_commit, freeze = _verify_report(
            report,
            expected_index=index,
            expected_source_commit=source_commit,
            expected_freeze=freeze,
        )
        reports.append(report)

    stage_rates: dict[tuple[str, str], float] = {}
    stage_bytes: dict[tuple[str, str], float] = {}
    for family in FAMILIES:
        for stage in CURRICULUM_STAGES:
            matching = tuple(
                report
                for report in reports
                if report["cell"]["family"] == family
                and report["cell"]["stage"] == stage
            )
            if not matching:
                raise PilotAggregateError("pilot stage coverage differs")
            stage_rates[(family, stage)] = min(
                _positive_number(
                    report["cores_per_cpu_second"],
                    "pilot cores per CPU second",
                )
                for report in matching
            )
            stage_bytes[(family, stage)] = max(
                _positive_number(
                    report["compressed_bytes_per_core"],
                    "pilot compressed bytes per core",
                )
                for report in matching
            )

    candidate_cores = 0
    projected_cpu_seconds = 0.0
    projected_compressed_bytes = 0.0
    candidate_cells: list[dict[str, object]] = []
    for split in SPLITS:
        matrix = split_stage_family_allocation(split)
        for stage in CURRICULUM_STAGES:
            for family in FAMILIES:
                selected_quota = matrix[stage][family]
                floor = candidate_floor(selected_quota)
                rate = stage_rates[(family, stage)]
                bytes_per_core = stage_bytes[(family, stage)]
                cpu_seconds = floor / rate
                compressed_bytes = floor * bytes_per_core
                candidate_cores += floor
                projected_cpu_seconds += cpu_seconds
                projected_compressed_bytes += compressed_bytes
                candidate_cells.append(
                    {
                        "candidate_floor": floor,
                        "conservative_compressed_bytes": compressed_bytes,
                        "conservative_cpu_seconds": cpu_seconds,
                        "family": family,
                        "selected_quota": selected_quota,
                        "split": split,
                        "stage": stage,
                    }
                )

    aggregate: dict[str, object] = {
        "candidate_cells": candidate_cells,
        "candidate_population_floor": candidate_cores,
        "cell_count": len(reports),
        "conservative_projected_compressed_bytes": projected_compressed_bytes,
        "conservative_projected_cpu_hours": projected_cpu_seconds / 3600,
        "max_observed_compressed_bytes_per_core": max(
            _positive_number(
                report["compressed_bytes_per_core"],
                "pilot compressed bytes per core",
            )
            for report in reports
        ),
        "max_observed_peak_rss_kib": max(
            _positive_number(report["peak_rss_kib"], "pilot peak RSS")
            for report in reports
        ),
        "min_observed_cores_per_cpu_second": min(
            _positive_number(
                report["cores_per_cpu_second"],
                "pilot cores per CPU second",
            )
            for report in reports
        ),
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": freeze,
        "schema": SCHEMA,
        "source_commit": source_commit,
        "status": "pass",
    }
    aggregate["aggregate_sha256"] = hashlib.sha256(
        canonical_json_bytes(aggregate)
    ).hexdigest()
    return aggregate


def write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    aggregate = aggregate_reports(args.reports)
    write_no_replace(args.out, canonical_json_bytes(aggregate))
    print(
        json.dumps(
            {
                "aggregate_sha256": aggregate["aggregate_sha256"],
                "candidate_population_floor": aggregate[
                    "candidate_population_floor"
                ],
                "out": str(args.out),
                "status": aggregate["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
