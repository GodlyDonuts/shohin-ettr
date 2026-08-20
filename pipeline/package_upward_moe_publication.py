#!/usr/bin/env python3
"""Bind upward-MoE scores, figure assets, and compute into one publication receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

ANALYSIS_SCHEMA = "shohin-upward-moe-scaling-analysis-v1"
FIGURE_SCHEMA = "shohin-upward-moe-scaling-figure-manifest-v1"
ACCOUNTING_SCHEMA = "shohin-upward-moe-slurm-accounting-v1"
SCHEMA = "shohin-upward-moe-publication-evidence-v1"
FIGURE_FILES = {
    "shohin-upward-moe-scaling.svg",
    "shohin-upward-moe-scaling-points.csv",
}


class UpwardMoEPublicationError(RuntimeError):
    """The publication artifact set is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise UpwardMoEPublicationError("publication input path differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UpwardMoEPublicationError("publication input is unreadable") from error
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise UpwardMoEPublicationError("publication input schema differs")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise UpwardMoEPublicationError("publication output differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def package(
    *,
    analysis_path: Path,
    figure_root: Path,
    accounting_paths: list[Path],
    output: Path,
) -> dict[str, Any]:
    analysis = _load(analysis_path, ANALYSIS_SCHEMA)
    if (
        analysis.get("status") != "complete_curve"
        or not isinstance(analysis.get("points"), list)
        or len(analysis["points"]) < 3
        or analysis.get("point_count") != len(analysis["points"])
    ):
        raise UpwardMoEPublicationError("analysis is not publication-complete")
    analysis_hosts = {
        point.get("host") for point in analysis["points"] if isinstance(point, dict)
    }
    point_sources = [
        point.get("source_sha256")
        for point in analysis["points"]
        if isinstance(point, dict)
    ]
    if (
        len(analysis_hosts) != len(analysis["points"])
        or any(not isinstance(host, str) or not host for host in analysis_hosts)
        or len(point_sources) != len(analysis["points"])
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in point_sources
        )
    ):
        raise UpwardMoEPublicationError("analysis point identity differs")
    if (
        not figure_root.is_absolute()
        or figure_root.is_symlink()
        or not figure_root.is_dir()
    ):
        raise UpwardMoEPublicationError("figure root differs")
    manifest_path = figure_root / "manifest.json"
    figure = _load(manifest_path, FIGURE_SCHEMA)
    analysis_sha256 = sha256_file(analysis_path)
    if (
        figure.get("status") != "complete"
        or figure.get("analysis_sha256") != analysis_sha256
        or figure.get("point_source_sha256s") != point_sources
        or figure.get("scientific_scores_changed") is not False
        or figure.get("automatic_successor_authorized") is not False
    ):
        raise UpwardMoEPublicationError("figure analysis binding differs")
    records = figure.get("records")
    if (
        not isinstance(records, list)
        or {record.get("name") for record in records if isinstance(record, dict)}
        != FIGURE_FILES
    ):
        raise UpwardMoEPublicationError("figure record set differs")
    figure_records = []
    for record in records:
        if not isinstance(record, dict):
            raise UpwardMoEPublicationError("figure record differs")
        name = record.get("name")
        path = figure_root / name
        if (
            not isinstance(name, str)
            or path.is_symlink()
            or not path.is_file()
            or record.get("sha256") != sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise UpwardMoEPublicationError("figure asset differs")
        figure_records.append(dict(record))

    if len(accounting_paths) != 2:
        raise UpwardMoEPublicationError("publication accounting count differs")
    accounting_records = []
    hosts: set[str] = set()
    charged_hours = 0.0
    for path in accounting_paths:
        accounting = _load(path, ACCOUNTING_SCHEMA)
        host = accounting.get("host")
        hours = accounting.get("charged_h100_hours")
        source_commit = accounting.get("source_commit")
        identity_sha256 = accounting.get("allocation_identity_sha256")
        allocation_count = accounting.get("allocation_count")
        if (
            accounting.get("status") != "complete"
            or accounting.get("all_required_complete") is not True
            or accounting.get("retry_count") != 0
            or not isinstance(host, str)
            or not host
            or host not in analysis_hosts
            or host in hosts
            or not isinstance(source_commit, str)
            or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
            or not isinstance(identity_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", identity_sha256)
            or isinstance(allocation_count, bool)
            or not isinstance(allocation_count, int)
            or allocation_count < 1
            or isinstance(hours, bool)
            or not isinstance(hours, (int, float))
            or not math.isfinite(float(hours))
            or hours < 0
        ):
            raise UpwardMoEPublicationError("publication accounting differs")
        hosts.add(host)
        charged_hours += float(hours)
        accounting_records.append(
            {
                "host": host,
                "source_commit": source_commit,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "allocation_count": allocation_count,
                "charged_h100_hours": float(hours),
                "allocation_identity_sha256": identity_sha256,
            }
        )
    accounting_records.sort(key=lambda value: value["host"])
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "analysis": {
            "sha256": analysis_sha256,
            "bytes": analysis_path.stat().st_size,
            "claim": analysis.get("claim"),
            "capability_curve_claim": analysis.get("capability_curve_claim"),
            "conservative_retention_curve_claim": analysis.get(
                "conservative_retention_curve_claim"
            ),
            "point_count": analysis.get("point_count"),
            "point_source_sha256s": point_sources,
        },
        "figure_manifest_sha256": sha256_file(manifest_path),
        "figure_records": sorted(figure_records, key=lambda value: value["name"]),
        "accounting_records": accounting_records,
        "total_charged_h100_hours": charged_hours,
        "scientific_scores_changed": False,
        "automatic_successor_authorized": False,
    }
    _atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--accounting", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(
        analysis_path=args.analysis,
        figure_root=args.figure_root,
        accounting_paths=args.accounting,
        output=args.output,
    )
    print(json.dumps({"status": result["status"], "analysis": result["analysis"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
