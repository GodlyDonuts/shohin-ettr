#!/usr/bin/env python3
"""Atomically mirror the completed upward-MoE publication into durable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from q36_mtr_evidence import verify_evidence_snapshot

PUBLICATION_SCHEMA = "shohin-upward-moe-publication-evidence-v1"
SCHEMA = "shohin-upward-moe-publication-mirror-v1"
FIGURE_FILES = {
    "shohin-upward-moe-scaling.svg": "figure_svg",
    "shohin-upward-moe-scaling-points.csv": "figure_points",
}


class UpwardMoEMirrorError(RuntimeError):
    """The upward-MoE publication cannot be durably mirrored exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise UpwardMoEMirrorError("mirror input path differs")
    return path.resolve(strict=True)


def _safe_output(authorized_root: Path, output_root: Path) -> Path:
    if (
        not authorized_root.is_absolute()
        or authorized_root.is_symlink()
        or not authorized_root.is_dir()
        or not output_root.is_absolute()
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise UpwardMoEMirrorError("authorized mirror root differs")
    authorized = authorized_root.resolve(strict=True)
    output = output_root.resolve(strict=False)
    if authorized in {Path("/"), Path.home().resolve()}:
        raise UpwardMoEMirrorError("authorized mirror root is too broad")
    try:
        relative = output.relative_to(authorized)
    except ValueError as error:
        raise UpwardMoEMirrorError("mirror output escapes authorization") from error
    if not relative.parts:
        raise UpwardMoEMirrorError("mirror output equals authorization root")
    return output


def _load_publication(path: Path) -> dict[str, Any]:
    path = _regular(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UpwardMoEMirrorError("publication receipt is unreadable") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != PUBLICATION_SCHEMA
        or value.get("status") != "complete"
        or value.get("scientific_scores_changed") is not False
        or value.get("automatic_successor_authorized") is not False
    ):
        raise UpwardMoEMirrorError("publication receipt differs")
    return value


def _expected_hashes(records: Any, label: str) -> set[str]:
    if not isinstance(records, list):
        raise UpwardMoEMirrorError(f"{label} records differ")
    values = {record.get("sha256") for record in records if isinstance(record, dict)}
    if len(values) != len(records) or any(
        not isinstance(value, str) or len(value) != 64 for value in values
    ):
        raise UpwardMoEMirrorError(f"{label} records differ")
    return values


def mirror(
    *,
    authorized_root: Path,
    output_root: Path,
    publication_receipt: Path,
    analysis: Path,
    figure_root: Path,
    accounting: list[Path],
    source_points: list[Path],
) -> dict[str, Any]:
    output = _safe_output(authorized_root, output_root)
    publication_path = _regular(publication_receipt)
    publication = _load_publication(publication_path)
    analysis_path = _regular(analysis)
    if sha256_file(analysis_path) != publication.get("analysis", {}).get("sha256"):
        raise UpwardMoEMirrorError("mirrored analysis differs")
    if (
        not figure_root.is_absolute()
        or figure_root.is_symlink()
        or not figure_root.is_dir()
    ):
        raise UpwardMoEMirrorError("figure root differs")
    figure_manifest = _regular(figure_root / "manifest.json")
    if sha256_file(figure_manifest) != publication.get("figure_manifest_sha256"):
        raise UpwardMoEMirrorError("figure manifest differs")
    figure_records = publication.get("figure_records")
    if not isinstance(figure_records, list):
        raise UpwardMoEMirrorError("figure records differ")
    figure_expected = {
        record.get("name"): record.get("sha256")
        for record in figure_records
        if isinstance(record, dict)
    }
    if set(figure_expected) != set(FIGURE_FILES) or len(figure_expected) != len(
        figure_records
    ):
        raise UpwardMoEMirrorError("figure records differ")
    figures: list[tuple[str, Path]] = []
    for filename, name in FIGURE_FILES.items():
        path = _regular(figure_root / filename)
        if sha256_file(path) != figure_expected[filename]:
            raise UpwardMoEMirrorError("figure asset differs")
        figures.append((name, path))

    if len(accounting) != 2 or len(source_points) != 3:
        raise UpwardMoEMirrorError("mirror input geometry differs")
    accounting_paths = [_regular(path) for path in accounting]
    point_paths = [_regular(path) for path in source_points]
    if {sha256_file(path) for path in accounting_paths} != _expected_hashes(
        publication.get("accounting_records"), "accounting"
    ):
        raise UpwardMoEMirrorError("accounting mirror set differs")
    expected_points = publication.get("analysis", {}).get("point_source_sha256s")
    if (
        not isinstance(expected_points, list)
        or len(set(expected_points)) != 3
        or {sha256_file(path) for path in point_paths} != set(expected_points)
    ):
        raise UpwardMoEMirrorError("source point mirror set differs")

    primary: dict[str, Path] = {
        "publication_evidence": publication_path,
        "analysis": analysis_path,
        "figure_manifest": figure_manifest,
        **dict(figures),
    }
    for index, path in enumerate(sorted(accounting_paths, key=sha256_file)):
        primary[f"accounting_{index:02d}"] = path
    for index, path in enumerate(sorted(point_paths, key=sha256_file)):
        primary[f"source_point_{index:02d}"] = path

    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.mkdir(parents=True, mode=0o700)
    records = []
    try:
        artifacts = temporary / "artifacts"
        artifacts.mkdir(mode=0o700)
        for name, source in sorted(primary.items()):
            suffix = "".join(source.suffixes)
            destination = artifacts / f"{name}{suffix}"
            with source.open("rb") as source_handle, destination.open(
                "xb"
            ) as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, 1 << 20)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            digest = sha256_file(destination)
            if digest != sha256_file(source):
                raise UpwardMoEMirrorError("mirrored bytes differ")
            os.chmod(destination, 0o444)
            records.append(
                {
                    "name": name,
                    "primary": str(source),
                    "mirror": str(output / "artifacts" / destination.name),
                    "sha256": digest,
                    "bytes": destination.stat().st_size,
                }
            )
        tree_rows = [
            {"name": row["name"], "sha256": row["sha256"], "bytes": row["bytes"]}
            for row in records
        ]
        tree_digest = hashlib.sha256(
            b"".join(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for row in sorted(tree_rows, key=lambda value: value["name"])
            )
        ).hexdigest()
        payload = {
            "schema": SCHEMA,
            "status": "complete",
            "artifact_count": len(records),
            "artifact_sha256s": {row["name"]: row["sha256"] for row in records},
            "records": records,
            "artifact_tree_sha256": tree_digest,
            "publication_receipt_sha256": sha256_file(publication_path),
            "primary_mirror_hashes_exact": True,
            "write_once_snapshot": True,
            "scientific_scores_changed": False,
            "automatic_successor_authorized": False,
        }
        manifest = temporary / "manifest.json"
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        with manifest.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(manifest, 0o444)
        os.chmod(artifacts, 0o555)
        os.chmod(temporary, 0o555)
        os.replace(temporary, output)
        replay = verify_evidence_snapshot(output / "manifest.json", payload)
        if replay.get("artifact_tree_sha256") != tree_digest:
            raise UpwardMoEMirrorError("durable mirror replay differs")
        return payload
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--accounting", type=Path, action="append", required=True)
    parser.add_argument("--source-point", type=Path, action="append", required=True)
    args = parser.parse_args()
    result = mirror(
        authorized_root=args.authorized_root,
        output_root=args.output_root,
        publication_receipt=args.publication_receipt,
        analysis=args.analysis,
        figure_root=args.figure_root,
        accounting=args.accounting,
        source_points=args.source_point,
    )
    print(
        json.dumps(
            {"status": result["status"], "artifact_count": result["artifact_count"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
