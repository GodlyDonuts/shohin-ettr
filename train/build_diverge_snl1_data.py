#!/usr/bin/env python3
"""Build fresh source-disjoint boards for DIVERGE-SNL1 composition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from build_diverge_sve1_data import (
    _atomic_json,
    _atomic_jsonl,
    _episode_identities,
    _episode_names,
    _episode_sources,
    _iter_jsonl,
    _serialized,
    sha256_path,
)
from diverge_eal1_data import canonical_sha256
from diverge_sve1_data import (
    DEVELOPMENT_EPISODES,
    augment_evaluation_episode,
    validate_evaluation_episode,
)


REPORT_SCHEMA = "shohin-diverge-snl1-data-report-v1"
DEVELOPMENT_SEED = 2026080852
CONFIRMATION_SEEDS = (
    2026080853,
    2026080854,
    2026080855,
    2026080856,
    2026080857,
)


def _load_report(root: Path, expected_sha256: str) -> dict[str, Any]:
    path = root / "report.json"
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"SNL1 parent report hash differs: {path}")
    return json.loads(path.read_text())


def _entry_path(root: Path, entry: Mapping[str, Any]) -> Path:
    return root / Path(str(entry["path"])).name


def _collect_lineage(
    root: Path, report_sha256: str, *, expected_schema: str
) -> tuple[set[str], set[str], set[str], dict[str, Any]]:
    report = _load_report(root, report_sha256)
    if report.get("schema") != expected_schema:
        raise RuntimeError("SNL1 parent data schema differs")
    sources: set[str] = set()
    names: set[str] = set()
    identities: set[str] = set()
    rows = 0
    for label, entries in report["files"].items():
        if label == "training":
            path = _entry_path(root, entries)
            for row in _iter_jsonl(path, str(entries["sha256"])):
                identities.add(str(row["identity_sha256"]))
                for key in ("evidence_sha256", "initial_sha256", "query_sha256"):
                    if key in row:
                        sources.add(str(row[key]))
                if "operation" in row:
                    names.add(str(row["operation"]))
                names.update(str(value) for value in row.get("register_table", ()))
                rows += 1
            continue
        public_entry = entries["public"]
        public_rows = list(
            _iter_jsonl(_entry_path(root, public_entry), str(public_entry["sha256"]))
        )
        sources |= _episode_sources(public_rows)
        names |= _episode_names(public_rows)
        identities |= _episode_identities(public_rows)
        rows += len(public_rows)
    return (
        sources,
        names,
        identities,
        {
            "report_sha256": report_sha256,
            "identity_sha256": report["identity_sha256"],
            "rows": rows,
        },
    )


def _build(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = [
        augment_evaluation_episode(index, seed=seed)
        for index in range(DEVELOPMENT_EPISODES)
    ]
    public = [value[0] for value in pairs]
    assessor = [value[1] for value in pairs]
    for visible, hidden in zip(public, assessor, strict=True):
        validate_evaluation_episode(visible, hidden)
    return public, assessor


def _digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(b"".join(_serialized(row) for row in rows)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sve1-data", type=Path, required=True)
    parser.add_argument("--sve1-data-report-sha256", required=True)
    parser.add_argument("--nls1-data", type=Path, required=True)
    parser.add_argument("--nls1-data-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing SNL1 data output: {args.output}")
    args.output.mkdir(parents=True)

    occupied_sources: set[str] = set()
    occupied_names: set[str] = set()
    occupied_identities: set[str] = set()
    parents = {}
    for label, root, digest, schema in (
        (
            "sve1",
            args.sve1_data,
            args.sve1_data_report_sha256,
            "shohin-diverge-sve1-data-report-v1",
        ),
        (
            "nls1",
            args.nls1_data,
            args.nls1_data_report_sha256,
            "shohin-diverge-nls1-data-report-v1",
        ),
    ):
        sources, names, identities, receipt = _collect_lineage(
            root, digest, expected_schema=schema
        )
        occupied_sources |= sources
        occupied_names |= names
        occupied_identities |= identities
        parents[label] = receipt
    historical = {
        "sources": len(occupied_sources),
        "names": len(occupied_names),
        "identities": len(occupied_identities),
    }

    files = {}
    split_reports = {}
    for label, seed in (
        ("development", DEVELOPMENT_SEED),
        *((f"confirmation_{seed}", seed) for seed in CONFIRMATION_SEEDS),
    ):
        public, assessor = _build(seed)
        sources = _episode_sources(public)
        names = _episode_names(public)
        identities = _episode_identities(public)
        rotations = [0, 0]
        for episode in public:
            rotations[int(episode["table_rotation"])] += 1
        if (
            len(identities) != DEVELOPMENT_EPISODES
            or sources & occupied_sources
            or names & occupied_names
            or identities & occupied_identities
            or min(rotations) < int(DEVELOPMENT_EPISODES * 0.40)
        ):
            raise SystemExit(f"SNL1 {label} overlap/balance audit failed")
        occupied_sources |= sources
        occupied_names |= names
        occupied_identities |= identities
        public_path = args.output / f"{label}_public.jsonl"
        assessor_path = args.output / f"{label}_assessor.jsonl"
        public_rows, public_sha = _atomic_jsonl(public_path, public)
        assessor_rows, assessor_sha = _atomic_jsonl(assessor_path, assessor)
        repeated_public, repeated_assessor = _build(seed)
        if (
            _digest(repeated_public) != public_sha
            or _digest(repeated_assessor) != assessor_sha
        ):
            raise SystemExit(f"SNL1 {label} deterministic regeneration failed")
        files[label] = {
            "public": {
                "path": public_path.name,
                "sha256": public_sha,
                "bytes": public_path.stat().st_size,
                "rows": public_rows,
            },
            "assessor": {
                "path": assessor_path.name,
                "sha256": assessor_sha,
                "bytes": assessor_path.stat().st_size,
                "rows": assessor_rows,
            },
        }
        split_reports[label] = {
            "seed": seed,
            "episodes": DEVELOPMENT_EPISODES,
            "sources": len(sources),
            "names": len(names),
            "identities": len(identities),
            "table_rotations": rotations,
            "deterministic_regeneration": True,
            "identity_sha256": canonical_sha256(
                [episode["identity_sha256"] for episode in public]
            ),
        }

    report = {
        "schema": REPORT_SCHEMA,
        "seeds": {
            "development": DEVELOPMENT_SEED,
            "confirmation": list(CONFIRMATION_SEEDS),
        },
        "parents": parents,
        "historical_counts": historical,
        "zero_source_name_and_identity_overlap": True,
        "split_reports": split_reports,
        "files": files,
    }
    report["identity_sha256"] = canonical_sha256(report)
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    for path in args.output.iterdir():
        os.chmod(path, 0o444)
    os.chmod(args.output, 0o555)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
