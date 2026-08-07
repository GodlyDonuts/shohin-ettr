#!/usr/bin/env python3
"""Build immutable source-disjoint DIVERGE-OPB1 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from diverge_opb1_data import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_EPISODES,
    DEVELOPMENT_SEED,
    REPORT_SCHEMA,
    TRAIN_ROWS,
    augment_evaluation_episode,
    build_training_record,
    validate_evaluation_episode,
)


SNL1_REPORT_SCHEMA = "shohin-diverge-snl1-data-report-v1"
EXTRA_SOURCE_KEYS = (
    "fully_renamed_source_sha256",
    "operation_scrubbed_sha256",
)


def _load_parent(root: Path, expected_sha256: str) -> dict[str, Any]:
    path = root / "report.json"
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("OPB1 SNL1 parent report hash differs")
    report = json.loads(path.read_text())
    if report.get("schema") != SNL1_REPORT_SCHEMA or not report.get(
        "zero_source_name_and_identity_overlap"
    ):
        raise RuntimeError("OPB1 SNL1 parent data is not qualified")
    return report


def _parent_lineage(
    root: Path, report: Mapping[str, Any]
) -> tuple[set[str], set[str], set[str]]:
    sources: set[str] = set()
    names: set[str] = set()
    identities: set[str] = set()
    for entries in report["files"].values():
        public_entry = entries["public"]
        path = root / Path(str(public_entry["path"])).name
        rows = list(_iter_jsonl(path, str(public_entry["sha256"])))
        sources |= _episode_sources(rows)
        names |= _episode_names(rows)
        identities |= _episode_identities(rows)
    return sources, names, identities


def _training_rows() -> Iterable[dict[str, Any]]:
    for serial in range(TRAIN_ROWS):
        yield build_training_record(serial)


def _training_audit() -> tuple[str, set[str], set[str], list[int]]:
    digest = hashlib.sha256()
    sources: set[str] = set()
    names: set[str] = set()
    targets = [0] * 8
    for row in _training_rows():
        digest.update(_serialized(row))
        sources.add(str(row["source_sha256"]))
        names.update(str(value) for value in row["aliases"])
        names.update(str(value) for value in row["decoy_aliases"])
        names.update(str(value) for value in row["registers"])
        targets[int(row["operation_target"])] += 1
    return digest.hexdigest(), sources, names, targets


def _opb_sources(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    sources = _episode_sources(rows)
    for episode in rows:
        for item in episode["evidence"]:
            sources.update(str(item[key]) for key in EXTRA_SOURCE_KEYS)
    return sources


def _opb_names(
    public: Sequence[Mapping[str, Any]], assessor: Sequence[Mapping[str, Any]]
) -> set[str]:
    names = _episode_names(public)
    names.update(
        str(value)
        for episode in assessor
        for value in episode["operation_scrub_aliases"]
    )
    return names


def _balanced(counts: Sequence[int], total: int, tolerance: float) -> bool:
    expected = total / len(counts)
    return sum(counts) == total and all(
        abs(value - expected) / expected <= tolerance for value in counts
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


def _digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(b"".join(_serialized(row) for row in rows)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snl1-data", type=Path, required=True)
    parser.add_argument("--snl1-data-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing OPB1 data output: {args.output}")
    args.output.mkdir(parents=True)

    parent_report = _load_parent(args.snl1_data, args.snl1_data_report_sha256)
    occupied_sources, occupied_names, occupied_identities = _parent_lineage(
        args.snl1_data, parent_report
    )
    historical = {
        "sources": len(occupied_sources),
        "names": len(occupied_names),
        "identities": len(occupied_identities),
    }

    training_path = args.output / "training.jsonl"
    training_rows, training_sha256 = _atomic_jsonl(training_path, _training_rows())
    regenerated, training_sources, training_names, target_counts = _training_audit()
    if (
        regenerated != training_sha256
        or training_rows != TRAIN_ROWS
        or len(training_sources) != TRAIN_ROWS
        or len(training_names) != TRAIN_ROWS * 18
        or training_sources & occupied_sources
        or training_names & occupied_names
        or not _balanced(target_counts, TRAIN_ROWS, 0.03)
    ):
        raise SystemExit("OPB1 training regeneration/balance/overlap audit failed")
    occupied_sources |= training_sources
    occupied_names |= training_names

    files: dict[str, Any] = {
        "training": {
            "path": training_path.name,
            "sha256": training_sha256,
            "bytes": training_path.stat().st_size,
            "rows": training_rows,
        }
    }
    split_reports = {}
    for label, seed in (
        ("development", DEVELOPMENT_SEED),
        *((f"confirmation_{seed}", seed) for seed in CONFIRMATION_SEEDS),
    ):
        public, assessor = _build(seed)
        sources = _opb_sources(public)
        names = _opb_names(public, assessor)
        identities = _episode_identities(public)
        targets = [0] * 8
        for episode in assessor:
            for target in episode["operation_targets"]:
                targets[int(target)] += 1
        if (
            len(identities) != DEVELOPMENT_EPISODES
            or sources & occupied_sources
            or names & occupied_names
            or identities & occupied_identities
            or not _balanced(targets, DEVELOPMENT_EPISODES * 24, 0.05)
        ):
            raise SystemExit(f"OPB1 {label} overlap/balance audit failed")
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
            raise SystemExit(f"OPB1 {label} deterministic regeneration failed")
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
            "target_counts": targets,
            "deterministic_regeneration": True,
            "identity_sha256": canonical_sha256(
                [episode["identity_sha256"] for episode in public]
            ),
        }

    report = {
        "schema": REPORT_SCHEMA,
        "seeds": {
            "train": 2026080861,
            "development": DEVELOPMENT_SEED,
            "confirmation": list(CONFIRMATION_SEEDS),
        },
        "parent_snl1": {
            "report_sha256": args.snl1_data_report_sha256,
            "identity_sha256": parent_report["identity_sha256"],
        },
        "historical_counts": historical,
        "zero_source_name_and_identity_overlap": True,
        "training_deterministic_regeneration": True,
        "training_target_counts": target_counts,
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
                "training_sha256": training_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
