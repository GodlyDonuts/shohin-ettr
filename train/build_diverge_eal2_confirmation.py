#!/usr/bin/env python3
"""Build five fixed source-disjoint EAL2 confirmation boards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from diverge_eal1_data import canonical_sha256
from diverge_eal2_data import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_EPISODES,
    build_evaluation_episode,
    validate_episode,
)


SCHEMA = "shohin-diverge-eal2-confirmation-data-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("EAL2 base data hash differs")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _serialized_sha256(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _sources(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str(item["source_sha256"]) for episode in rows for item in episode["evidence"]
    }


def _names(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {
        value
        for episode in rows
        for value in (*episode["aliases"], *episode["registers"])
    }


def _identities(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {str(episode["identity_sha256"]) for episode in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--development-public-sha256", required=True)
    parser.add_argument("--development-assessor-sha256", required=True)
    parser.add_argument("--base-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing EAL2 confirmation output: {args.output}")
    args.output.mkdir(parents=True)
    training = _load_jsonl(args.base_data / "training.jsonl", args.training_sha256)
    development = _load_jsonl(
        args.base_data / "development_public.jsonl",
        args.development_public_sha256,
    )
    development_assessor = _load_jsonl(
        args.base_data / "development_assessor.jsonl",
        args.development_assessor_sha256,
    )
    report_path = args.base_data / "report.json"
    if sha256_path(report_path) != args.base_report_sha256:
        raise SystemExit("EAL2 base data report hash differs")
    base_report = json.loads(report_path.read_text())
    expected_base_files = {
        "training": args.training_sha256,
        "development_public": args.development_public_sha256,
        "development_assessor": args.development_assessor_sha256,
    }
    if any(
        base_report["files"][name]["sha256"] != digest
        for name, digest in expected_base_files.items()
    ):
        raise SystemExit("EAL2 base report/file binding differs")
    if (
        len(development) != DEVELOPMENT_EPISODES
        or len(development_assessor) != DEVELOPMENT_EPISODES
    ):
        raise SystemExit("EAL2 base development geometry differs")
    for public, assessor in zip(development, development_assessor, strict=True):
        validate_episode(public, assessor)
    occupied_sources = {str(row["source_sha256"]) for row in training} | _sources(
        development
    )
    occupied_names = {
        value for row in training for value in (row["operation"], *row["registers"])
    } | _names(development)
    occupied_identities = _identities(development)
    files = {}
    seed_reports = {}
    for seed in CONFIRMATION_SEEDS:
        paired = [
            build_evaluation_episode(index, seed=seed)
            for index in range(DEVELOPMENT_EPISODES)
        ]
        public = [value[0] for value in paired]
        assessor = [value[1] for value in paired]
        for visible, hidden in zip(public, assessor, strict=True):
            validate_episode(visible, hidden)
        source_set = _sources(public)
        name_set = _names(public)
        identity_set = _identities(public)
        if len(source_set) != DEVELOPMENT_EPISODES * 24:
            raise SystemExit("EAL2 confirmation source uniqueness failed")
        if len(name_set) != DEVELOPMENT_EPISODES * 10:
            raise SystemExit("EAL2 confirmation name uniqueness failed")
        if len(identity_set) != DEVELOPMENT_EPISODES:
            raise SystemExit("EAL2 confirmation identity uniqueness failed")
        if (
            source_set & occupied_sources
            or name_set & occupied_names
            or identity_set & occupied_identities
        ):
            raise SystemExit("EAL2 confirmation overlap failed")
        occupied_sources |= source_set
        occupied_names |= name_set
        occupied_identities |= identity_set
        public_path = args.output / f"seed_{seed}_public.jsonl"
        assessor_path = args.output / f"seed_{seed}_assessor.jsonl"
        _atomic_jsonl(public_path, public)
        _atomic_jsonl(assessor_path, assessor)
        repeated = [
            build_evaluation_episode(index, seed=seed)
            for index in range(DEVELOPMENT_EPISODES)
        ]
        reproducible = _serialized_sha256(
            [value[0] for value in repeated]
        ) == sha256_path(public_path) and _serialized_sha256(
            [value[1] for value in repeated]
        ) == sha256_path(assessor_path)
        if not reproducible:
            raise SystemExit("EAL2 confirmation regeneration failed")
        files[str(seed)] = {
            "public": {
                "path": public_path.name,
                "sha256": sha256_path(public_path),
                "bytes": public_path.stat().st_size,
            },
            "assessor": {
                "path": assessor_path.name,
                "sha256": sha256_path(assessor_path),
                "bytes": assessor_path.stat().st_size,
            },
        }
        seed_reports[str(seed)] = {
            "episodes": len(public),
            "sources": len(source_set),
            "names": len(name_set),
            "identities": len(identity_set),
            "deterministic_regeneration": reproducible,
            "public_identity_sha256": canonical_sha256(
                [value["identity_sha256"] for value in public]
            ),
        }
    report = {
        "schema": SCHEMA,
        "seeds": list(CONFIRMATION_SEEDS),
        "base": {
            "training_sha256": args.training_sha256,
            "development_public_sha256": args.development_public_sha256,
            "development_assessor_sha256": args.development_assessor_sha256,
            "report_sha256": args.base_report_sha256,
        },
        "zero_source_name_and_identity_overlap": True,
        "seed_reports": seed_reports,
        "files": files,
    }
    report["identity_sha256"] = canonical_sha256(report)
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": sha256_path(report_path),
                "files": files,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
