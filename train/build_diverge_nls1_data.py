#!/usr/bin/env python3
"""Build immutable training and source-disjoint boards for DIVERGE-NLS1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from diverge_eal1_data import canonical_sha256
from diverge_eal2_data import (
    DEVELOPMENT_EPISODES,
    build_evaluation_episode,
    validate_episode,
)
from diverge_nls1_data import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEED,
    REPORT_SCHEMA,
    TRAIN_ROWS,
    TRAIN_SCHEMA,
    TRAIN_SEED,
    build_training_record,
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"NLS1 prerequisite hash differs: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


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


def _load_eal2_history(
    base: Path,
    base_report_sha256: str,
    confirmation: Path,
    confirmation_report_sha256: str,
) -> tuple[set[str], set[str], set[str], dict[str, Any]]:
    base_report_path = base / "report.json"
    confirmation_report_path = confirmation / "report.json"
    if sha256_path(base_report_path) != base_report_sha256:
        raise RuntimeError("NLS1 EAL2 base report hash differs")
    if sha256_path(confirmation_report_path) != confirmation_report_sha256:
        raise RuntimeError("NLS1 EAL2 confirmation report hash differs")
    base_report = json.loads(base_report_path.read_text())
    confirmation_report = json.loads(confirmation_report_path.read_text())
    base_public = _load_jsonl(
        base / "development_public.jsonl",
        base_report["files"]["development_public"]["sha256"],
    )
    historical = list(base_public)
    for seed in confirmation_report["seeds"]:
        entry = confirmation_report["files"][str(seed)]["public"]
        historical.extend(_load_jsonl(confirmation / entry["path"], entry["sha256"]))
    return (
        _sources(historical),
        _names(historical),
        _identities(historical),
        {
            "base_report_sha256": base_report_sha256,
            "confirmation_report_sha256": confirmation_report_sha256,
            "historical_episodes": len(historical),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eal2-base", type=Path, required=True)
    parser.add_argument("--eal2-base-report-sha256", required=True)
    parser.add_argument("--eal2-confirmation", type=Path, required=True)
    parser.add_argument("--eal2-confirmation-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NLS1 data output: {args.output}")
    args.output.mkdir(parents=True)

    occupied_sources, occupied_names, occupied_identities, parent = _load_eal2_history(
        args.eal2_base,
        args.eal2_base_report_sha256,
        args.eal2_confirmation,
        args.eal2_confirmation_report_sha256,
    )
    training = [build_training_record(index) for index in range(TRAIN_ROWS)]
    training_path = args.output / "training.jsonl"
    _atomic_jsonl(training_path, training)
    if _serialized_sha256(
        [build_training_record(index) for index in range(TRAIN_ROWS)]
    ) != sha256_path(training_path):
        raise SystemExit("NLS1 training regeneration failed")

    files: dict[str, Any] = {
        "training": {
            "path": training_path.name,
            "sha256": sha256_path(training_path),
            "bytes": training_path.stat().st_size,
            "rows": len(training),
        }
    }
    split_reports: dict[str, Any] = {}
    for label, seed in (
        ("development", DEVELOPMENT_SEED),
        *((f"confirmation_{seed}", seed) for seed in CONFIRMATION_SEEDS),
    ):
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
        if (
            len(source_set) != DEVELOPMENT_EPISODES * 24
            or len(name_set) != DEVELOPMENT_EPISODES * 10
            or len(identity_set) != DEVELOPMENT_EPISODES
            or source_set & occupied_sources
            or name_set & occupied_names
            or identity_set & occupied_identities
        ):
            raise SystemExit(f"NLS1 {label} uniqueness/overlap audit failed")
        occupied_sources |= source_set
        occupied_names |= name_set
        occupied_identities |= identity_set
        public_path = args.output / f"{label}_public.jsonl"
        assessor_path = args.output / f"{label}_assessor.jsonl"
        _atomic_jsonl(public_path, public)
        _atomic_jsonl(assessor_path, assessor)
        repeated = [
            build_evaluation_episode(index, seed=seed)
            for index in range(DEVELOPMENT_EPISODES)
        ]
        if _serialized_sha256([value[0] for value in repeated]) != sha256_path(
            public_path
        ) or _serialized_sha256([value[1] for value in repeated]) != sha256_path(
            assessor_path
        ):
            raise SystemExit(f"NLS1 {label} regeneration failed")
        files[label] = {
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
        split_reports[label] = {
            "seed": seed,
            "episodes": len(public),
            "sources": len(source_set),
            "names": len(name_set),
            "identities": len(identity_set),
            "deterministic_regeneration": True,
            "public_identity_sha256": canonical_sha256(
                [value["identity_sha256"] for value in public]
            ),
        }

    report = {
        "schema": REPORT_SCHEMA,
        "training_schema": TRAIN_SCHEMA,
        "seeds": {
            "training": TRAIN_SEED,
            "development": DEVELOPMENT_SEED,
            "confirmation": list(CONFIRMATION_SEEDS),
        },
        "parent": parent,
        "zero_source_name_and_identity_overlap": True,
        "training_deterministic_regeneration": True,
        "split_reports": split_reports,
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
                "training_sha256": files["training"]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
