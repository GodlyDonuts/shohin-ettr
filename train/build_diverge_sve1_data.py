#!/usr/bin/env python3
"""Build immutable, OQB1-lineage-disjoint DIVERGE-SVE1 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from diverge_sve1_data import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_EPISODES,
    DEVELOPMENT_SEED,
    REPORT_SCHEMA,
    TRAIN_ROWS,
    TRAIN_SCHEMA,
    TRAIN_SEED,
    augment_evaluation_episode,
    build_training_record,
    validate_evaluation_episode,
)
from diverge_eal1_data import canonical_sha256


OQB1_REPORT_SCHEMA = "shohin-diverge-oqb1-data-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialized(row: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    )


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = _serialized(row)
            handle.write(encoded)
            digest.update(encoded)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if sha256_path(path) != digest.hexdigest():
        raise RuntimeError("SVE1 serialized hash differs")
    return count, digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_report(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"SVE1 prerequisite report hash differs: {path}")
    return json.loads(path.read_text())


def _iter_jsonl(path: Path, expected_sha256: str) -> Iterator[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"SVE1 prerequisite data hash differs: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _report_file(root: Path, entry: Mapping[str, Any]) -> Path:
    return root / Path(str(entry["path"])).name


SOURCE_KEYS = (
    "source_sha256",
    "counterfactual_sha256",
    "scrubbed_sha256",
    "renamed_source_sha256",
    "register_scrubbed_sha256",
    "command_sha256",
    "reverse_command_sha256",
    "renamed_command_sha256",
    "scrubbed_command_sha256",
    "initial_sha256",
    "renamed_initial_sha256",
    "register_scrubbed_initial_sha256",
    "query_sha256",
    "renamed_query_sha256",
    "register_scrubbed_query_sha256",
)


def _episode_sources(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    output = set()
    for episode in rows:
        for group in ("evidence", "transfer", "queries"):
            for item in episode.get(group, ()):
                output.update(str(item[key]) for key in SOURCE_KEYS if key in item)
    return output


def _episode_names(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    keys = (
        "aliases",
        "renamed_aliases",
        "registers",
        "renamed_registers",
        "register_table",
        "renamed_register_table",
    )
    return {
        str(value) for episode in rows for key in keys for value in episode.get(key, ())
    }


def _episode_identities(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(episode["identity_sha256"]) for episode in rows}


def _historical_oqb1(
    root: Path, report_sha256: str
) -> tuple[set[str], set[str], set[str], dict[str, Any]]:
    report = _load_report(root / "report.json", report_sha256)
    if (
        report.get("schema") != OQB1_REPORT_SCHEMA
        or not report.get("zero_source_name_and_identity_overlap")
        or not report.get("training_deterministic_regeneration")
    ):
        raise RuntimeError("SVE1 OQB1 lineage report is not qualified")
    sources: set[str] = set()
    names: set[str] = set()
    identities: set[str] = set()
    training_entry = report["files"]["training"]
    training_rows = 0
    for row in _iter_jsonl(
        _report_file(root, training_entry), training_entry["sha256"]
    ):
        sources.update(
            str(row[key])
            for key in ("evidence_sha256", "initial_sha256", "query_sha256")
        )
        names.add(str(row["operation"]))
        names.update(str(value) for value in row["register_table"])
        identities.add(str(row["identity_sha256"]))
        training_rows += 1
    if training_rows != int(training_entry["rows"]):
        raise RuntimeError("SVE1 OQB1 training row receipt differs")
    evaluation_rows = 0
    for label, entries in report["files"].items():
        if label == "training":
            continue
        entry = entries["public"]
        rows = list(_iter_jsonl(_report_file(root, entry), entry["sha256"]))
        if len(rows) != int(entry["rows"]):
            raise RuntimeError("SVE1 OQB1 evaluation row receipt differs")
        sources |= _episode_sources(rows)
        names |= _episode_names(rows)
        identities |= _episode_identities(rows)
        evaluation_rows += len(rows)
    return (
        sources,
        names,
        identities,
        {
            "oqb1_data_report_sha256": report_sha256,
            "oqb1_identity_sha256": report["identity_sha256"],
            "oqb1_training_rows": training_rows,
            "oqb1_evaluation_episodes": evaluation_rows,
            "oqb1_transitive_overlap_gate": True,
        },
    )


def _training_rows() -> Iterable[dict[str, Any]]:
    for serial in range(TRAIN_ROWS):
        yield build_training_record(serial)


def _training_audit() -> tuple[str, set[str], set[str], dict[str, Any]]:
    digest = hashlib.sha256()
    sources = set()
    names = set()
    rotations = [0, 0]
    query_positions = [0, 0]
    renderers = {"initial": set(), "query": set()}
    for row in _training_rows():
        digest.update(_serialized(row))
        sources.update(
            (row["evidence_sha256"], row["initial_sha256"], row["query_sha256"])
        )
        names.add(str(row["operation"]))
        names.update(str(value) for value in row["register_table"])
        rotations[int(row["table_rotation"])] += 1
        query_positions[int(row["query_position_target"])] += 1
        renderers["initial"].update(tuple(value) for value in row["initial_renderer"])
        renderers["query"].add(tuple(row["query_renderer"]))
    return (
        digest.hexdigest(),
        sources,
        names,
        {
            "table_rotations": rotations,
            "query_positions": query_positions,
            "renderer_counts": {key: len(value) for key, value in renderers.items()},
        },
    )


def _balanced(counts: Sequence[int], total: int, tolerance: float = 0.01) -> bool:
    return (
        len(counts) == 2
        and sum(counts) == total
        and all(abs(value / total - 0.5) <= tolerance for value in counts)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oqb1-data", type=Path, required=True)
    parser.add_argument("--oqb1-data-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing SVE1 data output: {args.output}")
    args.output.mkdir(parents=True)

    occupied_sources, occupied_names, occupied_identities, parent = _historical_oqb1(
        args.oqb1_data, args.oqb1_data_report_sha256
    )
    historical_counts = {
        "sources": len(occupied_sources),
        "names": len(occupied_names),
        "identities": len(occupied_identities),
    }

    training_path = args.output / "training.jsonl"
    training_rows, training_sha256 = _atomic_jsonl(training_path, _training_rows())
    regenerated, training_sources, training_names, training_stats = _training_audit()
    if (
        regenerated != training_sha256
        or training_rows != TRAIN_ROWS
        or len(training_sources) != TRAIN_ROWS * 3
        or len(training_names) != TRAIN_ROWS * 3
        or training_sources & occupied_sources
        or training_names & occupied_names
        or not _balanced(training_stats["table_rotations"], TRAIN_ROWS)
        or not _balanced(training_stats["query_positions"], TRAIN_ROWS)
    ):
        raise SystemExit("SVE1 training regeneration/balance/overlap audit failed")

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
        paired = [
            augment_evaluation_episode(index, seed=seed)
            for index in range(DEVELOPMENT_EPISODES)
        ]
        public = [value[0] for value in paired]
        assessor = [value[1] for value in paired]
        for visible, hidden in zip(public, assessor, strict=True):
            validate_evaluation_episode(visible, hidden)
        sources = _episode_sources(public)
        names = _episode_names(public)
        identities = _episode_identities(public)
        rotations = [0, 0]
        for episode in public:
            rotations[int(episode["table_rotation"])] += 1
        if (
            len(identities) != len(public)
            or sources & occupied_sources
            or names & occupied_names
            or identities & occupied_identities
            or sources & training_sources
            or names & training_names
            or not _balanced(rotations, DEVELOPMENT_EPISODES, tolerance=0.10)
        ):
            raise SystemExit(f"SVE1 {label} balance/overlap audit failed")
        occupied_sources |= sources
        occupied_names |= names
        occupied_identities |= identities
        public_path = args.output / f"{label}_public.jsonl"
        assessor_path = args.output / f"{label}_assessor.jsonl"
        public_rows, public_sha = _atomic_jsonl(public_path, public)
        assessor_rows, assessor_sha = _atomic_jsonl(assessor_path, assessor)
        repeated = [
            augment_evaluation_episode(index, seed=seed)
            for index in range(DEVELOPMENT_EPISODES)
        ]
        if (
            hashlib.sha256(
                b"".join(_serialized(value[0]) for value in repeated)
            ).hexdigest()
            != public_sha
            or hashlib.sha256(
                b"".join(_serialized(value[1]) for value in repeated)
            ).hexdigest()
            != assessor_sha
        ):
            raise SystemExit(f"SVE1 {label} regeneration failed")
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
            "episodes": len(public),
            "sources": len(sources),
            "names": len(names),
            "identities": len(identities),
            "table_rotations": rotations,
            "deterministic_regeneration": True,
            "identity_sha256": canonical_sha256(
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
        "historical_counts": historical_counts,
        "zero_source_name_and_identity_overlap": True,
        "training_deterministic_regeneration": True,
        "training_stats": training_stats,
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
