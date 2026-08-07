#!/usr/bin/env python3
"""Build immutable, source-disjoint DIVERGE-JRB1 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from diverge_eal1_data import canonical_sha256
from diverge_jrb1_data import (
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
        raise RuntimeError("JRB1 serialized hash differs")
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
        raise RuntimeError(f"JRB1 prerequisite report hash differs: {path}")
    return json.loads(path.read_text())


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"JRB1 prerequisite data hash differs: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _report_file(root: Path, entry: Mapping[str, Any]) -> Path:
    return root / Path(str(entry["path"])).name


def _historical_boards(
    eal2_base: Path,
    eal2_base_sha256: str,
    eal2_confirmation: Path,
    eal2_confirmation_sha256: str,
    ncp1_data: Path,
    ncp1_data_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    base = _load_report(eal2_base / "report.json", eal2_base_sha256)
    confirmation = _load_report(
        eal2_confirmation / "report.json", eal2_confirmation_sha256
    )
    ncp1 = _load_report(ncp1_data / "report.json", ncp1_data_sha256)
    base_entry = base["files"]["development_public"]
    rows = _load_jsonl(_report_file(eal2_base, base_entry), base_entry["sha256"])
    for seed in confirmation["seeds"]:
        entry = confirmation["files"][str(seed)]["public"]
        rows.extend(
            _load_jsonl(_report_file(eal2_confirmation, entry), entry["sha256"])
        )
    for label in (
        "development",
        *(f"confirmation_{seed}" for seed in ncp1["seeds"]["confirmation"]),
    ):
        entry = ncp1["files"][label]["public"]
        rows.extend(_load_jsonl(_report_file(ncp1_data, entry), entry["sha256"]))
    return rows, {
        "eal2_base_report_sha256": eal2_base_sha256,
        "eal2_confirmation_report_sha256": eal2_confirmation_sha256,
        "ncp1_data_report_sha256": ncp1_data_sha256,
    }


def _episode_sources(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    keys = (
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
    output = set()
    for episode in rows:
        for group in ("evidence", "transfer", "queries"):
            for item in episode.get(group, ()):
                output.update(str(item[key]) for key in keys if key in item)
    return output


def _episode_names(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(value)
        for episode in rows
        for key in ("aliases", "renamed_aliases", "registers", "renamed_registers")
        for value in episode.get(key, ())
    }


def _episode_identities(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(episode["identity_sha256"]) for episode in rows}


def _training_rows() -> Iterable[dict[str, Any]]:
    for serial in range(TRAIN_ROWS):
        yield build_training_record(serial)


def _training_audit() -> tuple[str, set[str], set[str], dict[str, int]]:
    digest = hashlib.sha256()
    sources = set()
    names = set()
    renderers = {"initial": set(), "query": set()}
    for row in _training_rows():
        digest.update(_serialized(row))
        sources.update(
            (row["evidence_sha256"], row["initial_sha256"], row["query_sha256"])
        )
        names.add(str(row["operation"]))
        names.update(str(value) for value in row["registers"])
        renderers["initial"].update(tuple(value) for value in row["initial_renderer"])
        renderers["query"].add(tuple(row["query_renderer"]))
    return (
        digest.hexdigest(),
        sources,
        names,
        {key: len(value) for key, value in renderers.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eal2-base", type=Path, required=True)
    parser.add_argument("--eal2-base-report-sha256", required=True)
    parser.add_argument("--eal2-confirmation", type=Path, required=True)
    parser.add_argument("--eal2-confirmation-report-sha256", required=True)
    parser.add_argument("--ncp1-data", type=Path, required=True)
    parser.add_argument("--ncp1-data-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing JRB1 data output: {args.output}")
    args.output.mkdir(parents=True)

    historical, parent = _historical_boards(
        args.eal2_base,
        args.eal2_base_report_sha256,
        args.eal2_confirmation,
        args.eal2_confirmation_report_sha256,
        args.ncp1_data,
        args.ncp1_data_report_sha256,
    )
    occupied_sources = _episode_sources(historical)
    occupied_names = _episode_names(historical)
    occupied_identities = _episode_identities(historical)

    training_path = args.output / "training.jsonl"
    training_rows, training_sha256 = _atomic_jsonl(training_path, _training_rows())
    regenerated, training_sources, training_names, renderer_counts = _training_audit()
    if (
        regenerated != training_sha256
        or training_rows != TRAIN_ROWS
        or len(training_sources) != TRAIN_ROWS * 3
        or len(training_names) != TRAIN_ROWS * 3
        or training_sources & occupied_sources
        or training_names & occupied_names
    ):
        raise SystemExit("JRB1 training regeneration/overlap audit failed")

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
        source_occurrences = sum(
            len(_episode_sources([episode])) for episode in public
        )
        if (
            len(sources) != source_occurrences
            or len(identities) != len(public)
            or sources & occupied_sources
            or names & occupied_names
            or identities & occupied_identities
            or sources & training_sources
            or names & training_names
        ):
            raise SystemExit(f"JRB1 {label} overlap audit failed")
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
            raise SystemExit(f"JRB1 {label} regeneration failed")
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
        "historical_episodes": len(historical),
        "zero_source_name_and_identity_overlap": True,
        "training_deterministic_regeneration": True,
        "training_renderer_counts": renderer_counts,
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
