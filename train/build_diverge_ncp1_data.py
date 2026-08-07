#!/usr/bin/env python3
"""Build immutable training and source-disjoint boards for DIVERGE-NCP1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from diverge_eal1_data import canonical_sha256
from diverge_eal2_data import (
    DEVELOPMENT_EPISODES,
    build_evaluation_episode,
    validate_episode,
)
from diverge_ncp1_data import (
    CONFIRMATION_SEEDS,
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
        raise RuntimeError("NCP1 serialized hash differs")
    return count, digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"NCP1 prerequisite hash differs: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _episode_sources(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    output = {
        str(item["source_sha256"]) for episode in rows for item in episode["evidence"]
    }
    for episode in rows:
        for transfer in episode.get("transfer", ()):  # NCP1 boards add command views.
            for key in (
                "command_sha256",
                "reverse_command_sha256",
                "renamed_command_sha256",
                "scrubbed_command_sha256",
            ):
                if key in transfer:
                    output.add(str(transfer[key]))
    return output


def _episode_names(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(value)
        for episode in rows
        for key in ("aliases", "registers", "renamed_aliases")
        for value in episode.get(key, ())
    }


def _episode_identities(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(episode["identity_sha256"]) for episode in rows}


def _load_report(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"NCP1 report hash differs: {path}")
    return json.loads(path.read_text())


def _report_file(root: Path, entry: Mapping[str, Any]) -> Path:
    return root / Path(str(entry["path"])).name


def _historical_boards(
    eal2_base: Path,
    eal2_base_sha256: str,
    eal2_confirmation: Path,
    eal2_confirmation_sha256: str,
    nls1_data: Path,
    nls1_report_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    base = _load_report(eal2_base / "report.json", eal2_base_sha256)
    confirmation = _load_report(
        eal2_confirmation / "report.json", eal2_confirmation_sha256
    )
    nls1 = _load_report(nls1_data / "report.json", nls1_report_sha256)
    base_entry = base["files"]["development_public"]
    rows = _load_jsonl(_report_file(eal2_base, base_entry), base_entry["sha256"])
    for seed in confirmation["seeds"]:
        item = confirmation["files"][str(seed)]["public"]
        rows.extend(_load_jsonl(_report_file(eal2_confirmation, item), item["sha256"]))
    for label in (
        "development",
        *(f"confirmation_{seed}" for seed in nls1["seeds"]["confirmation"]),
    ):
        item = nls1["files"][label]["public"]
        rows.extend(_load_jsonl(_report_file(nls1_data, item), item["sha256"]))
    return rows, {
        "eal2_base_report_sha256": eal2_base_sha256,
        "eal2_confirmation_report_sha256": eal2_confirmation_sha256,
        "nls1_data_report_sha256": nls1_report_sha256,
    }


def _training_rows() -> Iterable[dict[str, Any]]:
    for serial in range(TRAIN_ROWS):
        yield build_training_record(serial)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eal2-base", type=Path, required=True)
    parser.add_argument("--eal2-base-report-sha256", required=True)
    parser.add_argument("--eal2-confirmation", type=Path, required=True)
    parser.add_argument("--eal2-confirmation-report-sha256", required=True)
    parser.add_argument("--nls1-data", type=Path, required=True)
    parser.add_argument("--nls1-data-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NCP1 data output: {args.output}")
    args.output.mkdir(parents=True)

    historical, parent = _historical_boards(
        args.eal2_base,
        args.eal2_base_report_sha256,
        args.eal2_confirmation,
        args.eal2_confirmation_report_sha256,
        args.nls1_data,
        args.nls1_data_report_sha256,
    )
    occupied_sources = _episode_sources(historical)
    occupied_names = _episode_names(historical)
    occupied_identities = _episode_identities(historical)

    training_path = args.output / "training.jsonl"
    training_rows, training_sha256 = _atomic_jsonl(training_path, _training_rows())
    repeated_digest = hashlib.sha256()
    training_aliases = set()
    training_sources = set()
    for record in _training_rows():
        repeated_digest.update(_serialized(record))
        training_sources.add(str(record["source_sha256"]))
        training_aliases.update(str(value) for value in record["aliases"])
    if (
        repeated_digest.hexdigest() != training_sha256
        or len(training_sources) != TRAIN_ROWS
        or len(training_aliases) != TRAIN_ROWS * 8
        or training_sources & occupied_sources
        or training_aliases & occupied_names
    ):
        raise SystemExit("NCP1 training regeneration/overlap audit failed")

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
        base = [
            build_evaluation_episode(index, seed=seed)
            for index in range(DEVELOPMENT_EPISODES)
        ]
        for visible, hidden in base:
            validate_episode(visible, hidden)
        paired = [
            augment_evaluation_episode(visible, hidden, seed=seed, serial=index)
            for index, (visible, hidden) in enumerate(base)
        ]
        public = [value[0] for value in paired]
        assessor = [value[1] for value in paired]
        for visible, hidden in zip(public, assessor, strict=True):
            validate_evaluation_episode(visible, hidden)
        sources = _episode_sources(public)
        names = _episode_names(public)
        identities = _episode_identities(public)
        if (
            sources & occupied_sources
            or names & occupied_names
            or identities & occupied_identities
            or sources & training_sources
            or names & training_aliases
        ):
            raise SystemExit(f"NCP1 {label} overlap audit failed")
        occupied_sources |= sources
        occupied_names |= names
        occupied_identities |= identities
        public_path = args.output / f"{label}_public.jsonl"
        assessor_path = args.output / f"{label}_assessor.jsonl"
        public_rows, public_sha = _atomic_jsonl(public_path, public)
        assessor_rows, assessor_sha = _atomic_jsonl(assessor_path, assessor)
        repeated = [
            augment_evaluation_episode(
                *build_evaluation_episode(index, seed=seed), seed=seed, serial=index
            )
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
            raise SystemExit(f"NCP1 {label} regeneration failed")
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
                "training_sha256": training_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
