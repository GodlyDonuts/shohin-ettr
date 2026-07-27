#!/usr/bin/env python3
"""Audit and deterministically select the ETTR-IL-v3 initializer corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Iterable, Mapping, Sequence

from ettr_il_v3_production import (
    ROW_SCHEMA,
    SCHEMA as PRODUCTION_SCHEMA,
    ProductionCell,
    base_owner_split,
    production_cells,
)
from ettr_il_v3_protocol import (
    CURRICULUM_STAGES,
    FAMILIES,
    MASTER_SEED,
    PROTOCOL,
    SPLIT_CORES,
    canonical_json_bytes,
    orbit_owner,
    split_stage_family_allocation,
)


SCHEMA = "r12-ettr-il-v3-selection-v1"
MANIFEST_SCHEMA = "r12-ettr-il-v3-selected-manifest-v1"
MAIN_SPLITS = (
    "train",
    "development",
    "train_reserve",
    "development_reserve",
)
CONFIRMATION_SPLITS = ("confirmation", "confirmation_reserve")
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_ROW_BYTES = 8 * 1024 * 1024


class SelectionError(ValueError):
    """Candidate audit or deterministic quota selection failed."""


@dataclass(frozen=True, slots=True)
class Candidate:
    episode_id: str
    split: str
    family: str
    stage: str
    depth: int
    row: Mapping[str, object]
    semantic_factor_sha256: str
    coverage_tokens: tuple[str, ...]


def _load_canonical_file(path: Path, *, maximum_bytes: int) -> dict[str, object]:
    if path.is_symlink():
        raise SelectionError(f"candidate metadata is a symlink: {path}")
    status = path.stat()
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or status.st_size > maximum_bytes
    ):
        raise SelectionError(f"candidate metadata is unsafe: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(
            payload,
            parse_constant=lambda item: (_ for _ in ()).throw(
                SelectionError(f"non-finite JSON value: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"candidate metadata is invalid JSON: {path}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise SelectionError(f"candidate metadata is not canonical: {path}")
    return value


def _hex_digest(value: object, label: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectionError(f"{label} differs")
    return value


def _verify_report(
    root: Path,
    cell: ProductionCell,
    *,
    source_commit: str | None,
    protocol_freeze: str | None,
) -> tuple[dict[str, object], str, str]:
    report = _load_canonical_file(
        root / "reports" / f"cell-{cell.index}.json",
        maximum_bytes=MAX_REPORT_BYTES,
    )
    expected_keys = {
        "cell",
        "compressed_bytes",
        "protocol",
        "protocol_freeze_sha256",
        "report_sha256",
        "row_count",
        "schema",
        "shard_name",
        "shard_sha256",
        "source_commit",
        "status",
        "uncompressed_bytes",
    }
    if (
        set(report) != expected_keys
        or report["schema"] != PRODUCTION_SCHEMA
        or report["protocol"] != PROTOCOL
        or report["status"] != "pass"
        or report["cell"] != cell.to_value()
        or report["row_count"] != cell.candidate_target
        or report["shard_name"] != f"cell-{cell.index}.jsonl.gz"
    ):
        raise SelectionError(f"production report differs for cell {cell.index}")
    report_sha256 = _hex_digest(report["report_sha256"], "report SHA-256")
    unhashed = dict(report)
    del unhashed["report_sha256"]
    if hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest() != report_sha256:
        raise SelectionError(f"report self-hash differs for cell {cell.index}")
    observed_commit = _hex_digest(
        report["source_commit"],
        "source commit",
        length=40,
    )
    observed_freeze = _hex_digest(
        report["protocol_freeze_sha256"],
        "protocol freeze",
    )
    if source_commit is not None and observed_commit != source_commit:
        raise SelectionError("production source commits are inconsistent")
    if protocol_freeze is not None and observed_freeze != protocol_freeze:
        raise SelectionError("production protocol freezes are inconsistent")
    return report, observed_commit, observed_freeze


def _coverage_tokens(
    family: str,
    stage: str,
    depth: int,
    episode: Mapping[str, object],
) -> tuple[str, ...]:
    coverage = episode.get("coverage")
    if not isinstance(coverage, Mapping):
        raise SelectionError("episode coverage differs")
    tokens = {
        f"family={family}",
        f"stage={stage}",
        f"depth={depth}",
    }
    for key in (
        "disposition",
        "status",
        "trace_length_bin",
        "packet_density_bin",
        "changed_slots",
        "prefix_dependent_steps",
    ):
        if key in coverage:
            tokens.add(f"{key}={json.dumps(coverage[key], sort_keys=True)}")
    query_ops = coverage.get("query_ops")
    if isinstance(query_ops, list):
        tokens.update(f"query_op={value}" for value in query_ops)
    query_answers = coverage.get("query_answers")
    if isinstance(query_answers, list):
        tokens.add(
            "query_answers="
            + json.dumps(query_answers, separators=(",", ":"), sort_keys=True)
        )
    for histogram_name in ("operation_histogram", "outcome_histogram"):
        histogram = coverage.get(histogram_name)
        if isinstance(histogram, list):
            for item in histogram:
                if isinstance(item, list) and item:
                    tokens.add(f"{histogram_name}={item[0]}")
    for key in ("applied_count", "blocked_count", "rejected_count"):
        if key in coverage:
            tokens.add(f"{key}={coverage[key]}")
    return tuple(sorted(tokens))


def _parse_candidate(
    payload: bytes,
    *,
    cell: ProductionCell,
    expected_ordinal: int,
) -> Candidate:
    if len(payload) > MAX_ROW_BYTES:
        raise SelectionError("candidate row exceeds bounded size")
    try:
        row = json.loads(
            payload,
            parse_constant=lambda item: (_ for _ in ()).throw(
                SelectionError(f"non-finite candidate value: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectionError("candidate row is invalid JSON") from error
    if not isinstance(row, dict) or canonical_json_bytes(row) != payload:
        raise SelectionError("candidate row is not canonical JSON")
    expected_keys = {
        "cell",
        "episode",
        "episode_id",
        "ordinal",
        "owner",
        "protocol",
        "schema",
    }
    if (
        set(row) != expected_keys
        or row["schema"] != ROW_SCHEMA
        or row["protocol"] != PROTOCOL
        or row["cell"] != cell.to_value()
        or row["ordinal"] != expected_ordinal
        or row["owner"] != base_owner_split(cell.split)
    ):
        raise SelectionError("candidate row identity differs")
    episode = row["episode"]
    if not isinstance(episode, dict):
        raise SelectionError("candidate episode differs")
    episode_id = _hex_digest(row["episode_id"], "episode ID")
    if hashlib.sha256(canonical_json_bytes(episode)).hexdigest() != episode_id:
        raise SelectionError("candidate episode ID differs")
    world = episode.get("world")
    command = episode.get("command")
    queries = episode.get("queries")
    if (
        not isinstance(world, dict)
        or not isinstance(command, dict)
        or not isinstance(queries, list)
    ):
        raise SelectionError("candidate semantic factors differ")
    expected_owner = orbit_owner({"family": cell.family, "world": world})
    if expected_owner != base_owner_split(cell.split):
        raise SelectionError("candidate split ownership differs")
    semantic_factor_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "command": command,
                "family": cell.family,
                "queries": queries,
                "world": world,
            }
        )
    ).hexdigest()
    return Candidate(
        episode_id=episode_id,
        split=cell.split,
        family=cell.family,
        stage=cell.stage,
        depth=cell.depth,
        row=row,
        semantic_factor_sha256=semantic_factor_sha256,
        coverage_tokens=_coverage_tokens(
            cell.family,
            cell.stage,
            cell.depth,
            episode,
        ),
    )


def _load_cell_candidates(
    root: Path,
    cell: ProductionCell,
    report: Mapping[str, object],
) -> tuple[Candidate, ...]:
    shard_path = root / "shards" / f"cell-{cell.index}.jsonl.gz"
    if shard_path.is_symlink():
        raise SelectionError("candidate shard is a symlink")
    status = shard_path.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise SelectionError("candidate shard is unsafe")
    compressed = shard_path.read_bytes()
    if (
        len(compressed) != report["compressed_bytes"]
        or hashlib.sha256(compressed).hexdigest() != report["shard_sha256"]
    ):
        raise SelectionError(f"candidate shard hash differs for cell {cell.index}")
    try:
        uncompressed = gzip.decompress(compressed)
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise SelectionError("candidate shard cannot be decompressed") from error
    if len(uncompressed) != report["uncompressed_bytes"]:
        raise SelectionError(
            f"candidate uncompressed size differs for cell {cell.index}"
        )
    rows = uncompressed.splitlines(keepends=True)
    if len(rows) != cell.candidate_target:
        raise SelectionError("candidate shard row count differs")
    return tuple(
        _parse_candidate(
            row,
            cell=cell,
            expected_ordinal=ordinal,
        )
        for ordinal, row in enumerate(rows)
    )


def select_pool(
    candidates: Iterable[Candidate],
    *,
    quota: int,
    group_context: Mapping[str, object],
    forbidden_semantic_factors: set[str],
) -> tuple[Candidate, ...]:
    values = tuple(candidates)
    if type(quota) is not int or quota < 1 or len(values) < quota:
        raise SelectionError("selection quota exceeds candidate population")
    frequencies = Counter(
        token for candidate in values for token in candidate.coverage_tokens
    )

    def score(candidate: Candidate) -> int:
        return sum(
            1_000_000_000 // frequencies[token]
            for token in candidate.coverage_tokens
        )

    ranked = sorted(
        values,
        key=lambda candidate: (
            -score(candidate),
            hashlib.sha256(
                MASTER_SEED
                + b"|selected-candidate|"
                + canonical_json_bytes(
                    {
                        "episode_id": candidate.episode_id,
                        "group": dict(group_context),
                    }
                )
            ).digest(),
            candidate.episode_id,
        ),
    )
    selected: list[Candidate] = []
    local_factors: set[str] = set()
    for candidate in ranked:
        factor = candidate.semantic_factor_sha256
        if factor in forbidden_semantic_factors or factor in local_factors:
            continue
        selected.append(candidate)
        local_factors.add(factor)
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise SelectionError(
            f"unique semantic-factor selection exhausted at "
            f"{len(selected)} of {quota}"
        )
    forbidden_semantic_factors.update(local_factors)
    return tuple(selected)


def _write_no_replace(path: Path, payload: bytes) -> None:
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


def _write_selected_shard(
    path: Path,
    selected: Sequence[Candidate],
) -> dict[str, object]:
    rows = tuple(canonical_json_bytes(candidate.row) for candidate in selected)
    compressed = gzip.compress(b"".join(rows), compresslevel=6, mtime=0)
    _write_no_replace(path, compressed)
    return {
        "bytes": len(compressed),
        "path": path.name,
        "rows": len(rows),
        "sha256": hashlib.sha256(compressed).hexdigest(),
    }


def audit_and_select(
    candidate_root: Path,
    *,
    main_output: Path,
    confirmation_output: Path,
) -> dict[str, object]:
    cells = production_cells()
    expected_reports = {f"cell-{cell.index}.json" for cell in cells}
    expected_shards = {f"cell-{cell.index}.jsonl.gz" for cell in cells}
    observed_reports = {
        path.name for path in (candidate_root / "reports").iterdir()
    }
    observed_shards = {
        path.name for path in (candidate_root / "shards").iterdir()
    }
    if observed_reports != expected_reports or observed_shards != expected_shards:
        raise SelectionError("candidate production inventory differs")
    if main_output.exists() or confirmation_output.exists():
        raise SelectionError("selected output root already exists")

    reports: dict[int, dict[str, object]] = {}
    source_commit: str | None = None
    protocol_freeze: str | None = None
    candidate_root_hasher = hashlib.sha256()
    for cell in cells:
        report, source_commit, protocol_freeze = _verify_report(
            candidate_root,
            cell,
            source_commit=source_commit,
            protocol_freeze=protocol_freeze,
        )
        reports[cell.index] = report
        candidate_root_hasher.update(
            str(cell.index).encode("ascii")
            + b"\0"
            + report["report_sha256"].encode("ascii")
            + b"\n"
        )

    selected_factor_sets: dict[str, set[str]] = {
        owner: set() for owner in ("train", "development", "confirmation")
    }
    all_episode_ids: set[str] = set()
    selected_episode_ids: set[str] = set()
    selected_groups: list[
        tuple[str, str, str, tuple[Candidate, ...]]
    ] = []
    selected_by_split: Counter[str] = Counter()
    main_descriptors: list[dict[str, object]] = []
    confirmation_descriptors: list[dict[str, object]] = []
    candidate_rows = 0
    selected_rows = 0

    for split in (
        "train",
        "development",
        "confirmation",
        "train_reserve",
        "development_reserve",
        "confirmation_reserve",
    ):
        matrix = split_stage_family_allocation(split)
        for family in FAMILIES:
            for stage in CURRICULUM_STAGES:
                group_cells = tuple(
                    cell
                    for cell in cells
                    if cell.split == split
                    and cell.family == family
                    and cell.stage == stage
                )
                candidates: list[Candidate] = []
                for cell in group_cells:
                    loaded = _load_cell_candidates(
                        candidate_root,
                        cell,
                        reports[cell.index],
                    )
                    for candidate in loaded:
                        if candidate.episode_id in all_episode_ids:
                            raise SelectionError("candidate episode ID repeats globally")
                        all_episode_ids.add(candidate.episode_id)
                    candidates.extend(loaded)
                candidate_rows += len(candidates)
                quota = matrix[stage][family]
                owner = base_owner_split(split)
                selected = select_pool(
                    candidates,
                    quota=quota,
                    group_context={
                        "family": family,
                        "split": split,
                        "stage": stage,
                    },
                    forbidden_semantic_factors=selected_factor_sets[owner],
                )
                for candidate in selected:
                    if candidate.episode_id in selected_episode_ids:
                        raise SelectionError("selected episode ID repeats")
                    selected_episode_ids.add(candidate.episode_id)
                selected_rows += len(selected)
                selected_by_split[split] += len(selected)
                selected_groups.append((split, family, stage, selected))

    expected_rows = sum(SPLIT_CORES.values())
    if (
        selected_rows != expected_rows
        or dict(selected_by_split) != dict(SPLIT_CORES)
        or len(selected_episode_ids) != expected_rows
    ):
        raise SelectionError("selected split cardinality differs")

    # No selected payload is written until every candidate and quota has passed.
    main_output.mkdir(parents=True)
    confirmation_output.mkdir(parents=True)
    for split, family, stage, selected in selected_groups:
        output_root = (
            main_output if split in MAIN_SPLITS else confirmation_output
        )
        filename = f"{split}-{family}-{stage}.jsonl.gz"
        descriptor = _write_selected_shard(
            output_root / filename,
            selected,
        )
        descriptor.update(
            {
                "family": family,
                "split": split,
                "stage": stage,
            }
        )
        (
            main_descriptors
            if split in MAIN_SPLITS
            else confirmation_descriptors
        ).append(descriptor)

    manifests = (
        (
            main_output,
            "main",
            main_descriptors,
        ),
        (
            confirmation_output,
            "sealed_confirmation",
            confirmation_descriptors,
        ),
    )
    manifest_hashes: dict[str, str] = {}
    for root, role, descriptors in manifests:
        manifest: dict[str, object] = {
            "candidate_root_sha256": candidate_root_hasher.hexdigest(),
            "protocol": PROTOCOL,
            "protocol_freeze_sha256": protocol_freeze,
            "role": role,
            "schema": MANIFEST_SCHEMA,
            "shards": descriptors,
            "source_commit": source_commit,
            "total_rows": sum(int(item["rows"]) for item in descriptors),
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest()
        _write_no_replace(root / "manifest.json", canonical_json_bytes(manifest))
        manifest_hashes[role] = str(manifest["manifest_sha256"])

    report: dict[str, object] = {
        "candidate_rows": candidate_rows,
        "candidate_root_sha256": candidate_root_hasher.hexdigest(),
        "confirmation_manifest_sha256": manifest_hashes["sealed_confirmation"],
        "main_manifest_sha256": manifest_hashes["main"],
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": protocol_freeze,
        "schema": SCHEMA,
        "selected_rows": selected_rows,
        "source_commit": source_commit,
        "status": "pass",
        "unique_candidate_episode_ids": len(all_episode_ids),
        "unique_selected_episode_ids": len(selected_episode_ids),
    }
    report["selection_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--confirmation-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_and_select(
        args.candidates,
        main_output=args.main_output,
        confirmation_output=args.confirmation_output,
    )
    _write_no_replace(args.report, canonical_json_bytes(report))
    print(
        json.dumps(
            {
                "candidate_rows": report["candidate_rows"],
                "selected_rows": report["selected_rows"],
                "selection_sha256": report["selection_sha256"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
