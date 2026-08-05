#!/usr/bin/env python3
"""Merge complete candidate arms while preserving arm order and provenance."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-verified-code-candidate-merge-v1"


class VerifiedCodeCandidateMergeError(RuntimeError):
    """Candidate arms do not describe the same complete task bank."""


def _groups(rows: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        identity = str(row.get("identity_sha256") or "")
        if not identity:
            raise VerifiedCodeCandidateMergeError("candidate identity is missing")
        groups.setdefault(identity, []).append(row)
    if not groups:
        raise VerifiedCodeCandidateMergeError("candidate arm is empty")
    for group in groups.values():
        group.sort(key=lambda row: int(row.get("sample_index", 0)))
        indices = [int(row.get("sample_index", 0)) for row in group]
        if indices != list(range(len(group))):
            raise VerifiedCodeCandidateMergeError("arm sample indices differ")
        if any(str(row.get("task")) != str(group[0].get("task")) for row in group):
            raise VerifiedCodeCandidateMergeError("arm candidate tasks differ")
    counts = {len(group) for group in groups.values()}
    if len(counts) != 1:
        raise VerifiedCodeCandidateMergeError("arm sample counts differ")
    return groups


def merge_arms(
    arms: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not arms:
        raise VerifiedCodeCandidateMergeError("no candidate arms were supplied")
    labels = [label for label, _ in arms]
    if len(set(labels)) != len(labels):
        raise VerifiedCodeCandidateMergeError("candidate arm labels differ")
    grouped = [(label, _groups(rows)) for label, rows in arms]
    identities = list(grouped[0][1])
    identity_set = set(identities)
    for _label, groups in grouped[1:]:
        if set(groups) != identity_set:
            raise VerifiedCodeCandidateMergeError("candidate arm identities differ")

    merged: list[dict[str, Any]] = []
    arm_samples: dict[str, int] = {}
    for identity in identities:
        next_index = 0
        task = str(grouped[0][1][identity][0].get("task"))
        for label, groups in grouped:
            group = groups[identity]
            if str(group[0].get("task")) != task:
                raise VerifiedCodeCandidateMergeError("candidate arm tasks differ")
            arm_samples[label] = len(group)
            for source in group:
                row = dict(source)
                row["candidate_arm"] = label
                row["source_sample_index"] = int(source.get("sample_index", 0))
                row["sample_index"] = next_index
                merged.append(row)
                next_index += 1
    return merged, {
        "schema": SCHEMA,
        "identities": len(identities),
        "rows": len(merged),
        "arms": labels,
        "arm_samples": arm_samples,
        "samples_per_identity": sum(arm_samples.values()),
    }


def _load(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if path.suffix == ".json":
        parsed = json.loads(payload)
        rows = parsed.get("results") if isinstance(parsed, dict) else parsed
        if not isinstance(rows, list):
            raise VerifiedCodeCandidateMergeError("JSON source has no result rows")
        task = parsed.get("task") if isinstance(parsed, dict) else None
        normalized: list[dict[str, Any]] = []
        for row in rows:
            candidate = dict(row)
            if task is not None:
                candidate.setdefault("task", task)
            candidate.setdefault("sample_index", 0)
            normalized.append(candidate)
        return normalized
    return [json.loads(line) for line in payload.splitlines() if line]


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VerifiedCodeCandidateMergeError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VerifiedCodeCandidateMergeError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Ordered LABEL=PATH candidate arm; JSON eval reports or JSONL are accepted.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    arms: list[tuple[str, list[dict[str, Any]]]] = []
    sources: list[dict[str, str]] = []
    for specification in args.arm:
        if "=" not in specification:
            parser.error("arm must be LABEL=PATH")
        label, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        payload = path.read_bytes()
        arms.append((label, _load(path)))
        sources.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    merged, report = merge_arms(arms)
    report["sources"] = sources
    report["output_sha256"] = _atomic_lines(args.output, merged)
    report["output"] = str(args.output.resolve())
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
