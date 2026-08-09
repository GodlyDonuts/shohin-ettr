#!/usr/bin/env python3
"""Read-only outcome attribution for completed OLMoE revision arms."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-moe-revision-error-attribution-v1"
GROUPS = ("corrected", "broken", "preserved_correct", "persistent_wrong")


class AttributionError(RuntimeError):
    """Completed MoE artifacts are incomplete or identity-misaligned."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get("identity_sha256", ""))
            if not identity or identity in rows:
                raise AttributionError(f"candidate identity differs: {path}")
            rows[identity] = row
    if len(rows) != 1289:
        raise AttributionError(f"candidate coverage differs: {path}")
    return rows


def transition(before: bool, after: bool) -> str:
    if before and after:
        return "preserved_correct"
    if before:
        return "broken"
    if after:
        return "corrected"
    return "persistent_wrong"


def summarize(
    baseline: dict[str, dict[str, Any]], arm: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, str]]:
    groups: dict[str, str] = {}
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    generated_tokens: dict[str, list[int]] = defaultdict(list)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for identity in sorted(baseline):
        before = baseline[identity]
        after = arm[identity]
        if before["task"] != after["task"]:
            raise AttributionError("task identity differs")
        group = transition(bool(before["correct"]), bool(after["correct"]))
        groups[identity] = group
        task = str(after["task"])
        by_task[task][group] += 1
        counters[group]["rows"] += 1
        counters[group]["baseline_exhausted"] += int(before["max_token_exhausted"])
        counters[group]["arm_exhausted"] += int(after["max_token_exhausted"])
        counters[group]["baseline_explicit_final"] += int(
            before.get("explicit_final_answer") is True
        )
        counters[group]["arm_explicit_final"] += int(
            after.get("explicit_final_answer") is True
        )
        generated_tokens[group].append(int(after["generated_tokens"]))
    group_summary = {}
    for group in GROUPS:
        values = counters[group]
        count = values["rows"]
        group_summary[group] = {
            **dict(values),
            "arm_mean_generated_tokens": (
                sum(generated_tokens[group]) / count if count else None
            ),
        }
    return {
        "groups": group_summary,
        "by_task": {task: dict(counts) for task, counts in sorted(by_task.items())},
    }, groups


def select_route_board(
    groups: dict[str, str], rows_by_identity: dict[str, dict[str, Any]], per_group: int
) -> list[dict[str, str]]:
    selected = []
    for group in GROUPS:
        identities = sorted(identity for identity, value in groups.items() if value == group)
        limit = min(per_group, len(identities))
        for identity in identities[:limit]:
            selected.append(
                {
                    "identity_sha256": identity,
                    "group": group,
                    "task": str(rows_by_identity[identity]["task"]),
                }
            )
    return selected


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing attribution: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unchanged", type=Path, required=True)
    parser.add_argument("--mtr", type=Path, required=True)
    parser.add_argument("--rcr", type=Path, required=True)
    parser.add_argument("--rank1-attention", type=Path, required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_group <= 0:
        parser.error("per-group must be positive")
    paths = {
        "unchanged": args.unchanged,
        "mtr_rank8_attention": args.mtr,
        "rcr_router": args.rcr,
        "rank1_attention": args.rank1_attention,
    }
    candidates = {name: load_candidates(path) for name, path in paths.items()}
    identities = set(candidates["unchanged"])
    if any(set(rows) != identities for rows in candidates.values()):
        raise AttributionError("arm identity coverage differs")
    data_rows = {
        str(row["identity_sha256"]): row
        for row in (
            json.loads(line)
            for line in args.development_data.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if set(data_rows) != identities:
        raise AttributionError("development data identity coverage differs")
    arms: dict[str, Any] = {}
    group_maps: dict[str, dict[str, str]] = {}
    for name in ("mtr_rank8_attention", "rcr_router", "rank1_attention"):
        arms[name], group_maps[name] = summarize(
            candidates["unchanged"], candidates[name]
        )
    route_board = select_route_board(
        group_maps["mtr_rank8_attention"], data_rows, args.per_group
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "candidate_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "development_data_sha256": sha256_file(args.development_data),
        "identities": len(identities),
        "arms": arms,
        "route_board_selection": "lowest_identity_sha256_within_mtr_transition_group",
        "route_board": route_board,
        "route_board_rows": len(route_board),
        "route_board_per_group_cap": args.per_group,
    }
    atomic_json(args.output, report)
    print(json.dumps({"arms": arms, "route_board_rows": len(route_board)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
