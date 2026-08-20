#!/usr/bin/env python3
"""Freeze an ordered manifest for a one-GPU public benchmark campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SCHEMA = "shohin-dense-public-campaign-manifest-v1"


class CampaignManifestError(RuntimeError):
    """A benchmark binding or output path differs."""


def parse_binding(value: str) -> dict[str, object]:
    try:
        name, path_text, rows_text, tokens_text = value.split("=", 3)
        path = Path(path_text).resolve()
        rows = int(rows_text)
        tokens = int(tokens_text)
    except (ValueError, TypeError) as exc:
        raise CampaignManifestError(
            "benchmark must be NAME=QUESTIONS=ROWS=MAX_NEW_TOKENS"
        ) from exc
    if not name or not path.is_file() or rows <= 0 or tokens <= 0:
        raise CampaignManifestError("benchmark binding differs")
    actual = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if actual != rows:
        raise CampaignManifestError(f"{name} row count differs: {actual} != {rows}")
    return {
        "name": name,
        "questions": str(path),
        "rows": rows,
        "max_new_tokens": tokens,
    }


def run(bindings: list[str], output: Path) -> dict[str, object]:
    if output.exists():
        raise CampaignManifestError("refusing to replace campaign manifest")
    benchmarks = [parse_binding(value) for value in bindings]
    names = [str(item["name"]) for item in benchmarks]
    if not benchmarks or len(names) != len(set(names)):
        raise CampaignManifestError("benchmark names are empty or duplicated")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": "frozen",
        "benchmarks": benchmarks,
        "rows": sum(int(item["rows"]) for item in benchmarks),
        "generation": {
            "direct_base": "one_pass_no_adapter_same_model_visible_envelope",
            "draft": "trained_draft_adapter",
            "control": "same_draft_adapter_second_pass",
            "treatment": "trained_revision_adapter_second_pass",
            "decoding": "greedy",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run(args.benchmark, args.output)
    print(json.dumps({"rows": payload["rows"], "benchmarks": len(payload["benchmarks"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
