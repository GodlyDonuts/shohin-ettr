#!/usr/bin/env python3
"""Create a hash-bound DSET report view exposing one existing split as diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


REPORT_SCHEMA = "shohin-dset1-span-edit-data-report-v1"
VIEW_SCHEMA = "shohin-dset-split-view-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.split not in {"train", "diagnostic"}:
        raise RuntimeError("DSET split view output or split differs")
    source = json.loads(args.report.read_text())
    selected = source.get("outputs", {}).get(args.split, {})
    data = Path(str(selected.get("path", ""))).resolve()
    if (
        source.get("schema") != REPORT_SCHEMA
        or source.get("status") != "complete"
        or source.get("holdout_used") is not False
        or not data.is_file()
        or selected.get("sha256") != sha256_file(data)
        or int(selected.get("rows", 0)) <= 0
        or int(selected.get("sources", 0)) <= 0
    ):
        raise RuntimeError("DSET split view source differs")
    view = dict(source)
    view["outputs"] = dict(source["outputs"])
    view["outputs"]["diagnostic"] = dict(selected)
    view["split_view"] = {
        "schema": VIEW_SCHEMA,
        "source_report": str(args.report.resolve()),
        "source_report_sha256": sha256_file(args.report),
        "selected_split": args.split,
        "holdout_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(view, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return view


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "diagnostic"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result["split_view"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
