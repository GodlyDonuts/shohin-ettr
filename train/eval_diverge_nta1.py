#!/usr/bin/env python3
"""Evaluate zero-shot FTA1 transfer to verified natural arithmetic traces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from diverge_ats1_product import compile_segments, load_jsonl, sha256_path
from diverge_fta1_autonomous import ABLATIONS, evaluate_autonomous_replay
from diverge_fta1_runtime import FTA1Config, FiniteStateSourceCompiler
from diverge_nta1_data import build_nta1_segments


def _require_gate(path: Path, expected_sha256: str, schema: str) -> None:
    if sha256_path(path) != expected_sha256:
        raise SystemExit("NTA1 prerequisite gate hash differs")
    payload = json.loads(path.read_text())
    if payload.get("schema") != schema or payload.get("status") != "pass":
        raise SystemExit("NTA1 prerequisite gate did not pass")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--component-gate", type=Path, required=True)
    parser.add_argument("--component-gate-sha256", required=True)
    parser.add_argument("--autonomous-gate", type=Path, required=True)
    parser.add_argument("--autonomous-gate-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    _require_gate(
        args.component_gate,
        args.component_gate_sha256,
        "shohin-diverge-fta1-component-gate-v1",
    )
    _require_gate(
        args.autonomous_gate,
        args.autonomous_gate_sha256,
        "shohin-diverge-fta1-autonomous-gate-v1",
    )
    if sha256_path(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("NTA1 checkpoint hash differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FiniteStateSourceCompiler(FTA1Config(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    rows = load_jsonl(args.data, args.data_sha256)
    if len(rows) != 279 or any(row.get("schema") != "shohin-diverge-nta1-board-v1" for row in rows):
        raise SystemExit("NTA1 board contract differs")
    segments = build_nta1_segments(rows)
    compiled, compiler = compile_segments(
        model, segments, device=torch.device("cpu"), batch_size=args.batch_size
    )
    arms = {
        ablation: evaluate_autonomous_replay(rows, compiled, ablation=ablation)
        for ablation in ABLATIONS
    }
    normal_per_error_operation = {}
    for operation in ("add", "subtract", "multiply"):
        subset = [
            row
            for row in rows
            if row["program"][int(row["error_index"]) - 1][0] == operation
        ]
        normal_per_error_operation[operation] = evaluate_autonomous_replay(
            subset, compiled, ablation="normal"
        )
    normal_per_depth = {
        str(depth): evaluate_autonomous_replay(
            [row for row in rows if int(row["depth"]) == depth],
            compiled,
            ablation="normal",
        )
        for depth in (2, 3, 4, 5)
    }
    report = {
        "schema": "shohin-diverge-nta1-evaluation-v1",
        "rows": len(rows),
        "checkpoint_sha256": args.checkpoint_sha256,
        "component_gate_sha256": args.component_gate_sha256,
        "autonomous_gate_sha256": args.autonomous_gate_sha256,
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "compiler": compiler,
        "arms": arms,
        "normal_per_error_operation": normal_per_error_operation,
        "normal_per_depth": normal_per_depth,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "valid_segments": compiler["counts"].get("valid", 0),
                "segments": compiler["counts"]["segments"],
                "selection_exact": arms["normal"]["counts"]["selection_exact"],
                "terminal_exact": arms["normal"]["counts"]["terminal_exact"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
