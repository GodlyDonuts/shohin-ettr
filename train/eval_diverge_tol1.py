#!/usr/bin/env python3
"""Evaluate one immutable DIVERGE-TOL1 checkpoint on the OOD board."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from diverge_tol1_product import evaluate_programs, load_rows, sha256_path
from diverge_tol1_runtime import TOL1Config, TypedOperationCompiler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing TOL1 evaluation: {args.output}")
    if sha256_path(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("TOL1 checkpoint hash differs")
    rows = load_rows(args.data, args.data_sha256, "ood")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-tol1-training-report-v1":
        raise SystemExit("TOL1 checkpoint schema differs")
    config = TOL1Config(**checkpoint["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TypedOperationCompiler(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    report = evaluate_programs(model, rows, device=device, batch_size=args.batch_size)
    report.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "data": str(args.data),
            "data_sha256": args.data_sha256,
        }
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "sha256": sha256_path(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
