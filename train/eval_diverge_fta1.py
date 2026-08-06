#!/usr/bin/env python3
"""Evaluate the one frozen DIVERGE-FTA1 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from diverge_ats1_product import evaluate_model, load_jsonl, sha256_path
from diverge_fta1_runtime import FTA1Config, FiniteStateSourceCompiler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-update", type=int, default=1600)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    if sha256_path(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("FTA1 checkpoint hash differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-diverge-fta1-training-report-v1":
        raise SystemExit("FTA1 checkpoint schema differs")
    if int(payload.get("update", -1)) != args.expected_update:
        raise SystemExit("FTA1 checkpoint update differs")
    model = FiniteStateSourceCompiler(FTA1Config(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = load_jsonl(args.data, args.data_sha256)
    result = evaluate_model(model, rows, device=device, batch_size=args.batch_size)
    report = {
        "schema": "shohin-diverge-fta1-forced-evaluation-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "rows": len(rows),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "terminal_exact": result["replay"]["normal"]["counts"]["terminal_exact"]}, sort_keys=True))


if __name__ == "__main__":
    main()
