#!/usr/bin/env python3
"""Evaluate full-document transaction scanning and typed replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from diverge_ats1_product import load_jsonl, sha256_path
from diverge_fta1_autonomous import ABLATIONS, evaluate_autonomous_replay
from diverge_fta1_runtime import FTA1Config, FiniteStateSourceCompiler
from diverge_nta1_data import natural_segment_target
from diverge_nta2_product import compile_nta2_segments
from diverge_nta3_scanner import NTA3ScannerError, scan_transactions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--nta2-gate", type=Path, required=True)
    parser.add_argument("--nta2-gate-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    if sha256_path(args.nta2_gate) != args.nta2_gate_sha256:
        raise SystemExit("NTA2 gate hash differs")
    prior = json.loads(args.nta2_gate.read_text())
    if prior.get("schema") != "shohin-diverge-nta2-gate-v1" or prior.get("status") != "pass":
        raise SystemExit("NTA3 requires the frozen NTA2 pass")
    if sha256_path(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("NTA3 checkpoint hash differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FiniteStateSourceCompiler(FTA1Config(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    rows = load_jsonl(args.data, args.data_sha256)
    if len(rows) != 279 or any(row.get("schema") != "shohin-diverge-nta3-board-v1" for row in rows):
        raise SystemExit("NTA3 board contract differs")
    segments = []
    scanner_rows = 0
    scanner_transactions = 0
    for row in rows:
        try:
            transactions = scan_transactions(str(row["document"]))
        except NTA3ScannerError:
            continue
        expected_hashes = list(map(str, row["transaction_sha256s"]))
        actual_hashes = [hashlib.sha256(value.encode()).hexdigest() for value in transactions]
        scanner_rows += actual_hashes == expected_hashes
        scanner_transactions += sum(
            left == right for left, right in zip(actual_hashes, expected_hashes, strict=False)
        )
        temporary_row = {
            "identity_sha256": row["identity_sha256"],
            "wrong_steps": list(transactions),
        }
        for step_index in range(len(transactions)):
            segments.append(
                natural_segment_target(
                    temporary_row, step_index, trace_kind="wrong"
                )
            )
    compiled, compiler = compile_nta2_segments(
        model, segments, device=torch.device("cpu"), batch_size=args.batch_size
    )
    arms = {
        ablation: evaluate_autonomous_replay(rows, compiled, ablation=ablation)
        for ablation in ABLATIONS
    }
    report = {
        "schema": "shohin-diverge-nta3-evaluation-v1",
        "rows": len(rows),
        "checkpoint_sha256": args.checkpoint_sha256,
        "nta2_gate_sha256": args.nta2_gate_sha256,
        "data_sha256": args.data_sha256,
        "updates_after_fta1": 0,
        "scanner": {
            "exact_rows": scanner_rows,
            "exact_transactions": scanner_transactions,
            "target_rows": len(rows),
            "target_transactions": sum(int(row["depth"]) for row in rows),
        },
        "compiler": compiler,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scanner_rows": scanner_rows,
                "terminal_exact": arms["normal"]["counts"]["terminal_exact"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
