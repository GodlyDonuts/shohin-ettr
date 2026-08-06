#!/usr/bin/env python3
"""Run the one frozen autonomous FTA1 contradiction-replay gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from diverge_ats1_data import build_segments
from diverge_ats1_product import compile_segments, load_jsonl, sha256_path
from diverge_fta1_autonomous import ABLATIONS, evaluate_autonomous_replay
from diverge_fta1_runtime import FTA1Config, FiniteStateSourceCompiler


REPORT_SCHEMA = "shohin-diverge-fta1-autonomous-evaluation-v1"


def _load_component_gate(path: Path, expected_sha256: str) -> dict[str, object]:
    if sha256_path(path) != expected_sha256:
        raise SystemExit("FTA1 component gate hash differs")
    gate = json.loads(path.read_text())
    if (
        gate.get("schema") != "shohin-diverge-fta1-component-gate-v1"
        or gate.get("status") != "pass"
        or gate.get("terminal_exact") != 480
    ):
        raise SystemExit("FTA1 component promotion receipt differs")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-update", type=int, default=1600)
    parser.add_argument("--component-gate", type=Path, required=True)
    parser.add_argument("--component-gate-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    if sha256_path(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("FTA1 checkpoint hash differs")
    gate = _load_component_gate(args.component_gate, args.component_gate_sha256)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-diverge-fta1-training-report-v1":
        raise SystemExit("FTA1 checkpoint schema differs")
    if int(payload.get("update", -1)) != args.expected_update:
        raise SystemExit("FTA1 checkpoint update differs")
    model = FiniteStateSourceCompiler(FTA1Config(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    rows = load_jsonl(args.data, args.data_sha256)
    if len(rows) != 480:
        raise SystemExit("autonomous gate requires exactly 480 rows")
    segments = build_segments(rows, trace_kinds=("wrong",))
    compiled, compiler = compile_segments(
        model, segments, device=torch.device("cpu"), batch_size=args.batch_size
    )
    arms = {
        ablation: evaluate_autonomous_replay(rows, compiled, ablation=ablation)
        for ablation in ABLATIONS
    }
    report = {
        "schema": REPORT_SCHEMA,
        "rows": len(rows),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "component_gate": str(args.component_gate),
        "component_gate_sha256": args.component_gate_sha256,
        "component_gate_receipt": gate,
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "source_contract": {
            "runtime_inputs": "compiled typed packets only",
            "answer_labels_available_to_runtime": False,
            "query_available_to_runtime": False,
            "first_conflict_commits_once": True,
            "later_rhs_claims_ignored_after_commit": True,
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
                "selection_exact": arms["normal"]["counts"]["selection_exact"],
                "terminal_exact": arms["normal"]["counts"]["terminal_exact"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
