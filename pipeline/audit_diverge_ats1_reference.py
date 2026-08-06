#!/usr/bin/env python3
"""Audit exact ATS1 packet/execution parity over frozen CRP1 boards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from diverge_ats1_data import segment_target, supervisor_states
from diverge_ats1_product import load_jsonl
from diverge_ats1_runtime import compile_segment, execute_step, render_typed_state


def main() -> None:
    parser = argparse.ArgumentParser()
    for split in ("train", "development", "evaluation"):
        parser.add_argument(f"--{split}-data", type=Path, required=True)
        parser.add_argument(f"--{split}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    digest = hashlib.sha256()
    splits: dict[str, dict[str, int]] = {}
    for split in ("train", "development", "evaluation"):
        rows = load_jsonl(
            getattr(args, f"{split}_data"), getattr(args, f"{split}_sha256")
        )
        steps = terminals = 0
        for row in rows:
            states = supervisor_states(row)
            state = None
            for step_index in range(int(row["depth"])):
                target = segment_target(row, step_index, trace_kind="correct")
                packet = compile_segment(
                    target.byte_ids, target.role_ids, target.operation_id
                )
                if (
                    render_typed_state(packet.lhs) != states[step_index]
                    or render_typed_state(packet.rhs_claim) != states[step_index + 1]
                ):
                    raise SystemExit("ATS1 source-role reference parity failed")
                if state is None:
                    state = packet.lhs
                state = execute_step(state, packet.operation_id, packet.arguments)
                rendered = render_typed_state(state)
                if rendered != states[step_index + 1]:
                    raise SystemExit("ATS1 transaction reference parity failed")
                digest.update(
                    (str(row["identity_sha256"]) + str(step_index) + rendered).encode()
                )
                steps += 1
            terminals += render_typed_state(state) == str(row["answer"])
        if terminals != len(rows):
            raise SystemExit("ATS1 terminal reference parity failed")
        splits[split] = {"rows": len(rows), "steps": steps, "terminals": terminals}
    report = {
        "schema": "shohin-diverge-ats1-reference-audit-v1",
        "splits": splits,
        "trajectory_digest": digest.hexdigest(),
        "extensional_parity": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
