#!/usr/bin/env python3
"""Attribute IEM1 failure by splicing its readers onto the qualified source.

This is a read-only postmortem. It cannot promote IEM1 and never changes a
checkpoint, board, prediction, or frozen gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from diverge_iem1_runtime import module_state_sha256, seal_natural_query
from diverge_nve1_runtime import seal_natural_evidence
from diverge_tfs1_runtime import execute_factorized, query_receipt
from eval_diverge_iem1 import (
    _answer_exact,
    _compile_evidence_for_packets,
    _compile_natural_queries,
    _compile_packets,
    _compiled_program_exact,
    _load_board,
    _load_evidence_model,
    _load_iem1,
    _load_tol3,
    _natural_query_exact,
    _natural_receipt_exact,
    _receipt_tuple,
    sha256_path,
)


SCHEMA = "shohin-diverge-iem1-stage-ownership-diagnostic-v1"


class StageOwnershipDiagnosticError(RuntimeError):
    """The read-only splice diagnostic violated its contract."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def diagnose(
    rows: list[dict[str, Any]],
    iem1,
    tol3,
    *,
    iem1_commitment: str,
    tol3_commitment: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    started = time.monotonic()
    packets, _, source_failures = _compile_packets(
        rows,
        tol3,
        commitment=tol3_commitment,
        device=device,
        integrated=False,
    )
    evidence_sets = _compile_evidence_for_packets(
        iem1,
        rows,
        packets,
        commitment=iem1_commitment,
        device=device,
        batch_size=batch_size,
    )
    query_sets = _compile_natural_queries(
        iem1,
        rows,
        packets,
        commitment=iem1_commitment,
        device=device,
    )

    counts = Counter()
    failures: list[dict[str, str]] = []
    for row, packet, evidence_set, query_set in zip(
        rows, packets, evidence_sets, query_sets, strict=True
    ):
        tfs1 = row["tfs1"]
        counts["episodes"] += 1
        program_exact = packet is not None and _compiled_program_exact(packet, tfs1)
        counts["source_program_exact"] += program_exact
        if packet is None or not program_exact:
            failures.append({"id": str(tfs1["id"]), "error": "source packet absent"})
            continue

        if evidence_set is None:
            failures.append({"id": str(tfs1["id"]), "error": "evidence set absent"})
            continue
        for compilation, supervisor in zip(
            evidence_set, row["natural_evidence"], strict=True
        ):
            counts["evidence_total"] += 1
            counts["evidence_compiled"] += compilation.receipt is not None
            counts["evidence_exact"] += _natural_receipt_exact(compilation, supervisor)

        if query_set is None:
            failures.append({"id": str(tfs1["id"]), "error": "query set absent"})
            continue
        for name, compilation in query_set.items():
            counts["query_total"] += 1
            counts["query_compiled"] += (
                compilation.receipt is not None and compilation.query is not None
            )
            counts["query_exact"] += _natural_query_exact(
                compilation, row["natural_queries"][name]
            )

        receipts = _receipt_tuple(evidence_set)
        sensitive = query_set["sensitive"]
        if receipts is None or sensitive.receipt is None:
            failures.append(
                {"id": str(tfs1["id"]), "error": "complete splice did not compile"}
            )
            continue
        try:
            typed = seal_natural_evidence(
                packet,
                receipts,
                expected_compiler_commitment=iem1_commitment,
            )
            query = seal_natural_query(
                packet,
                sensitive.receipt,
                expected_compiler_commitment=iem1_commitment,
            )
            execution = execute_factorized(packet, typed)
            decision = query_receipt(packet, execution, query)
        except Exception as error:  # fail closed and preserve the exact reason
            failures.append({"id": str(tfs1["id"]), "error": str(error)})
            continue
        counts["episodes_fully_sealed"] += 1
        counts["sensitive_answer_exact"] += _answer_exact(
            decision, str(tfs1["gold_answer"])
        )

    return {
        "counts": dict(counts),
        "source_failures": source_failures,
        "splice_failures": failures,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iem1-checkpoint", type=Path, required=True)
    parser.add_argument("--iem1-checkpoint-sha256", required=True)
    parser.add_argument("--ceiling-evidence-checkpoint", type=Path, required=True)
    parser.add_argument("--ceiling-evidence-checkpoint-sha256", required=True)
    parser.add_argument("--ceiling-tol3-checkpoint", type=Path, required=True)
    parser.add_argument("--ceiling-tol3-checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing diagnostic: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA diagnostic requested without CUDA")
    device = torch.device(args.device)

    rows = _load_board(args.data, args.data_sha256)
    iem1, iem1_checkpoint = _load_iem1(
        args.iem1_checkpoint, args.iem1_checkpoint_sha256, device
    )
    _, evidence_checkpoint = _load_evidence_model(
        args.ceiling_evidence_checkpoint,
        args.ceiling_evidence_checkpoint_sha256,
        device,
    )
    tol3, tol3_checkpoint = _load_tol3(
        args.ceiling_tol3_checkpoint,
        args.ceiling_tol3_checkpoint_sha256,
        device,
    )
    iem1_commitment = module_state_sha256(iem1)
    result = diagnose(
        rows,
        iem1,
        tol3,
        iem1_commitment=iem1_commitment,
        tol3_commitment=str(tol3_checkpoint["model_state_sha256"]),
        device=device,
        batch_size=args.batch_size,
    )
    payload = {
        "schema": SCHEMA,
        "status": "diagnostic_only",
        "device": str(device),
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "iem1_checkpoint": str(args.iem1_checkpoint),
        "iem1_checkpoint_sha256": args.iem1_checkpoint_sha256,
        "iem1_model_state_sha256": iem1_commitment,
        "ceiling_evidence_checkpoint": str(args.ceiling_evidence_checkpoint),
        "ceiling_evidence_checkpoint_sha256": (args.ceiling_evidence_checkpoint_sha256),
        "ceiling_evidence_model_state_sha256": str(
            evidence_checkpoint["model_state_sha256"]
        ),
        "ceiling_tol3_checkpoint": str(args.ceiling_tol3_checkpoint),
        "ceiling_tol3_checkpoint_sha256": args.ceiling_tol3_checkpoint_sha256,
        "ceiling_tol3_model_state_sha256": str(tol3_checkpoint["model_state_sha256"]),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, payload)
    output_sha256 = sha256_path(args.output)
    print(
        json.dumps(
            {
                "counts": payload["counts"],
                "output": str(args.output),
                "output_sha256": output_sha256,
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
