#!/usr/bin/env python3
"""Evaluate the pass-only PQI1 plus protected WORLD/EVIDENCE composition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from diverge_pqi1_composite import PretrainedStageTypedMachine
from eval_diverge_pqi1 import _load_board, _load_model, sha256_path
from eval_diverge_sot1 import _load_json, evaluate as evaluate_sot1_path
from eval_diverge_sti1 import _load_sti1


SCHEMA = "shohin-diverge-pqi1-composite-evaluation-v1"


class PQI1CompositeEvaluationError(RuntimeError):
    """A PQI1 composition input or gate differs."""


def _load_direct(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise PQI1CompositeEvaluationError("PQI1 direct confirmation hash differs")
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "shohin-diverge-pqi1-evaluation-v1"
        or report.get("board_type") != "confirmation"
        or report.get("status") != "pass"
        or report.get("promotion_gate", {}).get("passed") is not True
    ):
        raise PQI1CompositeEvaluationError("PQI1 direct confirmation did not pass")
    return report


def _conditions(
    report: Mapping[str, Any],
    direct: Mapping[str, Any],
    *,
    owner_hashes_exact: bool,
    stage_hashes_exact: bool,
) -> dict[str, bool]:
    counts = report["natural_query_path"]["counts"]
    underlying = report["promotion_gate"]["conditions"]
    return {
        "direct_confirmation_passed": direct["promotion_gate"]["passed"] is True,
        "underlying_transaction_gate_passed": all(underlying.values()),
        "world_256": int(counts.get("source_program_exact", 0)) == 256,
        "evidence_at_least_3070": int(counts.get("evidence_exact", 0)) >= 3070,
        "sealed_episodes_at_least_255": int(
            counts.get("episodes_fully_sealed", 0)
        )
        >= 255,
        "query_at_least_765": int(counts.get("query_total", 0)) == 768
        and int(counts.get("query_exact", 0)) >= 765,
        "sensitive_answers_at_least_254": int(counts.get("sensitive_exact", 0))
        >= 254,
        "extensional_parity_at_least_254": int(
            counts.get("extensional_parity", 0)
        )
        >= 254,
        "no_evidence_abstains_at_least_254": int(
            counts.get("no_evidence_abstain", 0)
        )
        >= 254,
        "invariant_answers_at_least_254": int(counts.get("invariant_exact", 0))
        >= 254,
        "partial_underdetermined_at_least_254": int(
            counts.get("partial_underdetermined_abstain", 0)
        )
        >= 254,
        "all_owner_hashes_bit_exact": owner_hashes_exact,
        "protected_world_evidence_hashes_exact": stage_hashes_exact,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--query-checkpoint-sha256", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--direct-confirmation", type=Path, required=True)
    parser.add_argument("--direct-confirmation-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--protected-nve1-result", type=Path, required=True)
    parser.add_argument("--protected-nve1-result-sha256", required=True)
    parser.add_argument("--protected-tol3-result", type=Path, required=True)
    parser.add_argument("--protected-tol3-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing PQI1 composite result: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("PQI1 composite requested unavailable CUDA")
    device = torch.device(args.device)

    rows = _load_board(args.data, args.data_sha256, "confirmation")
    direct = _load_direct(args.direct_confirmation, args.direct_confirmation_sha256)
    if (
        direct.get("data_sha256") != args.data_sha256
        or direct.get("checkpoint_sha256") != args.query_checkpoint_sha256
        or direct.get("base_sha256") != args.base_sha256
        or direct.get("tokenizer_sha256") != args.tokenizer_sha256
        or direct.get("backbone_name") != "smollm2"
        or direct.get("shuffle_supervision") is not False
    ):
        raise PQI1CompositeEvaluationError("PQI1 direct confirmation receipts differ")

    stage, _ = _load_sti1(
        args.stage_checkpoint, args.stage_checkpoint_sha256, device
    )
    stage_hashes = stage.owner_hashes()
    query, query_receipt = _load_model(
        args.query_checkpoint,
        args.query_checkpoint_sha256,
        args.base,
        args.base_sha256,
        args.tokenizer,
        args.tokenizer_sha256,
        device,
    )
    model = PretrainedStageTypedMachine(
        stage.source_owner,
        stage.evidence_owner,
        query,
        tokenizer_sha256=args.tokenizer_sha256,
    ).to(device).eval()
    before = model.owner_hashes()

    protected_nve1 = _load_json(
        args.protected_nve1_result, args.protected_nve1_result_sha256
    )
    protected_nve1["_sha256"] = args.protected_nve1_result_sha256
    protected_tol3 = _load_json(
        args.protected_tol3_result, args.protected_tol3_result_sha256
    )
    protected_tol3["_sha256"] = args.protected_tol3_result_sha256
    report = evaluate_sot1_path(
        rows,
        model,  # type: ignore[arg-type]
        protected_nve1=protected_nve1,
        protected_tol3=protected_tol3,
        device=device,
        batch_size=args.batch_size,
    )
    underlying_gate = report.pop("promotion_gate")
    report["underlying_transaction_gate"] = underlying_gate
    report["promotion_gate"] = underlying_gate
    after = model.owner_hashes()
    stage_hashes_exact = (
        before["WORLD"] == stage_hashes["WORLD"]
        and before["EVIDENCE"] == stage_hashes["EVIDENCE"]
    )
    conditions = _conditions(
        report,
        direct,
        owner_hashes_exact=before == after,
        stage_hashes_exact=stage_hashes_exact,
    )
    report.update(
        {
            "schema": SCHEMA,
            "status": "pass" if all(conditions.values()) else "fail",
            "promotion_gate": {
                "conditions": conditions,
                "passed": all(conditions.values()),
            },
            "source_commit": args.source_commit,
            "data": str(args.data),
            "data_sha256": args.data_sha256,
            "stage_checkpoint": str(args.stage_checkpoint),
            "stage_checkpoint_sha256": args.stage_checkpoint_sha256,
            "query_checkpoint": str(args.query_checkpoint),
            "query_checkpoint_sha256": args.query_checkpoint_sha256,
            "query_adapter_state_sha256": query_receipt["adapter_state_sha256"],
            "direct_confirmation": str(args.direct_confirmation),
            "direct_confirmation_sha256": args.direct_confirmation_sha256,
            "owner_manifest": model.owner_manifest(),
            "owner_hashes_before": before,
            "owner_hashes_after": after,
            "new_training_updates": 0,
            "device": str(device),
        }
    )
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "status": report["status"],
                "counts": report["natural_query_path"]["counts"],
                "promotion_gate": report["promotion_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
