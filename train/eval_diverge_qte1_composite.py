#!/usr/bin/env python3
"""Evaluate pass-only QTE1 with protected WORLD/EVIDENCE and exact execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from diverge_qte1_composite import QTE1StageTypedMachine, QwenEntailmentQueryOwner
from eval_diverge_pqi1 import _load_board, sha256_path
from eval_diverge_sot1 import _load_json, evaluate as evaluate_sot1_path
from eval_diverge_sti1 import _load_sti1


SCHEMA = "shohin-diverge-qte1-composite-evaluation-v1"


class QTE1CompositeEvaluationError(RuntimeError):
    """A QTE1 composition input or capability gate differs."""


def _direct_conditions(report: Mapping[str, Any]) -> dict[str, bool]:
    controls = report["controls"]
    normal = controls["normal"]
    scrub = controls["scrub_context"]
    swap = controls["swap_mentions"]
    exact = int(normal["overall"]["exact"])
    return {
        "query_at_least_765": exact >= 765,
        "every_mode_at_least_254": all(
            int(value["exact"]) >= 254 for value in normal["by_mode"].values()
        ),
        "every_renderer_at_least_127": all(
            int(value["exact"]) >= 127 for value in normal["by_renderer"].values()
        ),
        "scrub_at_most_430": int(scrub["overall"]["exact"]) <= 430,
        "scrub_loses_at_least_250": exact - int(scrub["overall"]["exact"]) >= 250,
        "swap_at_least_765": int(swap["overall"]["exact"]) >= 765,
    }


def _conditions(
    report: Mapping[str, Any],
    *,
    direct_passed: bool,
    owner_hashes_exact: bool,
    stage_hashes_exact: bool,
) -> dict[str, bool]:
    counts = report["natural_query_path"]["counts"]
    underlying = report["promotion_gate"]["conditions"]
    return {
        "direct_confirmation_passed": direct_passed,
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
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--direct-confirmation", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--protected-nve1-result", type=Path, required=True)
    parser.add_argument("--protected-nve1-result-sha256", required=True)
    parser.add_argument("--protected-tol3-result", type=Path, required=True)
    parser.add_argument("--protected-tol3-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing QTE1 composite: {args.output}")
    if not torch.cuda.is_available():
        raise SystemExit("QTE1 composite requires CUDA")
    direct = json.loads(args.direct_confirmation.read_text(encoding="utf-8"))
    if (
        direct.get("schema") != "shohin-diverge-gti1-qwen-entailment-attribution-v1"
        or direct.get("board_type") != "confirmation"
        or direct.get("data_sha256") != args.data_sha256
    ):
        raise QTE1CompositeEvaluationError("QTE1 direct confirmation receipt differs")
    direct_conditions = _direct_conditions(direct)
    if not all(direct_conditions.values()):
        raise QTE1CompositeEvaluationError("QTE1 direct confirmation did not pass")

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    device = torch.device("cuda:0")
    rows = _load_board(args.data, args.data_sha256, "confirmation")
    stage, _ = _load_sti1(
        args.stage_checkpoint, args.stage_checkpoint_sha256, device
    )
    stage_hashes = stage.owner_hashes()
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    backbone = AutoModelForMultimodalLM.from_pretrained(
        args.model_root,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    model_receipt = {
        name: sha256_path(args.model_root / name)
        for name in ("config.json", "model.safetensors-00001-of-00001.safetensors", "tokenizer.json")
    }
    query = QwenEntailmentQueryOwner(
        backbone, tokenizer, model_receipt=model_receipt, batch_size=args.batch_size
    )
    model = QTE1StageTypedMachine(
        stage.source_owner, stage.evidence_owner, query
    ).eval()
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
    underlying = report.pop("promotion_gate")
    report["underlying_transaction_gate"] = underlying
    report["promotion_gate"] = underlying
    after = model.owner_hashes()
    stage_hashes_exact = (
        before["WORLD"] == stage_hashes["WORLD"]
        and before["EVIDENCE"] == stage_hashes["EVIDENCE"]
    )
    conditions = _conditions(
        report,
        direct_passed=True,
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
            "direct_confirmation": str(args.direct_confirmation),
            "direct_confirmation_sha256": sha256_path(args.direct_confirmation),
            "direct_confirmation_conditions": direct_conditions,
            "data": str(args.data),
            "data_sha256": args.data_sha256,
            "source_commit": args.source_commit,
            "stage_checkpoint": str(args.stage_checkpoint),
            "stage_checkpoint_sha256": args.stage_checkpoint_sha256,
            "model_root": str(args.model_root),
            "model_receipt": model_receipt,
            "owner_manifest": model.owner_manifest(),
            "owner_hashes_before": before,
            "owner_hashes_after": after,
            "new_training_updates": 0,
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
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
