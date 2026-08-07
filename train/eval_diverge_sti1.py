#!/usr/bin/env python3
"""Evaluate the frozen zero-training DIVERGE-STI1 composition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from diverge_iem1_runtime import module_state_sha256
from diverge_rrg1_runtime import RRG1Config
from diverge_sti1_runtime import StageTypedInterfaceMachine, validate_owner_contract
from eval_diverge_ccr1 import _referent_records, _rename_records
from eval_diverge_iem1 import sha256_path
from eval_diverge_rrg1 import (
    _development_conditions,
    _fresh_conditions,
    _load_board,
    _public_score,
    _score_referent,
)
from eval_diverge_sot1 import _load_json, evaluate as evaluate_sot1_path
from eval_diverge_srp1 import _load_srp1, _score_query_owner


SCHEMA = "shohin-diverge-sti1-evaluation-v1"


class STI1EvaluationError(RuntimeError):
    """The frozen STI1 evaluation contract was violated."""


def _load_sti1(
    path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[StageTypedInterfaceMachine, dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise STI1EvaluationError("STI1 source checkpoint hash differs")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-rrg1-training-report-v1":
        raise STI1EvaluationError("STI1 source checkpoint schema differs")
    if int(checkpoint.get("update", -1)) != 2000:
        raise STI1EvaluationError("STI1 source checkpoint duration differs")
    model = StageTypedInterfaceMachine(RRG1Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.freeze_owners()
    validate_owner_contract(model)
    model.eval()
    if module_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise STI1EvaluationError("STI1 source model state differs")
    hashes = model.owner_hashes()
    expected_hashes = {
        "WORLD": checkpoint["final_owner_hashes"]["WORLD"],
        "EVIDENCE": checkpoint["final_owner_hashes"]["NUMERIC_EVIDENCE"],
        "QUERY": checkpoint["final_owner_hashes"]["REFERENT"],
    }
    if hashes != expected_hashes:
        raise STI1EvaluationError("STI1 routed owner hash differs")
    return model, checkpoint


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--srp1-checkpoint", type=Path)
    parser.add_argument("--srp1-checkpoint-sha256")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--protected-nve1-result", type=Path, required=True)
    parser.add_argument("--protected-nve1-result-sha256", required=True)
    parser.add_argument("--protected-tol3-result", type=Path, required=True)
    parser.add_argument("--protected-tol3-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing STI1 result: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("STI1 requested unavailable CUDA")
    if args.split == "confirmation" and (
        args.srp1_checkpoint is None or args.srp1_checkpoint_sha256 is None
    ):
        raise SystemExit("STI1 confirmation requires the frozen SRP1 checkpoint")
    device = torch.device(args.device)

    rows = _load_board(args.data, args.data_sha256, split=args.split)
    model, checkpoint = _load_sti1(args.checkpoint, args.checkpoint_sha256, device)
    hashes_before = model.owner_hashes()
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
    report["underlying_sot1_gate"] = report.pop("promotion_gate")
    records = _referent_records(rows)
    normal = _score_referent(
        model,  # type: ignore[arg-type]
        records,
        device=device,
        batch_size=args.batch_size,
    )
    hashes_after = model.owner_hashes()
    owner_hashes_exact = hashes_before == hashes_after
    report["schema"] = SCHEMA
    report["split"] = args.split
    report["semantic_owner"] = {"query_owner_normal": _public_score(normal)}

    if args.split == "development":
        conditions = _development_conditions(report)
    else:
        assert args.srp1_checkpoint is not None
        assert args.srp1_checkpoint_sha256 is not None
        srp1, _ = _load_srp1(
            args.srp1_checkpoint,
            args.srp1_checkpoint_sha256,
            device,
        )
        srp1_query = _score_query_owner(
            srp1, rows, device=device, batch_size=args.batch_size
        )
        srp1_end_to_end = evaluate_sot1_path(
            rows,
            srp1,  # type: ignore[arg-type]
            protected_nve1=protected_nve1,
            protected_tol3=protected_tol3,
            device=device,
            batch_size=args.batch_size,
        )
        role_slot_swap = _score_referent(
            model,  # type: ignore[arg-type]
            records,
            device=device,
            batch_size=args.batch_size,
            control="role_slot_swap",
        )
        marker_delete = _score_referent(
            model,  # type: ignore[arg-type]
            records,
            device=device,
            batch_size=args.batch_size,
            control="marker_delete",
        )
        renamed = _score_referent(
            model,  # type: ignore[arg-type]
            _rename_records(records),
            device=device,
            batch_size=args.batch_size,
        )
        baseline = {
            "checkpoint": str(args.srp1_checkpoint),
            "checkpoint_sha256": args.srp1_checkpoint_sha256,
            "direct_query": srp1_query,
            "direct_query_exact": int(srp1_query["overall"].get("exact", 0)),
            "natural_query_path": srp1_end_to_end["natural_query_path"],
            "fresh_nve1": srp1_end_to_end["fresh_nve1"],
        }
        conditions = _fresh_conditions(
            report,
            baseline,
            normal,
            role_slot_swap,
            marker_delete,
            renamed,
            owner_hashes_exact=owner_hashes_exact,
        )
        normal_query = int(normal["by_stage"]["QUERY"].get("exact", 0))
        report["semantic_owner"].update(
            {
                "role_slot_swap": _public_score(role_slot_swap),
                "marker_delete": _public_score(marker_delete),
                "entity_rename": {
                    **_public_score(renamed),
                    "assignment_mismatches": sum(
                        left != right
                        for left, right in zip(
                            normal["_predictions"],
                            renamed["_predictions"],
                            strict=True,
                        )
                    ),
                    "max_absolute_logit_difference": float(
                        (normal["_logits"] - renamed["_logits"]).abs().max()
                    ),
                },
                "query_loss_counts": {
                    "role_slot_swap": normal_query
                    - int(role_slot_swap["by_stage"]["QUERY"].get("exact", 0)),
                    "marker_delete": normal_query
                    - int(marker_delete["by_stage"]["QUERY"].get("exact", 0)),
                },
            }
        )
        report["frozen_srp1"] = baseline

    report["status"] = "pass" if all(conditions.values()) else "fail"
    report["promotion_gate"] = {
        "conditions": conditions,
        "passed": all(conditions.values()),
    }
    report.update(
        {
            "data": str(args.data),
            "data_sha256": args.data_sha256,
            "source_commit": args.source_commit,
            "source_checkpoint": str(args.checkpoint),
            "source_checkpoint_sha256": args.checkpoint_sha256,
            "source_model_state_sha256": checkpoint["model_state_sha256"],
            "owner_manifest": model.owner_manifest(),
            "owner_hashes_before": hashes_before,
            "owner_hashes_after": hashes_after,
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
                "split": args.split,
                "status": report["status"],
                "counts": report["natural_query_path"]["counts"],
                "promotion_gate": report["promotion_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
