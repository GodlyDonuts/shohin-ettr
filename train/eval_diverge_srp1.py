#!/usr/bin/env python3
"""Evaluate the one frozen DIVERGE-SRP1 semantic-primitive gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_iem1_runtime import module_state_sha256, tensorize_queries
from diverge_nve1_runtime import hard_role_permutation
from diverge_sot1_runtime import StageOwnedEpistemicMachine
from diverge_srp1_data import (
    SRP1_BOARD_ROWS,
    validate_srp1_board_row,
)
from diverge_srp1_runtime import (
    SRP1Config,
    SemanticPrimitiveEpistemicMachine,
    validate_owner_contract,
)
from eval_diverge_iem1 import sha256_path
from eval_diverge_sot1 import (
    _load_json,
    _load_sot1,
    evaluate as evaluate_sot1_path,
)


SCHEMA = "shohin-diverge-srp1-evaluation-v1"
QUERY_MODES = ("sensitive", "invariant", "underdetermined")


class SRP1EvaluationError(RuntimeError):
    """The frozen SRP1 evaluation contract was violated."""


def _load_board(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SRP1EvaluationError("SRP1 confirmation board hash differs")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_srp1_board_row(row)
            rows.append(row)
    if len(rows) != SRP1_BOARD_ROWS:
        raise SRP1EvaluationError("SRP1 confirmation row count differs")
    return rows


def _load_srp1(
    path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[SemanticPrimitiveEpistemicMachine, dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SRP1EvaluationError("SRP1 checkpoint hash differs")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-srp1-training-report-v1":
        raise SRP1EvaluationError("SRP1 checkpoint schema differs")
    if int(checkpoint.get("update", -1)) != 1000:
        raise SRP1EvaluationError("SRP1 checkpoint duration differs")
    model = SemanticPrimitiveEpistemicMachine(SRP1Config(**checkpoint["config"])).to(
        device
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.freeze_qualified_owners()
    validate_owner_contract(model)
    model.eval()
    if module_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise SRP1EvaluationError("SRP1 model state differs")
    owner_hashes = model.owner_hashes()
    if owner_hashes != checkpoint["final_owner_hashes"]:
        raise SRP1EvaluationError("SRP1 owner hashes differ")
    for owner in ("WORLD", "NUMERIC_EVIDENCE"):
        if owner_hashes[owner] != checkpoint["initial_owner_hashes"][owner]:
            raise SRP1EvaluationError(f"SRP1 immutable owner changed: {owner}")
    if model.owner_manifest() != checkpoint["owner_manifest"]:
        raise SRP1EvaluationError("SRP1 owner manifest differs")
    return model, checkpoint


def _query_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        symbols = list(row["tfs1"]["symbols"])
        for mode in QUERY_MODES:
            item = row["natural_queries"][mode]
            output.append(
                {
                    "mode": mode,
                    "renderer": int(item["renderer"]),
                    "source_text": str(item["source_text"]),
                    "symbols": symbols,
                    "symbol_role_ids": [
                        int(value) for value in item["symbol_role_ids"]
                    ],
                }
            )
    return output


@torch.no_grad()
def _score_query_owner(
    model: StageOwnedEpistemicMachine | SemanticPrimitiveEpistemicMachine,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    records = _query_records(rows)
    model.eval()
    overall = Counter()
    by_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    cursor = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        ids, mask, symbols, targets = tensorize_queries(batch, device)
        logits = model.forward_query(ids, mask, symbols)
        for index, record in enumerate(batch):
            exact = hard_role_permutation(logits[index]) == tuple(
                int(value) for value in targets[index].tolist()
            )
            mode = str(record["mode"])
            renderer = str(int(record["renderer"]))
            for counter in (overall, by_mode[mode], by_renderer[renderer]):
                counter["total"] += 1
                counter["exact"] += exact
            cursor += 1
    if cursor != len(records):
        raise SRP1EvaluationError("SRP1 query accounting differs")
    return {
        "overall": dict(overall),
        "by_mode": {key: dict(value) for key, value in sorted(by_mode.items())},
        "by_renderer": {
            key: dict(value) for key, value in sorted(by_renderer.items())
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--sot1-checkpoint", type=Path, required=True)
    parser.add_argument("--sot1-checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--protected-nve1-result", type=Path, required=True)
    parser.add_argument("--protected-nve1-result-sha256", required=True)
    parser.add_argument("--protected-tol3-result", type=Path, required=True)
    parser.add_argument("--protected-tol3-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing SRP1 result: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("SRP1 requested unavailable CUDA")
    device = torch.device(args.device)

    rows = _load_board(args.data, args.data_sha256)
    model, checkpoint = _load_srp1(args.checkpoint, args.checkpoint_sha256, device)
    sot1, _ = _load_sot1(
        args.sot1_checkpoint,
        args.sot1_checkpoint_sha256,
        device,
    )
    protected_nve1 = _load_json(
        args.protected_nve1_result, args.protected_nve1_result_sha256
    )
    protected_nve1["_sha256"] = args.protected_nve1_result_sha256
    protected_tol3 = _load_json(
        args.protected_tol3_result, args.protected_tol3_result_sha256
    )
    protected_tol3["_sha256"] = args.protected_tol3_result_sha256

    srp1_queries = _score_query_owner(
        model, rows, device=device, batch_size=args.batch_size
    )
    sot1_queries = _score_query_owner(
        sot1, rows, device=device, batch_size=args.batch_size
    )
    report = evaluate_sot1_path(
        rows,
        model,  # type: ignore[arg-type]
        protected_nve1=protected_nve1,
        protected_tol3=protected_tol3,
        device=device,
        batch_size=args.batch_size,
    )
    conditions = report["promotion_gate"]["conditions"]
    conditions.pop("owner_manifest_isolated", None)
    conditions["owner_manifest_semantic_sharing_exact"] = (
        model.owner_hashes()["QUERY"] == model.owner_hashes()["REFERENT"]
        and model.owner_manifest()["semantic_sharing"]
        == {"REFERENT": ["EVIDENCE.TARGET_DISTRACTOR", "QUERY.TARGET_DISTRACTOR"]}
    )
    conditions["every_renderer_at_least_122"] = all(
        int(values.get("total", 0)) == 128 and int(values.get("exact", 0)) >= 122
        for values in srp1_queries["by_renderer"].values()
    ) and len(srp1_queries["by_renderer"]) == 6
    srp1_exact = int(srp1_queries["overall"].get("exact", 0))
    sot1_exact = int(sot1_queries["overall"].get("exact", 0))
    conditions["beats_frozen_sot1_by_77_queries"] = srp1_exact - sot1_exact >= 77
    report["schema"] = SCHEMA
    report["status"] = "pass" if all(conditions.values()) else "fail"
    report["promotion_gate"]["passed"] = all(conditions.values())
    report["query_owner_comparison"] = {
        "srp1": srp1_queries,
        "frozen_sot1": sot1_queries,
        "exact_delta": srp1_exact - sot1_exact,
    }
    report.update(
        {
            "data": str(args.data),
            "data_sha256": args.data_sha256,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "sot1_checkpoint": str(args.sot1_checkpoint),
            "sot1_checkpoint_sha256": args.sot1_checkpoint_sha256,
            "model_state_sha256": checkpoint["model_state_sha256"],
            "owner_manifest": checkpoint["owner_manifest"],
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
                "query_owner_comparison": report["query_owner_comparison"],
                "promotion_gate": report["promotion_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

