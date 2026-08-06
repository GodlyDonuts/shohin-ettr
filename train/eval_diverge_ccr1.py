#!/usr/bin/env python3
"""Evaluate the frozen DIVERGE-CCR1 semantic-interface gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_ccr1_data import (
    CCR1_BOARD_ROWS,
    validate_ccr1_board_row,
)
from diverge_ccr1_runtime import (
    CCR1Config,
    CounterfactualReferentMachine,
    MarkerControl,
    validate_owner_contract,
)
from diverge_iem1_runtime import module_state_sha256, tensorize_queries
from diverge_nve1_runtime import hard_role_permutation
from diverge_srp1_data import validate_srp1_board_row
from eval_diverge_iem1 import sha256_path
from eval_diverge_sot1 import _load_json, evaluate as evaluate_sot1_path
from eval_diverge_srp1 import _load_srp1, _score_query_owner


SCHEMA = "shohin-diverge-ccr1-evaluation-v1"
QUERY_MODES = ("sensitive", "invariant", "underdetermined")


class CCR1EvaluationError(RuntimeError):
    """The frozen CCR1 evaluation contract was violated."""


def _load_board(
    path: Path,
    expected_sha256: str,
    *,
    split: str,
) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise CCR1EvaluationError("CCR1 evaluation board hash differs")
    validator = (
        validate_srp1_board_row if split == "development" else validate_ccr1_board_row
    )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validator(row)
            rows.append(row)
    if len(rows) != CCR1_BOARD_ROWS:
        raise CCR1EvaluationError("CCR1 evaluation row count differs")
    return rows


def _load_ccr1(
    path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[CounterfactualReferentMachine, dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise CCR1EvaluationError("CCR1 checkpoint hash differs")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-ccr1-training-report-v1":
        raise CCR1EvaluationError("CCR1 checkpoint schema differs")
    if int(checkpoint.get("update", -1)) != 1000:
        raise CCR1EvaluationError("CCR1 checkpoint duration differs")
    model = CounterfactualReferentMachine(CCR1Config(**checkpoint["config"])).to(
        device
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.freeze_qualified_owners()
    validate_owner_contract(model)
    model.eval()
    if module_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise CCR1EvaluationError("CCR1 model state differs")
    owner_hashes = model.owner_hashes()
    if owner_hashes != checkpoint["final_owner_hashes"]:
        raise CCR1EvaluationError("CCR1 owner hashes differ")
    for owner in ("WORLD", "NUMERIC_EVIDENCE"):
        if owner_hashes[owner] != checkpoint["initial_owner_hashes"][owner]:
            raise CCR1EvaluationError(f"CCR1 immutable owner changed: {owner}")
    if model.owner_manifest() != checkpoint["owner_manifest"]:
        raise CCR1EvaluationError("CCR1 owner manifest differs")
    return model, checkpoint


def _referent_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for episode_index, row in enumerate(rows):
        symbols = [str(value) for value in row["tfs1"]["symbols"]]
        for evidence in row["natural_evidence"]:
            records.append(
                {
                    "episode": episode_index,
                    "stage": "EVIDENCE",
                    "mode": "evidence",
                    "renderer": int(evidence["renderer"]),
                    "source_text": str(evidence["source_text"]),
                    "symbols": symbols,
                    "symbol_role_ids": [
                        int(value) for value in evidence["symbol_role_ids"]
                    ],
                }
            )
        for mode in QUERY_MODES:
            query = row["natural_queries"][mode]
            records.append(
                {
                    "episode": episode_index,
                    "stage": "QUERY",
                    "mode": mode,
                    "renderer": int(query["renderer"]),
                    "source_text": str(query["source_text"]),
                    "symbols": symbols,
                    "symbol_role_ids": [
                        int(value) for value in query["symbol_role_ids"]
                    ],
                }
            )
    return records


def _rename_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    symbols = sorted(
        {str(symbol) for record in records for symbol in record["symbols"]}
    )
    aliases = {symbol: f"referent_{index:03d}" for index, symbol in enumerate(symbols)}
    renamed = []
    for record in records:
        text = str(record["source_text"])
        local_symbols = [str(value) for value in record["symbols"]]
        placeholders = {
            symbol: f"__CCR1_ENTITY_{index:03d}__"
            for index, symbol in enumerate(local_symbols)
        }
        for symbol in sorted(local_symbols, key=len, reverse=True):
            text = text.replace(symbol, placeholders[symbol])
        for symbol in local_symbols:
            text = text.replace(placeholders[symbol], aliases[symbol])
        item = dict(record)
        item["source_text"] = text
        item["symbols"] = [aliases[symbol] for symbol in local_symbols]
        renamed.append(item)
    return renamed


@torch.no_grad()
def _score_referent(
    model: CounterfactualReferentMachine,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    marker_control: MarkerControl = "normal",
) -> dict[str, Any]:
    model.eval()
    overall = Counter()
    by_stage: defaultdict[str, Counter[str]] = defaultdict(Counter)
    query_by_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    query_by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    predictions = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        ids, mask, symbols, targets = tensorize_queries(batch, device)
        logits = model.referent_owner(
            ids,
            mask,
            symbols,
            marker_control=marker_control,
        )
        for index, record in enumerate(batch):
            prediction = hard_role_permutation(logits[index])
            target = tuple(int(value) for value in targets[index].tolist())
            exact = prediction == target
            predictions.append(prediction)
            counters = (overall, by_stage[str(record["stage"])])
            for counter in counters:
                counter["total"] += 1
                counter["exact"] += exact
            if str(record["stage"]) == "QUERY":
                mode = str(record["mode"])
                renderer = str(int(record["renderer"]))
                for counter in (
                    query_by_mode[mode],
                    query_by_renderer[renderer],
                ):
                    counter["total"] += 1
                    counter["exact"] += exact
    encoded_predictions = json.dumps(
        predictions, separators=(",", ":")
    ).encode("ascii")
    return {
        "overall": dict(overall),
        "by_stage": {key: dict(value) for key, value in sorted(by_stage.items())},
        "query_by_mode": {
            key: dict(value) for key, value in sorted(query_by_mode.items())
        },
        "query_by_renderer": {
            key: dict(value) for key, value in sorted(query_by_renderer.items())
        },
        "prediction_sha256": hashlib.sha256(encoded_predictions).hexdigest(),
        "_predictions": predictions,
    }


def _public_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in score.items() if key != "_predictions"}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _development_conditions(report: Mapping[str, Any]) -> dict[str, bool]:
    counts = report["natural_query_path"]["counts"]
    return {
        "query_at_least_765": int(counts.get("query_exact", 0)) >= 765,
        "evidence_at_least_3070": int(counts.get("evidence_exact", 0)) >= 3070,
        "sealed_episodes_at_least_255": int(
            counts.get("episodes_fully_sealed", 0)
        )
        >= 255,
    }


def _fresh_conditions(
    report: Mapping[str, Any],
    baseline: Mapping[str, Any],
    normal: Mapping[str, Any],
    marker_swap: Mapping[str, Any],
    marker_delete: Mapping[str, Any],
    renamed: Mapping[str, Any],
    *,
    owner_hashes_exact: bool,
) -> dict[str, bool]:
    counts = report["natural_query_path"]["counts"]
    modes = normal["query_by_mode"]
    renderers = normal["query_by_renderer"]
    query_exact = int(normal["by_stage"]["QUERY"].get("exact", 0))
    query_total = int(normal["by_stage"]["QUERY"].get("total", 0))
    swap_exact = int(marker_swap["by_stage"]["QUERY"].get("exact", 0))
    delete_exact = int(marker_delete["by_stage"]["QUERY"].get("exact", 0))
    nve_counts = report["fresh_nve1"]["counts"]
    baseline_counts = baseline["natural_query_path"]["counts"]
    control_drop = math.ceil(0.50 * CCR1_BOARD_ROWS)
    invalid_receipts = sum(
        int(nve_counts.get(key, 0))
        for key in (
            "invalid_distractor_accepted",
            "invalid_evidence_accepted",
            "invalid_packet_accepted",
            "invalid_receipts_accepted",
            "invalid_source_accepted",
            "invalid_step_accepted",
            "invalid_target_accepted",
            "invalid_value_accepted",
        )
    )
    return {
        "world_256": int(counts.get("source_program_exact", 0)) == 256,
        "evidence_at_least_3070": int(counts.get("evidence_exact", 0)) >= 3070,
        "sealed_episodes_at_least_255": int(
            counts.get("episodes_fully_sealed", 0)
        )
        >= 255,
        "query_at_least_765": query_total == 768 and query_exact >= 765,
        "every_mode_at_least_254": len(modes) == 3
        and all(
            int(value.get("total", 0)) == 256 and int(value.get("exact", 0)) >= 254
            for value in modes.values()
        ),
        "every_renderer_at_least_127": len(renderers) == 6
        and all(
            int(value.get("total", 0)) == 128 and int(value.get("exact", 0)) >= 127
            for value in renderers.values()
        ),
        "sensitive_answers_at_least_254": int(counts.get("sensitive_exact", 0))
        >= 254,
        "extensional_parity_at_least_254": int(counts.get("extensional_parity", 0))
        >= 254,
        "no_evidence_abstains_at_least_254": int(
            counts.get("no_evidence_abstain", 0)
        )
        >= 254,
        "invariant_answers_at_least_254": int(counts.get("invariant_exact", 0))
        >= 254,
        "partial_underdetermined_abstains_at_least_254": int(
            counts.get("partial_underdetermined_abstain", 0)
        )
        >= 254,
        "beats_frozen_srp1_by_4_queries": query_exact
        - int(baseline["direct_query_exact"])
        >= 4,
        "evidence_no_worse_than_frozen_srp1": int(counts.get("evidence_exact", 0))
        >= int(baseline_counts.get("evidence_exact", 0)),
        "marker_swap_drop_at_least_90_points": query_total == 768
        and query_exact - swap_exact >= math.ceil(0.90 * query_total),
        "marker_deletion_drop_at_least_49_points": query_total == 768
        and query_exact - delete_exact >= math.ceil(0.49 * query_total),
        "entity_rename_all_assignments_invariant": normal["_predictions"]
        == renamed["_predictions"],
        "shuffled_evidence_drop_at_least_50_points": int(
            nve_counts.get("learned_exact", 0)
        )
        - int(nve_counts.get("shuffled_exact", 0))
        >= control_drop,
        "state_reset_drop_at_least_50_points": int(
            nve_counts.get("learned_exact", 0)
        )
        - int(nve_counts.get("state_reset_exact", 0))
        >= control_drop,
        "operation_shift_drop_at_least_50_points": int(
            nve_counts.get("learned_exact", 0)
        )
        - int(nve_counts.get("operation_shift_exact", 0))
        >= control_drop,
        "packet_query_swaps_all_reject": int(
            counts.get("packet_query_swap_reject", 0)
        )
        == 256,
        "post_seal_poison_invariant": int(
            counts.get("post_seal_poison_invariant", 0)
        )
        == 256,
        "zero_invalid_transactions": invalid_receipts == 0
        and int(counts.get("invalid_queries_accepted", 0)) == 0,
        "zero_false_commitments": int(nve_counts.get("false_commitment", 0)) == 0,
        "zero_malformed_packets": int(nve_counts.get("malformed_accepted", 0)) == 0,
        "zero_gold_deletions": int(nve_counts.get("learned_gold_preserved", 0))
        == 256,
        "zero_overflow": int(nve_counts.get("overflow", 0)) == 0,
        "protected_owner_hashes_exact": owner_hashes_exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
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
        raise SystemExit(f"refusing existing CCR1 result: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CCR1 requested unavailable CUDA")
    if args.split == "confirmation" and (
        args.srp1_checkpoint is None or args.srp1_checkpoint_sha256 is None
    ):
        raise SystemExit("CCR1 confirmation requires the frozen SRP1 checkpoint")
    device = torch.device(args.device)

    rows = _load_board(args.data, args.data_sha256, split=args.split)
    model, checkpoint = _load_ccr1(
        args.checkpoint, args.checkpoint_sha256, device
    )
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
        model, records, device=device, batch_size=args.batch_size
    )
    owner_hashes_exact = model.owner_hashes() == checkpoint["final_owner_hashes"]
    report["schema"] = SCHEMA
    report["split"] = args.split
    report["semantic_owner"] = {"normal": _public_score(normal)}

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
        marker_swap = _score_referent(
            model,
            records,
            device=device,
            batch_size=args.batch_size,
            marker_control="swap",
        )
        marker_delete = _score_referent(
            model,
            records,
            device=device,
            batch_size=args.batch_size,
            marker_control="delete",
        )
        renamed = _score_referent(
            model,
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
            marker_swap,
            marker_delete,
            renamed,
            owner_hashes_exact=owner_hashes_exact,
        )
        normal_query = int(normal["by_stage"]["QUERY"].get("exact", 0))
        report["semantic_owner"].update(
            {
                "marker_swap": _public_score(marker_swap),
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
                },
                "query_drop_points": {
                    "marker_swap": 100.0
                    * (
                        normal_query
                        - int(marker_swap["by_stage"]["QUERY"].get("exact", 0))
                    )
                    / 768,
                    "marker_delete": 100.0
                    * (
                        normal_query
                        - int(marker_delete["by_stage"]["QUERY"].get("exact", 0))
                    )
                    / 768,
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
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
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
                "split": args.split,
                "status": report["status"],
                "counts": report["natural_query_path"]["counts"],
                "semantic_owner": report["semantic_owner"],
                "promotion_gate": report["promotion_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
