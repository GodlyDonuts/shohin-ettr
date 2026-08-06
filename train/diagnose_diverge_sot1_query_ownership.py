#!/usr/bin/env python3
"""Read-only attribution for the failed DIVERGE-SOT1 QUERY owner."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from diverge_iem1_data import (
    QUERY_TRAIN_ROWS,
    _symbol_role_ids,
    validate_query_training_record,
)
from diverge_iem1_runtime import tensorize_queries
from diverge_nve1_runtime import hard_role_permutation
from diverge_sot1_data import (
    SOT1_BOARD_ROWS,
    query_confirmation_text,
    validate_sot1_board_row,
)
from eval_diverge_sot1 import _load_sot1
from eval_diverge_iem1 import sha256_path


SCHEMA = "shohin-diverge-sot1-query-ownership-diagnostic-v1"
QUERY_MODES = ("sensitive", "invariant", "underdetermined")
QUERY_RENDERERS = tuple(range(3))


class SOT1DiagnosticError(RuntimeError):
    """The read-only SOT1 attribution contract was violated."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_jsonl(
    path: Path,
    expected_sha256: str,
    validator: Callable[[Mapping[str, Any]], None],
    expected_rows: int,
) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SOT1DiagnosticError(f"protected input hash differs: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validator(row)
            rows.append(row)
    if len(rows) != expected_rows:
        raise SOT1DiagnosticError(f"protected input row count differs: {path}")
    return rows


def _assignment_class(
    predicted: Sequence[int], gold: Sequence[int]
) -> str:
    predicted_tuple = tuple(int(value) for value in predicted)
    gold_tuple = tuple(int(value) for value in gold)
    if predicted_tuple == gold_tuple:
        return "exact"
    if predicted_tuple == tuple(1 - value for value in gold_tuple):
        return "complete_swap"
    return "other"


def _counter_record(counter: Counter[str]) -> dict[str, int]:
    return {
        key: int(counter.get(key, 0))
        for key in ("total", "exact", "complete_swap", "other")
    }


def _audit_training_labels(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    assignments: Counter[str] = Counter()
    for row in rows:
        renderer = str(int(row["renderer"]))
        assignment = tuple(int(value) for value in row["symbol_role_ids"])
        key = json.dumps(assignment, separators=(",", ":"))
        by_renderer[renderer]["total"] += 1
        by_renderer[renderer]["target_first"] += assignment == (0, 1)
        by_renderer[renderer]["distractor_first"] += assignment == (1, 0)
        assignments[key] += 1
    return {
        "rows": len(rows),
        "assignment_counts": dict(sorted(assignments.items())),
        "by_renderer": {
            key: {name: int(value) for name, value in sorted(counter.items())}
            for key, counter in sorted(by_renderer.items())
        },
    }


def _board_query_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        symbols = tuple(str(value) for value in row["tfs1"]["symbols"])
        episode_id = str(row["tfs1"]["id"])
        for mode in QUERY_MODES:
            item = row["natural_queries"][mode]
            records.append(
                {
                    "episode_id": episode_id,
                    "mode": mode,
                    "renderer": int(item["renderer"]),
                    "source_text": str(item["source_text"]),
                    "symbols": list(symbols),
                    "target": str(item["target"]),
                    "distractor": str(item["distractor"]),
                    "symbol_role_ids": [
                        int(value) for value in item["symbol_role_ids"]
                    ],
                }
            )
    return records


def _cross_renderer_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        symbols = tuple(str(value) for value in record["symbols"])
        target = str(record["target"])
        distractor = str(record["distractor"])
        for renderer in QUERY_RENDERERS:
            text = query_confirmation_text(
                renderer,
                target=target,
                distractor=distractor,
            )
            output.append(
                {
                    **record,
                    "renderer": renderer,
                    "source_text": text,
                    "symbol_role_ids": _symbol_role_ids(
                        text,
                        symbols,
                        target=target,
                        distractor=distractor,
                    ),
                }
            )
    return output


@torch.no_grad()
def _predict_assignments(
    model,
    records: Sequence[Mapping[str, Any]],
    *,
    owner: str,
    device: torch.device,
    batch_size: int,
) -> list[tuple[int, int]]:
    output = []
    model.eval()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        ids, attention, symbol_masks, _ = tensorize_queries(batch, device)
        if owner == "QUERY":
            logits = model.forward_query(ids, attention, symbol_masks)
        elif owner == "EVIDENCE":
            numeric_bounds = torch.tensor(
                [[[1, 2], [1, 2]]] * len(batch),
                dtype=torch.long,
                device=device,
            )
            _, logits = model.evidence_owner(
                ids,
                attention,
                numeric_bounds,
                symbol_masks,
            )
        else:
            raise SOT1DiagnosticError(f"unknown owner: {owner}")
        output.extend(hard_role_permutation(row) for row in logits)
    return output


def _score_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise SOT1DiagnosticError("prediction accounting differs")
    overall: Counter[str] = Counter()
    by_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_mode_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    confusion: Counter[str] = Counter()
    failures = []
    for record, prediction in zip(records, predictions, strict=True):
        gold = tuple(int(value) for value in record["symbol_role_ids"])
        label = _assignment_class(prediction, gold)
        mode = str(record["mode"])
        renderer = str(int(record["renderer"]))
        key = f"{mode}/r{renderer}"
        for counter in (overall, by_mode[mode], by_renderer[renderer], by_mode_renderer[key]):
            counter["total"] += 1
            counter[label] += 1
        confusion[
            f"gold={tuple(gold)} predicted={tuple(int(value) for value in prediction)}"
        ] += 1
        if label != "exact" and len(failures) < 24:
            failures.append(
                {
                    "episode_id": str(record["episode_id"]),
                    "mode": mode,
                    "renderer": int(record["renderer"]),
                    "source_text": str(record["source_text"]),
                    "gold": list(gold),
                    "predicted": [int(value) for value in prediction],
                    "classification": label,
                }
            )
    return {
        "overall": _counter_record(overall),
        "by_mode": {
            key: _counter_record(value) for key, value in sorted(by_mode.items())
        },
        "by_renderer": {
            key: _counter_record(value)
            for key, value in sorted(by_renderer.items())
        },
        "by_mode_renderer": {
            key: _counter_record(value)
            for key, value in sorted(by_mode_renderer.items())
        },
        "confusion": dict(sorted(confusion.items())),
        "failure_samples": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--board-sha256", required=True)
    parser.add_argument("--query-data", type=Path, required=True)
    parser.add_argument("--query-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.exists():
        raise SystemExit(f"refusing existing SOT1 diagnostic output: {args.output.parent}")
    if args.batch_size <= 0 or args.threads <= 0:
        raise SystemExit("SOT1 diagnostic runtime geometry differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("SOT1 diagnostic requested unavailable CUDA")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    board_rows = _load_jsonl(
        args.board,
        args.board_sha256,
        validate_sot1_board_row,
        SOT1_BOARD_ROWS,
    )
    training_rows = _load_jsonl(
        args.query_data,
        args.query_data_sha256,
        validate_query_training_record,
        QUERY_TRAIN_ROWS,
    )
    model, checkpoint = _load_sot1(
        args.checkpoint,
        args.checkpoint_sha256,
        device,
    )
    owner_hashes_before = model.owner_hashes()
    board_records = _board_query_records(board_rows)
    cross_records = _cross_renderer_records(board_records)

    query_board_predictions = _predict_assignments(
        model,
        board_records,
        owner="QUERY",
        device=device,
        batch_size=args.batch_size,
    )
    evidence_board_predictions = _predict_assignments(
        model,
        board_records,
        owner="EVIDENCE",
        device=device,
        batch_size=args.batch_size,
    )
    query_cross_predictions = _predict_assignments(
        model,
        cross_records,
        owner="QUERY",
        device=device,
        batch_size=args.batch_size,
    )
    evidence_cross_predictions = _predict_assignments(
        model,
        cross_records,
        owner="EVIDENCE",
        device=device,
        batch_size=args.batch_size,
    )

    query_board = _score_predictions(board_records, query_board_predictions)
    evidence_board = _score_predictions(board_records, evidence_board_predictions)
    query_cross = _score_predictions(cross_records, query_cross_predictions)
    evidence_cross = _score_predictions(cross_records, evidence_cross_predictions)
    renderer_to_modes = defaultdict(set)
    for record in board_records:
        renderer_to_modes[int(record["renderer"])].add(str(record["mode"]))
    mode_renderer_confounded = all(
        len(modes) == 1 for modes in renderer_to_modes.values()
    )
    renderer_zero = query_cross["by_renderer"]["0"]
    other_renderers = [query_cross["by_renderer"][str(index)] for index in (1, 2)]
    renderer_specific_inversion = (
        renderer_zero["complete_swap"] == renderer_zero["total"]
        and all(row["exact"] / row["total"] >= 0.9 for row in other_renderers)
    )
    evidence_exact_rate = (
        evidence_cross["overall"]["exact"] / evidence_cross["overall"]["total"]
    )
    owner_hashes_after = model.owner_hashes()
    if owner_hashes_before != owner_hashes_after:
        raise SOT1DiagnosticError("read-only diagnostic changed an owner")

    payload = {
        "schema": SCHEMA,
        "status": "complete_read_only_attribution",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "board": str(args.board),
        "board_sha256": args.board_sha256,
        "query_data": str(args.query_data),
        "query_data_sha256": args.query_data_sha256,
        "device": str(device),
        "owner_hashes_before": owner_hashes_before,
        "owner_hashes_after": owner_hashes_after,
        "checkpoint_training_query_evaluation": checkpoint.get("query_evaluation"),
        "training_label_audit": _audit_training_labels(training_rows),
        "board_query_owner": query_board,
        "board_evidence_owner_probe": evidence_board,
        "cross_renderer_query_owner": query_cross,
        "cross_renderer_evidence_owner_probe": evidence_cross,
        "attribution": {
            "mode_renderer_confounded_on_sealed_board": mode_renderer_confounded,
            "renderer_to_modes": {
                str(key): sorted(value) for key, value in sorted(renderer_to_modes.items())
            },
            "query_owner_renderer_specific_complete_inversion": renderer_specific_inversion,
            "evidence_owner_cross_renderer_exact_rate": evidence_exact_rate,
            "isolated_query_owner_family_closed": True,
            "result_is_training_or_promotion_claim": False,
            "successor_implication": (
                "share_the_qualified_referent_primitive_across_evidence_and_query"
                if evidence_exact_rate >= 0.95
                else "replace_stage_local_query_classification_with_a_structurally_different_semantic_primitive"
            ),
        },
    }
    _atomic_json(args.output, payload)
    output_sha256 = sha256_path(args.output)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": output_sha256,
                "attribution": payload["attribution"],
                "query_board": query_board["overall"],
                "evidence_board": evidence_board["overall"],
                "query_cross_renderer": query_cross["by_renderer"],
                "evidence_cross_renderer": evidence_cross["by_renderer"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
