#!/usr/bin/env python3
"""One read-only renderer attribution after the frozen CCR1 development miss."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_iem1_data import _evidence_confirmation_text, _symbol_role_ids
from diverge_iem1_runtime import tensorize_queries
from diverge_nve1_runtime import hard_role_permutation
from diverge_srp1_data import query_text
from eval_diverge_ccr1 import (
    _load_board,
    _load_ccr1,
    _public_score,
    _referent_records,
    _rename_records,
    _score_referent,
)
from eval_diverge_iem1 import sha256_path


SCHEMA = "shohin-diverge-ccr1-read-only-attribution-v1"
QUERY_MODES = ("sensitive", "invariant", "underdetermined")


def _counterfactual_records(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for episode_index, row in enumerate(rows):
        symbols = [str(value) for value in row["tfs1"]["symbols"]]
        for evidence in row["natural_evidence"]:
            target = str(evidence["target"])
            distractor = str(evidence["distractor"])
            for renderer in range(3):
                text = _evidence_confirmation_text(
                    renderer,
                    step=int(evidence["step_ordinal"]),
                    value=str(evidence["value"]),
                    target=target,
                    distractor=distractor,
                )
                records.append(
                    {
                        "episode": episode_index,
                        "stage": "EVIDENCE",
                        "mode": "evidence",
                        "renderer": renderer,
                        "source_text": text,
                        "symbols": symbols,
                        "symbol_role_ids": _symbol_role_ids(
                            text,
                            symbols,
                            target=target,
                            distractor=distractor,
                        ),
                    }
                )
        for mode in QUERY_MODES:
            query = row["natural_queries"][mode]
            target = str(query["target"])
            distractor = str(query["distractor"])
            for renderer in range(6):
                text = query_text(
                    renderer,
                    target=target,
                    distractor=distractor,
                )
                records.append(
                    {
                        "episode": episode_index,
                        "stage": "QUERY",
                        "mode": mode,
                        "renderer": renderer,
                        "source_text": text,
                        "symbols": symbols,
                        "symbol_role_ids": _symbol_role_ids(
                            text,
                            symbols,
                            target=target,
                            distractor=distractor,
                        ),
                    }
                )
    return records


@torch.no_grad()
def _score_detailed(
    model,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    counters: defaultdict[str, Counter[str]] = defaultdict(Counter)
    margins: defaultdict[str, list[float]] = defaultdict(list)
    model.eval()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        ids, mask, symbols, targets = tensorize_queries(batch, device)
        logits = model.referent_owner(ids, mask, symbols)
        for index, record in enumerate(batch):
            target = tuple(int(value) for value in targets[index].tolist())
            prediction = hard_role_permutation(logits[index])
            exact = prediction == target
            alternate = (1 - target[0], 1 - target[1])
            correct_score = sum(
                float(logits[index, mention, role])
                for mention, role in enumerate(target)
            )
            alternate_score = sum(
                float(logits[index, mention, role])
                for mention, role in enumerate(alternate)
            )
            keys = (
                "overall",
                str(record["stage"]),
                f"{record['stage']}:renderer:{int(record['renderer'])}",
                f"{record['stage']}:role_order:{target[0]}{target[1]}",
            )
            if str(record["stage"]) == "QUERY":
                keys = (*keys, f"QUERY:mode:{record['mode']}")
            for key in keys:
                counters[key]["total"] += 1
                counters[key]["exact"] += exact
                margins[key].append(correct_score - alternate_score)
    return {
        "counts": {key: dict(value) for key, value in sorted(counters.items())},
        "signed_margin": {
            key: {
                "mean": sum(values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for key, values in sorted(margins.items())
        },
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing CCR1 diagnostic: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CCR1 diagnostic requested unavailable CUDA")
    device = torch.device(args.device)
    rows = _load_board(args.data, args.data_sha256, split="development")
    model, checkpoint = _load_ccr1(
        args.checkpoint, args.checkpoint_sha256, device
    )
    original = _referent_records(rows)
    normal = _score_referent(
        model, original, device=device, batch_size=args.batch_size
    )
    marker_swap = _score_referent(
        model,
        original,
        device=device,
        batch_size=args.batch_size,
        marker_control="swap",
    )
    marker_delete = _score_referent(
        model,
        original,
        device=device,
        batch_size=args.batch_size,
        marker_control="delete",
    )
    renamed = _score_referent(
        model,
        _rename_records(original),
        device=device,
        batch_size=args.batch_size,
    )
    report = {
        "schema": SCHEMA,
        "status": "diagnostic_only",
        "decision": "close_ccr1_without_variants",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "model_state_sha256": checkpoint["model_state_sha256"],
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "owner_hashes": model.owner_hashes(),
        "original": _public_score(normal),
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
        "complete_renderer_matrix": _score_detailed(
            model,
            _counterfactual_records(rows),
            device=device,
            batch_size=args.batch_size,
        ),
        "claim_boundary": (
            "Read-only attribution on the opened SRP1 development board. "
            "It cannot promote CCR1 or open sealed confirmation."
        ),
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "decision": report["decision"],
                "original": report["original"],
                "entity_rename_mismatches": report["entity_rename"][
                    "assignment_mismatches"
                ],
                "renderer_counts": report["complete_renderer_matrix"]["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
