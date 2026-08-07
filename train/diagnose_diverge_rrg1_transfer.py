#!/usr/bin/env python3
"""One read-only attribution of the closed DIVERGE-RRG1 transfer miss."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_iem1_data import _symbol_role_ids
from diverge_iem1_runtime import tensorize_queries
from diverge_nve1_data import _render_confirmation, _render_training
from diverge_rrg1_runtime import permutation_scores, permutation_targets
from eval_diverge_ccr1 import _referent_records, _rename_records
from eval_diverge_rrg1 import _load_board, _load_rrg1, _score_referent
from eval_diverge_iem1 import sha256_path


SCHEMA = "shohin-diverge-rrg1-transfer-diagnostic-v1"


@torch.no_grad()
def _score_records(
    model,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    by_surface: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_role_order: defaultdict[str, Counter[str]] = defaultdict(Counter)
    overall = Counter()
    prediction_digest = hashlib.sha256()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        ids, mask, symbols, targets = tensorize_queries(batch, device)
        logits = model.referent_owner(ids, mask, symbols)
        scores = permutation_scores(logits)
        expected = permutation_targets(targets)
        predicted = scores.argmax(dim=-1)
        margins = (
            scores.gather(1, expected[:, None]).squeeze(1)
            - scores.gather(1, (1 - expected)[:, None]).squeeze(1)
        )
        for index, record in enumerate(batch):
            exact = bool(predicted[index].eq(expected[index]))
            margin = float(margins[index])
            order = str(int(targets[index, 0]))
            surface = str(record["surface"])
            for counter in (overall, by_surface[surface], by_role_order[order]):
                counter["total"] += 1
                counter["exact"] += exact
                counter["signed_margin_milli"] += round(1000.0 * margin)
        prediction_digest.update(predicted.detach().cpu().numpy().tobytes())

    def public(counter: Counter[str]) -> dict[str, Any]:
        total = int(counter["total"])
        return {
            "total": total,
            "exact": int(counter["exact"]),
            "exact_rate": counter["exact"] / total,
            "mean_signed_margin": counter["signed_margin_milli"] / (1000.0 * total),
        }

    return {
        "overall": public(overall),
        "by_surface": {
            key: public(value) for key, value in sorted(by_surface.items())
        },
        "by_role_order": {
            key: public(value) for key, value in sorted(by_role_order.items())
        },
        "prediction_sha256": prediction_digest.hexdigest(),
    }


def _established_evidence_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for episode, row in enumerate(rows):
        symbols = [str(value) for value in row["tfs1"]["symbols"]]
        for evidence in row["natural_evidence"]:
            fields = {
                "step": int(evidence["step_ordinal"]),
                "value": str(evidence["value"]),
                "target": str(evidence["target"]),
                "distractor": str(evidence["distractor"]),
            }
            surfaces = {
                **{
                    f"nve1_train_{renderer}": _render_training(renderer, **fields)
                    for renderer in range(6)
                },
                **{
                    f"nve1_confirmation_{renderer}": _render_confirmation(
                        renderer, **fields
                    )
                    for renderer in range(3)
                },
            }
            for surface, text in surfaces.items():
                records.append(
                    {
                        "episode": episode,
                        "surface": surface,
                        "source_text": text,
                        "symbols": symbols,
                        "symbol_role_ids": _symbol_role_ids(
                            text,
                            symbols,
                            target=fields["target"],
                            distractor=fields["distractor"],
                        ),
                    }
                )
    return records


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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing RRG1 diagnostic: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("RRG1 diagnostic requested unavailable CUDA")
    device = torch.device(args.device)
    rows = _load_board(args.data, args.data_sha256, split="development")
    model, checkpoint = _load_rrg1(args.checkpoint, args.checkpoint_sha256, device)
    initial_hashes = model.owner_hashes()

    original = _referent_records(rows)
    original_evidence = [
        {**record, "surface": f'evidence_renderer_{int(record["renderer"])}'}
        for record in original
        if record["stage"] == "EVIDENCE"
    ]
    original_queries = [
        {**record, "surface": f'query_renderer_{int(record["renderer"])}'}
        for record in original
        if record["stage"] == "QUERY"
    ]
    original_score = _score_records(
        model,
        (*original_evidence, *original_queries),
        device=device,
        batch_size=args.batch_size,
    )
    established_matrix = _score_records(
        model,
        _established_evidence_matrix(rows),
        device=device,
        batch_size=args.batch_size,
    )
    normal = _score_referent(
        model, original, device=device, batch_size=args.batch_size
    )
    renamed = _score_referent(
        model,
        _rename_records(original),
        device=device,
        batch_size=args.batch_size,
    )
    final_hashes = model.owner_hashes()
    report = {
        "schema": SCHEMA,
        "status": "read_only_attribution",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "model_state_sha256": checkpoint["model_state_sha256"],
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "original_surface_transfer": original_score,
        "established_evidence_surface_matrix": established_matrix,
        "entity_rename": {
            "assignment_mismatches": sum(
                left != right
                for left, right in zip(
                    normal["_predictions"], renamed["_predictions"], strict=True
                )
            ),
            "logits_bit_exact": torch.equal(normal["_logits"], renamed["_logits"]),
            "max_absolute_logit_difference": float(
                (normal["_logits"] - renamed["_logits"]).abs().max()
            ),
        },
        "owner_hashes_before": initial_hashes,
        "owner_hashes_after": final_hashes,
        "owner_hashes_exact": initial_hashes == final_hashes,
        "confirmation_opened": False,
        "model_or_threshold_changed": False,
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "original_surface_transfer": original_score,
                "established_evidence_surface_matrix": established_matrix,
                "entity_rename": report["entity_rename"],
                "owner_hashes_exact": report["owner_hashes_exact"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
