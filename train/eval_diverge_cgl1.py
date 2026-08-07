#!/usr/bin/env python3
"""Evaluate one frozen DIVERGE-CGL1 outcome-grounded interpreter."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tokenizers import Tokenizer

from diverge_cgl1_runtime import (
    CGL1Config,
    CausalGroundingInterpreter,
    adapter_state_sha256,
    frozen_backbone_state_sha256,
    load_adapter_state,
    render_claim_prompt,
)
from diverge_gti1_runtime import expected_transaction
from eval_diverge_ccr1 import _referent_records, _rename_records
from eval_diverge_pqi1 import _load_board, sha256_path
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-cgl1-evaluation-v1"


class CGL1EvaluationError(RuntimeError):
    """A CGL1 checkpoint, board, or evaluation receipt differs."""


def _load_model(
    checkpoint_path: Path,
    checkpoint_sha256: str,
    base_path: Path,
    base_sha256: str,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    device: torch.device,
) -> tuple[CausalGroundingInterpreter, dict[str, Any]]:
    for path, expected, label in (
        (checkpoint_path, checkpoint_sha256, "checkpoint"),
        (base_path, base_sha256, "base"),
        (tokenizer_path, tokenizer_sha256, "tokenizer"),
    ):
        if sha256_path(path) != expected:
            raise CGL1EvaluationError(f"CGL1 {label} hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-cgl1-training-report-v1":
        raise CGL1EvaluationError("CGL1 checkpoint schema differs")
    if (
        checkpoint.get("base_sha256") != base_sha256
        or checkpoint.get("tokenizer_sha256") != tokenizer_sha256
    ):
        raise CGL1EvaluationError("CGL1 parent receipt differs")
    backbone, _, _ = load_frozen_pointer_backbone(base_path, device=device)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = CausalGroundingInterpreter(
        backbone, tokenizer, CGL1Config(**checkpoint["config"])
    ).to(device)
    load_adapter_state(model, checkpoint["adapter_state"])
    model.requires_grad_(False).eval()
    if adapter_state_sha256(model) != checkpoint["adapter_state_sha256"]:
        raise CGL1EvaluationError("CGL1 adapter state differs")
    if (
        frozen_backbone_state_sha256(model.backbone)
        != checkpoint["frozen_backbone_state_sha256"]
    ):
        raise CGL1EvaluationError("CGL1 frozen backbone state differs")
    return model, checkpoint


@torch.no_grad()
def _score(
    model: CausalGroundingInterpreter,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    control: str = "normal",
    map_swapped_back: bool = False,
) -> dict[str, Any]:
    scores = model.candidate_scores(
        records,
        device=device,
        batch_size=batch_size,
        control=control,  # type: ignore[arg-type]
    )
    raw_predictions = scores.argmax(dim=-1).tolist()
    predictions = [1 - value for value in raw_predictions] if map_swapped_back else raw_predictions
    overall = Counter()
    by_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    margins = []
    for row, (record, prediction) in enumerate(zip(records, predictions, strict=True)):
        expected = expected_transaction(record)
        exact = prediction == expected
        comparison = 1 - expected if map_swapped_back else expected
        margins.append(float(scores[row, comparison] - scores[row, 1 - comparison]))
        for counter in (
            overall,
            by_mode[str(record["mode"])],
            by_renderer[str(int(record["renderer"]))],
        ):
            counter["total"] += 1
            counter["exact"] += exact
    return {
        "overall": dict(overall),
        "by_mode": {key: dict(value) for key, value in sorted(by_mode.items())},
        "by_renderer": {
            key: dict(value) for key, value in sorted(by_renderer.items())
        },
        "mean_signed_margin": sum(margins) / len(margins),
        "prediction_sha256": hashlib.sha256(
            json.dumps(predictions, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "raw_prediction_sha256": hashlib.sha256(
            json.dumps(raw_predictions, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "score_sha256": hashlib.sha256(
            scores.contiguous().numpy().tobytes()
        ).hexdigest(),
        "_predictions": predictions,
        "_scores": scores,
    }


def _public(score: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in score.items() if not key.startswith("_")}


def _conditions(
    normal: Mapping[str, Any],
    scrubbed: Mapping[str, Any],
    swapped: Mapping[str, Any],
    renamed: Mapping[str, Any],
    *,
    prompt_identity_exact: bool,
    frozen_backbone_exact: bool,
) -> dict[str, bool]:
    exact = int(normal["overall"].get("exact", 0))
    return {
        "query_at_least_765": int(normal["overall"].get("total", 0)) == 768
        and exact >= 765,
        "every_mode_at_least_254": len(normal["by_mode"]) == 3
        and all(
            int(value.get("total", 0)) == 256
            and int(value.get("exact", 0)) >= 254
            for value in normal["by_mode"].values()
        ),
        "every_renderer_at_least_127": len(normal["by_renderer"]) == 6
        and all(
            int(value.get("total", 0)) == 128
            and int(value.get("exact", 0)) >= 127
            for value in normal["by_renderer"].values()
        ),
        "context_scrub_loses_at_least_250": exact
        - int(scrubbed["overall"].get("exact", 0))
        >= 250,
        "mention_swap_equivariance_at_least_765": int(
            swapped["overall"].get("exact", 0)
        )
        >= 765,
        "entity_rename_prompts_exact": prompt_identity_exact,
        "entity_rename_predictions_exact": normal["_predictions"]
        == renamed["_predictions"],
        "entity_rename_scores_bit_exact": torch.equal(
            normal["_scores"], renamed["_scores"]
        ),
        "frozen_backbone_bit_exact": frozen_backbone_exact,
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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--board-type", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing CGL1 result: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CGL1 requested unavailable CUDA")
    device = torch.device(args.device)
    board = _load_board(args.data, args.data_sha256, args.board_type)
    model, checkpoint = _load_model(
        args.checkpoint,
        args.checkpoint_sha256,
        args.base,
        args.base_sha256,
        args.tokenizer,
        args.tokenizer_sha256,
        device,
    )
    records = [row for row in _referent_records(board) if row["stage"] == "QUERY"]
    renamed_records = [row for row in _rename_records(records) if row["stage"] == "QUERY"]
    normal_prompts = [
        render_claim_prompt(row, candidate)
        for row in records
        for candidate in (0, 1)
    ]
    renamed_prompts = [
        render_claim_prompt(row, candidate)
        for row in renamed_records
        for candidate in (0, 1)
    ]
    normal = _score(model, records, device=device, batch_size=args.batch_size)
    scrubbed = _score(
        model,
        records,
        device=device,
        batch_size=args.batch_size,
        control="scrub_context",
    )
    swapped = _score(
        model,
        records,
        device=device,
        batch_size=args.batch_size,
        control="swap_mentions",
        map_swapped_back=True,
    )
    renamed = _score(
        model, renamed_records, device=device, batch_size=args.batch_size
    )
    frozen_exact = (
        frozen_backbone_state_sha256(model.backbone)
        == checkpoint["frozen_backbone_state_sha256"]
    )
    conditions = _conditions(
        normal,
        scrubbed,
        swapped,
        renamed,
        prompt_identity_exact=normal_prompts == renamed_prompts,
        frozen_backbone_exact=frozen_exact,
    )
    report = {
        "schema": SCHEMA,
        "board_type": args.board_type,
        "status": "pass" if all(conditions.values()) else "fail",
        "promotion_gate": {"conditions": conditions, "passed": all(conditions.values())},
        "normal": _public(normal),
        "scrub_context": _public(scrubbed),
        "mention_swap": _public(swapped),
        "entity_rename": {
            **_public(renamed),
            "prompts_bit_exact": normal_prompts == renamed_prompts,
            "prediction_mismatches": sum(
                left != right
                for left, right in zip(
                    normal["_predictions"], renamed["_predictions"], strict=True
                )
            ),
            "max_absolute_score_difference": float(
                (normal["_scores"] - renamed["_scores"]).abs().max()
            ),
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "adapter_state_sha256": checkpoint["adapter_state_sha256"],
        "frozen_backbone_state_sha256": checkpoint["frozen_backbone_state_sha256"],
        "base": str(args.base),
        "base_sha256": args.base_sha256,
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": args.tokenizer_sha256,
        "backbone_name": checkpoint["backbone_name"],
        "flip_outcomes": checkpoint["flip_outcomes"],
        "data": str(args.data),
        "data_sha256": args.data_sha256,
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "status": report["status"],
                "normal": report["normal"]["overall"],
                "promotion_gate": report["promotion_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
