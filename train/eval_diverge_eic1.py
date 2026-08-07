#!/usr/bin/env python3
"""Evaluate one frozen DIVERGE-EIC1 identity-commit arm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from diverge_cgl1_runtime import (
    adapter_state_sha256,
    frozen_backbone_state_sha256,
    load_adapter_state,
)
from diverge_eic1_runtime import (
    EIC1Config,
    EquivariantIdentityCommitter,
    render_claim_prompt,
)
from diverge_eic1_confirmation_data import BOARD_ROWS, validate_confirmation_row
from eval_diverge_cgl1 import _conditions, _load_board as _load_cgl1_board, _public, _score
from eval_diverge_ccr1 import _referent_records, _rename_records
from eval_diverge_pqi1 import sha256_path
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-eic1-evaluation-v1"


def _load_board(path: Path, expected_sha256: str, board_type: str) -> list[dict]:
    if board_type == "development":
        return _load_cgl1_board(path, expected_sha256, "development")
    if board_type != "confirmation" or sha256_path(path) != expected_sha256:
        raise SystemExit("EIC1 confirmation board receipt differs")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_confirmation_row(row)
            rows.append(row)
    if len(rows) != BOARD_ROWS:
        raise SystemExit("EIC1 confirmation board count differs")
    return rows


def _load_model(
    checkpoint_path: Path,
    checkpoint_sha256: str,
    base_path: Path,
    base_sha256: str,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    device: torch.device,
) -> tuple[EquivariantIdentityCommitter, dict]:
    for path, expected, label in (
        (checkpoint_path, checkpoint_sha256, "checkpoint"),
        (base_path, base_sha256, "base"),
        (tokenizer_path, tokenizer_sha256, "tokenizer"),
    ):
        if sha256_path(path) != expected:
            raise SystemExit(f"EIC1 {label} hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-eic1-training-report-v1":
        raise SystemExit("EIC1 checkpoint schema differs")
    if (
        checkpoint.get("base_sha256") != base_sha256
        or checkpoint.get("tokenizer_sha256") != tokenizer_sha256
    ):
        raise SystemExit("EIC1 parent receipt differs")
    backbone, _, _ = load_frozen_pointer_backbone(base_path, device=device)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = EquivariantIdentityCommitter(
        backbone,
        tokenizer,
        EIC1Config(**checkpoint["config"]),
    ).to(device)
    load_adapter_state(model, checkpoint["adapter_state"])
    model.requires_grad_(False).eval()
    if adapter_state_sha256(model) != checkpoint["adapter_state_sha256"]:
        raise SystemExit("EIC1 adapter state differs")
    if (
        frozen_backbone_state_sha256(model.backbone)
        != checkpoint["frozen_backbone_state_sha256"]
    ):
        raise SystemExit("EIC1 frozen backbone state differs")
    return model, checkpoint


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
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
        raise SystemExit("refusing existing EIC1 result")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("EIC1 requested unavailable CUDA")
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
        model,
        renamed_records,
        device=device,
        batch_size=args.batch_size,
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
    equivariance_error = float(
        (normal["_scores"] - swapped["_scores"].flip(dims=(-1,))).abs().max()
    )
    if checkpoint["projection_mode"] == "involution":
        conditions["projection_identity_bit_exact"] = equivariance_error == 0.0
    passed = all(conditions.values())
    report = {
        "schema": SCHEMA,
        "board_type": args.board_type,
        "status": "pass" if passed else "fail",
        "promotion_gate": {"conditions": conditions, "passed": passed},
        "normal": _public(normal),
        "scrub_context": _public(scrubbed),
        "mention_swap": _public(swapped),
        "projection_identity_max_absolute_error": equivariance_error,
        "entity_rename": {
            **_public(renamed),
            "prompts_bit_exact": normal_prompts == renamed_prompts,
            "prediction_mismatches": sum(
                left != right
                for left, right in zip(
                    normal["_predictions"],
                    renamed["_predictions"],
                    strict=True,
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
        "projection_mode": checkpoint["projection_mode"],
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
                "normal": report["normal"]["overall"],
                "mapped_swap": report["mention_swap"]["overall"],
                "projection_error": equivariance_error,
                "status": report["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
