#!/usr/bin/env python3
"""Evaluate the frozen DIVERGE-PQI1 semantic query compiler."""

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

from diverge_ccr1_data import validate_ccr1_board_row
from diverge_iem1_runtime import tensorize_queries
from diverge_nve1_runtime import hard_role_permutation
from diverge_pqi1_data import validate_pqi1_board_row
from diverge_pqi1_runtime import (
    PQI1Config,
    PretrainedQueryGrounder,
    QueryControl,
    adapter_state_sha256,
    load_adapter_state,
)
from eval_diverge_ccr1 import _referent_records, _rename_records
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-pqi1-evaluation-v1"


class PQI1EvaluationError(RuntimeError):
    """A PQI1 checkpoint, board, or result contract differs."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_board(path: Path, expected_sha256: str, board_type: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise PQI1EvaluationError("PQI1 board hash differs")
    validator = validate_ccr1_board_row if board_type == "development" else validate_pqi1_board_row
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validator(row)
            rows.append(row)
    if len(rows) != 256:
        raise PQI1EvaluationError("PQI1 board row count differs")
    return rows


def _load_model(
    checkpoint_path: Path,
    checkpoint_sha256: str,
    base_path: Path,
    base_sha256: str,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    device: torch.device,
) -> tuple[PretrainedQueryGrounder, dict[str, Any]]:
    for path, expected, label in (
        (checkpoint_path, checkpoint_sha256, "checkpoint"),
        (base_path, base_sha256, "base"),
        (tokenizer_path, tokenizer_sha256, "tokenizer"),
    ):
        if sha256_path(path) != expected:
            raise PQI1EvaluationError(f"PQI1 {label} hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-pqi1-training-report-v1":
        raise PQI1EvaluationError("PQI1 checkpoint schema differs")
    if checkpoint.get("base_sha256") != base_sha256 or checkpoint.get("tokenizer_sha256") != tokenizer_sha256:
        raise PQI1EvaluationError("PQI1 parent receipt differs")
    backbone, _, _ = load_frozen_pointer_backbone(base_path, device=device)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = PretrainedQueryGrounder(
        backbone, tokenizer, PQI1Config(**checkpoint["config"])
    ).to(device)
    load_adapter_state(model, checkpoint["adapter_state"])
    model.requires_grad_(False).eval()
    if adapter_state_sha256(model) != checkpoint["adapter_state_sha256"]:
        raise PQI1EvaluationError("PQI1 adapter state differs")
    return model, checkpoint


@torch.no_grad()
def _score(
    model: PretrainedQueryGrounder,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    control: QueryControl = "normal",
) -> dict[str, Any]:
    overall = Counter()
    by_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    predictions = []
    digest = hashlib.sha256()
    logits_all = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        ids, mask, symbols, targets = tensorize_queries(batch, device)
        logits = model(ids, mask, symbols, control=control)
        cpu_logits = logits.detach().cpu().contiguous()
        logits_all.append(cpu_logits)
        digest.update(cpu_logits.numpy().tobytes())
        for index, row in enumerate(batch):
            prediction = hard_role_permutation(logits[index])
            expected = tuple(int(value) for value in targets[index].tolist())
            exact = prediction == expected
            predictions.append(prediction)
            for counter in (
                overall,
                by_mode[str(row["mode"])],
                by_renderer[str(int(row["renderer"]))],
            ):
                counter["total"] += 1
                counter["exact"] += exact
    return {
        "overall": dict(overall),
        "by_mode": {key: dict(value) for key, value in sorted(by_mode.items())},
        "by_renderer": {key: dict(value) for key, value in sorted(by_renderer.items())},
        "prediction_sha256": hashlib.sha256(
            json.dumps(predictions, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "logit_sha256": digest.hexdigest(),
        "_predictions": predictions,
        "_logits": torch.cat(logits_all, dim=0),
    }


def _public(score: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in score.items() if not key.startswith("_")}


def _conditions(
    normal: Mapping[str, Any],
    swapped: Mapping[str, Any],
    scrubbed: Mapping[str, Any],
    renamed: Mapping[str, Any],
) -> dict[str, bool]:
    exact = int(normal["overall"].get("exact", 0))
    return {
        "query_at_least_765": int(normal["overall"].get("total", 0)) == 768 and exact >= 765,
        "every_mode_at_least_254": len(normal["by_mode"]) == 3 and all(
            int(value.get("total", 0)) == 256 and int(value.get("exact", 0)) >= 254
            for value in normal["by_mode"].values()
        ),
        "every_renderer_at_least_127": len(normal["by_renderer"]) == 6 and all(
            int(value.get("total", 0)) == 128 and int(value.get("exact", 0)) >= 127
            for value in normal["by_renderer"].values()
        ),
        "role_swap_loses_at_least_500": exact - int(swapped["overall"].get("exact", 0)) >= 500,
        "context_scrub_loses_at_least_250": exact - int(scrubbed["overall"].get("exact", 0)) >= 250,
        "entity_rename_assignments_exact": normal["_predictions"] == renamed["_predictions"],
        "entity_rename_logits_bit_exact": torch.equal(normal["_logits"], renamed["_logits"]),
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing PQI1 result: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("PQI1 requested unavailable CUDA")
    device = torch.device(args.device)
    rows = _load_board(args.data, args.data_sha256, args.board_type)
    model, checkpoint = _load_model(
        args.checkpoint,
        args.checkpoint_sha256,
        args.base,
        args.base_sha256,
        args.tokenizer,
        args.tokenizer_sha256,
        device,
    )
    records = [row for row in _referent_records(rows) if row["stage"] == "QUERY"]
    normal = _score(model, records, device=device, batch_size=args.batch_size)
    swapped = _score(
        model, records, device=device, batch_size=args.batch_size, control="role_slot_swap"
    )
    scrubbed = _score(
        model, records, device=device, batch_size=args.batch_size, control="scrub_context"
    )
    renamed_records = [row for row in _rename_records(records) if row["stage"] == "QUERY"]
    renamed = _score(model, renamed_records, device=device, batch_size=args.batch_size)
    conditions = _conditions(normal, swapped, scrubbed, renamed)
    report = {
        "schema": SCHEMA,
        "board_type": args.board_type,
        "status": "pass" if all(conditions.values()) else "fail",
        "promotion_gate": {"conditions": conditions, "passed": all(conditions.values())},
        "normal": _public(normal),
        "role_slot_swap": _public(swapped),
        "scrub_context": _public(scrubbed),
        "entity_rename": {
            **_public(renamed),
            "assignment_mismatches": sum(
                left != right for left, right in zip(normal["_predictions"], renamed["_predictions"], strict=True)
            ),
            "max_absolute_logit_difference": float((normal["_logits"] - renamed["_logits"]).abs().max()),
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "adapter_state_sha256": checkpoint["adapter_state_sha256"],
        "base": str(args.base),
        "base_sha256": args.base_sha256,
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": args.tokenizer_sha256,
        "backbone_name": checkpoint["backbone_name"],
        "shuffle_supervision": checkpoint["shuffle_supervision"],
        "data": str(args.data),
        "data_sha256": args.data_sha256,
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": sha256_path(args.output),
        "status": report["status"],
        "normal": report["normal"]["overall"],
        "promotion_gate": report["promotion_gate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
