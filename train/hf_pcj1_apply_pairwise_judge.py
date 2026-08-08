#!/usr/bin/env python3
"""Apply a qualified PCJ1 judge to preserved whole-solution lineages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from hf_cvg1_apply_completion_verifier import (
    _summarize,
    load_evaluation_pairs,
)
from hf_cvg1_completion_verifier import configure_lora_scope, sha256_file
from hf_pcj1_pairwise_judge import (
    MODEL_SCHEMA,
    REPORT_SCHEMA,
    PairwiseJudgeHead,
    conservative_selection,
    forward_logits,
    pair_token_ids,
)

OUTPUT_SCHEMA = "shohin-pcj1-evaluation-selection-v1"


class PCJ1ApplicationError(RuntimeError):
    """The PCJ1 judge or evaluation-pair contract is not qualified."""


def _load_qualified_report(path: Path, judge: Path, adapter: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PCJ1ApplicationError(f"missing PCJ1 report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
        raise PCJ1ApplicationError("PCJ1 report is incomplete")
    if report.get("holdout", {}).get("gate_pass") is not True:
        raise PCJ1ApplicationError("source-disjoint PCJ1 holdout did not pass")
    if report.get("judge_sha256") != sha256_file(judge):
        raise PCJ1ApplicationError("PCJ1 judge hash differs")
    if report.get("adapter_checkpoint_sha256") != sha256_file(adapter):
        raise PCJ1ApplicationError("protected adapter checkpoint hash differs")
    if report.get("inference_fields") != ["question", "candidate_a", "candidate_b"]:
        raise PCJ1ApplicationError("PCJ1 inference fields differ")
    if report.get("task_or_benchmark_label_at_inference") is not False:
        raise PCJ1ApplicationError("PCJ1 admits a task label at inference")
    return report


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise PCJ1ApplicationError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def apply(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import _load_model

    rows = load_evaluation_pairs(args.pairs)
    judge_report = _load_qualified_report(
        args.judge_report, args.judge, args.adapter_checkpoint
    )
    payload = torch.load(args.judge, map_location="cpu", weights_only=True)
    if payload.get("schema") != MODEL_SCHEMA:
        raise PCJ1ApplicationError("PCJ1 checkpoint schema differs")
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("model_revision") != args.model_revision
    ):
        raise PCJ1ApplicationError("PCJ1 model metadata differs")
    if metadata.get("adapter_checkpoint_sha256") != sha256_file(
        args.adapter_checkpoint
    ):
        raise PCJ1ApplicationError("PCJ1 adapter metadata differs")

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, _, _ = _load_model(
        args.model_root, args.adapter_checkpoint, str(metadata["model_loader"])
    )
    trainable = dict(configure_lora_scope(model))
    saved_backbone = payload.get("backbone_state")
    if not isinstance(saved_backbone, dict) or set(saved_backbone) != set(trainable):
        raise PCJ1ApplicationError("PCJ1 backbone state coverage differs")
    with torch.no_grad():
        for name, parameter in trainable.items():
            saved = saved_backbone[name]
            if tuple(saved.shape) != tuple(parameter.shape):
                raise PCJ1ApplicationError("PCJ1 backbone tensor shape differs")
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = PairwiseJudgeHead(hidden_size, int(metadata["head_width"])).to("cuda:0")
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()

    started = time.monotonic()
    selected: dict[str, int] = {}
    selections: list[dict[str, Any]] = []
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_rows):
            batch = rows[start : start + args.batch_rows]
            token_rows: list[list[int]] = []
            for row in batch:
                for order in ((0, 1), (1, 0)):
                    tokens, was_truncated = pair_token_ids(
                        tokenizer,
                        row,
                        order,
                        int(metadata["max_sequence_length"]),
                    )
                    token_rows.append(tokens)
                    truncated += int(was_truncated)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_logits(
                    model, head, token_rows, tokenizer.pad_token_id
                ).float()
            predictions = logits.argmax(dim=-1).cpu().tolist()
            for index, row in enumerate(batch):
                prediction_ab = int(predictions[index * 2])
                prediction_ba = int(predictions[index * 2 + 1])
                chosen, consistent, verdict_ab, verdict_ba = conservative_selection(
                    prediction_ab, prediction_ba
                )
                identity = row["identity_sha256"]
                selected[identity] = chosen
                selections.append(
                    {
                        "identity_sha256": identity,
                        "order_consistent": consistent,
                        "selected_lineage": ("base", "expert")[chosen],
                        "selected_correct": row["candidates"][chosen]["correct"],
                        "verdict_ab": verdict_ab,
                        "verdict_ba": verdict_ba,
                    }
                )
            print(
                f"[pcj1-apply] {min(start + len(batch), len(rows))}/{len(rows)}",
                flush=True,
            )
    torch.cuda.synchronize()
    summary = _summarize(rows, selected)
    consistency = sum(item["order_consistent"] for item in selections) / len(selections)
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "judge": str(args.judge.resolve()),
        "judge_sha256": sha256_file(args.judge),
        "judge_report": str(args.judge_report.resolve()),
        "judge_report_sha256": sha256_file(args.judge_report),
        "protected_adapter": str(args.adapter_checkpoint.resolve()),
        "protected_adapter_sha256": sha256_file(args.adapter_checkpoint),
        "source_disjoint_holdout_gate_pass": judge_report["holdout"]["gate_pass"],
        "inference_fields": ["question", "candidate_a", "candidate_b"],
        "task_or_benchmark_label_at_inference": False,
        "order_consistency": consistency,
        "prompt_truncated": truncated,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "selections": selections,
        **summary,
        "required_next_step": (
            "compare_qualified_pcj1_arms"
            if summary["comparison"]["development_gate_pass"]
            else "close_this_pcj1_arm"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--judge-report", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=2)
    args = parser.parse_args()
    if args.batch_rows <= 0:
        parser.error("batch rows must be positive")
    report = apply(args)
    _atomic_json(args.output, report)
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
