#!/usr/bin/env python3
"""Apply a qualified AQC1 commit policy to the seven-task product board."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from build_aqc1_product import PAIR_SCHEMA, REPORT_SCHEMA, sha256_file
from hf_aqc1_train_commit import (
    MODEL_SCHEMA,
    candidate_text,
    hidden_states,
    make_head,
    select_candidate,
)
from hf_cvg1_completion_verifier import bounded_token_ids, configure_lora_scope
from hf_product_reasoning_eval import _load_model
from hf_vcr1_evaluate_reviser import _atomic_json, _atomic_lines

PRODUCT_REPORT_SCHEMA = "shohin-aqc1-product-application-v1"
PRODUCT_CANDIDATE_SCHEMA = "shohin-aqc1-product-selection-v1"
VCR_DOMAINS = {
    "grade_school_math": 0.92,
    "competition_math": 0.68,
    "code": 0.725,
    "science": 101 / 198,
    "logic": 0.78,
}


def arm_summary(rows: list[dict], selections: dict[str, int], arm: int | None) -> dict:
    tasks: dict[str, Counter] = {}
    for row in rows:
        task = str(row["task"])
        tasks.setdefault(task, Counter())["total"] += 1
        correct = [bool(candidate["correct"]) for candidate in row["candidates"]]
        if arm is None:
            value = any(correct)
        else:
            value = correct[selections[row["identity_sha256"]] if arm == 2 else arm]
        tasks[task]["correct"] += int(value)
    task_result = {
        task: {"correct": value["correct"], "total": value["total"]}
        for task, value in sorted(tasks.items())
    }
    domains = {
        "grade_school_math": task_result["gsm8k"],
        "competition_math": task_result["math500"],
        "science": task_result["gpqa"],
        "logic": task_result["bbh_logic"],
        "code": {
            "correct": task_result["humaneval"]["correct"]
            + task_result["mbpp"]["correct"],
            "total": task_result["humaneval"]["total"] + task_result["mbpp"]["total"],
        },
    }
    for value in domains.values():
        value["accuracy"] = value["correct"] / value["total"]
    solved = sum(value["correct"] for value in domains.values())
    total = sum(value["total"] for value in domains.values())
    aime = task_result["aime"]
    aime["accuracy"] = aime["correct"] / aime["total"]
    return {
        "tasks": task_result,
        "domains": domains,
        "aime": aime,
        "solved": solved,
        "total": total,
        "macro_accuracy": sum(value["accuracy"] for value in domains.values()) / 5,
    }


def apply(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    pair_report = json.loads(args.pairs_report.read_text(encoding="utf-8"))
    if (
        pair_report.get("schema") != REPORT_SCHEMA
        or pair_report.get("stage") != "pairs"
        or pair_report.get("output_sha256") != sha256_file(args.pairs)
    ):
        raise RuntimeError("AQC1 product pair receipt differs")
    rows = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 568 or any(
        row.get("schema") != PAIR_SCHEMA or row.get("split") != "product"
        for row in rows
    ):
        raise RuntimeError("AQC1 product pair coverage differs")

    training_report = json.loads(args.commit_report.read_text(encoding="utf-8"))
    payload = torch.load(args.commit, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != MODEL_SCHEMA
        or training_report.get("holdout_gate_pass") is not True
        or training_report.get("checkpoint_sha256") != sha256_file(args.commit)
        or training_report.get("arm") != args.arm
    ):
        raise RuntimeError("AQC1 qualified checkpoint differs")
    metadata = payload["metadata"]
    if metadata.get("adapter_checkpoint_sha256") != sha256_file(
        args.adapter_checkpoint
    ):
        raise RuntimeError("AQC1 protected adapter binding differs")

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, _, _ = _load_model(args.model_root, args.adapter_checkpoint, "multimodal")
    trainable = dict(configure_lora_scope(model))
    if set(trainable) != set(payload["backbone_state"]):
        raise RuntimeError("AQC1 backbone state coverage differs")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(
                payload["backbone_state"][name].to(parameter.device, parameter.dtype)
            )
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = make_head(
        args.arm,
        hidden_size,
        int(metadata["head_width"]),
        int(metadata["projection_width"]),
    ).to("cuda:0")
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()
    selections: dict[str, int] = {}
    output_rows = []
    maximum_swap_error = 0.0
    truncated = 0
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_pairs):
            batch = rows[start : start + args.batch_pairs]
            encoded = []
            for row in batch:
                for candidate in row["candidates"]:
                    tokens, hit_limit = bounded_token_ids(
                        tokenizer,
                        candidate_text(row["question"], candidate["completion"]),
                        int(metadata["max_sequence_length"]),
                    )
                    encoded.append(tokens)
                    truncated += int(hit_limit)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = hidden_states(model, encoded, tokenizer.pad_token_id)
                paired = hidden.reshape(-1, 2, hidden.shape[-1])
                direct = head.margin(paired[:, 0], paired[:, 1]).float()
                reverse = head.margin(paired[:, 1], paired[:, 0]).float()
            maximum_swap_error = max(
                maximum_swap_error, float((direct + reverse).abs().max().cpu())
            )
            for row, margin, swapped in zip(
                batch, direct.tolist(), reverse.tolist(), strict=True
            ):
                chosen = select_candidate(margin, row["candidates"])
                reverse_chosen = select_candidate(
                    swapped, list(reversed(row["candidates"]))
                )
                consistent = (
                    chosen == 1 - reverse_chosen
                    or row["candidates"][0]["completion"]
                    == row["candidates"][1]["completion"]
                )
                if not consistent:
                    raise RuntimeError("AQC1 product order consistency failed")
                selections[row["identity_sha256"]] = chosen
                selected = row["candidates"][chosen]
                output_rows.append(
                    {
                        "schema": PRODUCT_CANDIDATE_SCHEMA,
                        "identity_sha256": row["identity_sha256"],
                        "task": row["task"],
                        "selected_lineage": selected["lineage"],
                        "completion": selected["completion"],
                        "prediction": selected.get("prediction"),
                        "correct": bool(selected["correct"]),
                        "margin": margin,
                    }
                )
    elapsed = time.monotonic() - started
    candidate_hash = _atomic_lines(args.candidates_output, output_rows)
    arms = {
        "idr1": arm_summary(rows, selections, 0),
        "control": arm_summary(rows, selections, 1),
        "selected": arm_summary(rows, selections, 2),
        "oracle": arm_summary(rows, selections, None),
    }
    selected = arms["selected"]
    gates = {
        "solved_at_least_vcr1_368": selected["solved"] >= 368,
        "macro_at_least_vcr1": selected["macro_accuracy"] >= 0.723020202020202,
        "code_at_least_30_of_40": selected["domains"]["code"]["correct"] >= 30,
        "at_least_three_domains_match_vcr1": sum(
            selected["domains"][name]["accuracy"] >= floor
            for name, floor in VCR_DOMAINS.items()
        )
        >= 3,
    }
    report = {
        "schema": PRODUCT_REPORT_SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "commit": str(args.commit.resolve()),
        "commit_sha256": sha256_file(args.commit),
        "commit_report": str(args.commit_report.resolve()),
        "commit_report_sha256": sha256_file(args.commit_report),
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "arms": arms,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "maximum_swap_error": maximum_swap_error,
        "order_consistency": 1.0,
        "prompt_truncated": truncated,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidate_hash,
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("antisymmetric", "independent"), required=True
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--commit-report", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pairs-report", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-pairs", type=int, default=2)
    args = parser.parse_args()
    report = apply(args)
    print(
        json.dumps(
            {"gate_pass": report["gate_pass"], "selected": report["arms"]["selected"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
