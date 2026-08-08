#!/usr/bin/env python3
"""Train PCJ1 to compare two frozen whole-solution lineages jointly."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_cvg1_completion_verifier import (
    PAIR_SCHEMA as CVG_PAIR_SCHEMA,
    _atomic_json,
    _atomic_torch,
    _pad_rows,
    bounded_token_ids,
    configure_lora_scope,
    load_pairs,
    sha256_file,
)

PAIR_SCHEMA = CVG_PAIR_SCHEMA
MODEL_SCHEMA = "shohin-pcj1-pairwise-judge-v1"
REPORT_SCHEMA = "shohin-pcj1-pairwise-judge-report-v1"
SPLITS = ("train", "development", "holdout")
OUTCOMES = ("base_only", "both_correct", "both_wrong", "expert_only")
LABELS = ("a_better", "tie", "b_better")
SPLIT_SEED = 2026080811


class PCJ1Error(RuntimeError):
    """The frozen PCJ1 data, model, or gate contract differs."""


def assigned_split(identity: str, seed: int = SPLIT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_500:
        return "development"
    return "holdout"


def load_partitioned_pairs(path: Path, split_seed: int) -> list[dict[str, Any]]:
    rows = load_pairs(path)
    partitioned = [
        {
            **row,
            "source_split": row["split"],
            "split": assigned_split(row["identity_sha256"], split_seed),
        }
        for row in rows
    ]
    split_counts = Counter(row["split"] for row in partitioned)
    if set(split_counts) != set(SPLITS):
        raise PCJ1Error("PCJ1 split coverage differs")
    for split in SPLITS:
        outcomes = {
            row["outcome_class"] for row in partitioned if row["split"] == split
        }
        if outcomes != set(OUTCOMES):
            raise PCJ1Error(f"PCJ1 {split} lacks an outcome class")
    return partitioned


def partition_receipt(rows: list[dict[str, Any]], split_seed: int) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    identity_hashes: dict[str, str] = {}
    for split in SPLITS:
        identities = sorted(
            row["identity_sha256"] for row in rows if row["split"] == split
        )
        identity_hashes[split] = hashlib.sha256(
            ("\n".join(identities) + "\n").encode()
        ).hexdigest()
    for row in rows:
        counts[row["split"]]["total"] += 1
        counts[row["split"]][str(row["task"])] += 1
        counts[row["split"]][str(row["outcome_class"])] += 1
    return {
        "identity_sha256": identity_hashes,
        "seed": split_seed,
        "split_rule": "sha256(seed\\0identity)[:8] mod 10000; 0:7000 train, 7000:8500 development, 8500:10000 holdout",
        "counts": {split: dict(counts[split]) for split in SPLITS},
    }


def build_balanced_strata(
    rows: list[dict[str, Any]], seed: int
) -> OrderedDict[tuple[str, str], list[int]]:
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "train":
            strata[(str(row["task"]), str(row["outcome_class"]))].append(index)
    if not strata:
        raise PCJ1Error("PCJ1 has no training strata")
    generator = random.Random(seed)
    ordered: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for key in sorted(strata):
        generator.shuffle(strata[key])
        ordered[key] = strata[key]
    return ordered


def pairwise_text(question: str, candidate_a: str, candidate_b: str) -> str:
    return (
        "Compare two complete candidate solutions to the same problem. Check "
        "their reasoning, factual claims, calculations, and final answer. Decide "
        "whether Candidate A is better supported, both are equally supported, or "
        "Candidate B is better supported.\n\n"
        f"Problem:\n{question}\n\nCandidate A:\n{candidate_a}\n\n"
        f"Candidate B:\n{candidate_b}\n\nVerdict:"
    )


def ordered_indices(identity: str, seed: int, presentation: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"{seed}\0{identity}\0{presentation}".encode()).digest()
    return (0, 1) if digest[0] & 1 == 0 else (1, 0)


def ordered_label(row: dict[str, Any], order: tuple[int, int]) -> int:
    correct_a = bool(row["candidates"][order[0]]["correct"])
    correct_b = bool(row["candidates"][order[1]]["correct"])
    if correct_a == correct_b:
        return 1
    return 0 if correct_a else 2


def pair_token_ids(
    tokenizer: Any,
    row: dict[str, Any],
    order: tuple[int, int],
    max_sequence_length: int,
) -> tuple[list[int], bool]:
    candidates = row["candidates"]
    return bounded_token_ids(
        tokenizer,
        pairwise_text(
            row["question"],
            candidates[order[0]]["completion"],
            candidates[order[1]]["completion"],
        ),
        max_sequence_length,
    )


class PairwiseJudgeHead(nn.Module):
    def __init__(self, hidden_size: int, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, len(LABELS)),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden.float())


def forward_logits(
    model: nn.Module,
    head: PairwiseJudgeHead,
    token_rows: list[list[int]],
    pad_token_id: int,
) -> torch.Tensor:
    input_ids, attention = _pad_rows(token_rows, pad_token_id)
    outputs = model.text_model(
        input_ids=input_ids,
        attention_mask=attention,
        use_cache=False,
    )
    final_indices = attention.sum(dim=1) - 1
    hidden = outputs.last_hidden_state[
        torch.arange(len(token_rows), device="cuda:0"), final_indices
    ]
    return head(hidden)


def semantic_verdict(prediction: int, order: tuple[int, int]) -> str:
    if prediction == 1:
        return "tie"
    if prediction == 0:
        selected = order[0]
    elif prediction == 2:
        selected = order[1]
    else:
        raise PCJ1Error("PCJ1 prediction is outside the label space")
    return ("base", "expert")[selected]


def conservative_selection(
    prediction_ab: int, prediction_ba: int
) -> tuple[int, bool, str, str]:
    verdict_ab = semantic_verdict(prediction_ab, (0, 1))
    verdict_ba = semantic_verdict(prediction_ba, (1, 0))
    consistent = verdict_ab == verdict_ba
    selected = 0 if consistent and verdict_ab == "base" else 1
    return selected, consistent, verdict_ab, verdict_ba


def metrics_from_predictions(
    rows: list[dict[str, Any]],
    predictions: dict[str, tuple[int, int]],
    *,
    split: str,
) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    verdict_counts: Counter[str] = Counter()
    for row in rows:
        if row["split"] != split:
            continue
        identity = row["identity_sha256"]
        if identity not in predictions:
            raise PCJ1Error("PCJ1 prediction coverage is incomplete")
        prediction_ab, prediction_ba = predictions[identity]
        selected, consistent, verdict_ab, verdict_ba = conservative_selection(
            prediction_ab, prediction_ba
        )
        verdict_counts[f"ab:{verdict_ab}"] += 1
        verdict_counts[f"ba:{verdict_ba}"] += 1
        correct = [bool(candidate["correct"]) for candidate in row["candidates"]]
        disagreement = correct[0] != correct[1]
        for key in ("overall", str(row["task"])):
            bucket = buckets[key]
            bucket["total"] += 1
            bucket["base_correct"] += int(correct[0])
            bucket["expert_correct"] += int(correct[1])
            bucket["selected_correct"] += int(correct[selected])
            bucket["oracle_correct"] += int(any(correct))
            bucket["base_commits"] += int(selected == 0)
            bucket["order_consistent"] += int(consistent)
            bucket["disagreements"] += int(disagreement)
            bucket["disagreement_selected_correct"] += int(
                disagreement and correct[selected]
            )
    if "overall" not in buckets:
        raise PCJ1Error(f"PCJ1 {split} split is empty")
    metrics: dict[str, Any] = {}
    for key, bucket in sorted(buckets.items()):
        total = bucket["total"]
        disagreements = bucket["disagreements"]
        metrics[key] = {
            **dict(bucket),
            "base_accuracy": bucket["base_correct"] / total,
            "expert_accuracy": bucket["expert_correct"] / total,
            "selected_accuracy": bucket["selected_correct"] / total,
            "oracle_accuracy": bucket["oracle_correct"] / total,
            "base_commit_rate": bucket["base_commits"] / total,
            "order_consistency": bucket["order_consistent"] / total,
            "disagreement_selection_accuracy": (
                bucket["disagreement_selected_correct"] / disagreements
                if disagreements
                else None
            ),
        }
    overall = metrics["overall"]
    gate = {
        "order_consistency_at_least_0_90": overall["order_consistency"] >= 0.90,
        "base_commit_between_0_02_and_0_50": 0.02
        <= overall["base_commit_rate"]
        <= 0.50,
        "disagreement_selection_at_least_0_80": isinstance(
            overall["disagreement_selection_accuracy"], float
        )
        and overall["disagreement_selection_accuracy"] >= 0.80,
        "selected_beats_expert_by_0_02": overall["selected_accuracy"]
        >= overall["expert_accuracy"] + 0.02,
    }
    return {
        "gate": gate,
        "gate_pass": all(gate.values()),
        "metrics": metrics,
        "verdict_counts": dict(verdict_counts),
    }


def evaluate(
    model: nn.Module,
    head: PairwiseJudgeHead,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    split: str,
    max_sequence_length: int,
    batch_rows: int,
) -> tuple[dict[str, Any], int]:
    selected_rows = [row for row in rows if row["split"] == split]
    predictions: dict[str, tuple[int, int]] = {}
    truncated = 0
    model.eval()
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(selected_rows), batch_rows):
            batch = selected_rows[start : start + batch_rows]
            token_rows: list[list[int]] = []
            for row in batch:
                for order in ((0, 1), (1, 0)):
                    token_ids, was_truncated = pair_token_ids(
                        tokenizer, row, order, max_sequence_length
                    )
                    token_rows.append(token_ids)
                    truncated += int(was_truncated)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_logits(model, head, token_rows, tokenizer.pad_token_id)
            classes = logits.argmax(dim=-1).cpu().tolist()
            for index, row in enumerate(batch):
                predictions[row["identity_sha256"]] = (
                    int(classes[index * 2]),
                    int(classes[index * 2 + 1]),
                )
    return metrics_from_predictions(rows, predictions, split=split), truncated


def train(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import _load_model

    if args.output.exists():
        raise PCJ1Error(f"refusing existing PCJ1 output: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = load_partitioned_pairs(args.pairs, args.split_seed)
    receipt = partition_receipt(rows, args.split_seed)
    strata = build_balanced_strata(rows, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    protected_sha256_before = sha256_file(args.adapter_checkpoint)
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    trainable = configure_lora_scope(model)
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = PairwiseJudgeHead(hidden_size, args.head_width).to("cuda:0")
    model_parameters = [parameter for _, parameter in trainable]
    head_parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model_parameters, "lr": args.backbone_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True,
    )
    positions = dict.fromkeys(strata, 0)
    keys = list(strata)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    head.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = microstep = pair_presentations = truncated = 0
    trace: list[dict[str, Any]] = []
    while update < args.updates:
        key = keys[microstep % len(keys)]
        stratum = strata[key]
        row = rows[stratum[positions[key] % len(stratum)]]
        positions[key] += 1
        order = ordered_indices(row["identity_sha256"], args.seed, microstep)
        token_ids, was_truncated = pair_token_ids(
            tokenizer, row, order, args.max_sequence_length
        )
        truncated += int(was_truncated)
        label = torch.tensor([ordered_label(row, order)], device="cuda:0")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_logits(
                model, head, [token_ids], tokenizer.pad_token_id
            ).float()
            loss = F.cross_entropy(logits, label) / args.gradient_accumulation
        loss.backward()
        pair_presentations += 1
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model_parameters + head_parameters, args.max_gradient_norm
        )
        progress = update / max(args.updates - 1, 1)
        schedule = 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = args.backbone_learning_rate * schedule
        optimizer.param_groups[1]["lr"] = args.head_learning_rate * schedule
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % args.log_interval == 0:
            event = {
                "gradient_norm": float(gradient_norm),
                "pair_presentations": pair_presentations,
                "pairs_per_second": pair_presentations / (time.monotonic() - started),
                "prompt_truncated": truncated,
                "update": update,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)

    development, development_truncated = evaluate(
        model,
        head,
        tokenizer,
        rows,
        split="development",
        max_sequence_length=args.max_sequence_length,
        batch_rows=args.evaluation_batch_rows,
    )
    holdout, holdout_truncated = evaluate(
        model,
        head,
        tokenizer,
        rows,
        split="holdout",
        max_sequence_length=args.max_sequence_length,
        batch_rows=args.evaluation_batch_rows,
    )
    model_path = args.output / "judge.pt"
    metadata = {
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": protected_sha256_before,
        "adapter_metadata": adapter_metadata,
        "backbone_learning_rate": args.backbone_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "head_width": args.head_width,
        "inference_fields": ["question", "candidate_a", "candidate_b"],
        "max_sequence_length": args.max_sequence_length,
        "model_loader": model_loader,
        "model_revision": args.model_revision,
        "model_root": str(args.model_source_root.resolve()),
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "seed": args.seed,
        "split_receipt": receipt,
        "task_or_benchmark_label_at_inference": False,
        "trainable_parameter_names": [name for name, _ in trainable],
        "updates": args.updates,
    }
    _atomic_torch(
        model_path,
        {
            "schema": MODEL_SCHEMA,
            "metadata": metadata,
            "backbone_state": {
                name: parameter.detach().cpu() for name, parameter in trainable
            },
            "head_state": {
                name: tensor.detach().cpu()
                for name, tensor in head.state_dict().items()
            },
        },
    )
    protected_sha256_after = sha256_file(args.adapter_checkpoint)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        **metadata,
        "development": development,
        "holdout": holdout,
        "development_prompt_truncated": development_truncated,
        "holdout_prompt_truncated": holdout_truncated,
        "training_prompt_truncated": truncated,
        "gradient_accumulation": args.gradient_accumulation,
        "pair_presentations": pair_presentations,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "protected_adapter_sha256_after": protected_sha256_after,
        "protected_adapter_unchanged": protected_sha256_after
        == protected_sha256_before,
        "strata_counts": {
            f"{task}:{outcome}": len(indices)
            for (task, outcome), indices in strata.items()
        },
        "trace": trace,
        "judge": str(model_path.resolve()),
        "judge_sha256": sha256_file(model_path),
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", default="auto")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--head-width", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=3072)
    parser.add_argument("--evaluation-batch-rows", type=int, default=2)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026080812)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.gradient_accumulation,
        args.head_width,
        args.max_sequence_length,
        args.evaluation_batch_rows,
        args.log_interval,
    )
    if any(value <= 0 for value in positive):
        parser.error("PCJ1 dimensions must be positive")
    if args.backbone_learning_rate <= 0 or args.head_learning_rate <= 0:
        parser.error("PCJ1 learning rates must be positive")
    return args


def main() -> int:
    report = train(parse_args())
    print(
        json.dumps(
            {
                "holdout_gate_pass": report["holdout"]["gate_pass"],
                "judge_sha256": report["judge_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
