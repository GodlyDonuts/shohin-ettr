#!/usr/bin/env python3
"""Train AQC1 to commit to one coherent same-family trajectory."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_cvg1_completion_verifier import (
    _pad_rows,
    bounded_token_ids,
    configure_lora_scope,
    sha256_file,
)

PAIR_SCHEMA = "shohin-aqc1-whole-trajectory-pair-v1"
MODEL_SCHEMA = "shohin-aqc1-commit-model-v1"
REPORT_SCHEMA = "shohin-aqc1-commit-report-v1"
SPLITS = ("train", "development", "holdout")
TASKS = ("math500", "bbh_logic", "mbpp")
OUTCOMES = ("both_correct", "idr1_only", "both_wrong", "control_only")


class AQC1Error(RuntimeError):
    """The frozen AQC1 model, data, or gate contract differs."""


def expected_outcome(left: bool, right: bool) -> str:
    if left and right:
        return "both_correct"
    if left:
        return "idr1_only"
    if right:
        return "control_only"
    return "both_wrong"


def load_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AQC1Error(f"malformed AQC1 row {line_number}") from error
            identity = row.get("identity_sha256")
            candidates = row.get("candidates")
            if row.get("schema") != PAIR_SCHEMA:
                raise AQC1Error("AQC1 pair schema differs")
            if not isinstance(identity, str) or len(identity) != 64 or identity in identities:
                raise AQC1Error("AQC1 identity is invalid or duplicated")
            identities.add(identity)
            if row.get("split") not in SPLITS or row.get("task") not in TASKS:
                raise AQC1Error("AQC1 split or task differs")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise AQC1Error("AQC1 question is empty")
            if not isinstance(candidates, list) or len(candidates) != 2:
                raise AQC1Error("AQC1 requires two candidates")
            if [candidate.get("lineage") for candidate in candidates] != [
                "idr1",
                "control",
            ]:
                raise AQC1Error("AQC1 candidate order differs")
            for candidate in candidates:
                if not isinstance(candidate.get("completion"), str):
                    raise AQC1Error("AQC1 candidate completion differs")
                if not isinstance(candidate.get("correct"), bool):
                    raise AQC1Error("AQC1 candidate outcome differs")
            outcome = expected_outcome(
                bool(candidates[0]["correct"]), bool(candidates[1]["correct"])
            )
            if row.get("outcome_class") != outcome:
                raise AQC1Error("AQC1 outcome binding differs")
            rows.append(row)
    if not rows or {row["split"] for row in rows} != set(SPLITS):
        raise AQC1Error("AQC1 split coverage differs")
    return rows


def balanced_strata(
    rows: list[dict[str, Any]], seed: int
) -> OrderedDict[tuple[str, str], list[int]]:
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "train":
            strata[(str(row["task"]), str(row["outcome_class"]))].append(index)
    if {outcome for _, outcome in strata} != set(OUTCOMES):
        raise AQC1Error("AQC1 training outcome coverage differs")
    generator = random.Random(seed)
    result: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for key in sorted(strata):
        generator.shuffle(strata[key])
        result[key] = strata[key]
    return result


def candidate_text(question: str, completion: str) -> str:
    return (
        "Assess this complete candidate solution to the problem. Track whether "
        "its reasoning, calculations, and final answer are internally and "
        "externally supported.\n\n"
        f"Problem:\n{question}\n\nCandidate solution:\n{completion}\n\n"
        "Assessment state:"
    )


def token_rows(tokenizer: Any, row: dict[str, Any], maximum: int) -> tuple[list[list[int]], int]:
    rows: list[list[int]] = []
    truncated = 0
    for candidate in row["candidates"]:
        tokens, was_truncated = bounded_token_ids(
            tokenizer,
            candidate_text(row["question"], candidate["completion"]),
            maximum,
        )
        rows.append(tokens)
        truncated += int(was_truncated)
    return rows, truncated


class IndependentCommitHead(nn.Module):
    def __init__(self, hidden_size: int, width: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def margin(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.score(left.float()).squeeze(-1) - self.score(right.float()).squeeze(-1)


class AntisymmetricCommitHead(nn.Module):
    def __init__(self, hidden_size: int, width: int, projection: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection),
            nn.GELU(),
        )
        self.phi = nn.Sequential(
            nn.Linear(projection * 4, width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def _ordered(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        features = torch.cat((left, right, left - right, left * right), dim=-1)
        return self.phi(features).squeeze(-1)

    def margin(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_projected = self.project(left.float())
        right_projected = self.project(right.float())
        return self._ordered(left_projected, right_projected) - self._ordered(
            right_projected, left_projected
        )


def make_head(arm: str, hidden_size: int, width: int, projection: int) -> nn.Module:
    if arm == "independent":
        return IndependentCommitHead(hidden_size, width)
    if arm == "antisymmetric":
        return AntisymmetricCommitHead(hidden_size, width, projection)
    raise AQC1Error(f"unknown AQC1 arm: {arm}")


def hidden_states(
    model: nn.Module, rows: list[list[int]], pad_token_id: int
) -> torch.Tensor:
    input_ids, attention = _pad_rows(rows, pad_token_id)
    outputs = model.text_model(
        input_ids=input_ids,
        attention_mask=attention,
        use_cache=False,
    )
    final_indices = attention.sum(dim=1) - 1
    return outputs.last_hidden_state[
        torch.arange(len(rows), device="cuda:0"), final_indices
    ]


def margins_for_batch(
    model: nn.Module,
    head: nn.Module,
    rows: list[list[int]],
    pad_token_id: int,
) -> torch.Tensor:
    hidden = hidden_states(model, rows, pad_token_id)
    if hidden.shape[0] % 2:
        raise AQC1Error("AQC1 candidate batch is not paired")
    paired = hidden.reshape(-1, 2, hidden.shape[-1])
    return head.margin(paired[:, 0], paired[:, 1])


def metrics(rows: list[dict[str, Any]], selections: dict[str, tuple[int, bool]], split: str) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["split"] != split:
            continue
        identity = row["identity_sha256"]
        if identity not in selections:
            raise AQC1Error("AQC1 selection coverage differs")
        selected, consistent = selections[identity]
        correct = [bool(candidate["correct"]) for candidate in row["candidates"]]
        disagreement = correct[0] != correct[1]
        for key in ("overall", str(row["task"])):
            bucket = buckets[key]
            bucket["total"] += 1
            bucket["idr1_correct"] += int(correct[0])
            bucket["control_correct"] += int(correct[1])
            bucket["selected_correct"] += int(correct[selected])
            bucket["oracle_correct"] += int(any(correct))
            bucket["control_commits"] += int(selected == 1)
            bucket["order_consistent"] += int(consistent)
            bucket["disagreements"] += int(disagreement)
            bucket["disagreement_selected_correct"] += int(
                disagreement and correct[selected]
            )
    result: dict[str, Any] = {}
    for key, bucket in sorted(buckets.items()):
        total = bucket["total"]
        disagreement = bucket["disagreements"]
        result[key] = {
            **dict(bucket),
            "idr1_accuracy": bucket["idr1_correct"] / total,
            "control_accuracy": bucket["control_correct"] / total,
            "selected_accuracy": bucket["selected_correct"] / total,
            "oracle_accuracy": bucket["oracle_correct"] / total,
            "control_commit_rate": bucket["control_commits"] / total,
            "order_consistency": bucket["order_consistent"] / total,
            "disagreement_selection_accuracy": (
                bucket["disagreement_selected_correct"] / disagreement
                if disagreement
                else None
            ),
        }
    return result


def evaluate(
    model: nn.Module,
    head: nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    split: str,
    maximum: int,
    batch_pairs: int,
) -> tuple[dict[str, Any], int, float]:
    selected_rows = [row for row in rows if row["split"] == split]
    selections: dict[str, tuple[int, bool]] = {}
    truncated = 0
    maximum_swap_error = 0.0
    model.eval()
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(selected_rows), batch_pairs):
            batch = selected_rows[start : start + batch_pairs]
            encoded: list[list[int]] = []
            for row in batch:
                pair, local_truncated = token_rows(tokenizer, row, maximum)
                encoded.extend(pair)
                truncated += local_truncated
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = hidden_states(model, encoded, tokenizer.pad_token_id)
                paired = hidden.reshape(-1, 2, hidden.shape[-1])
                forward = head.margin(paired[:, 0], paired[:, 1]).float()
                swapped = head.margin(paired[:, 1], paired[:, 0]).float()
            maximum_swap_error = max(
                maximum_swap_error, float((forward + swapped).abs().max().cpu())
            )
            for row, direct, reverse in zip(batch, forward.tolist(), swapped.tolist(), strict=True):
                chosen = 0 if direct >= 0 else 1
                swapped_mapped = 1 if reverse >= 0 else 0
                selections[row["identity_sha256"]] = (
                    chosen,
                    chosen == swapped_mapped,
                )
    return metrics(rows, selections, split), truncated, maximum_swap_error


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from hf_product_reasoning_eval import _load_model

    if args.output.exists():
        raise AQC1Error(f"refusing existing AQC1 output: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = load_pairs(args.pairs)
    strata = balanced_strata(rows, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    protected_before = sha256_file(args.adapter_checkpoint)
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    trainable = configure_lora_scope(model)
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = make_head(args.arm, hidden_size, args.head_width, args.projection_width).to(
        "cuda:0"
    )
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
    update = microstep = presentations = truncated = 0
    trace: list[dict[str, Any]] = []
    while update < args.updates:
        key = keys[microstep % len(keys)]
        indices = strata[key]
        row = rows[indices[positions[key] % len(indices)]]
        positions[key] += 1
        encoded, local_truncated = token_rows(tokenizer, row, args.max_sequence_length)
        truncated += local_truncated
        left_correct = bool(row["candidates"][0]["correct"])
        right_correct = bool(row["candidates"][1]["correct"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            margin = margins_for_batch(model, head, encoded, tokenizer.pad_token_id)[0].float()
            if left_correct != right_correct:
                sign = 1.0 if left_correct else -1.0
                local_loss = F.softplus(-sign * margin)
            else:
                local_loss = args.tie_loss_weight * F.smooth_l1_loss(
                    margin, torch.zeros_like(margin)
                )
            loss = local_loss / args.gradient_accumulation
        loss.backward()
        microstep += 1
        presentations += 1
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
                "update": update,
                "presentations": presentations,
                "gradient_norm": float(gradient_norm),
                "pairs_per_second": presentations / (time.monotonic() - started),
                "prompt_truncated": truncated,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)

    development, development_truncated, development_swap_error = evaluate(
        model,
        head,
        tokenizer,
        rows,
        "development",
        args.max_sequence_length,
        args.evaluation_batch_pairs,
    )
    holdout, holdout_truncated, holdout_swap_error = evaluate(
        model,
        head,
        tokenizer,
        rows,
        "holdout",
        args.max_sequence_length,
        args.evaluation_batch_pairs,
    )
    holdout_gate = {
        "overall_at_least_646": holdout["overall"]["selected_correct"] >= 646,
        "math_at_least_255": holdout["math500"]["selected_correct"] >= 255,
        "logic_at_least_349": holdout["bbh_logic"]["selected_correct"] >= 349,
        "code_at_least_24": holdout["mbpp"]["selected_correct"] >= 24,
        "exact_order_consistency": holdout["overall"]["order_consistency"] == 1.0,
        "net_positive_over_idr1": holdout["overall"]["selected_correct"]
        > holdout["overall"]["idr1_correct"],
    }
    protected_after = sha256_file(args.adapter_checkpoint)
    checkpoint = args.output / "commit.pt"
    metadata = {
        "arm": args.arm,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": protected_before,
        "adapter_metadata": adapter_metadata,
        "backbone_learning_rate": args.backbone_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "head_width": args.head_width,
        "projection_width": args.projection_width,
        "inference_fields": ["question", "candidate_a", "candidate_b"],
        "max_sequence_length": args.max_sequence_length,
        "model_loader": model_loader,
        "model_revision": args.model_revision,
        "model_root": str(args.model_source_root.resolve()),
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "seed": args.seed,
        "task_or_benchmark_label_at_inference": False,
        "updates": args.updates,
    }
    atomic_torch(
        checkpoint,
        {
            "schema": MODEL_SCHEMA,
            "metadata": metadata,
            "backbone_state": {
                name: parameter.detach().cpu() for name, parameter in trainable
            },
            "head_state": {
                name: tensor.detach().cpu() for name, tensor in head.state_dict().items()
            },
        },
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        **metadata,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "development": development,
        "holdout": holdout,
        "holdout_gate": holdout_gate,
        "holdout_gate_pass": all(holdout_gate.values()),
        "development_prompt_truncated": development_truncated,
        "holdout_prompt_truncated": holdout_truncated,
        "training_prompt_truncated": truncated,
        "development_maximum_swap_error": development_swap_error,
        "holdout_maximum_swap_error": holdout_swap_error,
        "elapsed_seconds": time.monotonic() - started,
        "gradient_accumulation": args.gradient_accumulation,
        "head_parameters": sum(parameter.numel() for parameter in head_parameters),
        "pair_presentations": presentations,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "protected_adapter_sha256_after": protected_after,
        "protected_adapter_unchanged": protected_before == protected_after,
        "strata_counts": {
            f"{task}:{outcome}": len(indices)
            for (task, outcome), indices in strata.items()
        },
        "trace": trace,
    }
    atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("antisymmetric", "independent"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", default="multimodal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--head-width", type=int, default=512)
    parser.add_argument("--projection-width", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=3072)
    parser.add_argument("--evaluation-batch-pairs", type=int, default=2)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--tie-loss-weight", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.gradient_accumulation,
        args.head_width,
        args.projection_width,
        args.max_sequence_length,
        args.evaluation_batch_pairs,
        args.log_interval,
    )
    if any(value <= 0 for value in positive):
        parser.error("AQC1 dimensions must be positive")
    if args.backbone_learning_rate <= 0 or args.head_learning_rate <= 0:
        parser.error("AQC1 learning rates must be positive")
    return args


def main() -> int:
    report = train(parse_args())
    print(
        json.dumps(
            {
                "arm": report["arm"],
                "holdout_gate_pass": report["holdout_gate_pass"],
                "holdout_selected": report["holdout"]["overall"]["selected_correct"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
