#!/usr/bin/env python3
"""Train CVG1 to hard-select one frozen whole-completion lineage."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
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

PAIR_SCHEMA = "shohin-cvg1-whole-lineage-pairs-v1"
MODEL_SCHEMA = "shohin-cvg1-completion-verifier-v1"
REPORT_SCHEMA = "shohin-cvg1-completion-verifier-report-v1"
OUTCOMES = ("base_only", "both_correct", "both_wrong", "expert_only")
SPLITS = ("train", "development", "holdout")
EXPERT_LOGIT_MARGIN = 0.10
EXPERT_MIN_PROBABILITY = 0.50


class CVG1VerifierError(RuntimeError):
    """The frozen CVG1 data, model, or training contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_outcome(base_correct: bool, expert_correct: bool) -> str:
    if base_correct and expert_correct:
        return "both_correct"
    if base_correct:
        return "base_only"
    if expert_correct:
        return "expert_only"
    return "both_wrong"


def load_pairs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CVG1VerifierError(f"missing CVG1 pair corpus: {path}")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise CVG1VerifierError(
                    f"malformed CVG1 pair row at line {line_number}"
                ) from error
            if row.get("schema") != PAIR_SCHEMA:
                raise CVG1VerifierError("CVG1 pair schema differs")
            identity = row.get("identity_sha256")
            split = row.get("split")
            candidates = row.get("candidates")
            if not isinstance(identity, str) or len(identity) != 64:
                raise CVG1VerifierError("CVG1 identity is invalid")
            if identity in identities:
                raise CVG1VerifierError("CVG1 pair identity is duplicated")
            identities.add(identity)
            if split not in SPLITS:
                raise CVG1VerifierError("CVG1 split differs")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise CVG1VerifierError("CVG1 question is empty")
            if not isinstance(row.get("task"), str) or not row["task"].strip():
                raise CVG1VerifierError("CVG1 task is empty")
            if not isinstance(candidates, list) or len(candidates) != 2:
                raise CVG1VerifierError("CVG1 requires exactly two candidates")
            by_lineage = {
                candidate.get("lineage"): candidate for candidate in candidates
            }
            if set(by_lineage) != {"base", "expert"}:
                raise CVG1VerifierError("CVG1 candidate lineages differ")
            ordered = [by_lineage["base"], by_lineage["expert"]]
            for candidate in ordered:
                if (
                    not isinstance(candidate.get("completion"), str)
                    or not candidate["completion"].strip()
                ):
                    raise CVG1VerifierError("CVG1 candidate completion is empty")
                if not isinstance(candidate.get("correct"), bool):
                    raise CVG1VerifierError("CVG1 candidate outcome is invalid")
            outcome = _expected_outcome(ordered[0]["correct"], ordered[1]["correct"])
            if row.get("outcome_class") != outcome:
                raise CVG1VerifierError("CVG1 outcome class differs")
            rows.append({**row, "candidates": ordered})
    if not rows:
        raise CVG1VerifierError("CVG1 pair corpus is empty")
    return rows


def build_balanced_strata(
    rows: list[dict[str, Any]], seed: int
) -> OrderedDict[tuple[str, str], list[int]]:
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    train_outcomes: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if row["split"] != "train":
            continue
        key = (str(row["task"]), str(row["outcome_class"]))
        strata[key].append(index)
        train_outcomes[key[1]] += 1
    if set(train_outcomes) != set(OUTCOMES):
        raise CVG1VerifierError("CVG1 train split lacks an outcome class")
    generator = random.Random(seed)
    ordered: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for key in sorted(strata):
        values = strata[key]
        generator.shuffle(values)
        ordered[key] = values
    if not ordered:
        raise CVG1VerifierError("CVG1 has no training strata")
    return ordered


def verifier_text(question: str, completion: str) -> str:
    return (
        "Judge whether the candidate solution is fully correct. Check every "
        "logical, mathematical, factual, and algorithmic step; a plausible "
        "final answer is not enough.\n\n"
        f"Problem:\n{question}\n\nCandidate solution:\n{completion}\n\n"
        "Correctness evidence:"
    )


def bounded_token_ids(
    tokenizer: Any, text: str, max_length: int
) -> tuple[list[int], bool]:
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    if len(token_ids) <= max_length:
        return token_ids, False
    if max_length < 256:
        raise CVG1VerifierError("CVG1 max sequence length is too small")
    head = max_length // 4
    return token_ids[:head] + token_ids[-(max_length - head) :], True


class CompletionVerifierHead(nn.Module):
    def __init__(self, hidden_size: int, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden.float()).squeeze(-1)


def configure_lora_scope(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    model.requires_grad_(False)
    selected: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if ".lora_a." in name or ".lora_b." in name:
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    if not selected:
        raise CVG1VerifierError("CVG1 host exposes no LoRA parameters")
    return selected


def _pad_rows(
    token_rows: list[list[int]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, token_rows))
    input_ids = torch.full(
        (len(token_rows), width),
        pad_token_id,
        dtype=torch.long,
        device="cuda:0",
    )
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(token_rows):
        input_ids[index, : len(row)] = torch.tensor(row, device="cuda:0")
        attention[index, : len(row)] = 1
    return input_ids, attention


def forward_scores(
    model: nn.Module,
    head: CompletionVerifierHead,
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


def candidate_token_rows(
    tokenizer: Any,
    row: dict[str, Any],
    max_sequence_length: int,
) -> tuple[list[list[int]], int]:
    token_rows: list[list[int]] = []
    truncated = 0
    for candidate in row["candidates"]:
        tokens, was_truncated = bounded_token_ids(
            tokenizer,
            verifier_text(row["question"], candidate["completion"]),
            max_sequence_length,
        )
        token_rows.append(tokens)
        truncated += int(was_truncated)
    return token_rows, truncated


def choose_lineage(base_logit: float, expert_logit: float) -> int:
    expert_probability = torch.sigmoid(torch.tensor(expert_logit)).item()
    if (
        expert_logit - base_logit >= EXPERT_LOGIT_MARGIN
        and expert_probability >= EXPERT_MIN_PROBABILITY
    ):
        return 1
    return 0


def _random_lineage(identity: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    return int(digest[0] & 1)


def metrics_from_scores(
    rows: list[dict[str, Any]],
    scores: dict[str, tuple[float, float]],
    *,
    split: str,
    seed: int,
) -> dict[str, Any]:
    if split not in SPLITS:
        raise CVG1VerifierError("CVG1 metric split differs")
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    probabilities: list[float] = []
    labels: list[float] = []
    for row in rows:
        if row["split"] != split:
            continue
        identity = row["identity_sha256"]
        if identity not in scores:
            raise CVG1VerifierError("CVG1 score coverage is incomplete")
        logits = scores[identity]
        selected = choose_lineage(*logits)
        random_selected = _random_lineage(identity, seed)
        correct = [bool(candidate["correct"]) for candidate in row["candidates"]]
        disagreement = correct[0] != correct[1]
        for key in ("overall", str(row["task"])):
            bucket = buckets[key]
            bucket["total"] += 1
            bucket["base_correct"] += int(correct[0])
            bucket["expert_correct"] += int(correct[1])
            bucket["selected_correct"] += int(correct[selected])
            bucket["random_correct"] += int(correct[random_selected])
            bucket["oracle_correct"] += int(any(correct))
            bucket["expert_commits"] += int(selected == 1)
            bucket["disagreements"] += int(disagreement)
            bucket["disagreement_selected_correct"] += int(
                disagreement and correct[selected]
            )
        probabilities.extend(torch.sigmoid(torch.tensor(logits)).tolist())
        labels.extend(map(float, correct))
    if "overall" not in buckets:
        raise CVG1VerifierError(f"CVG1 {split} split is empty")

    metrics: dict[str, Any] = {}
    for key, bucket in sorted(buckets.items()):
        total = bucket["total"]
        disagreements = bucket["disagreements"]
        metrics[key] = {
            **dict(bucket),
            "base_accuracy": bucket["base_correct"] / total,
            "expert_accuracy": bucket["expert_correct"] / total,
            "selected_accuracy": bucket["selected_correct"] / total,
            "random_accuracy": bucket["random_correct"] / total,
            "oracle_accuracy": bucket["oracle_correct"] / total,
            "expert_commit_rate": bucket["expert_commits"] / total,
            "disagreement_selection_accuracy": (
                bucket["disagreement_selected_correct"] / disagreements
                if disagreements
                else None
            ),
        }
    brier = sum(
        (probability - label) ** 2 for probability, label in zip(probabilities, labels)
    ) / len(labels)
    overall = metrics["overall"]
    strongest = max(overall["base_accuracy"], overall["expert_accuracy"])
    gate = {
        "brier_at_most_0_24": brier <= 0.24,
        "disagreement_selection_at_least_0_60": isinstance(
            overall["disagreement_selection_accuracy"], float
        )
        and overall["disagreement_selection_accuracy"] >= 0.60,
        "expert_commit_between_0_05_and_0_95": 0.05
        <= overall["expert_commit_rate"]
        <= 0.95,
        "selected_beats_strongest_lineage_by_0_02": overall["selected_accuracy"]
        >= strongest + 0.02,
    }
    return {
        "brier_score": brier,
        "gate": gate,
        "gate_pass": all(gate.values()),
        "metrics": metrics,
    }


def evaluate(
    model: nn.Module,
    head: CompletionVerifierHead,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    split: str,
    max_sequence_length: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, tuple[float, float]], int]:
    model.eval()
    head.eval()
    scores: dict[str, tuple[float, float]] = {}
    truncated = 0
    with torch.inference_mode():
        for row in rows:
            if row["split"] != split:
                continue
            token_rows, local_truncated = candidate_token_rows(
                tokenizer, row, max_sequence_length
            )
            truncated += local_truncated
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_scores(model, head, token_rows, tokenizer.pad_token_id)
            scores[row["identity_sha256"]] = tuple(map(float, logits.float().cpu()))
    return metrics_from_scores(rows, scores, split=split, seed=seed), scores, truncated


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
        raise CVG1VerifierError(f"refusing existing CVG1 output: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = load_pairs(args.pairs)
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
    head = CompletionVerifierHead(hidden_size, args.head_width).to("cuda:0")
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
        token_rows, local_truncated = candidate_token_rows(
            tokenizer, row, args.max_sequence_length
        )
        truncated += local_truncated
        labels = torch.tensor(
            [float(candidate["correct"]) for candidate in row["candidates"]],
            device="cuda:0",
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_scores(
                model, head, token_rows, tokenizer.pad_token_id
            ).float()
            bce = F.binary_cross_entropy_with_logits(logits, labels)
            if labels[0] != labels[1]:
                correct_index = int(labels.argmax().item())
                wrong_index = 1 - correct_index
                pairwise = F.softplus(-(logits[correct_index] - logits[wrong_index]))
            else:
                pairwise = torch.zeros((), device="cuda:0")
            loss = (bce + pairwise) / args.gradient_accumulation
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

    development, _, development_truncated = evaluate(
        model,
        head,
        tokenizer,
        rows,
        split="development",
        max_sequence_length=args.max_sequence_length,
        seed=args.seed,
    )
    holdout, _, holdout_truncated = evaluate(
        model,
        head,
        tokenizer,
        rows,
        split="holdout",
        max_sequence_length=args.max_sequence_length,
        seed=args.seed,
    )
    model_path = args.output / "verifier.pt"
    metadata = {
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": protected_sha256_before,
        "adapter_metadata": adapter_metadata,
        "backbone_learning_rate": args.backbone_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "head_width": args.head_width,
        "inference_fields": ["question", "completion"],
        "max_sequence_length": args.max_sequence_length,
        "model_loader": model_loader,
        "model_revision": args.model_revision,
        "model_root": str(args.model_source_root.resolve()),
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "seed": args.seed,
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
        "verifier": str(model_path.resolve()),
        "verifier_sha256": sha256_file(model_path),
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
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026080716)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.gradient_accumulation,
        args.head_width,
        args.max_sequence_length,
        args.log_interval,
    )
    if any(value <= 0 for value in positive):
        parser.error("CVG1 dimensions must be positive")
    if args.backbone_learning_rate <= 0 or args.head_learning_rate <= 0:
        parser.error("CVG1 learning rates must be positive")
    return args


def main() -> int:
    report = train(parse_args())
    print(
        json.dumps(
            {
                "holdout_gate_pass": report["holdout"]["gate_pass"],
                "verifier_sha256": report["verifier_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
