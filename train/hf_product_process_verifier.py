#!/usr/bin/env python3
"""Train and apply a separate end-to-end candidate process verifier."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
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

from product_candidate_reranker import (
    CandidateReranker,
    FEATURE_NAMES,
    SCHEMA as SHAPE_SCHEMA,
    feature_vector,
)
from select_product_self_consistency import choose_modal_candidate


MODEL_SCHEMA = "shohin-product-process-verifier-v1"
REPORT_SCHEMA = "shohin-product-process-verifier-selection-v1"
SPLIT_NAMES = ("final", "dev", "train")


class ProcessVerifierError(RuntimeError):
    """The process-verifier data, model, or split contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_split(identity: str, seed: int) -> str:
    """Reserve two disjoint identity folds for development and final scoring."""

    digest = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    if bucket == 0:
        return "final"
    if bucket == 1:
        return "dev"
    return "train"


def load_grouped_candidates(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProcessVerifierError("candidate JSONL is malformed") from exc
            required = (
                "identity_sha256",
                "task",
                "question",
                "completion",
                "sample_index",
                "correct",
            )
            if any(key not in row for key in required):
                raise ProcessVerifierError("candidate row schema differs")
            identity = str(row["identity_sha256"])
            grouped.setdefault(identity, []).append(row)
    if not grouped:
        raise ProcessVerifierError("candidate source is empty")
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
        if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
            raise ProcessVerifierError("candidate sample indices differ")
        if len({str(row["question"]) for row in rows}) != 1:
            raise ProcessVerifierError("candidate questions differ within an identity")
        if len({str(row["task"]) for row in rows}) != 1:
            raise ProcessVerifierError("candidate tasks differ within an identity")
    return grouped


def _pair_rank(seed: int, identity: str, positive: int, negative: int) -> str:
    return hashlib.sha256(
        f"{seed}\0{identity}\0{positive}\0{negative}".encode()
    ).hexdigest()


def build_balanced_pairs(
    grouped: OrderedDict[str, list[dict[str, Any]]],
    *,
    seed: int,
    pairs_per_prompt: int,
) -> dict[str, list[tuple[str, int, int]]]:
    """Build deterministic within-prompt pairs grouped by task."""

    if pairs_per_prompt <= 0:
        raise ProcessVerifierError("pairs per prompt must be positive")
    by_task: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for identity, rows in grouped.items():
        if identity_split(identity, seed) != "train":
            continue
        positives = [int(row["sample_index"]) for row in rows if bool(row["correct"])]
        negatives = [
            int(row["sample_index"]) for row in rows if not bool(row["correct"])
        ]
        ranked = sorted(
            (
                _pair_rank(seed, identity, positive, negative),
                positive,
                negative,
            )
            for positive in positives
            for negative in negatives
        )
        task = str(rows[0]["task"])
        by_task[task].extend(
            (identity, positive, negative)
            for _, positive, negative in ranked[:pairs_per_prompt]
        )
    if not by_task or any(not pairs for pairs in by_task.values()):
        raise ProcessVerifierError("training split has no balanced candidate pairs")
    generator = random.Random(seed)
    for pairs in by_task.values():
        generator.shuffle(pairs)
    return dict(sorted(by_task.items()))


def verifier_text(question: str, completion: str) -> str:
    return (
        "Act as a strict process verifier. Check the candidate's logical, "
        "mathematical, and factual steps against the problem. A plausible final "
        "answer is insufficient when its derivation is invalid.\n\n"
        f"Problem:\n{question}\n\nCandidate solution:\n{completion}\n\n"
        "Assess whether the complete solution is correct.\nVerifier decision:"
    )


def bounded_token_ids(
    tokenizer: Any, text: str, max_length: int
) -> tuple[list[int], bool]:
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    if len(token_ids) <= max_length:
        return token_ids, False
    if max_length < 128:
        raise ProcessVerifierError("max sequence length is too small")
    head = max_length // 4
    return token_ids[:head] + token_ids[-(max_length - head) :], True


class ProcessVerifierHead(nn.Module):
    """Read a contextual verifier state and label-blind completion-shape features."""

    def __init__(self, hidden_size: int, shape_size: int, width: int) -> None:
        super().__init__()
        shape_width = max(width // 4, 32)
        self.hidden = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, width),
            nn.GELU(),
        )
        self.shape = nn.Sequential(
            nn.LayerNorm(shape_size),
            nn.Linear(shape_size, shape_width),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(width + shape_width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, hidden: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        return self.output(
            torch.cat((self.hidden(hidden.float()), self.shape(shape.float())), dim=-1)
        ).squeeze(-1)


def configure_train_scope(
    model: nn.Module, scope: str
) -> list[tuple[str, nn.Parameter]]:
    if scope not in {"leader", "lora"}:
        raise ProcessVerifierError("verifier train scope differs")
    if scope == "lora":
        model.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if ".lora_a." in name or ".lora_b." in name:
                parameter.requires_grad_(True)
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not selected:
        raise ProcessVerifierError("verifier has no trainable backbone parameters")
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
    head: ProcessVerifierHead,
    token_rows: list[list[int]],
    shape_rows: list[list[float]],
    pad_token_id: int,
) -> torch.Tensor:
    input_ids, attention = _pad_rows(token_rows, pad_token_id)
    outputs = model.text_model(
        input_ids=input_ids,
        attention_mask=attention,
        use_cache=False,
    )
    lengths = attention.sum(dim=1) - 1
    final_hidden = outputs.last_hidden_state[
        torch.arange(len(token_rows), device="cuda:0"), lengths
    ]
    shape = torch.tensor(shape_rows, dtype=torch.float32, device="cuda:0")
    return head(final_hidden, shape)


def _candidate_inputs(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    group: list[dict[str, Any]],
    max_sequence_length: int,
) -> tuple[list[list[int]], list[list[float]], int]:
    token_rows: list[list[int]] = []
    shape_rows: list[list[float]] = []
    truncated = 0
    for row in rows:
        tokens, was_truncated = bounded_token_ids(
            tokenizer,
            verifier_text(str(row["question"]), str(row["completion"])),
            max_sequence_length,
        )
        token_rows.append(tokens)
        shape_rows.append(feature_vector(row, group))
        truncated += int(was_truncated)
    return token_rows, shape_rows, truncated


def _load_shape_model(path: Path) -> tuple[CandidateReranker, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SHAPE_SCHEMA:
        raise ProcessVerifierError("shape reranker schema differs")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ProcessVerifierError("shape reranker features differ")
    model = CandidateReranker(len(FEATURE_NAMES), int(payload["hidden_size"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _shape_scores(
    rows: list[dict[str, Any]],
    model: CandidateReranker,
    payload: dict[str, Any],
) -> torch.Tensor:
    features = torch.tensor(
        [feature_vector(row, rows) for row in rows], dtype=torch.float32
    )
    normalized = (features - payload["feature_mean"]) / payload["feature_scale"]
    with torch.inference_mode():
        return model(normalized)


def evaluate_split(
    model: nn.Module,
    head: ProcessVerifierHead,
    tokenizer: Any,
    grouped: OrderedDict[str, list[dict[str, Any]]],
    shape_model: CandidateReranker,
    shape_payload: dict[str, Any],
    *,
    split: str,
    seed: int,
    max_sequence_length: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if split not in (*SPLIT_NAMES, "all"):
        raise ProcessVerifierError("evaluation split differs")
    model.eval()
    head.eval()
    counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    selections: list[dict[str, Any]] = []
    truncated = 0
    with torch.inference_mode():
        for identity, rows in grouped.items():
            if split != "all" and identity_split(identity, seed) != split:
                continue
            scores: list[torch.Tensor] = []
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset : offset + batch_size]
                tokens, shapes, local_truncated = _candidate_inputs(
                    tokenizer, batch, rows, max_sequence_length
                )
                truncated += local_truncated
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    scores.append(
                        forward_scores(
                            model,
                            head,
                            tokens,
                            shapes,
                            tokenizer.pad_token_id,
                        )
                        .float()
                        .cpu()
                    )
            process = torch.cat(scores)
            shape = _shape_scores(rows, shape_model, shape_payload)
            for index, row in enumerate(rows):
                if not str(row.get("completion") or "").strip():
                    process[index] = -1e9
                    shape[index] = -1e9
            process_winner = int(process.argmax().item())
            shape_winner = int(shape.argmax().item())
            modal = choose_modal_candidate(rows)
            task = str(rows[0]["task"])
            for key in ("overall", task):
                bucket = counters[key]
                bucket["total"] += 1
                bucket["first_correct"] += int(bool(rows[0]["correct"]))
                bucket["modal_correct"] += int(bool(modal["correct"]))
                bucket["shape_correct"] += int(bool(rows[shape_winner]["correct"]))
                bucket["process_correct"] += int(bool(rows[process_winner]["correct"]))
                bucket["oracle_correct"] += int(
                    any(bool(row["correct"]) for row in rows)
                )
            selections.append(
                {
                    "identity_sha256": identity,
                    "task": task,
                    "process_sample_index": process_winner,
                    "process_correct": bool(rows[process_winner]["correct"]),
                    "shape_sample_index": shape_winner,
                    "shape_correct": bool(rows[shape_winner]["correct"]),
                    "process_scores": process.tolist(),
                }
            )
    metrics: dict[str, Any] = {}
    for key, values in sorted(counters.items()):
        total = values["total"]
        metrics[key] = {
            **dict(values),
            "first_accuracy": values["first_correct"] / total,
            "modal_accuracy": values["modal_correct"] / total,
            "shape_accuracy": values["shape_correct"] / total,
            "process_accuracy": values["process_correct"] / total,
            "oracle_accuracy": values["oracle_correct"] / total,
            "process_minus_shape": (values["process_correct"] - values["shape_correct"])
            / total,
        }
    if "overall" not in metrics:
        raise ProcessVerifierError(f"{split} split is empty")
    metrics["prompt_truncated"] = truncated
    return metrics, selections


def _checkpoint_payload(
    model: nn.Module,
    head: ProcessVerifierHead,
    trainable_names: list[str],
    metadata: dict[str, Any],
    update: int,
    dev_metrics: dict[str, Any],
) -> dict[str, Any]:
    named = dict(model.named_parameters())
    return {
        "schema": MODEL_SCHEMA,
        "update": update,
        "metadata": metadata,
        "backbone_state": {
            name: named[name].detach().cpu() for name in trainable_names
        },
        "head_state": {
            name: tensor.detach().cpu() for name, tensor in head.state_dict().items()
        },
        "dev_metrics": dev_metrics,
    }


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _restore_verifier(
    path: Path,
    model: nn.Module,
    head: ProcessVerifierHead,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != MODEL_SCHEMA:
        raise ProcessVerifierError("process verifier schema differs")
    current = dict(model.named_parameters())
    state = payload.get("backbone_state")
    if not isinstance(state, dict) or any(name not in current for name in state):
        raise ProcessVerifierError("process verifier backbone state differs")
    with torch.no_grad():
        for name, tensor in state.items():
            current[name].copy_(
                tensor.to(device=current[name].device, dtype=current[name].dtype)
            )
    head.load_state_dict(payload["head_state"])
    return payload


def train(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import _load_model

    if args.output.exists():
        raise ProcessVerifierError(f"refusing existing output: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    grouped = load_grouped_candidates(args.candidates)
    pairs = build_balanced_pairs(
        grouped, seed=args.seed, pairs_per_prompt=args.pairs_per_prompt
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    trainable = configure_train_scope(model, args.train_scope)
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = ProcessVerifierHead(hidden_size, len(FEATURE_NAMES), args.head_width).to(
        "cuda:0"
    )
    shape_model, shape_payload = _load_shape_model(args.shape_model)
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
    split_counts: dict[str, int] = defaultdict(int)
    for identity in grouped:
        split_counts[identity_split(identity, args.seed)] += 1
    metadata = {
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_update": adapter_metadata.get("update") if adapter_metadata else None,
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": sha256_file(args.candidates),
        "shape_model": str(args.shape_model.resolve()),
        "shape_model_sha256": sha256_file(args.shape_model),
        "seed": args.seed,
        "split_counts": dict(split_counts),
        "pairs_per_prompt": args.pairs_per_prompt,
        "pair_counts": {task: len(rows) for task, rows in pairs.items()},
        "train_scope": args.train_scope,
        "trainable_backbone_parameters": sum(
            parameter.numel() for parameter in model_parameters
        ),
        "head_parameters": sum(parameter.numel() for parameter in head_parameters),
        "trainable_parameter_names": [name for name, _ in trainable],
        "head_width": args.head_width,
        "max_sequence_length": args.max_sequence_length,
    }
    tasks = list(pairs)
    positions = dict.fromkeys(tasks, 0)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    head.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = 0
    microstep = 0
    truncated = 0
    seen_pairs = 0
    loss_sum = 0.0
    best_correct = -1
    best_update = 0
    stale_evaluations = 0
    trace: list[dict[str, Any]] = []
    model_path = args.output / "model.pt"
    while update < args.updates:
        task = tasks[microstep % len(tasks)]
        pair_rows = pairs[task]
        identity, positive_index, negative_index = pair_rows[
            positions[task] % len(pair_rows)
        ]
        positions[task] += 1
        group = grouped[identity]
        batch = [group[positive_index], group[negative_index]]
        token_rows, shape_rows, local_truncated = _candidate_inputs(
            tokenizer, batch, group, args.max_sequence_length
        )
        truncated += local_truncated
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = forward_scores(
                model, head, token_rows, shape_rows, tokenizer.pad_token_id
            ).float()
            pairwise = F.softplus(-(scores[0] - scores[1]))
            bce = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    scores[0], torch.ones_like(scores[0])
                )
                + F.binary_cross_entropy_with_logits(
                    scores[1], torch.zeros_like(scores[1])
                )
            )
            loss = (pairwise + args.bce_weight * bce) / args.gradient_accumulation
        loss.backward()
        loss_sum += float(loss.detach()) * args.gradient_accumulation
        seen_pairs += 1
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
            elapsed = time.monotonic() - started
            event = {
                "update": update,
                "mean_pair_loss": loss_sum / max(seen_pairs, 1),
                "gradient_norm": float(gradient_norm),
                "pairs": seen_pairs,
                "pairs_per_second": seen_pairs / elapsed,
                "prompt_truncated": truncated,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
        if update % args.eval_interval and update != args.updates:
            continue
        dev_metrics, _ = evaluate_split(
            model,
            head,
            tokenizer,
            grouped,
            shape_model,
            shape_payload,
            split="dev",
            seed=args.seed,
            max_sequence_length=args.max_sequence_length,
            batch_size=args.eval_batch_size,
        )
        correct = int(dev_metrics["overall"]["process_correct"])
        print(
            json.dumps({"update": update, "dev": dev_metrics}, sort_keys=True),
            flush=True,
        )
        if correct > best_correct:
            best_correct = correct
            best_update = update
            stale_evaluations = 0
            _atomic_torch_save(
                model_path,
                _checkpoint_payload(
                    model,
                    head,
                    [name for name, _ in trainable],
                    metadata,
                    update,
                    dev_metrics,
                ),
            )
        else:
            stale_evaluations += 1
        model.train()
        head.train()
        if update >= args.min_updates and stale_evaluations >= args.early_stop_patience:
            break
    best_payload = _restore_verifier(model_path, model, head)
    final_metrics, final_selections = evaluate_split(
        model,
        head,
        tokenizer,
        grouped,
        shape_model,
        shape_payload,
        split="final",
        seed=args.seed,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.eval_batch_size,
    )
    dev_metrics = best_payload["dev_metrics"]
    elapsed = time.monotonic() - started
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        **metadata,
        "updates_completed": update,
        "best_update": best_update,
        "gradient_accumulation": args.gradient_accumulation,
        "backbone_learning_rate": args.backbone_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "bce_weight": args.bce_weight,
        "seen_pairs": seen_pairs,
        "training_prompt_truncated": truncated,
        "elapsed_seconds": elapsed,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "dev_metrics": dev_metrics,
        "final_metrics": final_metrics,
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "trace": trace,
        "final_selections": final_selections,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def select(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import _load_model

    for path in (args.output, args.report):
        if path.exists():
            raise ProcessVerifierError(f"refusing existing output: {path}")
    verifier = torch.load(args.verifier, map_location="cpu", weights_only=False)
    if verifier.get("schema") != MODEL_SCHEMA:
        raise ProcessVerifierError("process verifier schema differs")
    metadata = verifier["metadata"]
    if metadata["adapter_sha256"] != sha256_file(args.adapter_checkpoint):
        raise ProcessVerifierError("process verifier generator adapter differs")
    grouped = load_grouped_candidates(args.candidates)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, _, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    head = ProcessVerifierHead(
        int(model.text_model.embed_tokens.embedding_dim),
        len(FEATURE_NAMES),
        int(metadata["head_width"]),
    ).to("cuda:0")
    _restore_verifier(args.verifier, model, head)
    shape_model, shape_payload = _load_shape_model(args.shape_model)
    model.eval()
    head.eval()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    metrics, selections = evaluate_split(
        model,
        head,
        tokenizer,
        grouped,
        shape_model,
        shape_payload,
        split="all",
        seed=args.seed,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.eval_batch_size,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_sha256": sha256_file(args.adapter_checkpoint),
        "verifier": str(args.verifier.resolve()),
        "verifier_sha256": sha256_file(args.verifier),
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": sha256_file(args.candidates),
        "shape_model": str(args.shape_model.resolve()),
        "shape_model_sha256": sha256_file(args.shape_model),
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "selector_reads_gold": False,
        "selections": selections,
    }
    _atomic_json(args.report, report)
    _atomic_json(args.output, {"schema": REPORT_SCHEMA, "selections": selections})
    return report


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", default="auto")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--shape-model", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    _common_parser(train_parser)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--updates", type=int, default=300)
    train_parser.add_argument("--min-updates", type=int, default=100)
    train_parser.add_argument("--gradient-accumulation", type=int, default=8)
    train_parser.add_argument("--pairs-per-prompt", type=int, default=2)
    train_parser.add_argument("--head-width", type=int, default=512)
    train_parser.add_argument(
        "--train-scope", choices=("leader", "lora"), default="leader"
    )
    train_parser.add_argument("--backbone-learning-rate", type=float, default=2e-6)
    train_parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--bce-weight", type=float, default=0.25)
    train_parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    train_parser.add_argument("--eval-interval", type=int, default=50)
    train_parser.add_argument("--early-stop-patience", type=int, default=3)
    train_parser.add_argument("--log-interval", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=20260809)
    select_parser = subparsers.add_parser("select")
    _common_parser(select_parser)
    select_parser.add_argument("--verifier", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--report", type=Path, required=True)
    select_parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    positive = (args.max_sequence_length, args.eval_batch_size)
    if args.command == "train":
        positive += (
            args.updates,
            args.min_updates,
            args.gradient_accumulation,
            args.pairs_per_prompt,
            args.head_width,
            args.backbone_learning_rate,
            args.head_learning_rate,
            args.weight_decay,
            args.bce_weight,
            args.max_gradient_norm,
            args.eval_interval,
            args.early_stop_patience,
            args.log_interval,
        )
    if any(value <= 0 for value in positive):
        parser.error("process verifier dimensions must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = train(args) if args.command == "train" else select(args)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
