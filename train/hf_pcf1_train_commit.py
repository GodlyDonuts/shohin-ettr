#!/usr/bin/env python3
"""Train and apply the source-disjoint PCF1 whole-trajectory commit policy."""

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

from hf_aqc1_train_commit import (
    IndependentCommitHead,
    select_candidate,
    token_rows,
)
from hf_cvg1_completion_verifier import (
    _pad_rows,
    configure_lora_scope,
    sha256_file,
)
from hf_pcf1_evaluate import validate_adapter_trainables
from pcf1_environment import validate_environment_receipt

PAIR_SCHEMA = "shohin-pcf1-whole-trajectory-pair-v1"
MODEL_SCHEMA = "shohin-pcf1-commit-model-v1"
REPORT_SCHEMA = "shohin-pcf1-commit-training-report-v1"
PINNED_MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
SPLITS = ("calibration_train", "calibration_development")
TASKS = ("math500", "bbh_logic", "mbpp")
OUTCOMES = ("both_correct", "revision_only", "both_wrong", "unchanged_only")


class PCF1CommitError(RuntimeError):
    """The frozen PCF1 commit model, data, or custody contract differs."""


def expected_outcome(revision: bool, unchanged: bool) -> str:
    if revision and unchanged:
        return "both_correct"
    if revision:
        return "revision_only"
    if unchanged:
        return "unchanged_only"
    return "both_wrong"


def load_pairs(path: Path) -> list[dict[str, Any]]:
    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(
        word in rendered for word in ("confirmation", "holdout", "product", "public")
    ):
        raise PCF1CommitError("sealed path supplied to PCF1 commit")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise PCF1CommitError(
                    f"malformed PCF1 pair row {line_number}"
                ) from error
            identity = row.get("identity_sha256")
            split = row.get("split")
            candidates = row.get("candidates")
            if set(row) != {
                "schema",
                "identity_sha256",
                "split",
                "task",
                "question",
                "outcome_class",
                "candidates",
            }:
                raise PCF1CommitError("PCF1 pair exposes an unauthorized field")
            if row.get("schema") != PAIR_SCHEMA:
                raise PCF1CommitError("PCF1 pair schema differs")
            if split not in SPLITS:
                raise PCF1CommitError("PCF1 split differs")
            if (
                not isinstance(identity, str)
                or len(identity) != 64
                or identity in identities
            ):
                raise PCF1CommitError("PCF1 identity is invalid or duplicated")
            identities.add(identity)
            if row.get("task") not in TASKS:
                raise PCF1CommitError("PCF1 task differs")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise PCF1CommitError("PCF1 question is empty")
            if not isinstance(candidates, list) or len(candidates) != 2:
                raise PCF1CommitError("PCF1 requires two complete candidates")
            if [candidate.get("lineage") for candidate in candidates] != [
                "revision",
                "unchanged",
            ]:
                raise PCF1CommitError("PCF1 candidate order differs")
            for candidate in candidates:
                if (
                    set(candidate)
                    != {
                        "lineage",
                        "completion",
                        "correct",
                        "generated_tokens",
                        "max_token_exhausted",
                    }
                    or not isinstance(candidate.get("completion"), str)
                    or not isinstance(candidate.get("correct"), bool)
                    or isinstance(candidate.get("generated_tokens"), bool)
                    or not isinstance(candidate.get("generated_tokens"), int)
                    or candidate["generated_tokens"] < 0
                    or not isinstance(candidate.get("max_token_exhausted"), bool)
                ):
                    raise PCF1CommitError("PCF1 candidate content differs")
            outcome = expected_outcome(
                bool(candidates[0]["correct"]), bool(candidates[1]["correct"])
            )
            if row.get("outcome_class") != outcome:
                raise PCF1CommitError("PCF1 outcome binding differs")
            rows.append(row)
    split_counts = Counter(str(row["split"]) for row in rows)
    if set(split_counts) != set(SPLITS):
        raise PCF1CommitError("PCF1 split coverage differs")
    return rows


def balanced_strata(
    rows: list[dict[str, Any]], seed: int
) -> OrderedDict[tuple[str, str], list[int]]:
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    outcomes: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if row["split"] != "calibration_train":
            continue
        key = (str(row["task"]), str(row["outcome_class"]))
        strata[key].append(index)
        outcomes[key[1]] += 1
    if set(outcomes) != set(OUTCOMES):
        raise PCF1CommitError("PCF1 calibration lacks an outcome class")
    generator = random.Random(seed)
    result: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for key in sorted(strata):
        generator.shuffle(strata[key])
        result[key] = strata[key]
    return result


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
    head: IndependentCommitHead,
    rows: list[list[int]],
    pad_token_id: int,
) -> torch.Tensor:
    hidden = hidden_states(model, rows, pad_token_id)
    if hidden.shape[0] % 2:
        raise PCF1CommitError("PCF1 candidate batch is not paired")
    paired = hidden.reshape(-1, 2, hidden.shape[-1])
    return head.margin(paired[:, 0], paired[:, 1])


def summarize(
    rows: list[dict[str, Any]],
    selections: dict[str, tuple[int, bool, float]],
    split: str,
) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["split"] != split:
            continue
        identity = str(row["identity_sha256"])
        if identity not in selections:
            raise PCF1CommitError("PCF1 selection coverage differs")
        selected, consistent, _ = selections[identity]
        correct = [bool(candidate["correct"]) for candidate in row["candidates"]]
        for key in ("overall", str(row["task"])):
            bucket = buckets[key]
            bucket["total"] += 1
            bucket["revision_correct"] += int(correct[0])
            bucket["unchanged_correct"] += int(correct[1])
            bucket["selected_correct"] += int(correct[selected])
            bucket["oracle_correct"] += int(any(correct))
            bucket["unchanged_commits"] += int(selected == 1)
            bucket["order_consistent"] += int(consistent)
            bucket["revision_correct_retained"] += int(correct[0] and correct[selected])
            bucket["unchanged_correct_retained"] += int(
                correct[1] and correct[selected]
            )
    result: dict[str, Any] = {}
    for key, bucket in sorted(buckets.items()):
        total = bucket["total"]
        result[key] = {
            **dict(bucket),
            "selected_accuracy": bucket["selected_correct"] / total,
            "order_consistency": bucket["order_consistent"] / total,
            "revision_correct_retention": (
                bucket["revision_correct_retained"] / bucket["revision_correct"]
                if bucket["revision_correct"]
                else None
            ),
            "unchanged_correct_retention": (
                bucket["unchanged_correct_retained"] / bucket["unchanged_correct"]
                if bucket["unchanged_correct"]
                else None
            ),
        }
    return result


def evaluate(
    model: nn.Module,
    head: IndependentCommitHead,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    split: str,
    maximum: int,
    batch_pairs: int,
) -> tuple[dict[str, Any], int, float]:
    selected_rows = [row for row in rows if row["split"] == split]
    selections: dict[str, tuple[int, bool, float]] = {}
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
            for row, direct, reverse in zip(
                batch, forward.tolist(), swapped.tolist(), strict=True
            ):
                chosen = select_candidate(direct, row["candidates"])
                swapped_choice = select_candidate(
                    reverse, list(reversed(row["candidates"]))
                )
                consistent = chosen == 1 - swapped_choice or (
                    row["candidates"][0]["completion"]
                    == row["candidates"][1]["completion"]
                )
                identity = str(row["identity_sha256"])
                selections[identity] = (chosen, consistent, direct)
    return (
        summarize(rows, selections, split),
        truncated,
        maximum_swap_error,
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1CommitError(f"refusing existing PCF1 output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1CommitError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1CommitError(f"refusing existing PCF1 output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise PCF1CommitError(f"refusing existing PCF1 temporary output: {temporary}")
    torch.save(payload, temporary)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1CommitError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def train(args: argparse.Namespace) -> dict[str, Any]:
    environment = validate_environment_receipt(
        args.environment_receipt,
        args.environment_receipt_sha256,
        "train/hf_pcf1_train_commit.py",
    )
    from transformers import AutoTokenizer
    from hf_product_reasoning_eval import _load_model

    if args.output.exists() or args.output.is_symlink():
        raise PCF1CommitError(f"refusing existing PCF1 output: {args.output}")
    rendered_paths = "\n".join(
        f"{path}\n{path.resolve(strict=False)}"
        for path in (
            args.model_root,
            args.model_source_root,
            args.adapter_checkpoint,
            args.pairs,
            args.output,
            args.environment_receipt,
        )
    ).casefold()
    if any(
        word in rendered_paths
        for word in ("confirmation", "holdout", "product", "public")
    ):
        raise PCF1CommitError("sealed path supplied to PCF1 commit")
    if (
        args.model_revision != PINNED_MODEL_REVISION
        or args.model_loader != "multimodal"
        or args.updates != 128
        or args.gradient_accumulation != 8
        or args.head_width != 512
        or args.max_sequence_length != 3072
        or args.backbone_learning_rate != 2e-6
        or args.head_learning_rate != 2e-4
        or args.tie_loss_weight != 0.25
        or args.weight_decay != 0.01
        or args.max_gradient_norm != 1.0
        or args.seed != 2026080822
    ):
        raise PCF1CommitError("PCF1 pinned commit-training settings differ")
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
    if model_loader != "multimodal":
        raise PCF1CommitError("PCF1 multimodal loader differs")
    adapter_trainable_receipt = validate_adapter_trainables(model, adapter_metadata)
    trainable = configure_lora_scope(model)
    configured_names = sorted(name for name, _ in trainable)
    configured_parameters = sum(int(parameter.numel()) for _, parameter in trainable)
    configured_name_sha256 = hashlib.sha256(
        "\n".join(configured_names).encode()
    ).hexdigest()
    if (
        configured_parameters != adapter_trainable_receipt["trainable_parameters"]
        or configured_name_sha256
        != adapter_trainable_receipt["trainable_parameter_name_sha256"]
    ):
        raise PCF1CommitError("PCF1 commit trainable adapter scope differs")
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = IndependentCommitHead(hidden_size, args.head_width).to("cuda:0")
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
            margin = margins_for_batch(model, head, encoded, tokenizer.pad_token_id)[
                0
            ].float()
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

    development, development_truncated, development_swap = evaluate(
        model,
        head,
        tokenizer,
        rows,
        "calibration_development",
        args.max_sequence_length,
        args.evaluation_batch_pairs,
    )
    protected_after = sha256_file(args.adapter_checkpoint)
    if protected_after != protected_before:
        raise PCF1CommitError("PCF1 protected adapter changed during commit training")
    args.output.mkdir(parents=True)
    checkpoint = args.output / "commit.pt"
    metadata = {
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": protected_before,
        "adapter_metadata": adapter_metadata,
        **adapter_trainable_receipt,
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
                name: tensor.detach().cpu()
                for name, tensor in head.state_dict().items()
            },
        },
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        **metadata,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "calibration_development": development,
        "calibration_development_prompt_truncated": development_truncated,
        "training_prompt_truncated": truncated,
        "calibration_development_maximum_swap_error": development_swap,
        "elapsed_seconds": time.monotonic() - started,
        "gradient_accumulation": args.gradient_accumulation,
        "head_parameters": sum(parameter.numel() for parameter in head_parameters),
        "pair_presentations": presentations,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "protected_adapter_sha256_after": protected_after,
        "protected_adapter_unchanged": protected_before == protected_after,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        "strata_counts": {
            f"{task}:{outcome}": len(indices)
            for (task, outcome), indices in strata.items()
        },
        "trace": trace,
        "environment_verified": True,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": environment["environment_tree"]["sha256"],
    }
    atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("multimodal",), default="multimodal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--head-width", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=3072)
    parser.add_argument("--evaluation-batch-pairs", type=int, default=2)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--tie-loss-weight", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026080822)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.gradient_accumulation,
        args.head_width,
        args.max_sequence_length,
        args.evaluation_batch_pairs,
        args.log_interval,
    )
    if any(value <= 0 for value in positive):
        parser.error("PCF1 dimensions must be positive")
    if args.backbone_learning_rate <= 0 or args.head_learning_rate <= 0:
        parser.error("PCF1 learning rates must be positive")
    return args


def main() -> int:
    report = train(parse_args())
    print(
        json.dumps(
            {
                "calibration_development_selected": report["calibration_development"][
                    "overall"
                ]["selected_correct"],
                "calibration_development_order_consistency": report[
                    "calibration_development"
                ]["overall"]["order_consistency"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
