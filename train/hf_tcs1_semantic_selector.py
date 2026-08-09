#!/usr/bin/env python3
"""Train one same-host semantic scorer for the frozen TCS1 candidate pool."""

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
import torch.nn.functional as F

from hf_cvg1_completion_verifier import (
    CompletionVerifierHead,
    bounded_token_ids,
    configure_lora_scope,
    forward_scores,
    sha256_file,
    verifier_text,
)


CANDIDATE_SCHEMA = "shohin-tcs1-candidate-v1"
MODEL_SCHEMA = "shohin-tcs1-semantic-selector-v1"
REPORT_SCHEMA = "shohin-tcs1-semantic-selection-report-v1"


class TCS1SemanticError(RuntimeError):
    """TCS1 semantic scorer data, lineage, or output differs."""


def load_groups(
    path: Path, split: str, lineages: tuple[str, str, str]
) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema") != CANDIDATE_SCHEMA or row.get("split") != split:
            raise TCS1SemanticError("candidate schema or split differs")
        identity = str(row.get("identity_sha256") or "")
        groups.setdefault(identity, []).append(row)
    if not groups:
        raise TCS1SemanticError("candidate groups are empty")
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
        if tuple(str(row["lineage"]) for row in rows) != lineages:
            raise TCS1SemanticError("candidate lineage geometry differs")
        if [int(row["sample_index"]) for row in rows] != [0, 1, 2]:
            raise TCS1SemanticError("candidate positions differ")
        if len({row["task"] for row in rows}) != 1:
            raise TCS1SemanticError("candidate task binding differs")
    return groups


def balanced_strata(
    groups: OrderedDict[str, list[dict[str, Any]]], seed: int
) -> OrderedDict[tuple[str, str], list[str]]:
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for identity, rows in groups.items():
        pattern = "".join("1" if row["correct"] else "0" for row in rows)
        strata[(str(rows[0]["task"]), pattern)].append(identity)
    generator = random.Random(seed)
    result: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
    for key in sorted(strata):
        generator.shuffle(strata[key])
        result[key] = strata[key]
    if not result:
        raise TCS1SemanticError("training strata are empty")
    return result


def encode_group(
    tokenizer: Any, rows: list[dict[str, Any]], maximum: int
) -> tuple[list[list[int]], int, int]:
    encoded = []
    truncated = total_tokens = 0
    for row in rows:
        tokens, hit_limit = bounded_token_ids(
            tokenizer,
            verifier_text(str(row["question"]), str(row["completion"])),
            maximum,
        )
        encoded.append(tokens)
        truncated += int(hit_limit)
        total_tokens += len(tokens)
    return encoded, truncated, total_tokens


def selection_metrics(
    groups: OrderedDict[str, list[dict[str, Any]]],
    score_rows: list[list[float]],
    *,
    rotate_contents: bool = False,
) -> dict[str, Any]:
    if len(groups) != len(score_rows):
        raise TCS1SemanticError("score coverage differs")
    buckets: dict[str, Counter[str]] = {"overall": Counter()}
    lineages: Counter[str] = Counter()
    output_rows = []
    for (identity, rows), values in zip(groups.items(), score_rows, strict=True):
        if len(values) != 3:
            raise TCS1SemanticError("score group geometry differs")
        effective = values[-1:] + values[:-1] if rotate_contents else values
        winner = max(range(3), key=lambda index: (effective[index], -index))
        first, selected = rows[0], rows[winner]
        task = str(first["task"])
        buckets.setdefault(task, Counter())
        for bucket in (buckets["overall"], buckets[task]):
            bucket["total"] += 1
            bucket["first_correct"] += int(first["correct"])
            bucket["selected_correct"] += int(selected["correct"])
            bucket["oracle_correct"] += int(any(row["correct"] for row in rows))
            bucket["repaired"] += int(not first["correct"] and selected["correct"])
            bucket["broken"] += int(first["correct"] and not selected["correct"])
        lineages[str(selected["lineage"])] += 1
        output_rows.append(
            {
                "identity_sha256": identity,
                "task": task,
                "selected_index": winner,
                "selected_lineage": selected["lineage"],
                "selected_score": effective[winner],
                "correct": bool(selected["correct"]),
            }
        )
    return {
        "buckets": {name: dict(value) for name, value in sorted(buckets.items())},
        "selected_lineages": dict(sorted(lineages.items())),
        "rows": output_rows,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import _load_model

    if args.output.exists():
        raise TCS1SemanticError("refusing existing semantic-selector output")
    train_groups = load_groups(args.train, "train", ("base", "expert", "depth1"))
    development_groups = load_groups(
        args.development, "development", ("depth1", "depth2", "direct")
    )
    if set(train_groups) & set(development_groups):
        raise TCS1SemanticError("train and development identities overlap")
    shape_report = json.loads(args.shape_report.read_text())
    if (
        shape_report.get("schema") != "shohin-tcs1-shape-selection-report-v1"
        or shape_report.get("status") != "complete"
        or shape_report.get("development_sha256") != sha256_file(args.development)
    ):
        raise TCS1SemanticError("shape-control receipt differs")
    shape_correct = int(
        shape_report["metrics"]["buckets"]["overall"]["selected_correct"]
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    protected_before = sha256_file(args.adapter_checkpoint)
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
        weight_decay=0.01,
        fused=True,
    )
    strata = balanced_strata(train_groups, args.seed)
    positions = dict.fromkeys(strata, 0)
    keys = list(strata)
    all_labels = [
        float(row["correct"]) for rows in train_groups.values() for row in rows
    ]
    positives = sum(all_labels)
    positive_weight = torch.tensor(
        (len(all_labels) - positives) / max(positives, 1), device="cuda:0"
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    head.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = microstep = presentations = truncated = charged_tokens = 0
    trace = []
    while update < args.updates:
        key = keys[microstep % len(keys)]
        identities = strata[key]
        identity = identities[positions[key] % len(identities)]
        positions[key] += 1
        rows = train_groups[identity]
        encoded, local_truncated, local_tokens = encode_group(
            tokenizer, rows, args.max_sequence_length
        )
        labels = torch.tensor([float(row["correct"]) for row in rows], device="cuda:0")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = forward_scores(
                model, head, encoded, tokenizer.pad_token_id
            ).float()
            bce = F.binary_cross_entropy_with_logits(
                scores, labels, pos_weight=positive_weight
            )
            positives_local = torch.nonzero(labels == 1, as_tuple=False).flatten()
            negatives_local = torch.nonzero(labels == 0, as_tuple=False).flatten()
            if len(positives_local) and len(negatives_local):
                differences = (
                    scores[positives_local, None] - scores[negatives_local[None, :]]
                )
                pairwise = F.softplus(-differences).mean()
            else:
                pairwise = scores.new_zeros(())
            loss = (bce + pairwise) / args.gradient_accumulation
        loss.backward()
        microstep += 1
        presentations += 1
        truncated += local_truncated
        charged_tokens += local_tokens
        if microstep % args.gradient_accumulation:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model_parameters + head_parameters, 1.0
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
                "charged_tokens": charged_tokens,
                "gradient_norm": float(gradient_norm),
                "groups_per_second": presentations / (time.monotonic() - started),
                "truncated": truncated,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)

    evaluation_groups = (
        OrderedDict(list(development_groups.items())[: args.mechanics_groups])
        if args.mechanics_only
        else development_groups
    )
    model.eval()
    head.eval()
    score_rows: list[list[float]] = []
    evaluation_truncated = evaluation_tokens = 0
    with torch.inference_mode():
        for rows in evaluation_groups.values():
            encoded, local_truncated, local_tokens = encode_group(
                tokenizer, rows, args.max_sequence_length
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = forward_scores(
                    model, head, encoded, tokenizer.pad_token_id
                ).float()
            score_rows.append(scores.tolist())
            evaluation_truncated += local_truncated
            evaluation_tokens += local_tokens
    metrics = selection_metrics(evaluation_groups, score_rows)
    permuted = selection_metrics(evaluation_groups, score_rows, rotate_contents=True)
    protected_after = sha256_file(args.adapter_checkpoint)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "selector.pt"
    torch.save(
        {
            "schema": MODEL_SCHEMA,
            "metadata": {
                "model_root": str(args.model_source_root.resolve()),
                "model_revision": args.model_revision,
                "model_loader": model_loader,
                "adapter_checkpoint_sha256": protected_before,
                "train_sha256": sha256_file(args.train),
                "seed": args.seed,
                "updates": args.updates,
                "head_width": args.head_width,
                "max_sequence_length": args.max_sequence_length,
                "inference_fields": ["question", "candidate_completion"],
                "task_or_lineage_at_inference": False,
            },
            "backbone_state": {
                name: parameter.detach().cpu() for name, parameter in trainable
            },
            "head_state": {
                name: value.detach().cpu() for name, value in head.state_dict().items()
            },
        },
        checkpoint,
    )
    overall = metrics["buckets"]["overall"]
    permuted_overall = permuted["buckets"]["overall"]
    domain = metrics["buckets"]
    gates = (
        {"mechanics_only": True}
        if args.mechanics_only
        else {
            "absolute_603": overall["selected_correct"] >= 603,
            "beats_shape_by_10": overall["selected_correct"] >= shape_correct + 10,
            "math_nonnegative": domain["math500"]["selected_correct"] >= 223,
            "logic_nonnegative": domain["bbh_logic"]["selected_correct"] >= 349,
            "code_nonnegative": domain["mbpp"]["selected_correct"] >= 17,
            "content_permutation_drop_8": (
                overall["selected_correct"] - permuted_overall["selected_correct"] >= 8
            ),
        }
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "mechanics_only": args.mechanics_only,
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": protected_before,
        "protected_adapter_sha256_after": protected_after,
        "protected_adapter_unchanged": protected_before == protected_after,
        "train": str(args.train.resolve()),
        "train_sha256": sha256_file(args.train),
        "development": str(args.development.resolve()),
        "development_sha256": sha256_file(args.development),
        "shape_report": str(args.shape_report.resolve()),
        "shape_report_sha256": sha256_file(args.shape_report),
        "shape_correct": shape_correct,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "updates": update,
        "group_presentations": presentations,
        "charged_tokens": charged_tokens,
        "training_truncated": truncated,
        "evaluation_truncated": evaluation_truncated,
        "evaluation_tokens": evaluation_tokens,
        "model_trainable_parameters": sum(
            parameter.numel() for parameter in model_parameters
        ),
        "head_parameters": sum(parameter.numel() for parameter in head_parameters),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.monotonic() - started,
        "metrics": {key: value for key, value in metrics.items() if key != "rows"},
        "content_permutation": {
            key: value for key, value in permuted.items() if key != "rows"
        },
        "gates": gates,
        "gate_pass": all(gates.values()) and not args.mechanics_only,
        "trace": trace,
    }
    atomic_json(args.output / "report.json", report)
    atomic_json(args.output / "selections.json", {"rows": metrics["rows"]})
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", default="multimodal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--shape-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--head-width", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026080903)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--mechanics-only", action="store_true")
    parser.add_argument("--mechanics-groups", type=int, default=8)
    args = parser.parse_args()
    if (
        min(
            args.updates,
            args.gradient_accumulation,
            args.head_width,
            args.max_sequence_length,
            args.log_interval,
            args.mechanics_groups,
        )
        <= 0
    ):
        parser.error("TCS1 dimensions must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "gate_pass": report["gate_pass"],
                "mechanics_only": report["mechanics_only"],
                "metrics": report["metrics"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
