#!/usr/bin/env python3
"""Measure frozen OLMoE routing on MTR1 revision prompts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from hf_idr1_evaluate_reviser import load_rows
from hf_product_reasoning_eval import _load_model, _render_prompt
from hf_vcr1_evaluate_reviser import sha256_file


SCHEMA = "shohin-mtr1-router-accounting-v1"


class MTR1RouterError(RuntimeError):
    """The frozen MTR1 accounting contract differs."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise MTR1RouterError(f"refusing to replace report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _tensor_name_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def summarize_router_logits(
    router_logits: tuple[Any, ...],
    attention_mask: Any,
    top_k: int,
    normalize_topk: bool = False,
) -> list[dict[str, Any]]:
    """Return exact selected-expert accounting for each MoE layer."""

    import torch

    valid = attention_mask.reshape(-1).bool()
    if not router_logits or not bool(valid.any()) or top_k <= 0:
        raise MTR1RouterError("router accounting geometry is empty")
    layers: list[dict[str, Any]] = []
    for layer_index, logits in enumerate(router_logits):
        if logits.ndim != 2 or logits.shape[0] != valid.numel():
            raise MTR1RouterError("router logits do not match the token mask")
        num_experts = int(logits.shape[1])
        if top_k > num_experts:
            raise MTR1RouterError("router top-k exceeds the expert count")
        selected_logits = logits[valid].float()
        probabilities = selected_logits.softmax(dim=-1)
        top_values, top_indices = probabilities.topk(top_k, dim=-1)
        if normalize_topk:
            top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        counts = torch.bincount(top_indices.reshape(-1), minlength=num_experts)
        weights = torch.zeros(num_experts, device=logits.device, dtype=torch.float32)
        weights.scatter_add_(0, top_indices.reshape(-1), top_values.reshape(-1))
        count_share = counts.float() / counts.sum()
        nonzero = count_share[count_share > 0]
        entropy = float(-(nonzero * nonzero.log()).sum())
        normalized_entropy = entropy / math.log(num_experts)
        layers.append(
            {
                "layer": layer_index,
                "tokens": int(valid.sum()),
                "assignments": int(counts.sum()),
                "num_experts": num_experts,
                "active_experts": int((counts > 0).sum()),
                "count_share_min": float(count_share.min()),
                "count_share_max": float(count_share.max()),
                "count_share_entropy_normalized": normalized_entropy,
                "expert_counts": counts.cpu().tolist(),
                "expert_weight_share": weights.div(weights.sum()).cpu().tolist(),
            }
        )
    return layers


def _parameter_accounting(backbone: Any, top_k: int, num_experts: int) -> dict[str, Any]:
    named = list(backbone.named_parameters())
    expert = [(name, parameter) for name, parameter in named if ".experts." in name]
    router = [
        (name, parameter)
        for name, parameter in named
        if ".gate." in name or "router" in name.casefold()
    ]
    total = sum(parameter.numel() for _, parameter in named)
    expert_total = sum(parameter.numel() for _, parameter in expert)
    router_total = sum(parameter.numel() for _, parameter in router)
    shared = total - expert_total
    active_estimate = shared + expert_total * top_k // num_experts
    return {
        "total_parameters": total,
        "shared_parameters": shared,
        "expert_parameters": expert_total,
        "router_parameters": router_total,
        "active_parameters_estimate": active_estimate,
        "active_fraction_estimate": active_estimate / total,
        "expert_parameter_name_sha256": _tensor_name_sha256([name for name, _ in expert]),
        "router_parameter_name_sha256": _tensor_name_sha256([name for name, _ in router]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if args.output.exists():
        raise MTR1RouterError("router report already exists")
    data_report = json.loads(args.data_report.read_text())
    expected = data_report.get("outputs", {}).get(args.split, {})
    if (
        data_report.get("schema") != "shohin-idr1-revision-data-report-v1"
        or expected.get("sha256") != sha256_file(args.data)
        or Path(expected.get("path", "")).resolve() != args.data.resolve()
    ):
        raise MTR1RouterError("revision data receipt differs")
    all_rows = load_rows(args.data, args.split)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_task.setdefault(str(row["task"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for task in sorted(by_task):
        ranked = sorted(by_task[task], key=lambda row: row["identity_sha256"])
        if len(ranked) < args.rows_per_task:
            raise MTR1RouterError("router board is smaller than the frozen sample")
        rows.extend(ranked[: args.rows_per_task])

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(
        args.model_root, args.adapter_checkpoint, "causal"
    )
    backbone = getattr(model, "backbone", model)
    config = backbone.config
    top_k = int(config.num_experts_per_tok)
    num_experts = int(config.num_experts)
    normalize_topk = bool(config.norm_topk_prob)
    trainable_state_names: list[str] = []
    if args.adapter_checkpoint is not None:
        payload = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("trainable_state")
        if not isinstance(state, dict) or not state:
            raise MTR1RouterError("adapter trainable state is absent")
        trainable_state_names = sorted(state)
        if any(
            token in name.casefold()
            for name in trainable_state_names
            for token in ("experts", ".mlp.", ".gate.", "router")
        ):
            raise MTR1RouterError("adapter contains a protected router/expert tensor")

    aggregate: list[Counter[int]] = [Counter() for _ in range(config.num_hidden_layers)]
    weight_aggregate: list[list[float]] = [
        [0.0] * num_experts for _ in range(config.num_hidden_layers)
    ]
    tokens_by_layer = [0] * config.num_hidden_layers
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for index in range(0, len(rows), args.batch_size):
        batch = rows[index : index + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, str(row["question"]), True, False)
            for row in batch
        ]
        encoded = tokenizer(rendered, padding=True, return_tensors="pt")
        encoded = {name: value.to("cuda:0") for name, value in encoded.items()}
        with torch.inference_mode():
            output = backbone(
                **encoded,
                use_cache=False,
                output_router_logits=True,
                logits_to_keep=1,
            )
        summaries = summarize_router_logits(
            output.router_logits,
            encoded["attention_mask"],
            top_k,
            normalize_topk,
        )
        for summary in summaries:
            layer = int(summary["layer"])
            aggregate[layer].update(
                {expert: count for expert, count in enumerate(summary["expert_counts"])}
            )
            for expert, share in enumerate(summary["expert_weight_share"]):
                weight_aggregate[layer][expert] += float(share) * int(summary["tokens"])
            tokens_by_layer[layer] += int(summary["tokens"])
        if (index + len(batch)) % 32 == 0 or index + len(batch) == len(rows):
            print(f"[mtr1-router] {index + len(batch)}/{len(rows)}", flush=True)

    elapsed = time.monotonic() - started
    layers: list[dict[str, Any]] = []
    for layer, counts_counter in enumerate(aggregate):
        counts = [counts_counter[expert] for expert in range(num_experts)]
        assignments = sum(counts)
        shares = [count / assignments for count in counts]
        nonzero = [share for share in shares if share > 0]
        layers.append(
            {
                "layer": layer,
                "tokens": tokens_by_layer[layer],
                "assignments": assignments,
                "active_experts": sum(count > 0 for count in counts),
                "count_share_min": min(shares),
                "count_share_max": max(shares),
                "count_share_entropy_normalized": -sum(
                    share * math.log(share) for share in nonzero
                )
                / math.log(num_experts),
                "expert_counts": counts,
                "expert_weight_share": [
                    weight / tokens_by_layer[layer]
                    for weight in weight_aggregate[layer]
                ],
            }
        )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()) if args.adapter_checkpoint else None,
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint) if args.adapter_checkpoint else None,
        "adapter_metadata": metadata,
        "adapter_trainable_names": len(trainable_state_names),
        "adapter_trainable_name_sha256": _tensor_name_sha256(trainable_state_names),
        "protected_router_expert_trainables": 0,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "split": args.split,
        "selection": "lowest_identity_sha256_per_task",
        "rows_per_task": args.rows_per_task,
        "rows": len(rows),
        "batch_size": args.batch_size,
        "prompt_tokens": sum(tokens_by_layer) // len(tokens_by_layer),
        "elapsed_seconds": elapsed,
        "prompt_tokens_per_second": (sum(tokens_by_layer) // len(tokens_by_layer)) / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "num_experts": num_experts,
        "experts_per_token": top_k,
        "normalize_topk_prob": normalize_topk,
        "parameter_accounting": _parameter_accounting(backbone, top_k, num_experts),
        "layers": layers,
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("unchanged", "treatment"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--rows-per-task", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.arm == "treatment") != (args.adapter_checkpoint is not None):
        parser.error("treatment requires exactly one adapter checkpoint")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
