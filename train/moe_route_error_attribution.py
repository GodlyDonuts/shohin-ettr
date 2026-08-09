#!/usr/bin/env python3
"""Profile per-example OLMoE routes for completed temporal-revision arms."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-moe-route-error-attribution-v1"


class RouteAttributionError(RuntimeError):
    """The frozen route-attribution inputs or model geometry differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_route_logits(
    baseline_logits: Any,
    arm_logits: Any,
    top_k: int,
) -> dict[str, float]:
    """Compare two token-aligned router-logit matrices without retaining logits."""

    import torch

    if (
        baseline_logits.ndim != 2
        or arm_logits.shape != baseline_logits.shape
        or not 0 < top_k <= baseline_logits.shape[1]
    ):
        raise RouteAttributionError("router comparison geometry differs")
    baseline = baseline_logits.float()
    arm = arm_logits.float()
    base_probabilities = baseline.softmax(dim=-1)
    arm_probabilities = arm.softmax(dim=-1)
    base_top = base_probabilities.topk(top_k, dim=-1).indices
    arm_top = arm_probabilities.topk(top_k, dim=-1).indices
    top1_changed = (base_top[:, 0] != arm_top[:, 0]).float().mean()
    position_changed = (base_top != arm_top).float().mean()
    overlap = torch.stack(
        [
            torch.isin(base_row, arm_row).float().mean()
            for base_row, arm_row in zip(base_top, arm_top, strict=True)
        ]
    ).mean()
    base_counts = torch.bincount(
        base_top.reshape(-1), minlength=baseline.shape[1]
    ).float()
    arm_counts = torch.bincount(
        arm_top.reshape(-1), minlength=baseline.shape[1]
    ).float()
    base_counts /= base_counts.sum()
    arm_counts /= arm_counts.sum()
    return {
        "top1_change_rate": float(top1_changed),
        "topk_position_change_rate": float(position_changed),
        "topk_set_overlap": float(overlap),
        "probability_l1_mean": float(
            (base_probabilities - arm_probabilities).abs().sum(dim=-1).mean()
        ),
        "route_count_l1": float((base_counts - arm_counts).abs().sum()),
    }


def summarize_comparisons(
    rows: list[dict[str, Any]],
    arm: str,
) -> dict[str, Any]:
    metrics = (
        "top1_change_rate",
        "topk_position_change_rate",
        "topk_set_overlap",
        "probability_l1_mean",
        "route_count_l1",
    )
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        for layer in row["comparisons"][arm]:
            group = str(row["group"])
            layer_index = int(layer["layer"])
            scopes = ("all_layers", "last_four" if layer_index >= 12 else "first_twelve")
            for scope in scopes:
                for metric in metrics:
                    grouped[f"{group}:{scope}"][metric].append(float(layer[metric]))
    output: dict[str, Any] = {}
    for key, values in sorted(grouped.items()):
        output[key] = {
            "layer_observations": len(next(iter(values.values()))),
            **{
                metric: sum(observations) / len(observations)
                for metric, observations in values.items()
            },
        }
    return output


def route_path_sha256(logits: Any, top_k: int) -> str:
    import torch

    indices = logits.float().topk(top_k, dim=-1).indices.to(torch.int16).cpu()
    return hashlib.sha256(indices.numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RouteAttributionError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def _construct_model(model_root: Path, checkpoint: Path | None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoTokenizer

    from hf_product_reasoning_train import (
        ProductReasoningModel,
        load_product_backbone,
        load_trainable_checkpoint,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone, _ = load_product_backbone(
        model_root,
        "causal",
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    if checkpoint is None:
        return backbone.eval(), tokenizer
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RouteAttributionError("adapter metadata is absent")
    workspace = metadata.get("workspace_config") or {}
    model = ProductReasoningModel(
        backbone=backbone,
        arm=str(metadata["arm"]),
        lora_layers=int(metadata["lora_layers"]),
        lora_rank=int(metadata["lora_rank"]),
        lora_alpha=float(metadata["lora_alpha"]),
        workspace_width=int(workspace.get("workspace_width", 512)),
        workspace_slots=int(workspace.get("workspace_slots", 16)),
        recurrent_steps=int(workspace.get("recurrent_steps", 8)),
        unfreeze_layers=int(metadata.get("unfreeze_layers", 0)),
        lora_scope=str(metadata.get("lora_scope", "all")),
    )
    load_trainable_checkpoint(checkpoint, model)
    return model.backbone.eval(), tokenizer


def _collect_routes(
    model_root: Path,
    checkpoint: Path | None,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    import torch

    from hf_product_reasoning_eval import _render_prompt

    model, tokenizer = _construct_model(model_root, checkpoint)
    config = model.config
    top_k = int(config.num_experts_per_tok)
    num_layers = int(config.num_hidden_layers)
    collected: dict[str, list[Any]] = {}
    prompt_tokens = 0
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            prompt = _render_prompt(tokenizer, str(row["question"]), True, False)
            encoded = tokenizer(prompt, return_tensors="pt")
            output = model(
                **encoded,
                use_cache=False,
                output_router_logits=True,
                logits_to_keep=1,
            )
            router_logits = list(output.router_logits)
            if len(router_logits) != num_layers:
                raise RouteAttributionError("router layer count differs")
            identity = str(row["identity_sha256"])
            collected[identity] = [logit.detach().cpu() for logit in router_logits]
            prompt_tokens += int(encoded["attention_mask"].sum())
            if index % 8 == 0 or index == len(rows):
                print(f"[route-attribution] {index}/{len(rows)}", flush=True)
    del model
    return collected, {
        "top_k": top_k,
        "num_experts": int(config.num_experts),
        "num_layers": num_layers,
        "prompt_tokens": prompt_tokens,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    attribution = json.loads(args.error_attribution.read_text(encoding="utf-8"))
    if attribution.get("schema") != "shohin-moe-revision-error-attribution-v1":
        raise RouteAttributionError("error-attribution schema differs")
    development_rows = {
        str(row["identity_sha256"]): row
        for row in (
            json.loads(line)
            for line in args.development_data.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    board = attribution.get("route_board")
    if not isinstance(board, list) or not board:
        raise RouteAttributionError("route board is absent")
    rows = []
    for selected in board:
        identity = str(selected["identity_sha256"])
        row = dict(development_rows[identity])
        row["group"] = str(selected["group"])
        rows.append(row)
    checkpoints = {
        "unchanged": None,
        "mtr_rank8_attention": args.mtr_checkpoint,
        "rcr_router": args.rcr_checkpoint,
        "rank1_attention": args.rank1_checkpoint,
    }
    routes: dict[str, dict[str, list[Any]]] = {}
    geometries: dict[str, dict[str, Any]] = {}
    for arm, checkpoint in checkpoints.items():
        print(f"[route-attribution] loading {arm}", flush=True)
        routes[arm], geometries[arm] = _collect_routes(
            args.model_root, checkpoint, rows
        )
    if len({json.dumps(value, sort_keys=True) for value in geometries.values()}) != 1:
        raise RouteAttributionError("arm routing geometry differs")
    top_k = geometries["unchanged"]["top_k"]
    row_reports: list[dict[str, Any]] = []
    for row in rows:
        identity = str(row["identity_sha256"])
        baseline = routes["unchanged"][identity]
        report = {
            "identity_sha256": identity,
            "task": row["task"],
            "group": row["group"],
            "route_path_sha256": {
                arm: [route_path_sha256(logit, top_k) for logit in arm_routes[identity]]
                for arm, arm_routes in routes.items()
            },
            "comparisons": {},
        }
        for arm in ("mtr_rank8_attention", "rcr_router", "rank1_attention"):
            report["comparisons"][arm] = [
                {"layer": layer, **compare_route_logits(base, candidate, top_k)}
                for layer, (base, candidate) in enumerate(
                    zip(baseline, routes[arm][identity], strict=True)
                )
            ]
        row_reports.append(report)
    output = {
        "schema": SCHEMA,
        "status": "complete",
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_config_sha256": sha256_file(args.model_source_root / "config.json"),
        "error_attribution_sha256": sha256_file(args.error_attribution),
        "development_data_sha256": sha256_file(args.development_data),
        "checkpoint_sha256": {
            key: sha256_file(value) if value is not None else None
            for key, value in checkpoints.items()
        },
        "rows": len(rows),
        "geometry": geometries["unchanged"],
        "summaries": {
            arm: summarize_comparisons(row_reports, arm)
            for arm in ("mtr_rank8_attention", "rcr_router", "rank1_attention")
        },
        "row_reports": row_reports,
    }
    _atomic_json(args.output, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--error-attribution", type=Path, required=True)
    parser.add_argument("--mtr-checkpoint", type=Path, required=True)
    parser.add_argument("--rcr-checkpoint", type=Path, required=True)
    parser.add_argument("--rank1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"rows": report["rows"], "summaries": report["summaries"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
