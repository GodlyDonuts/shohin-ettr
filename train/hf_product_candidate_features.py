#!/usr/bin/env python3
"""Extract frozen late-layer features for autonomous reasoning candidates."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from product_candidate_reranker import FEATURE_NAMES, feature_vector


SCHEMA = "shohin-product-candidate-features-v1"
DEFAULT_LAYER_OFFSETS = (-1, -2, -4, -8)
POOLING = ("last", "tail_mean", "completion_mean")


class CandidateFeatureError(RuntimeError):
    """Candidate feature extraction violated its immutable contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_grouped(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get("identity_sha256") or "")
            if not identity:
                raise CandidateFeatureError("candidate identity is missing")
            grouped.setdefault(identity, []).append(row)
    if not grouped:
        raise CandidateFeatureError("candidate source is empty")
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
        if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
            raise CandidateFeatureError("candidate sample indices differ")
    return grouped


def bounded_token_rows(
    prompt_ids: list[int], completion_ids: list[int], max_length: int
) -> tuple[list[int], int, bool]:
    """Keep the system prefix, complete question tail, and completion tail."""

    if max_length < 128:
        raise CandidateFeatureError("max sequence length is too small")
    if len(prompt_ids) + len(completion_ids) <= max_length:
        return prompt_ids + completion_ids, len(prompt_ids), False
    completion_budget = min(len(completion_ids), max_length - 96)
    kept_completion = completion_ids[-completion_budget:]
    prompt_budget = max_length - len(kept_completion)
    if len(prompt_ids) <= prompt_budget:
        kept_prompt = prompt_ids
    else:
        head = min(64, prompt_budget // 2)
        kept_prompt = prompt_ids[:head] + prompt_ids[-(prompt_budget - head) :]
    return kept_prompt + kept_completion, len(kept_prompt), True


def pool_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    layer_offsets: tuple[int, ...],
    lengths: list[int],
    completion_starts: list[int],
    tail_tokens: int,
) -> torch.Tensor:
    pooled_rows: list[torch.Tensor] = []
    for batch_index, (length, completion_start) in enumerate(
        zip(lengths, completion_starts, strict=True)
    ):
        start = min(completion_start, length - 1)
        row_parts: list[torch.Tensor] = []
        for offset in layer_offsets:
            layer = hidden_states[offset][batch_index]
            row_parts.extend(
                (
                    layer[length - 1],
                    layer[max(start, length - tail_tokens) : length].mean(dim=0),
                    layer[start:length].mean(dim=0),
                )
            )
        pooled_rows.append(torch.cat(row_parts))
    return torch.stack(pooled_rows)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CandidateFeatureError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CandidateFeatureError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import _load_model, _render_prompt

    grouped = load_grouped(args.candidates)
    identities = list(grouped)
    if args.skip < 0 or args.count <= 0 or args.skip + args.count > len(identities):
        raise CandidateFeatureError("requested identity slice is outside candidate source")
    selected = identities[args.skip : args.skip + args.count]
    rows = [row for identity in selected for row in grouped[identity]]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    if not hasattr(model, "text_model"):
        raise CandidateFeatureError("adapter exposes no text model")
    text_model = model.text_model
    text_model.eval()
    layer_offsets = tuple(args.layer_offset)
    if not layer_offsets or any(offset >= 0 for offset in layer_offsets):
        raise CandidateFeatureError("layer offsets must be negative")

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    feature_batches: list[torch.Tensor] = []
    shape_rows: list[list[float]] = []
    metadata: list[dict[str, Any]] = []
    truncated = 0
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        token_rows: list[list[int]] = []
        completion_starts: list[int] = []
        batch_truncated: list[bool] = []
        for row in batch:
            rendered = _render_prompt(
                tokenizer, str(row.get("question") or ""), True, False
            )
            prompt_ids = tokenizer.encode(rendered, add_special_tokens=True)
            completion_ids = tokenizer.encode(
                str(row.get("completion") or ""), add_special_tokens=False
            )
            token_ids, completion_start, was_truncated = bounded_token_rows(
                prompt_ids, completion_ids, args.max_sequence_length
            )
            token_rows.append(token_ids)
            completion_starts.append(completion_start)
            batch_truncated.append(was_truncated)
        width = max(len(row) for row in token_rows)
        input_ids = torch.full(
            (len(batch), width),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device="cuda:0",
        )
        attention = torch.zeros_like(input_ids)
        for index, token_ids in enumerate(token_rows):
            input_ids[index, : len(token_ids)] = torch.tensor(
                token_ids, dtype=torch.long, device="cuda:0"
            )
            attention[index, : len(token_ids)] = 1
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = text_model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                output_hidden_states=True,
            )
            pooled = pool_hidden_states(
                outputs.hidden_states,
                layer_offsets,
                [len(row) for row in token_rows],
                completion_starts,
                args.tail_tokens,
            )
        feature_batches.append(pooled.to(dtype=torch.float16, device="cpu"))
        truncated += sum(batch_truncated)
        for row, was_truncated in zip(batch, batch_truncated, strict=True):
            group = grouped[str(row["identity_sha256"])]
            shape_rows.append(feature_vector(row, group))
            metadata.append(
                {
                    "identity_sha256": str(row["identity_sha256"]),
                    "task": str(row["task"]),
                    "sample_index": int(row["sample_index"]),
                    "prediction": row.get("prediction"),
                    "correct": bool(row["correct"]),
                    "empty_completion": not bool(str(row.get("completion") or "")),
                    "prompt_truncated": was_truncated,
                }
            )
        done = min(offset + len(batch), len(rows))
        if done % max(args.batch_size * 25, 1) == 0 or done == len(rows):
            print(
                f"[candidate-features] candidates={done}/{len(rows)} "
                f"truncated={truncated}",
                flush=True,
            )

    features = torch.cat(feature_batches, dim=0)
    shape = torch.tensor(shape_rows, dtype=torch.float32)
    payload = {
        "schema": SCHEMA,
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_update": adapter_metadata.get("update") if adapter_metadata else None,
        "candidate_source": str(args.candidates.resolve()),
        "candidate_sha256": sha256_file(args.candidates),
        "skip": args.skip,
        "count": args.count,
        "layer_offsets": layer_offsets,
        "pooling": POOLING,
        "tail_tokens": args.tail_tokens,
        "hidden_features": features,
        "shape_feature_names": FEATURE_NAMES,
        "shape_features": shape,
        "metadata": metadata,
    }
    _atomic_torch_save(args.output, payload)
    elapsed = time.monotonic() - started
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": sha256_file(args.candidates),
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_sha256": sha256_file(args.adapter_checkpoint),
        "model_revision": args.model_revision,
        "skip": args.skip,
        "count": args.count,
        "candidate_rows": len(rows),
        "feature_width": int(features.shape[1]),
        "shape_width": int(shape.shape[1]),
        "layer_offsets": layer_offsets,
        "pooling": POOLING,
        "tail_tokens": args.tail_tokens,
        "prompt_truncated": truncated,
        "elapsed_seconds": elapsed,
        "candidates_per_second": len(rows) / max(elapsed, 1e-9),
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", default="auto")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--tail-tokens", type=int, default=32)
    parser.add_argument(
        "--layer-offset", type=int, action="append", default=list(DEFAULT_LAYER_OFFSETS)
    )
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
