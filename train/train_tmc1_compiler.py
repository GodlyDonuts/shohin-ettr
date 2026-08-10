#!/usr/bin/env python3
"""Train the frozen TMC1 typed graph compiler over a frozen semantic owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch

from hf_product_reasoning_eval import _load_model
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from natural_microcode_program import parse_program
from typed_microcode_compiler import (
    MAX_SOURCE_SPANS,
    TypedMicrocodeCompiler,
    graph_labels,
    typed_compiler_loss,
)
from typed_microcode_graph import TypedMicrocodeGraph, compile_typed_graph

SCHEMA = "shohin-tmc1-typed-compiler-training-v1"


class TMC1TrainingError(RuntimeError):
    """Frozen TMC1 training custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_graphs(path: Path, expected_sha256: str) -> list[TypedMicrocodeGraph]:
    if sha256_file(path) != expected_sha256:
        raise TMC1TrainingError("training data SHA-256 differs")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 6333:
        raise TMC1TrainingError("training population differs")
    graphs = [
        compile_typed_graph(
            str(row["original_question"]), parse_program(str(row["gold_program"]))
        )
        for row in rows
    ]
    return graphs


def render_source(tokenizer: Any, source: str) -> str:
    return render_reasoning_messages(
        tokenizer,
        [
            {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
            {"role": "user", "content": source},
        ],
        enable_thinking=False,
    )


def tokenize_sources(
    tokenizer: Any,
    graphs: Sequence[TypedMicrocodeGraph],
    device: torch.device,
    maximum_tokens: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
    rendered = [render_source(tokenizer, graph.source) for graph in graphs]
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    if encoded["input_ids"].shape[1] > maximum_tokens:
        raise TMC1TrainingError("source prompt exceeds frozen context")
    candidate_mask = torch.zeros(
        len(graphs),
        MAX_SOURCE_SPANS,
        encoded["input_ids"].shape[1],
        dtype=torch.bool,
    )
    for row, (graph, prompt) in enumerate(zip(graphs, rendered, strict=True)):
        start = prompt.rfind(graph.source)
        if start < 0:
            raise TMC1TrainingError("source is absent from rendered prompt")
        for candidate, span in enumerate(graph.number_spans):
            absolute_start = start + span.start
            absolute_end = start + span.end
            for token, (token_start, token_end) in enumerate(offsets[row].tolist()):
                if token_end > absolute_start and token_start < absolute_end:
                    candidate_mask[row, candidate, token] = True
            if not candidate_mask[row, candidate].any():
                raise TMC1TrainingError("source span lacks token owner")
    receipt = {
        "maximum_tokens": int(encoded["attention_mask"].sum(1).max()),
        "charged_source_tokens": int(encoded["attention_mask"].sum()),
    }
    return (
        {key: value.to(device) for key, value in encoded.items()},
        candidate_mask.to(device),
        receipt,
    )


def shuffled(
    graphs: Sequence[TypedMicrocodeGraph], seed: int
) -> list[TypedMicrocodeGraph]:
    result = list(graphs)
    random.Random(seed).shuffle(result)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise TMC1TrainingError("refusing existing output")
    if (
        args.updates != 4096
        or args.batch_size != 32
        or args.max_source_tokens != 512
        or args.width != 512
        or args.source_layers != 2
        or args.decoder_layers != 4
        or args.heads != 8
        or args.learning_rate != 2e-4
        or args.seed != 2026081061
        or args.data_seed != 2026081062
    ):
        raise TMC1TrainingError("frozen training geometry differs")
    if sha256_file(args.owner_checkpoint) != args.expected_owner_sha256:
        raise TMC1TrainingError("semantic owner checkpoint SHA-256 differs")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise TMC1TrainingError("fast tokenizer offsets unavailable")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    owner, metadata, loader = _load_model(
        args.model_root, args.owner_checkpoint, "auto"
    )
    if (
        metadata is None
        or metadata.get("update") != 1024
        or metadata.get("model_revision") != args.model_revision
        or metadata.get("data_sha256") != args.expected_owner_data_sha256
    ):
        raise TMC1TrainingError("semantic owner metadata differs")
    owner.eval().requires_grad_(False)
    source_width = int(owner.text_model.embed_tokens.embedding_dim)
    compiler = TypedMicrocodeCompiler(
        source_width,
        width=args.width,
        source_layers=args.source_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
    ).to(device=device, dtype=torch.bfloat16)
    trainable = compiler.parameter_count()
    if trainable != 24_864_055:
        raise TMC1TrainingError("compiler parameter receipt differs")
    optimizer = torch.optim.AdamW(
        compiler.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    graphs = shuffled(load_graphs(args.data, args.expected_data_sha256), args.data_seed)
    cursor = 0
    charged_examples = 0
    charged_source_tokens = 0
    maximum_source_tokens = 0
    trace = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    compiler.train()
    for update in range(1, args.updates + 1):
        if cursor + args.batch_size > len(graphs):
            graphs = shuffled(graphs, args.data_seed + update)
            cursor = 0
        batch = graphs[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        encoded, candidate_mask, receipt = tokenize_sources(
            tokenizer, batch, device, args.max_source_tokens
        )
        labels = graph_labels(batch, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            source_states = owner.text_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state.detach()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = compiler(
                source_states,
                encoded["attention_mask"].bool(),
                candidate_mask,
                labels["source_count"],
            )
            loss, components = typed_compiler_loss(output, labels)
        if not torch.isfinite(loss):
            raise TMC1TrainingError("nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(compiler.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise TMC1TrainingError("nonfinite gradient")
        optimizer.step()
        charged_examples += len(batch)
        charged_source_tokens += receipt["charged_source_tokens"]
        maximum_source_tokens = max(maximum_source_tokens, receipt["maximum_tokens"])
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            event = {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **{
                    f"{name}_loss": float(value.detach())
                    for name, value in components.items()
                },
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    checkpoint = args.output / "compiler.pt"
    temporary_checkpoint = args.output / f".compiler.pt.tmp.{os.getpid()}"
    torch.save(
        {
            "schema": SCHEMA,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in compiler.state_dict().items()
            },
            "config": {
                "source_width": source_width,
                "width": args.width,
                "source_layers": args.source_layers,
                "decoder_layers": args.decoder_layers,
                "heads": args.heads,
            },
            "updates": args.updates,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "model_revision": args.model_revision,
            "owner_checkpoint_sha256": args.expected_owner_sha256,
            "data_sha256": args.expected_data_sha256,
        },
        temporary_checkpoint,
    )
    os.replace(temporary_checkpoint, checkpoint)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "semantic_owner_checkpoint": str(args.owner_checkpoint.resolve()),
        "semantic_owner_checkpoint_sha256": args.expected_owner_sha256,
        "semantic_owner_data_sha256": args.expected_owner_data_sha256,
        "data": str(args.data.resolve()),
        "data_sha256": args.expected_data_sha256,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "trainable_parameters": trainable,
        "charged_examples": charged_examples,
        "charged_source_tokens": charged_source_tokens,
        "maximum_source_tokens": maximum_source_tokens,
        "elapsed_seconds": elapsed,
        "examples_per_second": charged_examples / elapsed,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "trace": trace,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    atomic_json(args.output / "report.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "trace"},
            indent=2,
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-owner-sha256", required=True)
    parser.add_argument("--expected-owner-data-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-source-tokens", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--source-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026081061)
    parser.add_argument("--data-seed", type=int, default=2026081062)
    parser.add_argument("--log-interval", type=int, default=64)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
