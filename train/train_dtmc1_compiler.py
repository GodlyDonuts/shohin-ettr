#!/usr/bin/env python3
"""Train the frozen DTMC1 compiler on source plus model-owned drafts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from dtmc1_inputs import (
    DraftExample,
    load_examples,
    sha256_file,
    tokenize_draft_sources,
)
from hf_product_reasoning_eval import _load_model
from typed_microcode_compiler import (
    TypedMicrocodeCompiler,
    graph_labels,
    typed_compiler_loss,
)

SCHEMA = "shohin-dtmc1-typed-compiler-training-v1"


class DTMC1TrainingError(RuntimeError):
    """Frozen DTMC1 training custody differs."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def shuffled(examples: list[DraftExample], seed: int) -> list[DraftExample]:
    result = list(examples)
    random.Random(seed).shuffle(result)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise DTMC1TrainingError("refusing existing output")
    if (
        args.updates != 4096
        or args.batch_size != 32
        or args.max_source_tokens != 1024
        or args.width != 512
        or args.source_layers != 2
        or args.decoder_layers != 4
        or args.heads != 8
        or args.learning_rate != 2e-4
        or args.seed != 2026081061
        or args.data_seed != 2026081062
    ):
        raise DTMC1TrainingError("frozen training geometry differs")
    if sha256_file(args.owner_checkpoint) != args.expected_owner_sha256:
        raise DTMC1TrainingError("semantic owner checkpoint SHA-256 differs")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise DTMC1TrainingError("fast tokenizer offsets unavailable")
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
        raise DTMC1TrainingError("semantic owner metadata differs")
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
        raise DTMC1TrainingError("compiler parameter receipt differs")
    optimizer = torch.optim.AdamW(
        compiler.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    examples = shuffled(
        load_examples(args.data, args.expected_data_sha256, 6333), args.data_seed
    )
    cursor = 0
    charged_examples = 0
    charged_source_draft_tokens = 0
    maximum_source_tokens = 0
    trace = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    compiler.train()
    for update in range(1, args.updates + 1):
        if cursor + args.batch_size > len(examples):
            examples = shuffled(examples, args.data_seed + update)
            cursor = 0
        batch = examples[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        encoded, candidate_mask, receipt = tokenize_draft_sources(
            tokenizer, batch, device, args.max_source_tokens
        )
        labels = graph_labels([example.graph for example in batch], device)
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
            raise DTMC1TrainingError("nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(compiler.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise DTMC1TrainingError("nonfinite gradient")
        optimizer.step()
        charged_examples += len(batch)
        charged_source_draft_tokens += receipt["charged_source_draft_tokens"]
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
            "input_envelope": "PROBLEM/source + MODEL-OWNED DRAFT",
            "maximum_source_tokens": args.max_source_tokens,
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
        "charged_source_draft_tokens": charged_source_draft_tokens,
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
    parser.add_argument("--max-source-tokens", type=int, default=1024)
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
