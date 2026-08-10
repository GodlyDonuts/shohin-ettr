#!/usr/bin/env python3
"""Train the frozen PSTC1 hard pushdown-stack compiler."""

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

from fixed_slot_typed_compiler import MAX_SOURCE_NUMBERS
from hf_product_reasoning_train import load_product_backbone, resolve_product_backbone_layout
from pushdown_stack_typed_compiler import (
    PushdownStackCompiler,
    StackProgram,
    load_stack_program,
    stack_labels,
    stack_loss,
)


SCHEMA = "shohin-pstc1-stack-training-v1"


class PSTC1TrainingError(RuntimeError):
    """Raised when PSTC1 training custody or mechanics differ."""


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


def load_programs(path: Path, expected_sha256: str) -> list[StackProgram]:
    if sha256_file(path) != expected_sha256:
        raise PSTC1TrainingError("training data SHA-256 differs")
    programs = [
        load_stack_program(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(programs) != 75935:
        raise PSTC1TrainingError("training population differs")
    return programs


def shuffled(programs: Sequence[StackProgram], seed: int) -> list[StackProgram]:
    ordered = list(programs)
    random.Random(seed).shuffle(ordered)
    return ordered


def tokenize_sources(
    tokenizer: Any,
    programs: Sequence[StackProgram],
    device: torch.device,
    maximum: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
    encoded = tokenizer(
        [program.question for program in programs],
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    if encoded["input_ids"].shape[1] > maximum:
        raise PSTC1TrainingError("source exceeds frozen token context")
    candidate_mask = torch.zeros(
        len(programs),
        MAX_SOURCE_NUMBERS,
        encoded["input_ids"].shape[1],
        dtype=torch.bool,
    )
    for row, program in enumerate(programs):
        for candidate, span in enumerate(program.number_spans):
            for token, (start, end) in enumerate(offsets[row].tolist()):
                if end > span.start and start < span.end:
                    candidate_mask[row, candidate, token] = True
            if not candidate_mask[row, candidate].any():
                raise PSTC1TrainingError("numeric span has no tokenizer owner")
    receipt = {
        "maximum_tokens": int(encoded["attention_mask"].sum(1).max().item()),
        "charged_source_tokens": int(encoded["attention_mask"].sum().item()),
    }
    return (
        {key: value.to(device) for key, value in encoded.items()},
        candidate_mask.to(device),
        receipt,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise PSTC1TrainingError("refusing existing output")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True, use_fast=True
    )
    if not tokenizer.is_fast:
        raise PSTC1TrainingError("fast tokenizer offsets unavailable")
    backbone, loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization="none",
    )
    text_model, _, source_width, layout = resolve_product_backbone_layout(backbone)
    backbone.eval().requires_grad_(False)
    compiler = PushdownStackCompiler(
        source_width,
        width=args.width,
        encoder_layers=args.encoder_layers,
        heads=args.heads,
    ).to(device=device, dtype=torch.bfloat16)
    trainable = compiler.parameter_count()
    if trainable >= 30_000_000:
        raise PSTC1TrainingError("compiler parameter receipt differs")
    optimizer = torch.optim.AdamW(
        compiler.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    programs = shuffled(load_programs(args.data, args.expected_data_sha256), args.data_seed)
    cursor = charged_examples = charged_source_tokens = maximum_source_tokens = 0
    losses = []
    started = time.time()
    compiler.train()
    for update in range(1, args.updates + 1):
        if cursor + args.batch_size > len(programs):
            programs = shuffled(programs, args.data_seed + update)
            cursor = 0
        batch = programs[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        encoded, candidate_mask, token_receipt = tokenize_sources(
            tokenizer, batch, device, args.max_source_tokens
        )
        labels = stack_labels(batch, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            source = text_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state.detach()
        feedback = "gold" if update <= args.gold_feedback_updates else "hard"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = compiler(
                source,
                encoded["attention_mask"].bool(),
                candidate_mask,
                labels["candidate_count"],
                gold=labels,
                feedback=feedback,
            )
            loss, components = stack_loss(output, labels)
        if not torch.isfinite(loss):
            raise PSTC1TrainingError("nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(compiler.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise PSTC1TrainingError("nonfinite gradient")
        optimizer.step()
        charged_examples += len(batch)
        charged_source_tokens += token_receipt["charged_source_tokens"]
        maximum_source_tokens = max(maximum_source_tokens, token_receipt["maximum_tokens"])
        losses.append(float(loss.detach()))
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            print(
                f"update={update}/{args.updates} feedback={feedback} "
                f"loss={losses[-1]:.4f} gnorm={float(gradient_norm):.4f} "
                f"action={float(components['action']):.4f} "
                f"pointer={float(components['pointer']):.4f} "
                f"invalid_feedback={int(output.invalid_action_count)}",
                flush=True,
            )
    elapsed = time.time() - started
    checkpoint = args.output / "compiler.pt"
    temporary_checkpoint = args.output / f".compiler.pt.tmp.{os.getpid()}"
    torch.save(
        {
            "schema": SCHEMA,
            "state_dict": {key: value.detach().cpu() for key, value in compiler.state_dict().items()},
            "config": {
                "source_width": source_width,
                "width": args.width,
                "encoder_layers": args.encoder_layers,
                "heads": args.heads,
            },
            "updates": args.updates,
            "model_revision": args.model_revision,
            "data_sha256": args.expected_data_sha256,
        },
        temporary_checkpoint,
    )
    os.replace(temporary_checkpoint, checkpoint)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "model_revision": args.model_revision,
        "model_loader": loader,
        "backbone_layout": layout,
        "data": str(args.data.resolve()),
        "data_sha256": args.expected_data_sha256,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "gold_feedback_updates": args.gold_feedback_updates,
        "charged_examples": charged_examples,
        "charged_source_tokens": charged_source_tokens,
        "maximum_source_tokens": maximum_source_tokens,
        "learning_rate": args.learning_rate,
        "trainable_parameters": trainable,
        "total_parameters": trainable + sum(parameter.numel() for parameter in backbone.parameters()),
        "elapsed_seconds": elapsed,
        "examples_per_second": charged_examples / elapsed,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", default="auto")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gold-feedback-updates", type=int, default=128)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026081041)
    parser.add_argument("--data-seed", type=int, default=2026081042)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
