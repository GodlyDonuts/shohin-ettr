"""Evaluate complete-trace revision and causal packet interventions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable

import torch

from diverge_crp1_data import tokenize_revision_example
from diverge_crp1_product import CRP1ProductModel, load_crp1_checkpoint
from hf_product_reasoning_eval import _completion_usage, _generation_stop_token_ids
from hf_product_reasoning_train import (
    ProductReasoningModel,
    load_product_backbone,
    load_trainable_checkpoint,
)


BOARD_SCHEMA = "shohin-diverge-crp1-board-v1"
REPORT_SCHEMA = "shohin-diverge-crp1-evaluation-v1"
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
ERROR_STEP = re.compile(r"Error\s+step\s*:\s*(NONE|\d+)", re.IGNORECASE)


class CRP1EvalError(RuntimeError):
    """The causal-revision evaluation contract was violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CRP1EvalError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _batches(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def extract_answer(completion: str) -> str | None:
    matches = BOXED.findall(completion)
    return matches[-1].strip() if matches and matches[-1].strip() else None


def extract_error_step(completion: str) -> int | None:
    match = ERROR_STEP.search(completion)
    if match is None:
        return None
    value = match.group(1).upper()
    return 0 if value == "NONE" else int(value)


def _normalize_answer(value: str | None) -> str | None:
    return "".join(value.casefold().split()) if value is not None else None


def _read_board(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if _sha256_file(path) != expected_sha256:
        raise CRP1EvalError("CRP1 evaluation board hash differs")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != BOARD_SCHEMA or row.get("split") != "evaluation":
            raise CRP1EvalError("CRP1 evaluation schema or split differs")
        identity = str(row.get("identity_sha256") or "")
        if len(identity) != 64 or identity in identities:
            raise CRP1EvalError("CRP1 evaluation identity differs")
        identities.add(identity)
        rows.append(row)
    if not rows:
        raise CRP1EvalError("CRP1 evaluation board is empty")
    return rows


def _source_model(args: argparse.Namespace):
    payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CRP1EvalError("protected source metadata is missing")
    expected = {
        "arm": "baseline",
        "model_revision": args.model_revision,
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "unfreeze_layers": 2,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise CRP1EvalError("protected source metadata differs")
    backbone, resolved_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    source = ProductReasoningModel(
        backbone,
        arm="baseline",
        lora_layers=4,
        lora_rank=8,
        lora_alpha=16.0,
        workspace_width=512,
        workspace_slots=16,
        recurrent_steps=8,
        unfreeze_layers=2,
    ).to("cuda:0")
    update, restored = load_trainable_checkpoint(args.source_checkpoint, source)
    if restored != metadata:
        raise CRP1EvalError("protected source metadata replay differs")
    source.requires_grad_(False).eval()
    return source, update, resolved_loader


def _crp_model(args: argparse.Namespace):
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise CRP1EvalError("CRP1 checkpoint is missing")
    if _sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise CRP1EvalError("CRP1 checkpoint hash differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("arm") != args.arm:
        raise CRP1EvalError("CRP1 checkpoint metadata differs")
    if metadata.get("source_checkpoint_sha256") != args.source_checkpoint_sha256:
        raise CRP1EvalError("CRP1 source provenance differs")
    if metadata.get("model_revision") != args.model_revision:
        raise CRP1EvalError("CRP1 model revision differs")
    config = metadata.get("packet_config")
    if not isinstance(config, dict):
        raise CRP1EvalError("CRP1 packet metadata is missing")
    backbone, resolved_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = CRP1ProductModel(
        backbone,
        args.source_checkpoint,
        source_checkpoint_sha256=args.source_checkpoint_sha256,
        source_revision=args.model_revision,
        unguarded=args.arm == "unguarded",
        workspace_width=int(config["workspace_width"]),
        workspace_slots=int(config["workspace_slots"]),
        recurrent_steps=int(config["recurrent_steps"]),
        attention_heads=int(config["attention_heads"]),
        ff_multiplier=int(config["ff_multiplier"]),
        max_trace_steps=int(config["max_trace_steps"]),
        localization_weight=float(metadata["localization_weight"]),
    ).to("cuda:0")
    update, restored = load_crp1_checkpoint(args.checkpoint, model)
    if restored != metadata:
        raise CRP1EvalError("CRP1 checkpoint replay differs")
    model.set_ablation(args.ablation)
    model.eval()
    model.source.eval()
    return model, update, resolved_loader


def _left_pad(
    rows: list[Any],
    pad_token_id: int,
    max_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row.prompt_ids) for row in rows)
    batch = len(rows)
    ids = torch.full((batch, width), pad_token_id, dtype=torch.long, device=device)
    active = torch.zeros((batch, width), dtype=torch.long, device=device)
    problem = torch.zeros((batch, width), dtype=torch.bool, device=device)
    steps = torch.zeros((batch, max_steps, width), dtype=torch.bool, device=device)
    final = torch.zeros((batch, width), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        if len(row.step_masks) > max_steps:
            raise CRP1EvalError("evaluation trace exceeds packet width")
        start = width - len(row.prompt_ids)
        ids[index, start:] = torch.tensor(row.prompt_ids, device=device)
        active[index, start:] = 1
        problem[index, start:] = torch.tensor(row.problem_mask, device=device)
        final[index, start:] = torch.tensor(row.final_mask, device=device)
        for step, mask in enumerate(row.step_masks):
            steps[index, step, start:] = torch.tensor(mask, device=device)
    return ids, active, problem, steps, final


def _generate(
    model: ProductReasoningModel | CRP1ProductModel,
    tokenizer: Any,
    tokenized: list[Any],
    *,
    arm: str,
    max_steps: int,
    max_new_tokens: int,
    stop_ids: list[int],
) -> tuple[list[str], list[tuple[int, bool]], list[int | None]]:
    device = model.text_model.embed_tokens.weight.device
    ids, attention, problem, steps, final = _left_pad(
        tokenized, tokenizer.pad_token_id, max_steps, device
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if arm == "plain":
            embeddings = model.text_model.embed_tokens(ids)
            selected: list[int | None] = [None] * len(tokenized)
        else:
            assert isinstance(model, CRP1ProductModel)
            embeddings, attention, _, selected_tensor = (
                model.revision_generation_embeddings(
                    ids, attention, problem, steps, final
                )
            )
            selected = [int(value) for value in selected_tensor.cpu().tolist()]
        output = model.backbone.generate(
            inputs_embeds=embeddings,
            attention_mask=attention,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=stop_ids[0] if len(stop_ids) == 1 else stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    completions = tokenizer.batch_decode(output, skip_special_tokens=True)
    usage = [
        _completion_usage(row.tolist(), stop_ids, max_new_tokens) for row in output
    ]
    return completions, usage, selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if _sha256_file(args.source_checkpoint) != args.source_checkpoint_sha256:
        raise CRP1EvalError("protected source checkpoint hash differs")
    rows = _read_board(args.data, args.data_sha256)
    if args.count < len(rows):
        rows = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{args.subset_seed}\0{row['identity_sha256']}".encode()
            ).hexdigest(),
        )[: args.count]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.arm == "plain":
        model, update, resolved_loader = _source_model(args)
        checkpoint_sha256 = None
        max_steps = args.max_trace_steps
    else:
        model, update, resolved_loader = _crp_model(args)
        checkpoint_sha256 = _sha256_file(args.checkpoint)
        max_steps = model.packet_config.max_trace_steps
    stop_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    prompt_limit = args.max_sequence_length - args.max_new_tokens
    results: list[dict[str, Any]] = []
    skipped_length = 0
    generated_tokens = 0
    exhausted = 0
    for batch in _batches(rows, args.batch_size):
        admitted: list[dict[str, Any]] = []
        tokenized = []
        for row in batch:
            wrong = args.variant == "wrong"
            tokens = tokenize_revision_example(
                tokenizer,
                str(row["problem"]),
                list(map(str, row["wrong_steps"] if wrong else row["correct_steps"])),
                f"Final answer: \\boxed{{{row['wrong_answer'] if wrong else row['answer']}}}",
                None,
                max_sequence_length=prompt_limit,
                workspace_slots=args.workspace_slots,
            )
            if tokens is None:
                skipped_length += 1
                continue
            admitted.append(row)
            tokenized.append(tokens)
        if not admitted:
            continue
        completions, usage, selected = _generate(
            model,
            tokenizer,
            tokenized,
            arm=args.arm,
            max_steps=max_steps,
            max_new_tokens=args.max_new_tokens,
            stop_ids=stop_ids,
        )
        for row, completion, (used, hit_cap), selected_candidate in zip(
            admitted, completions, usage, selected, strict=True
        ):
            prediction = extract_answer(completion)
            predicted_error = extract_error_step(completion)
            target_error = int(row["error_index"]) if args.variant == "wrong" else 0
            answer_correct = _normalize_answer(prediction) == _normalize_answer(
                str(row["answer"])
            )
            error_correct = predicted_error == target_error
            packet_correct = (
                selected_candidate == target_error
                if selected_candidate is not None
                else None
            )
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "family": row["family"],
                    "depth": row["depth"],
                    "variant": args.variant,
                    "problem": row["problem"],
                    "draft_steps": row[
                        "wrong_steps" if args.variant == "wrong" else "correct_steps"
                    ],
                    "draft_answer": (
                        row["wrong_answer"] if args.variant == "wrong" else row["answer"]
                    ),
                    "gold_answer": row["answer"],
                    "target_error_step": target_error,
                    "completion": completion,
                    "prediction": prediction,
                    "predicted_error_step": predicted_error,
                    "selected_candidate": selected_candidate,
                    "answer_correct": answer_correct,
                    "error_localization_correct": error_correct,
                    "packet_localization_correct": packet_correct,
                    "joint_correct": answer_correct and error_correct,
                    "generated_tokens": used,
                    "exhausted": hit_cap,
                }
            )
            generated_tokens += used
            exhausted += int(hit_cap)
        print(f"[crp1-eval] completed={len(results)}/{len(rows)}", flush=True)
    if not results:
        raise CRP1EvalError("CRP1 evaluation admitted no rows")
    family_metrics: dict[str, dict[str, int]] = {}
    for family in sorted({str(row["family"]) for row in results}):
        subset = [row for row in results if row["family"] == family]
        family_metrics[family] = {
            "rows": len(subset),
            "exact_answers": sum(row["answer_correct"] for row in subset),
            "error_localizations": sum(
                row["error_localization_correct"] for row in subset
            ),
            "joint": sum(row["joint_correct"] for row in subset),
        }
    packet_rows = [
        row for row in results if row["packet_localization_correct"] is not None
    ]
    elapsed = time.monotonic() - started
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "ablation": args.ablation,
        "variant": args.variant,
        "data": str(args.data.resolve()),
        "data_sha256": args.data_sha256,
        "input_rows": len(rows),
        "evaluated_rows": len(results),
        "skipped_length": skipped_length,
        "exact_answers": sum(row["answer_correct"] for row in results),
        "error_localizations": sum(
            row["error_localization_correct"] for row in results
        ),
        "packet_localizations": (
            sum(row["packet_localization_correct"] for row in packet_rows)
            if packet_rows
            else None
        ),
        "joint_correct": sum(row["joint_correct"] for row in results),
        "family_metrics": family_metrics,
        "selected_candidate_histogram": dict(
            sorted(
                Counter(
                    str(row["selected_candidate"])
                    for row in results
                    if row["selected_candidate"] is not None
                ).items()
            )
        ),
        "generated_tokens": generated_tokens,
        "exhausted": exhausted,
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "generation_seed": args.generation_seed,
        "subset_seed": args.subset_seed,
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_update": update,
        "results": results,
    }
    _atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("auto", "causal"), default="causal")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("plain", "guarded", "unguarded"), required=True)
    parser.add_argument(
        "--ablation",
        choices=("normal", "reset", "force_no_error", "shift", "packet_swap"),
        default="normal",
    )
    parser.add_argument("--variant", choices=("wrong", "correct"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--count", type=int, default=480)
    parser.add_argument("--subset-seed", type=int, default=47)
    parser.add_argument("--generation-seed", type=int, default=2026080604)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--workspace-slots", type=int, default=6)
    parser.add_argument("--max-trace-steps", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    if min(
        args.count,
        args.batch_size,
        args.max_sequence_length,
        args.workspace_slots,
        args.max_trace_steps,
        args.max_new_tokens,
    ) <= 0:
        parser.error("CRP1 evaluation dimensions must be positive")
    if args.arm != "plain" and (
        args.checkpoint is None or args.checkpoint_sha256 is None
    ):
        parser.error("trained CRP1 arm requires a checkpoint and hash")
    if args.arm == "plain" and args.ablation != "normal":
        parser.error("plain correction supports only the normal ablation")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[crp1-eval] arm={report['arm']} variant={report['variant']} "
        f"answers={report['exact_answers']}/{report['evaluated_rows']} "
        f"joint={report['joint_correct']}/{report['evaluated_rows']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
