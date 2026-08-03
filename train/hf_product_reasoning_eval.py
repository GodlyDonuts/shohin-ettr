"""Evaluate a Hugging Face reasoning backbone on frozen local math boards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any


class ProductEvalError(RuntimeError):
    """The product reasoning evaluator contract was violated."""


def _clean_number(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "").rstrip(".")
    match = re.search(r"-?\d+(?:/\d+)?(?:\.\d+)?", cleaned)
    return match.group(0) if match else None


def extract_gsm8k(text: str) -> str | None:
    boxed_numbers = [
        _clean_number(value)
        for value in _boxed_values(text)
        if _clean_number(value) is not None
    ]
    if boxed_numbers:
        return boxed_numbers[-1]
    explicit = re.findall(
        r"(?:answer|final answer)\s*(?:is|:)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return _clean_number(explicit[-1])
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return _clean_number(numbers[-1]) if numbers else None


def _boxed_values(text: str) -> list[str]:
    values: list[str] = []
    cursor = 0
    while True:
        start = text.find(r"\boxed", cursor)
        if start < 0:
            break
        opening = text.find("{", start)
        if opening >= 0:
            depth = 0
            for index in range(opening, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        value = text[opening + 1 : index].strip()
                        if value:
                            values.append(value)
                        cursor = index + 1
                        break
            else:
                cursor = opening + 1
        else:
            cursor = start + len(r"\boxed")
    return values


def extract_boxed(text: str) -> str | None:
    boxed = _boxed_values(text)
    if boxed:
        return boxed[-1]
    fallback = re.findall(
        r"(?:answer|final answer)\s*(?:is|:)\s*\$?\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    return fallback[-1].strip().rstrip(".") if fallback else None


def gold_gsm8k(row: dict[str, Any]) -> str | None:
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", str(row.get("answer", "")))
    return _clean_number(match.group(1)) if match else None


def gold_math(row: dict[str, Any]) -> str | None:
    for key in ("answer", "solution", "expected_answer"):
        value = row.get(key)
        if value:
            boxed = extract_boxed(str(value))
            if boxed is not None:
                return boxed.replace(" ", "")
    return None


def match_gsm8k(prediction: str | None, gold: str | None) -> bool:
    return (
        prediction is not None
        and gold is not None
        and _clean_number(prediction) == _clean_number(gold)
    )


def _normalize_math(value: str | None) -> str | None:
    if value is None:
        return None
    return (
        value.replace(" ", "")
        .replace(r"\dfrac", r"\frac")
        .replace(r"\tfrac", r"\frac")
        .replace("$", "")
        .rstrip(".")
    )


def match_math(prediction: str | None, gold: str | None) -> bool:
    return (
        prediction is not None
        and gold is not None
        and _normalize_math(prediction) == _normalize_math(gold)
    )


TASKS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "extract": extract_gsm8k,
        "gold": gold_gsm8k,
        "match": match_gsm8k,
    },
    "math500": {
        "extract": extract_boxed,
        "gold": gold_math,
        "match": match_math,
    },
}


def _question(row: dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt"):
        value = row.get(key)
        if value:
            return str(value)
    raise ProductEvalError("evaluation row has no question")


def _row_identity(task: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{task}\0{_question(row)}".encode()).hexdigest()


def select_rows(
    task: str,
    rows: list[dict[str, Any]],
    count: int,
    subset_seed: int,
) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise ProductEvalError("requested row count is outside the board")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{subset_seed}\0{_row_identity(task, row)}".encode()
        ).hexdigest(),
    )
    selected = ranked[:count]
    identities = [_row_identity(task, row) for row in selected]
    if len(set(identities)) != len(identities):
        raise ProductEvalError("selected board contains duplicate prompts")
    return selected


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ProductEvalError(f"refusing to replace evaluation report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _load_model(model_root: Path, adapter_checkpoint: Path | None):
    import torch
    from transformers import AutoModelForMultimodalLM

    backbone = AutoModelForMultimodalLM.from_pretrained(
        model_root,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if adapter_checkpoint is None:
        return backbone.eval(), None

    from hf_product_reasoning_train import (
        ProductReasoningModel,
        load_trainable_checkpoint,
    )

    payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ProductEvalError("adapter checkpoint metadata is missing")
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
    ).to("cuda:0")
    update, restored_metadata = load_trainable_checkpoint(adapter_checkpoint, model)
    model.eval()
    return model, {"update": update, **restored_metadata}


def _generation_arguments(mode: str, max_new_tokens: int) -> dict[str, Any]:
    common: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if mode == "greedy":
        return {**common, "do_sample": False}
    if mode == "qwen-thinking":
        return {
            **common,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
        }
    raise ProductEvalError("unsupported generation mode")


def _make_prompt(question: str) -> str:
    return (
        "Solve the problem carefully. Show the reasoning needed to verify the "
        "result, then put only the final answer inside \\boxed{}.\n\nProblem:\n"
        f"{question}"
    )


def _render_prompt(
    tokenizer: Any,
    question: str,
    adapter: bool,
    enable_thinking: bool,
) -> str:
    if adapter:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful reasoning assistant. Give concise, "
                    "verifiable reasoning and a clearly marked final answer."
                ),
            },
            {"role": "user", "content": question},
        ]
        thinking = False
    else:
        messages = [{"role": "user", "content": _make_prompt(question)}]
        thinking = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def _generate_adapter(
    model: Any,
    encoded: dict[str, Any],
    generation_arguments: dict[str, Any],
    pad_token_id: int,
):
    from hf_product_reasoning_train import product_generation_embeddings

    embeddings, attention = product_generation_embeddings(
        model,
        encoded["input_ids"],
        encoded["attention_mask"],
    )
    return model.backbone.generate(
        inputs_embeds=embeddings,
        attention_mask=attention,
        pad_token_id=pad_token_id,
        **generation_arguments,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    task = TASKS[args.task]
    data_bytes = args.data.read_bytes()
    rows = [json.loads(line) for line in data_bytes.splitlines() if line.strip()]
    selected = select_rows(args.task, rows, args.count, args.subset_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata = _load_model(args.model_root, args.adapter_checkpoint)

    random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    correct = 0
    generated_tokens = 0
    results: list[dict[str, Any]] = []

    for offset in range(0, len(selected), args.batch_size):
        batch = selected[offset : offset + args.batch_size]
        rendered = [
            _render_prompt(
                tokenizer,
                _question(row),
                adapter_metadata is not None,
                args.enable_thinking,
            )
            for row in batch
        ]
        encoded = tokenizer(rendered, padding=True, return_tensors="pt")
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generation_arguments = _generation_arguments(
                args.generation_mode, args.max_new_tokens
            )
            if adapter_metadata is None:
                output = model.generate(
                    **encoded,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_arguments,
                )
                completion_ids = output[:, prompt_width:]
            else:
                output = _generate_adapter(
                    model,
                    encoded,
                    generation_arguments,
                    tokenizer.pad_token_id,
                )
                completion_ids = output
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for row, completion in zip(batch, completions, strict=True):
            prediction = task["extract"](completion)
            gold = task["gold"](row)
            is_correct = bool(task["match"](prediction, gold))
            correct += int(is_correct)
            identity = _row_identity(args.task, row)
            results.append(
                {
                    "identity_sha256": identity,
                    "question": _question(row),
                    "gold": gold,
                    "prediction": prediction,
                    "correct": is_correct,
                    "completion": completion,
                }
            )
        generated_tokens += int(completion_ids.shape[1]) * len(batch)
        print(
            f"[product-eval] {min(offset + len(batch), len(selected))}/"
            f"{len(selected)} correct={correct}",
            flush=True,
        )

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    return {
        "schema": "shohin-hf-product-reasoning-eval-v1",
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "adapter_checkpoint": (
            str(args.adapter_checkpoint.resolve()) if args.adapter_checkpoint else None
        ),
        "adapter_metadata": adapter_metadata,
        "task": args.task,
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "subset_seed": args.subset_seed,
        "generation_seed": args.generation_seed,
        "generation_mode": args.generation_mode,
        "enable_thinking": args.enable_thinking,
        "effective_enable_thinking": (
            args.enable_thinking if adapter_metadata is None else False
        ),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "correct": correct,
        "total": len(selected),
        "accuracy": correct / len(selected),
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "selection_sha256": hashlib.sha256(
            "\n".join(result["identity_sha256"] for result in results).encode()
        ).hexdigest(),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--subset-seed", type=int, default=20260802)
    parser.add_argument("--generation-seed", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--generation-mode", choices=("greedy", "qwen-thinking"), default="greedy"
    )
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("batch size and max-new-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    _atomic_json(args.output, report)
    print(
        f"[product-eval] {args.task} {report['correct']}/{report['total']} "
        f"= {report['accuracy']:.2%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
