"""Autonomous first-draft and temporal-correction evaluation for VCR1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch

from diverge_vcr1_data import tokenize_correction_example
from diverge_vcr1_product import VCR1ProductModel, load_vcr1_checkpoint
from hf_product_reasoning_eval import (
    TASKS,
    _completion_usage,
    _generation_stop_token_ids,
    _render_prompt,
    _task_prompt,
    select_rows,
)
from hf_product_reasoning_train import (
    ProductReasoningModel,
    load_product_backbone,
    load_trainable_checkpoint,
    product_generation_embeddings,
)


DRAFT_SCHEMA = "shohin-diverge-vcr1-autonomous-drafts-v1"
REPORT_SCHEMA = "shohin-diverge-vcr1-autonomous-correction-v1"


class VCR1EvalError(RuntimeError):
    """The autonomous correction evaluation contract was violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VCR1EvalError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _batches(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def _ordered_identity_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    identities: list[str] = []
    for row in rows:
        identity = str(row.get("identity_sha256", ""))
        if len(identity) != 64:
            raise VCR1EvalError("autonomous draft identity differs")
        identities.append(identity)
        digest.update(identity.encode())
        digest.update(b"\0")
    if len(set(identities)) != len(identities):
        raise VCR1EvalError("autonomous draft identities are not unique")
    return digest.hexdigest()


def _validate_draft_payload(
    payload: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if payload.get("schema") != DRAFT_SCHEMA or payload.get("status") != "complete":
        raise VCR1EvalError("autonomous draft envelope differs")
    expected = {
        "task": args.task,
        "model_revision": args.model_revision,
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise VCR1EvalError("autonomous draft provenance differs")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise VCR1EvalError("autonomous draft bank is empty")
    if payload.get("count") != len(rows):
        raise VCR1EvalError("autonomous draft count differs")
    if any(
        row.get("task") != args.task
        or not str(row.get("task_prompt", "")).strip()
        or not str(row.get("source_completion", "")).strip()
        for row in rows
    ):
        raise VCR1EvalError("autonomous draft row differs")
    _ordered_identity_sha256(rows)
    return rows


def _source_model(args: argparse.Namespace):
    payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise VCR1EvalError("protected source metadata is missing")
    expected = {
        "arm": "baseline",
        "model_revision": args.model_revision,
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "unfreeze_layers": 2,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise VCR1EvalError("protected source metadata differs")
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
        raise VCR1EvalError("protected source metadata replay differs")
    source.requires_grad_(False).eval()
    return source, update, resolved_loader


def _vcr_model(args: argparse.Namespace):
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise VCR1EvalError("VCR1 correction checkpoint is missing")
    if _sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise VCR1EvalError("VCR1 correction checkpoint hash differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise VCR1EvalError("VCR1 correction metadata is missing")
    if metadata.get("arm") != args.arm:
        raise VCR1EvalError("VCR1 correction arm differs")
    if metadata.get("source_checkpoint_sha256") != args.source_checkpoint_sha256:
        raise VCR1EvalError("VCR1 protected source provenance differs")
    if metadata.get("model_revision") != args.model_revision:
        raise VCR1EvalError("VCR1 model revision differs")
    workspace = metadata.get("workspace_config")
    if not isinstance(workspace, dict):
        raise VCR1EvalError("VCR1 workspace metadata is missing")
    backbone, resolved_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = VCR1ProductModel(
        backbone,
        args.source_checkpoint,
        source_checkpoint_sha256=args.source_checkpoint_sha256,
        source_revision=args.model_revision,
        role_blind=args.arm == "role_blind",
        workspace_width=int(workspace["workspace_width"]),
        workspace_slots=int(workspace["workspace_slots"]),
        recurrent_steps=int(workspace["recurrent_steps"]),
        attention_heads=int(workspace["attention_heads"]),
        ff_multiplier=int(workspace["ff_multiplier"]),
        validity_weight=float(metadata["validity_weight"]),
        correction_margin_weight=float(metadata["correction_margin_weight"]),
        correction_margin=float(metadata["correction_margin"]),
    ).to("cuda:0")
    update, restored = load_vcr1_checkpoint(args.checkpoint, model)
    if restored != metadata:
        raise VCR1EvalError("VCR1 correction metadata replay differs")
    model.set_ablation(args.ablation)
    model.eval()
    model.source.eval()
    return model, update, resolved_loader


def _generate_source(
    source: ProductReasoningModel,
    tokenizer: Any,
    rendered: list[str],
    *,
    max_new_tokens: int,
    stop_ids: list[int],
) -> tuple[list[str], list[tuple[int, bool]]]:
    encoded = tokenizer(rendered, padding=True, return_tensors="pt")
    encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings, attention = product_generation_embeddings(
            source, encoded["input_ids"], encoded["attention_mask"]
        )
        output = source.backbone.generate(
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
    return completions, usage


def _left_pad_corrections(
    rows: list[Any], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row.prompt_ids) for row in rows)
    batch = len(rows)
    ids = torch.full((batch, width), pad_token_id, dtype=torch.long, device=device)
    active = torch.zeros((batch, width), dtype=torch.long, device=device)
    question = torch.zeros((batch, width), dtype=torch.bool, device=device)
    draft = torch.zeros((batch, width), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        start = width - len(row.prompt_ids)
        ids[index, start:] = torch.tensor(row.prompt_ids, device=device)
        active[index, start:] = 1
        question[index, start:] = torch.tensor(row.question_mask, device=device)
        draft[index, start:] = torch.tensor(row.draft_mask, device=device)
    return ids, active, question, draft


def _generate_correction(
    model: ProductReasoningModel | VCR1ProductModel,
    tokenizer: Any,
    tokenized: list[Any],
    *,
    arm: str,
    max_new_tokens: int,
    stop_ids: list[int],
) -> tuple[list[str], list[tuple[int, bool]], list[float | None]]:
    device = model.text_model.embed_tokens.weight.device
    ids, attention, question, draft = _left_pad_corrections(
        tokenized, tokenizer.pad_token_id, device
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if arm == "plain":
            embeddings = model.text_model.embed_tokens(ids)
            validity = [None] * len(tokenized)
        else:
            assert isinstance(model, VCR1ProductModel)
            embeddings, attention, logits = model.correction_generation_embeddings(
                ids, attention, question, draft
            )
            validity = logits.sigmoid().float().cpu().tolist()
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
    return completions, usage, validity


def build_drafts(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if _sha256_file(args.source_checkpoint) != args.source_checkpoint_sha256:
        raise VCR1EvalError("protected source checkpoint hash differs")
    data_sha256 = _sha256_file(args.data)
    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = select_rows(args.task, rows, args.count, args.subset_seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    source, source_update, resolved_loader = _source_model(args)
    stop_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    task_contract = TASKS[args.task]
    output_rows: list[dict[str, Any]] = []
    generated_tokens = 0
    exhausted = 0
    for batch in _batches(selected, args.batch_size):
        prompts = [_task_prompt(args.task, row) for row in batch]
        rendered = [
            _render_prompt(tokenizer, prompt, True, False) for prompt in prompts
        ]
        completions, usage = _generate_source(
            source,
            tokenizer,
            rendered,
            max_new_tokens=args.max_new_tokens,
            stop_ids=stop_ids,
        )
        for row, prompt, completion, (used, hit_cap) in zip(
            batch, prompts, completions, usage, strict=True
        ):
            gold = task_contract["gold"](row)
            prediction = task_contract["extract"](completion)
            correct = bool(task_contract["match"](prediction, gold))
            identity = hashlib.sha256(f"{args.task}\0{prompt}".encode()).hexdigest()
            output_rows.append(
                {
                    "schema": DRAFT_SCHEMA,
                    "task": args.task,
                    "identity_sha256": identity,
                    "task_prompt": prompt,
                    "gold": gold,
                    "source_completion": completion,
                    "source_prediction": prediction,
                    "source_correct": correct,
                    "source_generated_tokens": used,
                    "source_exhausted": hit_cap,
                }
            )
            generated_tokens += used
            exhausted += int(hit_cap)
        print(f"[vcr1-drafts] completed={len(output_rows)}/{len(selected)}", flush=True)

    elapsed = time.monotonic() - started
    payload = {
        "schema": DRAFT_SCHEMA,
        "status": "complete",
        "task": args.task,
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "count": len(output_rows),
        "subset_seed": args.subset_seed,
        "generation_seed": args.generation_seed,
        "max_new_tokens": args.max_new_tokens,
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "source_checkpoint_update": source_update,
        "ordered_identity_sha256": _ordered_identity_sha256(output_rows),
        "correct": sum(row["source_correct"] for row in output_rows),
        "generated_tokens": generated_tokens,
        "exhausted": exhausted,
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "rows": output_rows,
    }
    _atomic_json(args.output, payload)
    return payload


def correct_drafts(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if _sha256_file(args.source_checkpoint) != args.source_checkpoint_sha256:
        raise VCR1EvalError("protected source checkpoint hash differs")
    draft_sha256 = _sha256_file(args.drafts)
    draft_payload = json.loads(args.drafts.read_text(encoding="utf-8"))
    rows = _validate_draft_payload(draft_payload, args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.arm == "plain":
        model, update, resolved_loader = _source_model(args)
        checkpoint_sha256 = None
    else:
        model, update, resolved_loader = _vcr_model(args)
        checkpoint_sha256 = _sha256_file(args.checkpoint)
    stop_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    task_contract = TASKS[args.task]
    results: list[dict[str, Any]] = []
    generated_tokens = 0
    exhausted = 0
    skipped_length = 0
    correction_prompt_limit = args.max_sequence_length - args.max_new_tokens
    if correction_prompt_limit <= args.workspace_slots + 8:
        raise VCR1EvalError("correction generation budget leaves no prompt capacity")
    for batch in _batches(rows, args.batch_size):
        admitted_rows: list[dict[str, Any]] = []
        tokenized = []
        for row in batch:
            tokens = tokenize_correction_example(
                tokenizer,
                str(row["task_prompt"]),
                str(row["source_completion"]),
                None,
                max_sequence_length=correction_prompt_limit,
                workspace_slots=args.workspace_slots,
            )
            if tokens is None:
                skipped_length += 1
                continue
            admitted_rows.append(row)
            tokenized.append(tokens)
        if not admitted_rows:
            continue
        completions, usage, validity = _generate_correction(
            model,
            tokenizer,
            tokenized,
            arm=args.arm,
            max_new_tokens=args.max_new_tokens,
            stop_ids=stop_ids,
        )
        for row, completion, (used, hit_cap), valid_probability in zip(
            admitted_rows, completions, usage, validity, strict=True
        ):
            prediction = task_contract["extract"](completion)
            correct = bool(task_contract["match"](prediction, row["gold"]))
            source_correct = bool(row["source_correct"])
            transition = (
                ("right" if source_correct else "wrong")
                + "_to_"
                + ("right" if correct else "wrong")
            )
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "task_prompt": row["task_prompt"],
                    "gold": row["gold"],
                    "source_completion": row["source_completion"],
                    "source_prediction": row["source_prediction"],
                    "source_correct": source_correct,
                    "corrected_completion": completion,
                    "corrected_prediction": prediction,
                    "corrected_correct": correct,
                    "transition": transition,
                    "draft_valid_probability": valid_probability,
                    "generated_tokens": used,
                    "exhausted": hit_cap,
                }
            )
            generated_tokens += used
            exhausted += int(hit_cap)
        print(f"[vcr1-correct] completed={len(results)}/{len(rows)}", flush=True)

    transitions = {
        key: sum(row["transition"] == key for row in results)
        for key in (
            "wrong_to_right",
            "right_to_right",
            "right_to_wrong",
            "wrong_to_wrong",
        )
    }
    validity_rows = [
        row for row in results if row["draft_valid_probability"] is not None
    ]
    validity_accuracy = None
    validity_brier = None
    validity_correct_mean = None
    validity_wrong_mean = None
    if validity_rows:
        validity_accuracy = sum(
            (float(row["draft_valid_probability"]) >= 0.5)
            == bool(row["source_correct"])
            for row in validity_rows
        ) / len(validity_rows)
        validity_brier = sum(
            (float(row["draft_valid_probability"]) - float(bool(row["source_correct"])))
            ** 2
            for row in validity_rows
        ) / len(validity_rows)
        correct_probabilities = [
            float(row["draft_valid_probability"])
            for row in validity_rows
            if row["source_correct"]
        ]
        wrong_probabilities = [
            float(row["draft_valid_probability"])
            for row in validity_rows
            if not row["source_correct"]
        ]
        validity_correct_mean = (
            sum(correct_probabilities) / len(correct_probabilities)
            if correct_probabilities
            else None
        )
        validity_wrong_mean = (
            sum(wrong_probabilities) / len(wrong_probabilities)
            if wrong_probabilities
            else None
        )
    elapsed = time.monotonic() - started
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "task": args.task,
        "arm": args.arm,
        "ablation": args.ablation,
        "drafts": str(args.drafts.resolve()),
        "drafts_sha256": draft_sha256,
        "draft_ordered_identity_sha256": _ordered_identity_sha256(rows),
        "evaluated_ordered_identity_sha256": _ordered_identity_sha256(results),
        "input_rows": len(rows),
        "evaluated_rows": len(results),
        "skipped_length": skipped_length,
        "source_correct": sum(row["source_correct"] for row in results),
        "corrected_correct": sum(row["corrected_correct"] for row in results),
        "net_correction": transitions["wrong_to_right"] - transitions["right_to_wrong"],
        "transitions": transitions,
        "validity_accuracy": validity_accuracy,
        "validity_brier": validity_brier,
        "validity_correct_mean": validity_correct_mean,
        "validity_wrong_mean": validity_wrong_mean,
        "generated_tokens": generated_tokens,
        "exhausted": exhausted,
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "max_sequence_length": args.max_sequence_length,
        "max_new_tokens": args.max_new_tokens,
        "generation_seed": args.generation_seed,
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
    parser.add_argument("--mode", choices=("drafts", "correct"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("auto", "causal"), default="causal")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("math500", "preformatted_short_answer"), required=True
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--drafts", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument(
        "--arm", choices=("plain", "vcr1", "role_blind"), default="plain"
    )
    parser.add_argument(
        "--ablation", choices=("normal", "reset", "swap_roles"), default="normal"
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--subset-seed", type=int, default=31)
    parser.add_argument("--generation-seed", type=int, default=2026080603)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--workspace-slots", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if (
        min(
            args.count,
            args.batch_size,
            args.max_sequence_length,
            args.workspace_slots,
            args.max_new_tokens,
        )
        <= 0
    ):
        parser.error("VCR1 evaluation dimensions must be positive")
    if args.mode == "drafts" and args.data is None:
        parser.error("draft mode requires --data")
    if args.mode == "correct" and args.drafts is None:
        parser.error("correct mode requires --drafts")
    if args.mode == "correct" and args.arm != "plain" and args.checkpoint is None:
        parser.error("trained correction arm requires --checkpoint")
    if (
        args.mode == "correct"
        and args.arm != "plain"
        and args.checkpoint_sha256 is None
    ):
        parser.error("trained correction arm requires --checkpoint-sha256")
    if args.arm == "plain" and args.ablation != "normal":
        parser.error("plain correction supports only the normal ablation")
    return args


def main() -> int:
    args = parse_args()
    report = build_drafts(args) if args.mode == "drafts" else correct_drafts(args)
    print(
        f"[vcr1-eval] mode={args.mode} task={args.task} "
        f"correct={report.get('corrected_correct', report.get('correct'))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
