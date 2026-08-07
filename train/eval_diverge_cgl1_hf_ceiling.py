#!/usr/bin/env python3
"""One-control pretrained-backbone ceiling for the open CGL1 development board."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from diverge_cgl1_runtime import render_claim_prompt
from diverge_gti1_runtime import expected_transaction
from eval_diverge_ccr1 import _referent_records
from eval_diverge_pqi1 import _load_board, sha256_path


SCHEMA = "shohin-diverge-cgl1-hf-ceiling-v1"


def _ids(tokenizer: Any, text: str) -> list[int]:
    values = list(tokenizer.encode(text, add_special_tokens=False))
    if not values:
        raise RuntimeError("CGL1 ceiling text tokenized empty")
    return values


@torch.no_grad()
def _scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    control: str,
    batch_size: int,
) -> torch.Tensor:
    suffixes = tuple(tuple(_ids(tokenizer, value)) for value in (" YES", " NO"))
    rows: list[tuple[int, int, int, list[int], tuple[int, ...]]] = []
    for row, record in enumerate(records):
        for candidate in (0, 1):
            prompt = _ids(
                tokenizer,
                render_claim_prompt(record, candidate, control=control),
            )
            for answer, suffix in enumerate(suffixes):
                rows.append((row, candidate, answer, prompt, suffix))

    values = torch.empty((len(records), 2, 2), dtype=torch.float32)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("CGL1 ceiling tokenizer has no pad or EOS token")
    device = torch.device("cuda:0")
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        maximum = max(len(prompt) + len(suffix) - 1 for *_, prompt, suffix in batch)
        inputs = torch.full(
            (len(batch), maximum), int(pad_id), dtype=torch.long, device=device
        )
        attention = torch.zeros_like(inputs)
        for index, (*_, prompt, suffix) in enumerate(batch):
            sequence = prompt + list(suffix)
            inputs[index, : len(sequence) - 1] = torch.tensor(
                sequence[:-1], dtype=torch.long, device=device
            )
            attention[index, : len(sequence) - 1] = 1
        logits = model(
            input_ids=inputs, attention_mask=attention, use_cache=False
        ).logits
        probabilities = F.log_softmax(logits.float(), dim=-1)
        for index, (row, candidate, answer, prompt, suffix) in enumerate(batch):
            positions = torch.arange(
                len(prompt) - 1,
                len(prompt) + len(suffix) - 1,
                device=device,
            )
            target = torch.tensor(suffix, dtype=torch.long, device=device)
            values[row, candidate, answer] = probabilities[
                index, positions, target
            ].sum().cpu()
    return values[:, :, 0] - values[:, :, 1]


def _summarize(
    records: Sequence[Mapping[str, Any]], scores: torch.Tensor, *, control: str
) -> dict[str, Any]:
    raw = scores.argmax(dim=-1).tolist()
    predictions = [1 - value for value in raw] if control == "swap_mentions" else raw
    overall = Counter()
    by_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    margins = []
    for row, (record, prediction) in enumerate(zip(records, predictions, strict=True)):
        expected = expected_transaction(record)
        exact = int(prediction) == expected
        comparison = 1 - expected if control == "swap_mentions" else expected
        margins.append(float(scores[row, comparison] - scores[row, 1 - comparison]))
        for counter in (
            overall,
            by_mode[str(record["mode"])],
            by_renderer[str(int(record["renderer"]))],
        ):
            counter["total"] += 1
            counter["exact"] += exact
    return {
        "overall": dict(overall),
        "by_mode": {key: dict(value) for key, value in sorted(by_mode.items())},
        "by_renderer": {
            key: dict(value) for key, value in sorted(by_renderer.items())
        },
        "mean_signed_margin": sum(margins) / len(margins),
        "prediction_sha256": hashlib.sha256(
            json.dumps(predictions, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "score_sha256": hashlib.sha256(scores.numpy().tobytes()).hexdigest(),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument(
        "--control", choices=("normal", "scrub_context", "swap_mentions"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available():
        raise SystemExit("CGL1 ceiling output exists or CUDA is unavailable")

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoTokenizer,
    )

    config = AutoConfig.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    loader = (
        AutoModelForMultimodalLM
        if str(getattr(config, "model_type", "")).startswith("qwen3_5")
        else AutoModelForCausalLM
    )
    model = loader.from_pretrained(
        args.model_root,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    board = _load_board(args.data, args.data_sha256, "development")
    records = [row for row in _referent_records(board) if row["stage"] == "QUERY"]
    scores = _scores(
        model,
        tokenizer,
        records,
        control=args.control,
        batch_size=args.batch_size,
    )
    summary = _summarize(records, scores, control=args.control)
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Zero-training pretrained-backbone ceiling on the already-open CGL1 "
            "development board; never a promotion result."
        ),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "model_root": str(args.model_root),
        "model_config_sha256": sha256_path(args.model_root / "config.json"),
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "control": args.control,
        "result": summary,
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "model": args.model_name,
                "control": args.control,
                "result": summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
