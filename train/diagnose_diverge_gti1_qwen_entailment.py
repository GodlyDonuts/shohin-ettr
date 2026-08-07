#!/usr/bin/env python3
"""Read-only Qwen entailment ceiling after the closed DIVERGE-GTI1 gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from diverge_gti1_runtime import canonical_query_text, expected_transaction
from eval_diverge_ccr1 import _referent_records
from eval_diverge_pqi1 import _load_board, sha256_path


SCHEMA = "shohin-diverge-gti1-qwen-entailment-attribution-v1"
SYSTEM = (
    "Judge whether a claim follows from an instruction. "
    "Answer exactly YES or NO without explanation."
)
ClaimControl = Literal["normal", "scrub_context", "swap_mentions"]


class QwenEntailmentError(RuntimeError):
    """The fixed read-only Qwen attribution contract was violated."""


def _prompt(tokenizer: Any, record: Mapping[str, Any], candidate: int, control: ClaimControl) -> str:
    if candidate not in (0, 1):
        raise QwenEntailmentError("candidate differs")
    query = canonical_query_text(record, control=control)
    target = "alpha" if candidate == 0 else "beta"
    distractor = "beta" if candidate == 0 else "alpha"
    user = (
        f"Instruction: {query}\n"
        f"Claim: {target} is the requested answer source and {distractor} is the distractor."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"System: {SYSTEM}\nUser: {user}\nAssistant:"


def _ids(tokenizer: Any, text: str) -> list[int]:
    values = list(tokenizer.encode(text, add_special_tokens=False))
    if not values:
        raise QwenEntailmentError("fixed text tokenized empty")
    return values


@torch.no_grad()
def _candidate_entailment_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    control: ClaimControl,
) -> tuple[torch.Tensor, tuple[int, int]]:
    answer_ids = tuple(tuple(_ids(tokenizer, answer)) for answer in ("YES", "NO"))
    rows: list[tuple[int, int, int, list[int], list[int]]] = []
    for row, record in enumerate(records):
        for candidate in (0, 1):
            prompt = _ids(tokenizer, _prompt(tokenizer, record, candidate, control))
            for answer, suffix in enumerate(answer_ids):
                rows.append((row, candidate, answer, prompt, list(suffix)))

    log_likelihoods = torch.empty((len(records), 2, 2), dtype=torch.float32)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise QwenEntailmentError("Qwen tokenizer exposes no pad or EOS token")
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        maximum = max(len(prompt) + len(suffix) - 1 for *_, prompt, suffix in batch)
        inputs = torch.full(
            (len(batch), maximum), int(pad_id), dtype=torch.long, device=device
        )
        attention = torch.zeros_like(inputs)
        for index, (*_, prompt, suffix) in enumerate(batch):
            sequence = prompt + suffix
            inputs[index, : len(sequence) - 1] = torch.tensor(
                sequence[:-1], dtype=torch.long, device=device
            )
            attention[index, : len(sequence) - 1] = 1
        output = model(input_ids=inputs, attention_mask=attention, use_cache=False)
        probabilities = F.log_softmax(output.logits.float(), dim=-1)
        for index, (row, candidate, answer, prompt, suffix) in enumerate(batch):
            positions = torch.arange(
                len(prompt) - 1,
                len(prompt) + len(suffix) - 1,
                device=device,
            )
            target = torch.tensor(suffix, dtype=torch.long, device=device)
            log_likelihoods[row, candidate, answer] = probabilities[
                index, positions, target
            ].sum().cpu()
    return log_likelihoods[:, :, 0] - log_likelihoods[:, :, 1], (
        len(answer_ids[0]),
        len(answer_ids[1]),
    )


def _score(
    records: Sequence[Mapping[str, Any]],
    scores: torch.Tensor,
    *,
    map_swapped_back: bool,
) -> dict[str, Any]:
    raw = scores.argmax(dim=-1).tolist()
    predictions = [1 - value for value in raw] if map_swapped_back else raw
    overall = Counter()
    by_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    margins = []
    for row, (record, prediction) in enumerate(zip(records, predictions, strict=True)):
        expected = expected_transaction(record)
        exact = int(prediction) == expected
        comparison = 1 - expected if map_swapped_back else expected
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
        "score_sha256": hashlib.sha256(
            scores.contiguous().numpy().tobytes()
        ).hexdigest(),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing Qwen attribution: {args.output}")
    if not torch.cuda.is_available():
        raise SystemExit("Qwen attribution requires CUDA")

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    board = _load_board(args.data, args.data_sha256, "development")
    records = [row for row in _referent_records(board) if row["stage"] == "QUERY"]
    controls = {}
    token_lengths = None
    for control in ("normal", "scrub_context", "swap_mentions"):
        scores, lengths = _candidate_entailment_scores(
            model,
            tokenizer,
            records,
            device=torch.device("cuda:0"),
            batch_size=args.batch_size,
            control=control,
        )
        if token_lengths is None:
            token_lengths = lengths
        elif lengths != token_lengths:
            raise SystemExit("Qwen attribution answer token lengths drift")
        controls[control] = _score(
            records,
            scores,
            map_swapped_back=control == "swap_mentions",
        )
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Zero-training Qwen3.5-0.8B candidate-entailment attribution on the "
            "opened GTI1 development board; not a promoted semantic interface."
        ),
        "model_root": str(args.model_root),
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "system_sha256": hashlib.sha256(SYSTEM.encode("ascii")).hexdigest(),
        "answer_token_lengths": list(token_lengths or ()),
        "controls": controls,
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "normal": controls["normal"],
                "scrub_context": controls["scrub_context"]["overall"],
                "swap_mentions": controls["swap_mentions"]["overall"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
