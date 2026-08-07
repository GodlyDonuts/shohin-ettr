#!/usr/bin/env python3
"""Read-only autoregressive attribution for the closed DIVERGE-PQI1 gate."""

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
from tokenizers import Tokenizer

from diverge_ats1_data import BYTE_OFFSET, PAD_ID
from diverge_iem1_runtime import tensorize_queries
from diverge_pqi1_runtime import canonicalize_query
from eval_diverge_ccr1 import _referent_records
from eval_diverge_pqi1 import _load_board, sha256_path
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-pqi1-autoregressive-attribution-v1"
PROMPT_PREFIX = (
    "Read the instruction and identify the placeholder whose value is requested.\n"
    "Instruction: "
)
PROMPT_SUFFIX = "\nRequested placeholder:"
COMPLETIONS = (" alpha", " beta")


class PQI1AutoregressiveError(RuntimeError):
    """The fixed read-only attribution contract was violated."""


def _canonical_texts(records: Sequence[Mapping[str, Any]]) -> list[str]:
    ids, attention, symbols, _ = tensorize_queries(records, torch.device("cpu"))
    output = []
    for row in range(len(records)):
        length = int(attention[row].sum())
        raw = ids[row, 1:length] - BYTE_OFFSET
        if torch.any((raw < 0) | (raw > 127)):
            raise PQI1AutoregressiveError("query source is not ASCII")
        text = bytes(int(value) for value in raw.tolist()).decode("ascii")
        masks = [symbols[row, group, 1:length].tolist() for group in range(2)]
        output.append(canonicalize_query(text, masks).text)
    return output


def _token_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    if not ids:
        raise PQI1AutoregressiveError("fixed attribution text tokenized empty")
    return ids


@torch.no_grad()
def _candidate_scores(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    completion_ids = [_token_ids(tokenizer, value) for value in COMPLETIONS]
    if len(completion_ids[0]) != len(completion_ids[1]):
        raise PQI1AutoregressiveError("alpha/beta completion token lengths differ")
    rows: list[tuple[int, int, list[int], list[int]]] = []
    for row, prompt in enumerate(prompts):
        prompt_ids = _token_ids(tokenizer, prompt)
        for candidate, suffix in enumerate(completion_ids):
            if len(prompt_ids) + len(suffix) > int(model.cfg.seq_len):
                raise PQI1AutoregressiveError("attribution sequence exceeds backbone context")
            rows.append((row, candidate, prompt_ids, suffix))

    scores = torch.empty((len(prompts), len(COMPLETIONS)), dtype=torch.float32)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        maximum = max(len(prompt) + len(suffix) - 1 for _, _, prompt, suffix in batch)
        inputs = torch.full(
            (len(batch), maximum), PAD_ID, dtype=torch.long, device=device
        )
        for index, (_, _, prompt, suffix) in enumerate(batch):
            sequence = prompt + suffix
            inputs[index, : len(sequence) - 1] = torch.tensor(
                sequence[:-1], dtype=torch.long, device=device
            )
        logits, _ = model(inputs)
        log_probabilities = F.log_softmax(logits.float(), dim=-1)
        for index, (row, candidate, prompt, suffix) in enumerate(batch):
            positions = torch.arange(
                len(prompt) - 1,
                len(prompt) + len(suffix) - 1,
                device=device,
            )
            target = torch.tensor(suffix, dtype=torch.long, device=device)
            scores[row, candidate] = log_probabilities[index, positions, target].sum().cpu()
    return scores


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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--backbone-name", choices=("shohin", "smollm2"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing attribution output: {args.output}")
    for path, expected, label in (
        (args.base, args.base_sha256, "base"),
        (args.tokenizer, args.tokenizer_sha256, "tokenizer"),
        (args.data, args.data_sha256, "data"),
    ):
        if sha256_path(path) != expected:
            raise SystemExit(f"PQI1 autoregressive {label} hash differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("PQI1 autoregressive attribution requested unavailable CUDA")

    device = torch.device(args.device)
    board = _load_board(args.data, args.data_sha256, "development")
    records = [row for row in _referent_records(board) if row["stage"] == "QUERY"]
    canonical = _canonical_texts(records)
    prompts = [PROMPT_PREFIX + value + PROMPT_SUFFIX for value in canonical]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model, _, receipt = load_frozen_pointer_backbone(args.base, device=device)
    scores = _candidate_scores(
        model, tokenizer, prompts, device=device, batch_size=args.batch_size
    )

    overall = Counter()
    by_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    predictions = scores.argmax(dim=-1).tolist()
    margins = []
    for row, (record, prediction) in enumerate(zip(records, predictions, strict=True)):
        roles = [int(value) for value in record["symbol_role_ids"]]
        expected = roles.index(0)
        exact = int(prediction) == expected
        margin = float(scores[row, expected] - scores[row, 1 - expected])
        margins.append(margin)
        for counter in (
            overall,
            by_mode[str(record["mode"])],
            by_renderer[str(int(record["renderer"]))],
        ):
            counter["total"] += 1
            counter["exact"] += exact

    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Read-only fixed-prompt attribution of the closed PQI1 development failure; "
            "not a promoted semantic interface or a confirmation result."
        ),
        "backbone_name": args.backbone_name,
        "base": str(args.base),
        "base_sha256": args.base_sha256,
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": args.tokenizer_sha256,
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "prompt_sha256": hashlib.sha256(
            (PROMPT_PREFIX + "{QUERY}" + PROMPT_SUFFIX).encode("ascii")
        ).hexdigest(),
        "completions": list(COMPLETIONS),
        "completion_token_lengths": [
            len(_token_ids(tokenizer, value)) for value in COMPLETIONS
        ],
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
        "backbone_receipt": {
            "checkpoint_format": receipt.checkpoint_format,
            "base_step": receipt.base_step,
            "initialization": receipt.initialization,
            "base_import": receipt.base_import,
            "base_rms_norm_eps": receipt.base_rms_norm_eps,
        },
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": sha256_path(args.output),
        "overall": report["overall"],
        "by_renderer": report["by_renderer"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
