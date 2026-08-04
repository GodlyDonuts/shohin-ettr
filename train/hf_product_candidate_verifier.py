#!/usr/bin/env python3
"""Score autonomous candidates with a counterbalanced semantic self-verifier."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


SCHEMA = "shohin-hf-product-candidate-verifier-v1"


class CandidateVerifierError(RuntimeError):
    """The semantic-verifier contract was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verifier_prompt(question: str, completion: str, *, reversed_labels: bool) -> str:
    if reversed_labels:
        mapping = "A means incorrect; B means correct"
    else:
        mapping = "A means correct; B means incorrect"
    return (
        "Act as a strict solution verifier. Check every important logical, numerical, "
        "and factual step. A plausible final answer is not enough if the reasoning is "
        "invalid.\n\n"
        f"Problem:\n{question}\n\nCandidate solution:\n{completion}\n\n"
        f"Verdict labels: {mapping}. Reply with only A or B.\nVerdict:"
    )


def counterbalanced_score(
    forward_score: float,
    reversed_score: float,
) -> float:
    """Combine oriented A/B log odds while cancelling a fixed label preference."""

    return 0.5 * (forward_score - reversed_score)


def _load_grouped(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get("identity_sha256") or "")
            if not identity:
                raise CandidateVerifierError("candidate identity is missing")
            grouped.setdefault(identity, []).append(row)
    if not grouped:
        raise CandidateVerifierError("candidate source is empty")
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
        if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
            raise CandidateVerifierError("candidate sample indices differ")
    return grouped


def _prompt_token_ids(tokenizer: Any, prompt: str, max_length: int) -> list[int]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(token_ids) <= max_length:
        return token_ids
    # Keep the verifier instruction and final answer region. Candidate generation
    # is already capped, so this path is rare and explicitly reported.
    head = max_length // 4
    return token_ids[:head] + token_ids[-(max_length - head) :]


def _score_pair(
    model: Any,
    tokenizer: Any,
    forward_prompt: str,
    reversed_prompt: str,
    max_length: int,
) -> tuple[float, float, bool]:
    import torch

    rows = [
        _prompt_token_ids(tokenizer, forward_prompt, max_length),
        _prompt_token_ids(tokenizer, reversed_prompt, max_length),
    ]
    truncated = any(
        len(tokenizer.encode(prompt, add_special_tokens=True)) > max_length
        for prompt in (forward_prompt, reversed_prompt)
    )
    width = max(len(row) for row in rows)
    input_ids = torch.full(
        (2, width), tokenizer.pad_token_id, dtype=torch.long, device="cuda:0"
    )
    attention = torch.zeros((2, width), dtype=torch.long, device="cuda:0")
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, device="cuda:0")
        attention[index, : len(row)] = 1
    if hasattr(model, "text_model") and hasattr(model, "lm_head"):
        text_model = model.text_model
        lm_head = model.lm_head
    else:
        from hf_product_reasoning_train import resolve_product_backbone_layout

        text_model, lm_head, _, _ = resolve_product_backbone_layout(model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = text_model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
        )
    lengths = attention.sum(dim=1) - 1
    final_hidden = outputs.last_hidden_state[
        torch.arange(2, device="cuda:0"), lengths
    ]
    logits = lm_head(final_hidden).float()
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    if len(a_ids) != 1 or len(b_ids) != 1 or a_ids == b_ids:
        raise CandidateVerifierError("A/B verdict labels are not distinct single tokens")
    logp = torch.log_softmax(logits, dim=-1)
    # Both values are log P(A) - log P(B). The second mapping reverses meaning.
    return (
        float(logp[0, a_ids[0]] - logp[0, b_ids[0]]),
        float(logp[1, a_ids[0]] - logp[1, b_ids[0]]),
        truncated,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import _load_model

    for output in (args.output, args.report):
        if output.exists():
            raise CandidateVerifierError(f"refusing existing output: {output}")
    grouped = _load_grouped(args.candidates)
    identities = list(grouped)
    if args.skip < 0 or args.count <= 0 or args.skip + args.count > len(identities):
        raise CandidateVerifierError("requested identity slice is outside candidate source")
    selected_identities = identities[args.skip : args.skip + args.count]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    output_rows: list[dict[str, Any]] = []
    truncated = 0
    empty_completions = 0
    for completed, identity in enumerate(selected_identities, start=1):
        for row in grouped[identity]:
            question = str(row.get("question") or "")
            completion = str(row.get("completion") or "")
            if not question:
                raise CandidateVerifierError("candidate question is empty")
            if completion:
                forward_prompt = verifier_prompt(
                    question, completion, reversed_labels=False
                )
                reversed_prompt = verifier_prompt(
                    question, completion, reversed_labels=True
                )
                forward_score, reversed_score, was_truncated = _score_pair(
                    model,
                    tokenizer,
                    forward_prompt,
                    reversed_prompt,
                    args.max_sequence_length,
                )
                score = counterbalanced_score(forward_score, reversed_score)
            else:
                # Empty autonomous samples contain no solution to verify and must
                # never outrank any finite semantic verdict.
                forward_score = reversed_score = 0.0
                score = -1e9
                was_truncated = False
                empty_completions += 1
            truncated += int(was_truncated)
            output_rows.append(
                {
                    "schema": SCHEMA,
                    "identity_sha256": identity,
                    "task": row["task"],
                    "sample_index": int(row["sample_index"]),
                    "prediction": row.get("prediction"),
                    "completion": completion,
                    "correct": bool(row["correct"]),
                    "forward_a_minus_b_logp": forward_score,
                    "reversed_a_minus_b_logp": reversed_score,
                    "verifier_score": score,
                    "prompt_truncated": was_truncated,
                    "empty_completion": not completion,
                }
            )
        if completed % 10 == 0 or completed == len(selected_identities):
            print(
                f"[candidate-verifier] groups={completed}/{len(selected_identities)} "
                f"candidates={len(output_rows)} truncated={truncated}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    elapsed = time.monotonic() - started
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_update": adapter_metadata.get("update") if adapter_metadata else None,
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": sha256_file(args.candidates),
        "skip": args.skip,
        "count": args.count,
        "scored_candidates": len(output_rows),
        "max_sequence_length": args.max_sequence_length,
        "prompt_truncated": truncated,
        "empty_completions": empty_completions,
        "counterbalanced_labels": True,
        "selector_reads_gold": False,
        "elapsed_seconds": elapsed,
        "candidates_per_second": len(output_rows) / max(elapsed, 1e-9),
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_suffix(args.report.suffix + ".partial")
    with temporary_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_report, args.report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", default="auto")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
