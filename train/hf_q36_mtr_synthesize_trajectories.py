#!/usr/bin/env python3
"""Synthesize three independent Q36 trajectories with the trained reviser."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from hf_q36_mtr_generate_drafts import (
    _atomic_json,
    _atomic_lines,
    load_sources,
    sha256_file,
)
from q36_mtr_roles import MODEL_REVISION, Q36MTRRoleError, validate_contract
import train_apply_q36_mtr_sparse_router as sparse

SCHEMA = "shohin-q36-mtr-model-draft-v1"
REPORT_SCHEMA = "shohin-q36-mtr-trajectory-synthesis-shard-v1"
SYNTHESIS_SCHEMA = "shohin-q36-mtr-trajectory-synthesis-v1"
SHARDS = 16
ROWS = 1_289
SEED = 2026081421
MAX_NEW_TOKENS = 768


class Q36MTRSynthesisError(RuntimeError):
    """Trajectory-synthesis model, inputs, or outputs differ."""


def _rotation(identity: str, offset: int = 0) -> int:
    if offset not in {0, 1, 2}:
        raise Q36MTRSynthesisError("synthesis rotation offset differs")
    digest = hashlib.sha256(f"q36-synthesis\0{identity}".encode()).digest()
    return (int.from_bytes(digest[:4], "big") + offset) % len(sparse.LINEAGES)


def synthesis_prompt(
    source_prompt: str,
    identity: str,
    candidates: list[dict[str, Any]],
    rotation_offset: int = 0,
) -> tuple[str, list[str]]:
    if len(candidates) != len(sparse.LINEAGES):
        raise Q36MTRSynthesisError("synthesis candidate geometry differs")
    rotation = _rotation(identity, rotation_offset)
    order = [
        (rotation + offset) % len(sparse.LINEAGES)
        for offset in range(len(sparse.LINEAGES))
    ]
    labels = ("A", "B", "C")
    attempts: list[str] = []
    ordered_lineages: list[str] = []
    for label, index in zip(labels, order, strict=True):
        completion = candidates[index].get("completion")
        if not isinstance(completion, str) or not completion.strip():
            raise Q36MTRSynthesisError("synthesis completion differs")
        attempts.append(f"Internal attempt {label}:\n{completion}")
        ordered_lineages.append(sparse.LINEAGES[index])
    prompt = (
        "Solve the original problem using the useful reasoning in three independent "
        "internal attempts. Check every attempt against the problem, reconcile "
        "disagreements, and recompute anything uncertain. Do not vote blindly, do not "
        "mention the attempts, and return one best solution in the original problem's "
        "requested output format.\n\n"
        f"Original problem:\n{source_prompt}\n\n"
        "Internal draft:\n"
        + "\n\n".join(attempts)
        + "\n\nFollow the original problem's requested output format."
        f"\n\nOriginal problem:\n{source_prompt}"
    )
    return prompt, ordered_lineages


def validate_aligned_metadata(metadata: dict[str, Any]) -> None:
    try:
        validate_contract(metadata, "aligned")
    except Q36MTRRoleError as error:
        raise Q36MTRSynthesisError(str(error)) from error
    if (
        metadata.get("role") != "aligned"
        or metadata.get("internal_draft_visible") is not True
        or metadata.get("draft_control") != "normal"
        or metadata.get("draft_information_available") is not True
    ):
        raise Q36MTRSynthesisError("synthesis reviser contract differs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import (
        GENERATED_ONLY_SEQUENCE_CONTRACT,
        _generate_completions,
        _generation_stop_token_ids,
        _render_prompt,
    )
    from hf_q36_mtr_evaluate import (
        load_q36_adapter_model,
        q36_nonpadding_prompt_tokens,
    )

    if (
        args.model_revision != MODEL_REVISION
        or args.seed != SEED
        or args.shard_count != SHARDS
        or args.max_new_tokens != MAX_NEW_TOKENS
        or args.batch_size != 2
        or args.rotation_offset not in {0, 1, 2}
        or not 0 <= args.shard_index < SHARDS
    ):
        raise Q36MTRSynthesisError("synthesis settings differ")
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTRSynthesisError("synthesis environment receipt differs")
    environment = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        environment.get("schema") != "shohin-q36-mtr-environment-v1"
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTRSynthesisError("synthesis environment contract differs")

    all_sources, freeze_report = load_sources(
        args.train_source, args.development_source, args.freeze_report
    )
    sources = {
        row["identity_sha256"]: row
        for row in all_sources
        if row["split"] == "development"
    }
    owners = [
        sparse.load_development_candidates(paths)
        for paths in (
            args.current_candidates,
            args.owner71_candidates,
            args.owner8_candidates,
        )
    ]
    if len(sources) != ROWS or any(set(owner) != set(sources) for owner in owners):
        raise Q36MTRSynthesisError("synthesis identity coverage differs")
    ordered_identities = sorted(sources)
    row_start = ROWS * args.shard_index // args.shard_count
    row_end = ROWS * (args.shard_index + 1) // args.shard_count
    shard_identities = ordered_identities[row_start:row_end]

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = load_q36_adapter_model(
        args.model_root, args.aligned_checkpoint
    )
    validate_aligned_metadata(metadata)
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    outputs: list[dict[str, Any]] = []
    prompt_tokens = generated_tokens = exhausted = 0
    started = time.monotonic()
    for offset in range(0, len(shard_identities), args.batch_size):
        identities = shard_identities[offset : offset + args.batch_size]
        prompts: list[str] = []
        permutations: list[list[str]] = []
        for identity in identities:
            candidates = [owner[identity] for owner in owners]
            prompt, order = synthesis_prompt(
                sources[identity]["source_prompt"],
                identity,
                candidates,
                args.rotation_offset,
            )
            prompts.append(prompt)
            permutations.append(order)
        rendered = [
            _render_prompt(tokenizer, prompt, True, False) for prompt in prompts
        ]
        prompt_tokens += q36_nonpadding_prompt_tokens(tokenizer, rendered)
        batch_started = time.monotonic()
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
            add_special_tokens=False,
        )
        batch_wall_seconds = (time.monotonic() - batch_started) / len(identities)
        for identity, prompt, order, completion, (token_count, hit_limit) in zip(
            identities, prompts, permutations, completions, usage, strict=True
        ):
            if not isinstance(completion, str) or not completion.strip():
                raise Q36MTRSynthesisError("synthesis emitted an empty completion")
            source = sources[identity]
            outputs.append(
                {
                    "schema": SCHEMA,
                    "identity_sha256": identity,
                    "split": "development",
                    "task": source["task"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "owner_checkpoint_sha256": sha256_file(args.aligned_checkpoint),
                    "model_revision": MODEL_REVISION,
                    "completion": completion,
                    "generated_tokens": int(token_count),
                    "max_token_exhausted": bool(hit_limit),
                    "finish_reason": "length" if hit_limit else "stop",
                    "wall_seconds": batch_wall_seconds,
                    "trajectory_synthesis": {
                        "schema": SYNTHESIS_SCHEMA,
                        "attempt_order": order,
                        "rotation_offset": args.rotation_offset,
                        "development_labels_read": 0,
                    },
                }
            )
            generated_tokens += int(token_count)
            exhausted += int(hit_limit)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output_sha256 = _atomic_lines(args.output, outputs)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "three_owner_revision_synthesis",
        "model_revision": MODEL_REVISION,
        "model_loader": loader,
        "aligned_checkpoint": str(args.aligned_checkpoint.resolve()),
        "aligned_checkpoint_sha256": sha256_file(args.aligned_checkpoint),
        "aligned_update": metadata["update"],
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "freeze_report_sha256": sha256_file(args.freeze_report),
        "freeze_identity_receipts": freeze_report["identity_receipts"],
        "train_source_sha256": sha256_file(args.train_source),
        "development_source_sha256": sha256_file(args.development_source),
        "owner_candidate_sha256": {
            lineage: [sha256_file(path) for path in paths]
            for lineage, paths in zip(
                sparse.LINEAGES,
                (
                    args.current_candidates,
                    args.owner71_candidates,
                    args.owner8_candidates,
                ),
                strict=True,
            )
        },
        "generation_mode": "greedy",
        "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
        "rendered_chat_tokenization": "add_special_tokens_false",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "rotation_offset": args.rotation_offset,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_rows": ROWS,
        "row_start": row_start,
        "row_end": row_end,
        "rows": len(outputs),
        "ordered_identity_sha256": hashlib.sha256(
            ("\n".join(row["identity_sha256"] for row in outputs) + "\n").encode()
        ).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_token_exhausted": exhausted,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "development_labels_read": 0,
        "capability_scored": False,
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--aligned-checkpoint", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    for owner in ("current", "owner71", "owner8"):
        parser.add_argument(
            f"--{owner}-candidates", type=Path, action="append", required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=SHARDS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rotation-offset", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
