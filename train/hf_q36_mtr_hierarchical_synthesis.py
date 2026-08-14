#!/usr/bin/env python3
"""Verify a synthesized Q36 answer against two independently derived alternatives."""

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
from hf_q36_mtr_synthesize_trajectories import validate_aligned_metadata
from q36_mtr_roles import MODEL_REVISION

SCHEMA = "shohin-q36-mtr-model-draft-v1"
REPORT_SCHEMA = "shohin-q36-mtr-hierarchical-synthesis-shard-v1"
HIERARCHY_SCHEMA = "shohin-q36-mtr-hierarchical-synthesis-v1"
SHARDS = 16
ROWS = 1_289
SEED = 2026081423
MAX_NEW_TOKENS = 768
TASKS = {"bbh_logic", "math500", "mbpp"}


class Q36MTRHierarchicalSynthesisError(RuntimeError):
    """Hierarchical synthesis inputs, generation, or custody differ."""


def hierarchical_prompt(
    source_prompt: str,
    synthesis: str,
    stacked: str,
    self_refinement: str,
) -> str:
    values = (source_prompt, synthesis, stacked, self_refinement)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise Q36MTRHierarchicalSynthesisError("hierarchical prompt input differs")
    return (
        "Produce the single most reliable answer to the original problem. Candidate A "
        "is the current integrated solution produced by reconciling three independent "
        "reasoning trajectories. Preserve Candidate A unless checking it against the "
        "original problem reveals a concrete error. Candidates B and C are independent "
        "alternatives: use them as evidence to locate and repair such an error, not as "
        "votes. Recompute disputed steps yourself. Do not mention the candidates or this "
        "review process, and return only one final solution in the original problem's "
        "requested output format.\n\n"
        f"Original problem:\n{source_prompt}\n\n"
        f"Candidate A — integrated solution:\n{synthesis}\n\n"
        f"Candidate B — preserved alternative:\n{stacked}\n\n"
        f"Candidate C — independent refinement:\n{self_refinement}\n\n"
        "Return the verified final solution in the original problem's requested output "
        "format."
    )


def load_candidate_group(
    paths: list[Path], *, expected_paths: int
) -> dict[str, dict[str, Any]]:
    if len(paths) != expected_paths:
        raise Q36MTRHierarchicalSynthesisError("candidate path geometry differs")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise Q36MTRHierarchicalSynthesisError("candidate path differs")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                identity = row.get("identity_sha256")
                if row.get("split", "development") != "development":
                    continue
                if (
                    row.get("schema") not in {SCHEMA, "shohin-q36-mtr-candidate-v1"}
                    or not isinstance(identity, str)
                    or len(identity) != 64
                    or identity in result
                    or row.get("task") not in TASKS
                    or not isinstance(row.get("completion"), str)
                    or not row["completion"].strip()
                    or isinstance(row.get("generated_tokens"), bool)
                    or not isinstance(row.get("generated_tokens"), int)
                    or row["generated_tokens"] <= 0
                    or not isinstance(row.get("max_token_exhausted"), bool)
                ):
                    raise Q36MTRHierarchicalSynthesisError("candidate payload differs")
                result[identity] = row
    if len(result) != ROWS:
        raise Q36MTRHierarchicalSynthesisError("candidate coverage differs")
    return result


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
        or not 0 <= args.shard_index < SHARDS
    ):
        raise Q36MTRHierarchicalSynthesisError("generation settings differ")
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTRHierarchicalSynthesisError("environment receipt differs")
    environment = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        environment.get("schema") != "shohin-q36-mtr-environment-v1"
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTRHierarchicalSynthesisError("environment contract differs")

    all_sources, freeze_report = load_sources(
        args.train_source, args.development_source, args.freeze_report
    )
    sources = {
        row["identity_sha256"]: row
        for row in all_sources
        if row["split"] == "development"
    }
    groups = {
        "integrated_synthesis": load_candidate_group(
            args.synthesis_candidates, expected_paths=16
        ),
        "stacked_preserved": load_candidate_group(
            args.stacked_candidates, expected_paths=1
        ),
        "self_refinement": load_candidate_group(
            args.self_refinement_candidates, expected_paths=8
        ),
    }
    if len(sources) != ROWS or any(
        set(group) != set(sources) for group in groups.values()
    ):
        raise Q36MTRHierarchicalSynthesisError("identity coverage differs")
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
        prompts = [
            hierarchical_prompt(
                sources[identity]["source_prompt"],
                groups["integrated_synthesis"][identity]["completion"],
                groups["stacked_preserved"][identity]["completion"],
                groups["self_refinement"][identity]["completion"],
            )
            for identity in identities
        ]
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
        for identity, prompt, completion, (token_count, hit_limit) in zip(
            identities, prompts, completions, usage, strict=True
        ):
            if not isinstance(completion, str) or not completion.strip():
                raise Q36MTRHierarchicalSynthesisError(
                    "generation emitted an empty completion"
                )
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
                    "hierarchical_synthesis": {
                        "schema": HIERARCHY_SCHEMA,
                        "input_roles": list(groups),
                        "development_labels_read": 0,
                    },
                }
            )
            generated_tokens += int(token_count)
            exhausted += int(hit_limit)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output_sha256 = _atomic_lines(args.output, outputs)
    candidate_paths = {
        "integrated_synthesis": args.synthesis_candidates,
        "stacked_preserved": args.stacked_candidates,
        "self_refinement": args.self_refinement_candidates,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "hierarchical_synthesis_with_conservative_retention",
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
        "candidate_sha256": {
            name: [sha256_file(path) for path in paths]
            for name, paths in candidate_paths.items()
        },
        "generation_mode": "greedy",
        "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
        "rendered_chat_tokenization": "add_special_tokens_false",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
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
    parser.add_argument(
        "--synthesis-candidates", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--stacked-candidates", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--self-refinement-candidates", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=SHARDS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
