#!/usr/bin/env python3
"""Generate source-only host-owned drafts for one pinned upward MoE host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from hf_q36_mtr_generate_drafts import (
    _atomic_json,
    _atomic_lines,
    exact_model_owned_completion,
    load_sources,
)
from hf_upward_moe_train_aligned import restore_exact_owner
from hf_upward_moe_train_owner import (
    OWNER_DATA_SHA256,
    SCHEMA as OWNER_TRAINING_SCHEMA,
    UpwardMoEOwnerTrainingError,
    _load_host,
)
from q36_mtr_roles import (
    DRAFT_IDENTITIES,
    DRAFT_MAX_NEW_TOKENS,
    DRAFT_SEED,
    DRAFT_SHARDS,
)
from upward_moe_role_lineage import load_role_checkpoint, sha256_file
from upward_moe_temporal_gate import MIXTRAL_SPEC, NEMOTRON_SPEC

SCHEMA = "shohin-upward-moe-model-draft-v1"
REPORT_SCHEMA = "shohin-upward-moe-draft-shard-v1"


class UpwardMoEDraftError(RuntimeError):
    """The upward-MoE host-owned draft contract differed."""


def host_spec(host: str) -> Any:
    if host == "nemotron-super":
        return NEMOTRON_SPEC
    if host == "mixtral-8x22b":
        return MIXTRAL_SPEC
    raise UpwardMoEDraftError("upward-MoE draft host differs")


def validate_owner_checkpoint(path: Path, spec: Any) -> dict[str, Any]:
    payload = load_role_checkpoint(path, spec)
    metadata = payload["metadata"]
    training = metadata["training_receipt"]
    if (
        metadata.get("role") != "owner"
        or training.get("schema") != OWNER_TRAINING_SCHEMA
        or training.get("role") != "owner"
        or training.get("host") != spec.host
        or training.get("data_sha256") != OWNER_DATA_SHA256
        or training.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise UpwardMoEDraftError("upward-MoE source owner lineage differs")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    from hf_product_reasoning_eval import (
        GENERATED_ONLY_SEQUENCE_CONTRACT,
        _generate_completions,
        _generation_stop_token_ids,
        _render_prompt,
    )
    from hf_q36_mtr_evaluate import q36_nonpadding_prompt_tokens

    spec = host_spec(args.host)
    if (
        args.seed != DRAFT_SEED
        or args.shard_count != DRAFT_SHARDS
        or args.max_new_tokens != DRAFT_MAX_NEW_TOKENS
        or args.batch_size != 1
        or not 0 <= args.shard_index < DRAFT_SHARDS
        or args.output.exists()
        or args.report.exists()
    ):
        raise UpwardMoEDraftError("upward-MoE draft settings differ")
    rows, freeze_report = load_sources(
        args.train_source, args.development_source, args.freeze_report
    )
    if len(rows) != DRAFT_IDENTITIES:
        raise UpwardMoEDraftError("upward-MoE draft source population differs")
    row_start = len(rows) * args.shard_index // args.shard_count
    row_end = len(rows) * (args.shard_index + 1) // args.shard_count
    shard_rows = rows[row_start:row_end]
    owner_payload = validate_owner_checkpoint(args.owner_checkpoint, spec)
    try:
        loaded = _load_host(args)
    except UpwardMoEOwnerTrainingError as error:
        raise UpwardMoEDraftError(str(error)) from error
    if loaded.spec != spec:
        raise UpwardMoEDraftError("upward-MoE loaded host differs")
    if loaded.tokenizer.pad_token_id is None:
        loaded.tokenizer.pad_token_id = loaded.tokenizer.eos_token_id
    owner_restore = restore_exact_owner(loaded.model, args.owner_checkpoint, spec)
    expected_state = owner_payload["metadata"]["final_trainable_state_sha256"]
    if owner_restore["owner_state_sha256"] != expected_state:
        raise UpwardMoEDraftError("upward-MoE owner restore differs")

    tokenizer = loaded.tokenizer
    generator = loaded.model.backbone
    generator.eval()
    loaded.model.reset_receipt()
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    outputs: list[dict[str, Any]] = []
    prompt_tokens = generated_tokens = exhausted = 0
    started = time.monotonic()
    for row in shard_rows:
        rendered = _render_prompt(tokenizer, str(row["source_prompt"]), True, False)
        prompt_tokens += q36_nonpadding_prompt_tokens(tokenizer, [rendered])
        row_started = time.monotonic()
        completions, usage = _generate_completions(
            generator,
            tokenizer,
            [rendered],
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
            add_special_tokens=False,
        )
        completion = exact_model_owned_completion(completions[0])
        token_count, hit_limit = usage[0]
        outputs.append(
            {
                "schema": SCHEMA,
                "host": spec.host,
                "identity_sha256": row["identity_sha256"],
                "split": row["split"],
                "task": row["task"],
                "prompt_sha256": hashlib.sha256(
                    str(row["source_prompt"]).encode()
                ).hexdigest(),
                "owner_checkpoint_sha256": sha256_file(args.owner_checkpoint),
                "owner_state_sha256": expected_state,
                "model_revision": spec.model_revision,
                "completion": completion,
                "generated_tokens": int(token_count),
                "max_token_exhausted": bool(hit_limit),
                "finish_reason": "length" if hit_limit else "stop",
                "wall_seconds": time.monotonic() - row_started,
            }
        )
        generated_tokens += int(token_count)
        exhausted += int(hit_limit)
    torch.cuda.synchronize()
    output_sha256 = _atomic_lines(args.output, outputs)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "host": spec.host,
        "model_revision": spec.model_revision,
        "host_contract": spec.receipt(),
        "model_receipt": loaded.model_receipt,
        "owner_checkpoint": str(args.owner_checkpoint.resolve()),
        "owner_checkpoint_sha256": sha256_file(args.owner_checkpoint),
        "owner_state_sha256": expected_state,
        "owner_role": "owner",
        "owner_update": owner_payload["update"],
        "owner_restore_exact": True,
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        "freeze_report_sha256": sha256_file(args.freeze_report),
        "freeze_identity_receipts": freeze_report["identity_receipts"],
        "train_source_sha256": sha256_file(args.train_source),
        "development_source_sha256": sha256_file(args.development_source),
        "generation_mode": "greedy",
        "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
        "rendered_chat_tokenization": "add_special_tokens_false",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_rows": len(rows),
        "row_start": row_start,
        "row_end": row_end,
        "rows": len(outputs),
        "ordered_identity_sha256": hashlib.sha256(
            ("\n".join(row["identity_sha256"] for row in outputs) + "\n").encode()
        ).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_token_exhausted": exhausted,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in range(2)
        },
        "routing_receipt": loaded.model.receipt(),
        "capability_scored": False,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", choices=("nemotron-super", "mixtral-8x22b"), required=True
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256")
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--overlay-manifest", type=Path)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=DRAFT_SHARDS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=DRAFT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DRAFT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
