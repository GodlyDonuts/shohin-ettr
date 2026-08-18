#!/usr/bin/env python3
"""Generate all matched Mixtral BF16-TP4 screen arms in one model load."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

import torch
import torch.distributed as dist

from hf_mixtral_8x22b_evaluate import (
    ARMS,
    CANDIDATE_SCHEMA,
    MAX_NEW_TOKENS,
    ROWS as SCREEN_ROWS,
    SEED,
    SHARDS as SCREEN_SHARDS,
    SOURCE_SHA256 as SCREEN_SOURCE_SHA256,
)
from hf_mixtral_8x22b_mechanics import EXPECTED_PACKAGES, verify_model_manifest
from hf_mixtral_8x22b_multinode_tp_mechanics import (
    EXPECTED_WORLD_SIZE,
    _broadcast_object,
    _gather_rank_receipts,
    _require_world,
)
from hf_mixtral_8x22b_multinode_tp_train_revision import (
    CHECKPOINT_SCHEMA,
    SCHEMA as TRAINING_SCHEMA,
    _validate_mechanics,
)
from hf_mixtral_8x22b_train_revision import (
    DATA_SHA256,
    DRAFT_ORIGIN_MODEL,
    DRAFT_ORIGIN_REVISION,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    SEED as TRAINING_SEED,
    TRANSFER_SCOPE,
    UPDATES,
    _package_versions,
    _state_sha256,
)
from hf_product_reasoning_eval import (
    GENERATED_ONLY_SEQUENCE_CONTRACT,
    _generate_completions,
    _generation_stop_token_ids,
    _render_prompt,
)
from hf_q36_mtr_evaluate import q36_nonpadding_prompt_tokens
from hf_q36_mtr_external_evaluate import load_drafts, load_sources, prompt_for
from hf_pcf1_evaluate import shard_bounds
from mixtral_post_mlp_revision import MixtralRevisionError, MixtralRevisionModel
from q36_upward_moe_mixtral_host import (
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

REPORT_SCHEMA = "shohin-mixtral-8x22b-bf16-tp4-matched-evaluation-v1"
VALIDATION_ROWS = 1023
VALIDATION_SHARDS = 16
VALIDATION_SOURCE_SHA256 = (
    "98c25465916f6275c49ccf9cec67db1236cf0c795db67246a774ea392c0cb778"
)
DATASET_SPECS = {
    (SCREEN_ROWS, SCREEN_SHARDS): {
        "source_sha256": SCREEN_SOURCE_SHA256,
        "split": "external_validation_screen",
    },
    (VALIDATION_ROWS, VALIDATION_SHARDS): {
        "source_sha256": VALIDATION_SOURCE_SHA256,
        "split": "external_validation_confirmation",
    },
}


class MixtralDistributedEvaluationError(RuntimeError):
    """The distributed matched evaluation contract differed."""


def _encoded_rows(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)


def _atomic_bytes(path: Path, payload: bytes) -> str:
    if path.exists() or path.is_symlink():
        raise MixtralDistributedEvaluationError("refusing existing evaluation output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    _atomic_bytes(path, encoded)


def _validate_host(
    model_root: Path,
    model_manifest: Path,
    expected_model_manifest_sha256: str,
) -> dict[str, Any]:
    model_root = model_root.resolve(strict=True)
    if (
        model_root.is_symlink()
        or not model_root.is_dir()
        or model_manifest.resolve(strict=True).parent != model_root
        or sha256_file(model_root / "config.json") != MODEL_CONFIG_SHA256
        or (model_root / "SOURCE_REVISION").is_symlink()
        or not (model_root / "SOURCE_REVISION").is_file()
        or (model_root / "SOURCE_REVISION").read_text().strip() != MODEL_REVISION
    ):
        raise MixtralDistributedEvaluationError("model identity differs")
    load_pinned_config(model_root / "config.json")
    return verify_model_manifest(
        model_root, model_manifest, expected_model_manifest_sha256
    )


def _load_checkpoint(
    path: Path,
    model: MixtralRevisionModel,
    expected_model_manifest_sha256: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MixtralDistributedEvaluationError("revision checkpoint is absent")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("trainable_state") if isinstance(payload, dict) else None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    expected = {
        "schema": TRAINING_SCHEMA,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": expected_model_manifest_sha256,
        "draft_origin_model": DRAFT_ORIGIN_MODEL,
        "draft_origin_revision": DRAFT_ORIGIN_REVISION,
        "transfer_scope": TRANSFER_SCOPE,
        "standalone_mixtral_owned_draft": False,
        "data_sha256": DATA_SHA256,
        "updates": UPDATES,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "seed": TRAINING_SEED,
        "world_size": EXPECTED_WORLD_SIZE,
        "parallelism": "native-transformers-tensor-parallel",
        "weight_dtype": "bfloat16",
        "quantization": "none",
        "trainable_parameters": TRAINABLE_PARAMETERS_PER_ROLE,
        "native_router_expert_trainables": 0,
        "optimizer_state_serialized": False,
        "checkpoint_trainable_only": True,
    }
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "update", "trainable_state", "metadata"}
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("update") != UPDATES
        or not isinstance(state, dict)
        or set(state) != set(current)
        or not isinstance(metadata, dict)
        or any(metadata.get(key) != value for key, value in expected.items())
        or metadata.get("trainable_parameter_name_sha256")
        != model.trainable_parameter_name_sha256()
        or metadata.get("final_trainable_state_sha256") != _state_sha256(state)
    ):
        raise MixtralDistributedEvaluationError("revision checkpoint contract differs")
    with torch.no_grad():
        for name, parameter in current.items():
            tensor = state[name]
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise MixtralDistributedEvaluationError("checkpoint geometry differs")
            parameter.copy_(tensor.to(parameter.device))
    if model.trainable_state_sha256() != metadata["final_trainable_state_sha256"]:
        raise MixtralDistributedEvaluationError("checkpoint restore differs")
    return metadata


def _generate_arm(
    *,
    arm: str,
    backbone: Any,
    tokenizer: Any,
    sources: list[dict[str, Any]],
    drafts: dict[str, dict[str, Any]] | None,
    stop_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    started = time.monotonic()
    for source in sources:
        draft = drafts[source["identity_sha256"]] if drafts is not None else None
        question = prompt_for(arm, source, draft)
        rendered = [_render_prompt(tokenizer, question, True, False)]
        counters["prompt_tokens"] += q36_nonpadding_prompt_tokens(tokenizer, rendered)
        completions, usage = _generate_completions(
            backbone,
            tokenizer,
            rendered,
            False,
            "greedy",
            MAX_NEW_TOKENS,
            stop_ids,
            add_special_tokens=False,
        )
        completion = completions[0]
        generated_tokens, exhausted = usage[0]
        candidates.append(
            {
                "schema": CANDIDATE_SCHEMA,
                "arm": arm,
                "identity_sha256": source["identity_sha256"],
                "task": source["task"],
                "completion": completion,
                "generated_tokens": generated_tokens,
                "max_token_exhausted": exhausted,
            }
        )
        counters["rows"] += 1
        counters["generated_tokens"] += generated_tokens
        counters["max_token_exhausted"] += int(exhausted)
        counters["empty_completions"] += int(not completion.strip())
    return candidates, dict(sorted(counters.items())), time.monotonic() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank, world = _require_world()
    dataset = DATASET_SPECS.get((args.expected_rows, args.shard_count))
    group_geometry = (
        args.shard_group_index,
        args.shard_group_count,
    )
    divisible_groups = (
        args.shard_group_count > 0 and args.shard_count % args.shard_group_count == 0
    )
    drafts_per_group = (
        args.shard_count // args.shard_group_count if divisible_groups else -1
    )
    if (
        dataset is None
        or args.seed != SEED
        or args.batch_size != 1
        or args.output_root.is_symlink()
        or sha256_file(args.source) != dataset["source_sha256"]
        or len(args.draft_candidates) != drafts_per_group
        or group_geometry
        not in (
            (0, 1),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
        )
        or (args.expected_rows == SCREEN_ROWS and group_geometry != (0, 1))
        or not divisible_groups
    ):
        raise MixtralDistributedEvaluationError("evaluation settings differ")
    rows = args.expected_rows
    shards = args.shard_count
    source_sha256 = dataset["source_sha256"]
    split = dataset["split"]

    mechanics = None
    model_receipt = None
    if rank == 0:
        mechanics = _validate_mechanics(
            args.mechanics_report, args.expected_model_manifest_sha256
        )
        model_receipt = _validate_host(
            args.model_root,
            args.model_manifest,
            args.expected_model_manifest_sha256,
        )
    mechanics = _broadcast_object(mechanics)
    model_receipt = _broadcast_object(model_receipt)
    if not isinstance(mechanics, dict) or not isinstance(model_receipt, dict):
        raise MixtralDistributedEvaluationError("evaluation authorization differs")
    if _package_versions() != EXPECTED_PACKAGES:
        raise MixtralDistributedEvaluationError("evaluation package versions differ")

    sources = load_sources(args.source, rows)
    shards_per_group = shards // args.shard_group_count
    first_shard = args.shard_group_index * shards_per_group
    shard_indices = list(range(first_shard, first_shard + shards_per_group))
    group_start, _ = shard_bounds(rows, shard_indices[0], shards, 1)
    _, group_end = shard_bounds(rows, shard_indices[-1], shards, 1)
    group_sources = sources[group_start:group_end]
    drafts = load_drafts(args.draft_candidates, group_sources)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.distributed.configuration_utils import DistributedConfig

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_root,
        local_files_only=True,
        trust_remote_code=False,
        distributed_config=DistributedConfig(tp_size=world),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    distributed_config = getattr(backbone.config, "distributed_config", None)
    if (
        distributed_config is None
        or distributed_config.tp_size != world
        or distributed_config.fsdp_size != 1
        or getattr(backbone, "_device_mesh", None) is None
        or getattr(backbone, "hf_device_map", None) is not None
    ):
        raise MixtralDistributedEvaluationError("tensor-parallel load differs")
    backbone.eval()
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    results: dict[str, dict[str, Any]] = {}

    for arm in ("unchanged", "self_refinement"):
        candidates, counters, elapsed = _generate_arm(
            arm=arm,
            backbone=backbone,
            tokenizer=tokenizer,
            sources=group_sources,
            drafts=drafts if arm == "self_refinement" else None,
            stop_ids=stop_ids,
        )
        results[arm] = {
            "candidates": candidates,
            "counters": counters,
            "elapsed_seconds": elapsed,
            "routing_receipt": None,
        }

    revision_model = MixtralRevisionModel(backbone)
    metadata = _load_checkpoint(
        args.revision_checkpoint,
        revision_model,
        args.expected_model_manifest_sha256,
    )
    revision_model.eval()
    revision_model.reset_receipt()
    candidates, counters, elapsed = _generate_arm(
        arm="revision",
        backbone=backbone,
        tokenizer=tokenizer,
        sources=group_sources,
        drafts=drafts,
        stop_ids=stop_ids,
    )
    results["revision"] = {
        "candidates": candidates,
        "counters": counters,
        "elapsed_seconds": elapsed,
        "routing_receipt": revision_model.receipt(),
    }
    torch.cuda.synchronize()

    rank_payload = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "candidate_sha256s": {
            arm: hashlib.sha256(_encoded_rows(results[arm]["candidates"])).hexdigest()
            for arm in ARMS
        },
        "counters": {arm: results[arm]["counters"] for arm in ARMS},
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "shard_group_index": args.shard_group_index,
        "shard_group_count": args.shard_group_count,
        "group_row_start": group_start,
        "group_row_end": group_end,
    }
    rank_receipts = _gather_rank_receipts(rank_payload, world)
    for arm in ARMS:
        if (
            len({item["candidate_sha256s"][arm] for item in rank_receipts}) != 1
            or len(
                {
                    json.dumps(item["counters"][arm], sort_keys=True)
                    for item in rank_receipts
                }
            )
            != 1
        ):
            raise MixtralDistributedEvaluationError("rank candidate output differs")

    reports: dict[str, dict[str, Any]] = {}
    if rank == 0:
        args.output_root.mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            group_candidates = results[arm]["candidates"]
            for shard_index in shard_indices:
                start, end = shard_bounds(rows, shard_index, shards, 1)
                shard_dir = args.output_root / arm / f"shard_{shard_index:02d}"
                if shard_dir.exists() or shard_dir.is_symlink():
                    raise MixtralDistributedEvaluationError(
                        "evaluation shard output exists"
                    )
                shard_dir.mkdir(parents=True)
                candidate_path = shard_dir / "candidates.jsonl"
                candidates_sha256 = _atomic_bytes(
                    candidate_path,
                    _encoded_rows(
                        group_candidates[start - group_start : end - group_start]
                    ),
                )
                report = {
                    "schema": REPORT_SCHEMA,
                    "status": "complete",
                    "arm": arm,
                    "split": split,
                    "model_revision": MODEL_REVISION,
                    "model_manifest_sha256": args.expected_model_manifest_sha256,
                    "model_receipt": model_receipt,
                    "world_size": world,
                    "parallelism": "native-transformers-tensor-parallel",
                    "weight_dtype": "bfloat16",
                    "quantization": "none",
                    "draft_origin_model": DRAFT_ORIGIN_MODEL,
                    "draft_origin_revision": DRAFT_ORIGIN_REVISION,
                    "transfer_scope": TRANSFER_SCOPE,
                    "mechanics_report_sha256": sha256_file(args.mechanics_report),
                    "mechanics_checkpoint_sha256": mechanics["checkpoint_sha256"],
                    "revision_checkpoint_sha256": (
                        sha256_file(args.revision_checkpoint)
                        if arm == "revision"
                        else None
                    ),
                    "revision_metadata_sha256": (
                        hashlib.sha256(
                            json.dumps(
                                metadata, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest()
                        if arm == "revision"
                        else None
                    ),
                    "trainable_parameters": (
                        TRAINABLE_PARAMETERS_PER_ROLE if arm == "revision" else 0
                    ),
                    "native_router_expert_trainables": 0,
                    "source_sha256": source_sha256,
                    "draft_candidate_sha256s": (
                        [sha256_file(path) for path in args.draft_candidates]
                        if arm != "unchanged"
                        else []
                    ),
                    "fixed_draft_control": arm != "unchanged",
                    "generation_mode": "greedy",
                    "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "seed": SEED,
                    "batch_size": 1,
                    "shard_index": shard_index,
                    "shard_count": shards,
                    "shard_group_index": args.shard_group_index,
                    "shard_group_count": args.shard_group_count,
                    "group_row_start": group_start,
                    "group_row_end": group_end,
                    "row_start": start,
                    "row_end": end,
                    "full_row_count": rows,
                    "candidates_output": str(candidate_path.resolve()),
                    "candidates_sha256": candidates_sha256,
                    "counters": {
                        "rows": end - start,
                        (
                            "full_arm_counters"
                            if args.shard_group_count == 1
                            else "group_arm_counters"
                        ): results[arm]["counters"],
                    },
                    (
                        "full_arm_elapsed_seconds"
                        if args.shard_group_count == 1
                        else "group_arm_elapsed_seconds"
                    ): results[arm]["elapsed_seconds"],
                    "rank_receipts": rank_receipts,
                    "routing_receipt": results[arm]["routing_receipt"],
                    "assessor_access_count": 0,
                    "development_labels_read": 0,
                    "sealed_access": {"holdout": 0, "product": 0, "public": 0},
                }
                _atomic_json(shard_dir / "report.json", report)
                reports[f"{arm}/{shard_index:02d}"] = report
    dist.barrier()
    return {
        "status": "complete",
        "reports": reports if rank == 0 else {},
        "total_elapsed_seconds": time.monotonic() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--revision-checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--draft-candidates", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=SCREEN_ROWS)
    parser.add_argument("--shard-count", type=int, default=SCREEN_SHARDS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shard-group-index", type=int, default=0)
    parser.add_argument("--shard-group-count", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
    except (MixtralDistributedEvaluationError, MixtralRevisionError) as error:
        raise SystemExit(str(error)) from error
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    if int(os.environ.get("RANK", "-1")) == 0:
        print(json.dumps({"status": result["status"], "reports": 12}, sort_keys=True))
