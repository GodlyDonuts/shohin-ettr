#!/usr/bin/env python3
"""Generate matched 550B-A55B arms on the fixed Q36 MoE screen."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from hf_nemotron_super_evaluate import (
    ARMS,
    MAX_NEW_TOKENS,
    ROWS,
    SEED,
    SHARDS,
    SOURCE_SHA256,
    _package_versions,
)
from hf_nemotron_ultra_mechanics import (
    CUDA_VERSION,
    MAMBA_VERSION,
    MODELOPT_VERSION,
    SCHEMA as MECHANICS_SCHEMA,
    TORCH_VERSION,
    verify_manifest,
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
from lift_nemotron_super_adapter_to_ultra import (
    CHECKPOINT_SCHEMA,
    FACTOR_SHA256,
    SCHEMA as TRANSFER_SCHEMA,
    _state_sha256,
)
from nemotron_ultra_post_mixer_revision import NemotronUltraRevisionModel
from q36_upward_moe_ultra_host import (
    MINIMUM_H100S,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

REPORT_SCHEMA = "shohin-nemotron-ultra-fixed-draft-evaluation-v1"
CANDIDATE_SCHEMA = "shohin-nemotron-ultra-fixed-draft-candidate-v1"


class NemotronUltraEvaluationError(RuntimeError):
    """The upward-MoE Ultra evaluation contract differed."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise NemotronUltraEvaluationError("refusing existing candidates")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NemotronUltraEvaluationError("refusing existing evaluation report")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_mechanics_report(
    path: Path, expected_model_manifest_sha256: str
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NemotronUltraEvaluationError("mechanics report is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    receipt = payload.get("model_receipt")
    if (
        payload.get("schema") != MECHANICS_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("score_rows_read") != 0
        or payload.get("benchmark_rows_read") != 0
        or payload.get("trainable_parameters") != TRAINABLE_PARAMETERS_PER_ROLE
        or payload.get("native_router_expert_trainables") != 0
        or payload.get("serialization_restore_exact") is not True
        or len(payload.get("devices", [])) != MINIMUM_H100S
        or not isinstance(receipt, dict)
        or receipt.get("manifest_sha256") != expected_model_manifest_sha256
        or receipt.get("exact_membership") is not True
    ):
        raise NemotronUltraEvaluationError("mechanics authorization differs")
    return payload


def validate_host(
    model_root: Path, model_manifest: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    model_root = model_root.resolve(strict=True)
    receipt = verify_manifest(
        model_root,
        model_manifest,
        expected_manifest_sha256,
        exact_membership=True,
    )
    if (
        sha256_file(model_root / "config.json") != MODEL_CONFIG_SHA256
        or (model_root / "SOURCE_REVISION").read_text(encoding="utf-8").strip()
        != MODEL_REVISION
    ):
        raise NemotronUltraEvaluationError("Ultra model identity differs")
    load_pinned_config(model_root / "config.json")
    return receipt


def load_transferred_checkpoint(
    checkpoint: Path,
    transfer_report: Path,
    model: NemotronUltraRevisionModel,
) -> dict[str, Any]:
    if any(
        path.is_symlink() or not path.is_file()
        for path in (checkpoint, transfer_report)
    ):
        raise NemotronUltraEvaluationError("Ultra transfer input is absent")
    report = json.loads(transfer_report.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("trainable_state") if isinstance(payload, dict) else None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if (
        not isinstance(payload, dict)
        or not isinstance(report, dict)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or not isinstance(state, dict)
        or set(state) != set(current)
        or not isinstance(metadata, dict)
        or metadata.get("schema") != TRANSFER_SCHEMA
        or metadata.get("target_model_revision") != MODEL_REVISION
        or metadata.get("ultra_config_sha256") != MODEL_CONFIG_SHA256
        or metadata.get("factor_sha256") != FACTOR_SHA256
        or metadata.get("trainable_parameters") != TRAINABLE_PARAMETERS_PER_ROLE
        or metadata.get("target_trainable_state_sha256") != _state_sha256(state)
        or metadata.get("label_rows_read") != 0
        or metadata.get("benchmark_rows_read") != 0
        or metadata.get("optimizer_updates") != 0
        or metadata.get("native_router_expert_trainables") != 0
        or report.get("schema") != TRANSFER_SCHEMA
        or report.get("status") != "complete"
        or report.get("checkpoint_sha256") != sha256_file(checkpoint)
        or any(report.get(key) != value for key, value in metadata.items())
    ):
        raise NemotronUltraEvaluationError("Ultra transfer checkpoint differs")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise NemotronUltraEvaluationError(
                    "Ultra transfer tensor geometry differs"
                )
            parameter.copy_(value.to(parameter.device))
    if model.trainable_state_sha256() != metadata["target_trainable_state_sha256"]:
        raise NemotronUltraEvaluationError("Ultra transfer restore differs")
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    from modelopt.torch.opt.plugins.huggingface import enable_huggingface_checkpointing
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (
        args.arm not in ARMS
        or args.seed != SEED
        or args.expected_rows != ROWS
        or args.shard_count != SHARDS
        or args.batch_size != 1
        or not 0 <= args.shard_index < SHARDS
        or args.candidates_output.exists()
        or args.report.exists()
        or sha256_file(args.source) != SOURCE_SHA256
        or (args.arm == "revision")
        != (args.revision_checkpoint is not None and args.transfer_report is not None)
        or (args.arm != "unchanged") != bool(args.draft_candidates)
    ):
        raise NemotronUltraEvaluationError("Ultra evaluation settings differ")
    mechanics = validate_mechanics_report(
        args.mechanics_report, args.expected_model_manifest_sha256
    )
    model_receipt = validate_host(
        args.model_root,
        args.model_manifest,
        args.expected_model_manifest_sha256,
    )
    expected_versions = {
        "mamba-ssm": MAMBA_VERSION,
        "nvidia-modelopt": MODELOPT_VERSION,
        "torch": TORCH_VERSION,
        "cuda": CUDA_VERSION,
    }
    if _package_versions() != expected_versions:
        raise NemotronUltraEvaluationError("Ultra package versions differ")
    if torch.cuda.device_count() != MINIMUM_H100S or any(
        "H100" not in torch.cuda.get_device_name(index).upper()
        for index in range(MINIMUM_H100S)
    ):
        raise NemotronUltraEvaluationError("Ultra evaluation requires eight H100s")

    sources = load_sources(args.source, ROWS)
    drafts = (
        load_drafts(args.draft_candidates, sources) if args.arm != "unchanged" else None
    )
    start, end = shard_bounds(ROWS, args.shard_index, SHARDS, args.batch_size)
    rows = sources[start:end]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    enable_huggingface_checkpointing()
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_root,
        local_files_only=True,
        trust_remote_code=False,
        device_map="balanced",
        max_memory={index: "77GiB" for index in range(MINIMUM_H100S)},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    device_map = getattr(backbone, "hf_device_map", None)
    if not isinstance(device_map, dict) or set(device_map.values()) != set(
        range(MINIMUM_H100S)
    ):
        raise NemotronUltraEvaluationError("Ultra evaluation device map differs")
    metadata = None
    revision_model = None
    if args.arm == "revision":
        revision_model = NemotronUltraRevisionModel(backbone)
        metadata = load_transferred_checkpoint(
            args.revision_checkpoint, args.transfer_report, revision_model
        )
        revision_model.eval()
        revision_model.reset_receipt()
    else:
        backbone.eval()

    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    for index in range(MINIMUM_H100S):
        torch.cuda.reset_peak_memory_stats(index)
    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    started = time.monotonic()
    for source in rows:
        draft = drafts[source["identity_sha256"]] if drafts is not None else None
        question = prompt_for(args.arm, source, draft)
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
                "arm": args.arm,
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
    torch.cuda.synchronize()
    candidates_sha256 = _atomic_lines(args.candidates_output, candidates)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "split": "source_disjoint_screen",
        "model_revision": MODEL_REVISION,
        "model_receipt": model_receipt,
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        "mechanics_checkpoint_sha256": mechanics["checkpoint_sha256"],
        "revision_checkpoint_sha256": (
            sha256_file(args.revision_checkpoint)
            if args.revision_checkpoint is not None
            else None
        ),
        "transfer_report_sha256": (
            sha256_file(args.transfer_report)
            if args.transfer_report is not None
            else None
        ),
        "transfer_metadata_sha256": (
            hashlib.sha256(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if metadata is not None
            else None
        ),
        "trainable_parameters": (
            revision_model.trainable_parameter_count()
            if revision_model is not None
            else 0
        ),
        "native_router_expert_trainables": 0,
        "source_sha256": SOURCE_SHA256,
        "draft_candidate_sha256s": (
            [sha256_file(path) for path in args.draft_candidates]
            if args.arm != "unchanged"
            else []
        ),
        "fixed_draft_control": args.arm != "unchanged",
        "generation_mode": "greedy",
        "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": start,
        "row_end": end,
        "full_row_count": ROWS,
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        "counters": dict(sorted(counters.items())),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in range(MINIMUM_H100S)
        },
        "routing_receipt": (
            revision_model.receipt() if revision_model is not None else None
        ),
        "assessor_access_count": 0,
        "development_labels_read": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--revision-checkpoint", type=Path)
    parser.add_argument("--transfer-report", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--draft-candidates", type=Path, action="append", default=[])
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=ROWS)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=SHARDS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {"arm": result["arm"], "rows": result["counters"]["rows"]},
            sort_keys=True,
        )
    )
