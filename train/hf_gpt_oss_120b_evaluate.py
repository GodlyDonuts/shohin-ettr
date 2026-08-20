#!/usr/bin/env python3
"""Generate matched GPT-OSS-120B arms on the fixed Q36 MoE screen."""

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

from gpt_oss_harmony import extract_final_completion, render_prompt
from gpt_oss_post_mlp_revision import GptOssRevisionModel
from hf_gpt_oss_120b_mechanics import (
    EXPECTED_PACKAGES,
    _package_receipt,
    _state_sha256,
    verify_manifest,
)
from hf_gpt_oss_120b_train_revision import (
    CHECKPOINT_SCHEMA,
    DATA_SHA256,
    DRAFT_ORIGIN_MODEL,
    DRAFT_ORIGIN_REVISION,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    SCHEMA as TRAINING_SCHEMA,
    SEED as TRAINING_SEED,
    TRANSFER_SCOPE,
    UPDATES,
    _load_backbone,
    validate_mechanics,
)
from hf_pcf1_evaluate import shard_bounds
from hf_q36_mtr_external_evaluate import (
    MMLU_CONFIRMATION_TASKS,
    TASKS,
    load_drafts,
    load_sources,
    prompt_for,
)
from q36_upward_moe_gpt_oss_host import (
    MODEL_CONFIG_SHA256,
    MODEL_MANIFEST_SHA256,
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

REPORT_SCHEMA = "shohin-gpt-oss-120b-fixed-draft-evaluation-v1"
CANDIDATE_SCHEMA = "shohin-gpt-oss-120b-fixed-draft-candidate-v1"
ARMS = ("unchanged", "self_refinement", "revision")
SOURCE_SHA256 = "f0b7830814762c6917363642e86edaaf192a8ab2834911c13c0cae9255ceefa9"
ROWS = 256
SHARDS = 4
CONFIRMATION_GEOMETRIES = ((256, 4), (1_023, 16))
SEED = 2026080816
MAX_NEW_TOKENS = 768


class GptOssEvaluationError(RuntimeError):
    """The GPT-OSS matched capability evaluation contract differed."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise GptOssEvaluationError("refusing existing candidates")
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
        raise GptOssEvaluationError("refusing existing evaluation report")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_revision_checkpoint(
    path: Path,
    model: GptOssRevisionModel,
    *,
    model_manifest_sha256: str,
    overlay_manifest_sha256: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GptOssEvaluationError("revision checkpoint is absent")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("trainable_state") if isinstance(payload, dict) else None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    expected = {
        "schema": TRAINING_SCHEMA,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": model_manifest_sha256,
        "overlay_manifest_sha256": overlay_manifest_sha256,
        "draft_origin_model": DRAFT_ORIGIN_MODEL,
        "draft_origin_revision": DRAFT_ORIGIN_REVISION,
        "transfer_scope": TRANSFER_SCOPE,
        "standalone_gpt_oss_owned_draft": False,
        "data_sha256": DATA_SHA256,
        "updates": UPDATES,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "seed": TRAINING_SEED,
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
        raise GptOssEvaluationError("revision checkpoint contract differs")
    with torch.no_grad():
        for name, parameter in current.items():
            tensor = state[name]
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise GptOssEvaluationError("revision checkpoint geometry differs")
            parameter.copy_(tensor.to(parameter.device))
    if model.trainable_state_sha256() != metadata["final_trainable_state_sha256"]:
        raise GptOssEvaluationError("revision checkpoint restore differs")
    return metadata


def _generate_one(
    model: Any, tokenizer: Any, question: str
) -> tuple[str, dict[str, Any]]:
    rendered = render_prompt(tokenizer, question)
    tokenized = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"].to("cuda:0")
    attention_mask = tokenized["attention_mask"].to("cuda:0")
    prompt_tokens = int(attention_mask.sum().item())
    with torch.inference_mode():
        sequence = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    generated_ids = sequence[0, input_ids.shape[1] :].detach().cpu().tolist()
    completion, harmony = extract_final_completion(tokenizer, generated_ids)
    return completion, {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated_ids),
        "max_token_exhausted": len(generated_ids) >= MAX_NEW_TOKENS,
        "harmony": harmony,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if (
        args.arm not in ARMS
        or args.seed != SEED
        or (
            (args.expected_rows, args.shard_count)
            not in (
                CONFIRMATION_GEOMETRIES
                if args.confirmation_mmlu_pro
                else ((ROWS, SHARDS),)
            )
        )
        or args.batch_size != 1
        or not 0 <= args.shard_index < args.shard_count
        or args.candidates_output.exists()
        or args.report.exists()
        or sha256_file(args.source) != args.expected_source_sha256
        or (args.arm == "revision") != (args.revision_checkpoint is not None)
        or (args.arm != "unchanged") != bool(args.draft_candidates)
        or args.expected_model_manifest_sha256 != MODEL_MANIFEST_SHA256
        or (args.expected_source_sha256 != SOURCE_SHA256) != args.confirmation_mmlu_pro
    ):
        raise GptOssEvaluationError("evaluation settings differ")
    mechanics = validate_mechanics(
        args.mechanics_report,
        model_manifest_sha256=args.expected_model_manifest_sha256,
        overlay_manifest_sha256=args.expected_overlay_manifest_sha256,
    )
    model_root = args.model_root.resolve(strict=True)
    overlay_root = args.overlay_root.resolve(strict=True)
    model_receipt = verify_manifest(
        model_root, args.model_manifest, args.expected_model_manifest_sha256
    )
    overlay_receipt = verify_manifest(
        overlay_root, args.overlay_manifest, args.expected_overlay_manifest_sha256
    )
    if (
        sha256_file(model_root / "config.json") != MODEL_CONFIG_SHA256
        or (model_root / "SOURCE_REVISION").read_text(encoding="utf-8")
        != f"{MODEL_REVISION}\n"
    ):
        raise GptOssEvaluationError("host identity differs")
    load_pinned_config(model_root / "config.json")
    packages = _package_receipt(overlay_root)
    if packages["versions"] != EXPECTED_PACKAGES:
        raise GptOssEvaluationError("evaluation package versions differ")
    if (
        torch.cuda.device_count() != 1
        or "H100" not in torch.cuda.get_device_name(0).upper()
    ):
        raise GptOssEvaluationError("evaluation requires exactly one H100")

    sources = load_sources(
        args.source,
        args.expected_rows,
        MMLU_CONFIRMATION_TASKS if args.confirmation_mmlu_pro else TASKS,
    )
    drafts = (
        load_drafts(args.draft_candidates, sources) if args.arm != "unchanged" else None
    )
    start, end = shard_bounds(
        args.expected_rows, args.shard_index, args.shard_count, args.batch_size
    )
    rows = sources[start:end]
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    backbone, native_load_receipt = _load_backbone(model_root)
    revision_model = None
    metadata = None
    if args.arm == "revision":
        revision_model = GptOssRevisionModel(backbone)
        metadata = load_revision_checkpoint(
            args.revision_checkpoint,
            revision_model,
            model_manifest_sha256=args.expected_model_manifest_sha256,
            overlay_manifest_sha256=args.expected_overlay_manifest_sha256,
        )
        revision_model.eval()
        revision_model.reset_receipt()
    else:
        backbone.eval()
    generation_model = backbone
    torch.cuda.reset_peak_memory_stats(0)
    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    started = time.monotonic()
    for source in rows:
        draft = drafts[source["identity_sha256"]] if drafts is not None else None
        question = prompt_for(args.arm, source, draft)
        completion, usage = _generate_one(generation_model, tokenizer, question)
        harmony = usage["harmony"]
        candidates.append(
            {
                "schema": CANDIDATE_SCHEMA,
                "arm": args.arm,
                "identity_sha256": source["identity_sha256"],
                "task": source["task"],
                "completion": completion,
                "generated_tokens": usage["generated_tokens"],
                "max_token_exhausted": usage["max_token_exhausted"],
                "harmony_trajectory_sha256": harmony["raw_trajectory_sha256"],
                "harmony_analysis_channel_present": harmony["analysis_channel_present"],
                "harmony_final_channel_present": harmony["final_channel_present"],
                "harmony_final_channel_terminated": harmony["final_channel_terminated"],
            }
        )
        counters["rows"] += 1
        counters["prompt_tokens"] += usage["prompt_tokens"]
        counters["generated_tokens"] += usage["generated_tokens"]
        counters["max_token_exhausted"] += int(usage["max_token_exhausted"])
        counters["empty_completions"] += int(not completion)
        counters["missing_final_channel"] += int(not harmony["final_channel_present"])
        counters["unterminated_final_channel"] += int(
            harmony["final_channel_present"] and not harmony["final_channel_terminated"]
        )
    torch.cuda.synchronize()
    candidates_sha256 = _atomic_lines(args.candidates_output, candidates)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "split": "external_validation",
        "model_revision": MODEL_REVISION,
        "model_receipt": model_receipt,
        "overlay_receipt": overlay_receipt,
        "packages": packages,
        "native_mxfp4_load_receipt": native_load_receipt,
        "draft_origin_model": DRAFT_ORIGIN_MODEL,
        "draft_origin_revision": DRAFT_ORIGIN_REVISION,
        "transfer_scope": TRANSFER_SCOPE,
        "standalone_gpt_oss_owned_draft": False,
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        "mechanics_checkpoint_sha256": mechanics["checkpoint_sha256"],
        "revision_checkpoint": (
            str(args.revision_checkpoint.resolve())
            if args.revision_checkpoint is not None
            else None
        ),
        "revision_checkpoint_sha256": (
            sha256_file(args.revision_checkpoint)
            if args.revision_checkpoint is not None
            else None
        ),
        "revision_metadata_sha256": (
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
        "source_sha256": args.expected_source_sha256,
        "draft_candidate_sha256s": (
            [sha256_file(path) for path in args.draft_candidates]
            if args.arm != "unchanged"
            else []
        ),
        "fixed_draft_control": args.arm != "unchanged",
        "generation_mode": "greedy",
        "harmony_reasoning_effort": "low",
        "generation_projection": "final_channel_only",
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": start,
        "row_end": end,
        "full_row_count": args.expected_rows,
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        "counters": dict(sorted(counters.items())),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "routing_receipt": (
            revision_model.receipt() if revision_model is not None else None
        ),
        "assessor_access_count": 0,
        "development_labels_read": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        "confirmation_mmlu_pro": args.confirmation_mmlu_pro,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-model-manifest-sha256", default=MODEL_MANIFEST_SHA256
    )
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--expected-overlay-manifest-sha256", required=True)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--revision-checkpoint", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", default=SOURCE_SHA256)
    parser.add_argument("--confirmation-mmlu-pro", action="store_true")
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
