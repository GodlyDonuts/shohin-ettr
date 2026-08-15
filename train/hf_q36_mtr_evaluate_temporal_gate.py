#!/usr/bin/env python3
"""Generate source-disjoint candidates with a trained Q36 temporal gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from build_pcf1_data import revision_prompt
from hf_pcf1_evaluate import shard_bounds
from hf_product_reasoning_eval import (
    GENERATED_ONLY_SEQUENCE_CONTRACT,
    _generate_completions,
    _generation_stop_token_ids,
    _render_prompt,
)
from hf_product_reasoning_train import (
    load_product_backbone,
    resolve_product_backbone_layout,
)
from hf_q36_mtr_evaluate import q36_nonpadding_prompt_tokens, sha256_file
from hf_q36_mtr_external_evaluate import (
    CANDIDATE_SCHEMA,
    SEED,
    SHARD_COUNTS,
    _atomic_json,
    _atomic_lines,
    load_drafts,
    load_sources,
)
from hf_q36_mtr_train_temporal_gate import (
    GATE_INITIAL_REVISION_WEIGHT,
    GATE_PARAMETERS,
    MULTI_BRANCHES,
    MULTI_CHECKPOINT_SCHEMA,
    MULTI_GATE_PARAMETERS,
    MULTI_INITIAL_WEIGHTS,
    _role_pair,
    _role_bank,
    restore_gate_checkpoint,
)
from q36_mtr_roles import (
    ALPHA,
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    QUANTIZATION,
    RANK,
    TRAINABLE_MASTER_DTYPE,
    validate_backbone_geometry,
    validate_backbone_moe_surface,
)
from temporal_residual_gate import (
    MultiTrajectoryGatedProductModel,
    MultiTrajectoryResidualGateConfig,
    TemporalGatedProductModel,
    TemporalResidualGateConfig,
)

ARM = "temporal_gate"
MULTI_ARM = "multi_trajectory_gate"
REPORT_SCHEMA = "shohin-q36-mtr-temporal-gate-evaluation-v1"


class Q36MTRTemporalGateEvaluationError(RuntimeError):
    """The trained temporal gate or evaluation projection differs."""


def load_temporal_gate_model(
    model_root: Path,
    owner_checkpoint: Path,
    revision_checkpoint: Path,
    gate_checkpoint: Path,
) -> tuple[TemporalGatedProductModel, dict[str, Any], str, dict[str, Any]]:
    owner_state, revision_state, role_receipt = _role_pair(
        owner_checkpoint, revision_checkpoint
    )
    backbone, loader = load_product_backbone(
        model_root,
        "causal",
        dtype=__import__("torch").bfloat16,
        device_map={"": 0},
        quantization=QUANTIZATION,
    )
    controlled = validate_backbone_geometry(backbone)
    moe_surface = validate_backbone_moe_surface(backbone)
    text_model, lm_head, hidden, layout = resolve_product_backbone_layout(backbone)
    if hidden != HIDDEN_SIZE or controlled != list(CONTROLLED_LAYER_INDICES):
        raise Q36MTRTemporalGateEvaluationError("temporal gate backbone differs")
    model = TemporalGatedProductModel(
        backbone,
        text_model,
        lm_head,
        TemporalResidualGateConfig(
            HIDDEN_SIZE, RANK, ALPHA, GATE_INITIAL_REVISION_WEIGHT
        ),
        owner_state=owner_state,
        revision_state=revision_state,
        controlled_layer_indices=CONTROLLED_LAYER_INDICES,
    )
    update, metadata = restore_gate_checkpoint(gate_checkpoint, model)
    expected = {
        "architecture": "q36-tokenwise-temporal-residual-gate-v1",
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "gate_parameters": GATE_PARAMETERS,
        "initial_revision_weight": GATE_INITIAL_REVISION_WEIGHT,
        "trainable_master_dtype": TRAINABLE_MASTER_DTYPE,
        "role_receipt": role_receipt,
    }
    if (
        update != 256
        or any(metadata.get(key) != value for key, value in expected.items())
        or model.trainable_parameter_count() != GATE_PARAMETERS
    ):
        raise Q36MTRTemporalGateEvaluationError("temporal gate checkpoint differs")
    model.eval()
    model.reset_routing_receipt()
    return (
        model,
        metadata,
        loader,
        {
            "backbone_layout": layout,
            "native_moe_surface": moe_surface,
            "role_receipt": role_receipt,
        },
    )


def load_multi_trajectory_gate_model(
    model_root: Path,
    owner_checkpoint: Path,
    revision_checkpoint: Path,
    draft_hidden_checkpoint: Path,
    gate_checkpoint: Path,
) -> tuple[MultiTrajectoryGatedProductModel, dict[str, Any], str, dict[str, Any]]:
    role_states, role_receipt = _role_bank(
        owner_checkpoint, revision_checkpoint, draft_hidden_checkpoint
    )
    backbone, loader = load_product_backbone(
        model_root,
        "causal",
        dtype=__import__("torch").bfloat16,
        device_map={"": 0},
        quantization=QUANTIZATION,
    )
    controlled = validate_backbone_geometry(backbone)
    moe_surface = validate_backbone_moe_surface(backbone)
    text_model, lm_head, hidden, layout = resolve_product_backbone_layout(backbone)
    if hidden != HIDDEN_SIZE or controlled != list(CONTROLLED_LAYER_INDICES):
        raise Q36MTRTemporalGateEvaluationError(
            "multi-trajectory gate backbone differs"
        )
    model = MultiTrajectoryGatedProductModel(
        backbone,
        text_model,
        lm_head,
        MultiTrajectoryResidualGateConfig(
            HIDDEN_SIZE, RANK, ALPHA, MULTI_BRANCHES, MULTI_INITIAL_WEIGHTS
        ),
        role_states=role_states,
        controlled_layer_indices=CONTROLLED_LAYER_INDICES,
    )
    update, metadata = restore_gate_checkpoint(
        gate_checkpoint,
        model,
        checkpoint_schema=MULTI_CHECKPOINT_SCHEMA,
        gate_parameters=MULTI_GATE_PARAMETERS,
    )
    expected = {
        "architecture": "q36-tokenwise-multi-trajectory-residual-gate-v1",
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "gate_parameters": MULTI_GATE_PARAMETERS,
        "branch_names": list(MULTI_BRANCHES),
        "initial_branch_weights": list(MULTI_INITIAL_WEIGHTS),
        "trainable_master_dtype": TRAINABLE_MASTER_DTYPE,
        "role_receipt": role_receipt,
    }
    if (
        update != 256
        or any(metadata.get(key) != value for key, value in expected.items())
        or model.trainable_parameter_count() != MULTI_GATE_PARAMETERS
    ):
        raise Q36MTRTemporalGateEvaluationError(
            "multi-trajectory gate checkpoint differs"
        )
    model.eval()
    model.reset_routing_receipt()
    return (
        model,
        metadata,
        loader,
        {
            "backbone_layout": layout,
            "native_moe_surface": moe_surface,
            "role_receipt": role_receipt,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    architecture = getattr(args, "architecture", "temporal")
    multi = architecture == "multi_trajectory"
    if (
        architecture not in {"temporal", "multi_trajectory"}
        or args.model_revision != MODEL_REVISION
        or args.seed != SEED
        or args.expected_rows not in SHARD_COUNTS
        or args.shard_count != SHARD_COUNTS[args.expected_rows]
        or args.batch_size != 2
        or not 0 <= args.shard_index < args.shard_count
        or sha256_file(args.model_source_root / "config.json") != MODEL_CONFIG_SHA256
        or args.candidates_output.exists()
        or args.report.exists()
        or sha256_file(args.source) != args.source_sha256
    ):
        raise Q36MTRTemporalGateEvaluationError("temporal evaluation settings differ")
    sources = load_sources(args.source, args.expected_rows)
    drafts = load_drafts(args.draft_candidates, sources)
    start, end = shard_bounds(
        len(sources), args.shard_index, args.shard_count, args.batch_size
    )
    rows = sources[start:end]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if multi:
        draft_hidden_checkpoint = getattr(args, "draft_hidden_checkpoint", None)
        if not isinstance(draft_hidden_checkpoint, Path):
            raise Q36MTRTemporalGateEvaluationError(
                "multi-trajectory draft-hidden checkpoint is absent"
            )
        model, metadata, loader, model_receipt = load_multi_trajectory_gate_model(
            args.model_root,
            args.owner_checkpoint,
            args.revision_checkpoint,
            draft_hidden_checkpoint,
            args.gate_checkpoint,
        )
        arm = MULTI_ARM
    else:
        model, metadata, loader, model_receipt = load_temporal_gate_model(
            args.model_root,
            args.owner_checkpoint,
            args.revision_checkpoint,
            args.gate_checkpoint,
        )
        arm = ARM
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    started = time.monotonic()
    for source in rows:
        draft = drafts[source["identity_sha256"]]
        question = revision_prompt(source["source_prompt"], draft["completion"])
        rendered = [_render_prompt(tokenizer, question, True, False)]
        counters["prompt_tokens"] += q36_nonpadding_prompt_tokens(tokenizer, rendered)
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            768,
            stop_ids,
            add_special_tokens=False,
        )
        completion = completions[0]
        token_count, exhausted = usage[0]
        candidates.append(
            {
                "schema": CANDIDATE_SCHEMA,
                "arm": arm,
                "identity_sha256": source["identity_sha256"],
                "task": source["task"],
                "completion": completion,
                "generated_tokens": token_count,
                "max_token_exhausted": exhausted,
            }
        )
        counters["rows"] += 1
        counters["generated_tokens"] += token_count
        counters["max_token_exhausted"] += int(exhausted)
        counters["empty_completions"] += int(not completion.strip())
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output_sha256 = _atomic_lines(args.candidates_output, candidates)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "arm": arm,
        "split": "external_validation",
        "model_revision": MODEL_REVISION,
        "model_loader": loader,
        "gate_checkpoint": str(args.gate_checkpoint.resolve()),
        "gate_checkpoint_sha256": sha256_file(args.gate_checkpoint),
        "gate_metadata_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "owner_checkpoint_sha256": sha256_file(args.owner_checkpoint),
        "revision_checkpoint_sha256": sha256_file(args.revision_checkpoint),
        "draft_hidden_checkpoint_sha256": (
            sha256_file(args.draft_hidden_checkpoint) if multi else None
        ),
        "trainable_parameters": model.trainable_parameter_count(),
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        **model_receipt,
        "routing_receipt": model.routing_receipt(),
        "source_sha256": args.source_sha256,
        "draft_candidate_sha256s": [
            sha256_file(path) for path in args.draft_candidates
        ],
        "generation_mode": "greedy",
        "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
        "max_new_tokens": 768,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": start,
        "row_end": end,
        "full_row_count": len(sources),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": output_sha256,
        "counters": dict(sorted(counters.items())),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "assessor_access_count": 0,
        "development_labels_read": 0,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("temporal", "multi_trajectory"),
        default="temporal",
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--revision-checkpoint", type=Path, required=True)
    parser.add_argument("--draft-hidden-checkpoint", type=Path)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--draft-candidates", type=Path, action="append", required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"arm": report["arm"], "rows": report["counters"]["rows"]}))


if __name__ == "__main__":
    main()
