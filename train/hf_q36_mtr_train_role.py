#!/usr/bin/env python3
"""Train one frozen Q36-MTR natural-trajectory role on a single H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from hf_product_reasoning_train import (
    PRODUCT_SYSTEM_PROMPT,
    ProductReasoningTrainError,
    _batches,
    load_product_backbone,
    load_trainable_checkpoint,
    pack_training_embeddings,
    render_reasoning_messages,
    reservoir_rows_with_sha256,
)
from q36_mtr_roles import (
    ALPHA,
    CONTROLLED_LAYERS,
    HIDDEN_SIZE,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    QUANTIZATION,
    RANK,
    ROLE_CHECKPOINT_SCHEMA,
    TRAINABLE_PARAMETERS,
    TRAINABLE_MASTER_DTYPE,
    Q36MTRRoleError,
    role_contract,
    role_spec,
    sequence_geometry_receipt,
    validate_owner_warm_start,
)
from shared_post_mlp_revision import (
    SharedPostMLPConfig,
    SharedPostMLPProductModel,
    trainable_state,
    trainable_state_sha256,
)
from ttr1_revision import DRAFT_MARKER, tokenize_with_draft_mask

SCHEMA = "shohin-q36-mtr-role-training-v1"


class Q36MTRTrainingError(RuntimeError):
    """The Q36-MTR role-training contract was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_role_checkpoint(
    path: Path,
    model: SharedPostMLPProductModel,
    update: int,
    metadata: dict[str, Any],
) -> None:
    """Publish only the role-owned residual state; optimizer carryover is forbidden."""

    if path.exists() or path.is_symlink():
        raise Q36MTRTrainingError("Q36-MTR role checkpoint already exists")
    state = trainable_state(model)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise Q36MTRTrainingError("Q36-MTR role checkpoint temporary exists")
    torch.save(
        {
            "schema": ROLE_CHECKPOINT_SCHEMA,
            "update": update,
            "trainable_state": state,
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def tokenize_role_rows(
    tokenizer: Any,
    rows: list[dict[str, str]],
    *,
    role: str,
    max_sequence_length: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], dict[str, Any]]:
    """Tokenize without truncation and retain the matched causal mask."""

    spec = role_spec(role)
    prompts: list[list[int]] = []
    responses: list[list[int]] = []
    draft_masks: list[list[int]] = []
    maximum_observed = 0
    for index, row in enumerate(rows):
        question = str(row["question"])
        if spec.data_kind == "source_only" and DRAFT_MARKER in question:
            raise Q36MTRTrainingError("source-only owner received an internal draft")
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            enable_thinking=False,
        )
        if spec.data_kind == "source_only":
            prompt = [
                int(value)
                for value in tokenizer.encode(rendered, add_special_tokens=False)
            ]
            mask = [1] * len(prompt)
        else:
            prompt, mask, _ = tokenize_with_draft_mask(tokenizer, rendered)
        response = [
            int(value)
            for value in tokenizer.encode(
                str(row["response"]), add_special_tokens=False
            )
        ]
        response.append(int(tokenizer.eos_token_id))
        total = len(prompt) + len(response)
        maximum_observed = max(maximum_observed, total)
        if not prompt or len(response) < 2 or total > max_sequence_length:
            raise Q36MTRTrainingError(
                f"Q36-MTR row {index} is empty or requires {total} tokens"
            )
        prompts.append(prompt)
        responses.append(response)
        draft_masks.append(mask)
    receipt = sequence_geometry_receipt(prompts, responses, draft_masks)
    receipt.update(
        {
            "maximum_observed_tokens": maximum_observed,
            "maximum_sequence_length": max_sequence_length,
            "truncated_rows": 0,
            "token_positions_deleted": 0,
            "source_only": spec.data_kind == "source_only",
        }
    )
    if spec.data_kind == "source_only" and receipt["draft_masked_tokens"] != 0:
        raise Q36MTRTrainingError("source-only owner has a masked draft")
    if spec.data_kind != "source_only" and receipt["draft_masked_tokens"] <= 0:
        raise Q36MTRTrainingError("revision role has no informative draft span")
    return prompts, responses, draft_masks, receipt


def full_sequence_position_ids(attention: torch.Tensor) -> torch.Tensor:
    """Return positions before applying the hidden arm's attention intervention."""

    if attention.ndim != 2 or attention.shape[1] < 1:
        raise Q36MTRTrainingError("Q36-MTR packed attention geometry differs")
    return (
        torch.arange(attention.shape[1], device=attention.device, dtype=torch.long)
        .unsqueeze(0)
        .expand(attention.shape[0], -1)
    )


def training_consumption_receipt(
    examples: list[tuple[list[int], list[int], list[int]]],
    *,
    updates: int,
    gradient_accumulation: int,
    batch_size: int,
) -> dict[str, Any]:
    """Hash the exact deterministic presentation prefix consumed by training."""

    if (
        not examples
        or min(updates, gradient_accumulation, batch_size) <= 0
        or len(examples) % batch_size
    ):
        raise Q36MTRTrainingError("Q36-MTR consumption geometry differs")
    batches = list(_batches(examples, batch_size))
    microsteps = updates * gradient_accumulation
    indices: list[int] = []
    token_digest = hashlib.sha256()
    mask_digest = hashlib.sha256()
    for microstep in range(microsteps):
        batch_index = microstep % len(batches)
        first_index = batch_index * batch_size
        for offset, (prompt, response, mask) in enumerate(batches[batch_index]):
            index = first_index + offset
            indices.append(index)
            token_digest.update(
                (
                    json.dumps(
                        {"prompt": prompt, "response": response},
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode()
            )
            mask_digest.update(
                (json.dumps(mask, separators=(",", ":")) + "\n").encode()
            )
    index_preimage = ("\n".join(map(str, indices)) + "\n").encode()
    return {
        "dataset_presentations": len(examples),
        "optimizer_updates": updates,
        "gradient_accumulation": gradient_accumulation,
        "batch_size": batch_size,
        "microsteps": microsteps,
        "consumed_presentations": len(indices),
        "unique_consumed_presentations": len(set(indices)),
        "complete_dataset_cycles": len(indices) // len(examples),
        "partial_cycle_presentations": len(indices) % len(examples),
        "presentation_index_sha256": hashlib.sha256(index_preimage).hexdigest(),
        "consumed_token_geometry_sha256": token_digest.hexdigest(),
        "consumed_draft_attention_sha256": mask_digest.hexdigest(),
    }


def _validate_arguments(args: argparse.Namespace) -> None:
    spec = role_spec(args.role)
    expected = {
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "quantization": QUANTIZATION,
        "updates": spec.updates,
        "max_rows": spec.max_rows,
        "max_sequence_length": spec.max_sequence_length,
        "learning_rate": spec.learning_rate,
        "gradient_accumulation": spec.gradient_accumulation,
        "seed": spec.seed,
        "data_seed": spec.data_seed,
        "controlled_layers": CONTROLLED_LAYERS,
        "rank": RANK,
        "alpha": ALPHA,
    }
    observed = {key: getattr(args, key) for key in expected}
    if observed != expected:
        raise Q36MTRTrainingError(
            f"Q36-MTR role settings differ: expected={expected} observed={observed}"
        )
    if (args.warm_start_checkpoint is None) != (spec.warm_start_role is None):
        raise Q36MTRTrainingError("Q36-MTR warm-start role differs")
    if args.batch_size != 1 or args.checkpoint_interval != spec.updates:
        raise Q36MTRTrainingError("Q36-MTR batch/checkpoint geometry differs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    _validate_arguments(args)
    spec = role_spec(args.role)
    if args.output.exists() or args.output.is_symlink():
        raise Q36MTRTrainingError("Q36-MTR role output already exists")
    if sha256_file(args.model_source_root / "config.json") != MODEL_CONFIG_SHA256:
        raise Q36MTRTrainingError("Q36-MTR host config differs")
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTRTrainingError("Q36-MTR environment receipt differs")
    environment = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        environment.get("schema") != "shohin-q36-mtr-environment-v1"
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTRTrainingError("Q36-MTR environment contract differs")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rows, data_sha256 = reservoir_rows_with_sha256(
        args.data, args.max_rows, args.data_seed
    )
    if len(rows) != args.max_rows:
        raise Q36MTRTrainingError("Q36-MTR selected-row geometry differs")
    prompts, responses, draft_masks, sequence_receipt = tokenize_role_rows(
        tokenizer,
        rows,
        role=args.role,
        max_sequence_length=args.max_sequence_length,
    )

    backbone, loader = load_product_backbone(
        args.model_root,
        "causal",
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization=args.quantization,
    )
    text_config = getattr(backbone.config, "text_config", backbone.config)
    if int(text_config.hidden_size) != HIDDEN_SIZE:
        raise Q36MTRTrainingError("Q36-MTR hidden size differs")
    config = SharedPostMLPConfig(
        hidden_size=HIDDEN_SIZE,
        controlled_layers=args.controlled_layers,
        rank=args.rank,
        alpha=args.alpha,
    )
    # device_map already placed the NF4 backbone. Newly created residuals
    # inherit the base MLP device, so the quantized wrapper must not be moved.
    model = SharedPostMLPProductModel(
        backbone, config, draft_control=spec.draft_control
    )
    trainable_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if (
        model.trainable_parameter_count() != TRAINABLE_PARAMETERS
        or not trainable_names
        or any(
            parameter.dtype != torch.float32
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        or any(
            not (name.endswith("adapter_a.weight") or name.endswith("adapter_b.weight"))
            for name in trainable_names
        )
    ):
        raise Q36MTRTrainingError("Q36-MTR trainable surface differs")
    trainable_name_digest = model.trainable_parameter_name_sha256()
    controlled_indices = list(
        range(
            len(model.text_model.layers) - CONTROLLED_LAYERS,
            len(model.text_model.layers),
        )
    )

    warm_start_update = None
    warm_start_sha256 = None
    if args.warm_start_checkpoint is not None:
        warm_start_update, owner_metadata = load_trainable_checkpoint(
            args.warm_start_checkpoint, model
        )
        loaded_trainable_state_sha256 = trainable_state_sha256(trainable_state(model))
        try:
            validate_owner_warm_start(
                owner_metadata,
                checkpoint_update=warm_start_update,
                trainable_parameters=model.trainable_parameter_count(),
                trainable_parameter_name_sha256=trainable_name_digest,
                loaded_trainable_state_sha256=loaded_trainable_state_sha256,
            )
        except Q36MTRRoleError as error:
            raise Q36MTRTrainingError(str(error)) from error
        warm_start_sha256 = sha256_file(args.warm_start_checkpoint)
    initial_trainable_state_sha256 = trainable_state_sha256(trainable_state(model))

    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    examples = list(zip(prompts, responses, draft_masks, strict=True))
    batches = list(_batches(examples, args.batch_size))
    consumption_receipt = training_consumption_receipt(
        examples,
        updates=args.updates,
        gradient_accumulation=args.gradient_accumulation,
        batch_size=args.batch_size,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    optimizer_state_entries_before_training = len(optimizer.state)
    if optimizer_state_entries_before_training != 0:
        raise Q36MTRTrainingError("Q36-MTR optimizer did not start empty")
    metadata = {
        **role_contract(args.role),
        "schema": SCHEMA,
        "shared_post_mlp_config": {
            "hidden_size": HIDDEN_SIZE,
            "controlled_layers": CONTROLLED_LAYERS,
            "rank": RANK,
            "alpha": ALPHA,
        },
        "draft_control": spec.draft_control,
        "model_root": str(args.model_source_root.resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_loader": loader,
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "selected_rows": len(rows),
        "trainable_parameter_name_sha256": trainable_name_digest,
        "trainable_master_dtype": TRAINABLE_MASTER_DTYPE,
        "trainable_compute_dtype": "bfloat16",
        "controlled_layer_indices": controlled_indices,
        "source_only_model_visible": args.role == "owner",
        "internal_draft_visible": args.role == "aligned",
        "draft_token_bytes_present": args.role != "owner",
        "draft_information_available": args.role == "aligned",
        "draft_attention_applied": args.role == "draft_hidden",
        "sequence_custody": sequence_receipt,
        "training_consumption": consumption_receipt,
        "warm_start_checkpoint": (
            str(args.warm_start_checkpoint.resolve())
            if args.warm_start_checkpoint is not None
            else None
        ),
        "warm_start_checkpoint_sha256": warm_start_sha256,
        "warm_start_update": warm_start_update,
        "optimizer_restored": False,
        "optimizer_initial_state_empty": True,
        "optimizer_state_entries_before_training": 0,
        "optimizer_state_serialized": False,
        "checkpoint_trainable_only": True,
        "router_expert_checkpoint_tensors": 0,
        "initial_trainable_state_sha256": initial_trainable_state_sha256,
        "environment_receipt": str(args.environment_receipt.resolve()),
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "assessor_board_access_count": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }

    model.train()
    model.reset_routing_receipt()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = microstep = charged_tokens = 0
    trace: list[dict[str, Any]] = []
    while update < args.updates:
        raw_batch = batches[microstep % len(batches)]
        batch_prompts = [item[0] for item in raw_batch]
        batch_responses = [item[1] for item in raw_batch]
        batch_masks = [item[2] for item in raw_batch]
        applied_masks = batch_masks if args.role == "draft_hidden" else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs, attention, labels, charged = pack_training_embeddings(
                model.text_model.embed_tokens,
                batch_prompts,
                batch_responses,
                None,
                tokenizer.pad_token_id,
                prompt_attention_rows=applied_masks,
            )
            positions = full_sequence_position_ids(attention)
            outputs = model.text_model(
                inputs_embeds=inputs,
                attention_mask=attention,
                position_ids=positions,
                use_cache=False,
            )
            logits = model.lm_head(outputs.last_hidden_state)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        charged_tokens += int(charged)
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue
        trainables = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        progress = update / max(args.updates - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % args.log_interval == 0:
            elapsed = time.monotonic() - started
            event = {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "charged_tokens": charged_tokens,
                "charged_tokens_per_second": charged_tokens / elapsed,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    final_trainable_state_sha256 = trainable_state_sha256(trainable_state(model))
    metadata["final_trainable_state_sha256"] = final_trainable_state_sha256
    metadata["serialization_restore_exact"] = True
    checkpoint = args.output / f"checkpoint_{update:07d}.pt"
    _save_role_checkpoint(checkpoint, model, update, metadata)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        saved.get("schema") != ROLE_CHECKPOINT_SCHEMA
        or saved.get("update") != update
        or saved.get("metadata") != metadata
        or not isinstance(saved.get("trainable_state"), dict)
        or trainable_state_sha256(saved["trainable_state"])
        != final_trainable_state_sha256
        or "optimizer" in saved
        or set(saved) != {"schema", "update", "trainable_state", "metadata"}
        or set(saved["trainable_state"]) != set(trainable_names)
    ):
        raise Q36MTRTrainingError("Q36-MTR saved role state differs")
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    restored_update, restored_metadata = load_trainable_checkpoint(checkpoint, model)
    if (
        restored_update != update
        or restored_metadata != metadata
        or trainable_state_sha256(trainable_state(model))
        != final_trainable_state_sha256
    ):
        raise Q36MTRTrainingError("Q36-MTR live role restore differs")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    report = {
        **metadata,
        "status": "complete",
        "update": update,
        "updates": update,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "charged_tokens": charged_tokens,
        "elapsed_seconds": elapsed,
        "charged_tokens_per_second": charged_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "residual_receipt": model.routing_receipt(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "trace": trace,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(role_contracts()), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-config-sha256", default=MODEL_CONFIG_SHA256)
    parser.add_argument("--quantization", default=QUANTIZATION)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--controlled-layers", type=int, default=CONTROLLED_LAYERS)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int)
    args = parser.parse_args()
    spec = role_spec(args.role)
    for field in (
        "updates",
        "gradient_accumulation",
        "max_rows",
        "max_sequence_length",
        "learning_rate",
        "seed",
        "data_seed",
        "checkpoint_interval",
    ):
        if getattr(args, field) is None:
            value = (
                spec.updates if field == "checkpoint_interval" else getattr(spec, field)
            )
            setattr(args, field, value)
    return args


def role_contracts() -> tuple[str, ...]:
    return ("owner", "aligned", "draft_hidden")


def main() -> int:
    try:
        report = run(parse_args())
    except (ProductReasoningTrainError, Q36MTRRoleError) as error:
        raise Q36MTRTrainingError(str(error)) from error
    print(
        json.dumps(
            {
                "role": report["role"],
                "updates": report["updates"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
