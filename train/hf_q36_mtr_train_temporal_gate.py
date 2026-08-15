#!/usr/bin/env python3
"""Train the Q36 tokenwise temporal residual gate on one H100."""

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

from hf_product_reasoning_train import (
    ProductReasoningTrainError,
    _batches,
    load_product_backbone,
    pack_training_embeddings,
    resolve_product_backbone_layout,
)
from hf_q36_mtr_train_role import (
    chunked_causal_cross_entropy,
    full_sequence_position_ids,
    sha256_file,
    tokenize_role_rows,
    training_consumption_receipt,
)
from q36_mtr_roles import (
    ALPHA,
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    QUANTIZATION,
    RANK,
    REVISION_DATA_SEED,
    REVISION_MAX_SEQUENCE_LENGTH,
    REVISION_PRESENTATIONS,
    REVISION_UPDATES,
    TRAINABLE_MASTER_DTYPE,
    Q36MTRRoleError,
    load_role_checkpoint_payload,
    validate_backbone_geometry,
    validate_backbone_moe_surface,
)
from shared_post_mlp_revision import trainable_state, trainable_state_sha256
from temporal_residual_gate import (
    MultiTrajectoryGatedProductModel,
    MultiTrajectoryResidualGateConfig,
    TemporalGatedProductModel,
    TemporalResidualGateConfig,
    TemporalResidualGateError,
)

SCHEMA = "shohin-q36-mtr-temporal-gate-training-v1"
CHECKPOINT_SCHEMA = "shohin-q36-mtr-temporal-gate-checkpoint-v1"
MULTI_SCHEMA = "shohin-q36-mtr-multi-trajectory-gate-training-v1"
MULTI_CHECKPOINT_SCHEMA = "shohin-q36-mtr-multi-trajectory-gate-checkpoint-v1"
TRI_SCHEMA = "shohin-q36-mtr-tri-trajectory-gate-training-v1"
TRI_CHECKPOINT_SCHEMA = "shohin-q36-mtr-tri-trajectory-gate-checkpoint-v1"
GATE_SEED = 2026081511
GATE_LEARNING_RATE = 2e-4
GATE_GRADIENT_ACCUMULATION = 8
GATE_BATCH_SIZE = 1
GATE_INITIAL_REVISION_WEIGHT = 0.1
GATE_ROUTING_SUPERVISION_WEIGHT = 0.1
GATE_CAUSAL_LOSS_WEIGHTS = (0.0, 1.0)
GATE_PARAMETERS = len(CONTROLLED_LAYER_INDICES) * (HIDDEN_SIZE + 1)
MULTI_BRANCHES = ("revision", "draft_hidden")
MULTI_INITIAL_WEIGHTS = (0.9, 0.1)
MULTI_PRESENTATIONS = 1_167
MULTI_DATA_SEED = 2026081514
MULTI_GATE_PARAMETERS = (
    len(CONTROLLED_LAYER_INDICES) * len(MULTI_BRANCHES) * (HIDDEN_SIZE + 1)
)
TRI_BRANCHES = ("owner", "revision", "draft_hidden")
TRI_INITIAL_WEIGHTS = (0.1, 0.8, 0.1)
TRI_PRESENTATIONS = 1_189
TRI_DATA_SEED = 2026081517
TRI_GATE_PARAMETERS = (
    len(CONTROLLED_LAYER_INDICES) * len(TRI_BRANCHES) * (HIDDEN_SIZE + 1)
)
LOSS_CHUNK_SIZE = 512
ROUTING_TARGETS = {
    "base_only": 0.0,
    "both_correct": 0.0,
    "both_wrong": None,
    "expert_only": 1.0,
}
MULTI_ROW_SCHEMA = "shohin-q36-mtr-multi-trajectory-gate-train-v1"
TRI_ROW_SCHEMA = "shohin-q36-mtr-tri-trajectory-gate-train-v1"


class Q36MTRTemporalGateTrainingError(RuntimeError):
    """Temporal gate inputs, training, or checkpointing differ."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _role_pair(
    owner_checkpoint: Path, revision_checkpoint: Path
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    try:
        owner = load_role_checkpoint_payload(owner_checkpoint)
        revision = load_role_checkpoint_payload(revision_checkpoint)
    except Q36MTRRoleError as error:
        raise Q36MTRTemporalGateTrainingError(str(error)) from error
    owner_metadata = owner["metadata"]
    revision_metadata = revision["metadata"]
    shared = (
        "model_revision",
        "model_config_sha256",
        "controlled_layer_indices",
        "trainable_parameter_name_sha256",
        "trainable_parameters",
        "trainable_master_dtype",
    )
    if (
        owner_metadata.get("role") != "owner"
        or revision_metadata.get("role") != "aligned"
        or any(owner_metadata.get(key) != revision_metadata.get(key) for key in shared)
        or owner_metadata.get("model_revision") != MODEL_REVISION
        or owner_metadata.get("model_config_sha256") != MODEL_CONFIG_SHA256
        or owner_metadata.get("controlled_layer_indices")
        != list(CONTROLLED_LAYER_INDICES)
        or owner_metadata.get("trainable_master_dtype") != TRAINABLE_MASTER_DTYPE
        or set(owner["trainable_state"]) != set(revision["trainable_state"])
    ):
        raise Q36MTRTemporalGateTrainingError("temporal gate role pair differs")
    owner_checkpoint_sha = sha256_file(owner_checkpoint)
    owner_sha = trainable_state_sha256(owner["trainable_state"])
    revision_sha = trainable_state_sha256(revision["trainable_state"])
    if (
        owner_sha == revision_sha
        or owner_metadata.get("final_trainable_state_sha256") != owner_sha
        or owner_metadata.get("warm_start_checkpoint") is not None
        or owner_metadata.get("warm_start_checkpoint_sha256") is not None
        or revision_metadata.get("warm_start_checkpoint_sha256") != owner_checkpoint_sha
        or revision_metadata.get("warm_start_update") != REVISION_UPDATES
        or revision_metadata.get("initial_trainable_state_sha256") != owner_sha
    ):
        raise Q36MTRTemporalGateTrainingError("temporal gate role lineage differs")
    return (
        owner["trainable_state"],
        revision["trainable_state"],
        {
            "owner_checkpoint_sha256": owner_checkpoint_sha,
            "revision_checkpoint_sha256": sha256_file(revision_checkpoint),
            "owner_state_sha256": owner_sha,
            "revision_state_sha256": revision_sha,
        },
    )


def _role_bank(
    owner_checkpoint: Path,
    revision_checkpoint: Path,
    draft_hidden_checkpoint: Path,
    *,
    include_owner: bool = False,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Load sibling trajectories and optionally expose their shared owner."""

    try:
        owner = load_role_checkpoint_payload(owner_checkpoint)
        revision = load_role_checkpoint_payload(revision_checkpoint)
        hidden = load_role_checkpoint_payload(draft_hidden_checkpoint)
    except Q36MTRRoleError as error:
        raise Q36MTRTemporalGateTrainingError(str(error)) from error
    owner_metadata = owner["metadata"]
    owner_checkpoint_sha = sha256_file(owner_checkpoint)
    owner_state_sha = trainable_state_sha256(owner["trainable_state"])
    shared = (
        "model_revision",
        "model_config_sha256",
        "controlled_layer_indices",
        "trainable_parameter_name_sha256",
        "trainable_parameters",
        "trainable_master_dtype",
    )
    branches = (
        ("revision", "aligned", revision),
        ("draft_hidden", "draft_hidden", hidden),
    )
    if (
        owner_metadata.get("role") != "owner"
        or owner_metadata.get("model_revision") != MODEL_REVISION
        or owner_metadata.get("model_config_sha256") != MODEL_CONFIG_SHA256
        or owner_metadata.get("controlled_layer_indices")
        != list(CONTROLLED_LAYER_INDICES)
        or owner_metadata.get("trainable_master_dtype") != TRAINABLE_MASTER_DTYPE
        or owner_metadata.get("warm_start_checkpoint") is not None
        or owner_metadata.get("warm_start_checkpoint_sha256") is not None
        or owner_metadata.get("final_trainable_state_sha256") != owner_state_sha
    ):
        raise Q36MTRTemporalGateTrainingError("multi-trajectory owner differs")
    role_states: dict[str, dict[str, torch.Tensor]] = {}
    receipt: dict[str, Any] = {
        "owner_checkpoint_sha256": owner_checkpoint_sha,
        "owner_state_sha256": owner_state_sha,
    }
    if include_owner:
        role_states["owner"] = owner["trainable_state"]
    for branch, role, payload in branches:
        metadata = payload["metadata"]
        state = payload["trainable_state"]
        state_sha = trainable_state_sha256(state)
        if (
            metadata.get("role") != role
            or any(metadata.get(key) != owner_metadata.get(key) for key in shared)
            or metadata.get("warm_start_checkpoint_sha256") != owner_checkpoint_sha
            or metadata.get("warm_start_update") != REVISION_UPDATES
            or metadata.get("initial_trainable_state_sha256") != owner_state_sha
            or metadata.get("final_trainable_state_sha256") != state_sha
            or set(state) != set(owner["trainable_state"])
            or state_sha == owner_state_sha
        ):
            raise Q36MTRTemporalGateTrainingError(
                f"multi-trajectory {branch} lineage differs"
            )
        role_states[branch] = state
        receipt[f"{branch}_checkpoint_sha256"] = sha256_file(
            revision_checkpoint if branch == "revision" else draft_hidden_checkpoint
        )
        receipt[f"{branch}_state_sha256"] = state_sha
    if receipt["revision_state_sha256"] == receipt["draft_hidden_state_sha256"]:
        raise Q36MTRTemporalGateTrainingError("multi-trajectory branches are identical")
    return role_states, receipt


def _routing_rows_with_sha256(
    path: Path, limit: int, seed: int, *, architecture: str
) -> tuple[list[dict[str, Any]], str]:
    """Hash and shuffle routing rows without dropping supervisor targets."""

    categorical = architecture in {"multi_trajectory", "tri_trajectory"}
    branch_names = TRI_BRANCHES if architecture == "tri_trajectory" else MULTI_BRANCHES
    expected_schema = {
        "temporal": "shohin-q36-mtr-revision-train-v1",
        "multi_trajectory": MULTI_ROW_SCHEMA,
        "tri_trajectory": TRI_ROW_SCHEMA,
    }.get(architecture)
    if expected_schema is None:
        raise Q36MTRTemporalGateTrainingError("temporal architecture differs")
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise Q36MTRTemporalGateTrainingError(
                    "temporal routing row is unreadable"
                ) from error
            if (
                not isinstance(row, dict)
                or row.get("schema") != expected_schema
                or not isinstance(row.get("question"), str)
                or not row["question"].strip()
                or not isinstance(row.get("response"), str)
                or not row["response"].strip()
                or not isinstance(row.get("outcome_class"), str)
            ):
                raise Q36MTRTemporalGateTrainingError("temporal routing row differs")
            if categorical:
                target = row.get("routing_target")
                if (
                    row.get("branch_names") != list(branch_names)
                    or not isinstance(target, list)
                    or len(target) != len(branch_names)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) < 0.0
                        for value in target
                    )
                    or not math.isclose(sum(float(value) for value in target), 1.0)
                ):
                    raise Q36MTRTemporalGateTrainingError(
                        "multi-trajectory routing target differs"
                    )
            elif row["outcome_class"] not in ROUTING_TARGETS:
                raise Q36MTRTemporalGateTrainingError(
                    "temporal routing outcome differs"
                )
            rows.append(row)
    if len(rows) != limit:
        raise Q36MTRTemporalGateTrainingError("temporal routing data geometry differs")
    random.Random(seed).shuffle(rows)
    return rows, digest.hexdigest()


def _validate_gate_state(
    state: dict[str, torch.Tensor], gate_parameters: int = GATE_PARAMETERS
) -> None:
    if (
        len(state) != len(CONTROLLED_LAYER_INDICES) * 2
        or sum(tensor.numel() for tensor in state.values()) != gate_parameters
        or any(
            not isinstance(name, str)
            or not (name.endswith("gate_weight") or name.endswith("gate_bias"))
            or not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
            for name, tensor in state.items()
        )
    ):
        raise Q36MTRTemporalGateTrainingError("temporal gate state differs")


def _response_routing_mask(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 2 or labels.dtype != torch.long:
        raise Q36MTRTemporalGateTrainingError("temporal routing label geometry differs")
    mask = labels.ne(-100)
    if not mask.any():
        raise Q36MTRTemporalGateTrainingError("temporal routing response mask is empty")
    return mask


def save_gate_checkpoint(
    path: Path,
    model: TemporalGatedProductModel | MultiTrajectoryGatedProductModel,
    update: int,
    metadata: dict[str, Any],
    *,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
    gate_parameters: int = GATE_PARAMETERS,
) -> None:
    if path.exists() or path.is_symlink() or update != REVISION_UPDATES:
        raise Q36MTRTemporalGateTrainingError("temporal gate checkpoint target differs")
    state = trainable_state(model)
    _validate_gate_state(state, gate_parameters)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "schema": checkpoint_schema,
            "update": update,
            "trainable_state": state,
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def restore_gate_checkpoint(
    path: Path,
    model: TemporalGatedProductModel | MultiTrajectoryGatedProductModel,
    *,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
    gate_parameters: int = GATE_PARAMETERS,
) -> tuple[int, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRTemporalGateTrainingError("temporal gate checkpoint is absent")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "update", "trainable_state", "metadata"}
        or payload.get("schema") != checkpoint_schema
        or payload.get("update") != REVISION_UPDATES
        or not isinstance(payload.get("trainable_state"), dict)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise Q36MTRTemporalGateTrainingError("temporal gate checkpoint differs")
    saved = payload["trainable_state"]
    _validate_gate_state(saved, gate_parameters)
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(saved) != set(current):
        raise Q36MTRTemporalGateTrainingError("temporal gate restore names differ")
    with torch.no_grad():
        for name, parameter in current.items():
            tensor = saved[name]
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise Q36MTRTemporalGateTrainingError(
                    "temporal gate restore geometry differs"
                )
            parameter.copy_(tensor.to(device=parameter.device))
    return int(payload["update"]), payload["metadata"]


def _validate_args(args: argparse.Namespace) -> None:
    architecture = getattr(args, "architecture", "temporal")
    categorical = architecture in {"multi_trajectory", "tri_trajectory"}
    if architecture not in {"temporal", "multi_trajectory", "tri_trajectory"}:
        raise Q36MTRTemporalGateTrainingError("temporal gate architecture differs")
    presentations = {
        "temporal": REVISION_PRESENTATIONS,
        "multi_trajectory": MULTI_PRESENTATIONS,
        "tri_trajectory": TRI_PRESENTATIONS,
    }[architecture]
    data_seed = {
        "temporal": REVISION_DATA_SEED,
        "multi_trajectory": MULTI_DATA_SEED,
        "tri_trajectory": TRI_DATA_SEED,
    }[architecture]
    initial_weights = (
        TRI_INITIAL_WEIGHTS
        if architecture == "tri_trajectory"
        else MULTI_INITIAL_WEIGHTS
    )
    expected = {
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "updates": REVISION_UPDATES,
        "max_rows": presentations,
        "max_sequence_length": REVISION_MAX_SEQUENCE_LENGTH,
        "learning_rate": GATE_LEARNING_RATE,
        "gradient_accumulation": GATE_GRADIENT_ACCUMULATION,
        "batch_size": GATE_BATCH_SIZE,
        "seed": GATE_SEED,
        "data_seed": data_seed,
        "initial_revision_weight": GATE_INITIAL_REVISION_WEIGHT,
        "loss_chunk_size": LOSS_CHUNK_SIZE,
    }
    observed = {key: getattr(args, key) for key in expected}
    supervision = (
        {GATE_ROUTING_SUPERVISION_WEIGHT}
        if categorical
        else {
            0.0,
            GATE_ROUTING_SUPERVISION_WEIGHT,
        }
    )
    causal_loss_weight = getattr(args, "causal_loss_weight", 1.0)
    if (
        observed != expected
        or args.routing_supervision_weight not in supervision
        or causal_loss_weight not in GATE_CAUSAL_LOSS_WEIGHTS
        or (not categorical and causal_loss_weight != 1.0)
        or (
            categorical
            and tuple(getattr(args, "initial_branch_weights", ())) != initial_weights
        )
    ):
        raise Q36MTRTemporalGateTrainingError(
            f"temporal gate settings differ: expected={expected} observed={observed}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    _validate_args(args)
    architecture = getattr(args, "architecture", "temporal")
    categorical = architecture in {"multi_trajectory", "tri_trajectory"}
    branch_names = TRI_BRANCHES if architecture == "tri_trajectory" else MULTI_BRANCHES
    initial_weights = (
        TRI_INITIAL_WEIGHTS
        if architecture == "tri_trajectory"
        else MULTI_INITIAL_WEIGHTS
    )
    if not torch.cuda.is_available():
        raise Q36MTRTemporalGateTrainingError("temporal gate training requires CUDA")
    if args.output.exists() or args.output.is_symlink():
        raise Q36MTRTemporalGateTrainingError("temporal gate output exists")
    if sha256_file(args.model_source_root / "config.json") != MODEL_CONFIG_SHA256:
        raise Q36MTRTemporalGateTrainingError("temporal gate host config differs")
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTRTemporalGateTrainingError("temporal gate environment differs")
    environment = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        environment.get("schema") != "shohin-q36-mtr-environment-v1"
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTRTemporalGateTrainingError(
            "temporal gate environment contract differs"
        )
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rows, data_sha256 = _routing_rows_with_sha256(
        args.data, args.max_rows, args.data_seed, architecture=architecture
    )
    prompts, responses, draft_masks, sequence_receipt = tokenize_role_rows(
        tokenizer,
        rows,
        role="aligned",
        max_sequence_length=args.max_sequence_length,
    )
    outcomes = [str(row["outcome_class"]) for row in rows]
    routing_targets: list[float | list[float] | None] = [
        (
            [float(value) for value in row["routing_target"]]
            if categorical
            else ROUTING_TARGETS[row["outcome_class"]]
        )
        for row in rows
    ]
    sequence_examples = list(zip(prompts, responses, draft_masks, strict=True))
    examples = list(
        zip(prompts, responses, draft_masks, outcomes, routing_targets, strict=True)
    )
    batches = list(_batches(examples, args.batch_size))
    consumption = training_consumption_receipt(
        sequence_examples,
        updates=args.updates,
        gradient_accumulation=args.gradient_accumulation,
        batch_size=args.batch_size,
    )
    if categorical:
        draft_hidden_checkpoint = getattr(args, "draft_hidden_checkpoint", None)
        if not isinstance(draft_hidden_checkpoint, Path):
            raise Q36MTRTemporalGateTrainingError(
                "multi-trajectory draft-hidden checkpoint is absent"
            )
        role_states, role_receipt = _role_bank(
            args.owner_checkpoint,
            args.revision_checkpoint,
            draft_hidden_checkpoint,
            include_owner=architecture == "tri_trajectory",
        )
    else:
        owner_state, revision_state, role_receipt = _role_pair(
            args.owner_checkpoint, args.revision_checkpoint
        )
    backbone, loader = load_product_backbone(
        args.model_root,
        "causal",
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization=QUANTIZATION,
    )
    try:
        controlled_indices = validate_backbone_geometry(backbone)
        moe_surface = validate_backbone_moe_surface(backbone)
    except Q36MTRRoleError as error:
        raise Q36MTRTemporalGateTrainingError(str(error)) from error
    text_model, lm_head, hidden_size, backbone_layout = resolve_product_backbone_layout(
        backbone
    )
    if hidden_size != HIDDEN_SIZE or controlled_indices != list(
        CONTROLLED_LAYER_INDICES
    ):
        raise Q36MTRTemporalGateTrainingError("temporal gate backbone differs")
    if categorical:
        model = MultiTrajectoryGatedProductModel(
            backbone,
            text_model,
            lm_head,
            MultiTrajectoryResidualGateConfig(
                HIDDEN_SIZE,
                RANK,
                ALPHA,
                branch_names,
                initial_weights,
            ),
            role_states=role_states,
            controlled_layer_indices=CONTROLLED_LAYER_INDICES,
        )
        gate_parameters = (
            TRI_GATE_PARAMETERS
            if architecture == "tri_trajectory"
            else MULTI_GATE_PARAMETERS
        )
        checkpoint_schema = (
            TRI_CHECKPOINT_SCHEMA
            if architecture == "tri_trajectory"
            else MULTI_CHECKPOINT_SCHEMA
        )
    else:
        model = TemporalGatedProductModel(
            backbone,
            text_model,
            lm_head,
            TemporalResidualGateConfig(
                HIDDEN_SIZE,
                RANK,
                ALPHA,
                args.initial_revision_weight,
            ),
            owner_state=owner_state,
            revision_state=revision_state,
            controlled_layer_indices=CONTROLLED_LAYER_INDICES,
        )
        gate_parameters = GATE_PARAMETERS
        checkpoint_schema = CHECKPOINT_SCHEMA
    trainables = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    initial_state = trainable_state(model)
    _validate_gate_state(initial_state, gate_parameters)
    if model.trainable_parameter_count() != gate_parameters or any(
        parameter.dtype != torch.float32 for parameter in trainables
    ):
        raise Q36MTRTemporalGateTrainingError("temporal gate trainables differ")
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()

    optimizer = torch.optim.AdamW(
        trainables,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    if optimizer.state:
        raise Q36MTRTemporalGateTrainingError("temporal gate optimizer is not fresh")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    model.reset_routing_receipt()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = microstep = charged_tokens = 0
    trace: list[dict[str, Any]] = []
    while update < args.updates:
        raw_batch = batches[microstep % len(batches)]
        batch_prompts = [item[0] for item in raw_batch]
        batch_responses = [item[1] for item in raw_batch]
        batch_outcomes = [item[3] for item in raw_batch]
        batch_targets = [item[4] for item in raw_batch]
        if len(batch_outcomes) != 1:
            raise Q36MTRTemporalGateTrainingError(
                "temporal routing batch geometry differs"
            )
        routing_target = batch_targets[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs, attention, labels, charged = pack_training_embeddings(
                model.text_model.embed_tokens,
                batch_prompts,
                batch_responses,
                None,
                tokenizer.pad_token_id,
            )
            outputs = model.text_model(
                inputs_embeds=inputs,
                attention_mask=attention,
                position_ids=full_sequence_position_ids(attention),
                use_cache=False,
            )
            causal_loss = (
                chunked_causal_cross_entropy(
                    outputs.last_hidden_state,
                    labels,
                    model.lm_head,
                    args.loss_chunk_size,
                )
                if args.causal_loss_weight
                else None
            )
            routing_mask = _response_routing_mask(labels)
            routing_loss = None
            loss = (
                args.causal_loss_weight * causal_loss
                if causal_loss is not None
                else None
            )
            if args.routing_supervision_weight and routing_target is not None:
                routing_loss = model.routing_supervision_loss(
                    routing_target, routing_mask
                )
                routing_term = args.routing_supervision_weight * routing_loss
                loss = routing_term if loss is None else loss + routing_term
            if loss is None:
                raise Q36MTRTemporalGateTrainingError(
                    "temporal gate has no training objective"
                )
            scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        charged_tokens += int(charged)
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        progress = update / max(args.updates - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % args.log_interval == 0:
            event = {
                "update": update,
                "loss": float(loss.detach()),
                "causal_loss": (
                    float(causal_loss.detach()) if causal_loss is not None else None
                ),
                "causal_loss_weight": args.causal_loss_weight,
                "routing_supervision_loss": (
                    float(routing_loss.detach()) if routing_loss is not None else None
                ),
                "routing_target": routing_target,
                "routing_supervised_tokens": (
                    int(routing_mask.sum()) if routing_target is not None else 0
                ),
                "outcome_class": batch_outcomes[0],
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "charged_tokens": charged_tokens,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)

    final_state = trainable_state(model)
    _validate_gate_state(final_state, gate_parameters)
    initial_sha256 = trainable_state_sha256(initial_state)
    final_sha256 = trainable_state_sha256(final_state)
    if initial_sha256 == final_sha256:
        raise Q36MTRTemporalGateTrainingError("temporal gate update is absent")
    metadata = {
        "schema": (
            TRI_SCHEMA
            if architecture == "tri_trajectory"
            else MULTI_SCHEMA if categorical else SCHEMA
        ),
        "architecture": (
            "q36-tokenwise-tri-trajectory-residual-gate-v1"
            if architecture == "tri_trajectory"
            else (
                "q36-tokenwise-multi-trajectory-residual-gate-v1"
                if categorical
                else "q36-tokenwise-temporal-residual-gate-v1"
            )
        ),
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_loader": loader,
        "backbone_layout": backbone_layout,
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "gate_parameters": gate_parameters,
        "branch_names": list(branch_names) if categorical else ["owner", "revision"],
        "initial_branch_weights": (
            list(initial_weights)
            if categorical
            else [1.0 - args.initial_revision_weight, args.initial_revision_weight]
        ),
        "initial_revision_weight": args.initial_revision_weight,
        "routing_supervision_weight": args.routing_supervision_weight,
        "causal_loss_weight": args.causal_loss_weight,
        "routing_supervision_targets": (
            "per-row-soft-target" if categorical else ROUTING_TARGETS
        ),
        "routing_supervision_mask": "response_tokens_only",
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "trainable_master_dtype": TRAINABLE_MASTER_DTYPE,
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "sequence_custody": sequence_receipt,
        "training_consumption": consumption,
        "role_receipt": role_receipt,
        "native_moe_surface": moe_surface,
        "initial_trainable_state_sha256": initial_sha256,
        "final_trainable_state_sha256": final_sha256,
        "optimizer_restored": False,
        "optimizer_state_serialized": False,
        "router_expert_checkpoint_tensors": 0,
        "assessor_access_count": 0,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
    }
    checkpoint = args.output / f"checkpoint_{update:07d}.pt"
    save_gate_checkpoint(
        checkpoint,
        model,
        update,
        metadata,
        checkpoint_schema=checkpoint_schema,
        gate_parameters=gate_parameters,
    )
    with torch.no_grad():
        for parameter in trainables:
            parameter.zero_()
    restored_update, restored_metadata = restore_gate_checkpoint(
        checkpoint,
        model,
        checkpoint_schema=checkpoint_schema,
        gate_parameters=gate_parameters,
    )
    if (
        restored_update != update
        or restored_metadata != metadata
        or trainable_state_sha256(trainable_state(model)) != final_sha256
    ):
        raise Q36MTRTemporalGateTrainingError("temporal gate restore differs")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    report = {
        **metadata,
        "status": "complete",
        "updates": update,
        "learning_rate": args.learning_rate,
        "routing_supervision_weight": args.routing_supervision_weight,
        "gradient_accumulation": args.gradient_accumulation,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "charged_tokens": charged_tokens,
        "elapsed_seconds": elapsed,
        "charged_tokens_per_second": charged_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "routing_receipt": model.routing_receipt(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "trace": trace,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("temporal", "multi_trajectory", "tri_trajectory"),
        default="temporal",
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-config-sha256", default=MODEL_CONFIG_SHA256)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--revision-checkpoint", type=Path, required=True)
    parser.add_argument("--draft-hidden-checkpoint", type=Path)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=REVISION_UPDATES)
    parser.add_argument("--max-rows", type=int, default=REVISION_PRESENTATIONS)
    parser.add_argument(
        "--max-sequence-length", type=int, default=REVISION_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument("--learning-rate", type=float, default=GATE_LEARNING_RATE)
    parser.add_argument(
        "--gradient-accumulation", type=int, default=GATE_GRADIENT_ACCUMULATION
    )
    parser.add_argument("--batch-size", type=int, default=GATE_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=GATE_SEED)
    parser.add_argument("--data-seed", type=int, default=REVISION_DATA_SEED)
    parser.add_argument(
        "--initial-revision-weight",
        type=float,
        default=GATE_INITIAL_REVISION_WEIGHT,
    )
    parser.add_argument(
        "--initial-branch-weights",
        type=float,
        nargs="+",
        default=MULTI_INITIAL_WEIGHTS,
    )
    parser.add_argument("--routing-supervision-weight", type=float, default=0.0)
    parser.add_argument("--causal-loss-weight", type=float, default=1.0)
    parser.add_argument("--loss-chunk-size", type=int, default=LOSS_CHUNK_SIZE)
    parser.add_argument("--log-interval", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (
        ProductReasoningTrainError,
        Q36MTRRoleError,
        TemporalResidualGateError,
    ) as error:
        raise Q36MTRTemporalGateTrainingError(str(error)) from error
    print(
        json.dumps(
            {
                "updates": report["updates"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
