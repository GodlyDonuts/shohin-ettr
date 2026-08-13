#!/usr/bin/env python3
"""Run the no-score Q36-MTR load/train/mask/restore mechanics gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from build_pcf1_data import revision_prompt
from hf_product_reasoning_eval import (
    GENERATED_ONLY_SEQUENCE_CONTRACT,
    _generate_adapter,
    _generation_arguments,
    _generation_stop_token_ids,
    _render_prompt,
)
from hf_product_reasoning_train import (
    _save_checkpoint,
    load_product_backbone,
    load_trainable_checkpoint,
    pack_training_embeddings,
    reservoir_rows_with_sha256,
)
from hf_q36_mtr_train_role import (
    full_sequence_position_ids,
    sha256_file,
    tokenize_role_rows,
)
from q36_mtr_roles import (
    ALPHA,
    CONTROLLED_LAYERS,
    HIDDEN_SIZE,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    Q36MTRRoleError,
    RANK,
    TRAINABLE_PARAMETERS,
    TRAINABLE_MASTER_DTYPE,
    role_contract,
    validate_matched_revision_geometry,
)
from shared_post_mlp_revision import SharedPostMLPConfig, SharedPostMLPProductModel

SCHEMA = "shohin-q36-mtr-mechanics-v1"
ROWS = 24
SEED = 2026080825
DATA_SEED = 2026080824
B1_SHA256 = "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549"


class Q36MTRMechanicsError(RuntimeError):
    """The Q36-MTR no-score mechanics contract failed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRMechanicsError(f"refusing existing mechanics report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _digest_rows(rows: Iterable[dict[str, Any]]) -> str:
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the exact stored tensor bytes without changing its dtype."""

    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def protected_parameter_receipt(model: Any) -> dict[str, Any]:
    """Bind every frozen parameter and specifically every MoE router/expert tensor."""

    protected: list[dict[str, Any]] = []
    moe: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        row = {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
            "version": int(parameter._version),
            "data_ptr": int(parameter.data_ptr()),
        }
        protected.append(row)
        lowered = name.casefold()
        if any(token in lowered for token in ("expert", "router", ".gate.")):
            moe.append({**row, "tensor_sha256": _tensor_sha256(parameter)})
    if not protected or not moe:
        raise Q36MTRMechanicsError("Q36-MTR protected MoE parameters are absent")
    return {
        "protected_parameter_count": len(protected),
        "protected_numel": sum(row["numel"] for row in protected),
        "protected_receipt_sha256": _digest_rows(protected),
        "router_expert_parameter_count": len(moe),
        "router_expert_numel": sum(row["numel"] for row in moe),
        "router_expert_receipt_sha256": _digest_rows(moe),
    }


def _trainable_state(model: Any) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise Q36MTRMechanicsError("Q36-MTR trainable state is empty")
    return state


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def causal_draft_intervention_receipt(
    model: Any,
    prompt: list[int],
    response: list[int],
    draft_mask: list[int],
    pad_token_id: int,
) -> dict[str, Any]:
    """Exercise the causal intervention on identical model-facing geometry."""

    if (
        not prompt
        or not response
        or len(prompt) != len(draft_mask)
        or not any(value == 0 for value in draft_mask)
        or any(value not in (0, 1) for value in draft_mask)
    ):
        raise Q36MTRMechanicsError("Q36-MTR causal counterfactual geometry differs")
    embedding = model.text_model.embed_tokens
    vocabulary = int(getattr(embedding, "num_embeddings", 0))
    if vocabulary < 2:
        raise Q36MTRMechanicsError("Q36-MTR embedding vocabulary differs")
    counterfactual = list(prompt)
    for index, visible in enumerate(draft_mask):
        if not visible:
            counterfactual[index] = (
                counterfactual[index] + max(vocabulary // 2, 1)
            ) % vocabulary
            if counterfactual[index] == prompt[index]:
                raise Q36MTRMechanicsError("Q36-MTR draft perturbation is empty")
    prompt_rows = [prompt, counterfactual]
    response_rows = [response, response]
    text_config = getattr(model.text_model, "config", None)
    router_top_k = int(getattr(text_config, "num_experts_per_tok", 0))
    if router_top_k <= 0:
        raise Q36MTRMechanicsError("Q36-MTR native router geometry is absent")

    def response_states(
        hidden: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        inputs, attention, labels, _ = pack_training_embeddings(
            embedding,
            prompt_rows,
            response_rows,
            None,
            pad_token_id,
            prompt_attention_rows=([draft_mask, draft_mask] if hidden else None),
        )
        positions = full_sequence_position_ids(attention)
        output = model.text_model(
            inputs_embeds=inputs,
            attention_mask=attention,
            position_ids=positions,
            use_cache=False,
            output_router_logits=True,
        )
        # The last prompt state predicts the first response token; subsequent
        # response states predict the remainder.  This is the complete
        # teacher-forced target-facing trajectory.
        states = output.last_hidden_state[
            :, len(prompt) - 1 : len(prompt) + len(response) - 1
        ]
        if states.shape[0] != 2 or states.shape[1] != len(response):
            raise Q36MTRMechanicsError("Q36-MTR causal state geometry differs")
        router_logits = getattr(output, "router_logits", None)
        if not isinstance(router_logits, (tuple, list)) or not router_logits:
            raise Q36MTRMechanicsError("Q36-MTR native router logits are absent")
        return states, positions, tuple(router_logits)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        aligned_states, aligned_positions, aligned_routes = response_states(False)
        hidden_states, hidden_positions, hidden_routes = response_states(True)
    model.train(was_training)
    aligned_delta = float(
        (aligned_states[0].float() - aligned_states[1].float()).abs().max().cpu()
    )
    hidden_delta = float(
        (hidden_states[0].float() - hidden_states[1].float()).abs().max().cpu()
    )
    positions_exact = torch.equal(aligned_positions, hidden_positions)

    def route_receipt(layers: tuple[torch.Tensor, ...], control: str) -> dict[str, Any]:
        layer_rows: list[dict[str, Any]] = []
        route_digest = hashlib.sha256()
        for layer_index, raw in enumerate(layers):
            if raw.ndim == 2:
                if raw.shape[0] != 2 * (len(prompt) + len(response)):
                    raise Q36MTRMechanicsError(
                        "Q36-MTR flattened router geometry differs"
                    )
                logits = raw.reshape(2, len(prompt) + len(response), raw.shape[-1])
            elif raw.ndim == 3 and tuple(raw.shape[:2]) == (
                2,
                len(prompt) + len(response),
            ):
                logits = raw
            else:
                raise Q36MTRMechanicsError("Q36-MTR router geometry differs")
            if router_top_k > logits.shape[-1]:
                raise Q36MTRMechanicsError("Q36-MTR router top-k differs")
            target = logits[:, len(prompt) - 1 : len(prompt) + len(response) - 1]
            indices = target.float().topk(router_top_k, dim=-1).indices
            route_digest.update(indices.to(torch.int16).cpu().numpy().tobytes())
            top1_changes = int((indices[0, :, 0] != indices[1, :, 0]).sum().cpu())
            assignment_changes = int((indices[0] != indices[1]).sum().cpu())
            layer_rows.append(
                {
                    "layer": layer_index,
                    "target_positions": len(response),
                    "experts": int(logits.shape[-1]),
                    "top1_changes": top1_changes,
                    "topk_assignment_changes": assignment_changes,
                    "router_max_abs_delta": float(
                        (target[0].float() - target[1].float()).abs().max().cpu()
                    ),
                }
            )
        return {
            "control": control,
            "layers": len(layer_rows),
            "top_k": router_top_k,
            "target_positions_per_layer": len(response),
            "top1_changes": sum(row["top1_changes"] for row in layer_rows),
            "topk_assignment_changes": sum(
                row["topk_assignment_changes"] for row in layer_rows
            ),
            "sensitive_layers": sum(
                row["topk_assignment_changes"] > 0 for row in layer_rows
            ),
            "router_max_abs_delta": max(
                row["router_max_abs_delta"] for row in layer_rows
            ),
            "route_path_sha256": route_digest.hexdigest(),
            "layer_receipts": layer_rows,
        }

    aligned_router = route_receipt(aligned_routes, "aligned")
    hidden_router = route_receipt(hidden_routes, "draft_hidden")
    # BF16 MoE kernels can change low-order accumulation when masked tokens
    # route differently, so admit only a tiny numerical residue while requiring
    # a materially larger aligned response.
    invariant_tolerance = 2e-3
    sensitivity_floor = 1e-2
    router_invariant_tolerance = 2e-3
    router_sensitivity_floor = 1e-2
    hidden_route_invariant = (
        hidden_router["topk_assignment_changes"] == 0
        and hidden_router["router_max_abs_delta"] <= router_invariant_tolerance
    )
    aligned_route_sensitive = (
        aligned_router["router_max_abs_delta"] >= router_sensitivity_floor
    )
    receipt = {
        "counterfactual_draft_tokens": sum(value == 0 for value in draft_mask),
        "token_count_exact": True,
        "position_geometry_exact": positions_exact,
        "aligned_response_max_abs_delta": aligned_delta,
        "draft_hidden_response_max_abs_delta": hidden_delta,
        "draft_hidden_invariant_tolerance": invariant_tolerance,
        "aligned_sensitivity_floor": sensitivity_floor,
        "draft_hidden_counterfactual_invariant": hidden_delta <= invariant_tolerance,
        "aligned_counterfactual_sensitive": aligned_delta >= sensitivity_floor,
        "native_router": {
            "aligned": aligned_router,
            "draft_hidden": hidden_router,
            "invariant_tolerance": router_invariant_tolerance,
            "sensitivity_floor": router_sensitivity_floor,
            "draft_hidden_route_invariant": hidden_route_invariant,
            "aligned_route_sensitive": aligned_route_sensitive,
            "aligned_expert_selection_changed": (
                aligned_router["topk_assignment_changes"] > 0
            ),
        },
    }
    if (
        not positions_exact
        or not receipt["draft_hidden_counterfactual_invariant"]
        or not receipt["aligned_counterfactual_sensitive"]
        or not hidden_route_invariant
        or not aligned_route_sensitive
    ):
        raise Q36MTRMechanicsError(
            f"Q36-MTR causal intervention failed: {json.dumps(receipt, sort_keys=True)}"
        )
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if (
        args.model_revision != MODEL_REVISION
        or args.model_config_sha256 != MODEL_CONFIG_SHA256
        or args.seed != SEED
        or args.data_seed != DATA_SEED
        or args.rows != ROWS
        or args.quantization != "nf4"
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise Q36MTRMechanicsError("Q36-MTR mechanics settings differ")
    if sha256_file(args.model_source_root / "config.json") != MODEL_CONFIG_SHA256:
        raise Q36MTRMechanicsError("Q36-MTR mechanics host differs")
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTRMechanicsError("Q36-MTR mechanics environment differs")
    environment = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        environment.get("schema") != "shohin-q36-mtr-environment-v1"
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTRMechanicsError("Q36-MTR mechanics environment contract differs")
    args.output.mkdir(parents=True)
    rows, data_sha256 = reservoir_rows_with_sha256(args.data, args.rows, args.data_seed)
    if len(rows) != ROWS or data_sha256 != B1_SHA256:
        raise Q36MTRMechanicsError("Q36-MTR mechanics source rows differ")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts, responses, masks, owner_geometry = tokenize_role_rows(
        tokenizer, rows, role="owner", max_sequence_length=1024
    )
    sample_source = rows[0]["question"]
    sample_draft = rows[0]["response"]
    revision_rows = [
        {
            "question": revision_prompt(sample_source, sample_draft),
            "response": rows[0]["response"],
        }
    ]
    aligned_tokens, aligned_responses, aligned_masks, aligned_geometry = (
        tokenize_role_rows(
            tokenizer, revision_rows, role="aligned", max_sequence_length=4096
        )
    )
    hidden_tokens, hidden_responses, hidden_masks, hidden_geometry = tokenize_role_rows(
        tokenizer, revision_rows, role="draft_hidden", max_sequence_length=4096
    )
    try:
        validate_matched_revision_geometry(aligned_geometry, hidden_geometry)
    except Q36MTRRoleError as error:
        raise Q36MTRMechanicsError(str(error)) from error
    if (
        aligned_tokens != hidden_tokens
        or aligned_responses != hidden_responses
        or aligned_masks != hidden_masks
    ):
        raise Q36MTRMechanicsError("Q36-MTR aligned/hidden token preimages differ")

    started = time.monotonic()
    backbone, loader = load_product_backbone(
        args.model_root,
        "causal",
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization="nf4",
    )
    text_config = getattr(backbone.config, "text_config", backbone.config)
    if int(text_config.hidden_size) != HIDDEN_SIZE:
        raise Q36MTRMechanicsError("Q36-MTR mechanics hidden size differs")
    config = SharedPostMLPConfig(
        hidden_size=HIDDEN_SIZE,
        controlled_layers=CONTROLLED_LAYERS,
        rank=RANK,
        alpha=ALPHA,
    )
    # device_map already placed the NF4 backbone. Moving this wrapper would
    # recursively invoke bitsandbytes' unsupported 4-bit `.to()` path.
    model = SharedPostMLPProductModel(backbone, config, draft_control="normal")
    controlled_indices = list(
        range(
            len(model.text_model.layers) - CONTROLLED_LAYERS,
            len(model.text_model.layers),
        )
    )
    trainable_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if (
        model.trainable_parameter_count() != TRAINABLE_PARAMETERS
        or any(
            not (name.endswith("adapter_a.weight") or name.endswith("adapter_b.weight"))
            for name in trainable_names
        )
        or any(
            parameter.dtype != torch.float32
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    ):
        raise Q36MTRMechanicsError("Q36-MTR mechanics trainable surface differs")
    rendered_generation_probe = _render_prompt(tokenizer, sample_source, True, False)
    generation_encoded = tokenizer(
        [rendered_generation_probe],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    generation_encoded = {
        key: value.to("cuda:0") for key, value in generation_encoded.items()
    }
    generation_prompt_width = int(generation_encoded["input_ids"].shape[1])
    generation_arguments = _generation_arguments("greedy", 1)
    generation_arguments["eos_token_id"] = _generation_stop_token_ids(tokenizer)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        generation_output = _generate_adapter(
            model,
            generation_encoded,
            generation_arguments,
            tokenizer.pad_token_id,
        )
    model.train(was_training)
    if generation_prompt_width <= 1 or tuple(generation_output.shape) != (1, 1):
        raise Q36MTRMechanicsError(
            "Q36-MTR adapter generation sequence semantics differ"
        )
    generation_sequence_receipt = {
        "contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
        "inputs_embeds_only": True,
        "input_ids_supplied_to_backbone_generate": False,
        "rendered_chat_tokenization": "add_special_tokens_false",
        "prompt_width": generation_prompt_width,
        "max_new_tokens": 1,
        "output_width": int(generation_output.shape[1]),
        "prompt_tokens_returned": 0,
        "generated_tokens_returned": int(generation_output.shape[1]),
    }
    before = protected_parameter_receipt(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-5,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs, attention, labels, charged = pack_training_embeddings(
            model.text_model.embed_tokens,
            [prompts[0]],
            [responses[0]],
            None,
            tokenizer.pad_token_id,
        )
        positions = full_sequence_position_ids(attention)
        output = model.text_model(
            inputs_embeds=inputs,
            attention_mask=attention,
            position_ids=positions,
            use_cache=False,
        )
        logits = model.lm_head(output.last_hidden_state)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
    if not torch.isfinite(loss):
        raise Q36MTRMechanicsError("Q36-MTR mechanics loss is nonfinite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
    )
    if not torch.isfinite(gradient_norm):
        raise Q36MTRMechanicsError("Q36-MTR mechanics gradient is nonfinite")
    optimizer.step()
    after = protected_parameter_receipt(model)
    if before != after:
        raise Q36MTRMechanicsError("Q36-MTR mechanics changed a protected parameter")
    trained_state = _trainable_state(model)
    trained_sha256 = _state_sha256(trained_state)
    metadata = {
        **role_contract("owner"),
        "shared_post_mlp_config": {
            "hidden_size": HIDDEN_SIZE,
            "controlled_layers": CONTROLLED_LAYERS,
            "rank": RANK,
            "alpha": ALPHA,
        },
        "draft_control": "normal",
        "model_root": str(args.model_source_root.resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_loader": loader,
        "data_sha256": data_sha256,
        "selected_rows": ROWS,
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "trainable_master_dtype": TRAINABLE_MASTER_DTYPE,
        "trainable_compute_dtype": "bfloat16",
        "controlled_layer_indices": controlled_indices,
        "source_only_model_visible": True,
        "internal_draft_visible": False,
    }
    checkpoint = args.output / "checkpoint_0000001.pt"
    _save_checkpoint(checkpoint, model, optimizer, 1, metadata)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    restored_update, restored_metadata = load_trainable_checkpoint(checkpoint, model)
    restored_state = _trainable_state(model)
    if (
        restored_update != 1
        or restored_metadata != metadata
        or _state_sha256(restored_state) != trained_sha256
        or any(
            not torch.equal(trained_state[name], restored_state[name])
            for name in trained_state
        )
    ):
        raise Q36MTRMechanicsError("Q36-MTR mechanics checkpoint restore differs")
    # The hidden arm uses identical token IDs and full-sequence positions; only
    # the informative draft keys are zeroed in its attention mask.
    aligned_attention = torch.ones(
        (1, len(aligned_tokens[0]) + len(aligned_responses[0])),
        dtype=torch.long,
        device="cuda:0",
    )
    hidden_attention = aligned_attention.clone()
    hidden_attention[0, : len(hidden_masks[0])] = torch.tensor(
        hidden_masks[0], device="cuda:0", dtype=torch.long
    )
    aligned_positions = full_sequence_position_ids(aligned_attention)
    hidden_positions = full_sequence_position_ids(hidden_attention)
    if not torch.equal(aligned_positions, hidden_positions):
        raise Q36MTRMechanicsError("Q36-MTR hidden intervention changed positions")
    causal_intervention = causal_draft_intervention_receipt(
        model,
        aligned_tokens[0],
        aligned_responses[0],
        hidden_masks[0],
        tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    report = {
        "schema": SCHEMA,
        "status": "pass",
        "capability_scored": False,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_loader": loader,
        "quantization": "nf4",
        "compute_dtype": "bfloat16",
        "rows": ROWS,
        "data_sha256": data_sha256,
        "controlled_layer_indices": controlled_indices,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "protected_router_expert_trainables": 0,
        "protected_parameter_receipt_before": before,
        "protected_parameter_receipt_after": after,
        "one_finite_update": True,
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "charged_tokens": int(charged),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "trained_state_sha256": trained_sha256,
        "serialization_restore_exact": True,
        "owner_sequence_geometry": owner_geometry,
        "aligned_sequence_geometry": aligned_geometry,
        "draft_hidden_sequence_geometry": hidden_geometry,
        "aligned_hidden_token_geometry_exact": True,
        "aligned_hidden_position_geometry_exact": True,
        "draft_hidden_mask_nonempty": hidden_geometry["draft_masked_tokens"] > 0,
        "draft_tokens_deleted": 0,
        "causal_draft_intervention": causal_intervention,
        "generation_sequence_receipt": generation_sequence_receipt,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-config-sha256", default=MODEL_CONFIG_SHA256)
    parser.add_argument("--quantization", default="nf4")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-seed", type=int, default=DATA_SEED)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "peak_gpu_memory_bytes": report["peak_gpu_memory_bytes"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
