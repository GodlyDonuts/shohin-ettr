"""Matched LoRA and integrated-workspace SFT for product reasoning backbones."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from integrated_reasoning_workspace import (
    DenseReasoningWorkspace,
    IntegratedReasoningWorkspace,
    IntegratedWorkspaceConfig,
    dense_workspace_architecture_sha256,
    residual_workspace_architecture_sha256,
    workspace_architecture_sha256,
)


class ProductReasoningTrainError(RuntimeError):
    """The matched product-reasoning training contract was violated."""


PRODUCT_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Give concise, verifiable reasoning "
    "and a clearly marked final answer."
)


def render_reasoning_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> str:
    """Render native chat when available, otherwise use one stable text envelope."""

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    rendered = "\n\n".join(
        f"{message['role'].capitalize()}: {message['content']}" for message in messages
    )
    return f"{rendered}\n\nAssistant:"


class LoRALinear(nn.Module):
    """A frozen linear projection plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ProductReasoningTrainError("LoRA rank and alpha must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(inputs)) * self.scale


def install_lora(module: nn.Module, rank: int, alpha: float) -> int:
    """Replace every descendant linear projection and return the count."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            replaced += 1
        else:
            replaced += install_lora(child, rank, alpha)
    return replaced


def resolve_product_backbone_layout(backbone: nn.Module) -> tuple[nn.Module, nn.Module, int, str]:
    """Expose a common decoder-only text path for Qwen multimodal and causal LMs."""

    model = getattr(backbone, "model", None)
    if model is None:
        raise ProductReasoningTrainError("backbone exposes no decoder model")
    if hasattr(model, "language_model"):
        text_model = model.language_model
        layout = "multimodal-language-model"
    else:
        text_model = model
        layout = "causal-language-model"
    lm_head = getattr(backbone, "lm_head", None)
    if lm_head is None:
        raise ProductReasoningTrainError("backbone exposes no language-model head")
    if not hasattr(text_model, "layers") or not hasattr(text_model, "embed_tokens"):
        raise ProductReasoningTrainError("backbone text path differs")
    text_config = getattr(backbone.config, "text_config", backbone.config)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ProductReasoningTrainError("backbone text width is unavailable")
    return text_model, lm_head, int(hidden_size), layout


def resolve_product_model_loader(model_root: Path, requested: str) -> str:
    """Select the exact HF auto class without guessing from a failed load."""

    if requested not in {"auto", "causal", "multimodal"}:
        raise ProductReasoningTrainError("model loader differs")
    if requested != "auto":
        return requested
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_root, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", ""))
    if hasattr(config, "vision_config") or model_type.startswith("qwen3_5"):
        return "multimodal"
    return "causal"


def load_product_backbone(
    model_root: Path,
    requested_loader: str,
    *,
    dtype: torch.dtype,
    device_map: dict[str, int],
) -> tuple[nn.Module, str]:
    """Load either a multimodal wrapper or an ordinary causal LM."""

    loader = resolve_product_model_loader(model_root, requested_loader)
    if loader == "multimodal":
        from transformers import AutoModelForMultimodalLM

        auto_class = AutoModelForMultimodalLM
    else:
        from transformers import AutoModelForCausalLM

        auto_class = AutoModelForCausalLM
    backbone = auto_class.from_pretrained(
        model_root,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    resolve_product_backbone_layout(backbone)
    return backbone, loader


def _question_response(row: dict[str, Any]) -> tuple[str, str] | None:
    question = row.get("question") or row.get("problem") or row.get("prompt")
    response = (
        row.get("response")
        or row.get("solution")
        or row.get("completion")
        or row.get("answer")
    )
    if not question or not response:
        return None
    return str(question), str(response)


def reservoir_rows_with_sha256(
    path: Path,
    limit: int,
    seed: int,
) -> tuple[list[dict[str, str]], str]:
    """Hash and select a deterministic bounded population in one exact pass."""

    if limit <= 0:
        raise ProductReasoningTrainError("row limit must be positive")
    generator = random.Random(seed)
    digest = hashlib.sha256()
    selected: list[dict[str, str]] = []
    valid = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            pair = _question_response(row)
            if pair is None:
                continue
            normalized = {"question": pair[0], "response": pair[1]}
            valid += 1
            if len(selected) < limit:
                selected.append(normalized)
            else:
                position = generator.randrange(valid)
                if position < limit:
                    selected[position] = normalized
    if not selected:
        raise ProductReasoningTrainError("training source has no valid rows")
    generator.shuffle(selected)
    return selected, digest.hexdigest()


def reservoir_rows(path: Path, limit: int, seed: int) -> list[dict[str, str]]:
    """Compatibility wrapper returning only the selected deterministic rows."""

    return reservoir_rows_with_sha256(path, limit, seed)[0]


def _pad_token_rows(
    rows: list[list[int]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row)] = 1
    return ids, mask


def pack_training_embeddings(
    embedding: nn.Module,
    prompt_rows: list[list[int]],
    response_rows: list[list[int]],
    prefix_states: torch.Tensor | None,
    pad_token_id: int,
    prompt_residuals: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack prompt, optional workspace, and target with causal-LM labels."""

    if len(prompt_rows) != len(response_rows) or not prompt_rows:
        raise ProductReasoningTrainError("prompt/response batch differs")
    prefix_slots = 0 if prefix_states is None else int(prefix_states.shape[1])
    if prefix_states is not None and prompt_residuals is not None:
        raise ProductReasoningTrainError("workspace prefix and residual are mutually exclusive")
    if prefix_states is not None and prefix_states.shape[0] != len(prompt_rows):
        raise ProductReasoningTrainError("workspace batch differs")
    if prompt_residuals is not None and prompt_residuals.shape[0] != len(prompt_rows):
        raise ProductReasoningTrainError("workspace residual batch differs")
    sequences: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    charged_tokens = 0
    device = next(embedding.parameters()).device
    for index, (prompt, response) in enumerate(
        zip(prompt_rows, response_rows, strict=True)
    ):
        prompt_tensor = torch.tensor(prompt, device=device, dtype=torch.long)
        response_tensor = torch.tensor(response, device=device, dtype=torch.long)
        prompt_embeddings = embedding(prompt_tensor)
        if prompt_residuals is not None:
            residual_count = min(len(prompt), int(prompt_residuals.shape[1]))
            prompt_embeddings = prompt_embeddings.clone()
            prompt_embeddings[-residual_count:] = (
                prompt_embeddings[-residual_count:]
                + prompt_residuals[index, :residual_count]
            )
        parts = [prompt_embeddings]
        if prefix_states is not None:
            parts.append(prefix_states[index])
        parts.append(embedding(response_tensor))
        sequences.append(torch.cat(parts, dim=0))
        row_labels = torch.full(
            (len(prompt) + prefix_slots + len(response),),
            -100,
            device=device,
            dtype=torch.long,
        )
        row_labels[len(prompt) + prefix_slots :] = response_tensor
        labels.append(row_labels)
        charged_tokens += len(response)

    width = max(sequence.shape[0] for sequence in sequences)
    hidden = sequences[0].shape[-1]
    dtype = sequences[0].dtype
    packed = torch.zeros(len(sequences), width, hidden, device=device, dtype=dtype)
    attention = torch.zeros(len(sequences), width, device=device, dtype=torch.long)
    packed_labels = torch.full(
        (len(sequences), width), -100, device=device, dtype=torch.long
    )
    pad_embedding = embedding(
        torch.tensor([pad_token_id], device=device, dtype=torch.long)
    )[0]
    packed[:] = pad_embedding
    for index, (sequence, row_labels) in enumerate(
        zip(sequences, labels, strict=True)
    ):
        length = sequence.shape[0]
        packed[index, :length] = sequence
        attention[index, :length] = 1
        packed_labels[index, :length] = row_labels
    return packed, attention, packed_labels, charged_tokens


class ProductReasoningModel(nn.Module):
    """Text-only Qwen path with matched LoRA and optional recurrent workspace."""

    def __init__(
        self,
        backbone: nn.Module,
        arm: str,
        lora_layers: int,
        lora_rank: int,
        lora_alpha: float,
        workspace_width: int,
        workspace_slots: int,
        recurrent_steps: int,
        dense_width: int = 192,
    ) -> None:
        super().__init__()
        if arm not in {
            "baseline",
            "ettr",
            "dense",
            "ettr_residual",
            "dense_residual",
        }:
            raise ProductReasoningTrainError("training arm differs")
        self.backbone = backbone
        self.arm = arm
        self.backbone.requires_grad_(False)
        (
            self.text_model,
            self.lm_head,
            hidden_size,
            self.backbone_layout,
        ) = resolve_product_backbone_layout(self.backbone)
        layers = self.text_model.layers
        if not 0 < lora_layers <= len(layers):
            raise ProductReasoningTrainError("LoRA layer count differs")
        self.lora_projection_count = 0
        for layer in layers[-lora_layers:]:
            self.lora_projection_count += install_lora(layer, lora_rank, lora_alpha)
        if self.lora_projection_count == 0:
            raise ProductReasoningTrainError("no text projections received LoRA")

        self.workspace_config: IntegratedWorkspaceConfig | None = None
        self.workspace: IntegratedReasoningWorkspace | DenseReasoningWorkspace | None = None
        self.workspace_injection = (
            "prompt_residual" if arm.endswith("_residual") else "soft_prefix"
        )
        self.residual_gate: nn.Parameter | None = None
        if arm != "baseline":
            effective_width = workspace_width if arm.startswith("ettr") else dense_width
            self.workspace_config = IntegratedWorkspaceConfig(
                backbone_width=hidden_size,
                workspace_width=effective_width,
                workspace_slots=workspace_slots,
                recurrent_steps=recurrent_steps,
                attention_heads=8,
                ff_multiplier=4,
            )
            if arm.startswith("ettr"):
                self.workspace = IntegratedReasoningWorkspace(self.workspace_config)
            else:
                self.workspace = DenseReasoningWorkspace(self.workspace_config)
            if self.workspace_injection == "prompt_residual":
                self.residual_gate = nn.Parameter(torch.tensor(-4.0))

    def sequence_workspace_slots(self) -> int:
        if self.workspace_config is None or self.workspace_injection != "soft_prefix":
            return 0
        return self.workspace_config.workspace_slots

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        response_rows: list[list[int]],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        embedding = self.text_model.embed_tokens
        prefix = None
        prompt_residuals = None
        halting_loss = torch.zeros((), device=embedding.weight.device)
        stop_probability = 0.0
        delta_norm = 0.0
        if self.workspace is not None:
            prompt_ids, prompt_mask = _pad_token_rows(prompt_rows, pad_token_id)
            prompt_ids = prompt_ids.to(embedding.weight.device)
            prompt_mask = prompt_mask.to(embedding.weight.device)
            if self.workspace_injection == "prompt_residual":
                prompt_features = embedding(prompt_ids)
            else:
                with torch.no_grad():
                    prompt_features = self.text_model(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        use_cache=False,
                    ).last_hidden_state
            workspace_output = self.workspace(prompt_features, prompt_mask)
            workspace_states = workspace_output.prefix_states.to(
                dtype=embedding.weight.dtype
            )
            if self.workspace_injection == "prompt_residual":
                assert self.residual_gate is not None
                prompt_residuals = workspace_states * self.residual_gate.sigmoid()
            else:
                prefix = workspace_states
            halting_loss = self.workspace.halting_regularizer(workspace_output)
            stop_probability = float(
                workspace_output.stop_logits[:, -1].sigmoid().detach().mean()
            )
            delta_norm = float(workspace_output.step_deltas.detach().mean())

        inputs, attention, labels, charged = pack_training_embeddings(
            embedding,
            prompt_rows,
            response_rows,
            prefix,
            pad_token_id,
            prompt_residuals,
        )
        outputs = self.text_model(
            inputs_embeds=inputs,
            attention_mask=attention,
            use_cache=False,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        language_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss = language_loss + 0.01 * halting_loss
        return loss, {
            "language_loss": float(language_loss.detach()),
            "halting_loss": float(halting_loss.detach()),
            "final_stop_probability": stop_probability,
            "mean_step_delta": delta_norm,
            "residual_gate": (
                float(self.residual_gate.sigmoid().detach())
                if self.residual_gate is not None
                else 0.0
            ),
            "charged_tokens": float(charged),
        }


def _tokenize_rows(
    tokenizer: Any,
    rows: list[dict[str, str]],
    max_sequence_length: int,
    workspace_slots: int,
) -> tuple[list[list[int]], list[list[int]]]:
    prompt_rows: list[list[int]] = []
    response_rows: list[list[int]] = []
    response_budget_floor = min(256, max_sequence_length // 2)
    for row in rows:
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {
                    "role": "system",
                    "content": PRODUCT_SYSTEM_PROMPT,
                },
                {"role": "user", "content": row["question"]},
            ],
            enable_thinking=False,
        )
        prompt = tokenizer.encode(rendered, add_special_tokens=False)
        response = tokenizer.encode(row["response"], add_special_tokens=False)
        target_budget = max_sequence_length - workspace_slots
        if len(response) > target_budget - 9:
            response = response[: target_budget - 9]
        prompt_budget = target_budget - len(response)
        if prompt_budget < 8:
            response = response[:response_budget_floor]
            prompt_budget = target_budget - len(response)
        prompt = prompt[-prompt_budget:]
        response.append(tokenizer.eos_token_id)
        if prompt and response:
            prompt_rows.append(prompt)
            response_rows.append(response)
    if not prompt_rows:
        raise ProductReasoningTrainError("tokenization removed every row")
    return prompt_rows, response_rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _save_checkpoint(
    path: Path,
    model: ProductReasoningModel,
    optimizer: torch.optim.Optimizer,
    update: int,
    metadata: dict[str, Any],
) -> None:
    trainable = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "shohin-hf-product-reasoning-checkpoint-v1",
            "update": update,
            "trainable_state": trainable,
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_trainable_checkpoint(
    path: Path,
    model: ProductReasoningModel,
) -> tuple[int, dict[str, Any]]:
    """Restore only the explicitly trainable product-reasoning parameters."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-hf-product-reasoning-checkpoint-v1":
        raise ProductReasoningTrainError("product checkpoint schema differs")
    metadata = payload.get("metadata")
    saved = payload.get("trainable_state")
    if not isinstance(metadata, dict) or not isinstance(saved, dict):
        raise ProductReasoningTrainError("product checkpoint payload is incomplete")
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(saved) != set(current):
        missing = sorted(set(current) - set(saved))
        unexpected = sorted(set(saved) - set(current))
        raise ProductReasoningTrainError(
            f"product checkpoint parameter contract differs: "
            f"missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            tensor = saved[name]
            if tensor.shape != parameter.shape:
                raise ProductReasoningTrainError(
                    f"product checkpoint tensor shape differs: {name}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return int(payload["update"]), metadata


def product_generation_embeddings(
    model: ProductReasoningModel,
    prompt_ids: torch.Tensor,
    prompt_attention: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact prompt plus learned-workspace prefix used at inference."""

    if prompt_ids.ndim != 2 or prompt_attention.shape != prompt_ids.shape:
        raise ProductReasoningTrainError("generation prompt geometry differs")
    embedding = model.text_model.embed_tokens
    prompt_ids = prompt_ids.to(embedding.weight.device)
    prompt_attention = prompt_attention.to(embedding.weight.device)
    prompt_embeddings = embedding(prompt_ids)
    if model.workspace is None:
        return prompt_embeddings, prompt_attention
    if model.workspace_injection == "prompt_residual":
        prompt_features = prompt_embeddings
    else:
        prompt_features = model.text_model(
            input_ids=prompt_ids,
            attention_mask=prompt_attention,
            use_cache=False,
        ).last_hidden_state
    workspace_output = model.workspace(prompt_features, prompt_attention)
    prefix = workspace_output.prefix_states.to(dtype=prompt_embeddings.dtype)
    if model.workspace_injection == "prompt_residual":
        assert model.residual_gate is not None
        residual_count = min(prompt_embeddings.shape[1], prefix.shape[1])
        prompt_embeddings = prompt_embeddings.clone()
        prompt_embeddings[:, -residual_count:] = (
            prompt_embeddings[:, -residual_count:]
            + prefix[:, :residual_count] * model.residual_gate.sigmoid()
        )
        return prompt_embeddings, prompt_attention
    prefix_attention = torch.ones(
        prefix.shape[:2], device=prompt_attention.device, dtype=prompt_attention.dtype
    )
    return (
        torch.cat((prompt_embeddings, prefix), dim=1),
        torch.cat((prompt_attention, prefix_attention), dim=1),
    )


def _batches(rows: list[dict[str, str]], batch_size: int) -> Iterable[list[dict[str, str]]]:
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        if len(batch) == batch_size:
            yield batch


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise ProductReasoningTrainError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone, resolved_model_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = ProductReasoningModel(
        backbone,
        args.arm,
        args.lora_layers,
        args.lora_rank,
        args.lora_alpha,
        args.workspace_width,
        args.workspace_slots,
        args.recurrent_steps,
        args.dense_width,
    ).to("cuda:0")
    model.train()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )

    rows, data_hash = reservoir_rows_with_sha256(
        args.data, args.max_rows, args.data_seed
    )
    batch_stream = list(_batches(rows, args.batch_size))
    if not batch_stream:
        raise ProductReasoningTrainError("training population is smaller than a batch")
    metadata = {
        "arm": args.arm,
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_model_loader,
        "backbone_layout": model.backbone_layout,
        "data": str(args.data.resolve()),
        "data_sha256": data_hash,
        "selected_rows": len(rows),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "lora_layers": args.lora_layers,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_projection_count": model.lora_projection_count,
        "trainable_parameters": model.trainable_parameter_count(),
        "workspace_config": (
            asdict(model.workspace_config) if model.workspace_config else None
        ),
        "workspace_architecture_sha256": None,
    }
    if model.workspace_config is not None:
        if args.arm.endswith("_residual"):
            metadata["workspace_architecture_sha256"] = (
                residual_workspace_architecture_sha256(
                    model.workspace_config,
                    dense=args.arm.startswith("dense"),
                )
            )
        elif args.arm == "ettr":
            metadata["workspace_architecture_sha256"] = workspace_architecture_sha256(
                model.workspace_config
            )
        else:
            metadata["workspace_architecture_sha256"] = (
                dense_workspace_architecture_sha256(model.workspace_config)
            )
        metadata["workspace_injection"] = model.workspace_injection

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    total_charged = 0
    trace: list[dict[str, float | int]] = []
    update = 0
    microstep = 0
    while update < args.updates:
        raw_batch = batch_stream[microstep % len(batch_stream)]
        workspace_slots = model.sequence_workspace_slots()
        prompt_rows, response_rows = _tokenize_rows(
            tokenizer,
            raw_batch,
            args.max_sequence_length,
            workspace_slots,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                prompt_rows,
                response_rows,
                tokenizer.pad_token_id,
            )
            scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        total_charged += int(metrics["charged_tokens"])
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue

        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        progress = update / max(args.updates - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % args.log_interval == 0:
            elapsed = time.monotonic() - started
            event: dict[str, float | int] = {
                "update": update,
                "loss": float(loss.detach()),
                "language_loss": metrics["language_loss"],
                "halting_loss": metrics["halting_loss"],
                "final_stop_probability": metrics["final_stop_probability"],
                "mean_step_delta": metrics["mean_step_delta"],
                "residual_gate": metrics["residual_gate"],
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "charged_tokens": total_charged,
                "charged_tokens_per_second": total_charged / elapsed,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
        if update % args.checkpoint_interval == 0 or update == args.updates:
            _save_checkpoint(
                args.output / f"checkpoint_{update:07d}.pt",
                model,
                optimizer,
                update,
                metadata,
            )

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    report = {
        "schema": "shohin-hf-product-reasoning-training-v1",
        "status": "complete",
        **metadata,
        "updates": update,
        "gradient_accumulation": args.gradient_accumulation,
        "batch_size": args.batch_size,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "charged_tokens": total_charged,
        "elapsed_seconds": elapsed,
        "charged_tokens_per_second": total_charged / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "trace": trace,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--model-loader",
        choices=("auto", "causal", "multimodal"),
        default="auto",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("baseline", "ettr", "dense", "ettr_residual", "dense_residual"),
        required=True,
    )
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--workspace-width", type=int, default=512)
    parser.add_argument("--dense-width", type=int, default=192)
    parser.add_argument("--workspace-slots", type=int, default=16)
    parser.add_argument("--recurrent-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--data-seed", type=int, default=20260802)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.batch_size,
        args.gradient_accumulation,
        args.max_rows,
        args.max_sequence_length,
        args.lora_layers,
        args.lora_rank,
        args.workspace_width,
        args.dense_width,
        args.workspace_slots,
        args.recurrent_steps,
        args.log_interval,
        args.checkpoint_interval,
    )
    if any(value <= 0 for value in positive) or args.learning_rate <= 0:
        parser.error("training dimensions and learning rate must be positive")
    reserved_slots = args.workspace_slots if args.arm in {"ettr", "dense"} else 0
    if args.max_sequence_length <= reserved_slots + 16:
        parser.error("maximum sequence length leaves no prompt/target budget")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    print(
        f"[product-train] arm={args.arm} updates={report['updates']} "
        f"tokens/s={report['charged_tokens_per_second']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
