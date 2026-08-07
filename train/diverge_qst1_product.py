"""Stage-owned transaction workspace for Qwen product-reasoning experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_product_reasoning_train import (
    ProductReasoningTrainError,
    _pad_token_rows,
    install_lora,
    pack_training_embeddings,
    resolve_product_backbone_layout,
)


@dataclass(frozen=True)
class QST1Config:
    """Geometry of the source-sealed, single-lineage transaction workspace."""

    backbone_width: int
    workspace_width: int = 384
    source_slots: int = 8
    state_slots: int = 8
    query_slots: int = 4
    recurrent_steps: int = 8
    attention_heads: int = 8
    ff_multiplier: int = 2
    binding_temperature: float = 0.10

    def validate(self) -> None:
        dimensions = (
            self.backbone_width,
            self.workspace_width,
            self.source_slots,
            self.state_slots,
            self.query_slots,
            self.recurrent_steps,
            self.attention_heads,
            self.ff_multiplier,
        )
        if any(value <= 0 for value in dimensions):
            raise ProductReasoningTrainError("QST1 dimensions must be positive")
        if self.workspace_width % self.attention_heads:
            raise ProductReasoningTrainError(
                "QST1 workspace width must divide attention heads"
            )
        if self.binding_temperature <= 0.0:
            raise ProductReasoningTrainError(
                "QST1 binding temperature must be positive"
            )

    @property
    def prefix_slots(self) -> int:
        return self.source_slots + self.state_slots + self.query_slots


@dataclass
class QST1Output:
    """Complete stage-owned transaction trace for one prompt batch."""

    prefix_states: torch.Tensor
    source_packet: torch.Tensor
    initial_state: torch.Tensor
    final_state: torch.Tensor
    query_state: torch.Tensor
    cumulative_halt: torch.Tensor
    update_norms: torch.Tensor
    write_gates: torch.Tensor


def qst1_architecture_sha256(config: QST1Config) -> str:
    """Bind checkpoints to the exact QST1 mechanism and geometry."""

    payload = {
        "architecture": "diverge-qst1-stage-owned-transaction-v1",
        "config": asdict(config),
        "invariants": [
            "immutable_source_packet",
            "separate_tied_recurrent_state",
            "copy_on_write_slot_gates",
            "monotone_adaptive_stop",
            "late_query_owner",
            "single_coherent_lineage",
            "batch_contrastive_provenance",
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class _ResidualFFN(nn.Module):
    def __init__(self, width: int, multiplier: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, width * multiplier),
            nn.SiLU(),
            nn.Linear(width * multiplier, width),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return states + self.net(self.norm(states))


class StageOwnedTransactionWorkspace(nn.Module):
    """Compile once, evolve separate state, and read one coherent lineage."""

    def __init__(self, config: QST1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.workspace_width
        heads = config.attention_heads

        self.prompt_projection = nn.Linear(config.backbone_width, width)
        self.prompt_norm = nn.LayerNorm(width)

        self.source_queries = nn.Parameter(torch.empty(config.source_slots, width))
        self.source_identities = nn.Parameter(torch.empty(config.source_slots, width))
        self.source_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.source_norm = nn.LayerNorm(width)
        self.source_ffn = _ResidualFFN(width, config.ff_multiplier)

        self.state_seed = nn.Parameter(torch.empty(config.state_slots, width))
        self.state_identities = nn.Parameter(torch.empty(config.state_slots, width))
        self.state_source_attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.state_self_attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.state_norm = nn.LayerNorm(width)
        self.state_ffn = _ResidualFFN(width, config.ff_multiplier)
        self.write_gate = nn.Linear(width * 3, width)
        self.stop_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )

        self.query_seed = nn.Parameter(torch.empty(config.query_slots, width))
        self.query_identities = nn.Parameter(torch.empty(config.query_slots, width))
        self.query_state_attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.query_source_attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.query_norm = nn.LayerNorm(width)
        self.query_ffn = _ResidualFFN(width, config.ff_multiplier)

        self.stage_identities = nn.Parameter(torch.empty(3, width))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, config.backbone_width)
        self.binding_source = nn.Linear(width, width, bias=False)
        self.binding_state = nn.Linear(width, width, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.source_queries,
            self.source_identities,
            self.state_seed,
            self.state_identities,
            self.query_seed,
            self.query_identities,
            self.stage_identities,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        nn.init.constant_(self.stop_head[-1].bias, -2.0)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _attend(
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return attended

    def _compile_source(
        self,
        prompt_features: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = prompt_features.shape[0]
        prompt = self.prompt_norm(self.prompt_projection(prompt_features))
        queries = (
            (self.source_queries + self.source_identities)
            .unsqueeze(0)
            .expand(batch, -1, -1)
        )
        source = queries + self._attend(
            self.source_attention,
            self.source_norm(queries),
            prompt,
            key_padding_mask=~prompt_mask.bool(),
        )
        return self.source_ffn(source)

    def _initial_state(self, source_packet: torch.Tensor) -> torch.Tensor:
        batch = source_packet.shape[0]
        state = (
            (self.state_seed + self.state_identities).unsqueeze(0).expand(batch, -1, -1)
        )
        state = state + self._attend(
            self.state_source_attention,
            self.state_norm(state),
            source_packet,
        )
        return self.state_ffn(state)

    def _transition(
        self,
        source_packet: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = initial_state
        alive = torch.ones(state.shape[0], 1, 1, device=state.device, dtype=state.dtype)
        cumulative_halt: list[torch.Tensor] = []
        update_norms: list[torch.Tensor] = []
        write_gates: list[torch.Tensor] = []
        for _ in range(self.config.recurrent_steps):
            normalized = self.state_norm(state)
            self_context = self._attend(
                self.state_self_attention, normalized, normalized
            )
            source_context = self._attend(
                self.state_source_attention, normalized, source_packet
            )
            proposal = self.state_ffn(state + self_context + source_context)
            gate = torch.sigmoid(
                self.write_gate(torch.cat((state, proposal, source_context), dim=-1))
            )
            update = gate * (proposal - state)
            next_state = state + alive * update
            stop = torch.sigmoid(self.stop_head(next_state.mean(dim=1))).unsqueeze(-1)
            alive = alive * (1.0 - stop)
            state = next_state
            cumulative_halt.append(1.0 - alive.squeeze(-1).squeeze(-1))
            update_norms.append(update.float().pow(2).mean(dim=(1, 2)).sqrt())
            write_gates.append(gate.float().mean(dim=(1, 2)))
        return (
            state,
            torch.stack(cumulative_halt, dim=1),
            torch.stack(update_norms, dim=1),
            torch.stack(write_gates, dim=1),
        )

    def _read_query(
        self,
        source_packet: torch.Tensor,
        final_state: torch.Tensor,
    ) -> torch.Tensor:
        batch = source_packet.shape[0]
        query = (
            (self.query_seed + self.query_identities).unsqueeze(0).expand(batch, -1, -1)
        )
        query = query + self._attend(
            self.query_state_attention,
            self.query_norm(query),
            final_state,
        )
        query = query + self._attend(
            self.query_source_attention,
            self.query_norm(query),
            source_packet,
        )
        return self.query_ffn(query)

    def forward(
        self,
        prompt_features: torch.Tensor,
        prompt_mask: torch.Tensor,
        *,
        control: str = "normal",
    ) -> QST1Output:
        if prompt_features.ndim != 3 or prompt_mask.shape != prompt_features.shape[:2]:
            raise ProductReasoningTrainError("QST1 prompt geometry differs")
        if control not in {"normal", "packet_swap", "state_reset"}:
            raise ProductReasoningTrainError("QST1 control differs")
        source_packet = self._compile_source(prompt_features, prompt_mask)
        if control == "packet_swap":
            if source_packet.shape[0] > 1:
                source_packet = source_packet.roll(1, dims=0)
            else:
                source_packet = source_packet.roll(1, dims=1)
        initial_state = self._initial_state(source_packet)
        if control == "state_reset":
            final_state = initial_state
            shape = (source_packet.shape[0], self.config.recurrent_steps)
            cumulative_halt = torch.zeros(
                shape, device=source_packet.device, dtype=source_packet.dtype
            )
            update_norms = torch.zeros_like(cumulative_halt)
            write_gates = torch.zeros_like(cumulative_halt)
        else:
            final_state, cumulative_halt, update_norms, write_gates = self._transition(
                source_packet, initial_state
            )
        query_state = self._read_query(source_packet, final_state)
        stage_states = torch.cat(
            (
                source_packet + self.stage_identities[0],
                final_state + self.stage_identities[1],
                query_state + self.stage_identities[2],
            ),
            dim=1,
        )
        prefix = self.output_projection(self.output_norm(stage_states))
        return QST1Output(
            prefix_states=prefix,
            source_packet=source_packet,
            initial_state=initial_state,
            final_state=final_state,
            query_state=query_state,
            cumulative_halt=cumulative_halt,
            update_norms=update_norms,
            write_gates=write_gates,
        )

    def binding_vectors(
        self,
        source_packet: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = F.normalize(
            self.binding_source(source_packet.mean(dim=1)).float(), dim=-1
        )
        state_vector = F.normalize(
            self.binding_state(state.mean(dim=1)).float(), dim=-1
        )
        return source, state_vector


class QST1ProductModel(nn.Module):
    """Pinned Qwen plus LoRA and a stage-owned transaction workspace."""

    architecture = "diverge-qst1"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        lora_layers: int,
        lora_rank: int,
        lora_alpha: float,
        workspace_width: int,
        source_slots: int,
        state_slots: int,
        query_slots: int,
        recurrent_steps: int,
        attention_heads: int,
        ff_multiplier: int,
        binding_temperature: float,
        binding_weight: float,
        reset_weight: float,
        halting_weight: float,
    ) -> None:
        super().__init__()
        if min(binding_weight, reset_weight, halting_weight) < 0.0:
            raise ProductReasoningTrainError("QST1 loss weights must be nonnegative")
        self.backbone = backbone
        self.arm = "diverge_qst1"
        self.backbone.requires_grad_(False)
        (
            self.text_model,
            self.lm_head,
            hidden_size,
            self.backbone_layout,
        ) = resolve_product_backbone_layout(backbone)
        layers = self.text_model.layers
        if not 0 < lora_layers <= len(layers):
            raise ProductReasoningTrainError("LoRA layer count differs")
        self.lora_projection_count = 0
        for layer in layers[-lora_layers:]:
            self.lora_projection_count += install_lora(layer, lora_rank, lora_alpha)
        if self.lora_projection_count == 0:
            raise ProductReasoningTrainError("no text projections received LoRA")

        self.workspace_config = QST1Config(
            backbone_width=hidden_size,
            workspace_width=workspace_width,
            source_slots=source_slots,
            state_slots=state_slots,
            query_slots=query_slots,
            recurrent_steps=recurrent_steps,
            attention_heads=attention_heads,
            ff_multiplier=ff_multiplier,
            binding_temperature=binding_temperature,
        )
        self.workspace_config.validate()
        self.workspace = StageOwnedTransactionWorkspace(self.workspace_config)
        self.binding_weight = binding_weight
        self.reset_weight = reset_weight
        self.halting_weight = halting_weight
        self.control = "normal"

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def sequence_workspace_slots(self) -> int:
        return self.workspace_config.prefix_slots

    def set_control(self, control: str) -> None:
        if control not in {"normal", "packet_swap", "state_reset"}:
            raise ProductReasoningTrainError("QST1 control differs")
        self.control = control

    def _workspace_output(
        self,
        prompt_rows: list[list[int]],
        pad_token_id: int,
        *,
        control: str = "normal",
    ) -> QST1Output:
        embedding = self.text_model.embed_tokens
        prompt_ids, prompt_mask = _pad_token_rows(prompt_rows, pad_token_id)
        prompt_ids = prompt_ids.to(embedding.weight.device)
        prompt_mask = prompt_mask.to(embedding.weight.device)
        # The qualified host supplies contextual observations, while trainable
        # LoRA is optimized only on the causal answer path below.
        with torch.no_grad():
            prompt_features = self.text_model(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                use_cache=False,
            ).last_hidden_state
        return self.workspace(prompt_features, prompt_mask, control=control)

    def _binding_losses(self, output: QST1Output) -> tuple[torch.Tensor, torch.Tensor]:
        source, final_state = self.workspace.binding_vectors(
            output.source_packet, output.final_state
        )
        _, initial_state = self.workspace.binding_vectors(
            output.source_packet, output.initial_state
        )
        if source.shape[0] > 1:
            logits = torch.matmul(final_state, source.transpose(0, 1))
            logits = logits / self.workspace_config.binding_temperature
            labels = torch.arange(source.shape[0], device=source.device)
            binding_loss = 0.5 * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.transpose(0, 1), labels)
            )
        else:
            corrupted = output.source_packet.roll(1, dims=1)
            corrupt_source, _ = self.workspace.binding_vectors(
                corrupted, output.final_state
            )
            clean_score = (source * final_state).sum(dim=-1)
            corrupt_score = (corrupt_source * final_state).sum(dim=-1)
            binding_loss = F.softplus(0.20 - clean_score + corrupt_score).mean()
        clean_score = (source * final_state).sum(dim=-1)
        reset_score = (source * initial_state).sum(dim=-1)
        reset_margin_loss = F.relu(0.10 - clean_score + reset_score).mean()
        return binding_loss, reset_margin_loss

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        response_rows: list[list[int]],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if len(prompt_rows) != len(response_rows) or not prompt_rows:
            raise ProductReasoningTrainError("QST1 batch geometry differs")
        embedding = self.text_model.embed_tokens
        workspace_output = self._workspace_output(prompt_rows, pad_token_id)
        prefix = workspace_output.prefix_states.to(dtype=embedding.weight.dtype)
        inputs, attention, labels, charged = pack_training_embeddings(
            embedding,
            prompt_rows,
            response_rows,
            prefix,
            pad_token_id,
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
        binding_loss, reset_loss = self._binding_losses(workspace_output)
        final_halt = workspace_output.cumulative_halt[:, -1]
        premature_halt = workspace_output.cumulative_halt[:, :-1].mean()
        halting_loss = (1.0 - final_halt).mean() + 0.10 * premature_halt
        loss = (
            language_loss
            + self.binding_weight * binding_loss
            + self.reset_weight * reset_loss
            + self.halting_weight * halting_loss
        )
        return loss, {
            "language_loss": float(language_loss.detach()),
            "binding_loss": float(binding_loss.detach()),
            "reset_loss": float(reset_loss.detach()),
            "halting_loss": float(halting_loss.detach()),
            "final_halt_probability": float(final_halt.detach().mean()),
            "mean_step_delta": float(workspace_output.update_norms.detach().mean()),
            "mean_write_gate": float(workspace_output.write_gates.detach().mean()),
            "logical_charged_tokens": float(charged),
        }

    def generation_embeddings(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_ids.ndim != 2 or prompt_attention.shape != prompt_ids.shape:
            raise ProductReasoningTrainError("QST1 generation geometry differs")
        embedding = self.text_model.embed_tokens
        prompt_ids = prompt_ids.to(embedding.weight.device)
        prompt_attention = prompt_attention.to(embedding.weight.device)
        prompt_embeddings = embedding(prompt_ids)
        prompt_features = self.text_model(
            input_ids=prompt_ids,
            attention_mask=prompt_attention,
            use_cache=False,
        ).last_hidden_state
        output = self.workspace(
            prompt_features,
            prompt_attention,
            control=self.control,
        )
        prefix = output.prefix_states.to(dtype=prompt_embeddings.dtype)
        prefix_attention = torch.ones(
            prefix.shape[:2],
            device=prompt_attention.device,
            dtype=prompt_attention.dtype,
        )
        return (
            torch.cat((prompt_embeddings, prefix), dim=1),
            torch.cat((prompt_attention, prefix_attention), dim=1),
        )


def frozen_parameter_sha256(model: nn.Module) -> str:
    """Hash every protected parameter through exact raw bytes."""

    digest = hashlib.sha256()
    frozen = 0
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            continue
        frozen += 1
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        raw = parameter.detach().to("cpu").contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    if frozen == 0:
        raise ProductReasoningTrainError("model exposes no protected parameters")
    return digest.hexdigest()
