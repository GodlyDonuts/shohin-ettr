"""Query-conditioned pointer transactions for product reasoning."""

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
class QPT1Config:
    """Geometry of the coherent hard-pointer transaction workspace."""

    backbone_width: int
    workspace_width: int = 512
    source_slots: int = 8
    query_slots: int = 4
    recurrent_steps: int = 8
    attention_heads: int = 8
    ff_multiplier: int = 2
    pointer_temperature: float = 0.50

    def validate(self) -> None:
        dimensions = (
            self.backbone_width,
            self.workspace_width,
            self.source_slots,
            self.query_slots,
            self.recurrent_steps,
            self.attention_heads,
            self.ff_multiplier,
        )
        if any(value <= 0 for value in dimensions):
            raise ProductReasoningTrainError("QPT1 dimensions must be positive")
        if self.workspace_width % self.attention_heads:
            raise ProductReasoningTrainError(
                "QPT1 workspace width must divide attention heads"
            )
        if self.pointer_temperature <= 0.0:
            raise ProductReasoningTrainError(
                "QPT1 pointer temperature must be positive"
            )


@dataclass
class QPT1Output:
    """Complete pointer and transaction trace for one prompt batch."""

    prompt_residuals: torch.Tensor
    source_packet: torch.Tensor
    initial_state: torch.Tensor
    final_state: torch.Tensor
    query_state: torch.Tensor
    source_assignments: torch.Tensor
    query_assignments: torch.Tensor
    transaction_reads: torch.Tensor
    transaction_writes: torch.Tensor
    update_norms: torch.Tensor
    commit_gates: torch.Tensor
    release_gates: torch.Tensor


def qpt1_architecture_sha256(config: QPT1Config) -> str:
    """Bind checkpoints to the exact QPT1 mechanism and geometry."""

    payload = {
        "architecture": "diverge-qpt1-hard-pointer-transactions-v1",
        "config": asdict(config),
        "invariants": [
            "hard_query_conditioned_source_pointers",
            "whole_source_reads",
            "whole_state_writes",
            "fixed_depth_tied_transactions",
            "no_fieldwise_hypothesis_averaging",
            "zero_initialized_identity_residual",
            "unchanged_prompt_sequence_geometry",
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


class QueryPointerTransactionWorkspace(nn.Module):
    """Select complete observations and apply one coherent write per step."""

    def __init__(self, config: QPT1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.workspace_width

        self.prompt_projection = nn.Linear(config.backbone_width, width)
        self.prompt_norm = nn.LayerNorm(width)
        self.prompt_key = nn.Linear(width, width, bias=False)
        self.prompt_value = nn.Linear(width, width, bias=False)

        self.query_seeds = nn.Parameter(torch.empty(config.query_slots, width))
        self.query_pointer = nn.Linear(width, width, bias=False)
        self.query_ffn = _ResidualFFN(width, config.ff_multiplier)

        self.source_seeds = nn.Parameter(torch.empty(config.source_slots, width))
        self.source_identities = nn.Parameter(torch.empty(config.source_slots, width))
        self.query_to_source = nn.Linear(width, width, bias=False)
        self.source_pointer = nn.Linear(width, width, bias=False)
        self.source_ffn = _ResidualFFN(width, config.ff_multiplier)

        self.state_seeds = nn.Parameter(torch.empty(config.source_slots, width))
        self.state_ffn = _ResidualFFN(width, config.ff_multiplier)
        self.step_identities = nn.Parameter(torch.empty(config.recurrent_steps, width))
        self.controller_norm = nn.LayerNorm(width)
        self.read_query = nn.Linear(width, width, bias=False)
        self.read_key = nn.Linear(width, width, bias=False)
        self.write_query = nn.Linear(width, width, bias=False)
        self.write_key = nn.Linear(width, width, bias=False)
        self.transaction = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width * config.ff_multiplier),
            nn.SiLU(),
            nn.Linear(width * config.ff_multiplier, width),
        )
        self.commit_gate = nn.Linear(width * 4, 1)

        self.query_state_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.query_source_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.query_norm = nn.LayerNorm(width)
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, config.backbone_width)
        self.release_gate = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )
        self.binding_source = nn.Linear(width, width, bias=False)
        self.binding_state = nn.Linear(width, width, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.query_seeds,
            self.source_seeds,
            self.source_identities,
            self.state_seeds,
            self.step_identities,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.constant_(self.release_gate[-1].bias, -1.0)

    def _hard_pointer(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            logits = logits.masked_fill(~mask[:, None, :].bool(), float("-inf"))
        soft = torch.softmax(
            logits.float() / self.config.pointer_temperature, dim=-1
        ).to(logits.dtype)
        hard = F.one_hot(soft.argmax(dim=-1), soft.shape[-1]).to(soft.dtype)
        if self.training:
            return hard + soft - soft.detach()
        return hard

    @staticmethod
    def _attend(
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = attention(query, key_value, key_value, need_weights=False)
        return output

    def _compile(
        self,
        prompt_features: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = prompt_features.shape[0]
        prompt = self.prompt_norm(self.prompt_projection(prompt_features))
        keys = F.normalize(self.prompt_key(prompt).float(), dim=-1).to(prompt.dtype)
        values = self.prompt_value(prompt)

        query_seeds = self.query_seeds.unsqueeze(0).expand(batch, -1, -1)
        query_vectors = F.normalize(self.query_pointer(query_seeds).float(), dim=-1).to(
            prompt.dtype
        )
        query_logits = torch.einsum("bqd,btd->bqt", query_vectors, keys)
        query_assignments = self._hard_pointer(query_logits, prompt_mask)
        query_context = torch.einsum("bqt,btd->bqd", query_assignments, values)
        query_context = self.query_ffn(query_context + query_seeds)

        query_summary = query_context.mean(dim=1, keepdim=True)
        source_seeds = self.source_seeds.unsqueeze(0).expand(
            batch, -1, -1
        ) + self.query_to_source(query_summary)
        source_vectors = F.normalize(
            self.source_pointer(source_seeds).float(), dim=-1
        ).to(prompt.dtype)
        source_logits = torch.einsum("bsd,btd->bst", source_vectors, keys)
        source_assignments = self._hard_pointer(source_logits, prompt_mask)
        source_packet = torch.einsum("bst,btd->bsd", source_assignments, values)
        source_packet = self.source_ffn(
            source_packet + source_seeds + self.source_identities
        )
        return source_packet, query_context, source_assignments, query_assignments

    def _transition(
        self,
        source_packet: torch.Tensor,
        query_context: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = source_packet.shape[0]
        state = self.state_ffn(
            source_packet + self.state_seeds.unsqueeze(0).expand(batch, -1, -1)
        )
        initial_state = state
        query_summary = query_context.mean(dim=1)
        reads: list[torch.Tensor] = []
        writes: list[torch.Tensor] = []
        update_norms: list[torch.Tensor] = []
        commit_gates: list[torch.Tensor] = []
        source_keys = F.normalize(self.read_key(source_packet).float(), dim=-1).to(
            source_packet.dtype
        )
        for step in range(self.config.recurrent_steps):
            step_state = self.step_identities[step].unsqueeze(0).expand(batch, -1)
            controller = self.controller_norm(
                state.mean(dim=1) + query_summary + step_state
            )
            read_vector = F.normalize(self.read_query(controller).float(), dim=-1).to(
                state.dtype
            )
            read_logits = torch.einsum("bd,bsd->bs", read_vector, source_keys)
            read = self._hard_pointer(read_logits[:, None, :]).squeeze(1)
            selected_source = torch.einsum("bs,bsd->bd", read, source_packet)

            state_keys = F.normalize(self.write_key(state).float(), dim=-1).to(
                state.dtype
            )
            write_vector = F.normalize(self.write_query(controller).float(), dim=-1).to(
                state.dtype
            )
            write_logits = torch.einsum("bd,bsd->bs", write_vector, state_keys)
            write = self._hard_pointer(write_logits[:, None, :]).squeeze(1)
            selected_state = torch.einsum("bs,bsd->bd", write, state)

            transaction_input = torch.cat(
                (selected_state, selected_source, query_summary, step_state), dim=-1
            )
            proposal = self.transaction(transaction_input)
            gate = torch.sigmoid(self.commit_gate(transaction_input))
            update = gate * (proposal - selected_state)
            state = state + write.unsqueeze(-1) * update.unsqueeze(1)
            reads.append(read)
            writes.append(write)
            update_norms.append(update.float().pow(2).mean(dim=-1).sqrt())
            commit_gates.append(gate.squeeze(-1).float())
        return (
            initial_state,
            state,
            torch.stack(reads, dim=1),
            torch.stack(writes, dim=1),
            torch.stack(update_norms, dim=1),
            torch.stack(commit_gates, dim=1),
        )

    def forward(
        self,
        prompt_features: torch.Tensor,
        prompt_mask: torch.Tensor,
        *,
        control: str = "normal",
    ) -> QPT1Output:
        if prompt_features.ndim != 3 or prompt_mask.shape != prompt_features.shape[:2]:
            raise ProductReasoningTrainError("QPT1 prompt geometry differs")
        if control not in {"normal", "packet_swap", "state_reset", "release_off"}:
            raise ProductReasoningTrainError("QPT1 control differs")
        source, query_context, source_assignments, query_assignments = self._compile(
            prompt_features, prompt_mask
        )
        if control == "packet_swap":
            source = source.roll(1, dims=0 if source.shape[0] > 1 else 1)
        (
            initial_state,
            final_state,
            reads,
            writes,
            update_norms,
            commit_gates,
        ) = self._transition(source, query_context)
        if control == "state_reset":
            final_state = initial_state
        query_state = query_context + self._attend(
            self.query_state_attention,
            self.query_norm(query_context),
            final_state,
        )
        query_state = self.query_ffn(
            query_state
            + self._attend(
                self.query_source_attention,
                self.query_norm(query_state),
                source,
            )
        )
        release_gates = torch.sigmoid(self.release_gate(query_state))
        residuals = self.output_projection(self.output_norm(query_state))
        residuals = residuals * release_gates
        if control == "release_off":
            residuals = torch.zeros_like(residuals)
        return QPT1Output(
            prompt_residuals=residuals,
            source_packet=source,
            initial_state=initial_state,
            final_state=final_state,
            query_state=query_state,
            source_assignments=source_assignments,
            query_assignments=query_assignments,
            transaction_reads=reads,
            transaction_writes=writes,
            update_norms=update_norms,
            commit_gates=commit_gates,
            release_gates=release_gates,
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


class QPT1ProductModel(nn.Module):
    """Pinned reasoning backbone plus hard pointer transactions and LoRA."""

    architecture = "diverge-qpt1"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        lora_layers: int,
        lora_rank: int,
        lora_alpha: float,
        workspace_width: int,
        source_slots: int,
        query_slots: int,
        recurrent_steps: int,
        attention_heads: int,
        ff_multiplier: int,
        pointer_temperature: float,
        binding_weight: float,
        coverage_weight: float,
        reset_weight: float,
    ) -> None:
        super().__init__()
        if min(binding_weight, coverage_weight, reset_weight) < 0.0:
            raise ProductReasoningTrainError("QPT1 loss weights must be nonnegative")
        self.backbone = backbone
        self.arm = "diverge_qpt1"
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

        self.workspace_config = QPT1Config(
            backbone_width=hidden_size,
            workspace_width=workspace_width,
            source_slots=source_slots,
            query_slots=query_slots,
            recurrent_steps=recurrent_steps,
            attention_heads=attention_heads,
            ff_multiplier=ff_multiplier,
            pointer_temperature=pointer_temperature,
        )
        self.workspace_config.validate()
        self.workspace = QueryPointerTransactionWorkspace(self.workspace_config)
        self.binding_weight = binding_weight
        self.coverage_weight = coverage_weight
        self.reset_weight = reset_weight
        self.control = "normal"

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def sequence_workspace_slots(self) -> int:
        return 0

    def set_control(self, control: str) -> None:
        if control not in {"normal", "packet_swap", "state_reset", "release_off"}:
            raise ProductReasoningTrainError("QPT1 control differs")
        self.control = control

    def _workspace_output(
        self,
        prompt_rows: list[list[int]],
        pad_token_id: int,
        *,
        control: str = "normal",
    ) -> QPT1Output:
        embedding = self.text_model.embed_tokens
        prompt_ids, prompt_mask = _pad_token_rows(prompt_rows, pad_token_id)
        prompt_ids = prompt_ids.to(embedding.weight.device)
        prompt_mask = prompt_mask.to(embedding.weight.device)
        with torch.no_grad():
            prompt_features = self.text_model(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                use_cache=False,
            ).last_hidden_state
        return self.workspace(prompt_features, prompt_mask, control=control)

    @staticmethod
    def _pointer_overlap(assignments: torch.Tensor) -> torch.Tensor:
        slots = assignments.shape[1]
        if slots <= 1:
            return assignments.new_zeros(())
        gram = torch.matmul(assignments, assignments.transpose(1, 2))
        diagonal = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
        return ((gram.sum(dim=(1, 2)) - diagonal) / (slots * (slots - 1))).mean()

    def _auxiliary_losses(
        self,
        output: QPT1Output,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, final_state = self.workspace.binding_vectors(
            output.source_packet, output.final_state
        )
        _, initial_state = self.workspace.binding_vectors(
            output.source_packet, output.initial_state
        )
        if source.shape[0] > 1:
            logits = torch.matmul(final_state, source.transpose(0, 1)) / 0.10
            labels = torch.arange(source.shape[0], device=source.device)
            binding_loss = 0.5 * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.transpose(0, 1), labels)
            )
        else:
            binding_loss = F.softplus(0.20 - (source * final_state).sum(dim=-1)).mean()
        clean_score = (source * final_state).sum(dim=-1)
        reset_score = (source * initial_state).sum(dim=-1)
        reset_loss = F.relu(0.10 - clean_score + reset_score).mean()

        pointer_overlap = self._pointer_overlap(output.source_assignments)
        source_count = output.transaction_reads.shape[-1]
        state_count = output.transaction_writes.shape[-1]
        read_frequency = output.transaction_reads.mean(dim=1)
        write_frequency = output.transaction_writes.mean(dim=1)
        read_balance = ((read_frequency - 1.0 / source_count) ** 2).mean()
        write_balance = ((write_frequency - 1.0 / state_count) ** 2).mean()
        coverage_loss = pointer_overlap + read_balance + write_balance
        return binding_loss, coverage_loss, reset_loss

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        response_rows: list[list[int]],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if len(prompt_rows) != len(response_rows) or not prompt_rows:
            raise ProductReasoningTrainError("QPT1 batch geometry differs")
        embedding = self.text_model.embed_tokens
        output = self._workspace_output(prompt_rows, pad_token_id)
        residuals = output.prompt_residuals.to(dtype=embedding.weight.dtype)
        inputs, attention, labels, charged = pack_training_embeddings(
            embedding,
            prompt_rows,
            response_rows,
            None,
            pad_token_id,
            prompt_residuals=residuals,
        )
        backbone_output = self.text_model(
            inputs_embeds=inputs,
            attention_mask=attention,
            use_cache=False,
        )
        logits = self.lm_head(backbone_output.last_hidden_state)
        language_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        binding_loss, coverage_loss, reset_loss = self._auxiliary_losses(output)
        loss = (
            language_loss
            + self.binding_weight * binding_loss
            + self.coverage_weight * coverage_loss
            + self.reset_weight * reset_loss
        )
        return loss, {
            "language_loss": float(language_loss.detach()),
            "binding_loss": float(binding_loss.detach()),
            "coverage_loss": float(coverage_loss.detach()),
            "reset_loss": float(reset_loss.detach()),
            "mean_step_delta": float(output.update_norms.detach().mean()),
            "mean_commit_gate": float(output.commit_gates.detach().mean()),
            "mean_release_gate": float(output.release_gates.detach().mean()),
            "logical_charged_tokens": float(charged),
        }

    def generation_embeddings(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_ids.ndim != 2 or prompt_attention.shape != prompt_ids.shape:
            raise ProductReasoningTrainError("QPT1 generation geometry differs")
        embedding = self.text_model.embed_tokens
        prompt_ids = prompt_ids.to(embedding.weight.device)
        prompt_attention = prompt_attention.to(embedding.weight.device)
        prompt_embeddings = embedding(prompt_ids)
        with torch.no_grad():
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
        residuals = output.prompt_residuals.to(dtype=prompt_embeddings.dtype)
        result = prompt_embeddings.clone()
        for batch_index in range(result.shape[0]):
            valid = torch.nonzero(
                prompt_attention[batch_index], as_tuple=False
            ).flatten()
            count = min(int(valid.numel()), int(residuals.shape[1]))
            if count:
                result[batch_index, valid[-count:]] += residuals[batch_index, :count]
        return result, prompt_attention


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
