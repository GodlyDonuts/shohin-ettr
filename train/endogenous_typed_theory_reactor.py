"""Trainable endogenous typed-theory reactor for Shohin.

The module adds an explicit source-deleted typed graph state to the language
model. A raw-token compiler writes the state, a domain-neutral recurrent
controller updates it through structural transactions, and a late-query
reader consumes only the resulting state. No task-family solver or semantic
host callback appears in this runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, _supervised_lm_loss


SYSTEM_PARAMETER_CAP = 200_000_000
TRANSACTION_COUNT = 9
DISPOSITION_COUNT = 4
HARD_SURROGATE_GRADIENT_CAP = 1.0


class TheoryReactorError(ValueError):
    """An architecture shape, state, or parameter contract failed."""


@dataclass(frozen=True, slots=True)
class TheoryReactorConfig:
    d_model: int = 576
    state_width: int = 512
    # Production geometry reserves room for 32 object nodes plus reified
    # ordered hyperedge/value-byte nodes. Small synthetic tests override it.
    num_slots: int = 64
    num_types: int = 8
    num_relations: int = 16
    num_value_codes: int = 256
    max_edges: int = 256
    num_heads: int = 8
    compiler_layers: int = 3
    reactor_layers: int = 6
    query_layers: int = 2
    ff_multiplier: int = 4
    max_steps: int = 64
    stage_after_block: int = 19
    parameter_cap: int = SYSTEM_PARAMETER_CAP

    def validate(self, *, n_layer: int | None = None) -> None:
        positive = (
            self.d_model,
            self.state_width,
            self.num_slots,
            self.num_types,
            self.num_relations,
            self.num_value_codes,
            self.max_edges,
            self.num_heads,
            self.compiler_layers,
            self.reactor_layers,
            self.query_layers,
            self.ff_multiplier,
            self.max_steps,
        )
        if any(value <= 0 for value in positive):
            raise TheoryReactorError("all reactor dimensions must be positive")
        if self.state_width % self.num_heads:
            raise TheoryReactorError("state width must divide evenly across heads")
        if self.max_edges > (self.num_relations * self.num_slots * self.num_slots):
            raise TheoryReactorError("max_edges exceeds the relation ledger")
        if not 0 <= self.stage_after_block:
            raise TheoryReactorError("stage_after_block must be nonnegative")
        if n_layer is not None and self.stage_after_block >= n_layer - 1:
            raise TheoryReactorError("reactor stage must leave a decoder block")
        if self.parameter_cap > SYSTEM_PARAMETER_CAP:
            raise TheoryReactorError("parameter cap exceeds the system maximum")


@dataclass(frozen=True, slots=True)
class TypedTheoryState:
    """Exclusive state allowed across the source-deletion boundary."""

    value_probabilities: torch.Tensor
    type_probabilities: torch.Tensor
    relations: torch.Tensor
    active: torch.Tensor
    root: torch.Tensor
    committed: torch.Tensor
    halted: torch.Tensor
    step: int

    def detached_clone(self) -> "TypedTheoryState":
        return TypedTheoryState(
            *(
                value.detach().clone() if isinstance(value, torch.Tensor) else value
                for value in (
                    self.value_probabilities,
                    self.type_probabilities,
                    self.relations,
                    self.active,
                    self.root,
                    self.committed,
                    self.halted,
                    self.step,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class TransactionPolicy:
    """Discrete transition choices plus their differentiable supervision path."""

    opcode: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    relation: torch.Tensor
    type_index: torch.Tensor
    value_code: torch.Tensor
    opcode_probabilities: torch.Tensor
    source_probabilities: torch.Tensor
    target_probabilities: torch.Tensor
    relation_probabilities: torch.Tensor
    type_probabilities: torch.Tensor
    value_probabilities: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReactorTrace:
    """Differentiable policy probabilities and the choices applied to state."""

    opcode: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    relation: torch.Tensor
    type_index: torch.Tensor
    value_code: torch.Tensor
    applied_opcode: torch.Tensor
    applied_source: torch.Tensor
    applied_target: torch.Tensor
    applied_relation: torch.Tensor
    applied_type_index: torch.Tensor
    applied_value_code: torch.Tensor
    active: torch.Tensor
    committed: torch.Tensor
    halted: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReactorParameterReceipt:
    base_parameters: int
    architecture_parameters: int
    complete_system_parameters: int
    remaining_under_cap: int
    parameter_cap: int


def _hard_one_hot(probabilities: torch.Tensor) -> torch.Tensor:
    return _ExactOneHot.apply(probabilities)


def _hard_binary(probabilities: torch.Tensor) -> torch.Tensor:
    return _ExactBinary.apply(probabilities)


def _hard_capped_binary(
    probabilities: torch.Tensor,
    maximum: int,
) -> torch.Tensor:
    return _ExactCappedBinary.apply(probabilities, maximum)


def _bounded_hard_adjoint(value: torch.Tensor) -> torch.Tensor:
    return _BoundedHardAdjoint.apply(value)


def _bounded_surrogate_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return gradient.clamp(
        min=-HARD_SURROGATE_GRADIENT_CAP,
        max=HARD_SURROGATE_GRADIENT_CAP,
    )


def _disposition_probabilities(state: TypedTheoryState) -> torch.Tensor:
    """Return OPEN/ANSWER/ABSTAIN/REJECT without hiding terminal status."""

    committed = state.committed
    halted = state.halted
    return torch.stack(
        (
            (1.0 - committed) * (1.0 - halted),
            committed * (1.0 - halted),
            (1.0 - committed) * halted,
            committed * halted,
        ),
        dim=-1,
    )


class _ExactOneHot(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        return F.one_hot(
            probabilities.argmax(-1),
            probabilities.shape[-1],
        ).to(probabilities.dtype)

    @staticmethod
    def backward(
        ctx: object,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        del ctx
        return (_bounded_surrogate_gradient(gradient),)


class _ExactBinary(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        return probabilities.ge(0.5).to(probabilities.dtype)

    @staticmethod
    def backward(
        ctx: object,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        del ctx
        return (_bounded_surrogate_gradient(gradient),)


class _ExactCappedBinary(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        probabilities: torch.Tensor,
        maximum: int,
    ) -> torch.Tensor:
        del ctx
        flat = probabilities.flatten(1)
        binary = flat.ge(0.5)
        indices = flat.topk(
            min(maximum, flat.shape[1]),
            dim=-1,
        ).indices
        capped = torch.zeros_like(flat)
        capped.scatter_(1, indices, 1.0)
        over_capacity = binary.sum(-1, keepdim=True).gt(maximum)
        hard = torch.where(
            over_capacity,
            capped,
            binary.to(flat.dtype),
        )
        return hard.view_as(probabilities)

    @staticmethod
    def backward(
        ctx: object,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        del ctx
        return _bounded_surrogate_gradient(gradient), None


class _BoundedHardAdjoint(torch.autograd.Function):
    """Exact identity forward with a bounded recurrent-state adjoint."""

    @staticmethod
    def forward(
        ctx: object,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        return value

    @staticmethod
    def backward(
        ctx: object,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        del ctx
        return (_bounded_surrogate_gradient(gradient),)


def _edge_aware_relation_context(
    relations: torch.Tensor,
    slot_features: torch.Tensor,
    projection: nn.Linear,
) -> torch.Tensor:
    """Preserve neighbor identity while reading the typed relation ledger."""

    outgoing = torch.einsum(
        "brst,btw->bsrw",
        relations,
        slot_features,
    )
    incoming = torch.einsum(
        "brst,bsw->btrw",
        relations,
        slot_features,
    )
    messages = torch.cat((incoming, outgoing), dim=2).flatten(2)
    return projection(messages)


class _CompilerLayer(nn.Module):
    def __init__(self, config: TheoryReactorConfig):
        super().__init__()
        width = config.state_width
        heads = config.num_heads
        ff_width = width * config.ff_multiplier
        self.slot_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width,
            heads,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            width,
            heads,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, ff_width),
            nn.GELU(),
            nn.Linear(ff_width, width),
        )

    def forward(
        self,
        slots: torch.Tensor,
        tokens: torch.Tensor,
        token_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.slot_norm(slots)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        slots = slots + attended
        attended, _ = self.cross_attention(
            self.slot_norm(slots),
            self.token_norm(tokens),
            self.token_norm(tokens),
            key_padding_mask=token_padding_mask,
            need_weights=False,
        )
        slots = slots + attended
        return slots + self.ff(self.ff_norm(slots))


class EndogenousTheoryCompiler(nn.Module):
    """Compile transformer residuals into an anonymous typed graph."""

    def __init__(self, config: TheoryReactorConfig):
        super().__init__()
        config.validate()
        self.config = config
        width = config.state_width
        self.token_projection = nn.Linear(config.d_model, width)
        self.slot_queries = nn.Parameter(torch.empty(config.num_slots, width))
        self.layers = nn.ModuleList(
            _CompilerLayer(config) for _ in range(config.compiler_layers)
        )
        self.value_norm = nn.LayerNorm(width)
        self.value_head = nn.Linear(
            width,
            config.num_value_codes,
        )
        self.type_head = nn.Linear(width, config.num_types)
        self.active_head = nn.Linear(width, 1)
        self.root_query = nn.Parameter(torch.empty(width))
        self.relation_left = nn.Linear(
            width,
            config.num_relations * width,
            bias=False,
        )
        self.relation_right = nn.Linear(
            width,
            config.num_relations * width,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.slot_queries, std=0.02)
        nn.init.normal_(self.root_query, std=0.02)

    def forward(
        self,
        token_hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        hard: bool = False,
    ) -> TypedTheoryState:
        if token_hidden.ndim != 3 or token_hidden.shape[-1] != self.config.d_model:
            raise TheoryReactorError("token_hidden must be [batch,tokens,d_model]")
        batch, tokens, _ = token_hidden.shape
        padding_mask = _padding_mask(
            attention_mask,
            batch,
            tokens,
            token_hidden.device,
        )
        projected = self.token_projection(token_hidden)
        slots = self.slot_queries.to(
            dtype=projected.dtype,
            device=projected.device,
        )
        slots = slots.unsqueeze(0).expand(batch, -1, -1)
        for layer in self.layers:
            slots = layer(slots, projected, padding_mask)
        slots = self.value_norm(slots)
        value_probabilities = self.value_head(slots).float().softmax(-1)
        type_probabilities = self.type_head(slots).float().softmax(-1)
        active = self.active_head(slots).float().sigmoid().squeeze(-1)
        left = self.relation_left(slots).view(
            batch,
            self.config.num_slots,
            self.config.num_relations,
            self.config.state_width,
        )
        right = self.relation_right(slots).view_as(left)
        relation_logits = torch.einsum(
            "bsrw,btrw->brst",
            left,
            right,
        ) / math.sqrt(self.config.state_width)
        relations = relation_logits.sigmoid()
        root_logits = torch.einsum(
            "bsw,w->bs",
            slots,
            self.root_query.to(slots.dtype),
        )
        root = root_logits.float().softmax(-1)
        if hard:
            value_probabilities = _hard_one_hot(value_probabilities)
            type_probabilities = _hard_one_hot(type_probabilities)
            active = _hard_binary(active)
            root = _hard_one_hot(
                root_logits.float()
                .masked_fill(
                    active.eq(0),
                    torch.finfo(torch.float32).min,
                )
                .softmax(-1)
            )
        value_probabilities = value_probabilities * active.unsqueeze(-1)
        type_probabilities = type_probabilities * active.unsqueeze(-1)
        root = root * active
        root = root / root.sum(-1, keepdim=True).clamp_min(1e-6)
        pair_active = active[:, None, :, None] * active[:, None, None, :]
        relations = relations * pair_active
        if hard:
            relations = (
                _hard_capped_binary(
                    relations,
                    self.config.max_edges,
                )
                * pair_active
            )
        state = TypedTheoryState(
            value_probabilities=value_probabilities.to(slots.dtype),
            type_probabilities=type_probabilities.to(slots.dtype),
            relations=relations.to(slots.dtype),
            active=active.to(slots.dtype),
            root=root.to(slots.dtype),
            committed=torch.zeros(
                batch,
                device=slots.device,
                dtype=slots.dtype,
            ),
            halted=torch.zeros(
                batch,
                device=slots.device,
                dtype=slots.dtype,
            ),
            step=0,
        )
        validate_state(state, self.config)
        return state


class GenericTransactionReactor(nn.Module):
    """Emit and apply only ontology-neutral structural transactions."""

    def __init__(self, config: TheoryReactorConfig):
        super().__init__()
        config.validate()
        self.config = config
        width = config.state_width
        self.control_seed = nn.Parameter(torch.empty(width))
        self.step_embedding = nn.Embedding(config.max_steps, width)
        self.type_embedding = nn.Parameter(torch.empty(config.num_types, width))
        self.value_embedding = nn.Parameter(torch.empty(config.num_value_codes, width))
        self.active_projection = nn.Linear(1, width, bias=False)
        self.root_projection = nn.Linear(1, width, bias=False)
        self.status_projection = nn.Linear(DISPOSITION_COUNT, width, bias=False)
        self.relation_projection = nn.Linear(
            2 * config.num_relations * width,
            width,
            bias=False,
        )
        self.command_projection = nn.Linear(
            config.d_model,
            width,
        )
        self.command_norm = nn.LayerNorm(width)
        self.command_attention = nn.MultiheadAttention(
            width,
            config.num_heads,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=width * config.ff_multiplier,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.core = nn.TransformerEncoder(
            layer,
            num_layers=config.reactor_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(width)
        self.opcode_head = nn.Linear(width, TRANSACTION_COUNT)
        self.source_query = nn.Linear(width, width, bias=False)
        self.target_query = nn.Linear(width, width, bias=False)
        self.slot_key = nn.Linear(width, width, bias=False)
        self.relation_head = nn.Linear(width, config.num_relations)
        self.type_head = nn.Linear(width, config.num_types)
        self.value_head = nn.Linear(width, config.num_value_codes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.control_seed, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.normal_(self.value_embedding, std=0.02)

    def policy(
        self,
        state: TypedTheoryState,
        *,
        hard: bool,
        command_hidden: torch.Tensor | None = None,
        command_attention_mask: torch.Tensor | None = None,
        validate: bool = True,
    ) -> TransactionPolicy:
        command, command_padding = self._prepare_command(
            state,
            command_hidden,
            command_attention_mask,
        )
        return self._policy(
            state,
            hard=hard,
            command=command,
            command_padding=command_padding,
            validate=validate,
        )

    def _prepare_command(
        self,
        state: TypedTheoryState,
        command_hidden: torch.Tensor | None,
        command_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        command_padding: torch.Tensor | None = None
        command: torch.Tensor | None = None
        if command_hidden is not None:
            if (
                command_hidden.ndim != 3
                or command_hidden.shape[0] != state.value_probabilities.shape[0]
                or command_hidden.shape[-1] != self.config.d_model
            ):
                raise TheoryReactorError("command hidden geometry differs")
            command_padding = _padding_mask(
                command_attention_mask,
                command_hidden.shape[0],
                command_hidden.shape[1],
                command_hidden.device,
            )
            command = self.command_projection(command_hidden)
        elif command_attention_mask is not None:
            raise TheoryReactorError("command mask requires command hidden")
        return command, command_padding

    def _policy(
        self,
        state: TypedTheoryState,
        *,
        hard: bool,
        command: torch.Tensor | None,
        command_padding: torch.Tensor | None,
        validate: bool,
    ) -> TransactionPolicy:
        if validate:
            validate_state(state, self.config)
        if state.step >= self.config.max_steps:
            raise TheoryReactorError("reactor step exceeds maximum")
        type_context = torch.einsum(
            "bst,tw->bsw",
            state.type_probabilities,
            self.type_embedding,
        )
        value_context = torch.einsum(
            "bsc,cw->bsw",
            state.value_probabilities,
            self.value_embedding,
        )
        slots = (
            value_context
            + type_context
            + self.active_projection(state.active.unsqueeze(-1))
            + self.root_projection(state.root.unsqueeze(-1))
        )
        slots = slots + _edge_aware_relation_context(
            state.relations,
            slots,
            self.relation_projection,
        )
        pooled = (slots * state.active.unsqueeze(-1)).sum(1) / state.active.sum(
            1, keepdim=True
        ).clamp_min(1.0)
        control = (
            self.control_seed.to(slots.dtype).unsqueeze(0)
            + self.step_embedding.weight[state.step].to(slots.dtype)
            + pooled
            + self.status_projection(
                _disposition_probabilities(state)
            )
        )
        if command is not None:
            command_read, _ = self.command_attention(
                control[:, None, :],
                self.command_norm(command),
                self.command_norm(command),
                key_padding_mask=command_padding,
                need_weights=False,
            )
            control = control + command_read[:, 0]
        encoded = self.core(torch.cat((control[:, None, :], slots), dim=1))
        control = self.output_norm(encoded[:, 0])
        encoded_slots = self.output_norm(encoded[:, 1:])
        keys = self.slot_key(encoded_slots)
        source_probabilities = (
            torch.einsum(
                "bw,bsw->bs",
                self.source_query(control),
                keys,
            )
            .float()
            .softmax(-1)
        )
        target_probabilities = (
            torch.einsum(
                "bw,bsw->bs",
                self.target_query(control),
                keys,
            )
            .float()
            .softmax(-1)
        )
        opcode_probabilities = self.opcode_head(control).float().softmax(-1)
        relation_probabilities = self.relation_head(control).float().softmax(-1)
        type_probabilities = self.type_head(control).float().softmax(-1)
        value_probabilities = self.value_head(control).float().softmax(-1)
        opcode = opcode_probabilities
        source = source_probabilities
        target = target_probabilities
        relation = relation_probabilities
        type_index = type_probabilities
        value_code = value_probabilities
        if hard:
            opcode = _hard_one_hot(opcode_probabilities)
            source = _hard_one_hot(source_probabilities)
            target = _hard_one_hot(target_probabilities)
            relation = _hard_one_hot(relation_probabilities)
            type_index = _hard_one_hot(type_probabilities)
            value_code = _hard_one_hot(value_probabilities)
        dtype = state.value_probabilities.dtype
        return TransactionPolicy(
            opcode=opcode.to(dtype),
            source=source.to(dtype),
            target=target.to(dtype),
            relation=relation.to(dtype),
            type_index=type_index.to(dtype),
            value_code=value_code.to(dtype),
            opcode_probabilities=opcode_probabilities,
            source_probabilities=source_probabilities,
            target_probabilities=target_probabilities,
            relation_probabilities=relation_probabilities,
            type_probabilities=type_probabilities,
            value_probabilities=value_probabilities,
        )

    def apply(
        self,
        state: TypedTheoryState,
        policy: TransactionPolicy,
        *,
        hard: bool = False,
        validate: bool = True,
    ) -> TypedTheoryState:
        open_state = _disposition_probabilities(state)[:, 0:1]
        opcode = policy.opcode * open_state
        alloc = opcode[:, 0:1] * policy.source
        write = opcode[:, 1:2] * policy.source
        clear = opcode[:, 2:3] * policy.source
        link = opcode[:, 3]
        unlink = opcode[:, 4]
        set_root = opcode[:, 5:6] * policy.source
        commit = opcode[:, 6]
        halt = opcode[:, 7]
        reject = opcode[:, 8]

        allocated = alloc * (1.0 - state.active)
        cleared = clear * state.active
        active = (state.active + allocated * (1.0 - state.active)) * (1.0 - cleared)
        type_write = allocated.unsqueeze(-1)
        type_probabilities = (
            state.type_probabilities * (1.0 - type_write)
            + policy.type_index[:, None, :] * type_write
        )
        type_probabilities = type_probabilities * (1.0 - cleared.unsqueeze(-1))
        value_write = ((write * state.active) + allocated).clamp(max=1.0).unsqueeze(-1)
        value_probabilities = (
            state.value_probabilities * (1.0 - value_write)
            + policy.value_code[:, None, :] * value_write
        )
        value_probabilities = value_probabilities * (1.0 - cleared.unsqueeze(-1))

        pair = (
            policy.relation[:, :, None, None]
            * policy.source[:, None, :, None]
            * policy.target[:, None, None, :]
        )
        relations = state.relations + link[:, None, None, None] * pair * (
            1.0 - state.relations
        )
        relations = relations * (1.0 - unlink[:, None, None, None] * pair)
        clear_pair = (cleared[:, None, :, None] + cleared[:, None, None, :]).clamp(
            max=1.0
        )
        relations = relations * (1.0 - clear_pair)
        relations = relations * (active[:, None, :, None] * active[:, None, None, :])
        if hard:
            relations = _hard_capped_binary(
                relations,
                self.config.max_edges,
            ) * (active[:, None, :, None] * active[:, None, None, :])

        root = state.root * (1.0 - set_root.sum(-1, keepdim=True)) + set_root
        root = root * active
        root = root / root.sum(-1, keepdim=True).clamp_min(1e-6)
        committed = state.committed + (1.0 - state.committed) * (commit + reject)
        halted = state.halted + (1.0 - state.halted) * (halt + reject)
        if hard:
            (
                value_probabilities,
                type_probabilities,
                relations,
                active,
                root,
                committed,
                halted,
            ) = tuple(
                _bounded_hard_adjoint(value)
                for value in (
                    value_probabilities,
                    type_probabilities,
                    relations,
                    active,
                    root,
                    committed,
                    halted,
                )
            )
        result = TypedTheoryState(
            value_probabilities=value_probabilities,
            type_probabilities=type_probabilities,
            relations=relations,
            active=active,
            root=root,
            committed=committed,
            halted=halted,
            step=state.step + 1,
        )
        if validate:
            validate_state(result, self.config)
        return result

    def forward(
        self,
        state: TypedTheoryState,
        *,
        steps: int,
        hard: bool = False,
        command_hidden: torch.Tensor | None = None,
        command_attention_mask: torch.Tensor | None = None,
    ) -> tuple[TypedTheoryState, ReactorTrace]:
        if not 1 <= steps <= self.config.max_steps - state.step:
            raise TheoryReactorError("requested reactor steps differ")
        validate_state(state, self.config)
        command, command_padding = self._prepare_command(
            state,
            command_hidden,
            command_attention_mask,
        )
        policies: list[TransactionPolicy] = []
        states: list[TypedTheoryState] = []
        for _ in range(steps):
            policy = self._policy(
                state,
                hard=hard,
                command=command,
                command_padding=command_padding,
                validate=False,
            )
            state = self.apply(
                state,
                policy,
                hard=hard,
                validate=False,
            )
            policies.append(policy)
            states.append(state)
        validate_state(state, self.config)
        return state, ReactorTrace(
            opcode=torch.stack(
                [item.opcode_probabilities for item in policies],
                dim=1,
            ),
            source=torch.stack(
                [item.source_probabilities for item in policies],
                dim=1,
            ),
            target=torch.stack(
                [item.target_probabilities for item in policies],
                dim=1,
            ),
            relation=torch.stack(
                [item.relation_probabilities for item in policies],
                dim=1,
            ),
            type_index=torch.stack(
                [item.type_probabilities for item in policies],
                dim=1,
            ),
            value_code=torch.stack(
                [item.value_probabilities for item in policies],
                dim=1,
            ),
            applied_opcode=torch.stack(
                [item.opcode for item in policies],
                dim=1,
            ),
            applied_source=torch.stack(
                [item.source for item in policies],
                dim=1,
            ),
            applied_target=torch.stack(
                [item.target for item in policies],
                dim=1,
            ),
            applied_relation=torch.stack(
                [item.relation for item in policies],
                dim=1,
            ),
            applied_type_index=torch.stack(
                [item.type_index for item in policies],
                dim=1,
            ),
            applied_value_code=torch.stack(
                [item.value_code for item in policies],
                dim=1,
            ),
            active=torch.stack([item.active for item in states], dim=1),
            committed=torch.stack(
                [item.committed for item in states],
                dim=1,
            ),
            halted=torch.stack([item.halted for item in states], dim=1),
        )


class SourceDeletedQueryReader(nn.Module):
    def __init__(self, config: TheoryReactorConfig):
        super().__init__()
        config.validate()
        width = config.state_width
        self.query_projection = nn.Linear(config.d_model, width)
        self.value_embedding = nn.Parameter(torch.empty(config.num_value_codes, width))
        self.type_embedding = nn.Parameter(torch.empty(config.num_types, width))
        self.active_projection = nn.Linear(1, width, bias=False)
        self.root_projection = nn.Linear(1, width, bias=False)
        self.relation_projection = nn.Linear(
            2 * config.num_relations * width,
            width,
            bias=False,
        )
        self.status_projection = nn.Linear(DISPOSITION_COUNT, width, bias=False)
        self.state_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width,
            config.num_heads,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=width * config.ff_multiplier,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.query_core = nn.TransformerEncoder(
            layer,
            num_layers=config.query_layers,
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(width, config.d_model)
        self.gate = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.value_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(
        self,
        query_hidden: torch.Tensor,
        state: TypedTheoryState,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, _ = query_hidden.shape
        query_padding = _padding_mask(
            attention_mask,
            batch,
            tokens,
            query_hidden.device,
        )
        query = self.query_projection(query_hidden)
        state_padding = state.active.lt(0.5)
        empty = state_padding.all(-1)
        first_slot = F.one_hot(
            torch.zeros(
                state_padding.shape[0],
                dtype=torch.long,
                device=state_padding.device,
            ),
            state_padding.shape[1],
        ).bool()
        state_padding = state_padding & ~(empty[:, None] & first_slot)
        state_values = torch.einsum(
            "bsc,cw->bsw",
            state.value_probabilities,
            self.value_embedding,
        )
        state_types = torch.einsum(
            "bst,tw->bsw",
            state.type_probabilities,
            self.type_embedding,
        )
        status = self.status_projection(
            _disposition_probabilities(state)
        )[:, None, :]
        state_values = (
            state_values
            + state_types
            + self.active_projection(state.active.unsqueeze(-1))
            + self.root_projection(state.root.unsqueeze(-1))
        )
        semantic_state = (
            state_values
            + _edge_aware_relation_context(
                state.relations,
                state_values,
                self.relation_projection,
            )
        )
        answer = _disposition_probabilities(state)[:, 1:2]
        state_values = semantic_state * answer[:, :, None] + status
        read, _ = self.cross_attention(
            query,
            self.state_norm(state_values),
            self.state_norm(state_values),
            key_padding_mask=state_padding,
            need_weights=False,
        )
        causal_mask = torch.ones(
            tokens,
            tokens,
            dtype=torch.bool,
            device=query.device,
        ).triu(diagonal=1)
        query = self.query_core(
            query + read,
            mask=causal_mask,
            src_key_padding_mask=query_padding,
        )
        return torch.tanh(self.gate) * self.output_projection(query)


class EndogenousTypedTheoryReactorGPT(nn.Module):
    """Checkpoint-compatible Shohin plus compiler, reactor, and query reader."""

    def __init__(self, base: GPT, config: TheoryReactorConfig):
        super().__init__()
        config.validate(n_layer=base.cfg.n_layer)
        if base.cfg.d_model != config.d_model:
            raise TheoryReactorError("reactor d_model must match Shohin")
        if base.cfg.n_loop != 1:
            raise TheoryReactorError("reactor reference requires n_loop=1")
        self.base = base
        self.config = config
        self.compiler = EndogenousTheoryCompiler(config)
        self.reactor = GenericTransactionReactor(config)
        self.query_reader = SourceDeletedQueryReader(config)

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def parameter_receipt(self) -> ReactorParameterReceipt:
        base_ids = {id(parameter) for parameter in self.base.parameters()}
        base_parameters = sum(parameter.numel() for parameter in self.base.parameters())
        architecture_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) not in base_ids
        )
        complete = base_parameters + architecture_parameters
        if complete > self.config.parameter_cap:
            raise TheoryReactorError("complete system exceeds parameter cap")
        return ReactorParameterReceipt(
            base_parameters=base_parameters,
            architecture_parameters=architecture_parameters,
            complete_system_parameters=complete,
            remaining_under_cap=self.config.parameter_cap - complete,
            parameter_cap=self.config.parameter_cap,
        )

    def compile_world(
        self,
        world_idx: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        hard: bool = False,
    ) -> TypedTheoryState:
        hidden = self._encode_to_stage(world_idx, pos=0)
        return self.compiler(
            hidden,
            attention_mask=attention_mask,
            hard=hard,
        )

    def execute(
        self,
        state: TypedTheoryState,
        *,
        steps: int,
        hard: bool = False,
        command_idx: torch.Tensor | None = None,
        command_attention_mask: torch.Tensor | None = None,
    ) -> tuple[TypedTheoryState, ReactorTrace]:
        command_hidden = (
            None if command_idx is None else self._encode_to_stage(command_idx, pos=0)
        )
        return self.reactor(
            state,
            steps=steps,
            hard=hard,
            command_hidden=command_hidden,
            command_attention_mask=command_attention_mask,
        )

    def answer_query(
        self,
        state: TypedTheoryState,
        query_idx: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self._encode_to_stage(query_idx, pos=0)
        hidden = hidden + self.query_reader(
            hidden,
            state,
            attention_mask=attention_mask,
        )
        hidden = self._decode_from_stage(hidden, pos=0)
        logits = self.base.head(self.base.norm(hidden))
        loss = None
        if targets is not None:
            loss = _supervised_lm_loss(
                logits,
                targets,
                self.base.cfg.zloss,
            )
        return logits, loss

    def forward_staged(
        self,
        world_idx: torch.Tensor,
        query_idx: torch.Tensor,
        *,
        reactor_steps: int,
        command_idx: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        world_attention_mask: torch.Tensor | None = None,
        command_attention_mask: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        hard: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        TypedTheoryState,
        ReactorTrace,
    ]:
        state = self.compile_world(
            world_idx,
            attention_mask=world_attention_mask,
            hard=hard,
        )
        state, trace = self.execute(
            state,
            steps=reactor_steps,
            hard=hard,
            command_idx=command_idx,
            command_attention_mask=command_attention_mask,
        )
        logits, loss = self.answer_query(
            state,
            query_idx,
            targets=targets,
            attention_mask=query_attention_mask,
        )
        return logits, loss, state, trace

    def _encode_to_stage(
        self,
        idx: torch.Tensor,
        *,
        pos: int,
    ) -> torch.Tensor:
        self._validate_tokens(idx, pos=pos)
        hidden = self.base.tok(idx)
        cos = self.base.cos[pos : pos + idx.shape[1]].to(hidden.device)
        sin = self.base.sin[pos : pos + idx.shape[1]].to(hidden.device)
        for block in self.base.blocks[: self.config.stage_after_block + 1]:
            hidden, _ = block(hidden, cos, sin)
        return hidden

    def _decode_from_stage(
        self,
        hidden: torch.Tensor,
        *,
        pos: int,
    ) -> torch.Tensor:
        cos = self.base.cos[pos : pos + hidden.shape[1]].to(hidden.device)
        sin = self.base.sin[pos : pos + hidden.shape[1]].to(hidden.device)
        for block in self.base.blocks[self.config.stage_after_block + 1 :]:
            hidden, _ = block(hidden, cos, sin)
        return hidden

    def _validate_tokens(
        self,
        idx: torch.Tensor,
        *,
        pos: int,
    ) -> None:
        if idx.ndim != 2 or idx.dtype != torch.long:
            raise TheoryReactorError("token ids must be a rank-two long tensor")
        if pos < 0 or pos + idx.shape[1] > self.base.cfg.seq_len:
            raise TheoryReactorError(
                "token positions exceed configured sequence length"
            )


def validate_state(
    state: TypedTheoryState,
    config: TheoryReactorConfig,
) -> None:
    config.validate()
    if not isinstance(state, TypedTheoryState):
        raise TheoryReactorError("typed theory state differs")
    batch = state.value_probabilities.shape[0]
    expected = {
        "value_probabilities": (
            batch,
            config.num_slots,
            config.num_value_codes,
        ),
        "type_probabilities": (
            batch,
            config.num_slots,
            config.num_types,
        ),
        "relations": (
            batch,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ),
        "active": (batch, config.num_slots),
        "root": (batch, config.num_slots),
        "committed": (batch,),
        "halted": (batch,),
    }
    devices = set()
    for field in fields(state):
        value = getattr(state, field.name)
        if field.name == "step":
            if not isinstance(value, int) or not 0 <= value <= config.max_steps:
                raise TheoryReactorError("state step differs")
            continue
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[field.name]
            or not value.is_floating_point()
        ):
            raise TheoryReactorError(f"state {field.name} differs")
        _require_tensor(
            torch.isfinite(value).all(),
            f"state {field.name} is nonfinite",
        )
        devices.add(value.device)
    if len(devices) != 1:
        raise TheoryReactorError("state tensors must share one device")


def validate_deployed_state(
    state: TypedTheoryState,
    config: TheoryReactorConfig,
) -> None:
    """Require a bounded discrete packet at a process boundary."""

    validate_state(state, config)

    def require_binary(
        name: str,
        value: torch.Tensor,
    ) -> None:
        _require_tensor(
            ((value == 0) | (value == 1)).all(),
            f"deployed state {name} is not binary",
        )

    require_binary("active", state.active)
    require_binary("committed", state.committed)
    require_binary("halted", state.halted)
    require_binary("relations", state.relations)
    require_binary("root", state.root)
    require_binary(
        "type_probabilities",
        state.type_probabilities,
    )
    require_binary(
        "value_probabilities",
        state.value_probabilities,
    )
    _require_tensor(
        state.value_probabilities.sum(-1).eq(state.active).all(),
        "deployed value codes are not active-slot one-hot",
    )
    _require_tensor(
        state.type_probabilities.sum(-1).eq(state.active).all(),
        "deployed types are not active-slot one-hot",
    )
    _require_tensor(
        (state.root.sum(-1) <= 1).all() & (state.root <= state.active).all(),
        "deployed root is not an active-slot pointer",
    )
    pair_active = state.active[:, None, :, None] * state.active[:, None, None, :]
    _require_tensor(
        (state.relations <= pair_active).all()
        & (state.relations.sum(dim=(1, 2, 3)) <= config.max_edges).all(),
        "deployed relation ledger exceeds its sparse bounds",
    )


def _require_tensor(
    condition: torch.Tensor,
    message: str,
) -> None:
    """Assert a scalar tensor without synchronizing the CUDA hot loop."""

    if condition.ndim:
        raise TheoryReactorError("internal tensor assertion must be scalar")
    if condition.device.type == "cuda" or torch.compiler.is_compiling():
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise TheoryReactorError(message)


def _padding_mask(
    attention_mask: torch.Tensor | None,
    batch: int,
    tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.shape != (batch, tokens):
        raise TheoryReactorError("attention mask has the wrong shape")
    if attention_mask.dtype != torch.bool:
        _require_tensor(
            ((attention_mask == 0) | (attention_mask == 1)).all(),
            "attention mask must be binary",
        )
        attention_mask = attention_mask.bool()
    _require_tensor(
        ~(
            attention_mask[:, 1:].to(torch.int8) > attention_mask[:, :-1].to(torch.int8)
        ).any(),
        "attention mask must be right padded",
    )
    return ~attention_mask.to(device=device)


__all__ = [
    "EndogenousTheoryCompiler",
    "EndogenousTypedTheoryReactorGPT",
    "GenericTransactionReactor",
    "ReactorParameterReceipt",
    "ReactorTrace",
    "SYSTEM_PARAMETER_CAP",
    "SourceDeletedQueryReader",
    "TRANSACTION_COUNT",
    "TheoryReactorConfig",
    "TheoryReactorError",
    "TransactionPolicy",
    "TypedTheoryState",
    "validate_deployed_state",
    "validate_state",
]
