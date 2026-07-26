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
TRANSACTION_COUNT = 8


class TheoryReactorError(ValueError):
    """An architecture shape, state, or parameter contract failed."""


@dataclass(frozen=True, slots=True)
class TheoryReactorConfig:
    d_model: int = 576
    state_width: int = 512
    num_slots: int = 24
    num_types: int = 8
    num_relations: int = 8
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
            self.num_heads,
            self.compiler_layers,
            self.reactor_layers,
            self.query_layers,
            self.ff_multiplier,
            self.max_steps,
        )
        if any(value <= 0 for value in positive):
            raise TheoryReactorError(
                "all reactor dimensions must be positive"
            )
        if self.state_width % self.num_heads:
            raise TheoryReactorError(
                "state width must divide evenly across heads"
            )
        if not 0 <= self.stage_after_block:
            raise TheoryReactorError(
                "stage_after_block must be nonnegative"
            )
        if (
            n_layer is not None
            and self.stage_after_block >= n_layer - 1
        ):
            raise TheoryReactorError(
                "reactor stage must leave a decoder block"
            )
        if self.parameter_cap > SYSTEM_PARAMETER_CAP:
            raise TheoryReactorError(
                "parameter cap exceeds the system maximum"
            )


@dataclass(frozen=True, slots=True)
class TypedTheoryState:
    """Exclusive state allowed across the source-deletion boundary."""

    values: torch.Tensor
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
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else value
                for value in (
                    self.values,
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
    opcode: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    relation: torch.Tensor
    type_index: torch.Tensor
    payload: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReactorTrace:
    opcode: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    relation: torch.Tensor
    type_index: torch.Tensor
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
        return (gradient,)


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
        return (gradient,)


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
        self.slot_queries = nn.Parameter(
            torch.empty(config.num_slots, width)
        )
        self.layers = nn.ModuleList(
            _CompilerLayer(config)
            for _ in range(config.compiler_layers)
        )
        self.value_norm = nn.LayerNorm(width)
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
        if (
            token_hidden.ndim != 3
            or token_hidden.shape[-1] != self.config.d_model
        ):
            raise TheoryReactorError(
                "token_hidden must be [batch,tokens,d_model]"
            )
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
        values = self.value_norm(slots)
        type_probabilities = self.type_head(values).float().softmax(-1)
        active = self.active_head(values).float().sigmoid().squeeze(-1)
        left = self.relation_left(values).view(
            batch,
            self.config.num_slots,
            self.config.num_relations,
            self.config.state_width,
        )
        right = self.relation_right(values).view_as(left)
        relation_logits = torch.einsum(
            "bsrw,btrw->brst",
            left,
            right,
        ) / math.sqrt(self.config.state_width)
        relations = relation_logits.sigmoid()
        root_logits = torch.einsum(
            "bsw,w->bs",
            values,
            self.root_query.to(values.dtype),
        )
        root = root_logits.float().softmax(-1)
        if hard:
            type_probabilities = _hard_one_hot(type_probabilities)
            active = _hard_binary(active)
            relations = _hard_binary(relations)
            root = _hard_one_hot(root)
        pair_active = active[:, None, :, None] * active[:, None, None, :]
        relations = relations * pair_active
        state = TypedTheoryState(
            values=values,
            type_probabilities=type_probabilities.to(values.dtype),
            relations=relations.to(values.dtype),
            active=active.to(values.dtype),
            root=root.to(values.dtype),
            committed=torch.zeros(
                batch,
                device=values.device,
                dtype=values.dtype,
            ),
            halted=torch.zeros(
                batch,
                device=values.device,
                dtype=values.dtype,
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
        self.type_embedding = nn.Parameter(
            torch.empty(config.num_types, width)
        )
        self.active_projection = nn.Linear(1, width, bias=False)
        self.relation_projection = nn.Linear(
            2 * config.num_relations,
            width,
            bias=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=width * config.ff_multiplier,
            batch_first=True,
            norm_first=True,
            activation="gelu",
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
        self.payload_head = nn.Linear(width, width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.control_seed, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

    def policy(
        self,
        state: TypedTheoryState,
        *,
        hard: bool,
    ) -> TransactionPolicy:
        validate_state(state, self.config)
        if state.step >= self.config.max_steps:
            raise TheoryReactorError("reactor step exceeds maximum")
        type_context = torch.einsum(
            "bst,tw->bsw",
            state.type_probabilities,
            self.type_embedding,
        )
        outgoing = state.relations.sum(-1).transpose(1, 2)
        incoming = state.relations.sum(-2).transpose(1, 2)
        relation_context = self.relation_projection(
            torch.cat((incoming, outgoing), dim=-1)
        )
        slots = (
            state.values
            + type_context
            + relation_context
            + self.active_projection(state.active.unsqueeze(-1))
        )
        pooled = (
            slots * state.active.unsqueeze(-1)
        ).sum(1) / state.active.sum(1, keepdim=True).clamp_min(1.0)
        control = (
            self.control_seed.to(slots.dtype).unsqueeze(0)
            + self.step_embedding.weight[state.step].to(slots.dtype)
            + pooled
        )
        encoded = self.core(torch.cat((control[:, None, :], slots), dim=1))
        control = self.output_norm(encoded[:, 0])
        encoded_slots = self.output_norm(encoded[:, 1:])
        keys = self.slot_key(encoded_slots)
        source = torch.einsum(
            "bw,bsw->bs",
            self.source_query(control),
            keys,
        ).float().softmax(-1)
        target = torch.einsum(
            "bw,bsw->bs",
            self.target_query(control),
            keys,
        ).float().softmax(-1)
        opcode = self.opcode_head(control).float().softmax(-1)
        relation = self.relation_head(control).float().softmax(-1)
        type_index = self.type_head(control).float().softmax(-1)
        if hard:
            opcode = _hard_one_hot(opcode)
            source = _hard_one_hot(source)
            target = _hard_one_hot(target)
            relation = _hard_one_hot(relation)
            type_index = _hard_one_hot(type_index)
        return TransactionPolicy(
            opcode=opcode.to(state.values.dtype),
            source=source.to(state.values.dtype),
            target=target.to(state.values.dtype),
            relation=relation.to(state.values.dtype),
            type_index=type_index.to(state.values.dtype),
            payload=self.payload_head(control),
        )

    def apply(
        self,
        state: TypedTheoryState,
        policy: TransactionPolicy,
    ) -> TypedTheoryState:
        opcode = policy.opcode * (1.0 - state.halted[:, None])
        alloc = opcode[:, 0:1] * policy.source
        write = opcode[:, 1:2] * policy.source
        clear = opcode[:, 2:3] * policy.source
        link = opcode[:, 3]
        unlink = opcode[:, 4]
        set_root = opcode[:, 5:6] * policy.source
        commit = opcode[:, 6]
        halt = opcode[:, 7]

        allocated = alloc * (1.0 - state.active)
        cleared = clear * state.active
        active = (
            state.active + allocated * (1.0 - state.active)
        ) * (1.0 - cleared)
        type_write = allocated.unsqueeze(-1)
        type_probabilities = (
            state.type_probabilities * (1.0 - type_write)
            + policy.type_index[:, None, :] * type_write
        )
        value_write = (
            (write * state.active) + allocated
        ).clamp(max=1.0).unsqueeze(-1)
        values = (
            state.values * (1.0 - value_write)
            + policy.payload[:, None, :] * value_write
        )
        values = values * (1.0 - cleared.unsqueeze(-1))

        pair = (
            policy.relation[:, :, None, None]
            * policy.source[:, None, :, None]
            * policy.target[:, None, None, :]
        )
        relations = (
            state.relations + link[:, None, None, None]
            * pair * (1.0 - state.relations)
        )
        relations = relations * (
            1.0 - unlink[:, None, None, None] * pair
        )
        clear_pair = (
            cleared[:, None, :, None]
            + cleared[:, None, None, :]
        ).clamp(max=1.0)
        relations = relations * (1.0 - clear_pair)
        relations = relations * (
            active[:, None, :, None] * active[:, None, None, :]
        )

        root = (
            state.root * (1.0 - set_root.sum(-1, keepdim=True))
            + set_root
        )
        root = root * active
        root = root / root.sum(-1, keepdim=True).clamp_min(1e-6)
        committed = state.committed + (
            1.0 - state.committed
        ) * commit
        halted = state.halted + (1.0 - state.halted) * halt
        result = TypedTheoryState(
            values=values,
            type_probabilities=type_probabilities,
            relations=relations,
            active=active,
            root=root,
            committed=committed,
            halted=halted,
            step=state.step + 1,
        )
        validate_state(result, self.config)
        return result

    def forward(
        self,
        state: TypedTheoryState,
        *,
        steps: int,
        hard: bool = False,
    ) -> tuple[TypedTheoryState, ReactorTrace]:
        if not 1 <= steps <= self.config.max_steps - state.step:
            raise TheoryReactorError("requested reactor steps differ")
        policies: list[TransactionPolicy] = []
        states: list[TypedTheoryState] = []
        for _ in range(steps):
            policy = self.policy(state, hard=hard)
            state = self.apply(state, policy)
            policies.append(policy)
            states.append(state)
        return state, ReactorTrace(
            opcode=torch.stack([item.opcode for item in policies], dim=1),
            source=torch.stack([item.source for item in policies], dim=1),
            target=torch.stack([item.target for item in policies], dim=1),
            relation=torch.stack(
                [item.relation for item in policies],
                dim=1,
            ),
            type_index=torch.stack(
                [item.type_index for item in policies],
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
        )
        self.query_core = nn.TransformerEncoder(
            layer,
            num_layers=config.query_layers,
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(width, config.d_model)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        query_hidden: torch.Tensor,
        state: TypedTheoryState,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, _ = query_hidden.shape
        _padding_mask(
            attention_mask,
            batch,
            tokens,
            query_hidden.device,
        )
        query = self.query_projection(query_hidden)
        state_padding = state.active.lt(0.5)
        empty = state_padding.all(-1)
        if bool(empty.any()):
            state_padding = state_padding.clone()
            state_padding[empty, 0] = False
        read, _ = self.cross_attention(
            query,
            self.state_norm(state.values),
            self.state_norm(state.values),
            key_padding_mask=state_padding,
            need_weights=False,
        )
        query = self.query_core(query + read)
        return torch.tanh(self.gate) * self.output_projection(query)


class EndogenousTypedTheoryReactorGPT(nn.Module):
    """Checkpoint-compatible Shohin plus compiler, reactor, and query reader."""

    def __init__(self, base: GPT, config: TheoryReactorConfig):
        super().__init__()
        config.validate(n_layer=base.cfg.n_layer)
        if base.cfg.d_model != config.d_model:
            raise TheoryReactorError(
                "reactor d_model must match Shohin"
            )
        if base.cfg.n_loop != 1:
            raise TheoryReactorError(
                "reactor reference requires n_loop=1"
            )
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
        base_parameters = sum(
            parameter.numel()
            for parameter in self.base.parameters()
        )
        architecture_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) not in base_ids
        )
        complete = base_parameters + architecture_parameters
        if complete > self.config.parameter_cap:
            raise TheoryReactorError(
                "complete system exceeds parameter cap"
            )
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
    ) -> tuple[TypedTheoryState, ReactorTrace]:
        return self.reactor(state, steps=steps, hard=hard)

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
        targets: torch.Tensor | None = None,
        world_attention_mask: torch.Tensor | None = None,
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
            raise TheoryReactorError(
                "token ids must be a rank-two long tensor"
            )
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
    batch = state.values.shape[0]
    expected = {
        "values": (
            batch,
            config.num_slots,
            config.state_width,
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
            or not bool(torch.isfinite(value).all())
        ):
            raise TheoryReactorError(
                f"state {field.name} differs"
            )
        devices.add(value.device)
    if len(devices) != 1:
        raise TheoryReactorError(
            "state tensors must share one device"
        )


def _padding_mask(
    attention_mask: torch.Tensor | None,
    batch: int,
    tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.shape != (batch, tokens):
        raise TheoryReactorError(
            "attention mask has the wrong shape"
        )
    if attention_mask.dtype != torch.bool:
        if not bool(
            ((attention_mask == 0) | (attention_mask == 1)).all()
        ):
            raise TheoryReactorError(
                "attention mask must be binary"
            )
        attention_mask = attention_mask.bool()
    if bool(
        (
            attention_mask[:, 1:].to(torch.int8)
            > attention_mask[:, :-1].to(torch.int8)
        ).any()
    ):
        raise TheoryReactorError(
            "attention mask must be right padded"
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
    "validate_state",
]
