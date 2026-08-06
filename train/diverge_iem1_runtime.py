"""Integrated learned interfaces for the bounded DIVERGE-IEM1 gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_OFFSET, BYTE_VOCAB_SIZE, CLS_ID, PAD_ID
from diverge_iem1_data import MAX_QUERY_BYTES
from diverge_nve1_data import MAX_EVIDENCE_BYTES, symbol_occurrence_groups
from diverge_nve1_runtime import hard_role_permutation
from diverge_tfs1_runtime import (
    AnchorProvenance,
    CompiledPacket,
    CompiledQuery,
    LocalScorer,
    compile_source,
)
from diverge_tol3_semantic_anchor import (
    COMPARATOR_NAMES,
    OPERATION_NAMES,
    tensorize_texts,
)


SCHEMA = "shohin-diverge-iem1-runtime-v1"
DIRECT_SLICE = slice(1, 5)


class IEM1RuntimeError(RuntimeError):
    """An IEM1 model or compiled interface violates the frozen contract."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def module_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("ascii"))
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class IEM1Config:
    width: int = 192
    layers: int = 2
    max_bytes: int = MAX_EVIDENCE_BYTES
    sinkhorn_iterations: int = 12

    def validate(self) -> None:
        if self.width != 192 or self.layers != 2 or self.width % 2:
            raise IEM1RuntimeError("IEM1 encoder geometry differs")
        if self.max_bytes != MAX_EVIDENCE_BYTES:
            raise IEM1RuntimeError("IEM1 source width differs")
        if self.sinkhorn_iterations != 12:
            raise IEM1RuntimeError("IEM1 transport iterations differ")


def _sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    values = logits.float()
    for _ in range(iterations):
        values = values - torch.logsumexp(values, dim=-1, keepdim=True)
        values = values - torch.logsumexp(values, dim=-2, keepdim=True)
    return values.exp()


class IntegratedEpistemicMachine(nn.Module):
    """One shared encoder for source, evidence, query, and semantic dispatch."""

    def __init__(self, config: IEM1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, config.width)
        self.encoder = nn.GRU(
            input_size=config.width,
            hidden_size=config.width // 2,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.numeric_head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, 2),
        )
        self.symbol_head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, 2),
        )
        self.query_head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, 2),
        )
        self.operation_head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, len(OPERATION_NAMES)),
        )
        self.comparator_head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, len(COMPARATOR_NAMES)),
        )
        self.operation_transport_logits = nn.Parameter(torch.empty(4, 4))
        self.comparator_transport_logits = nn.Parameter(
            torch.empty(len(COMPARATOR_NAMES), len(COMPARATOR_NAMES))
        )
        nn.init.normal_(self.operation_transport_logits, std=0.02)
        nn.init.normal_(self.comparator_transport_logits, std=0.02)

    def _encode(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
            or attention_mask.dtype != torch.bool
        ):
            raise IEM1RuntimeError("IEM1 encoder tensor interface differs")
        lengths = attention_mask.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise IEM1RuntimeError("IEM1 source mask or CLS differs")
        packed = pack_padded_sequence(
            self.embedding(byte_ids),
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=self.config.max_bytes,
        )
        return self.output_norm(hidden)

    @staticmethod
    def _group_pool(hidden: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if (
            masks.ndim != 3
            or masks.shape[0] != hidden.shape[0]
            or masks.shape[2] != hidden.shape[1]
        ):
            raise IEM1RuntimeError("IEM1 mention mask geometry differs")
        counts = masks.sum(dim=-1, keepdim=True)
        if torch.any(counts < 1):
            raise IEM1RuntimeError("IEM1 mention group is empty")
        return torch.einsum("bms,bsw->bmw", masks.to(hidden.dtype), hidden) / counts.to(
            hidden.dtype
        )

    def forward_evidence(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_bounds: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(byte_ids, attention_mask)
        if numeric_bounds.shape != (byte_ids.shape[0], 2, 2):
            raise IEM1RuntimeError("IEM1 numeric mention geometry differs")
        positions = torch.arange(self.config.max_bytes, device=byte_ids.device).view(
            1, 1, -1
        )
        numeric_masks = (positions >= numeric_bounds[:, :, 0].unsqueeze(-1)) & (
            positions < numeric_bounds[:, :, 1].unsqueeze(-1)
        )
        numeric_hidden = self._group_pool(hidden, numeric_masks)
        symbol_hidden = self._group_pool(hidden, symbol_masks)
        return self.numeric_head(numeric_hidden).float(), self.symbol_head(
            symbol_hidden
        ).float()

    def forward_query(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._encode(byte_ids, attention_mask)
        return self.query_head(self._group_pool(hidden, symbol_masks)).float()

    def transports(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            _sinkhorn(
                self.operation_transport_logits,
                self.config.sinkhorn_iterations,
            ),
            _sinkhorn(
                self.comparator_transport_logits,
                self.config.sinkhorn_iterations,
            ),
        )

    def forward_local(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        hard_transport: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(byte_ids, attention_mask)
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        raw_operation = self.operation_head(pooled).float()
        raw_comparator = self.comparator_head(pooled).float()
        operation_transport, comparator_transport = self.transports()

        operation_probability = raw_operation.softmax(-1)
        direct_probability = operation_probability[:, DIRECT_SLICE]
        if hard_transport:
            latent_operation = direct_probability.argmax(-1)
            mapped_direct = operation_transport[
                latent_operation
            ] * direct_probability.sum(-1, keepdim=True)
        else:
            mapped_direct = direct_probability @ operation_transport
        operation_probability = torch.cat(
            (
                operation_probability[:, :1],
                mapped_direct,
                operation_probability[:, 5:],
            ),
            dim=-1,
        )
        raw_comparator_probability = raw_comparator.softmax(-1)
        if hard_transport:
            latent_comparator = raw_comparator_probability.argmax(-1)
            comparator_probability = comparator_transport[latent_comparator]
        else:
            comparator_probability = raw_comparator_probability @ comparator_transport
        return (
            operation_probability.clamp_min(1e-12).log(),
            comparator_probability.clamp_min(1e-12).log(),
        )

    def transport_penalty(self) -> torch.Tensor:
        operation, comparator = self.transports()
        balance = sum(
            (matrix.sum(-1) - 1.0).square().mean()
            + (matrix.sum(-2) - 1.0).square().mean()
            for matrix in (operation, comparator)
        )
        hardness = sum(
            (matrix * (1.0 - matrix)).mean() for matrix in (operation, comparator)
        )
        return balance + 0.05 * hardness

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_bounds: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compatibility with the protected NVE1 sealing adapter.
        return self.forward_evidence(
            byte_ids,
            attention_mask,
            numeric_bounds,
            symbol_masks,
        )

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "config": asdict(self.config)}


def load_nve1_state(
    model: IntegratedEpistemicMachine,
    state: Mapping[str, torch.Tensor],
) -> None:
    required = {
        "embedding.weight",
        *{f"encoder.{name}" for name in model.encoder.state_dict()},
        *{f"output_norm.{name}" for name in model.output_norm.state_dict()},
        *{f"numeric_head.{name}" for name in model.numeric_head.state_dict()},
        *{f"symbol_head.{name}" for name in model.symbol_head.state_dict()},
    }
    missing = sorted(required - set(state))
    if missing:
        raise IEM1RuntimeError(f"NVE1 warm-start state is missing {missing}")
    own = model.state_dict()
    with torch.no_grad():
        for name in required:
            if own[name].shape != state[name].shape:
                raise IEM1RuntimeError(f"NVE1 warm-start tensor {name} differs")
            own[name].copy_(state[name])


def encode_source(text: str, max_bytes: int = MAX_EVIDENCE_BYTES) -> tuple[int, ...]:
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise IEM1RuntimeError("IEM1 source is not ASCII") from error
    if not raw or len(raw) + 1 > max_bytes:
        raise IEM1RuntimeError("IEM1 source width differs")
    return (CLS_ID, *(value + BYTE_OFFSET for value in raw))


def tensorize_queries(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(records)
    byte_ids = torch.full((batch, MAX_QUERY_BYTES), PAD_ID, dtype=torch.long)
    attention = torch.zeros((batch, MAX_QUERY_BYTES), dtype=torch.bool)
    symbol_masks = torch.zeros((batch, 2, MAX_QUERY_BYTES), dtype=torch.bool)
    targets = torch.zeros((batch, 2), dtype=torch.long)
    for row_index, record in enumerate(records):
        text = str(record["source_text"])
        encoded = encode_source(text, MAX_QUERY_BYTES)
        byte_ids[row_index, : len(encoded)] = torch.tensor(encoded)
        attention[row_index, : len(encoded)] = True
        symbols = tuple(str(value) for value in record["symbols"])
        groups = symbol_occurrence_groups(text, symbols)
        if len(groups) != 2:
            raise IEM1RuntimeError("IEM1 query does not expose two symbol groups")
        for group_index, (_, occurrences) in enumerate(groups):
            for start, end in occurrences:
                symbol_masks[row_index, group_index, start + 1 : end + 1] = True
        targets[row_index] = torch.tensor(
            tuple(int(value) for value in record.get("symbol_role_ids", (0, 1)))
        )
    return (
        byte_ids.to(device),
        attention.to(device),
        symbol_masks.to(device),
        targets.to(device),
    )


def tensorize_local_texts(
    texts: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    short_ids, short_mask = tensorize_texts(texts, device)
    ids = torch.full(
        (len(texts), MAX_EVIDENCE_BYTES),
        PAD_ID,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros_like(ids, dtype=torch.bool)
    ids[:, : short_ids.shape[1]] = short_ids
    mask[:, : short_mask.shape[1]] = short_mask
    return ids, mask


class IEM1LocalView:
    """Expose the TOL3 two-output interface without duplicating parameters."""

    def __init__(self, model: IntegratedEpistemicMachine) -> None:
        self.model = model

    def eval(self) -> IEM1LocalView:
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def __call__(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.shape[1] > MAX_EVIDENCE_BYTES:
            raise IEM1RuntimeError("IEM1 local source width differs")
        padded_ids = torch.full(
            (ids.shape[0], MAX_EVIDENCE_BYTES),
            PAD_ID,
            dtype=ids.dtype,
            device=ids.device,
        )
        padded_mask = torch.zeros_like(padded_ids, dtype=torch.bool)
        padded_ids[:, : ids.shape[1]] = ids
        padded_mask[:, : mask.shape[1]] = mask
        return self.model.forward_local(
            padded_ids,
            padded_mask,
            hard_transport=True,
        )


def compile_integrated_source(
    model: IntegratedEpistemicMachine,
    source: str,
    *,
    expected_source_commitment: str,
    compiler_commitment: str,
    device: torch.device,
) -> tuple[CompiledPacket, LocalScorer]:
    return compile_source(
        IEM1LocalView(model),  # type: ignore[arg-type]
        source,
        expected_source_commitment=expected_source_commitment,
        compiler_commitment=compiler_commitment,
        device=device,
    )


@dataclass(frozen=True, slots=True)
class NaturalQueryReceipt:
    packet_commitment: str
    query_source_sha256: str
    compiler_commitment: str
    target: str
    distractor: str
    symbol_provenance: tuple[tuple[str, str, tuple[tuple[int, int], ...]], ...]
    commitment: str

    def payload(self) -> dict[str, object]:
        return {
            "packet_commitment": self.packet_commitment,
            "query_source_sha256": self.query_source_sha256,
            "compiler_commitment": self.compiler_commitment,
            "target": self.target,
            "distractor": self.distractor,
            "symbol_provenance": [
                [role, symbol, [list(span) for span in spans]]
                for role, symbol, spans in self.symbol_provenance
            ],
        }

    def record(self) -> dict[str, object]:
        return {**self.payload(), "commitment": self.commitment}


@dataclass(frozen=True, slots=True)
class QueryCompilation:
    query: CompiledQuery | None
    receipt: NaturalQueryReceipt | None
    error: str | None


def _compile_query_one(
    text: str,
    packet: CompiledPacket,
    assignment: Sequence[int],
    *,
    compiler_commitment: str,
    logits: torch.Tensor,
) -> QueryCompilation:
    try:
        groups = symbol_occurrence_groups(text, packet.symbols)
        if len(groups) != 2 or sorted(int(value) for value in assignment) != [0, 1]:
            raise IEM1RuntimeError("IEM1 query candidate geometry differs")
        roles = {
            ("TARGET", "DISTRACTOR")[int(role)]: group
            for group, role in zip(groups, assignment, strict=True)
        }
        target, target_spans = roles["TARGET"]
        distractor, distractor_spans = roles["DISTRACTOR"]
        query_sha256 = hashlib.sha256(text.encode("ascii")).hexdigest()
        provisional = NaturalQueryReceipt(
            packet.commitment,
            query_sha256,
            compiler_commitment,
            target,
            distractor,
            (
                ("TARGET", target, target_spans),
                ("DISTRACTOR", distractor, distractor_spans),
            ),
            "",
        )
        receipt = replace(
            provisional,
            commitment=canonical_sha256(provisional.payload()),
        )
        target_index = next(
            index for index, role in enumerate(assignment) if int(role) == 0
        )
        target_span = groups[target_index][1][0]
        margin = float(logits[target_index, 0] - logits[target_index, 1])
        query = seal_natural_query(
            packet,
            receipt,
            expected_compiler_commitment=compiler_commitment,
            provenance=AnchorProvenance(
                query_sha256,
                target_span[0],
                target_span[1],
                margin,
            ),
        )
        return QueryCompilation(query, receipt, None)
    except (IEM1RuntimeError, StopIteration, UnicodeEncodeError) as error:
        return QueryCompilation(None, None, str(error))


@torch.no_grad()
def compile_query_batch(
    model: IntegratedEpistemicMachine,
    texts: Sequence[str],
    packets: Sequence[CompiledPacket],
    *,
    compiler_commitment: str,
    device: torch.device,
    swap_roles: bool = False,
) -> tuple[QueryCompilation, ...]:
    if len(texts) != len(packets):
        raise IEM1RuntimeError("IEM1 query batch geometry differs")
    records = [
        {"source_text": text, "symbols": list(packet.symbols)}
        for text, packet in zip(texts, packets, strict=True)
    ]
    ids, mask, groups, _ = tensorize_queries(records, device)
    model.eval()
    logits = model.forward_query(ids, mask, groups)
    output = []
    for row, (text, packet) in enumerate(zip(texts, packets, strict=True)):
        assignment = hard_role_permutation(logits[row])
        if swap_roles:
            assignment = tuple(1 - int(value) for value in assignment)
        output.append(
            _compile_query_one(
                text,
                packet,
                assignment,
                compiler_commitment=compiler_commitment,
                logits=logits[row],
            )
        )
    return tuple(output)


def seal_natural_query(
    packet: CompiledPacket,
    receipt: NaturalQueryReceipt,
    *,
    expected_compiler_commitment: str,
    provenance: AnchorProvenance | None = None,
) -> CompiledQuery:
    if receipt.commitment != canonical_sha256(receipt.payload()):
        raise IEM1RuntimeError("IEM1 query receipt commitment differs")
    if receipt.packet_commitment != packet.commitment:
        raise IEM1RuntimeError("IEM1 query packet commitment differs")
    if receipt.compiler_commitment != expected_compiler_commitment:
        raise IEM1RuntimeError("IEM1 query compiler commitment differs")
    if len(receipt.query_source_sha256) != 64:
        raise IEM1RuntimeError("IEM1 query source commitment width differs")
    if (
        receipt.target not in packet.symbols
        or receipt.distractor not in packet.symbols
        or receipt.target == receipt.distractor
    ):
        raise IEM1RuntimeError("IEM1 query symbol binding differs")
    roles = tuple(value[0] for value in receipt.symbol_provenance)
    symbols = tuple(value[1] for value in receipt.symbol_provenance)
    if roles != ("TARGET", "DISTRACTOR") or symbols != (
        receipt.target,
        receipt.distractor,
    ):
        raise IEM1RuntimeError("IEM1 query provenance roles differ")
    if any(
        not spans or any(start < 0 or end <= start for start, end in spans)
        for _, _, spans in receipt.symbol_provenance
    ):
        raise IEM1RuntimeError("IEM1 query provenance spans differ")
    if provenance is None:
        target_span = receipt.symbol_provenance[0][2][0]
        provenance = AnchorProvenance(
            receipt.query_source_sha256,
            target_span[0],
            target_span[1],
            0.0,
        )
    return CompiledQuery(
        packet.commitment,
        receipt.query_source_sha256,
        receipt.target,
        provenance,
    )


def mutate_query_receipt(
    receipt: NaturalQueryReceipt,
    field: str,
) -> NaturalQueryReceipt:
    if field == "packet":
        return replace(receipt, packet_commitment="0" * 64)
    if field == "source":
        return replace(receipt, query_source_sha256="1" * 64)
    if field == "compiler":
        return replace(receipt, compiler_commitment="2" * 64)
    if field == "target":
        return replace(receipt, target=receipt.distractor)
    if field == "distractor":
        return replace(receipt, distractor=receipt.target)
    if field == "commitment":
        return replace(receipt, commitment="3" * 64)
    raise IEM1RuntimeError("unknown IEM1 query mutation")


__all__ = [
    "IEM1Config",
    "IEM1LocalView",
    "IEM1RuntimeError",
    "IntegratedEpistemicMachine",
    "NaturalQueryReceipt",
    "QueryCompilation",
    "compile_integrated_source",
    "compile_query_batch",
    "load_nve1_state",
    "module_state_sha256",
    "mutate_query_receipt",
    "seal_natural_query",
    "tensorize_local_texts",
    "tensorize_queries",
]
