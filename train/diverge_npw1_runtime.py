"""Document-level event ownership for the DIVERGE-NPW1 WORLD ingress."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_sot1_runtime import SOT1Config, StageOwnedEpistemicMachine


SCHEMA = "shohin-diverge-npw1-runtime-v1"
PAD_ID = 0
CLS_ID = 1
BYTE_OFFSET = 2
BYTE_VOCAB_SIZE = 130
MAX_SOURCE_BYTES = 8192
MAX_CANDIDATES = 384
MAX_EVENTS = 32

FORM_NAMES = ("DIRECT", "SWAP", "GUARD", "AMBIGUOUS", "STOP")
FORM_TO_ID = {name: index for index, name in enumerate(FORM_NAMES)}
ROLE_NAMES = (
    "OPERATION",
    "TARGET",
    "OPERAND",
    "OPTION_A_OPERATION",
    "OPTION_B_OPERATION",
    "LEFT",
    "RIGHT",
    "PRED_LEFT",
    "PRED_RIGHT",
    "TRUE_OPERATION",
    "TRUE_TARGET",
    "TRUE_OPERAND",
    "FALSE_OPERATION",
    "FALSE_TARGET",
    "FALSE_OPERAND",
)
ROLE_TO_ID = {name: index for index, name in enumerate(ROLE_NAMES)}

_CANDIDATE = re.compile(r"-?(?:0|[1-9]\d*)(?:/[1-9]\d*)?|[a-z][a-z0-9_]{1,31}")


class NPW1RuntimeError(RuntimeError):
    """An NPW1 tensor or autonomous event violates the runtime contract."""


@dataclass(frozen=True, slots=True)
class NPW1Config:
    width: int = 256
    layers: int = 2
    max_source_bytes: int = MAX_SOURCE_BYTES
    max_candidates: int = MAX_CANDIDATES
    max_events: int = MAX_EVENTS

    def validate(self) -> None:
        if self.width != 256 or self.layers != 2 or self.width % 2:
            raise NPW1RuntimeError("NPW1 ingress geometry differs")
        if (
            self.max_source_bytes != MAX_SOURCE_BYTES
            or self.max_candidates != MAX_CANDIDATES
            or self.max_events != MAX_EVENTS
        ):
            raise NPW1RuntimeError("NPW1 bounded geometry differs")


@dataclass(frozen=True, slots=True)
class LexicalCandidate:
    text: str
    start: int
    end: int
    kind: int


def lexical_candidates(source: str) -> tuple[LexicalCandidate, ...]:
    try:
        source.encode("ascii")
    except UnicodeEncodeError as error:
        raise NPW1RuntimeError("NPW1 source must be ASCII") from error
    output = []
    for match in _CANDIDATE.finditer(source.lower()):
        text = match.group(0)
        kind = int(text[0].isdigit() or text[0] == "-")
        output.append(LexicalCandidate(text, match.start(), match.end(), kind))
    if not output or len(output) > MAX_CANDIDATES:
        raise NPW1RuntimeError("NPW1 candidate count differs")
    return tuple(output)


def _mention_candidate(
    mention: Mapping[str, object],
    candidates: Sequence[LexicalCandidate],
) -> int:
    start, end = int(mention["start"]), int(mention["end"])
    matches = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.start == start and candidate.end == end
    ]
    if len(matches) != 1:
        raise NPW1RuntimeError("NPW1 labelled mention is absent or ambiguous")
    return matches[0]


def tensorize_records(
    rows: Sequence[Mapping[str, object]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not rows:
        raise NPW1RuntimeError("empty NPW1 batch")
    batch = len(rows)
    byte_ids = torch.full(
        (batch, MAX_SOURCE_BYTES), PAD_ID, dtype=torch.long, device=device
    )
    byte_mask = torch.zeros_like(byte_ids, dtype=torch.bool)
    candidate_masks = torch.zeros(
        (batch, MAX_CANDIDATES, MAX_SOURCE_BYTES),
        dtype=torch.bool,
        device=device,
    )
    candidate_valid = torch.zeros(
        (batch, MAX_CANDIDATES), dtype=torch.bool, device=device
    )
    candidate_kind = torch.zeros(
        (batch, MAX_CANDIDATES), dtype=torch.long, device=device
    )
    form_targets = torch.full(
        (batch, MAX_EVENTS + 1),
        FORM_TO_ID["STOP"],
        dtype=torch.long,
        device=device,
    )
    start_targets = torch.full(
        (batch, MAX_EVENTS), MAX_CANDIDATES, dtype=torch.long, device=device
    )
    end_targets = torch.full_like(start_targets, MAX_CANDIDATES)
    role_targets = torch.full(
        (batch, MAX_EVENTS, len(ROLE_NAMES)),
        MAX_CANDIDATES,
        dtype=torch.long,
        device=device,
    )
    event_mask = torch.zeros(
        (batch, MAX_EVENTS), dtype=torch.bool, device=device
    )
    event_count = torch.zeros(batch, dtype=torch.long, device=device)

    for batch_index, row in enumerate(rows):
        world = row.get("natural_world")
        if not isinstance(world, Mapping):
            raise NPW1RuntimeError("NPW1 natural WORLD is absent")
        source = str(world["source_text"])
        payload = source.encode("ascii")
        encoded = (CLS_ID, *(value + BYTE_OFFSET for value in payload))
        if len(encoded) > MAX_SOURCE_BYTES:
            raise NPW1RuntimeError("NPW1 source exceeds byte bound")
        byte_ids[batch_index, : len(encoded)] = torch.tensor(
            encoded, dtype=torch.long, device=device
        )
        byte_mask[batch_index, : len(encoded)] = True
        candidates = lexical_candidates(source)
        for candidate_index, candidate in enumerate(candidates):
            candidate_valid[batch_index, candidate_index] = True
            candidate_kind[batch_index, candidate_index] = candidate.kind
            candidate_masks[
                batch_index,
                candidate_index,
                candidate.start + 1 : candidate.end + 1,
            ] = True
        events = world["events"]
        if not isinstance(events, list) or len(events) > MAX_EVENTS:
            raise NPW1RuntimeError("NPW1 event count exceeds bound")
        event_count[batch_index] = len(events)
        for event_index, event in enumerate(events):
            event_mask[batch_index, event_index] = True
            form_targets[batch_index, event_index] = FORM_TO_ID[str(event["form"])]
            in_event = [
                index
                for index, candidate in enumerate(candidates)
                if int(event["start"]) <= candidate.start
                and candidate.end <= int(event["end"])
            ]
            if not in_event:
                raise NPW1RuntimeError("NPW1 event has no lexical candidate")
            start_targets[batch_index, event_index] = in_event[0]
            end_targets[batch_index, event_index] = in_event[-1]
            seen_roles = set()
            for mention in event["mentions"]:
                role = str(mention["role"])
                if role == "COMPARATOR":
                    continue
                if role not in ROLE_TO_ID or role in seen_roles:
                    raise NPW1RuntimeError("NPW1 event role is unknown or repeated")
                seen_roles.add(role)
                role_targets[
                    batch_index,
                    event_index,
                    ROLE_TO_ID[role],
                ] = _mention_candidate(mention, candidates)

    return {
        "byte_ids": byte_ids,
        "byte_mask": byte_mask,
        "candidate_masks": candidate_masks,
        "candidate_valid": candidate_valid,
        "candidate_kind": candidate_kind,
        "form_targets": form_targets,
        "start_targets": start_targets,
        "end_targets": end_targets,
        "role_targets": role_targets,
        "event_mask": event_mask,
        "event_count": event_count,
    }


class NarrativeProgramWeaver(nn.Module):
    """Monotone recurrent owner that emits complete source-owned events."""

    def __init__(self, config: NPW1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, width)
        self.encoder = nn.GRU(
            width,
            width // 2,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.norm = nn.LayerNorm(width)
        self.kind_embedding = nn.Embedding(2, width)
        self.candidate_projection = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.null_candidate = nn.Parameter(torch.zeros(width))
        self.initial_state = nn.Parameter(torch.zeros(width))
        self.decoder = nn.GRUCell(width * 2, width)
        self.context_query = nn.Linear(width, width, bias=False)
        self.start_query = nn.Linear(width, width, bias=False)
        self.end_query = nn.Linear(width, width, bias=False)
        self.role_queries = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in ROLE_NAMES
        )
        self.form_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, len(FORM_NAMES)),
        )

    @staticmethod
    def _pointer(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,bcd->bc", query, candidates).float()

    def forward(
        self,
        byte_ids: torch.Tensor,
        byte_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_kind: torch.Tensor,
        *,
        teacher_starts: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, source_width = byte_ids.shape
        if (
            source_width != MAX_SOURCE_BYTES
            or byte_mask.shape != byte_ids.shape
            or candidate_masks.shape != (batch, MAX_CANDIDATES, MAX_SOURCE_BYTES)
            or candidate_valid.shape != (batch, MAX_CANDIDATES)
            or candidate_kind.shape != (batch, MAX_CANDIDATES)
        ):
            raise NPW1RuntimeError("NPW1 tensor geometry differs")
        lengths = byte_mask.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise NPW1RuntimeError("NPW1 source mask differs")
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
            total_length=MAX_SOURCE_BYTES,
        )
        hidden = self.norm(hidden)
        weights = candidate_masks.to(hidden.dtype)
        candidates = torch.einsum("bcs,bsd->bcd", weights, hidden)
        candidates = candidates / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        candidates = self.candidate_projection(
            candidates + self.kind_embedding(candidate_kind)
        )
        null = self.null_candidate.view(1, 1, -1).expand(batch, 1, -1)
        candidates = torch.cat((candidates, null), dim=1)
        valid = torch.cat(
            (
                candidate_valid,
                torch.ones(batch, 1, dtype=torch.bool, device=byte_ids.device),
            ),
            dim=1,
        )
        state = self.initial_state.view(1, -1).expand(batch, -1)
        document = (
            hidden * byte_mask.to(hidden.dtype).unsqueeze(-1)
        ).sum(dim=1) / byte_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        forms = []
        starts = []
        ends = []
        roles = []
        for event_index in range(MAX_EVENTS + 1):
            context_logits = self._pointer(self.context_query(state), candidates)
            context_logits = context_logits.masked_fill(~valid, float("-inf"))
            context = torch.einsum(
                "bc,bcd->bd", context_logits.softmax(dim=-1), candidates
            )
            state = self.decoder(torch.cat((context, document), dim=-1), state)
            forms.append(self.form_head(state))
            if event_index == MAX_EVENTS:
                continue
            start_logits = self._pointer(self.start_query(state), candidates)
            end_logits = self._pointer(self.end_query(state), candidates)
            start_logits = start_logits.masked_fill(~valid, float("-inf"))
            end_logits = end_logits.masked_fill(~valid, float("-inf"))
            starts.append(start_logits)
            ends.append(end_logits)
            roles.append(
                torch.stack(
                    [
                        self._pointer(head(state), candidates).masked_fill(
                            ~valid, float("-inf")
                        )
                        for head in self.role_queries
                    ],
                    dim=1,
                )
            )
            if teacher_starts is not None:
                selected = teacher_starts[:, event_index].clamp_max(MAX_CANDIDATES)
            else:
                selected = start_logits.argmax(dim=-1)
            state = state + candidates[
                torch.arange(batch, device=byte_ids.device), selected
            ]
        return {
            "form_logits": torch.stack(forms, dim=1),
            "start_logits": torch.stack(starts, dim=1),
            "end_logits": torch.stack(ends, dim=1),
            "role_logits": torch.stack(roles, dim=1),
        }


class NarrativeStageOwnedMachine(nn.Module):
    """SOT1 composite plus one isolated narrative WORLD ingress owner."""

    def __init__(self, sot1_config: SOT1Config, npw1_config: NPW1Config) -> None:
        super().__init__()
        self.sot1 = StageOwnedEpistemicMachine(sot1_config)
        self.world_ingress = NarrativeProgramWeaver(npw1_config)

    def freeze_inherited_owners(self) -> None:
        self.sot1.requires_grad_(False)
        self.world_ingress.requires_grad_(True)

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "sot1": self.sot1.owner_manifest(),
            "npw1": asdict(self.world_ingress.config),
        }


__all__ = [
    "FORM_NAMES",
    "FORM_TO_ID",
    "MAX_CANDIDATES",
    "MAX_EVENTS",
    "MAX_SOURCE_BYTES",
    "NPW1Config",
    "NPW1RuntimeError",
    "NarrativeProgramWeaver",
    "NarrativeStageOwnedMachine",
    "ROLE_NAMES",
    "ROLE_TO_ID",
    "SCHEMA",
    "lexical_candidates",
    "tensorize_records",
]
