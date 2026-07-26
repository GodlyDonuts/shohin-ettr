"""Source-deleted causal qualification controls for a trained ETTR model.

This module is assessor-side infrastructure. It never constructs semantic
answers or executes a task-family oracle. Model forwards receive only a
deployed terminal packet and a physically truncated query prefix. Targets and
factor identities are consulted only after every readout is sealed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import re

import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
    validate_deployed_state,
)


ETTR_QUALIFICATION_SCHEMA = "shohin-ettr-causal-qualification-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _tensor_receipt(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": digest.hexdigest(),
    }


def typed_state_row_sha256(state: TypedTheoryState, row: int) -> str:
    """Hash every deployed packet field for one row."""

    batch = state.active.shape[0]
    if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < batch:
        raise TheoryReactorError("qualification state row leaves the batch")
    payload: dict[str, object] = {"step": state.step}
    for item in fields(state):
        if item.name == "step":
            continue
        payload[item.name] = _tensor_receipt(getattr(state, item.name)[row])
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _typed_state_receipt(state: TypedTheoryState) -> dict[str, object]:
    payload: dict[str, object] = {"step": state.step}
    for item in fields(state):
        if item.name == "step":
            continue
        payload[item.name] = _tensor_receipt(getattr(state, item.name))
    return payload


def _index_state(
    state: TypedTheoryState,
    index: torch.Tensor,
) -> TypedTheoryState:
    return TypedTheoryState(
        value_probabilities=state.value_probabilities.index_select(0, index),
        type_probabilities=state.type_probabilities.index_select(0, index),
        relations=state.relations.index_select(0, index),
        active=state.active.index_select(0, index),
        root=state.root.index_select(0, index),
        committed=state.committed.index_select(0, index),
        halted=state.halted.index_select(0, index),
        step=state.step,
    )


def _empty_state(state: TypedTheoryState) -> TypedTheoryState:
    """Return the canonical query-only packet on the same device and dtype."""

    return TypedTheoryState(
        value_probabilities=torch.zeros_like(state.value_probabilities),
        type_probabilities=torch.zeros_like(state.type_probabilities),
        relations=torch.zeros_like(state.relations),
        active=torch.zeros_like(state.active),
        root=torch.zeros_like(state.root),
        committed=torch.zeros_like(state.committed),
        halted=torch.zeros_like(state.halted),
        step=state.step,
    )


def _validate_identifier_tuple(
    values: tuple[str, ...],
    *,
    rows: int,
    name: str,
) -> None:
    if (
        not isinstance(values, tuple)
        or len(values) != rows
        or any(_SHA256.fullmatch(value) is None for value in values)
    ):
        raise TheoryReactorError(
            f"qualification {name} identities differ"
        )


def _prefix_bytes(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    read_index: torch.Tensor,
    row: int,
) -> bytes:
    stop = int(read_index[row]) + 1
    payload = {
        "mask": mask[row, :stop].detach().cpu().bool().tolist(),
        "tokens": tokens[row, :stop].detach().cpu().tolist(),
    }
    return _canonical_json_bytes(payload)


@dataclass(frozen=True, slots=True)
class ETTRQualificationBatch:
    """Immutable treatment rows plus assessor-only matched-control indices."""

    terminal_state: TypedTheoryState
    query_tokens: torch.Tensor
    query_attention_mask: torch.Tensor
    query_read_index: torch.Tensor
    targets: torch.Tensor
    packet_ids: tuple[str, ...]
    world_factor_ids: tuple[str, ...]
    command_factor_ids: tuple[str, ...]
    query_semantic_ids: tuple[str, ...]
    query_paraphrase_ids: tuple[str, ...]
    shuffled_state_index: torch.Tensor
    wrong_world_state_index: torch.Tensor
    wrong_command_state_index: torch.Tensor
    query_twin_index: torch.Tensor
    target_derangement_index: torch.Tensor

    def sha256(self) -> str:
        payload = {
            "command_factor_ids": list(self.command_factor_ids),
            "packet_ids": list(self.packet_ids),
            "query_attention_mask": _tensor_receipt(
                self.query_attention_mask
            ),
            "query_paraphrase_ids": list(self.query_paraphrase_ids),
            "query_read_index": _tensor_receipt(self.query_read_index),
            "query_semantic_ids": list(self.query_semantic_ids),
            "query_tokens": _tensor_receipt(self.query_tokens),
            "query_twin_index": _tensor_receipt(self.query_twin_index),
            "shuffled_state_index": _tensor_receipt(
                self.shuffled_state_index
            ),
            "target_derangement_index": _tensor_receipt(
                self.target_derangement_index
            ),
            "targets": _tensor_receipt(self.targets),
            "terminal_state": _typed_state_receipt(self.terminal_state),
            "wrong_command_state_index": _tensor_receipt(
                self.wrong_command_state_index
            ),
            "wrong_world_state_index": _tensor_receipt(
                self.wrong_world_state_index
            ),
            "world_factor_ids": list(self.world_factor_ids),
        }
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()

    def validate(
        self,
        config: TheoryReactorConfig,
        *,
        vocab_size: int,
    ) -> None:
        validate_deployed_state(self.terminal_state, config)
        rows = self.terminal_state.active.shape[0]
        tokens = self.query_tokens
        mask = self.query_attention_mask
        read = self.query_read_index
        if (
            rows < 16
            or tokens.ndim != 2
            or tokens.dtype != torch.long
            or tokens.shape[0] != rows
            or tokens.shape[1] < 2
            or mask.shape != tokens.shape
            or mask.dtype != torch.bool
            or read.shape != (rows,)
            or read.dtype != torch.long
            or self.targets.shape != (rows,)
            or self.targets.dtype != torch.long
            or not isinstance(vocab_size, int)
            or isinstance(vocab_size, bool)
            or vocab_size < 2
        ):
            raise TheoryReactorError(
                "qualification query or target geometry differs"
            )
        devices = {
            tokens.device,
            mask.device,
            read.device,
            self.targets.device,
            self.terminal_state.active.device,
        }
        if len(devices) != 1:
            raise TheoryReactorError(
                "qualification tensors must share one device"
            )
        if (
            not bool(((tokens >= 0) & (tokens < vocab_size)).all())
            or not bool(
                ((self.targets >= 0) & (self.targets < vocab_size)).all()
            )
            or bool((mask[:, 1:].to(torch.int8) > mask[:, :-1]).any())
            or not bool(
                (
                    (read >= 0)
                    & (read < tokens.shape[1] - 1)
                ).all()
            )
        ):
            raise TheoryReactorError(
                "qualification query values leave their declared range"
            )
        row = torch.arange(rows, device=tokens.device)
        if not bool(mask[row, read].all()) or not bool(
            mask[row, read + 1].all()
        ):
            raise TheoryReactorError(
                "qualification read and target positions must be visible"
            )
        if not torch.equal(self.targets, tokens[row, read + 1]):
            raise TheoryReactorError(
                "qualification targets must be the causal next token"
            )

        identifiers = (
            ("packet", self.packet_ids),
            ("world factor", self.world_factor_ids),
            ("command factor", self.command_factor_ids),
            ("query semantic", self.query_semantic_ids),
            ("query paraphrase", self.query_paraphrase_ids),
        )
        for name, values in identifiers:
            _validate_identifier_tuple(values, rows=rows, name=name)
        expected_packets = tuple(
            typed_state_row_sha256(self.terminal_state, index)
            for index in range(rows)
        )
        if self.packet_ids != expected_packets:
            raise TheoryReactorError(
                "qualification packet identities do not bind packet bytes"
            )

        indices = (
            ("shuffled state", self.shuffled_state_index),
            ("wrong WORLD state", self.wrong_world_state_index),
            ("wrong COMMAND state", self.wrong_command_state_index),
            ("query twin", self.query_twin_index),
            ("target derangement", self.target_derangement_index),
        )
        canonical = list(range(rows))
        for name, index in indices:
            if (
                index.shape != (rows,)
                or index.dtype != torch.long
                or index.device != tokens.device
                or sorted(index.detach().cpu().tolist()) != canonical
                or bool((index == row).any())
            ):
                raise TheoryReactorError(
                    f"qualification {name} index is not a derangement"
                )

        self._validate_packet_groups()
        self._validate_state_controls()
        self._validate_query_twins()
        deranged_targets = self.targets.index_select(
            0,
            self.target_derangement_index,
        )
        if bool((deranged_targets == self.targets).any()):
            raise TheoryReactorError(
                "qualification target derangement has a fixed label"
            )

    def _validate_packet_groups(self) -> None:
        groups: dict[str, list[int]] = {}
        for row, packet in enumerate(self.packet_ids):
            groups.setdefault(packet, []).append(row)
        if len(groups) < 4:
            raise TheoryReactorError(
                "qualification needs at least four distinct packets"
            )
        factor_to_packet: dict[tuple[str, str], str] = {}
        for packet, rows in groups.items():
            factors = {
                (self.world_factor_ids[row], self.command_factor_ids[row])
                for row in rows
            }
            if len(factors) != 1:
                raise TheoryReactorError(
                    "qualification packet has inconsistent factor identity"
                )
            factor = next(iter(factors))
            prior = factor_to_packet.setdefault(factor, packet)
            if prior != packet:
                raise TheoryReactorError(
                    "qualification factor pair maps to multiple packets"
                )

            semantics: dict[str, list[int]] = {}
            for row in rows:
                semantics.setdefault(
                    self.query_semantic_ids[row],
                    [],
                ).append(row)
            if len(semantics) < 2:
                raise TheoryReactorError(
                    "qualification packet lacks independent query semantics"
                )
            semantic_targets: set[int] = set()
            for semantic_rows in semantics.values():
                paraphrases = {
                    self.query_paraphrase_ids[row]
                    for row in semantic_rows
                }
                targets = {
                    int(self.targets[row])
                    for row in semantic_rows
                }
                prefixes = {
                    _prefix_bytes(
                        self.query_tokens,
                        self.query_attention_mask,
                        self.query_read_index,
                        row,
                    )
                    for row in semantic_rows
                }
                if (
                    len(paraphrases) < 2
                    or len(targets) != 1
                    or len(prefixes) != len(semantic_rows)
                ):
                    raise TheoryReactorError(
                        "qualification query paraphrase twin differs"
                    )
                semantic_targets.update(targets)
            if len(semantic_targets) < 2:
                raise TheoryReactorError(
                    "qualification query semantics lack target contrast"
                )

    def _validate_state_controls(self) -> None:
        controls = (
            (
                "shuffled state",
                self.shuffled_state_index,
                None,
            ),
            (
                "wrong WORLD state",
                self.wrong_world_state_index,
                "world",
            ),
            (
                "wrong COMMAND state",
                self.wrong_command_state_index,
                "command",
            ),
        )
        for name, index, factor in controls:
            for row, donor in enumerate(index.detach().cpu().tolist()):
                if (
                    self.packet_ids[row] == self.packet_ids[donor]
                    or self.query_semantic_ids[row]
                    != self.query_semantic_ids[donor]
                    or self.query_paraphrase_ids[row]
                    != self.query_paraphrase_ids[donor]
                    or int(self.query_read_index[row])
                    != int(self.query_read_index[donor])
                    or _prefix_bytes(
                        self.query_tokens,
                        self.query_attention_mask,
                        self.query_read_index,
                        row,
                    )
                    != _prefix_bytes(
                        self.query_tokens,
                        self.query_attention_mask,
                        self.query_read_index,
                        donor,
                    )
                ):
                    raise TheoryReactorError(
                        f"qualification {name} does not isolate packet state"
                    )
                world_same = (
                    self.world_factor_ids[row]
                    == self.world_factor_ids[donor]
                )
                command_same = (
                    self.command_factor_ids[row]
                    == self.command_factor_ids[donor]
                )
                if factor == "world" and (
                    world_same or not command_same
                ):
                    raise TheoryReactorError(
                        "qualification wrong WORLD state is not factorial"
                    )
                if factor == "command" and (
                    not world_same or command_same
                ):
                    raise TheoryReactorError(
                        "qualification wrong COMMAND state is not factorial"
                    )

    def _validate_query_twins(self) -> None:
        for row, donor in enumerate(
            self.query_twin_index.detach().cpu().tolist()
        ):
            if (
                self.packet_ids[row] != self.packet_ids[donor]
                or self.world_factor_ids[row]
                != self.world_factor_ids[donor]
                or self.command_factor_ids[row]
                != self.command_factor_ids[donor]
                or self.query_semantic_ids[row]
                == self.query_semantic_ids[donor]
                or self.query_paraphrase_ids[row]
                != self.query_paraphrase_ids[donor]
                or int(self.query_read_index[row])
                != int(self.query_read_index[donor])
                or int(self.targets[row]) == int(self.targets[donor])
                or _prefix_bytes(
                    self.query_tokens,
                    self.query_attention_mask,
                    self.query_read_index,
                    row,
                )
                == _prefix_bytes(
                    self.query_tokens,
                    self.query_attention_mask,
                    self.query_read_index,
                    donor,
                )
            ):
                raise TheoryReactorError(
                    "qualification query twin is not an independent query"
                )


@dataclass(frozen=True, slots=True)
class ETTRQualificationReadouts:
    """Sealed read-position logits; targets were absent from every forward."""

    treatment: torch.Tensor
    query_only: torch.Tensor
    zero_reader: torch.Tensor
    shuffled_state: torch.Tensor
    wrong_world_state: torch.Tensor
    wrong_command_state: torch.Tensor
    query_twin: torch.Tensor
    targets: torch.Tensor
    query_twin_targets: torch.Tensor
    deranged_targets: torch.Tensor
    batch_sha256: str

    def validate(self, *, rows: int, vocab_size: int) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "batch_sha256":
                if (
                    not isinstance(value, str)
                    or _SHA256.fullmatch(value) is None
                ):
                    raise TheoryReactorError(
                        "qualification batch SHA-256 differs"
                    )
                continue
            expected = (
                (rows,)
                if item.name.endswith("targets") or item.name == "targets"
                else (rows, vocab_size)
            )
            if (
                not torch.is_tensor(value)
                or value.shape != expected
                or (
                    len(expected) == 2
                    and (
                        not value.is_floating_point()
                        or not bool(torch.isfinite(value).all())
                    )
                )
                or (
                    len(expected) == 1
                    and value.dtype != torch.long
                )
            ):
                raise TheoryReactorError(
                    f"qualification {item.name} readout differs"
                )


@dataclass(frozen=True, slots=True)
class ETTRQualificationScore:
    schema: str
    rows: int
    packet_groups: int
    treatment_exact: int
    query_only_exact: int
    zero_reader_exact: int
    shuffled_state_exact: int
    wrong_world_state_exact: int
    wrong_command_state_exact: int
    wrong_query_exact: int
    target_deranged_exact: int
    query_twin_exact: int
    packet_groups_all_exact: int
    causal_packet_effect_rows: int
    query_sensitivity_rows: int

    @property
    def strongest_negative_control_exact(self) -> int:
        return max(
            self.query_only_exact,
            self.zero_reader_exact,
            self.shuffled_state_exact,
            self.wrong_world_state_exact,
            self.wrong_command_state_exact,
            self.wrong_query_exact,
            self.target_deranged_exact,
        )


class ETTRQualificationHarness:
    """Execute all source-deleted controls from one immutable batch."""

    def __init__(self, model: EndogenousTypedTheoryReactorGPT):
        self.model = model

    @torch.no_grad()
    def run(
        self,
        batch: ETTRQualificationBatch,
    ) -> ETTRQualificationReadouts:
        if self.model.training:
            raise TheoryReactorError(
                "qualification requires eval mode"
            )
        vocab_size = self.model.base.cfg.vocab_size
        batch.validate(self.model.config, vocab_size=vocab_size)
        tokens, mask = _autonomous_prefix(
            batch.query_tokens,
            batch.query_attention_mask,
            batch.query_read_index,
        )
        treatment = self._read(
            batch.terminal_state,
            tokens,
            mask,
            batch.query_read_index,
        )
        query_only = self._read(
            _empty_state(batch.terminal_state),
            tokens,
            mask,
            batch.query_read_index,
        )
        zero_reader = self._read_without_reader(
            tokens,
            mask,
            batch.query_read_index,
        )
        shuffled_state = self._read(
            _index_state(
                batch.terminal_state,
                batch.shuffled_state_index,
            ),
            tokens,
            mask,
            batch.query_read_index,
        )
        wrong_world_state = self._read(
            _index_state(
                batch.terminal_state,
                batch.wrong_world_state_index,
            ),
            tokens,
            mask,
            batch.query_read_index,
        )
        wrong_command_state = self._read(
            _index_state(
                batch.terminal_state,
                batch.wrong_command_state_index,
            ),
            tokens,
            mask,
            batch.query_read_index,
        )
        twin = batch.query_twin_index
        twin_tokens, twin_mask = _autonomous_prefix(
            batch.query_tokens.index_select(0, twin),
            batch.query_attention_mask.index_select(0, twin),
            batch.query_read_index.index_select(0, twin),
        )
        query_twin = self._read(
            batch.terminal_state,
            twin_tokens,
            twin_mask,
            batch.query_read_index.index_select(0, twin),
        )
        result = ETTRQualificationReadouts(
            treatment=treatment,
            query_only=query_only,
            zero_reader=zero_reader,
            shuffled_state=shuffled_state,
            wrong_world_state=wrong_world_state,
            wrong_command_state=wrong_command_state,
            query_twin=query_twin,
            targets=batch.targets.detach().clone(),
            query_twin_targets=batch.targets.index_select(0, twin).detach(),
            deranged_targets=batch.targets.index_select(
                0,
                batch.target_derangement_index,
            ).detach(),
            batch_sha256=batch.sha256(),
        )
        result.validate(
            rows=batch.targets.shape[0],
            vocab_size=vocab_size,
        )
        return result

    def _read(
        self,
        state: TypedTheoryState,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        read_index: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.model.answer_query(
            state,
            tokens,
            targets=None,
            attention_mask=mask,
        )
        return _gather_read_logits(logits, read_index)

    def _read_without_reader(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        read_index: torch.Tensor,
    ) -> torch.Tensor:
        del mask
        hidden = self.model._encode_to_stage(tokens, pos=0)
        hidden = self.model._decode_from_stage(hidden, pos=0)
        logits = self.model.base.head(self.model.base.norm(hidden))
        return _gather_read_logits(logits, read_index)


def _autonomous_prefix(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    read_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(
        tokens.shape[1],
        device=tokens.device,
    )[None, :]
    visible = positions <= read_index[:, None]
    prefix_mask = mask & visible
    prefix_tokens = torch.where(
        prefix_mask,
        tokens,
        torch.zeros_like(tokens),
    )
    return prefix_tokens, prefix_mask


def _gather_read_logits(
    logits: torch.Tensor,
    read_index: torch.Tensor,
) -> torch.Tensor:
    return logits.gather(
        1,
        read_index[:, None, None].expand(-1, 1, logits.shape[-1]),
    ).squeeze(1)


def score_ettr_qualification(
    batch: ETTRQualificationBatch,
    readouts: ETTRQualificationReadouts,
) -> ETTRQualificationScore:
    """Score sealed readouts after every model forward has completed."""

    rows = batch.targets.shape[0]
    vocab_size = readouts.treatment.shape[-1]
    readouts.validate(rows=rows, vocab_size=vocab_size)
    if readouts.batch_sha256 != batch.sha256():
        raise TheoryReactorError(
            "qualification readouts belong to another batch"
        )
    expected_targets = (
        ("targets", batch.targets),
        (
            "query twin targets",
            batch.targets.index_select(0, batch.query_twin_index),
        ),
        (
            "deranged targets",
            batch.targets.index_select(
                0,
                batch.target_derangement_index,
            ),
        ),
    )
    for name, expected in expected_targets:
        observed = getattr(
            readouts,
            name.replace(" ", "_"),
        )
        if not torch.equal(observed, expected):
            raise TheoryReactorError(
                f"qualification {name} changed after forward"
            )

    def exact(logits: torch.Tensor, targets: torch.Tensor) -> int:
        return int(logits.argmax(-1).eq(targets).sum().detach().cpu())

    treatment_prediction = readouts.treatment.argmax(-1)
    packet_effect = (
        treatment_prediction.ne(readouts.shuffled_state.argmax(-1))
        & treatment_prediction.ne(
            readouts.wrong_world_state.argmax(-1)
        )
        & treatment_prediction.ne(
            readouts.wrong_command_state.argmax(-1)
        )
    )
    query_sensitivity = treatment_prediction.ne(
        readouts.query_twin.argmax(-1)
    )
    packet_groups: dict[str, list[int]] = {}
    for row, packet in enumerate(batch.packet_ids):
        packet_groups.setdefault(packet, []).append(row)
    treatment_correct = treatment_prediction.eq(readouts.targets)
    groups_exact = sum(
        int(bool(treatment_correct[indices].all()))
        for indices in packet_groups.values()
    )
    return ETTRQualificationScore(
        schema=ETTR_QUALIFICATION_SCHEMA,
        rows=rows,
        packet_groups=len(packet_groups),
        treatment_exact=exact(
            readouts.treatment,
            readouts.targets,
        ),
        query_only_exact=exact(
            readouts.query_only,
            readouts.targets,
        ),
        zero_reader_exact=exact(
            readouts.zero_reader,
            readouts.targets,
        ),
        shuffled_state_exact=exact(
            readouts.shuffled_state,
            readouts.targets,
        ),
        wrong_world_state_exact=exact(
            readouts.wrong_world_state,
            readouts.targets,
        ),
        wrong_command_state_exact=exact(
            readouts.wrong_command_state,
            readouts.targets,
        ),
        wrong_query_exact=exact(
            readouts.query_twin,
            readouts.targets,
        ),
        target_deranged_exact=exact(
            readouts.treatment,
            readouts.deranged_targets,
        ),
        query_twin_exact=exact(
            readouts.query_twin,
            readouts.query_twin_targets,
        ),
        packet_groups_all_exact=groups_exact,
        causal_packet_effect_rows=int(
            packet_effect.sum().detach().cpu()
        ),
        query_sensitivity_rows=int(
            query_sensitivity.sum().detach().cpu()
        ),
    )


__all__ = [
    "ETTRQualificationBatch",
    "ETTRQualificationHarness",
    "ETTRQualificationReadouts",
    "ETTRQualificationScore",
    "ETTR_QUALIFICATION_SCHEMA",
    "score_ettr_qualification",
    "typed_state_row_sha256",
]
