"""Zero-parameter source-law residuals for anonymous EFC machines.

The current identifiable board states that each action is a permutation and
each observer emits every answer exactly twice.  This module evaluates finite
hidden-row candidates against those public laws.  It never accepts a target
machine, supervisor, query, or answer label.
"""

from __future__ import annotations

import random
import secrets
from dataclasses import dataclass
from hashlib import sha256

import torch

from episode_functor_runtime_constants import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)


ANSWER_MULTIPLICITY = PRIMARY_STATES // PRIMARY_ANSWERS
EVIDENCE_CAPABILITY_SCHEMA = "shohin.efc.source-law-evidence.v1"
CONTROL_CAPABILITY_SCHEMA = "shohin.efc.source-law-control.v1"
CONTROL_SEED_SHA256 = sha256(
    b"R12-EFC-SLRA-CONTROL-SEED-v1"
).hexdigest()


class SourceLawResidualError(ValueError):
    """Source-law tensors or the unique-completion contract differ."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    for dimension in contiguous.shape:
        digest.update(int(dimension).to_bytes(8, "little"))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class SourceLawEvidenceCapability:
    """Opaque identity for one issuer-owned source batch."""

    schema: str
    source_sha256: tuple[str, ...]
    issuer_nonce: str
    capability: object

    def __post_init__(self) -> None:
        if (
            self.schema != EVIDENCE_CAPABILITY_SCHEMA
            or not self.source_sha256
            or any(not _is_sha256(value) for value in self.source_sha256)
            or not _is_sha256(self.issuer_nonce)
            or type(self.capability) is not object
        ):
            raise SourceLawResidualError(
                "source-law evidence capability differs"
            )


@dataclass(frozen=True, slots=True, eq=False)
class SourceLawControlCapability:
    """Opaque identity for a source-bound precommitted derangement."""

    schema: str
    source_sha256: tuple[str, ...]
    control_seed_sha256: str
    issuer_nonce: str
    capability: object

    def __post_init__(self) -> None:
        if (
            self.schema != CONTROL_CAPABILITY_SCHEMA
            or not self.source_sha256
            or any(not _is_sha256(value) for value in self.source_sha256)
            or not _is_sha256(self.control_seed_sha256)
            or not _is_sha256(self.issuer_nonce)
            or type(self.capability) is not object
        ):
            raise SourceLawResidualError(
                "source-law control capability differs"
            )


@dataclass(frozen=True, slots=True)
class _IssuedEvidence:
    capability: SourceLawEvidenceCapability
    source_sha256: tuple[str, ...]
    issuer_nonce: str
    sources: tuple[bytes, ...]
    transition: torch.Tensor
    observer: torch.Tensor
    transition_visible_rows: torch.Tensor
    observer_visible_rows: torch.Tensor
    tensor_sha256: tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class _IssuedControl:
    capability: SourceLawControlCapability
    evidence_capability: SourceLawEvidenceCapability
    source_sha256: tuple[str, ...]
    control_seed_sha256: str
    issuer_nonce: str
    transition_transport: torch.Tensor
    observer_transport: torch.Tensor
    tensor_sha256: tuple[str, str]


@dataclass(frozen=True, slots=True)
class SourceLawResidualResult:
    """Candidate law violations and compiler-owned hidden-row masks."""

    transition_residuals: torch.Tensor
    observer_residuals: torch.Tensor
    transition_hidden_rows: torch.Tensor
    observer_hidden_rows: torch.Tensor

    def __post_init__(self) -> None:
        batch = int(self.transition_residuals.shape[0])
        if (
            self.transition_residuals.shape
            != (
                batch,
                PRIMARY_ACTIONS,
                PRIMARY_STATES,
                PRIMARY_STATES,
            )
            or self.observer_residuals.shape
            != (
                batch,
                PRIMARY_OBSERVERS,
                PRIMARY_STATES,
                PRIMARY_ANSWERS,
            )
            or self.transition_hidden_rows.shape
            != (batch, PRIMARY_ACTIONS, PRIMARY_STATES)
            or self.observer_hidden_rows.shape
            != (batch, PRIMARY_OBSERVERS, PRIMARY_STATES)
            or self.transition_hidden_rows.dtype != torch.bool
            or self.observer_hidden_rows.dtype != torch.bool
            or not self.transition_residuals.is_floating_point()
            or not self.observer_residuals.is_floating_point()
            or not bool(torch.isfinite(self.transition_residuals).all())
            or not bool(torch.isfinite(self.observer_residuals).all())
            or bool(self.transition_residuals.lt(0).any())
            or bool(self.observer_residuals.lt(0).any())
        ):
            raise SourceLawResidualError(
                "source-law residual result differs"
            )


def _validate_visible(
    name: str,
    values: torch.Tensor,
    visible_rows: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if (
        values.shape != shape
        or not values.is_floating_point()
        or not bool(torch.isfinite(values).all())
        or bool(values.lt(0).any())
        or visible_rows.shape != shape[:-1]
        or visible_rows.dtype != torch.bool
        or visible_rows.device != values.device
    ):
        raise SourceLawResidualError(f"{name} differs")
    row_mass = values.sum(-1)
    if (
        not torch.equal(row_mass.gt(0), visible_rows)
        or bool(values.logical_and(values.ne(1)).any())
        or bool(
            (
                row_mass[visible_rows] - 1.0
            ).abs().gt(2e-6).any()
        )
        or bool(values[~visible_rows].ne(0).any())
    ):
        raise SourceLawResidualError(
            f"{name} visibility does not match row mass"
        )
    hidden_per_relation = (~visible_rows).sum(-1)
    if not bool(hidden_per_relation.eq(1).all()):
        raise SourceLawResidualError(
            f"{name} does not hide exactly one row per relation"
        )
    return values * visible_rows[..., None].to(values.dtype)


def _source_law_residuals_from_visible(
    transition_visible: torch.Tensor,
    observer_visible: torch.Tensor,
    transition_visible_rows: torch.Tensor,
    observer_visible_rows: torch.Tensor,
) -> SourceLawResidualResult:
    """Enumerate hidden-row candidates under source-declared laws."""

    batch = int(transition_visible.shape[0])
    transition_shape = (
        batch,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
    )
    observer_shape = (
        batch,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
    )
    transition = _validate_visible(
        "transition source law",
        transition_visible,
        transition_visible_rows,
        transition_shape,
    )
    observer = _validate_visible(
        "observer source law",
        observer_visible,
        observer_visible_rows,
        observer_shape,
    )

    state_eye = torch.eye(
        PRIMARY_STATES,
        dtype=transition.dtype,
        device=transition.device,
    )
    transition_interventions = torch.einsum(
        "rs,cd->rcsd",
        state_eye,
        state_eye,
    )
    transition_candidates = (
        transition[:, :, None, None]
        + (~transition_visible_rows)[..., None, None, None].to(
            transition.dtype
        )
        * transition_interventions[None, None]
    )
    transition_row_error = (
        transition_candidates.sum(-1) - 1.0
    ).square().sum(-1)
    transition_column_error = (
        transition_candidates.sum(-2) - 1.0
    ).square().sum(-1)
    transition_residual = (
        transition_row_error + transition_column_error
    )

    answer_eye = torch.eye(
        PRIMARY_ANSWERS,
        dtype=observer.dtype,
        device=observer.device,
    )
    observer_row_eye = torch.eye(
        PRIMARY_STATES,
        dtype=observer.dtype,
        device=observer.device,
    )
    observer_interventions = torch.einsum(
        "rs,cy->rcsy",
        observer_row_eye,
        answer_eye,
    )
    observer_candidates = (
        observer[:, :, None, None]
        + (~observer_visible_rows)[..., None, None, None].to(observer.dtype)
        * observer_interventions[None, None]
    )
    observer_row_error = (
        observer_candidates.sum(-1) - 1.0
    ).square().sum(-1)
    observer_column_error = (
        observer_candidates.sum(-2) - float(ANSWER_MULTIPLICITY)
    ).square().sum(-1)
    observer_residual = observer_row_error + observer_column_error

    return SourceLawResidualResult(
        transition_residuals=transition_residual,
        observer_residuals=observer_residual,
        transition_hidden_rows=~transition_visible_rows,
        observer_hidden_rows=~observer_visible_rows,
    )


def _complete_from_visible_source_law(
    transition_visible: torch.Tensor,
    observer_visible: torch.Tensor,
    transition_visible_rows: torch.Tensor,
    observer_visible_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Harden unique minimum-law candidates and reject every tie."""

    residuals = _source_law_residuals_from_visible(
        transition_visible,
        observer_visible,
        transition_visible_rows,
        observer_visible_rows,
    )

    transition_minimum = residuals.transition_residuals.amin(
        -1,
        keepdim=True,
    )
    observer_minimum = residuals.observer_residuals.amin(
        -1,
        keepdim=True,
    )
    transition_winners = residuals.transition_residuals.eq(
        transition_minimum
    )
    observer_winners = residuals.observer_residuals.eq(observer_minimum)
    if (
        bool(
            transition_winners[
                residuals.transition_hidden_rows
            ].sum(-1).ne(1).any()
        )
        or bool(
            observer_winners[
                residuals.observer_hidden_rows
            ].sum(-1).ne(1).any()
        )
    ):
        raise SourceLawResidualError(
            "source law does not identify a unique completion"
        )
    transition_choice = transition_winners.to(
        transition_visible.dtype
    )
    observer_choice = observer_winners.to(observer_visible.dtype)
    transition = (
        transition_visible
        + residuals.transition_hidden_rows[..., None].to(
            transition_visible.dtype
        )
        * transition_choice
    )
    observer = (
        observer_visible
        + residuals.observer_hidden_rows[..., None].to(
            observer_visible.dtype
        )
        * observer_choice
    )
    return transition, observer


def _transport_candidate_residuals(
    residuals: torch.Tensor,
    candidate_transport: torch.Tensor,
) -> torch.Tensor:
    """Transport a candidate packet by an explicit hard permutation."""

    if (
        residuals.ndim < 2
        or candidate_transport.shape
        != residuals.shape + (residuals.shape[-1],)
        or not residuals.is_floating_point()
        or not candidate_transport.is_floating_point()
        or residuals.device != candidate_transport.device
        or not bool(torch.isfinite(candidate_transport).all())
        or bool(
            candidate_transport.logical_and(
                candidate_transport.ne(1)
            ).any()
        )
        or bool(candidate_transport.sum(-1).ne(1).any())
        or bool(candidate_transport.sum(-2).ne(1).any())
        or bool(
            candidate_transport.diagonal(dim1=-2, dim2=-1).ne(0).any()
        )
    ):
        raise SourceLawResidualError(
            "candidate transport is not a hard derangement"
        )
    return torch.einsum(
        "...ij,...j->...i",
        candidate_transport,
        residuals,
    )


def _visible_from_sources(
    sources: tuple[bytes, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from pipeline.episode_functor_identifiable_board import decode_source

    transitions: list[torch.Tensor] = []
    observers: list[torch.Tensor] = []
    transition_masks: list[torch.Tensor] = []
    observer_masks: list[torch.Tensor] = []
    for source in sources:
        if not isinstance(source, bytes) or not source:
            raise SourceLawResidualError("source text differs")
        evidence = decode_source(source)
        states = tuple(sorted(evidence.state_keys))
        actions = tuple(
            sorted({record[0] for record in evidence.transition_events})
        )
        observers_keys = tuple(
            sorted({record[0] for record in evidence.observation_events})
        )
        if (
            len(states) != PRIMARY_STATES
            or len(actions) != PRIMARY_ACTIONS
            or len(observers_keys) != PRIMARY_OBSERVERS
        ):
            raise SourceLawResidualError(
                "source does not declare primary geometry"
            )
        state_index = {key: index for index, key in enumerate(states)}
        action_index = {key: index for index, key in enumerate(actions)}
        observer_index = {
            key: index for index, key in enumerate(observers_keys)
        }
        transition = torch.zeros(
            PRIMARY_ACTIONS,
            PRIMARY_STATES,
            PRIMARY_STATES,
        )
        transition_mask = torch.zeros(
            PRIMARY_ACTIONS,
            PRIMARY_STATES,
            dtype=torch.bool,
        )
        for action, state, destination in evidence.transition_events:
            action_slot = action_index[action]
            state_slot = state_index[state]
            if bool(transition_mask[action_slot, state_slot]):
                raise SourceLawResidualError(
                    "source repeats a transition row"
                )
            transition[
                action_slot,
                state_slot,
                state_index[destination],
            ] = 1.0
            transition_mask[action_slot, state_slot] = True
        observer = torch.zeros(
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            PRIMARY_ANSWERS,
        )
        observer_mask = torch.zeros(
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            dtype=torch.bool,
        )
        for item, state, answer in evidence.observation_events:
            item_slot = observer_index[item]
            state_slot = state_index[state]
            if (
                not isinstance(answer, int)
                or isinstance(answer, bool)
                or not 0 <= answer < PRIMARY_ANSWERS
                or bool(observer_mask[item_slot, state_slot])
            ):
                raise SourceLawResidualError(
                    "source observer row differs"
                )
            observer[item_slot, state_slot, answer] = 1.0
            observer_mask[item_slot, state_slot] = True
        transitions.append(transition)
        observers.append(observer)
        transition_masks.append(transition_mask)
        observer_masks.append(observer_mask)
    return (
        torch.stack(transitions),
        torch.stack(observers),
        torch.stack(transition_masks),
        torch.stack(observer_masks),
    )


def _source_without_keys(source: bytes, spans: tuple[tuple[int, int], ...]) -> bytes:
    output = bytearray()
    cursor = 0
    for start, end in spans:
        output.extend(source[cursor:start])
        output.extend(b"<OPAQUE-KEY>")
        cursor = end
    output.extend(source[cursor:])
    return bytes(output)


def _raw_key_recode_orders(
    source: bytes,
    recoded_source: bytes,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    from episode_functor_witness_compiler import scan_witness_source
    from pipeline.episode_functor_identifiable_board import decode_source

    original_scan = scan_witness_source(source)
    recoded_scan = scan_witness_source(recoded_source)
    if (
        original_scan.pointer.occurrence_to_unique
        != recoded_scan.pointer.occurrence_to_unique
        or _source_without_keys(source, original_scan.pointer.spans)
        != _source_without_keys(
            recoded_source,
            recoded_scan.pointer.spans,
        )
    ):
        raise SourceLawResidualError(
            "raw source recoding changes non-key structure"
        )
    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}
    for old_unique, new_unique in zip(
        original_scan.pointer.occurrence_to_unique,
        recoded_scan.pointer.occurrence_to_unique,
        strict=True,
    ):
        old = int.from_bytes(
            original_scan.pointer.unique_keys[old_unique],
            "little",
        )
        new = int.from_bytes(
            recoded_scan.pointer.unique_keys[new_unique],
            "little",
        )
        if (
            old in mapping and mapping[old] != new
        ) or (
            new in inverse and inverse[new] != old
        ):
            raise SourceLawResidualError(
                "raw source recoding is not bijective"
            )
        mapping[old] = new
        inverse[new] = old
    original = decode_source(source)
    recoded = decode_source(recoded_source)

    def order(
        old_keys: tuple[int, ...],
        new_keys: tuple[int, ...],
        label: str,
    ) -> tuple[int, ...]:
        try:
            result = tuple(
                old_keys.index(inverse[key]) for key in new_keys
            )
        except (KeyError, ValueError) as error:
            raise SourceLawResidualError(
                f"raw {label} recoding differs"
            ) from error
        if set(result) != set(range(len(old_keys))):
            raise SourceLawResidualError(
                f"raw {label} recoding is not a permutation"
            )
        return result

    old_states = tuple(sorted(original.state_keys))
    new_states = tuple(sorted(recoded.state_keys))
    old_actions = tuple(
        sorted({record[0] for record in original.transition_events})
    )
    new_actions = tuple(
        sorted({record[0] for record in recoded.transition_events})
    )
    old_observers = tuple(
        sorted({record[0] for record in original.observation_events})
    )
    new_observers = tuple(
        sorted({record[0] for record in recoded.observation_events})
    )
    return (
        order(old_states, new_states, "state"),
        order(old_actions, new_actions, "action"),
        order(old_observers, new_observers, "observer"),
    )


def _hard_derangement(
    *,
    source_sha256: str,
    control_seed_sha256: str,
    relation: str,
    relation_index: int,
    row_index: int,
    categories: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    material = (
        f"{control_seed_sha256}:{source_sha256}:{relation}:"
        f"{relation_index}:{row_index}"
    ).encode("ascii")
    rng = random.Random(int(sha256(material).hexdigest(), 16))
    order = list(range(categories))
    for _ in range(10_000):
        rng.shuffle(order)
        if all(index != value for index, value in enumerate(order)):
            matrix = torch.zeros(
                categories,
                categories,
                dtype=dtype,
                device=device,
            )
            matrix[
                torch.arange(categories, device=device),
                torch.tensor(order, device=device),
            ] = 1
            return matrix
    raise SourceLawResidualError("could not derive hard derangement")


class SourceLawResidualIssuer:
    """Sole issuer and verifier for source-law evidence and controls."""

    def __init__(self) -> None:
        self._nonce = secrets.token_hex(32)
        self._evidence: dict[object, _IssuedEvidence] = {}
        self._controls: dict[object, _IssuedControl] = {}
        self._control_by_evidence: dict[object, object] = {}

    def issue(
        self,
        sources: tuple[bytes, ...],
    ) -> SourceLawEvidenceCapability:
        if (
            not isinstance(sources, tuple)
            or not sources
            or any(
                not isinstance(source, bytes) or not source
                for source in sources
            )
        ):
            raise SourceLawResidualError("source batch differs")
        source_sha256 = tuple(
            sha256(source).hexdigest() for source in sources
        )
        bundle = tuple(
            tensor.detach().clone() for tensor in _visible_from_sources(sources)
        )
        token = object()
        capability = SourceLawEvidenceCapability(
            schema=EVIDENCE_CAPABILITY_SCHEMA,
            source_sha256=source_sha256,
            issuer_nonce=self._nonce,
            capability=token,
        )
        self._evidence[token] = _IssuedEvidence(
            capability=capability,
            source_sha256=source_sha256,
            issuer_nonce=self._nonce,
            sources=sources,
            transition=bundle[0],
            observer=bundle[1],
            transition_visible_rows=bundle[2],
            observer_visible_rows=bundle[3],
            tensor_sha256=tuple(_tensor_sha256(tensor) for tensor in bundle),
        )
        return capability

    def _verified_evidence(
        self,
        capability: SourceLawEvidenceCapability,
    ) -> _IssuedEvidence:
        if type(capability) is not SourceLawEvidenceCapability:
            raise SourceLawResidualError(
                "source-law evidence capability type differs"
            )
        issued = self._evidence.get(capability.capability)
        if (
            issued is None
            or issued.capability is not capability
            or capability.schema != EVIDENCE_CAPABILITY_SCHEMA
            or capability.source_sha256 != issued.source_sha256
            or capability.issuer_nonce != issued.issuer_nonce
            or capability.issuer_nonce != self._nonce
            or tuple(
                sha256(source).hexdigest() for source in issued.sources
            )
            != issued.source_sha256
            or tuple(
                _tensor_sha256(tensor)
                for tensor in (
                    issued.transition,
                    issued.observer,
                    issued.transition_visible_rows,
                    issued.observer_visible_rows,
                )
            )
            != issued.tensor_sha256
        ):
            raise SourceLawResidualError(
                "source-law evidence provenance differs"
            )
        return issued

    def residuals(
        self,
        capability: SourceLawEvidenceCapability,
    ) -> SourceLawResidualResult:
        issued = self._verified_evidence(capability)
        return _source_law_residuals_from_visible(
            issued.transition,
            issued.observer,
            issued.transition_visible_rows,
            issued.observer_visible_rows,
        )

    def complete(
        self,
        capability: SourceLawEvidenceCapability,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        issued = self._verified_evidence(capability)
        return _complete_from_visible_source_law(
            issued.transition,
            issued.observer,
            issued.transition_visible_rows,
            issued.observer_visible_rows,
        )

    def issue_control(
        self,
        evidence_capability: SourceLawEvidenceCapability,
    ) -> SourceLawControlCapability:
        issued = self._verified_evidence(evidence_capability)
        if evidence_capability.capability in self._control_by_evidence:
            raise SourceLawResidualError(
                "source-law control was already issued"
            )
        # Keep the committed seed in bytecode, not a mutable module lookup.
        control_seed_sha256 = (
            "f1571580393d3b13f9443f85a35bedcc"
            "15a6c1b7ced559561f3a52c9f6600868"
        )
        transition_shape = issued.transition.shape[:-1] + (
            PRIMARY_STATES,
            PRIMARY_STATES,
        )
        observer_shape = issued.observer.shape[:-1] + (
            PRIMARY_ANSWERS,
            PRIMARY_ANSWERS,
        )
        transition_transport = torch.empty(
            transition_shape,
            dtype=issued.transition.dtype,
            device=issued.transition.device,
        )
        observer_transport = torch.empty(
            observer_shape,
            dtype=issued.observer.dtype,
            device=issued.observer.device,
        )
        for batch, source_hash in enumerate(
            evidence_capability.source_sha256
        ):
            for relation in range(PRIMARY_ACTIONS):
                for row in range(PRIMARY_STATES):
                    transition_transport[batch, relation, row] = (
                        _hard_derangement(
                            source_sha256=source_hash,
                            control_seed_sha256=control_seed_sha256,
                            relation="transition",
                            relation_index=relation,
                            row_index=row,
                            categories=PRIMARY_STATES,
                            dtype=issued.transition.dtype,
                            device=issued.transition.device,
                        )
                    )
            for relation in range(PRIMARY_OBSERVERS):
                for row in range(PRIMARY_STATES):
                    observer_transport[batch, relation, row] = (
                        _hard_derangement(
                            source_sha256=source_hash,
                            control_seed_sha256=control_seed_sha256,
                            relation="observer",
                            relation_index=relation,
                            row_index=row,
                            categories=PRIMARY_ANSWERS,
                            dtype=issued.observer.dtype,
                            device=issued.observer.device,
                        )
                    )
        token = object()
        capability = SourceLawControlCapability(
            schema=CONTROL_CAPABILITY_SCHEMA,
            source_sha256=evidence_capability.source_sha256,
            control_seed_sha256=control_seed_sha256,
            issuer_nonce=self._nonce,
            capability=token,
        )
        self._controls[token] = _IssuedControl(
            capability=capability,
            evidence_capability=evidence_capability,
            source_sha256=evidence_capability.source_sha256,
            control_seed_sha256=control_seed_sha256,
            issuer_nonce=self._nonce,
            transition_transport=transition_transport,
            observer_transport=observer_transport,
            tensor_sha256=(
                _tensor_sha256(transition_transport),
                _tensor_sha256(observer_transport),
            ),
        )
        self._control_by_evidence[evidence_capability.capability] = token
        return capability

    @staticmethod
    def _permutation_order(
        values: tuple[int, ...],
        size: int,
        label: str,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            not isinstance(values, tuple)
            or len(values) != size
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                for value in values
            )
            or set(values) != set(range(size))
        ):
            raise SourceLawResidualError(
                f"{label} recoding is not a permutation"
            )
        return torch.tensor(values, dtype=torch.long, device=device)

    def recode_control(
        self,
        evidence_capability: SourceLawEvidenceCapability,
        control_capability: SourceLawControlCapability,
        recoded_sources: tuple[bytes, ...],
    ) -> tuple[SourceLawEvidenceCapability, SourceLawControlCapability]:
        """Issue a verified recoding and conjugate the realized control."""

        evidence = self._verified_evidence(evidence_capability)
        self.controlled_residuals(
            evidence_capability,
            control_capability,
        )
        control = self._controls[control_capability.capability]
        if (
            not isinstance(recoded_sources, tuple)
            or len(recoded_sources) != len(evidence.sources)
        ):
            raise SourceLawResidualError(
                "raw source recoding batch differs"
            )
        recode_orders = tuple(
            _raw_key_recode_orders(source, recoded)
            for source, recoded in zip(
                evidence.sources,
                recoded_sources,
                strict=True,
            )
        )
        expected_transition = []
        expected_observer = []
        expected_transition_mask = []
        expected_observer_mask = []
        recoded_transition_transport = []
        recoded_observer_transport = []
        for batch, (state_values, action_values, observer_values) in enumerate(
            recode_orders
        ):
            state = self._permutation_order(
                state_values,
                PRIMARY_STATES,
                "state",
                device=evidence.transition.device,
            )
            action = self._permutation_order(
                action_values,
                PRIMARY_ACTIONS,
                "action",
                device=evidence.transition.device,
            )
            observer = self._permutation_order(
                observer_values,
                PRIMARY_OBSERVERS,
                "observer",
                device=evidence.transition.device,
            )
            expected_transition.append(
                evidence.transition[batch][
                    action
                ][:, state][:, :, state]
            )
            expected_observer.append(
                evidence.observer[batch][
                    observer
                ][:, state]
            )
            expected_transition_mask.append(
                evidence.transition_visible_rows[batch][
                    action
                ][:, state]
            )
            expected_observer_mask.append(
                evidence.observer_visible_rows[batch][
                    observer
                ][:, state]
            )
            recoded_transition_transport.append(
                control.transition_transport[batch][
                    action
                ][:, state][:, :, state][:, :, :, state]
            )
            recoded_observer_transport.append(
                control.observer_transport[batch][
                    observer
                ][:, state]
            )
        expected_bundle = (
            torch.stack(expected_transition),
            torch.stack(expected_observer),
            torch.stack(expected_transition_mask),
            torch.stack(expected_observer_mask),
        )
        observed_bundle = _visible_from_sources(recoded_sources)
        if any(
            not torch.equal(expected, observed)
            for expected, observed in zip(
                expected_bundle,
                observed_bundle,
                strict=True,
            )
        ):
            raise SourceLawResidualError(
                "raw source recoding does not match declared permutations"
            )
        recoded_evidence = self.issue(recoded_sources)
        recoded_transition = torch.stack(recoded_transition_transport)
        recoded_observer = torch.stack(recoded_observer_transport)
        token = object()
        recoded_control = SourceLawControlCapability(
            schema=CONTROL_CAPABILITY_SCHEMA,
            source_sha256=recoded_evidence.source_sha256,
            control_seed_sha256=(
                "f1571580393d3b13f9443f85a35bedcc"
                "15a6c1b7ced559561f3a52c9f6600868"
            ),
            issuer_nonce=self._nonce,
            capability=token,
        )
        self._controls[token] = _IssuedControl(
            capability=recoded_control,
            evidence_capability=recoded_evidence,
            source_sha256=recoded_evidence.source_sha256,
            control_seed_sha256=(
                "f1571580393d3b13f9443f85a35bedcc"
                "15a6c1b7ced559561f3a52c9f6600868"
            ),
            issuer_nonce=self._nonce,
            transition_transport=recoded_transition,
            observer_transport=recoded_observer,
            tensor_sha256=(
                _tensor_sha256(recoded_transition),
                _tensor_sha256(recoded_observer),
            ),
        )
        self._control_by_evidence[recoded_evidence.capability] = token
        return recoded_evidence, recoded_control

    def controlled_residuals(
        self,
        evidence_capability: SourceLawEvidenceCapability,
        control_capability: SourceLawControlCapability,
    ) -> SourceLawResidualResult:
        evidence = self._verified_evidence(evidence_capability)
        if type(control_capability) is not SourceLawControlCapability:
            raise SourceLawResidualError(
                "source-law control capability type differs"
            )
        control = self._controls.get(control_capability.capability)
        if (
            control is None
            or control.capability is not control_capability
            or control.evidence_capability is not evidence_capability
            or control_capability.schema != CONTROL_CAPABILITY_SCHEMA
            or control_capability.source_sha256 != control.source_sha256
            or control_capability.control_seed_sha256
            != control.control_seed_sha256
            or control_capability.issuer_nonce != control.issuer_nonce
            or control_capability.issuer_nonce != self._nonce
            or control_capability.source_sha256
            != evidence_capability.source_sha256
            or (
                _tensor_sha256(control.transition_transport),
                _tensor_sha256(control.observer_transport),
            )
            != control.tensor_sha256
        ):
            raise SourceLawResidualError(
                "source-law control provenance differs"
            )
        residuals = _source_law_residuals_from_visible(
            evidence.transition,
            evidence.observer,
            evidence.transition_visible_rows,
            evidence.observer_visible_rows,
        )
        return SourceLawResidualResult(
            transition_residuals=_transport_candidate_residuals(
                residuals.transition_residuals,
                control.transition_transport,
            ),
            observer_residuals=_transport_candidate_residuals(
                residuals.observer_residuals,
                control.observer_transport,
            ),
            transition_hidden_rows=residuals.transition_hidden_rows,
            observer_hidden_rows=residuals.observer_hidden_rows,
        )


def source_law_residuals(
    issuer: SourceLawResidualIssuer,
    capability: SourceLawEvidenceCapability,
) -> SourceLawResidualResult:
    if type(issuer) is not SourceLawResidualIssuer:
        raise SourceLawResidualError("source-law issuer type differs")
    return issuer.residuals(capability)


def complete_from_source_law(
    issuer: SourceLawResidualIssuer,
    capability: SourceLawEvidenceCapability,
) -> tuple[torch.Tensor, torch.Tensor]:
    if type(issuer) is not SourceLawResidualIssuer:
        raise SourceLawResidualError("source-law issuer type differs")
    return issuer.complete(capability)


__all__ = [
    "ANSWER_MULTIPLICITY",
    "CONTROL_SEED_SHA256",
    "SourceLawControlCapability",
    "SourceLawEvidenceCapability",
    "SourceLawResidualError",
    "SourceLawResidualIssuer",
    "SourceLawResidualResult",
    "complete_from_source_law",
    "source_law_residuals",
]
