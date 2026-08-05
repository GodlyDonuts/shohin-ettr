#!/usr/bin/env python3
"""Exact source-sealed factorized epistemic packet mechanics for DIVERGE-v0."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Iterable, Mapping

from version_space_accounting import canonical_json_bytes, integer_bit_growth


SCHEMA = "shohin-diverge-v0-packet-v1"
ANSWER = "ANSWER"
ABSTAIN = "ABSTAIN"
REJECT = "REJECT"
OVERFLOW = "OVERFLOW"


class DivergeContractError(ValueError):
    """Raised when a packet violates the frozen semantic contract."""


def _exact_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DivergeContractError(f"{name} must be an exact integer")
    return value


def _nonnegative(value: object, name: str) -> int:
    value = _exact_int(value, name)
    if value < 0:
        raise DivergeContractError(f"{name} must be nonnegative")
    return value


def _commit(domain: str, payload: object) -> str:
    digest = hashlib.sha256()
    domain_bytes = domain.encode("ascii")
    payload_bytes = canonical_json_bytes(payload)
    for part in (domain_bytes, payload_bytes):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def validate_commitment(value: object, name: str = "commitment") -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DivergeContractError(f"{name} must be 64-character SHA-256 hex")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise DivergeContractError(f"{name} must be SHA-256 hex") from error
    return value.lower()


def named_commitment(domain: str, name: str) -> str:
    """Create a sealed test/supervisor commitment without storing the name."""

    return _commit(domain, {"name": str(name)})


@dataclass(frozen=True)
class PacketCaps:
    max_variables: int = 6
    max_domain: int = 4
    max_worlds: int = 64
    max_hard_factors: int = 32
    max_support_factors: int = 32
    max_factor_rows: int = 256
    max_patches: int = 64
    max_guard_literals: int = 6
    max_nogoods: int = 32
    max_cells: int = 64
    max_edges: int = 256
    max_integer_bits: int = 256

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if _exact_int(value, name) <= 0:
                raise DivergeContractError(f"{name} must be positive")


@dataclass(frozen=True, order=True)
class TypedCell:
    slot: int
    type_id: int
    value: int
    live: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _nonnegative(self.slot, "cell slot"))
        object.__setattr__(self, "type_id", _nonnegative(self.type_id, "cell type"))
        object.__setattr__(self, "value", _exact_int(self.value, "cell value"))
        if not isinstance(self.live, bool):
            raise DivergeContractError("cell liveness must be boolean")


@dataclass(frozen=True, order=True)
class TypedEdge:
    source: int
    relation: int
    target: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _nonnegative(self.source, "edge source"))
        object.__setattr__(self, "relation", _nonnegative(self.relation, "edge relation"))
        object.__setattr__(self, "target", _nonnegative(self.target, "edge target"))


@dataclass(frozen=True)
class TypedState:
    cells: tuple[TypedCell, ...]
    edges: tuple[TypedEdge, ...] = ()

    def __post_init__(self) -> None:
        cells = tuple(sorted(self.cells))
        edges = tuple(sorted(self.edges))
        if not cells:
            raise DivergeContractError("typed state requires at least one cell")
        if len({cell.slot for cell in cells}) != len(cells):
            raise DivergeContractError("typed state contains duplicate slots")
        if len(set(edges)) != len(edges):
            raise DivergeContractError("typed state contains duplicate edges")
        live = {cell.slot for cell in cells if cell.live}
        if any(edge.source not in live or edge.target not in live for edge in edges):
            raise DivergeContractError("edge references a missing or dead slot")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "edges", edges)

    def record(self) -> dict[str, object]:
        return {
            "cells": [
                [cell.slot, cell.type_id, cell.value, int(cell.live)]
                for cell in self.cells
            ],
            "edges": [
                [edge.source, edge.relation, edge.target] for edge in self.edges
            ],
        }


@dataclass(frozen=True)
class FaultLine:
    variable_id: int
    options: tuple[str, ...]
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "variable_id", _nonnegative(self.variable_id, "fault-line ID")
        )
        options = tuple(validate_commitment(value, "option commitment") for value in self.options)
        if len(options) < 2 or len(set(options)) != len(options):
            raise DivergeContractError("fault line requires distinct finite options")
        object.__setattr__(self, "options", options)
        object.__setattr__(
            self, "provenance", validate_commitment(self.provenance, "fault-line provenance")
        )


@dataclass(frozen=True, order=True)
class Literal:
    variable_id: int
    option: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "variable_id", _nonnegative(self.variable_id, "literal variable")
        )
        object.__setattr__(self, "option", _nonnegative(self.option, "literal option"))


@dataclass(frozen=True)
class Guard:
    literals: tuple[Literal, ...] = ()

    def __post_init__(self) -> None:
        literals = tuple(sorted(self.literals))
        by_variable: dict[int, int] = {}
        for literal in literals:
            previous = by_variable.setdefault(literal.variable_id, literal.option)
            if previous != literal.option:
                raise DivergeContractError("guard contains contradictory literals")
        literals = tuple(Literal(variable, option) for variable, option in sorted(by_variable.items()))
        object.__setattr__(self, "literals", literals)

    def matches(self, assignment: tuple[int, ...]) -> bool:
        return all(
            literal.variable_id < len(assignment)
            and assignment[literal.variable_id] == literal.option
            for literal in self.literals
        )

    def record(self) -> list[list[int]]:
        return [[literal.variable_id, literal.option] for literal in self.literals]


@dataclass(frozen=True)
class HardFactor:
    scope: tuple[int, ...]
    allowed: tuple[tuple[int, ...], ...]
    provenance: str

    def __post_init__(self) -> None:
        scope = tuple(_nonnegative(value, "hard-factor scope") for value in self.scope)
        if not scope or len(set(scope)) != len(scope):
            raise DivergeContractError("hard factor requires a nonempty unique scope")
        allowed = tuple(sorted(set(tuple(row) for row in self.allowed)))
        if not allowed or any(len(row) != len(scope) for row in allowed):
            raise DivergeContractError("hard-factor rows do not match scope")
        for row in allowed:
            for option in row:
                _nonnegative(option, "hard-factor option")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "allowed", allowed)
        object.__setattr__(
            self, "provenance", validate_commitment(self.provenance, "hard-factor provenance")
        )


@dataclass(frozen=True)
class SupportFactor:
    scope: tuple[int, ...]
    masses: tuple[tuple[tuple[int, ...], int], ...]
    provenance: str

    def __post_init__(self) -> None:
        scope = tuple(_nonnegative(value, "support-factor scope") for value in self.scope)
        if not scope or len(set(scope)) != len(scope):
            raise DivergeContractError("support factor requires a nonempty unique scope")
        by_row: dict[tuple[int, ...], int] = {}
        for row, mass in self.masses:
            row = tuple(row)
            if len(row) != len(scope):
                raise DivergeContractError("support-factor row does not match scope")
            for option in row:
                _nonnegative(option, "support-factor option")
            mass = _exact_int(mass, "support-factor mass")
            if mass <= 0:
                raise DivergeContractError("support-factor mass must be positive")
            if row in by_row and by_row[row] != mass:
                raise DivergeContractError("support-factor row has conflicting masses")
            by_row[row] = mass
        if not by_row:
            raise DivergeContractError("support factor requires at least one explicit mass")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "masses", tuple(sorted(by_row.items())))
        object.__setattr__(
            self, "provenance", validate_commitment(self.provenance, "support-factor provenance")
        )


TRANSACTION_ARITY = {
    "SET_VALUE": 2,
    "ADD_VALUE": 2,
    "COPY_VALUE": 2,
    "SWAP_VALUE": 2,
    "SET_TYPE": 2,
    "LINK": 3,
    "UNLINK": 3,
}


@dataclass(frozen=True)
class TypedTransaction:
    opcode: str
    arguments: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.opcode not in TRANSACTION_ARITY:
            raise DivergeContractError(f"unknown transaction opcode: {self.opcode}")
        arguments = tuple(_exact_int(value, "transaction argument") for value in self.arguments)
        if len(arguments) != TRANSACTION_ARITY[self.opcode]:
            raise DivergeContractError("transaction arity differs from opcode")
        object.__setattr__(self, "arguments", arguments)

    def record(self) -> list[object]:
        return [self.opcode, list(self.arguments)]


@dataclass(frozen=True)
class GuardedPatch:
    index: int
    guard: Guard
    transaction: TypedTransaction
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _nonnegative(self.index, "patch index"))
        object.__setattr__(
            self, "provenance", validate_commitment(self.provenance, "patch provenance")
        )


@dataclass(frozen=True)
class VerifiedNogood:
    guard: Guard
    evidence_commitment: str
    verifier_commitment: str
    deletion_minimal: bool

    def __post_init__(self) -> None:
        if not self.guard.literals:
            raise DivergeContractError("nogood guard must be nonempty")
        object.__setattr__(
            self,
            "evidence_commitment",
            validate_commitment(self.evidence_commitment, "nogood evidence"),
        )
        object.__setattr__(
            self,
            "verifier_commitment",
            validate_commitment(self.verifier_commitment, "nogood verifier"),
        )
        if not isinstance(self.deletion_minimal, bool):
            raise DivergeContractError("nogood minimality must be boolean")


@dataclass(frozen=True)
class OptionEvidenceCertificate:
    variable_id: int
    confirmed_option: int
    nogood: VerifiedNogood


def certify_binary_option_evidence(
    packet: "EpistemicPacket",
    *,
    option_commitment: str,
    evidence_commitment: str,
) -> OptionEvidenceCertificate | None:
    """Bind delayed evidence to one sealed binary option without source access."""

    if packet.overflow:
        return None
    option_commitment = validate_commitment(option_commitment, "evidence option")
    evidence_commitment = validate_commitment(evidence_commitment, "option evidence")
    matches = [
        (variable.variable_id, option)
        for variable in packet.variables
        for option, commitment in enumerate(variable.options)
        if commitment == option_commitment
    ]
    if len(matches) != 1:
        return None
    variable_id, confirmed = matches[0]
    variable = next(
        item for item in packet.variables if item.variable_id == variable_id
    )
    if len(variable.options) != 2:
        return None
    rejected = 1 - confirmed
    guard = Guard((Literal(variable_id, rejected),))
    verifier_commitment = _commit(
        "diverge-v0-sealed-option-evidence",
        {
            "source_commitment": packet.source_commitment,
            "variable_provenance": variable.provenance,
            "confirmed_option_commitment": option_commitment,
            "evidence_commitment": evidence_commitment,
            "rejected_guard": guard.record(),
        },
    )
    return OptionEvidenceCertificate(
        variable_id,
        confirmed,
        VerifiedNogood(guard, evidence_commitment, verifier_commitment, True),
    )


@dataclass(frozen=True)
class EpistemicPacket:
    source_commitment: str
    shared_state: TypedState
    variables: tuple[FaultLine, ...]
    hard_factors: tuple[HardFactor, ...]
    support_factors: tuple[SupportFactor, ...]
    patches: tuple[GuardedPatch, ...]
    nogoods: tuple[VerifiedNogood, ...]
    caps: PacketCaps
    overflow: bool = False
    overflow_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_commitment", validate_commitment(self.source_commitment, "source")
        )
        if self.overflow:
            if not self.overflow_reason:
                raise DivergeContractError("overflowed packet requires a reason")
            if any((self.variables, self.hard_factors, self.support_factors, self.patches, self.nogoods)):
                raise DivergeContractError("overflowed packet cannot expose partial support")
        elif self.overflow_reason is not None:
            raise DivergeContractError("exact packet cannot carry an overflow reason")


def _guard_record(guard: Guard) -> list[list[int]]:
    return guard.record()


def _factor_record(factor: HardFactor) -> dict[str, object]:
    return {
        "scope": list(factor.scope),
        "allowed": [list(row) for row in factor.allowed],
        "provenance": factor.provenance,
    }


def _support_record(factor: SupportFactor) -> dict[str, object]:
    return {
        "scope": list(factor.scope),
        "masses": [[list(row), mass] for row, mass in factor.masses],
        "provenance": factor.provenance,
    }


def _patch_record(patch: GuardedPatch) -> dict[str, object]:
    return {
        "index": patch.index,
        "guard": _guard_record(patch.guard),
        "transaction": patch.transaction.record(),
        "provenance": patch.provenance,
    }


def _nogood_record(nogood: VerifiedNogood) -> dict[str, object]:
    return {
        "guard": _guard_record(nogood.guard),
        "evidence_commitment": nogood.evidence_commitment,
        "verifier_commitment": nogood.verifier_commitment,
        "deletion_minimal": nogood.deletion_minimal,
    }


def _canonicalize_guard(guard: Guard, remap: Mapping[int, int]) -> Guard:
    try:
        return Guard(tuple(Literal(remap[item.variable_id], item.option) for item in guard.literals))
    except KeyError as error:
        raise DivergeContractError("guard references an unknown fault line") from error


def _canonicalize_factor(
    factor: HardFactor,
    remap: Mapping[int, int],
) -> HardFactor:
    mapped = [remap[variable] for variable in factor.scope]
    order = sorted(range(len(mapped)), key=mapped.__getitem__)
    scope = tuple(mapped[index] for index in order)
    allowed = tuple(tuple(row[index] for index in order) for row in factor.allowed)
    return HardFactor(scope, allowed, factor.provenance)


def _canonicalize_support(
    factor: SupportFactor,
    remap: Mapping[int, int],
) -> SupportFactor:
    mapped = [remap[variable] for variable in factor.scope]
    order = sorted(range(len(mapped)), key=mapped.__getitem__)
    scope = tuple(mapped[index] for index in order)
    masses = tuple(
        (tuple(row[index] for index in order), mass) for row, mass in factor.masses
    )
    return SupportFactor(scope, masses, factor.provenance)


def _overflow_packet(
    source_commitment: str,
    shared_state: TypedState,
    caps: PacketCaps,
    reason: str,
) -> EpistemicPacket:
    return EpistemicPacket(
        source_commitment=source_commitment,
        shared_state=shared_state,
        variables=(),
        hard_factors=(),
        support_factors=(),
        patches=(),
        nogoods=(),
        caps=caps,
        overflow=True,
        overflow_reason=reason,
    )


def _resource_overflow(
    state: TypedState,
    variables: tuple[FaultLine, ...],
    hard_factors: tuple[HardFactor, ...],
    support_factors: tuple[SupportFactor, ...],
    patches: tuple[GuardedPatch, ...],
    nogoods: tuple[VerifiedNogood, ...],
    caps: PacketCaps,
) -> str | None:
    checks = (
        (len(state.cells), caps.max_cells, "cell cap"),
        (len(state.edges), caps.max_edges, "edge cap"),
        (len(variables), caps.max_variables, "fault-line cap"),
        (max((len(item.options) for item in variables), default=0), caps.max_domain, "domain cap"),
        (len(hard_factors), caps.max_hard_factors, "hard-factor cap"),
        (len(support_factors), caps.max_support_factors, "support-factor cap"),
        (sum(len(item.allowed) for item in hard_factors), caps.max_factor_rows, "hard-factor row cap"),
        (sum(len(item.masses) for item in support_factors), caps.max_factor_rows, "support-factor row cap"),
        (len(patches), caps.max_patches, "patch cap"),
        (len(nogoods), caps.max_nogoods, "nogood cap"),
        (
            max(
                (
                    *(len(item.guard.literals) for item in patches),
                    *(len(item.guard.literals) for item in nogoods),
                    0,
                )
            ),
            caps.max_guard_literals,
            "guard-literal cap",
        ),
    )
    for observed, limit, reason in checks:
        if observed > limit:
            return reason
    growth = integer_bit_growth(
        {
            "state": state.record(),
            "hard_factors": [_factor_record(item) for item in hard_factors],
            "support_factors": [_support_record(item) for item in support_factors],
            "patches": [_patch_record(item) for item in patches],
            "nogoods": [_nogood_record(item) for item in nogoods],
        }
    )
    if growth.max_magnitude_bits > caps.max_integer_bits:
        return "integer-bit cap"
    return None


def build_packet(
    *,
    source_commitment: str,
    shared_state: TypedState,
    variables: Iterable[FaultLine],
    hard_factors: Iterable[HardFactor] = (),
    support_factors: Iterable[SupportFactor] = (),
    patches: Iterable[GuardedPatch] = (),
    nogoods: Iterable[VerifiedNogood] = (),
    caps: PacketCaps | None = None,
) -> EpistemicPacket:
    """Validate, anonymize, canonicalize, and seal one exact packet."""

    source_commitment = validate_commitment(source_commitment, "source")
    caps = caps or PacketCaps()
    variables = tuple(variables)
    hard_factors = tuple(hard_factors)
    support_factors = tuple(support_factors)
    patches = tuple(patches)
    nogoods = tuple(nogoods)
    identifiers = [item.variable_id for item in variables]
    if len(set(identifiers)) != len(identifiers):
        raise DivergeContractError("fault-line IDs must be unique")
    provenances = [item.provenance for item in variables]
    if len(set(provenances)) != len(provenances):
        raise DivergeContractError("fault-line provenance must be unique")
    ordered = tuple(sorted(variables, key=lambda item: (item.provenance, item.options)))
    remap = {item.variable_id: index for index, item in enumerate(ordered)}
    canonical_variables = tuple(
        FaultLine(index, item.options, item.provenance) for index, item in enumerate(ordered)
    )
    try:
        canonical_hard = tuple(
            sorted(
                (_canonicalize_factor(item, remap) for item in hard_factors),
                key=lambda item: canonical_json_bytes(_factor_record(item)),
            )
        )
        canonical_support = tuple(
            sorted(
                (_canonicalize_support(item, remap) for item in support_factors),
                key=lambda item: canonical_json_bytes(_support_record(item)),
            )
        )
        canonical_patches = tuple(
            sorted(
                (
                    GuardedPatch(
                        item.index,
                        _canonicalize_guard(item.guard, remap),
                        item.transaction,
                        item.provenance,
                    )
                    for item in patches
                ),
                key=lambda item: item.index,
            )
        )
        canonical_nogoods = tuple(
            sorted(
                (
                    VerifiedNogood(
                        _canonicalize_guard(item.guard, remap),
                        item.evidence_commitment,
                        item.verifier_commitment,
                        item.deletion_minimal,
                    )
                    for item in nogoods
                ),
                key=lambda item: canonical_json_bytes(_nogood_record(item)),
            )
        )
    except KeyError as error:
        raise DivergeContractError("factor references an unknown fault line") from error
    if [item.index for item in canonical_patches] != list(range(len(canonical_patches))):
        raise DivergeContractError("patch indices must be unique and chronological from zero")
    for literal in itertools.chain(
        *(item.guard.literals for item in canonical_patches),
        *(item.guard.literals for item in canonical_nogoods),
    ):
        if literal.variable_id >= len(canonical_variables):
            raise DivergeContractError("guard references an unknown fault line")
        if literal.option >= len(canonical_variables[literal.variable_id].options):
            raise DivergeContractError("guard option is outside its fault-line domain")
    for factor in (*canonical_hard, *canonical_support):
        for variable in factor.scope:
            if variable >= len(canonical_variables):
                raise DivergeContractError("factor references an unknown fault line")
        rows = factor.allowed if isinstance(factor, HardFactor) else tuple(row for row, _ in factor.masses)
        for row in rows:
            if any(option >= len(canonical_variables[variable].options) for variable, option in zip(factor.scope, row)):
                raise DivergeContractError("factor option is outside its fault-line domain")
    reason = _resource_overflow(
        shared_state,
        canonical_variables,
        canonical_hard,
        canonical_support,
        canonical_patches,
        canonical_nogoods,
        caps,
    )
    if reason:
        return _overflow_packet(source_commitment, shared_state, caps, reason)
    packet = EpistemicPacket(
        source_commitment,
        shared_state,
        canonical_variables,
        canonical_hard,
        canonical_support,
        canonical_patches,
        canonical_nogoods,
        caps,
    )
    assignments = enumerate_assignments(packet)
    if not assignments:
        raise DivergeContractError("hard factors and nogoods remove every world")
    if len(assignments) > caps.max_worlds:
        return _overflow_packet(source_commitment, shared_state, caps, "represented-world cap")
    return packet


def _factor_allows(factor: HardFactor, assignment: tuple[int, ...]) -> bool:
    row = tuple(assignment[variable] for variable in factor.scope)
    return row in factor.allowed


def enumerate_assignments(packet: EpistemicPacket) -> tuple[tuple[int, ...], ...]:
    if packet.overflow:
        return ()
    domains = tuple(range(len(variable.options)) for variable in packet.variables)
    assignments = []
    for assignment in itertools.product(*domains):
        if not all(_factor_allows(factor, assignment) for factor in packet.hard_factors):
            continue
        if any(nogood.guard.matches(assignment) for nogood in packet.nogoods):
            continue
        assignments.append(tuple(assignment))
    return tuple(assignments)


def assignment_mass(packet: EpistemicPacket, assignment: tuple[int, ...]) -> int:
    mass = 1
    for factor in packet.support_factors:
        row = tuple(assignment[variable] for variable in factor.scope)
        mass *= dict(factor.masses).get(row, 1)
    return mass


def _state_cells(state: TypedState) -> dict[int, TypedCell]:
    return {cell.slot: cell for cell in state.cells}


def apply_transaction(state: TypedState, transaction: TypedTransaction) -> TypedState:
    """Apply one complete typed transaction or raise a hard contradiction."""

    cells = _state_cells(state)

    def cell(slot: int) -> TypedCell:
        if slot not in cells or not cells[slot].live:
            raise DivergeContractError("transaction references a missing or dead slot")
        return cells[slot]

    args = transaction.arguments
    opcode = transaction.opcode
    edges = set(state.edges)
    if opcode == "SET_VALUE":
        current = cell(args[0])
        cells[args[0]] = replace(current, value=args[1])
    elif opcode == "ADD_VALUE":
        current = cell(args[0])
        cells[args[0]] = replace(current, value=current.value + args[1])
    elif opcode == "COPY_VALUE":
        source = cell(args[0])
        target = cell(args[1])
        if source.type_id != target.type_id:
            raise DivergeContractError("COPY_VALUE crosses incompatible types")
        cells[args[1]] = replace(target, value=source.value)
    elif opcode == "SWAP_VALUE":
        left = cell(args[0])
        right = cell(args[1])
        if left.type_id != right.type_id:
            raise DivergeContractError("SWAP_VALUE crosses incompatible types")
        cells[args[0]] = replace(left, value=right.value)
        cells[args[1]] = replace(right, value=left.value)
    elif opcode == "SET_TYPE":
        current = cell(args[0])
        cells[args[0]] = replace(current, type_id=_nonnegative(args[1], "new type"))
    elif opcode == "LINK":
        cell(args[0])
        cell(args[2])
        edge = TypedEdge(args[0], args[1], args[2])
        if edge in edges:
            raise DivergeContractError("LINK duplicates an existing edge")
        edges.add(edge)
    elif opcode == "UNLINK":
        edge = TypedEdge(args[0], args[1], args[2])
        if edge not in edges:
            raise DivergeContractError("UNLINK removes a missing edge")
        edges.remove(edge)
    else:  # pragma: no cover - TypedTransaction validates the vocabulary.
        raise DivergeContractError("unknown transaction")
    return TypedState(tuple(cells.values()), tuple(edges))


def transaction_touched_slots(transaction: TypedTransaction) -> frozenset[int]:
    """Return a conservative slot footprint for certified commutation."""

    opcode = transaction.opcode
    arguments = transaction.arguments
    if opcode in {"SET_VALUE", "ADD_VALUE", "SET_TYPE"}:
        return frozenset((arguments[0],))
    if opcode in {"COPY_VALUE", "SWAP_VALUE"}:
        return frozenset((arguments[0], arguments[1]))
    if opcode in {"LINK", "UNLINK"}:
        return frozenset((arguments[0], arguments[2]))
    raise DivergeContractError("unknown transaction")  # pragma: no cover


def commuting_patch_schedule(
    patches: Iterable[GuardedPatch],
) -> tuple[GuardedPatch, ...]:
    """Canonicalize only adjacent transactions proven disjoint.

    Guard truth is assignment-owned and cannot be changed by a transaction.
    Transactions with disjoint conservative slot footprints therefore commute.
    All overlapping pairs retain their original relative order, preserving the
    packet's noncommuting chronological semantics.
    """

    scheduled = list(patches)

    def key(patch: GuardedPatch) -> tuple[object, ...]:
        slots = tuple(sorted(transaction_touched_slots(patch.transaction)))
        return slots, patch.provenance, patch.index

    changed = True
    while changed:
        changed = False
        for index in range(len(scheduled) - 1):
            left = scheduled[index]
            right = scheduled[index + 1]
            if (
                transaction_touched_slots(left.transaction).isdisjoint(
                    transaction_touched_slots(right.transaction)
                )
                and key(right) < key(left)
            ):
                scheduled[index], scheduled[index + 1] = right, left
                changed = True
    return tuple(scheduled)


@dataclass(frozen=True)
class WorldResult:
    assignment: tuple[int, ...]
    mass: int
    state: TypedState | None
    contradiction: bool

    def record(self) -> dict[str, object]:
        return {
            "assignment": list(self.assignment),
            "mass": self.mass,
            "state": None if self.state is None else self.state.record(),
            "contradiction": self.contradiction,
        }


@dataclass(frozen=True)
class ExecutionReceipt:
    worlds: tuple[WorldResult, ...]
    unique_transactions: int
    duplicated_transactions: int
    overflow: bool

    @property
    def shared_transactions(self) -> int:
        return self.duplicated_transactions - self.unique_transactions


def execute_packet(
    packet: EpistemicPacket,
    *,
    patch_limit: int | None = None,
    commute_disjoint: bool = False,
) -> ExecutionReceipt:
    if packet.overflow:
        return ExecutionReceipt((), 0, 0, True)
    if patch_limit is None:
        patches = packet.patches
    else:
        patch_limit = _nonnegative(patch_limit, "patch limit")
        patches = packet.patches[:patch_limit]
    if commute_disjoint:
        patches = commuting_patch_schedule(patches)
    assignments = enumerate_assignments(packet)
    states: dict[tuple[int, ...], TypedState | None] = {
        assignment: packet.shared_state for assignment in assignments
    }
    unique_transactions = 0
    duplicated_transactions = 0
    for patch in patches:
        groups: dict[bytes, list[tuple[int, ...]]] = {}
        state_by_key: dict[bytes, TypedState] = {}
        for assignment in assignments:
            state = states[assignment]
            if state is None or not patch.guard.matches(assignment):
                continue
            key = canonical_json_bytes(state.record())
            groups.setdefault(key, []).append(assignment)
            state_by_key[key] = state
        duplicated_transactions += sum(len(group) for group in groups.values())
        for key, group in groups.items():
            unique_transactions += 1
            try:
                updated = apply_transaction(state_by_key[key], patch.transaction)
            except DivergeContractError:
                updated = None
            if (
                updated is not None
                and integer_bit_growth(updated.record()).max_magnitude_bits
                > packet.caps.max_integer_bits
            ):
                return ExecutionReceipt((), unique_transactions, duplicated_transactions, True)
            for assignment in group:
                states[assignment] = updated
    worlds = tuple(
        WorldResult(
            assignment,
            assignment_mass(packet, assignment),
            states[assignment],
            states[assignment] is None,
        )
        for assignment in assignments
    )
    return ExecutionReceipt(worlds, unique_transactions, duplicated_transactions, False)


def append_verified_nogood(
    packet: EpistemicPacket,
    nogood: VerifiedNogood,
) -> EpistemicPacket:
    """Append an assessor-issued certificate without running a hidden verifier.

    The candidate runtime deliberately has no access to valid assignments or raw
    evidence. The independent assessor is responsible for constructing the
    certificate and proving that it cannot remove a valid world.
    """

    if packet.overflow:
        return packet
    return build_packet(
        source_commitment=packet.source_commitment,
        shared_state=packet.shared_state,
        variables=packet.variables,
        hard_factors=packet.hard_factors,
        support_factors=packet.support_factors,
        patches=packet.patches,
        nogoods=packet.nogoods + (nogood,),
        caps=packet.caps,
    )


@dataclass(frozen=True)
class Query:
    opcode: str
    arguments: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.opcode not in {"READ_VALUE", "READ_TYPE", "HAS_EDGE", "SUM_VALUES", "EDGE_COUNT"}:
            raise DivergeContractError("unknown query opcode")
        arguments = tuple(_exact_int(value, "query argument") for value in self.arguments)
        expected = {
            "READ_VALUE": 1,
            "READ_TYPE": 1,
            "HAS_EDGE": 3,
            "SUM_VALUES": None,
            "EDGE_COUNT": 0,
        }[self.opcode]
        if expected is not None and len(arguments) != expected:
            raise DivergeContractError("query arity differs from opcode")
        if self.opcode == "SUM_VALUES" and not arguments:
            raise DivergeContractError("SUM_VALUES requires one or more slots")
        object.__setattr__(self, "arguments", arguments)


def read_query(state: TypedState, query: Query) -> int:
    cells = _state_cells(state)

    def cell(slot: int) -> TypedCell:
        if slot not in cells or not cells[slot].live:
            raise DivergeContractError("query references a missing or dead slot")
        return cells[slot]

    if query.opcode == "READ_VALUE":
        return cell(query.arguments[0]).value
    if query.opcode == "READ_TYPE":
        return cell(query.arguments[0]).type_id
    if query.opcode == "HAS_EDGE":
        return int(TypedEdge(*query.arguments) in state.edges)
    if query.opcode == "SUM_VALUES":
        return sum(cell(slot).value for slot in query.arguments)
    if query.opcode == "EDGE_COUNT":
        return len(state.edges)
    raise DivergeContractError("unknown query")  # pragma: no cover


@dataclass(frozen=True)
class QueryDecision:
    disposition: str
    answer: int | None
    marginals: tuple[tuple[int, int], ...]
    total_mass: int

    def probability(self, answer: int) -> Fraction:
        if self.total_mass <= 0:
            return Fraction(0, 1)
        return Fraction(dict(self.marginals).get(answer, 0), self.total_mass)


def query_execution(receipt: ExecutionReceipt, query: Query) -> QueryDecision:
    if receipt.overflow:
        return QueryDecision(OVERFLOW, None, (), 0)
    if not receipt.worlds or any(world.contradiction for world in receipt.worlds):
        return QueryDecision(REJECT, None, (), 0)
    masses: dict[int, int] = {}
    for world in receipt.worlds:
        assert world.state is not None
        answer = read_query(world.state, query)
        masses[answer] = masses.get(answer, 0) + world.mass
    marginals = tuple(sorted(masses.items()))
    total = sum(masses.values())
    if len(marginals) == 1:
        return QueryDecision(ANSWER, marginals[0][0], marginals, total)
    return QueryDecision(ABSTAIN, None, marginals, total)


def structural_merge_classes(
    packet: EpistemicPacket,
    *,
    after_patches: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Merge only identical states with identical remaining guarded behavior."""

    receipt = execute_packet(packet, patch_limit=after_patches)
    if receipt.overflow:
        return ()
    remaining = packet.patches[after_patches:]
    groups: dict[bytes, list[tuple[int, ...]]] = {}
    for world in receipt.worlds:
        if world.state is None:
            key_record: object = {"contradiction": True}
        else:
            key_record = {
                "state": world.state.record(),
                "remaining_guard_truth": [
                    int(patch.guard.matches(world.assignment)) for patch in remaining
                ],
            }
        key = canonical_json_bytes(key_record)
        groups.setdefault(key, []).append(world.assignment)
    return tuple(tuple(sorted(group)) for _, group in sorted(groups.items()))


@dataclass(frozen=True)
class MergedWorld:
    assignments: tuple[tuple[int, ...], ...]
    mass: int
    state: TypedState | None
    contradiction: bool

    def record(self) -> dict[str, object]:
        return {
            "assignments": [list(item) for item in self.assignments],
            "mass": self.mass,
            "state": None if self.state is None else self.state.record(),
            "contradiction": self.contradiction,
        }


def merge_certified_classes(
    packet: EpistemicPacket,
    *,
    after_patches: int,
    certified_classes: Iterable[Iterable[tuple[int, ...]]],
) -> tuple[MergedWorld, ...]:
    """Merge only assessor-certified classes that are also structurally safe."""

    receipt = execute_packet(packet, patch_limit=after_patches)
    if receipt.overflow:
        return ()
    support = {world.assignment for world in receipt.worlds}
    classes = tuple(
        tuple(sorted({tuple(assignment) for assignment in group}))
        for group in certified_classes
    )
    flattened = [assignment for group in classes for assignment in group]
    if any(not group for group in classes):
        raise DivergeContractError("merge certificate contains an empty class")
    if len(flattened) != len(set(flattened)) or set(flattened) != support:
        raise DivergeContractError("merge certificate must partition exact support")
    allowed = {
        frozenset(group)
        for group in structural_merge_classes(packet, after_patches=after_patches)
    }
    by_assignment = {world.assignment: world for world in receipt.worlds}
    merged = []
    for group in classes:
        if frozenset(group) not in allowed:
            raise DivergeContractError("merge certificate crosses a structural class")
        worlds = [by_assignment[assignment] for assignment in group]
        representative = worlds[0]
        merged.append(
            MergedWorld(
                assignments=group,
                mass=sum(world.mass for world in worlds),
                state=representative.state,
                contradiction=representative.contradiction,
            )
        )
    return tuple(sorted(merged, key=lambda item: item.assignments))


def packet_record(packet: EpistemicPacket) -> dict[str, object]:
    if packet.overflow:
        return {
            "schema": SCHEMA,
            "source_commitment": packet.source_commitment,
            "shared_state": packet.shared_state.record(),
            "overflow": True,
            "overflow_reason": packet.overflow_reason,
            "caps": dict(sorted(packet.caps.__dict__.items())),
        }
    return {
        "schema": SCHEMA,
        "source_commitment": packet.source_commitment,
        "shared_state": packet.shared_state.record(),
        "variables": [
            {
                "variable_id": item.variable_id,
                "options": list(item.options),
                "provenance": item.provenance,
            }
            for item in packet.variables
        ],
        "hard_factors": [_factor_record(item) for item in packet.hard_factors],
        "support_factors": [_support_record(item) for item in packet.support_factors],
        "patches": [_patch_record(item) for item in packet.patches],
        "nogoods": [_nogood_record(item) for item in packet.nogoods],
        "caps": dict(sorted(packet.caps.__dict__.items())),
        "overflow": False,
    }


def packet_bytes(packet: EpistemicPacket) -> bytes:
    return canonical_json_bytes(packet_record(packet))


def packet_commitment(packet: EpistemicPacket) -> str:
    return _commit("diverge-v0-packet", packet_record(packet))


@dataclass(frozen=True)
class PacketAccounting:
    represented_worlds: int
    packet_bytes: int
    materialized_world_bytes: int
    unique_transactions: int
    duplicated_transactions: int
    shared_transactions: int
    worlds_per_packet_byte: Fraction
    worlds_per_materialized_byte: Fraction
    integer_max_bits: int


def _materialized_world_record(
    packet: EpistemicPacket,
    world: WorldResult,
) -> dict[str, object]:
    """Serialize one complete whole-particle control with no shared packet fields."""

    return {
        "source_commitment": packet.source_commitment,
        "assignment": [
            {
                "fault_line_provenance": variable.provenance,
                "option_commitment": variable.options[world.assignment[index]],
            }
            for index, variable in enumerate(packet.variables)
        ],
        "initial_state": packet.shared_state.record(),
        "terminal_state": None if world.state is None else world.state.record(),
        "contradiction": world.contradiction,
        "mass": world.mass,
        "factor_provenance": [
            item.provenance for item in (*packet.hard_factors, *packet.support_factors)
        ],
        "evidence_provenance": [
            {
                "evidence_commitment": item.evidence_commitment,
                "verifier_commitment": item.verifier_commitment,
            }
            for item in packet.nogoods
        ],
        "program": [
            {
                "index": patch.index,
                "transaction": patch.transaction.record(),
                "provenance": patch.provenance,
            }
            for patch in packet.patches
            if patch.guard.matches(world.assignment)
        ],
    }


def materialized_world_bytes(packet: EpistemicPacket, world: WorldResult) -> int:
    """Return the canonical storage charged to one complete particle."""

    if world.assignment not in enumerate_assignments(packet):
        raise DivergeContractError("materialized world is outside packet support")
    return len(canonical_json_bytes(_materialized_world_record(packet, world)))


def account_packet(packet: EpistemicPacket, receipt: ExecutionReceipt) -> PacketAccounting:
    serialized = packet_bytes(packet)
    world_bytes = sum(
        materialized_world_bytes(packet, world)
        for world in receipt.worlds
    )
    worlds = len(receipt.worlds)
    return PacketAccounting(
        represented_worlds=worlds,
        packet_bytes=len(serialized),
        materialized_world_bytes=world_bytes,
        unique_transactions=receipt.unique_transactions,
        duplicated_transactions=receipt.duplicated_transactions,
        shared_transactions=receipt.shared_transactions,
        worlds_per_packet_byte=Fraction(worlds, max(1, len(serialized))),
        worlds_per_materialized_byte=Fraction(worlds, max(1, world_bytes)),
        integer_max_bits=integer_bit_growth(packet_record(packet)).max_magnitude_bits,
    )
