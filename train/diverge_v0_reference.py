#!/usr/bin/env python3
"""Independent enumerative assessor for DIVERGE-v0 packet semantics."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from version_space_accounting import canonical_json_bytes

from diverge_v0 import (
    ANSWER,
    ABSTAIN,
    OVERFLOW,
    REJECT,
    DivergeContractError,
    EpistemicPacket,
    Guard,
    Query,
    QueryDecision,
    TypedCell,
    TypedEdge,
    TypedState,
    TypedTransaction,
    VerifiedNogood,
)


@dataclass(frozen=True)
class ReferenceWorld:
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
class ReferenceExecution:
    worlds: tuple[ReferenceWorld, ...]
    overflow: bool


def _reference_guard(guard: Guard, assignment: tuple[int, ...]) -> bool:
    for literal in guard.literals:
        if literal.variable_id >= len(assignment):
            return False
        if assignment[literal.variable_id] != literal.option:
            return False
    return True


def reference_assignments(packet: EpistemicPacket) -> tuple[tuple[int, ...], ...]:
    """Enumerate support without calling the candidate packet enumerator."""

    if packet.overflow:
        return ()
    domains = [range(len(variable.options)) for variable in packet.variables]
    support = []
    for raw in itertools.product(*domains):
        assignment = tuple(raw)
        allowed = True
        for factor in packet.hard_factors:
            row = tuple(assignment[variable] for variable in factor.scope)
            if row not in factor.allowed:
                allowed = False
                break
        if not allowed:
            continue
        if any(_reference_guard(nogood.guard, assignment) for nogood in packet.nogoods):
            continue
        support.append(assignment)
    return tuple(support)


def _reference_mass(packet: EpistemicPacket, assignment: tuple[int, ...]) -> int:
    product = 1
    for factor in packet.support_factors:
        row = tuple(assignment[variable] for variable in factor.scope)
        match = 1
        for candidate, mass in factor.masses:
            if candidate == row:
                match = mass
                break
        product *= match
    return product


def _reference_apply(state: TypedState, transaction: TypedTransaction) -> TypedState:
    """Separate transaction interpreter used only by the assessor."""

    cells = {item.slot: item for item in state.cells}

    def get(slot: int) -> TypedCell:
        current = cells.get(slot)
        if current is None or not current.live:
            raise DivergeContractError("reference transaction targets unavailable slot")
        return current

    edges = set(state.edges)
    opcode = transaction.opcode
    arguments = transaction.arguments
    if opcode == "SET_VALUE":
        old = get(arguments[0])
        cells[arguments[0]] = TypedCell(old.slot, old.type_id, arguments[1], old.live)
    elif opcode == "ADD_VALUE":
        old = get(arguments[0])
        cells[arguments[0]] = TypedCell(
            old.slot, old.type_id, old.value + arguments[1], old.live
        )
    elif opcode == "COPY_VALUE":
        source = get(arguments[0])
        target = get(arguments[1])
        if source.type_id != target.type_id:
            raise DivergeContractError("reference COPY_VALUE type conflict")
        cells[arguments[1]] = replace(target, value=source.value)
    elif opcode == "SWAP_VALUE":
        left = get(arguments[0])
        right = get(arguments[1])
        if left.type_id != right.type_id:
            raise DivergeContractError("reference SWAP_VALUE type conflict")
        cells[arguments[0]] = replace(left, value=right.value)
        cells[arguments[1]] = replace(right, value=left.value)
    elif opcode == "SET_TYPE":
        old = get(arguments[0])
        if arguments[1] < 0:
            raise DivergeContractError("reference SET_TYPE negative type")
        cells[arguments[0]] = replace(old, type_id=arguments[1])
    elif opcode == "LINK":
        get(arguments[0])
        get(arguments[2])
        edge = TypedEdge(arguments[0], arguments[1], arguments[2])
        if edge in edges:
            raise DivergeContractError("reference LINK duplicate")
        edges.add(edge)
    elif opcode == "UNLINK":
        edge = TypedEdge(arguments[0], arguments[1], arguments[2])
        if edge not in edges:
            raise DivergeContractError("reference UNLINK missing edge")
        edges.remove(edge)
    else:
        raise DivergeContractError("reference unknown transaction")
    return TypedState(tuple(cells.values()), tuple(edges))


def reference_execute(
    packet: EpistemicPacket,
    *,
    patch_limit: int | None = None,
) -> ReferenceExecution:
    """Execute every complete world separately with no candidate-runtime calls."""

    if packet.overflow:
        return ReferenceExecution((), True)
    patches = packet.patches if patch_limit is None else packet.patches[:patch_limit]
    worlds = []
    for assignment in reference_assignments(packet):
        state: TypedState | None = packet.shared_state
        for patch in patches:
            if state is None or not _reference_guard(patch.guard, assignment):
                continue
            try:
                state = _reference_apply(state, patch.transaction)
            except DivergeContractError:
                state = None
            if (
                state is not None
                and max(
                    (
                        abs(cell.value).bit_length()
                        for cell in state.cells
                    ),
                    default=0,
                )
                > packet.caps.max_integer_bits
            ):
                return ReferenceExecution((), True)
        worlds.append(
            ReferenceWorld(
                assignment,
                _reference_mass(packet, assignment),
                state,
                state is None,
            )
        )
    return ReferenceExecution(tuple(worlds), False)


def _reference_read(state: TypedState, query: Query) -> int:
    cells = {item.slot: item for item in state.cells}

    def get(slot: int) -> TypedCell:
        current = cells.get(slot)
        if current is None or not current.live:
            raise DivergeContractError("reference query targets unavailable slot")
        return current

    if query.opcode == "READ_VALUE":
        return get(query.arguments[0]).value
    if query.opcode == "READ_TYPE":
        return get(query.arguments[0]).type_id
    if query.opcode == "HAS_EDGE":
        return int(TypedEdge(*query.arguments) in state.edges)
    if query.opcode == "SUM_VALUES":
        return sum(get(slot).value for slot in query.arguments)
    if query.opcode == "EDGE_COUNT":
        return len(state.edges)
    raise DivergeContractError("reference unknown query")


def reference_query(packet: EpistemicPacket, query: Query) -> QueryDecision:
    if packet.overflow:
        return QueryDecision(OVERFLOW, None, (), 0)
    receipt = reference_execute(packet)
    if receipt.overflow:
        return QueryDecision(OVERFLOW, None, (), 0)
    if not receipt.worlds or any(world.contradiction for world in receipt.worlds):
        return QueryDecision(REJECT, None, (), 0)
    masses: dict[int, int] = {}
    for world in receipt.worlds:
        assert world.state is not None
        answer = _reference_read(world.state, query)
        masses[answer] = masses.get(answer, 0) + world.mass
    marginal = tuple(sorted(masses.items()))
    total = sum(masses.values())
    if len(marginal) == 1:
        return QueryDecision(ANSWER, marginal[0][0], marginal, total)
    return QueryDecision(ABSTAIN, None, marginal, total)


@dataclass(frozen=True)
class ParityReport:
    exact: bool
    candidate_worlds: int
    reference_worlds: int
    mismatches: tuple[str, ...]


def compare_execution(candidate_receipt: object, packet: EpistemicPacket) -> ParityReport:
    reference = reference_execute(packet)
    candidate_overflow = bool(getattr(candidate_receipt, "overflow"))
    candidate_worlds = tuple(getattr(candidate_receipt, "worlds"))
    candidate_records = tuple(world.record() for world in candidate_worlds)
    reference_records = tuple(world.record() for world in reference.worlds)
    mismatches = []
    if candidate_overflow != reference.overflow:
        mismatches.append("overflow")
    if len(candidate_records) != len(reference_records):
        mismatches.append("world-count")
    for index, (candidate, expected) in enumerate(
        itertools.zip_longest(candidate_records, reference_records)
    ):
        if candidate != expected:
            mismatches.append(f"world-{index}")
    return ParityReport(
        exact=not mismatches,
        candidate_worlds=len(candidate_records),
        reference_worlds=len(reference_records),
        mismatches=tuple(mismatches),
    )


@dataclass(frozen=True)
class NogoodVerification:
    accepted: bool
    reason: str
    nogood: VerifiedNogood | None
    removed_worlds: int
    deletion_minimal: bool


def verify_nogood(
    packet: EpistemicPacket,
    *,
    guard: Guard,
    evidence_commitment: str,
    valid_assignments: Iterable[tuple[int, ...]],
) -> NogoodVerification:
    """Independently prove that a conflict core excludes no valid world."""

    if packet.overflow:
        return NogoodVerification(False, "packet-overflow", None, 0, False)
    support = set(reference_assignments(packet))
    valid = {tuple(item) for item in valid_assignments}
    if not valid or not valid.issubset(support):
        raise DivergeContractError("reference valid set is empty or outside support")
    removed = {item for item in support if _reference_guard(guard, item)}
    if not removed:
        return NogoodVerification(False, "vacuous-core", None, 0, False)
    if removed & valid:
        return NogoodVerification(
            False, "would-remove-valid-world", None, 0, False
        )
    minimal = True
    for index in range(len(guard.literals)):
        reduced = Guard(guard.literals[:index] + guard.literals[index + 1 :])
        if not any(_reference_guard(reduced, item) for item in valid):
            minimal = False
    evidence_commitment = str(evidence_commitment).lower()
    try:
        bytes.fromhex(evidence_commitment)
    except ValueError as error:
        raise DivergeContractError("reference evidence commitment is invalid") from error
    if len(evidence_commitment) != 64:
        raise DivergeContractError("reference evidence commitment is invalid")
    payload = {
        "evidence": evidence_commitment,
        "guard": guard.record(),
        "valid": [list(item) for item in sorted(valid)],
    }
    body = canonical_json_bytes(payload)
    verifier = hashlib.sha256(
        len(b"diverge-v0-reference-verifier").to_bytes(8, "big")
        + b"diverge-v0-reference-verifier"
        + len(body).to_bytes(8, "big")
        + body
    ).hexdigest()
    nogood = VerifiedNogood(guard, evidence_commitment, verifier, minimal)
    return NogoodVerification(True, "accepted", nogood, len(removed), minimal)


def reference_behavioral_classes(
    packet: EpistemicPacket,
    *,
    after_patches: int,
    queries: Sequence[Query],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Certify equivalence over the complete declared future patch/query universe."""

    prefix = {
        world.assignment: world
        for world in reference_execute(packet, patch_limit=after_patches).worlds
    }
    groups: dict[bytes, list[tuple[int, ...]]] = {}
    for assignment in reference_assignments(packet):
        initial = prefix[assignment]
        state = initial.state
        signatures: list[object] = []
        if state is None:
            signatures.append({"contradiction": True})
        else:
            signatures.append({"prefix_state": state.record()})
            for patch in packet.patches[after_patches:]:
                if _reference_guard(patch.guard, assignment):
                    try:
                        state = _reference_apply(state, patch.transaction)
                    except DivergeContractError:
                        state = None
                signatures.append(None if state is None else state.record())
            signatures.append(
                None
                if state is None
                else [_reference_read(state, query) for query in queries]
            )
        key = canonical_json_bytes(signatures)
        groups.setdefault(key, []).append(assignment)
    return tuple(tuple(sorted(group)) for _, group in sorted(groups.items()))
