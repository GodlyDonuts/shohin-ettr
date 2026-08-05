#!/usr/bin/env python3
"""Deterministic Delayed Disambiguation/Recovery board for DIVERGE-v0."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from version_space_accounting import canonical_json_bytes

from diverge_v0 import (
    EpistemicPacket,
    FaultLine,
    Guard,
    GuardedPatch,
    Literal,
    PacketCaps,
    Query,
    SupportFactor,
    TypedCell,
    TypedState,
    TypedTransaction,
    assignment_mass,
    build_packet,
    enumerate_assignments,
    named_commitment,
)


BOARD_SCHEMA = "shohin-diverge-delayed-board-v1"


@dataclass(frozen=True)
class DelayedEvidence:
    reject_guard: Guard
    valid_assignments: tuple[tuple[int, ...], ...]
    evidence_commitment: str


@dataclass(frozen=True)
class DelayedEpisode:
    episode_id: str
    split: str
    ontology: str
    renderer: str
    source_text: str
    packet: EpistemicPacket
    gold_assignment: tuple[int, ...]
    initial_top1: tuple[int, ...]
    evidence: DelayedEvidence
    sensitive_query: Query
    invariant_query: Query
    underdetermined_query: Query
    command_depth: int

    @property
    def represented_worlds(self) -> int:
        return len(enumerate_assignments(self.packet))

    def public_record(self) -> dict[str, object]:
        """Return assessor metadata without raw source or gold support."""

        return {
            "schema": BOARD_SCHEMA,
            "episode_id": self.episode_id,
            "split": self.split,
            "ontology": self.ontology,
            "renderer": self.renderer,
            "represented_worlds": self.represented_worlds,
            "command_depth": self.command_depth,
            "source_commitment": self.packet.source_commitment,
            "evidence_commitment": self.evidence.evidence_commitment,
        }


ONTOLOGY_TEXT = {
    "register-workshop": (
        "A workshop note leaves several role bindings unresolved. "
        "Execute the sealed register instructions only after the later inspection."
    ),
    "parcel-relation": (
        "A parcel ledger contains ambiguous aliases and relation bindings. "
        "Preserve every coherent ledger until the delayed scan arrives."
    ),
    "signal-routing": (
        "A signal-routing brief permits multiple switch interpretations. "
        "Keep the coherent routes separate until the late diagnostic packet."
    ),
}


def _digest(domain: str, payload: object) -> str:
    body = canonical_json_bytes(payload)
    digest = hashlib.sha256()
    for part in (domain.encode("ascii"), body):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _canonical_index(packet: EpistemicPacket, provenance: str) -> int:
    for variable in packet.variables:
        if variable.provenance == provenance:
            return variable.variable_id
    raise AssertionError("fault-line provenance disappeared during sealing")


def build_delayed_episode(
    *,
    split: str,
    ontology: str,
    renderer: str,
    width: int,
    serial: int,
    extra_depth: int = 0,
) -> DelayedEpisode:
    """Build one source-sealed episode with a deliberately wrong initial top-1."""

    if split not in {"calibration", "development", "confirmation"}:
        raise ValueError("unknown board split")
    if ontology not in ONTOLOGY_TEXT:
        raise ValueError("unknown board ontology")
    if width < 1 or width > 6:
        raise ValueError("board width must be one through six binary fault lines")
    if extra_depth < 0 or extra_depth % 2:
        raise ValueError("extra depth must be a nonnegative even number")

    episode_key = {
        "split": split,
        "ontology": ontology,
        "renderer": renderer,
        "width": width,
        "serial": serial,
        "extra_depth": extra_depth,
    }
    episode_id = _digest("diverge-board-episode", episode_key)[:20]
    source_text = (
        f"{ONTOLOGY_TEXT[ontology]} Renderer {renderer}; case {serial}; "
        f"ambiguity width {width}."
    )
    source_commitment = _digest("diverge-board-source", source_text)
    variables = []
    support_factors = []
    provenances = []
    for variable_id in range(width):
        provenance = named_commitment(
            "diverge-board-fault-line",
            f"{episode_id}:fault:{variable_id}",
        )
        provenances.append(provenance)
        variables.append(
            FaultLine(
                variable_id,
                (
                    named_commitment("diverge-board-option", f"{episode_id}:{variable_id}:0"),
                    named_commitment("diverge-board-option", f"{episode_id}:{variable_id}:1"),
                ),
                provenance,
            )
        )
        support_factors.append(
            SupportFactor(
                (variable_id,),
                (((0,), 3), ((1,), 1)),
                named_commitment(
                    "diverge-board-support",
                    f"{episode_id}:support:{variable_id}",
                ),
            )
        )

    state = TypedState(
        tuple(TypedCell(slot, 0, value) for slot, value in enumerate((1, 10, 20, 30, 40)))
    )
    patches = []

    def patch(guard: Guard, transaction: TypedTransaction) -> None:
        index = len(patches)
        patches.append(
            GuardedPatch(
                index,
                guard,
                transaction,
                named_commitment("diverge-board-patch", f"{episode_id}:patch:{index}"),
            )
        )

    primary_zero = Guard((Literal(0, 0),))
    primary_one = Guard((Literal(0, 1),))
    # These branches are deliberately noncommuting: ADD->SWAP differs from
    # SWAP->ADD while adding the same amount to the global value sum.
    patch(primary_zero, TypedTransaction("ADD_VALUE", (0, 3)))
    patch(primary_zero, TypedTransaction("SWAP_VALUE", (0, 1)))
    patch(primary_one, TypedTransaction("SWAP_VALUE", (0, 1)))
    patch(primary_one, TypedTransaction("ADD_VALUE", (0, 3)))
    for variable_id in range(1, width):
        patch(
            Guard((Literal(variable_id, 0),)),
            TypedTransaction("SWAP_VALUE", (2, 3)),
        )
        patch(
            Guard((Literal(variable_id, 1),)),
            TypedTransaction("SWAP_VALUE", (3, 4)),
        )
    for _ in range(extra_depth // 2):
        patch(Guard(), TypedTransaction("SWAP_VALUE", (3, 4)))
        patch(Guard(), TypedTransaction("SWAP_VALUE", (3, 4)))

    packet = build_packet(
        source_commitment=source_commitment,
        shared_state=state,
        variables=variables,
        support_factors=support_factors,
        patches=patches,
        caps=PacketCaps(max_patches=32),
    )
    primary = _canonical_index(packet, provenances[0])
    gold = tuple(1 for _ in packet.variables)
    support = enumerate_assignments(packet)
    top1 = min(support, key=lambda item: (-assignment_mass(packet, item), item))
    valid = tuple(item for item in support if item[primary] == 1)
    evidence = DelayedEvidence(
        reject_guard=Guard((Literal(primary, 0),)),
        valid_assignments=valid,
        evidence_commitment=_digest(
            "diverge-board-delayed-evidence",
            {"episode": episode_id, "primary_option": 1},
        ),
    )
    return DelayedEpisode(
        episode_id=episode_id,
        split=split,
        ontology=ontology,
        renderer=renderer,
        source_text=source_text,
        packet=packet,
        gold_assignment=gold,
        initial_top1=top1,
        evidence=evidence,
        sensitive_query=Query("READ_VALUE", (0,)),
        invariant_query=Query("SUM_VALUES", (0, 1, 2, 3, 4)),
        underdetermined_query=Query("READ_VALUE", (0,)),
        command_depth=width + 1 + extra_depth,
    )


def build_delayed_board(seed: int = 20260805) -> tuple[DelayedEpisode, ...]:
    """Build frozen calibration/development/confirmation coverage."""

    rng = random.Random(seed)
    specifications = (
        ("calibration", "register-workshop", (1, 2, 3), ("ledger-a", "ledger-b"), 0),
        ("development", "parcel-relation", (4, 5), ("manifest-x", "manifest-y"), 0),
        ("confirmation", "signal-routing", (6,), ("diagnostic-z", "diagnostic-q"), 2),
    )
    episodes = []
    serial = 0
    for split, ontology, widths, renderers, extra_depth in specifications:
        pairs = [(width, renderer) for width in widths for renderer in renderers]
        rng.shuffle(pairs)
        for width, renderer in pairs:
            episodes.append(
                build_delayed_episode(
                    split=split,
                    ontology=ontology,
                    renderer=renderer,
                    width=width,
                    serial=serial,
                    extra_depth=extra_depth,
                )
            )
            serial += 1
    return tuple(episodes)


def board_commitment(episodes: tuple[DelayedEpisode, ...]) -> str:
    return _digest(
        "diverge-delayed-board",
        [episode.public_record() for episode in episodes],
    )
