#!/usr/bin/env python3
"""Independent board and exact assessor for DIVERGE-ULC1."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from version_space_accounting import canonical_json_bytes

from diverge_ulc1 import (
    ACTIVE_LEFT,
    ACTIVE_RIGHT,
    BACKGROUND,
    CertifiedObservation,
    DelayedObservation,
    ParseAlternative,
    RecordLattice,
    SealedULC1Packet,
    selected_parse_record,
)
from diverge_v0 import (
    ANSWER,
    ABSTAIN,
    DivergeContractError,
    FaultLine,
    Guard,
    GuardedPatch,
    HardFactor,
    Literal,
    PacketCaps,
    Query,
    QueryDecision,
    SupportFactor,
    TypedCell,
    TypedState,
    TypedTransaction,
    WorldResult,
    assignment_mass,
    build_packet,
    enumerate_assignments,
    factorized_query_execution,
    named_commitment,
)
from diverge_v0_reference import reference_execute, reference_query, verify_nogood

BOARD_SCHEMA = "shohin-diverge-ulc1-delayed-board-v1"


def _digest(domain: str, payload: object) -> str:
    body = canonical_json_bytes(payload)
    digest = hashlib.sha256()
    for part in (domain.encode("ascii"), body):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


@dataclass(frozen=True)
class ULC1Episode:
    episode_id: str
    split: str
    ontology: str
    renderer: str
    source_text: str
    sealed: SealedULC1Packet
    gold_assignment: tuple[int, ...]
    initial_top1: tuple[int, ...]
    observations: tuple[DelayedObservation, ...]
    sensitive_query: Query
    invariant_query: Query
    underdetermined_query: Query
    record_count: int
    command_depth: int

    @property
    def represented_worlds(self) -> int:
        return len(enumerate_assignments(self.sealed.packet))

    def public_record(self) -> dict[str, object]:
        return {
            "schema": BOARD_SCHEMA,
            "episode_id": self.episode_id,
            "split": self.split,
            "ontology": self.ontology,
            "renderer": self.renderer,
            "source_commitment": self.sealed.packet.source_commitment,
            "record_count": self.record_count,
            "command_depth": self.command_depth,
            "represented_worlds": self.represented_worlds,
            "observation_commitments": [
                item.evidence_commitment for item in self.observations
            ],
        }


ONTOLOGY_PREAMBLES = {
    "register-workshop": "Keep each disputed register note unresolved until inspection.",
    "parcel-network": "Preserve every coherent parcel routing interpretation.",
    "molecular-switch": "Track competing switch-state readings without blending them.",
}

RENDERER_PHRASES = {
    "calibration-ledger": ("may be background", "left procedure", "right procedure"),
    "calibration-brief": ("possibly inert", "first protocol", "second protocol"),
    "development-manifest": ("can be commentary", "alpha route", "beta route"),
    "development-note": ("may not execute", "near branch", "far branch"),
    "confirmation-diagram": ("could be annotation", "upper path", "lower path"),
    "confirmation-report": ("might be nonoperative", "cold mode", "hot mode"),
}


def _variable_id(sealed: SealedULC1Packet, provenance: str) -> int:
    return sealed.variable_id(provenance)


def _world_count(record_count: int) -> int:
    return (2, 6, 16, 42)[record_count - 1]


def build_ulc1_episode(
    *,
    seed: int,
    serial: int,
    split: str,
    ontology: str,
    renderer: str,
    record_count: int,
    extra_depth: int = 0,
) -> ULC1Episode:
    """Build one exact episode with a wrong high-support parse and late evidence."""

    if split not in {"calibration", "development", "confirmation"}:
        raise ValueError("unknown split")
    if ontology not in ONTOLOGY_PREAMBLES:
        raise ValueError("unknown ontology")
    if renderer not in RENDERER_PHRASES:
        raise ValueError("unknown renderer")
    if record_count < 1 or record_count > 4:
        raise ValueError("ULC1 board uses one through four records")
    if extra_depth < 0 or extra_depth % 2:
        raise ValueError("extra depth must be nonnegative and even")

    rng = random.Random((seed << 20) ^ serial)
    episode_key = {
        "seed": seed,
        "serial": serial,
        "split": split,
        "ontology": ontology,
        "renderer": renderer,
        "record_count": record_count,
        "extra_depth": extra_depth,
    }
    episode_id = _digest("diverge-ulc1-episode", episode_key)[:20]
    background_phrase, left_phrase, right_phrase = RENDERER_PHRASES[renderer]

    source_records = []
    variables = []
    hard_factors = []
    support_factors = []
    lattices = []
    patches = []
    raw_variable_provenance: list[str] = []
    witness_slots = []

    def add_patch(
        guard: Guard, transaction: TypedTransaction, record_index: int
    ) -> None:
        index = len(patches)
        patches.append(
            GuardedPatch(
                index,
                guard,
                transaction,
                named_commitment(
                    "diverge-ulc1-patch",
                    f"{episode_id}:{record_index}:{index}",
                ),
            )
        )

    for record_index in range(record_count):
        interpretation_id = record_index
        record_provenance = named_commitment(
            "diverge-ulc1-record", f"{episode_id}:{record_index}"
        )
        interpretation_provenance = named_commitment(
            "diverge-ulc1-interpretation-variable", f"{episode_id}:{record_index}"
        )
        raw_variable_provenance.append(interpretation_provenance)
        domain_interpretations = (
            (ACTIVE_LEFT, ACTIVE_RIGHT)
            if record_index == 0
            else (BACKGROUND, ACTIVE_LEFT, ACTIVE_RIGHT)
        )
        variables.append(
            FaultLine(
                interpretation_id,
                tuple(
                    named_commitment(
                        "diverge-ulc1-interpretation-value",
                        f"{episode_id}:{record_index}:{interpretation}",
                    )
                    for interpretation in domain_interpretations
                ),
                interpretation_provenance,
            )
        )
        support_by_interpretation = {
            BACKGROUND: 20,
            ACTIVE_LEFT: 5,
            ACTIVE_RIGHT: 1,
        }
        support_factors.append(
            SupportFactor(
                (interpretation_id,),
                tuple(
                    ((domain_value,), support_by_interpretation[interpretation])
                    for domain_value, interpretation in enumerate(
                        domain_interpretations
                    )
                ),
                named_commitment(
                    "diverge-ulc1-interpretation-support",
                    f"{episode_id}:{record_index}",
                ),
            )
        )
        # This sparse circuit is nontrivial only from the third record onward:
        # a background predecessor cannot license an ACTIVE_LEFT successor.
        if record_index >= 2:
            previous_domain = (BACKGROUND, ACTIVE_LEFT, ACTIVE_RIGHT)
            current_domain = domain_interpretations
            allowed = tuple(
                (left, right)
                for left, left_interpretation in enumerate(previous_domain)
                for right, right_interpretation in enumerate(current_domain)
                if not (
                    left_interpretation == BACKGROUND
                    and right_interpretation == ACTIVE_LEFT
                )
            )
            hard_factors.append(
                HardFactor(
                    (record_index - 1, record_index),
                    allowed,
                    named_commitment(
                        "diverge-ulc1-guard-circuit",
                        f"{episode_id}:{record_index - 1}:{record_index}",
                    ),
                )
            )

        alias = f"{split[:3]}-{record_index}-{rng.randrange(10_000_000):07d}"
        record_text = (
            f"{alias}: {background_phrase}; {left_phrase} changes the accumulator before exchange; "
            f"{right_phrase} exchanges before changing the accumulator."
        )
        source_records.append(record_text)
        record_source_commitment = _digest("diverge-ulc1-source-record", record_text)
        occurrences = tuple(
            named_commitment(
                "diverge-ulc1-occurrence", f"{episode_id}:{record_index}:{index}"
            )
            for index in range(3)
        )
        phase_base = 2 + record_index
        alternatives = tuple(
            ParseAlternative(
                interpretation=interpretation,
                membership=membership,
                option=option,
                phase_cuts=(phase_base, phase_base + 2, phase_base + 4),
                cue_kind=interpretation,
                semantic_template=16 * record_index + interpretation,
                occurrence_commitments=occurrences,
                support_mass=mass,
                provenance=named_commitment(
                    "diverge-ulc1-parse-alternative",
                    f"{episode_id}:{record_index}:{interpretation}",
                ),
            )
            for interpretation, membership, option, mass in (
                (BACKGROUND, 0, 0, 20),
                (ACTIVE_LEFT, 1, 0, 5),
                (ACTIVE_RIGHT, 1, 1, 1),
            )
        )
        lattices.append(
            RecordLattice(
                record_index,
                record_source_commitment,
                record_provenance,
                interpretation_provenance,
                domain_interpretations,
                alternatives,
            )
        )

        witness_slot = 3 + record_index
        witness_slots.append(witness_slot)
        delta = 2 + (record_index % 3)
        left_value = domain_interpretations.index(ACTIVE_LEFT)
        right_value = domain_interpretations.index(ACTIVE_RIGHT)
        left_guard = Guard((Literal(interpretation_id, left_value),))
        right_guard = Guard((Literal(interpretation_id, right_value),))
        # Both active branches preserve the global sum but differ because ADD
        # and SWAP do not commute. Background leaves the state unchanged.
        add_patch(left_guard, TypedTransaction("ADD_VALUE", (0, delta)), record_index)
        add_patch(left_guard, TypedTransaction("SWAP_VALUE", (0, 1)), record_index)
        add_patch(left_guard, TypedTransaction("ADD_VALUE", (2, -delta)), record_index)
        add_patch(
            left_guard, TypedTransaction("SET_VALUE", (witness_slot, 1)), record_index
        )
        add_patch(right_guard, TypedTransaction("SWAP_VALUE", (0, 1)), record_index)
        add_patch(right_guard, TypedTransaction("ADD_VALUE", (0, delta)), record_index)
        add_patch(right_guard, TypedTransaction("ADD_VALUE", (2, -delta)), record_index)
        add_patch(
            right_guard, TypedTransaction("SET_VALUE", (witness_slot, 2)), record_index
        )

    for depth_index in range(extra_depth // 2):
        add_patch(
            Guard(), TypedTransaction("SWAP_VALUE", (1, 2)), record_count + depth_index
        )
        add_patch(
            Guard(), TypedTransaction("SWAP_VALUE", (1, 2)), record_count + depth_index
        )

    source_text = ONTOLOGY_PREAMBLES[ontology] + "\n" + "\n".join(source_records)
    source_commitment = _digest("diverge-ulc1-source", source_text)
    shared_state = TypedState(
        tuple(
            [TypedCell(0, 0, 3), TypedCell(1, 0, 11), TypedCell(2, 0, 29)]
            + [TypedCell(slot, 1, 0) for slot in witness_slots]
        )
    )
    packet = build_packet(
        source_commitment=source_commitment,
        shared_state=shared_state,
        variables=variables,
        hard_factors=hard_factors,
        support_factors=support_factors,
        patches=patches,
        caps=PacketCaps(
            max_variables=4,
            max_worlds=64,
            max_patches=40,
            max_guard_literals=4,
            max_cells=8,
            max_factor_rows=64,
        ),
    )
    if packet.overflow:
        raise AssertionError(
            f"calibrated ULC1 episode overflowed: {packet.overflow_reason}"
        )
    sealed = SealedULC1Packet(packet, tuple(lattices))
    gold = [0] * len(packet.variables)
    observations = []
    for record_index, interpretation_provenance in enumerate(raw_variable_provenance):
        record = lattices[record_index]
        gold[_variable_id(sealed, interpretation_provenance)] = (
            record.domain_interpretations.index(ACTIVE_RIGHT)
        )
        observations.append(
            DelayedObservation(
                source_commitment,
                lattices[record_index].record_provenance,
                witness_slots[record_index],
                2,
                _digest(
                    "diverge-ulc1-delayed-state-observation",
                    {
                        "episode": episode_id,
                        "record": record_index,
                        "slot": witness_slots[record_index],
                        "value": 2,
                    },
                ),
            )
        )
    gold_assignment = tuple(gold)
    support = enumerate_assignments(packet)
    if len(support) != _world_count(record_count):
        raise AssertionError("ULC1 coherence circuit represents the wrong world count")
    if gold_assignment not in support:
        raise AssertionError("gold interpretation disappeared during packet sealing")
    initial_top1 = min(support, key=lambda item: (-assignment_mass(packet, item), item))
    if initial_top1 == gold_assignment:
        raise AssertionError("calibrated ULC1 episode does not begin with wrong top-1")
    initial_reference = reference_execute(packet)
    if (
        len(
            {
                world.state.cells[0].value
                for world in initial_reference.worlds
                if world.state
            }
        )
        < 2
    ):
        raise AssertionError("underdetermined query does not separate hypotheses")
    return ULC1Episode(
        episode_id=episode_id,
        split=split,
        ontology=ontology,
        renderer=renderer,
        source_text=source_text,
        sealed=sealed,
        gold_assignment=gold_assignment,
        initial_top1=initial_top1,
        observations=tuple(observations),
        sensitive_query=Query("READ_VALUE", (0,)),
        invariant_query=Query("SUM_VALUES", (0, 1, 2)),
        underdetermined_query=Query("READ_VALUE", (0,)),
        record_count=record_count,
        command_depth=4 * record_count + extra_depth,
    )


def build_ulc1_board(
    seed: int = 202608057400, episodes: int = 1024
) -> tuple[ULC1Episode, ...]:
    if episodes < 16 or episodes % 4:
        raise ValueError("board size must be a multiple of four and at least sixteen")
    calibration = episodes // 2
    development = episodes // 4
    records = []
    for serial in range(episodes):
        if serial < calibration:
            split = "calibration"
            ontology = "register-workshop"
            renderer = ("calibration-ledger", "calibration-brief")[serial % 2]
            record_count = 1 + (serial % 2)
            extra_depth = 0
        elif serial < calibration + development:
            split = "development"
            ontology = "parcel-network"
            renderer = ("development-manifest", "development-note")[serial % 2]
            record_count = 3
            extra_depth = 0
        else:
            split = "confirmation"
            ontology = "molecular-switch"
            renderer = ("confirmation-diagram", "confirmation-report")[serial % 2]
            record_count = 4
            extra_depth = 2
        records.append(
            build_ulc1_episode(
                seed=seed,
                serial=serial,
                split=split,
                ontology=ontology,
                renderer=renderer,
                record_count=record_count,
                extra_depth=extra_depth,
            )
        )
    return tuple(records)


def board_commitment(episodes: tuple[ULC1Episode, ...]) -> str:
    return _digest("diverge-ulc1-board", [item.public_record() for item in episodes])


def certify_observation(
    sealed: SealedULC1Packet,
    observation: DelayedObservation,
) -> tuple[CertifiedObservation, ...]:
    """Derive and prove exact one-literal conflict cores by enumeration."""

    if observation.source_commitment != sealed.packet.source_commitment:
        raise DivergeContractError("observation belongs to a different source packet")
    record = sealed.record_lattice(observation.record_provenance)
    variable = sealed.variable_id(record.interpretation_provenance)
    right_value = record.domain_interpretations.index(ACTIVE_RIGHT)
    reference = reference_execute(sealed.packet)
    valid = []
    rejected = []
    for world in reference.worlds:
        if world.state is None:
            rejected.append(world.assignment)
            continue
        cells = {item.slot: item for item in world.state.cells}
        cell = cells.get(observation.state_slot)
        if cell is not None and cell.live and cell.value == observation.observed_value:
            valid.append(world.assignment)
        else:
            rejected.append(world.assignment)
    certificates = []
    covered = set()
    for domain_value in range(len(record.domain_interpretations)):
        if domain_value == right_value:
            continue
        proposed_guard = Guard((Literal(variable, domain_value),))
        matched = {
            item
            for item in enumerate_assignments(sealed.packet)
            if proposed_guard.matches(item)
        }
        if not matched:
            continue
        if not matched.issubset(set(rejected)):
            raise DivergeContractError(
                "proposed conflict core rejects observation-consistent worlds"
            )
        verification = verify_nogood(
            sealed.packet,
            guard=proposed_guard,
            evidence_commitment=observation.evidence_commitment,
            valid_assignments=valid,
        )
        if (
            not verification.accepted
            or verification.nogood is None
            or not verification.deletion_minimal
        ):
            raise DivergeContractError(
                f"independent conflict verifier rejected core: {verification.reason}"
            )
        certificates.append(
            CertifiedObservation(
                observation,
                verification.nogood,
                len(reference.worlds),
                verification.removed_worlds,
            )
        )
        covered.update(matched)
    if covered != set(rejected):
        raise DivergeContractError(
            "verified conflict cores do not explain every rejected world"
        )
    return tuple(certificates)


def _domains_and_strides(
    sealed: SealedULC1Packet,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    domains = tuple(len(item.options) for item in sealed.packet.variables)
    strides = tuple(math.prod(domains[index + 1 :]) for index in range(len(domains)))
    return domains, strides


def _assignment_from_index(
    index: int, domains: tuple[int, ...], strides: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        (index // stride) % domain
        for domain, stride in zip(domains, strides, strict=True)
    )


def factorized_world_records(execution) -> tuple[dict[str, object], ...]:
    """Expand only in the assessor to compare the packed runtime with enumeration."""

    sealed = execution.sealed
    domains, strides = _domains_and_strides(sealed)
    worlds = []
    seen = set()
    for group in execution.receipt.groups:
        remaining = group.support_mask
        while remaining:
            bit = remaining & -remaining
            index = bit.bit_length() - 1
            assignment = _assignment_from_index(index, domains, strides)
            if assignment in seen:
                raise AssertionError("factorized state groups overlap")
            seen.add(assignment)
            worlds.append(
                WorldResult(
                    assignment,
                    assignment_mass(sealed.packet, assignment),
                    group.state,
                    group.contradiction,
                ).record()
            )
            remaining ^= bit
    return tuple(sorted(worlds, key=lambda item: item["assignment"]))


def reference_world_records(sealed: SealedULC1Packet) -> tuple[dict[str, object], ...]:
    return tuple(
        sorted(
            (item.record() for item in reference_execute(sealed.packet).worlds),
            key=lambda item: item["assignment"],
        )
    )


def exact_factorized_parity(execution) -> bool:
    return factorized_world_records(execution) == reference_world_records(
        execution.sealed
    )


def materialized_particle_bytes(execution) -> int:
    """Charge every full-particle control for its selected parse and program."""

    sealed = execution.sealed
    reference = reference_execute(sealed.packet)
    total = 0
    for world in reference.worlds:
        program = [
            {
                "index": patch.index,
                "transaction": patch.transaction.record(),
                "provenance": patch.provenance,
            }
            for patch in sealed.packet.patches
            if patch.guard.matches(world.assignment)
        ]
        total += len(
            canonical_json_bytes(
                {
                    "source_commitment": sealed.packet.source_commitment,
                    "selected_parse": selected_parse_record(sealed, world.assignment),
                    "initial_state": sealed.packet.shared_state.record(),
                    "program": program,
                    "terminal_world": world.record(),
                    "factor_provenance": [
                        item.provenance
                        for item in (
                            *sealed.packet.hard_factors,
                            *sealed.packet.support_factors,
                        )
                    ],
                }
            )
        )
    return total


def expected_query_decisions(
    episode: ULC1Episode, refined: SealedULC1Packet
) -> dict[str, QueryDecision]:
    initial_invariant = reference_query(episode.sealed.packet, episode.invariant_query)
    initial_uncertain = reference_query(
        episode.sealed.packet, episode.underdetermined_query
    )
    sensitive = reference_query(refined.packet, episode.sensitive_query)
    if initial_invariant.disposition != ANSWER:
        raise AssertionError("calibrated invariant query is not invariant")
    if initial_uncertain.disposition != ABSTAIN:
        raise AssertionError("calibrated underdetermined query does not abstain")
    if sensitive.disposition != ANSWER:
        raise AssertionError("calibrated evidence does not resolve sensitive query")
    return {
        "sensitive": sensitive,
        "invariant": initial_invariant,
        "underdetermined": initial_uncertain,
    }


def query_decisions(execution, episode: ULC1Episode) -> dict[str, QueryDecision]:
    return {
        "sensitive": factorized_query_execution(
            execution.sealed.packet, execution.receipt, episode.sensitive_query
        ),
        "invariant": reference_query(episode.sealed.packet, episode.invariant_query),
        "underdetermined": reference_query(
            episode.sealed.packet, episode.underdetermined_query
        ),
    }
