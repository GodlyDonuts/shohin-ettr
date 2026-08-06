#!/usr/bin/env python3
"""Source-sealed uncertainty lifting for DIVERGE-ULC1.

ULC1 retains coherent parse alternatives as discrete guarded variables.  It
never averages fields across interpretations.  Independent assessors issue
verified nogoods; the candidate runtime can only intersect support and reuse
the already executed state partition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from version_space_accounting import canonical_json_bytes

from diverge_v0 import (
    DivergeContractError,
    EpistemicPacket,
    FactorizedExecutionReceipt,
    VerifiedNogood,
    append_verified_nogood,
    execute_packet_factorized,
    packet_record,
    refine_factorized_receipt,
    validate_commitment,
)

SCHEMA = "shohin-diverge-ulc1-packet-v1"
BACKGROUND = 0
ACTIVE_LEFT = 1
ACTIVE_RIGHT = 2


def _exact_nonnegative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DivergeContractError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class ParseAlternative:
    """One complete, internally coherent interpretation of a source record."""

    interpretation: int
    membership: int
    option: int
    phase_cuts: tuple[int, int, int]
    cue_kind: int
    semantic_template: int
    occurrence_commitments: tuple[str, ...]
    support_mass: int
    provenance: str

    def __post_init__(self) -> None:
        interpretation = _exact_nonnegative(self.interpretation, "interpretation")
        membership = _exact_nonnegative(self.membership, "membership")
        option = _exact_nonnegative(self.option, "option")
        expected = {
            BACKGROUND: (0, 0),
            ACTIVE_LEFT: (1, 0),
            ACTIVE_RIGHT: (1, 1),
        }
        if (
            interpretation not in expected
            or (membership, option) != expected[interpretation]
        ):
            raise DivergeContractError("parse alternative mixes incompatible fields")
        cuts = tuple(
            _exact_nonnegative(value, "phase cut") for value in self.phase_cuts
        )
        if len(cuts) != 3 or not cuts[0] < cuts[1] < cuts[2]:
            raise DivergeContractError(
                "phase cuts must be three strictly increasing offsets"
            )
        _exact_nonnegative(self.cue_kind, "cue kind")
        _exact_nonnegative(self.semantic_template, "semantic template")
        occurrences = tuple(
            validate_commitment(value, "occurrence commitment")
            for value in self.occurrence_commitments
        )
        if not occurrences or len(set(occurrences)) != len(occurrences):
            raise DivergeContractError(
                "parse alternative needs unique physical occurrences"
            )
        if isinstance(self.support_mass, bool) or not isinstance(
            self.support_mass, int
        ):
            raise DivergeContractError("support mass must be an exact integer")
        if self.support_mass <= 0:
            raise DivergeContractError("support mass must be positive")
        object.__setattr__(self, "phase_cuts", cuts)
        object.__setattr__(self, "occurrence_commitments", occurrences)
        object.__setattr__(
            self,
            "provenance",
            validate_commitment(self.provenance, "alternative provenance"),
        )

    def record(self) -> dict[str, object]:
        return {
            "interpretation": self.interpretation,
            "membership": self.membership,
            "option": self.option,
            "phase_cuts": list(self.phase_cuts),
            "cue_kind": self.cue_kind,
            "semantic_template": self.semantic_template,
            "occurrence_commitments": list(self.occurrence_commitments),
            "support_mass": self.support_mass,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class RecordLattice:
    """Packed source witness for one record; no raw source bytes are retained."""

    record_index: int
    source_commitment: str
    record_provenance: str
    interpretation_provenance: str
    domain_interpretations: tuple[int, ...]
    alternatives: tuple[ParseAlternative, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_index", _exact_nonnegative(self.record_index, "record index")
        )
        for field in (
            "source_commitment",
            "record_provenance",
            "interpretation_provenance",
        ):
            object.__setattr__(
                self, field, validate_commitment(getattr(self, field), field)
            )
        alternatives = tuple(
            sorted(self.alternatives, key=lambda item: item.interpretation)
        )
        if tuple(item.interpretation for item in alternatives) != (
            BACKGROUND,
            ACTIVE_LEFT,
            ACTIVE_RIGHT,
        ):
            raise DivergeContractError(
                "record lattice must contain exactly three coherent alternatives"
            )
        domain = tuple(
            _exact_nonnegative(value, "domain interpretation")
            for value in self.domain_interpretations
        )
        if len(domain) < 2 or len(set(domain)) != len(domain):
            raise DivergeContractError(
                "record domain needs at least two unique interpretations"
            )
        if any(
            value not in (BACKGROUND, ACTIVE_LEFT, ACTIVE_RIGHT) for value in domain
        ):
            raise DivergeContractError(
                "record domain contains an unknown interpretation"
            )
        object.__setattr__(self, "domain_interpretations", domain)
        object.__setattr__(self, "alternatives", alternatives)

    def selected(self, domain_value: int) -> ParseAlternative:
        if domain_value < 0 or domain_value >= len(self.domain_interpretations):
            raise DivergeContractError("record interpretation is outside its domain")
        return self.alternatives[self.domain_interpretations[domain_value]]

    def record(self) -> dict[str, object]:
        return {
            "record_index": self.record_index,
            "source_commitment": self.source_commitment,
            "record_provenance": self.record_provenance,
            "interpretation_provenance": self.interpretation_provenance,
            "domain_interpretations": list(self.domain_interpretations),
            "alternatives": [item.record() for item in self.alternatives],
        }


@dataclass(frozen=True)
class SealedULC1Packet:
    """A complete parse lattice coupled to an exact guarded execution packet."""

    packet: EpistemicPacket
    records: tuple[RecordLattice, ...]
    source_deleted: bool = True

    def __post_init__(self) -> None:
        if self.source_deleted is not True:
            raise DivergeContractError("ULC1 runtime must not retain raw source")
        records = tuple(sorted(self.records, key=lambda item: item.record_index))
        if not records or tuple(item.record_index for item in records) != tuple(
            range(len(records))
        ):
            raise DivergeContractError(
                "record lattices must be complete and chronologically indexed"
            )
        if len({item.record_provenance for item in records}) != len(records):
            raise DivergeContractError("record provenance must be unique")
        if not self.packet.overflow:
            variable_provenance = {item.provenance for item in self.packet.variables}
            required = {record.interpretation_provenance for record in records}
            if not required.issubset(variable_provenance):
                raise DivergeContractError(
                    "lattice variables are absent from sealed packet"
                )
        object.__setattr__(self, "records", records)

    def variable_id(self, provenance: str) -> int:
        provenance = validate_commitment(provenance, "variable provenance")
        matches = [
            item.variable_id
            for item in self.packet.variables
            if item.provenance == provenance
        ]
        if len(matches) != 1:
            raise DivergeContractError("sealed variable provenance is not unique")
        return matches[0]

    def record_lattice(self, provenance: str) -> RecordLattice:
        provenance = validate_commitment(provenance, "record provenance")
        matches = [
            item for item in self.records if item.record_provenance == provenance
        ]
        if len(matches) != 1:
            raise DivergeContractError("sealed record provenance is not unique")
        return matches[0]


@dataclass(frozen=True)
class DelayedObservation:
    """Post-source state evidence; it contains no answer or raw-language field."""

    source_commitment: str
    record_provenance: str
    state_slot: int
    observed_value: int
    evidence_commitment: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_commitment",
            validate_commitment(self.source_commitment, "observation source"),
        )
        object.__setattr__(
            self,
            "record_provenance",
            validate_commitment(self.record_provenance, "observation record"),
        )
        object.__setattr__(
            self, "state_slot", _exact_nonnegative(self.state_slot, "state slot")
        )
        if isinstance(self.observed_value, bool) or not isinstance(
            self.observed_value, int
        ):
            raise DivergeContractError("observed value must be an exact integer")
        object.__setattr__(
            self,
            "evidence_commitment",
            validate_commitment(self.evidence_commitment, "evidence"),
        )


@dataclass(frozen=True)
class CertifiedObservation:
    observation: DelayedObservation
    nogood: VerifiedNogood
    valid_worlds_before: int
    removed_worlds: int

    def __post_init__(self) -> None:
        _exact_nonnegative(self.valid_worlds_before, "valid worlds before")
        if _exact_nonnegative(self.removed_worlds, "removed worlds") <= 0:
            raise DivergeContractError("certified observation must remove support")
        if self.observation.evidence_commitment != self.nogood.evidence_commitment:
            raise DivergeContractError("observation and nogood evidence do not match")


@dataclass(frozen=True)
class ULC1Execution:
    sealed: SealedULC1Packet
    receipt: FactorizedExecutionReceipt
    certificates: tuple[CertifiedObservation, ...] = ()


def execute_ulc1(sealed: SealedULC1Packet) -> ULC1Execution:
    return ULC1Execution(sealed, execute_packet_factorized(sealed.packet))


def apply_certified_observation(
    execution: ULC1Execution,
    certificate: CertifiedObservation,
) -> ULC1Execution:
    """Apply one independently certified conflict without replay or source access."""

    if (
        certificate.observation.source_commitment
        != execution.sealed.packet.source_commitment
    ):
        raise DivergeContractError("packet/evidence source commitment mismatch")
    execution.sealed.record_lattice(certificate.observation.record_provenance)
    refined_packet = append_verified_nogood(execution.sealed.packet, certificate.nogood)
    if refined_packet.overflow:
        raise DivergeContractError("verified refinement overflowed the packet")
    refined_receipt = refine_factorized_receipt(refined_packet, execution.receipt)
    return ULC1Execution(
        replace(execution.sealed, packet=refined_packet),
        refined_receipt,
        execution.certificates + (certificate,),
    )


def packet_record_ulc1(sealed: SealedULC1Packet) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "source_deleted": True,
        "packet": packet_record(sealed.packet),
        "record_lattices": [item.record() for item in sealed.records],
    }


def packet_bytes_ulc1(sealed: SealedULC1Packet) -> bytes:
    return canonical_json_bytes(packet_record_ulc1(sealed))


def selected_parse_record(
    sealed: SealedULC1Packet,
    assignment: tuple[int, ...],
) -> list[dict[str, object]]:
    """Materialize the complete coherent parse selected by one control particle."""

    selected = []
    for record in sealed.records:
        value = assignment[sealed.variable_id(record.interpretation_provenance)]
        selected.append(record.selected(value).record())
    return selected


def apply_certificates(
    execution: ULC1Execution,
    certificates: Iterable[CertifiedObservation],
) -> ULC1Execution:
    for certificate in certificates:
        execution = apply_certified_observation(execution, certificate)
    return execution
