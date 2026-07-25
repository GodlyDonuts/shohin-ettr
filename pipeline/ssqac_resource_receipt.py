"""Exact, independently replayable resource receipts for SSQAC artifacts.

The receipt in this module measures structural work.  It does not trust the
quotient compiler and does not import it.  A quotient or consequence artifact
is first replayed by :mod:`pipeline.verify_ssqac_quotient_artifact`; all
resource counts are then reconstructed from the plain portable payload.

Primitive ALU accounting is accepted only when the receipt includes its full
canonical instruction stream.  An opaque trace digest or an aggregate
instruction count is not replay evidence and is rejected.

``primitive_cycles`` are abstract one-instruction issue cycles for the serial
SSQAC primitive VM.  They are not GPU/CPU clock cycles.  Wall time and peak
device memory are runtime observations.  An exact structural receipt cannot
derive or fabricate them, so both are always recorded as ``None`` with an
explicit external-observer requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Sequence

from pipeline.verify_ssqac_quotient_artifact import (
    ArtifactVerificationError,
    verify_outcome_artifact,
    verify_quotient_artifact,
)


RESOURCE_RECEIPT_SCHEMA = "ssqac_exact_resource_receipt_v1"
RESOURCE_VECTOR_SCHEMA = "ssqac_exact_resource_vector_v1"
RESOURCE_VERIFICATION_SCHEMA = "ssqac_exact_resource_verification_v1"
REPLAYABLE_ALU_RECEIPT_SCHEMA = "ssqac_replayable_primitive_alu_receipt_v1"

PRIMITIVE_OPCODES = (
    "LOAD",
    "INV",
    "NEG",
    "SCALE",
    "AXPY",
    "SWAP",
    "HALT",
)

PEAK_WORKSPACE_KEYS = (
    "artifact_bytes",
    "monomial_slots",
    "rref_nonzeros",
    "provenance_terms",
    "row_support",
    "provenance_support",
    "quotient_dimension",
    "primitive_instruction_slots",
)

DEFAULT_RESOURCE_CAPS: dict[str, int] = {
    "source_independent_artifact_bytes": 64 * 1024 * 1024,
    "variable_count": 128,
    "generator_count": 16_384,
    "polynomial_terms": 1_000_000,
    "main_degree": 8,
    "prolongation_degree": 9,
    "monomial_count": 4096,
    "rank": 4096,
    "rref_nonzeros": 16_000_000,
    "provenance_terms": 16_000_000,
    "row_support": 4096,
    "provenance_support": 1_000_000,
    "quotient_dimension": 256,
    "primitive_program_count": 64,
    "primitive_instructions": 1_000_000,
    "primitive_cycles": 1_000_000,
    "sequential_depth": 1_000_000,
}


class ResourceReceiptError(ValueError):
    """A resource artifact, declaration, or limit failed closed."""


@dataclass(frozen=True, slots=True)
class ResourceReceiptVerification:
    schema: str
    receipt_sha256: str
    artifact_sha256: str
    resource_vector_sha256: str
    artifact_kind: str
    gates: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.gates)


@dataclass(frozen=True, slots=True)
class _DerivedResources:
    artifact_kind: str
    artifact_sha256: str
    primitive_program_sha256s: tuple[str, ...]
    vector: dict[str, object]
    actual_peak_workspace: dict[str, int]
    declared_peak_workspace: dict[str, int]


def _fail(message: str) -> None:
    raise ResourceReceiptError(message)


def _plain_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be a plain integer")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        _fail(f"{label} keys must be strings")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{label} must be a sequence")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(f"{label} keys differ; missing={missing}, extra={extra}")


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ResourceReceiptError(
            "resource input is not canonical ASCII JSON data"
        ) from error


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _normalize_caps(overrides: Mapping[str, int] | None) -> dict[str, int]:
    caps = dict(DEFAULT_RESOURCE_CAPS)
    if overrides is None:
        return caps
    raw = _mapping(overrides, "resource caps")
    unknown = sorted(set(raw) - set(caps))
    if unknown:
        _fail(f"resource caps contain unknown keys: {unknown}")
    for key, raw_value in raw.items():
        value = _plain_int(raw_value, f"resource cap {key}")
        if value < 0:
            _fail(f"resource cap {key} must be nonnegative")
        caps[key] = value
    return caps


def _normalize_peak_workspace(
    value: Mapping[str, int],
    label: str,
) -> dict[str, int]:
    raw = _mapping(value, label)
    _exact_keys(raw, set(PEAK_WORKSPACE_KEYS), label)
    result: dict[str, int] = {}
    for key in PEAK_WORKSPACE_KEYS:
        bound = _plain_int(raw[key], f"{label} {key}")
        if bound < 0:
            _fail(f"{label} {key} must be nonnegative")
        result[key] = bound
    return result


def _runtime_observations() -> dict[str, object]:
    return {
        "device_peak_memory_bytes": None,
        "status": (
            "not-derived: wall time and device memory require an external "
            "runtime observer"
        ),
        "wall_time_seconds": None,
    }


def make_replayable_primitive_alu_receipt(
    instructions: Iterable[Sequence[object]],
) -> dict[str, object]:
    """Create a plain ALU receipt whose resource claims can be replayed.

    This helper certifies only the instruction-resource schedule.  Matrix
    semantics remain the responsibility of the primitive VM verifier.
    """

    normalized, counts = _normalize_instructions(tuple(instructions))
    cycles = len(normalized)
    program_payload = {
        "instructions": normalized,
        "schema": REPLAYABLE_ALU_RECEIPT_SCHEMA,
    }
    return {
        "cycles": cycles,
        "executed_instructions": cycles,
        "instructions": normalized,
        "opcode_counts": counts,
        "program_sha256": _sha256(program_payload),
        "schema": REPLAYABLE_ALU_RECEIPT_SCHEMA,
        "sequential_depth": cycles,
    }


def _normalize_instructions(
    instructions: Sequence[object],
) -> tuple[list[list[object]], dict[str, int]]:
    if not instructions:
        _fail("primitive ALU instruction stream must not be empty")
    normalized: list[list[object]] = []
    counts = {opcode: 0 for opcode in PRIMITIVE_OPCODES}
    for index, raw_instruction in enumerate(instructions):
        instruction = _sequence(raw_instruction, f"instruction {index}")
        if len(instruction) != 4:
            _fail(f"instruction {index} must contain opcode and three operands")
        opcode = instruction[0]
        if not isinstance(opcode, str) or opcode not in counts:
            _fail(f"instruction {index} has an unknown primitive opcode")
        operands = [
            _plain_int(instruction[position], f"instruction {index} operand")
            for position in range(1, 4)
        ]
        normalized.append([opcode, *operands])
        counts[opcode] += 1
    if counts["HALT"] != 1 or normalized[-1][0] != "HALT":
        _fail("primitive ALU stream must contain exactly one final HALT")
    return normalized, counts


def _replay_primitive_receipt(
    value: object,
    index: int,
) -> tuple[str, dict[str, int], int, int, int]:
    receipt = _mapping(value, f"primitive receipt {index}")
    _exact_keys(
        receipt,
        {
            "cycles",
            "executed_instructions",
            "instructions",
            "opcode_counts",
            "program_sha256",
            "schema",
            "sequential_depth",
        },
        f"primitive receipt {index}",
    )
    if receipt["schema"] != REPLAYABLE_ALU_RECEIPT_SCHEMA:
        _fail(
            f"primitive receipt {index} is not independently replayable; "
            "a full canonical instruction stream is required"
        )
    instructions, actual_counts = _normalize_instructions(
        _sequence(receipt["instructions"], f"primitive receipt {index} instructions")
    )
    claimed_counts = _mapping(
        receipt["opcode_counts"], f"primitive receipt {index} opcode counts"
    )
    _exact_keys(
        claimed_counts,
        set(PRIMITIVE_OPCODES),
        f"primitive receipt {index} opcode counts",
    )
    normalized_claimed: dict[str, int] = {}
    for opcode in PRIMITIVE_OPCODES:
        count = _plain_int(
            claimed_counts[opcode],
            f"primitive receipt {index} {opcode} count",
        )
        if count < 0:
            _fail(f"primitive receipt {index} opcode counts must be nonnegative")
        normalized_claimed[opcode] = count
    if normalized_claimed != actual_counts:
        _fail(f"primitive receipt {index} opcode counts differ from replay")
    instruction_count = len(instructions)
    if (
        _plain_int(
            receipt["executed_instructions"],
            f"primitive receipt {index} executed instructions",
        )
        != instruction_count
    ):
        _fail(f"primitive receipt {index} instruction count differs from replay")
    cycles = _plain_int(receipt["cycles"], f"primitive receipt {index} cycles")
    depth = _plain_int(
        receipt["sequential_depth"],
        f"primitive receipt {index} sequential depth",
    )
    if cycles != instruction_count:
        _fail(
            f"primitive receipt {index} abstract cycles differ from serial replay"
        )
    if depth != instruction_count:
        _fail(
            f"primitive receipt {index} sequential depth differs from serial replay"
        )
    program_payload = {
        "instructions": instructions,
        "schema": REPLAYABLE_ALU_RECEIPT_SCHEMA,
    }
    program_sha = _digest(
        receipt["program_sha256"],
        f"primitive receipt {index} program digest",
    )
    if program_sha != _sha256(program_payload):
        _fail(f"primitive receipt {index} program digest differs from replay")
    return _sha256(receipt), actual_counts, instruction_count, cycles, depth


def _row_resource_counts(
    rows_value: object,
    label: str,
) -> tuple[int, int, int, int]:
    rows = _sequence(rows_value, label)
    nonzeros = 0
    provenance_terms = 0
    max_row_support = 0
    max_provenance_support = 0
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"{label}[{index}]")
        coefficients = _sequence(
            row.get("coefficients"),
            f"{label}[{index}] coefficients",
        )
        provenance = _sequence(
            row.get("provenance"),
            f"{label}[{index}] provenance",
        )
        nonzeros += len(coefficients)
        provenance_terms += len(provenance)
        max_row_support = max(max_row_support, len(coefficients))
        max_provenance_support = max(
            max_provenance_support,
            len(provenance),
        )
    return (
        nonzeros,
        provenance_terms,
        max_row_support,
        max_provenance_support,
    )


def _polynomial_term_count(value: object, label: str) -> int:
    return len(_sequence(value, label))


def _artifact_resources(
    artifact: object,
    expected_generators: tuple[object, ...],
    expected_query: object | None,
    expected_allowed_values: tuple[int, ...] | None,
) -> tuple[str, Mapping[str, object], dict[str, int]]:
    envelope = _mapping(artifact, "quotient/outcome artifact")
    keys = set(envelope)
    if keys == {"certificate", "certificate_sha256"}:
        if expected_query is not None or expected_allowed_values is not None:
            _fail("a quotient-only artifact cannot bind a consequence query")
        try:
            verified = verify_quotient_artifact(artifact, expected_generators)
        except ArtifactVerificationError as error:
            raise ResourceReceiptError(
                f"independent quotient replay failed: {error}"
            ) from error
        artifact_kind = "quotient"
        query_terms = 0
        evidence_terms = 0
    elif keys == {"certificate", "outcome", "outcome_sha256"}:
        if expected_query is None or expected_allowed_values is None:
            _fail(
                "an outcome artifact requires external query and value-domain bindings"
            )
        try:
            outcome_verified = verify_outcome_artifact(
                artifact,
                expected_generators,
                expected_query,
                expected_allowed_values=expected_allowed_values,
            )
            quotient_verified = verify_quotient_artifact(
                {
                    "certificate": envelope["certificate"],
                    "certificate_sha256": outcome_verified.certificate_sha256,
                },
                expected_generators,
            )
        except ArtifactVerificationError as error:
            raise ResourceReceiptError(
                f"independent outcome replay failed: {error}"
            ) from error
        verified = quotient_verified
        artifact_kind = "outcome"
        outcome = _mapping(envelope["outcome"], "outcome")
        query_terms = _polynomial_term_count(outcome["query"], "outcome query")
        evidence_terms = 0
        consequence_value = outcome["consequence"]
        if consequence_value is not None:
            consequence = _mapping(consequence_value, "outcome consequence")
            evidence_value = consequence.get("evidence")
            if evidence_value is not None:
                evidence = _mapping(evidence_value, "outcome evidence")
                evidence_terms = len(
                    _sequence(evidence.get("terms"), "outcome evidence terms")
                )
    else:
        _fail(
            "artifact must be a plain quotient certificate or consequence outcome "
            "transport envelope"
        )

    certificate = _mapping(envelope["certificate"], "certificate")
    generators = _sequence(certificate["generators"], "certificate generators")
    generator_terms = sum(
        _polynomial_term_count(generator, f"generator {index}")
        for index, generator in enumerate(generators)
    )
    main_monomials = len(
        _sequence(certificate["admitted_monomials"], "admitted monomials")
    )
    prolongation_monomials = len(
        _sequence(
            certificate["prolongation_monomials"],
            "prolongation monomials",
        )
    )
    main = _row_resource_counts(certificate["rows"], "main rows")
    prolongation = _row_resource_counts(
        certificate["prolongation_rows"],
        "prolongation rows",
    )
    counts = {
        "evidence_term_count": evidence_terms,
        "generator_count": len(generators),
        "generator_term_count": generator_terms,
        "main_degree": _plain_int(certificate["degree_limit"], "main degree"),
        "main_monomial_count": main_monomials,
        "main_provenance_terms": main[1],
        "main_rank": verified.main_rank,
        "main_rref_nonzeros": main[0],
        "max_provenance_support": max(main[3], prolongation[3]),
        "max_row_support": max(main[2], prolongation[2]),
        "prolongation_degree": _plain_int(
            certificate["prolongation_degree"],
            "prolongation degree",
        ),
        "prolongation_monomial_count": prolongation_monomials,
        "prolongation_provenance_terms": prolongation[1],
        "prolongation_rank": verified.prolongation_rank,
        "prolongation_rref_nonzeros": prolongation[0],
        "query_term_count": query_terms,
        "quotient_dimension": verified.quotient_dimension,
        "total_polynomial_term_count": (
            generator_terms + query_terms + evidence_terms
        ),
        "variable_count": _plain_int(
            certificate["variable_count"],
            "variable count",
        ),
    }
    return artifact_kind, certificate, counts


def _enforce_resource_caps(
    vector: Mapping[str, object],
    declared_peak: Mapping[str, int],
    caps: Mapping[str, int],
) -> None:
    checks = {
        "source_independent_artifact_bytes": vector[
            "source_independent_artifact_bytes"
        ],
        "variable_count": vector["variable_count"],
        "generator_count": vector["generator_count"],
        "polynomial_terms": vector["total_polynomial_term_count"],
        "main_degree": vector["main_degree"],
        "prolongation_degree": vector["prolongation_degree"],
        "monomial_count": max(
            _plain_int(vector["main_monomial_count"], "main monomial count"),
            _plain_int(
                vector["prolongation_monomial_count"],
                "prolongation monomial count",
            ),
        ),
        "rank": max(
            _plain_int(vector["main_rank"], "main rank"),
            _plain_int(vector["prolongation_rank"], "prolongation rank"),
        ),
        "rref_nonzeros": (
            _plain_int(vector["main_rref_nonzeros"], "main RREF nonzeros")
            + _plain_int(
                vector["prolongation_rref_nonzeros"],
                "prolongation RREF nonzeros",
            )
        ),
        "provenance_terms": (
            _plain_int(
                vector["main_provenance_terms"],
                "main provenance terms",
            )
            + _plain_int(
                vector["prolongation_provenance_terms"],
                "prolongation provenance terms",
            )
        ),
        "row_support": vector["max_row_support"],
        "provenance_support": vector["max_provenance_support"],
        "quotient_dimension": vector["quotient_dimension"],
        "primitive_program_count": vector["primitive_program_count"],
        "primitive_instructions": sum(
            _plain_int(count, "primitive opcode count")
            for count in _mapping(
                vector["primitive_opcode_counts"],
                "primitive opcode counts",
            ).values()
        ),
        "primitive_cycles": vector["primitive_cycles"],
        "sequential_depth": vector["sequential_depth"],
    }
    for key, raw_actual in checks.items():
        actual = _plain_int(raw_actual, f"resource {key}")
        if actual < 0:
            _fail(f"resource {key} must be nonnegative")
        if actual > caps[key]:
            _fail(
                f"resource overflow: {key}={actual} exceeds cap {caps[key]}"
            )

    peak_to_cap = {
        "artifact_bytes": "source_independent_artifact_bytes",
        "monomial_slots": "monomial_count",
        "rref_nonzeros": "rref_nonzeros",
        "provenance_terms": "provenance_terms",
        "row_support": "row_support",
        "provenance_support": "provenance_support",
        "quotient_dimension": "quotient_dimension",
        "primitive_instruction_slots": "primitive_instructions",
    }
    for peak_key, cap_key in peak_to_cap.items():
        if declared_peak[peak_key] > caps[cap_key]:
            _fail(
                "declared workspace overflow: "
                f"{peak_key}={declared_peak[peak_key]} exceeds cap {caps[cap_key]}"
            )


def _derive_resources(
    artifact: object,
    expected_generators: Iterable[object],
    *,
    expected_query: object | None,
    expected_allowed_values: Iterable[int] | None,
    primitive_program_receipts: Iterable[object],
    declared_peak_workspace_bounds: Mapping[str, int],
    resource_caps: Mapping[str, int] | None,
) -> _DerivedResources:
    generators = tuple(expected_generators)
    allowed = (
        None
        if expected_allowed_values is None
        else tuple(
            _plain_int(value, "expected allowed value")
            for value in expected_allowed_values
        )
    )
    artifact_kind, _certificate, structural = _artifact_resources(
        artifact,
        generators,
        expected_query,
        allowed,
    )
    programs = tuple(primitive_program_receipts)
    aggregate_counts = {opcode: 0 for opcode in PRIMITIVE_OPCODES}
    program_hashes: list[str] = []
    total_cycles = 0
    total_depth = 0
    max_instruction_slots = 0
    for index, program in enumerate(programs):
        digest, counts, instructions, cycles, depth = _replay_primitive_receipt(
            program,
            index,
        )
        program_hashes.append(digest)
        for opcode in PRIMITIVE_OPCODES:
            aggregate_counts[opcode] += counts[opcode]
        total_cycles += cycles
        total_depth += depth
        max_instruction_slots = max(max_instruction_slots, instructions)

    artifact_bytes = len(_canonical_bytes(artifact)) + sum(
        len(_canonical_bytes(program)) for program in programs
    )
    vector: dict[str, object] = {
        "evidence_term_count": structural["evidence_term_count"],
        "generator_count": structural["generator_count"],
        "generator_term_count": structural["generator_term_count"],
        "main_degree": structural["main_degree"],
        "main_monomial_count": structural["main_monomial_count"],
        "main_provenance_terms": structural["main_provenance_terms"],
        "main_rank": structural["main_rank"],
        "main_rref_nonzeros": structural["main_rref_nonzeros"],
        "max_provenance_support": structural["max_provenance_support"],
        "max_row_support": structural["max_row_support"],
        "primitive_cycles": total_cycles,
        "primitive_opcode_counts": aggregate_counts,
        "primitive_program_count": len(programs),
        "prolongation_degree": structural["prolongation_degree"],
        "prolongation_monomial_count": structural[
            "prolongation_monomial_count"
        ],
        "prolongation_provenance_terms": structural[
            "prolongation_provenance_terms"
        ],
        "prolongation_rank": structural["prolongation_rank"],
        "prolongation_rref_nonzeros": structural[
            "prolongation_rref_nonzeros"
        ],
        "query_term_count": structural["query_term_count"],
        "quotient_dimension": structural["quotient_dimension"],
        "schema": RESOURCE_VECTOR_SCHEMA,
        "sequential_depth": total_depth,
        "source_independent_artifact_bytes": artifact_bytes,
        "total_polynomial_term_count": structural[
            "total_polynomial_term_count"
        ],
        "variable_count": structural["variable_count"],
    }
    actual_peak = {
        "artifact_bytes": artifact_bytes,
        "monomial_slots": max(
            structural["main_monomial_count"],
            structural["prolongation_monomial_count"],
        ),
        "primitive_instruction_slots": max_instruction_slots,
        "provenance_support": structural["max_provenance_support"],
        "provenance_terms": max(
            structural["main_provenance_terms"],
            structural["prolongation_provenance_terms"],
        ),
        "quotient_dimension": structural["quotient_dimension"],
        "row_support": structural["max_row_support"],
        "rref_nonzeros": max(
            structural["main_rref_nonzeros"],
            structural["prolongation_rref_nonzeros"],
        ),
    }
    declared_peak = _normalize_peak_workspace(
        declared_peak_workspace_bounds,
        "declared peak workspace bounds",
    )
    for key in PEAK_WORKSPACE_KEYS:
        if declared_peak[key] < actual_peak[key]:
            _fail(
                "declared peak workspace underflow: "
                f"{key}={declared_peak[key]} is below required {actual_peak[key]}"
            )
    caps = _normalize_caps(resource_caps)
    _enforce_resource_caps(vector, declared_peak, caps)
    return _DerivedResources(
        artifact_kind=artifact_kind,
        artifact_sha256=_sha256(artifact),
        primitive_program_sha256s=tuple(program_hashes),
        vector=vector,
        actual_peak_workspace=actual_peak,
        declared_peak_workspace=declared_peak,
    )


def build_ssqac_resource_receipt(
    artifact: object,
    expected_generators: Iterable[object],
    *,
    expected_query: object | None = None,
    expected_allowed_values: Iterable[int] | None = None,
    primitive_program_receipts: Iterable[object] = (),
    declared_peak_workspace_bounds: Mapping[str, int],
    resource_caps: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Build a hash-bound resource receipt after independent exact replay."""

    programs = tuple(primitive_program_receipts)
    derived = _derive_resources(
        artifact,
        expected_generators,
        expected_query=expected_query,
        expected_allowed_values=expected_allowed_values,
        primitive_program_receipts=programs,
        declared_peak_workspace_bounds=declared_peak_workspace_bounds,
        resource_caps=resource_caps,
    )
    payload: dict[str, object] = {
        "actual_peak_workspace": derived.actual_peak_workspace,
        "artifact_kind": derived.artifact_kind,
        "artifact_sha256": derived.artifact_sha256,
        "declared_peak_workspace_bounds": derived.declared_peak_workspace,
        "primitive_program_sha256s": list(derived.primitive_program_sha256s),
        "resource_vector": derived.vector,
        "resource_vector_sha256": _sha256(derived.vector),
        "runtime_observations": _runtime_observations(),
        "schema": RESOURCE_RECEIPT_SCHEMA,
    }
    return {
        "receipt": payload,
        "receipt_sha256": _sha256(payload),
    }


def verify_ssqac_resource_receipt(
    receipt_artifact: object,
    artifact: object,
    expected_generators: Iterable[object],
    *,
    expected_query: object | None = None,
    expected_allowed_values: Iterable[int] | None = None,
    primitive_program_receipts: Iterable[object] = (),
    expected_declared_peak_workspace_bounds: Mapping[str, int],
    resource_caps: Mapping[str, int] | None = None,
) -> ResourceReceiptVerification:
    """Recompute a claimed resource vector and reject every mismatch."""

    envelope = _mapping(receipt_artifact, "resource receipt artifact")
    _exact_keys(
        envelope,
        {"receipt", "receipt_sha256"},
        "resource receipt artifact",
    )
    payload = _mapping(envelope["receipt"], "resource receipt")
    _exact_keys(
        payload,
        {
            "actual_peak_workspace",
            "artifact_kind",
            "artifact_sha256",
            "declared_peak_workspace_bounds",
            "primitive_program_sha256s",
            "resource_vector",
            "resource_vector_sha256",
            "runtime_observations",
            "schema",
        },
        "resource receipt",
    )
    if payload["schema"] != RESOURCE_RECEIPT_SCHEMA:
        _fail("unexpected resource receipt schema")
    claimed_receipt_sha = _digest(
        envelope["receipt_sha256"],
        "resource receipt digest",
    )
    actual_receipt_sha = _sha256(payload)
    if claimed_receipt_sha != actual_receipt_sha:
        _fail("resource receipt digest differs from canonical payload")
    declared = _normalize_peak_workspace(
        payload["declared_peak_workspace_bounds"],
        "receipt declared peak workspace bounds",
    )
    expected_declared = _normalize_peak_workspace(
        expected_declared_peak_workspace_bounds,
        "externally expected peak workspace bounds",
    )
    if declared != expected_declared:
        _fail("declared peak workspace differs from external declaration")

    programs = tuple(primitive_program_receipts)
    derived = _derive_resources(
        artifact,
        expected_generators,
        expected_query=expected_query,
        expected_allowed_values=expected_allowed_values,
        primitive_program_receipts=programs,
        declared_peak_workspace_bounds=expected_declared,
        resource_caps=resource_caps,
    )
    claimed_vector = _mapping(payload["resource_vector"], "resource vector")
    claimed_vector_sha = _digest(
        payload["resource_vector_sha256"],
        "resource vector digest",
    )
    if claimed_vector_sha != _sha256(claimed_vector):
        _fail("resource vector digest differs from claimed vector")
    if dict(claimed_vector) != derived.vector:
        _fail("claimed resource vector differs from independent replay")
    if payload["artifact_kind"] != derived.artifact_kind:
        _fail("claimed artifact kind differs from independent replay")
    if _digest(payload["artifact_sha256"], "bound artifact digest") != (
        derived.artifact_sha256
    ):
        _fail("resource receipt does not bind the supplied artifact")
    claimed_program_hashes = tuple(
        _digest(value, "bound primitive program digest")
        for value in _sequence(
            payload["primitive_program_sha256s"],
            "primitive program digests",
        )
    )
    if claimed_program_hashes != derived.primitive_program_sha256s:
        _fail("resource receipt does not bind the primitive program receipts")
    claimed_actual_peak = _normalize_peak_workspace(
        payload["actual_peak_workspace"],
        "claimed actual peak workspace",
    )
    if claimed_actual_peak != derived.actual_peak_workspace:
        _fail("claimed actual peak workspace differs from independent replay")
    if payload["runtime_observations"] != _runtime_observations():
        _fail(
            "structural receipts cannot fabricate wall time or device-memory "
            "observations"
        )
    return ResourceReceiptVerification(
        schema=RESOURCE_VERIFICATION_SCHEMA,
        receipt_sha256=actual_receipt_sha,
        artifact_sha256=derived.artifact_sha256,
        resource_vector_sha256=_sha256(derived.vector),
        artifact_kind=derived.artifact_kind,
        gates=(
            "receipt_digest",
            "external_problem_binding",
            "independent_quotient_replay",
            "artifact_digest_binding",
            "primitive_instruction_replay",
            "resource_vector_replay",
            "workspace_declaration_binding",
            "resource_caps",
            "runtime_observation_nonfabrication",
        ),
    )
