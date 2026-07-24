"""Standalone verifier for portable SSQAC quotient and outcome artifacts.

This module intentionally has no dependency on the quotient compiler.  It
accepts plain JSON-compatible mappings and independently checks the algebra
using exact arithmetic in F_257.

Transport envelopes have these shapes::

    {
        "certificate": { ... ssqac_quotient_algebra_certificate_v1 ... },
        "certificate_sha256": "<canonical payload digest>"
    }

    {
        "certificate": { ... certificate payload ... },
        "outcome": { ... ssqac_quotient_consequence_outcome_v1 ... },
        "outcome_sha256": "<canonical outcome payload digest>"
    }

The expected generators, query, and allowed values are supplied separately.
That external binding prevents a self-consistent artifact from changing the
problem it claims to certify.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Sequence


FIELD_MODULUS = 257
MAX_VARIABLES = 128
MAX_DEGREE = 8
MAX_MONOMIALS = 4096
MAX_QUOTIENT_DIMENSION = 256
CERTIFICATE_SCHEMA = "ssqac_quotient_algebra_certificate_v1"
OUTCOME_SCHEMA = "ssqac_quotient_consequence_outcome_v1"
VERIFICATION_SCHEMA = "ssqac_independent_quotient_verification_v1"
OUTCOME_VERIFICATION_SCHEMA = "ssqac_independent_outcome_verification_v1"

Monomial = tuple[int, ...]
Polynomial = dict[Monomial, int]
SparseVector = dict[int, int]


class ArtifactVerificationError(ValueError):
    """A portable artifact failed a mandatory verification gate."""


@dataclass(frozen=True, slots=True)
class QuotientVerification:
    schema: str
    certificate_sha256: str
    quotient_digest: str
    main_rank: int
    prolongation_rank: int
    quotient_dimension: int
    inconsistent: bool
    gates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutcomeVerification:
    schema: str
    outcome_sha256: str
    certificate_sha256: str
    quotient_digest: str
    status: str
    value: int | None
    gates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Row:
    pivot: int
    coefficients: SparseVector


@dataclass(frozen=True, slots=True)
class _VerifiedQuotient:
    public: QuotientVerification
    certificate: Mapping[str, object]
    generators: tuple[Polynomial, ...]
    monomials: tuple[Monomial, ...]
    rows: tuple[_Row, ...]
    basis_columns: tuple[int, ...]
    operators: tuple[tuple[SparseVector, ...], ...]


def _fail(message: str) -> None:
    raise ArtifactVerificationError(message)


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


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ArtifactVerificationError(
            "artifact is not canonical ASCII JSON"
        ) from error


def _sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _monomial(
    value: object,
    width: int,
    degree_limit: int,
    label: str,
) -> Monomial:
    raw = _sequence(value, label)
    if len(raw) != width:
        _fail(f"{label} has width {len(raw)}, expected {width}")
    result = tuple(_plain_int(exponent, f"{label} exponent") for exponent in raw)
    if any(exponent < 0 for exponent in result):
        _fail(f"{label} has a negative exponent")
    if sum(result) > degree_limit:
        _fail(f"{label} exceeds degree {degree_limit}")
    return result


def _monomial_key(value: Monomial) -> tuple[int, Monomial]:
    return sum(value), value


def _all_monomials(width: int, degree_limit: int) -> tuple[Monomial, ...]:
    values: list[Monomial] = []

    def append_compositions(
        variable: int,
        remaining: int,
        prefix: list[int],
    ) -> None:
        if variable == width - 1:
            values.append(tuple((*prefix, remaining)))
            return
        for exponent in range(remaining, -1, -1):
            append_compositions(
                variable + 1,
                remaining - exponent,
                [*prefix, exponent],
            )

    for degree in range(degree_limit + 1):
        append_compositions(0, degree, [])
        if len(values) > MAX_MONOMIALS:
            _fail(f"complete monomial workspace exceeds {MAX_MONOMIALS}")
    return tuple(sorted(values, key=_monomial_key, reverse=True))


def _parse_polynomial(
    value: object,
    width: int,
    degree_limit: int,
    label: str,
) -> Polynomial:
    raw_terms = _sequence(value, label)
    result: Polynomial = {}
    order: list[Monomial] = []
    for term_index, raw_term in enumerate(raw_terms):
        term = _sequence(raw_term, f"{label}[{term_index}]")
        if len(term) != 2:
            _fail(f"{label}[{term_index}] must contain monomial and coefficient")
        monomial = _monomial(
            term[0],
            width,
            degree_limit,
            f"{label}[{term_index}] monomial",
        )
        coefficient = _plain_int(
            term[1], f"{label}[{term_index}] coefficient"
        )
        if not 1 <= coefficient < FIELD_MODULUS:
            _fail(f"{label} coefficients must be canonical nonzero residues")
        if monomial in result:
            _fail(f"{label} repeats a monomial")
        result[monomial] = coefficient
        order.append(monomial)
    if order != sorted(order, key=_monomial_key, reverse=True):
        _fail(f"{label} terms are not in canonical graded-lex order")
    return result


def _polynomial_data(polynomial: Polynomial) -> list[list[object]]:
    return [
        [list(monomial), coefficient]
        for monomial, coefficient in sorted(
            polynomial.items(), key=lambda item: _monomial_key(item[0]), reverse=True
        )
        if coefficient
    ]


def _canonical_generators(
    generators: Iterable[object],
    width: int,
    degree_limit: int,
    label: str,
    *,
    require_canonical: bool,
) -> tuple[Polynomial, ...]:
    parsed = tuple(
        _parse_polynomial(generator, width, degree_limit, f"{label}[{index}]")
        for index, generator in enumerate(generators)
    )
    if any(not generator for generator in parsed) and require_canonical:
        _fail(f"{label} contain a zero polynomial")

    def generator_key(
        generator: Polynomial,
    ) -> tuple[tuple[tuple[int, Monomial], int], ...]:
        return tuple(
            (_monomial_key(monomial), coefficient)
            for monomial, coefficient in sorted(
                generator.items(),
                key=lambda item: _monomial_key(item[0]),
                reverse=True,
            )
        )

    unique = {generator_key(generator): generator for generator in parsed if generator}
    canonical = tuple(unique[key] for key in sorted(unique, reverse=True))
    if require_canonical and parsed != canonical:
        _fail(f"{label} are not in canonical order")
    return canonical


def _add_scaled(target: Polynomial | SparseVector, source: Mapping, scale: int) -> None:
    factor = scale % FIELD_MODULUS
    for key, coefficient in source.items():
        value = (target.get(key, 0) + factor * coefficient) % FIELD_MODULUS
        if value:
            target[key] = value
        else:
            target.pop(key, None)


def _subtract_scaled(
    target: Polynomial | SparseVector,
    source: Mapping,
    scale: int,
) -> None:
    _add_scaled(target, source, -scale)


def _multiply_monomials(left: Monomial, right: Monomial) -> Monomial:
    return tuple(a + b for a, b in zip(left, right))


def _multiply_generator(
    generator: Polynomial,
    multiplier: Monomial,
    degree_limit: int,
) -> Polynomial:
    result: Polynomial = {}
    for monomial, coefficient in generator.items():
        output = _multiply_monomials(monomial, multiplier)
        if sum(output) > degree_limit:
            _fail("generator multiple leaves the admitted degree")
        result[output] = coefficient
    return result


def _parse_monomial_workspace(
    value: object,
    width: int,
    degree_limit: int,
    label: str,
) -> tuple[Monomial, ...]:
    raw = _sequence(value, label)
    result = tuple(
        _monomial(item, width, degree_limit, f"{label}[{index}]")
        for index, item in enumerate(raw)
    )
    expected = _all_monomials(width, degree_limit)
    if result != expected:
        _fail(f"{label} is not the complete canonical degree workspace")
    return result


def _parse_provenance(
    value: object,
    *,
    width: int,
    degree_limit: int,
    generators: tuple[Polynomial, ...],
    label: str,
) -> Polynomial:
    raw = _sequence(value, label)
    seen: set[tuple[int, Monomial]] = set()
    reconstructed: Polynomial = {}
    canonical_keys: list[tuple[int, tuple[int, Monomial]]] = []
    for index, raw_term in enumerate(raw):
        term = _mapping(raw_term, f"{label}[{index}]")
        _exact_keys(
            term,
            {"coefficient", "generator_index", "multiplier"},
            f"{label}[{index}]",
        )
        generator_index = _plain_int(
            term["generator_index"], f"{label}[{index}] generator index"
        )
        if not 0 <= generator_index < len(generators):
            _fail(f"{label}[{index}] generator index is out of range")
        multiplier = _monomial(
            term["multiplier"],
            width,
            degree_limit,
            f"{label}[{index}] multiplier",
        )
        coefficient = _plain_int(
            term["coefficient"], f"{label}[{index}] coefficient"
        )
        if not 1 <= coefficient < FIELD_MODULUS:
            _fail(f"{label} coefficients must be canonical nonzero residues")
        source = (generator_index, multiplier)
        if source in seen:
            _fail(f"{label} repeats a generator multiple")
        seen.add(source)
        canonical_keys.append((generator_index, _monomial_key(multiplier)))
        multiple = _multiply_generator(
            generators[generator_index], multiplier, degree_limit
        )
        _add_scaled(reconstructed, multiple, coefficient)
    if canonical_keys != sorted(canonical_keys):
        _fail(f"{label} is not in canonical source order")
    return reconstructed


def _parse_rows(
    value: object,
    *,
    width: int,
    degree_limit: int,
    monomials: tuple[Monomial, ...],
    generators: tuple[Polynomial, ...],
    label: str,
) -> tuple[_Row, ...]:
    raw_rows = _sequence(value, label)
    column = {monomial: index for index, monomial in enumerate(monomials)}
    rows: list[_Row] = []
    for row_index, raw_row in enumerate(raw_rows):
        row = _mapping(raw_row, f"{label}[{row_index}]")
        _exact_keys(
            row,
            {"coefficients", "pivot", "provenance"},
            f"{label}[{row_index}]",
        )
        pivot_monomial = _monomial(
            row["pivot"], width, degree_limit, f"{label}[{row_index}] pivot"
        )
        if pivot_monomial not in column:
            _fail(f"{label}[{row_index}] pivot leaves the workspace")
        raw_coefficients = _sequence(
            row["coefficients"], f"{label}[{row_index}] coefficients"
        )
        coefficients: SparseVector = {}
        coefficient_columns: list[int] = []
        for term_index, raw_term in enumerate(raw_coefficients):
            term = _sequence(
                raw_term,
                f"{label}[{row_index}] coefficients[{term_index}]",
            )
            if len(term) != 2:
                _fail("row coefficient terms must have monomial and coefficient")
            monomial = _monomial(
                term[0],
                width,
                degree_limit,
                f"{label}[{row_index}] coefficient monomial",
            )
            if monomial not in column:
                _fail(f"{label}[{row_index}] coefficient leaves the workspace")
            coefficient = _plain_int(
                term[1], f"{label}[{row_index}] coefficient"
            )
            if not 1 <= coefficient < FIELD_MODULUS:
                _fail("row coefficients must be canonical nonzero residues")
            target_column = column[monomial]
            if target_column in coefficients:
                _fail(f"{label}[{row_index}] repeats a coefficient column")
            coefficients[target_column] = coefficient
            coefficient_columns.append(target_column)
        pivot = column[pivot_monomial]
        if coefficient_columns != sorted(coefficient_columns):
            _fail(f"{label}[{row_index}] coefficients are not canonical")
        if not coefficients or min(coefficients) != pivot:
            _fail(f"{label}[{row_index}] pivot is not the leading column")
        if coefficients[pivot] != 1:
            _fail(f"{label}[{row_index}] pivot is not monic")
        reconstructed = _parse_provenance(
            row["provenance"],
            width=width,
            degree_limit=degree_limit,
            generators=generators,
            label=f"{label}[{row_index}] provenance",
        )
        certified = {
            monomials[target_column]: coefficient
            for target_column, coefficient in coefficients.items()
        }
        if reconstructed != certified:
            _fail(f"{label}[{row_index}] provenance does not reconstruct the row")
        rows.append(_Row(pivot=pivot, coefficients=coefficients))
    if [row.pivot for row in rows] != sorted(row.pivot for row in rows):
        _fail(f"{label} are not in canonical pivot order")
    if len({row.pivot for row in rows}) != len(rows):
        _fail(f"{label} repeat a pivot")
    for row in rows:
        for other in rows:
            expected = 1 if row.pivot == other.pivot else 0
            if row.coefficients.get(other.pivot, 0) != expected:
                _fail(f"{label} are not reduced row echelon form")
    return tuple(rows)


def _reduce(row: SparseVector, rows: Sequence[_Row]) -> SparseVector:
    result = dict(row)
    for pivot_row in rows:
        factor = result.get(pivot_row.pivot, 0)
        if factor:
            _subtract_scaled(result, pivot_row.coefficients, factor)
    return result


def _verify_complete_span(
    generators: tuple[Polynomial, ...],
    monomials: tuple[Monomial, ...],
    rows: tuple[_Row, ...],
    degree_limit: int,
) -> None:
    column = {monomial: index for index, monomial in enumerate(monomials)}
    for generator in generators:
        for multiplier in reversed(monomials):
            products = {
                _multiply_monomials(monomial, multiplier): coefficient
                for monomial, coefficient in generator.items()
            }
            if not products or any(
                sum(monomial) > degree_limit or monomial not in column
                for monomial in products
            ):
                continue
            vector = {
                column[monomial]: coefficient
                for monomial, coefficient in products.items()
            }
            if _reduce(vector, rows):
                _fail("certified RREF does not span every Macaulay generator multiple")


def _apply_operator(
    operator: tuple[SparseVector, ...],
    vector: Mapping[int, int],
) -> SparseVector:
    result: SparseVector = {}
    for input_index, input_coefficient in vector.items():
        if not 0 <= input_index < len(operator):
            _fail("operator input coordinate is out of range")
        _add_scaled(result, operator[input_index], input_coefficient)
    return result


def _build_operators(
    width: int,
    degree_limit: int,
    monomials: tuple[Monomial, ...],
    rows: tuple[_Row, ...],
    basis_columns: tuple[int, ...],
) -> tuple[tuple[SparseVector, ...], ...]:
    column = {monomial: index for index, monomial in enumerate(monomials)}
    basis_index = {
        basis_column: index for index, basis_column in enumerate(basis_columns)
    }
    operators: list[tuple[SparseVector, ...]] = []
    for variable in range(width):
        coordinate = tuple(1 if index == variable else 0 for index in range(width))
        operator_columns: list[SparseVector] = []
        for basis_column in basis_columns:
            product_monomial = _multiply_monomials(
                monomials[basis_column], coordinate
            )
            if sum(product_monomial) > degree_limit:
                _fail("quotient border exceeds the admitted degree")
            product_column = column.get(product_monomial)
            if product_column is None:
                _fail("quotient border leaves the admitted workspace")
            remainder = _reduce({product_column: 1}, rows)
            if any(output not in basis_index for output in remainder):
                _fail("normal form retains a pivot column")
            operator_columns.append(
                {
                    basis_index[output]: coefficient
                    for output, coefficient in remainder.items()
                }
            )
        operators.append(tuple(operator_columns))
    frozen = tuple(operators)
    for left in range(width):
        for right in range(left, width):
            for coordinate in range(len(basis_columns)):
                unit = {coordinate: 1}
                left_right = _apply_operator(
                    frozen[right], _apply_operator(frozen[left], unit)
                )
                right_left = _apply_operator(
                    frozen[left], _apply_operator(frozen[right], unit)
                )
                if left_right != right_left:
                    _fail("quotient multiplication operators do not commute")
    return frozen


def _verify_order_ideal(
    width: int,
    monomials: tuple[Monomial, ...],
    basis_columns: tuple[int, ...],
) -> None:
    basis = {monomials[column] for column in basis_columns}
    zero = (0,) * width
    if zero not in basis:
        _fail("quotient basis is not connected to one")
    for monomial in basis:
        for variable, exponent in enumerate(monomial):
            if exponent:
                divisor = list(monomial)
                divisor[variable] -= 1
                if tuple(divisor) not in basis:
                    _fail("quotient basis is not divisor-closed")


def _boolean_idempotent(
    operators: tuple[tuple[SparseVector, ...], ...],
) -> bool:
    for operator in operators:
        for coordinate in range(len(operator)):
            unit = {coordinate: 1}
            once = _apply_operator(operator, unit)
            twice = _apply_operator(operator, once)
            if twice != once:
                return False
    return True


def _semantic_digest(
    *,
    width: int,
    degree_limit: int,
    monomials: tuple[Monomial, ...],
    rows: tuple[_Row, ...],
    basis_columns: tuple[int, ...],
    prolongation_degree: int,
    prolongation_monomial_count: int,
    prolongation_rank: int,
    boolean_verified: bool,
    inconsistent: bool,
) -> str:
    payload = {
        "basis": [list(monomials[column]) for column in basis_columns],
        "boolean_schema_verified": boolean_verified,
        "degree_limit": degree_limit,
        "field_modulus": FIELD_MODULUS,
        "monomials": [list(monomial) for monomial in monomials],
        "prolongation_degree": prolongation_degree,
        "prolongation_monomial_count": prolongation_monomial_count,
        "prolongation_rank": prolongation_rank,
        "rref": [
            [
                row.pivot,
                [
                    [column, coefficient]
                    for column, coefficient in sorted(row.coefficients.items())
                ],
            ]
            for row in rows
        ],
        "status": "inconsistent" if inconsistent else "ok",
        "variable_count": width,
    }
    return _sha256(payload)


def _verify_quotient(
    artifact: object,
    expected_generators: Iterable[object],
) -> _VerifiedQuotient:
    envelope = _mapping(artifact, "quotient artifact")
    _exact_keys(
        envelope,
        {"certificate", "certificate_sha256"},
        "quotient artifact",
    )
    certificate = _mapping(envelope["certificate"], "certificate")
    _exact_keys(
        certificate,
        {
            "admitted_monomials",
            "degree_limit",
            "field_modulus",
            "generators",
            "prolongation_degree",
            "prolongation_monomials",
            "prolongation_rows",
            "quotient_digest",
            "require_boolean_schema",
            "rows",
            "schema",
            "variable_count",
        },
        "certificate",
    )
    if certificate["schema"] != CERTIFICATE_SCHEMA:
        _fail("unexpected certificate schema")
    if _plain_int(certificate["field_modulus"], "field modulus") != FIELD_MODULUS:
        _fail("SSQAC certificate field must be F_257")
    width = _plain_int(certificate["variable_count"], "variable count")
    degree_limit = _plain_int(certificate["degree_limit"], "degree limit")
    if not 1 <= width <= MAX_VARIABLES:
        _fail(f"variable count must be in [1, {MAX_VARIABLES}]")
    if not 1 <= degree_limit <= MAX_DEGREE:
        _fail(f"degree limit must be in [1, {MAX_DEGREE}]")
    require_boolean = certificate["require_boolean_schema"]
    if not isinstance(require_boolean, bool):
        _fail("require_boolean_schema must be Boolean")
    claimed_certificate_digest = _digest(
        envelope["certificate_sha256"], "certificate digest"
    )
    actual_certificate_digest = _sha256(certificate)
    if claimed_certificate_digest != actual_certificate_digest:
        _fail("certificate artifact digest does not match canonical payload")
    claimed_quotient_digest = _digest(
        certificate["quotient_digest"], "quotient digest"
    )

    raw_generators = _sequence(certificate["generators"], "certificate generators")
    generators = _canonical_generators(
        raw_generators,
        width,
        degree_limit,
        "certificate generators",
        require_canonical=True,
    )
    expected = _canonical_generators(
        tuple(expected_generators),
        width,
        degree_limit,
        "expected generators",
        require_canonical=False,
    )
    if tuple(_polynomial_data(item) for item in generators) != tuple(
        _polynomial_data(item) for item in expected
    ):
        _fail("certificate generators differ from the external source")

    monomials = _parse_monomial_workspace(
        certificate["admitted_monomials"],
        width,
        degree_limit,
        "admitted monomials",
    )
    rows = _parse_rows(
        certificate["rows"],
        width=width,
        degree_limit=degree_limit,
        monomials=monomials,
        generators=generators,
        label="main rows",
    )
    _verify_complete_span(generators, monomials, rows, degree_limit)

    prolongation_degree = _plain_int(
        certificate["prolongation_degree"], "prolongation degree"
    )
    if prolongation_degree != degree_limit + 1:
        _fail("prolongation must be the complete next degree")
    prolongation_monomials = _parse_monomial_workspace(
        certificate["prolongation_monomials"],
        width,
        prolongation_degree,
        "prolongation monomials",
    )
    prolongation_rows = _parse_rows(
        certificate["prolongation_rows"],
        width=width,
        degree_limit=prolongation_degree,
        monomials=prolongation_monomials,
        generators=generators,
        label="prolongation rows",
    )
    _verify_complete_span(
        generators,
        prolongation_monomials,
        prolongation_rows,
        prolongation_degree,
    )
    quotient_dimension = len(monomials) - len(rows)
    prolonged_dimension = len(prolongation_monomials) - len(prolongation_rows)
    if quotient_dimension != prolonged_dimension:
        _fail("quotient dimension is not stable under complete prolongation")
    if quotient_dimension > MAX_QUOTIENT_DIMENSION:
        _fail(f"quotient dimension exceeds {MAX_QUOTIENT_DIMENSION}")

    pivot_columns = {row.pivot for row in rows}
    basis_columns = tuple(
        column for column in range(len(monomials)) if column not in pivot_columns
    )
    constant_column = monomials.index((0,) * width)
    inconsistent = constant_column in pivot_columns
    operators: tuple[tuple[SparseVector, ...], ...] = ()
    boolean_verified = False
    if not inconsistent:
        _verify_order_ideal(width, monomials, basis_columns)
        operators = _build_operators(
            width, degree_limit, monomials, rows, basis_columns
        )
        boolean_verified = _boolean_idempotent(operators)
        if require_boolean and not boolean_verified:
            _fail("certificate does not prove Boolean operator idempotence")
    elif require_boolean:
        boolean_verified = False

    actual_quotient_digest = _semantic_digest(
        width=width,
        degree_limit=degree_limit,
        monomials=monomials,
        rows=rows,
        basis_columns=basis_columns,
        prolongation_degree=prolongation_degree,
        prolongation_monomial_count=len(prolongation_monomials),
        prolongation_rank=len(prolongation_rows),
        boolean_verified=boolean_verified,
        inconsistent=inconsistent,
    )
    if actual_quotient_digest != claimed_quotient_digest:
        _fail("quotient semantic digest does not match independent replay")
    public = QuotientVerification(
        schema=VERIFICATION_SCHEMA,
        certificate_sha256=actual_certificate_digest,
        quotient_digest=actual_quotient_digest,
        main_rank=len(rows),
        prolongation_rank=len(prolongation_rows),
        quotient_dimension=quotient_dimension,
        inconsistent=inconsistent,
        gates=(
            "artifact_digest",
            "external_generator_binding",
            "rref",
            "provenance",
            "main_span",
            "complete_prolongation",
            "stable_dimension",
            "order_ideal",
            "multiplication_commutation",
            "boolean_idempotence",
            "quotient_digest",
        ),
    )
    return _VerifiedQuotient(
        public=public,
        certificate=certificate,
        generators=generators,
        monomials=monomials,
        rows=rows,
        basis_columns=basis_columns,
        operators=operators,
    )


def verify_quotient_artifact(
    artifact: object,
    expected_generators: Iterable[object],
) -> QuotientVerification:
    """Independently verify a quotient certificate transport envelope."""

    return _verify_quotient(artifact, expected_generators).public


def _polynomial_to_vector(
    polynomial: Polynomial,
    monomials: tuple[Monomial, ...],
) -> SparseVector:
    column = {monomial: index for index, monomial in enumerate(monomials)}
    try:
        return {
            column[monomial]: coefficient
            for monomial, coefficient in polynomial.items()
        }
    except KeyError as error:
        raise ArtifactVerificationError(
            "polynomial leaves the admitted workspace"
        ) from error


def _normal_form(
    polynomial: Polynomial,
    quotient: _VerifiedQuotient,
) -> Polynomial:
    reduced = _reduce(
        _polynomial_to_vector(polynomial, quotient.monomials),
        quotient.rows,
    )
    return {
        quotient.monomials[column]: coefficient
        for column, coefficient in reduced.items()
    }


def _apply_polynomial(
    polynomial: Polynomial,
    vector: Mapping[int, int],
    operators: tuple[tuple[SparseVector, ...], ...],
) -> SparseVector:
    result: SparseVector = {}
    for monomial, coefficient in polynomial.items():
        term = dict(vector)
        for variable, exponent in enumerate(monomial):
            for _ in range(exponent):
                term = _apply_operator(operators[variable], term)
        _add_scaled(result, term, coefficient)
    return result


def _domain_verified(
    normal_form: Polynomial,
    allowed_values: tuple[int, ...],
    quotient: _VerifiedQuotient,
) -> bool:
    for coordinate in range(len(quotient.basis_columns)):
        vector: SparseVector = {coordinate: 1}
        for value in allowed_values:
            transformed = _apply_polynomial(
                normal_form, vector, quotient.operators
            )
            _subtract_scaled(transformed, vector, value)
            vector = transformed
        if vector:
            return False
    return True


def _difference_from_value(polynomial: Polynomial, value: int, width: int) -> Polynomial:
    result = dict(polynomial)
    zero = (0,) * width
    updated = (result.get(zero, 0) - value) % FIELD_MODULUS
    if updated:
        result[zero] = updated
    else:
        result.pop(zero, None)
    return result


def _parse_membership_evidence(
    value: object,
    *,
    quotient: _VerifiedQuotient,
    expected_target: Polynomial,
) -> None:
    evidence = _mapping(value, "membership evidence")
    _exact_keys(evidence, {"target", "terms"}, "membership evidence")
    width = _plain_int(
        quotient.certificate["variable_count"], "certificate variable count"
    )
    degree_limit = _plain_int(
        quotient.certificate["degree_limit"], "certificate degree limit"
    )
    target = _parse_polynomial(
        evidence["target"], width, degree_limit, "membership target"
    )
    if target != expected_target:
        _fail("membership evidence target is not query minus claimed value")
    reconstructed = _parse_provenance(
        evidence["terms"],
        width=width,
        degree_limit=degree_limit,
        generators=quotient.generators,
        label="membership terms",
    )
    if reconstructed != target:
        _fail("membership evidence does not reconstruct its bound target")


def verify_outcome_artifact(
    artifact: object,
    expected_generators: Iterable[object],
    expected_query: object,
    *,
    expected_allowed_values: Iterable[int],
) -> OutcomeVerification:
    """Verify a consequence outcome, including its embedded quotient proof."""

    envelope = _mapping(artifact, "outcome artifact")
    _exact_keys(
        envelope,
        {"certificate", "outcome", "outcome_sha256"},
        "outcome artifact",
    )
    outcome = _mapping(envelope["outcome"], "outcome")
    _exact_keys(
        outcome,
        {
            "allowed_values",
            "certificate_sha256",
            "consequence",
            "diagnostic",
            "query",
            "quotient_digest",
            "schema",
            "status",
            "value",
        },
        "outcome",
    )
    if outcome["schema"] != OUTCOME_SCHEMA:
        _fail("unexpected outcome schema")
    claimed_outcome_digest = _digest(
        envelope["outcome_sha256"], "outcome digest"
    )
    actual_outcome_digest = _sha256(outcome)
    if claimed_outcome_digest != actual_outcome_digest:
        _fail("outcome artifact digest does not match canonical payload")

    certificate = _mapping(envelope["certificate"], "embedded certificate")
    certificate_digest = _sha256(certificate)
    if _digest(outcome["certificate_sha256"], "bound certificate digest") != (
        certificate_digest
    ):
        _fail("outcome does not bind the embedded certificate")
    quotient = _verify_quotient(
        {
            "certificate": certificate,
            "certificate_sha256": certificate_digest,
        },
        expected_generators,
    )
    if _digest(outcome["quotient_digest"], "bound quotient digest") != (
        quotient.public.quotient_digest
    ):
        _fail("outcome does not bind the verified quotient digest")

    width = _plain_int(certificate["variable_count"], "variable count")
    degree_limit = _plain_int(certificate["degree_limit"], "degree limit")
    query = _parse_polynomial(outcome["query"], width, degree_limit, "outcome query")
    expected = _parse_polynomial(
        expected_query, width, degree_limit, "expected query"
    )
    if query != expected:
        _fail("outcome query differs from the externally supplied query")

    raw_allowed = _sequence(outcome["allowed_values"], "allowed values")
    allowed = tuple(
        _plain_int(value, "allowed value") % FIELD_MODULUS
        for value in raw_allowed
    )
    expected_allowed = tuple(
        _plain_int(value, "expected allowed value") % FIELD_MODULUS
        for value in expected_allowed_values
    )
    if allowed != expected_allowed:
        _fail("outcome value domain differs from the external value domain")
    if not allowed or len(set(allowed)) != len(allowed):
        _fail("allowed values must be nonempty and unique modulo 257")

    status = outcome["status"]
    if status not in {"CERTIFIED", "AMBIGUOUS", "UNSAT", "INCOMPLETE"}:
        _fail("outcome status is not independently certifiable")
    claimed_value = outcome["value"]
    value = None if claimed_value is None else _plain_int(claimed_value, "value")
    consequence_value = outcome["consequence"]
    diagnostic = outcome["diagnostic"]

    if quotient.public.inconsistent:
        if (
            status != "UNSAT"
            or value is not None
            or consequence_value is not None
            or diagnostic is not None
        ):
            _fail("inconsistent quotient is not represented as a clean UNSAT outcome")
    else:
        consequence = _mapping(consequence_value, "consequence")
        _exact_keys(
            consequence,
            {
                "domain_verified",
                "evidence",
                "normal_form",
                "quotient_digest",
                "status",
                "value",
            },
            "consequence",
        )
        if _digest(
            consequence["quotient_digest"], "consequence quotient digest"
        ) != quotient.public.quotient_digest:
            _fail("consequence does not bind the verified quotient")
        normal_form = _normal_form(query, quotient)
        claimed_normal_form = _parse_polynomial(
            consequence["normal_form"],
            width,
            degree_limit,
            "claimed normal form",
        )
        if claimed_normal_form != normal_form:
            _fail("claimed normal form differs from independent reduction")
        domain_verified = _domain_verified(normal_form, allowed, quotient)
        if consequence["domain_verified"] is not domain_verified:
            _fail("claimed value-domain gate differs from independent replay")
        forced_values = tuple(
            candidate
            for candidate in allowed
            if not _normal_form(
                _difference_from_value(query, candidate, width), quotient
            )
        )
        consequence_status = consequence["status"]
        consequence_claim = consequence["value"]
        evidence = consequence["evidence"]

        if not domain_verified:
            if (
                status != "INCOMPLETE"
                or value is not None
                or consequence_status != "out_of_domain"
                or consequence_claim is not None
                or evidence is not None
                or diagnostic
                != "query is not confined to the declared value domain"
            ):
                _fail("out-of-domain query has an inconsistent outcome claim")
        elif len(forced_values) == 1:
            forced = forced_values[0]
            if (
                status != "CERTIFIED"
                or value != forced
                or consequence_status != "forced"
                or consequence_claim != forced
                or diagnostic is not None
                or evidence is None
            ):
                _fail("forced consequence fields do not match independent replay")
            _parse_membership_evidence(
                evidence,
                quotient=quotient,
                expected_target=_difference_from_value(query, forced, width),
            )
        elif not forced_values:
            if (
                status != "AMBIGUOUS"
                or value is not None
                or consequence_status != "ambiguous"
                or consequence_claim is not None
                or evidence is not None
                or diagnostic is not None
            ):
                _fail("ambiguous consequence fields do not match independent replay")
        else:
            _fail("consistent quotient forces multiple declared values")

    return OutcomeVerification(
        schema=OUTCOME_VERIFICATION_SCHEMA,
        outcome_sha256=actual_outcome_digest,
        certificate_sha256=certificate_digest,
        quotient_digest=quotient.public.quotient_digest,
        status=status,
        value=value,
        gates=(
            "outcome_digest",
            "certificate_binding",
            "quotient_verification",
            "query_binding",
            "allowed_value_binding",
            "normal_form",
            "value_domain",
            "status_and_value",
            "membership_evidence",
        ),
    )
