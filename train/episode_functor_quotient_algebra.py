"""Exact bounded quotient-algebra mechanics for SSQAC.

The implementation is deliberately small and auditable:

* coefficients are Python integers reduced modulo the prime 257;
* polynomials and Macaulay rows are sparse;
* columns use deterministic graded lexicographic order;
* row reduction never uses floating point arithmetic; and
* every workspace or closure failure raises instead of returning a guess.

This is a CPU reference mechanism.  It does not parse source, search a
semantic state space, or invoke an external algebra or constraint solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from typing import Iterable, Mapping, Sequence


FIELD_MODULUS = 257
MAX_VARIABLES = 128
MAX_GENERATORS = 128
MAX_MONOMIALS = 4096
MAX_DEGREE = 4
MAX_CLOSURE_DEGREE = 8
MAX_QUOTIENT_DIMENSION = 256
QUOTIENT_CERTIFICATE_SCHEMA = "ssqac_quotient_algebra_certificate_v1"
QUOTIENT_VERIFICATION_SCHEMA = "ssqac_quotient_algebra_verification_v1"
SSQAC_OUTCOME_SCHEMA = "ssqac_quotient_consequence_outcome_v1"
SSQAC_OUTCOME_VERIFICATION_SCHEMA = "ssqac_quotient_outcome_verification_v1"
FIELD_SEMANTICS_SCHEMA = "ssqac_boolean_field_semantics_v1"
STATUS_CERTIFIED = "CERTIFIED"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_UNSAT = "UNSAT"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_RESOURCE_OVERFLOW = "RESOURCE_OVERFLOW"

Monomial = tuple[int, ...]
TermInput = Mapping[Monomial, int] | Iterable[tuple[Monomial, int]]


class QuotientAlgebraError(ValueError):
    """The exact quotient-algebra contract failed."""


class QuotientAlgebraLimitError(QuotientAlgebraError):
    """A fixed SSQAC workspace limit was exceeded."""


class QuotientAlgebraClosureError(QuotientAlgebraError):
    """The admitted Macaulay workspace did not close the quotient border."""


def _require_plain_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuotientAlgebraError(f"{label} must be an integer")
    return value


def _validate_variable_count(num_variables: int) -> int:
    value = _require_plain_int(num_variables, label="variable count")
    if not 1 <= value <= MAX_VARIABLES:
        raise QuotientAlgebraLimitError(
            f"variable count must be in [1, {MAX_VARIABLES}]"
        )
    return value


def _validate_degree_limit(degree_limit: int) -> int:
    value = _require_plain_int(degree_limit, label="degree limit")
    if not 1 <= value <= MAX_CLOSURE_DEGREE:
        raise QuotientAlgebraLimitError(
            f"degree limit must be in [1, {MAX_CLOSURE_DEGREE}]"
        )
    return value


def monomial_degree(monomial: Monomial) -> int:
    """Return the total degree of a validated dense exponent tuple."""

    return sum(monomial)


def monomial_order_key(monomial: Monomial) -> tuple[int, Monomial]:
    """Return the public graded-lex order key.

    Larger total degree leads.  Within a degree, the exponent of ``z_0`` is
    compared first, then ``z_1``, and so on.
    """

    return monomial_degree(monomial), monomial


def _validate_monomial(
    monomial: object,
    *,
    num_variables: int,
    degree_limit: int = MAX_DEGREE,
) -> Monomial:
    if not isinstance(monomial, tuple):
        raise QuotientAlgebraError("monomials must be exponent tuples")
    if len(monomial) != num_variables:
        raise QuotientAlgebraError("monomial width differs from variable count")
    exponents = []
    for exponent in monomial:
        value = _require_plain_int(exponent, label="monomial exponent")
        if value < 0:
            raise QuotientAlgebraError("monomial exponents must be nonnegative")
        exponents.append(value)
    result = tuple(exponents)
    if monomial_degree(result) > degree_limit:
        raise QuotientAlgebraLimitError(f"polynomial degree exceeds {degree_limit}")
    return result


def _zero_monomial(num_variables: int) -> Monomial:
    return (0,) * num_variables


def _coordinate_monomial(
    num_variables: int,
    variable: int,
    exponent: int = 1,
) -> Monomial:
    index = _require_plain_int(variable, label="variable index")
    power = _require_plain_int(exponent, label="variable exponent")
    if not 0 <= index < num_variables:
        raise QuotientAlgebraError("variable index is out of range")
    if power < 0:
        raise QuotientAlgebraError("variable exponent must be nonnegative")
    values = [0] * num_variables
    values[index] = power
    return tuple(values)


def _multiply_monomials(
    left: Monomial,
    right: Monomial,
    *,
    degree_limit: int,
) -> Monomial:
    result = tuple(a + b for a, b in zip(left, right))
    if monomial_degree(result) > degree_limit:
        raise QuotientAlgebraLimitError(f"polynomial degree exceeds {degree_limit}")
    return result


def _subtract_scaled_sparse(
    target: dict[object, int],
    source: Mapping[object, int],
    scale: int,
) -> None:
    factor = scale % FIELD_MODULUS
    if factor == 0:
        return
    for key, coefficient in source.items():
        value = (target.get(key, 0) - factor * coefficient) % FIELD_MODULUS
        if value:
            target[key] = value
        else:
            target.pop(key, None)


def _add_scaled_sparse(
    target: dict[object, int],
    source: Mapping[object, int],
    scale: int,
) -> None:
    factor = scale % FIELD_MODULUS
    if factor == 0:
        return
    for key, coefficient in source.items():
        value = (target.get(key, 0) + factor * coefficient) % FIELD_MODULUS
        if value:
            target[key] = value
        else:
            target.pop(key, None)


@dataclass(frozen=True, slots=True, init=False)
class SparsePolynomial:
    """Canonical sparse polynomial over :math:`F_257`."""

    num_variables: int
    terms: tuple[tuple[Monomial, int], ...]

    def __init__(
        self,
        num_variables: int,
        terms: TermInput = (),
    ) -> None:
        width = _validate_variable_count(num_variables)
        items = terms.items() if isinstance(terms, Mapping) else terms
        combined: dict[Monomial, int] = {}
        for raw_monomial, raw_coefficient in items:
            monomial = _validate_monomial(
                raw_monomial,
                num_variables=width,
            )
            coefficient = (
                _require_plain_int(
                    raw_coefficient,
                    label="polynomial coefficient",
                )
                % FIELD_MODULUS
            )
            if coefficient:
                value = (combined.get(monomial, 0) + coefficient) % FIELD_MODULUS
                if value:
                    combined[monomial] = value
                else:
                    combined.pop(monomial, None)
        canonical = tuple(
            sorted(
                combined.items(),
                key=lambda item: monomial_order_key(item[0]),
                reverse=True,
            )
        )
        if len(canonical) > MAX_MONOMIALS:
            raise QuotientAlgebraLimitError(
                f"polynomial support exceeds {MAX_MONOMIALS}"
            )
        object.__setattr__(self, "num_variables", width)
        object.__setattr__(self, "terms", canonical)

    @classmethod
    def zero(cls, num_variables: int) -> SparsePolynomial:
        return cls(num_variables)

    @classmethod
    def one(cls, num_variables: int) -> SparsePolynomial:
        return cls(
            num_variables,
            {_zero_monomial(num_variables): 1},
        )

    @classmethod
    def constant(
        cls,
        num_variables: int,
        value: int,
    ) -> SparsePolynomial:
        coefficient = _require_plain_int(
            value,
            label="constant coefficient",
        )
        return cls(
            num_variables,
            {_zero_monomial(num_variables): coefficient},
        )

    @classmethod
    def variable(
        cls,
        num_variables: int,
        variable: int,
    ) -> SparsePolynomial:
        return cls(
            num_variables,
            {_coordinate_monomial(num_variables, variable): 1},
        )

    @classmethod
    def monomial(
        cls,
        num_variables: int,
        monomial: Monomial,
        coefficient: int = 1,
    ) -> SparsePolynomial:
        return cls(num_variables, {monomial: coefficient})

    @property
    def degree(self) -> int:
        if not self.terms:
            return -1
        return monomial_degree(self.terms[0][0])

    @property
    def is_zero(self) -> bool:
        return not self.terms

    def as_dict(self) -> dict[Monomial, int]:
        return dict(self.terms)

    def coefficient(self, monomial: Monomial) -> int:
        validated = _validate_monomial(
            monomial,
            num_variables=self.num_variables,
        )
        return self.as_dict().get(validated, 0)

    def _require_same_ring(
        self,
        other: SparsePolynomial,
    ) -> None:
        if not isinstance(other, SparsePolynomial):
            raise QuotientAlgebraError("polynomial operand has the wrong type")
        if self.num_variables != other.num_variables:
            raise QuotientAlgebraError("polynomial variable counts differ")

    def __add__(
        self,
        other: SparsePolynomial,
    ) -> SparsePolynomial:
        self._require_same_ring(other)
        terms = self.as_dict()
        _add_scaled_sparse(terms, other.as_dict(), 1)
        return SparsePolynomial(self.num_variables, terms)

    def __sub__(
        self,
        other: SparsePolynomial,
    ) -> SparsePolynomial:
        self._require_same_ring(other)
        terms = self.as_dict()
        _subtract_scaled_sparse(terms, other.as_dict(), 1)
        return SparsePolynomial(self.num_variables, terms)

    def __neg__(self) -> SparsePolynomial:
        return self.scale(-1)

    def scale(self, coefficient: int) -> SparsePolynomial:
        value = (
            _require_plain_int(
                coefficient,
                label="scale coefficient",
            )
            % FIELD_MODULUS
        )
        return SparsePolynomial(
            self.num_variables,
            (
                (monomial, value * term_coefficient)
                for monomial, term_coefficient in self.terms
            ),
        )

    def __mul__(
        self,
        other: SparsePolynomial | int,
    ) -> SparsePolynomial:
        if isinstance(other, int) and not isinstance(other, bool):
            return self.scale(other)
        if not isinstance(other, SparsePolynomial):
            return NotImplemented
        self._require_same_ring(other)
        terms: dict[Monomial, int] = {}
        for left_monomial, left_coefficient in self.terms:
            for right_monomial, right_coefficient in other.terms:
                monomial = _multiply_monomials(
                    left_monomial,
                    right_monomial,
                    degree_limit=MAX_DEGREE,
                )
                coefficient = (
                    terms.get(monomial, 0) + left_coefficient * right_coefficient
                ) % FIELD_MODULUS
                if coefficient:
                    terms[monomial] = coefficient
                else:
                    terms.pop(monomial, None)
                if len(terms) > MAX_MONOMIALS:
                    raise QuotientAlgebraLimitError(
                        f"polynomial support exceeds {MAX_MONOMIALS}"
                    )
        return SparsePolynomial(self.num_variables, terms)

    def __rmul__(
        self,
        other: int,
    ) -> SparsePolynomial:
        if isinstance(other, int) and not isinstance(other, bool):
            return self.scale(other)
        return NotImplemented

    def multiply_monomial(
        self,
        multiplier: Monomial,
        *,
        degree_limit: int = MAX_DEGREE,
    ) -> SparsePolynomial:
        limit = _validate_degree_limit(degree_limit)
        monomial = _validate_monomial(
            multiplier,
            num_variables=self.num_variables,
            degree_limit=limit,
        )
        return SparsePolynomial(
            self.num_variables,
            (
                (
                    _multiply_monomials(
                        term_monomial,
                        monomial,
                        degree_limit=limit,
                    ),
                    coefficient,
                )
                for term_monomial, coefficient in self.terms
            ),
        )

    def evaluate(self, values: Sequence[int]) -> int:
        if len(values) != self.num_variables:
            raise QuotientAlgebraError("evaluation point has the wrong width")
        point = [
            _require_plain_int(value, label="evaluation value") % FIELD_MODULUS
            for value in values
        ]
        result = 0
        for monomial, coefficient in self.terms:
            term = coefficient
            for value, exponent in zip(point, monomial):
                term = (term * pow(value, exponent, FIELD_MODULUS)) % FIELD_MODULUS
            result = (result + term) % FIELD_MODULUS
        return result

    def recode_variables(
        self,
        permutation: Sequence[int],
    ) -> SparsePolynomial:
        """Rename variables with ``old_index -> new_index`` semantics."""

        if len(permutation) != self.num_variables:
            raise QuotientAlgebraError("variable recoding has the wrong width")
        mapping = tuple(
            _require_plain_int(index, label="recoded variable index")
            for index in permutation
        )
        if sorted(mapping) != list(range(self.num_variables)):
            raise QuotientAlgebraError("variable recoding must be a permutation")
        terms = []
        for monomial, coefficient in self.terms:
            recoded = [0] * self.num_variables
            for old_index, exponent in enumerate(monomial):
                recoded[mapping[old_index]] = exponent
            terms.append((tuple(recoded), coefficient))
        return SparsePolynomial(self.num_variables, terms)

    def canonical_data(self) -> list[list[object]]:
        return [[list(monomial), coefficient] for monomial, coefficient in self.terms]


@dataclass(frozen=True, slots=True, init=False)
class IntegerPolynomial:
    """Canonical unreduced integer polynomial for cross-field validation."""

    num_variables: int
    terms: tuple[tuple[Monomial, int], ...]

    def __init__(
        self,
        num_variables: int,
        terms: TermInput = (),
    ) -> None:
        width = _validate_variable_count(num_variables)
        items = terms.items() if isinstance(terms, Mapping) else terms
        combined: dict[Monomial, int] = {}
        for raw_monomial, raw_coefficient in items:
            monomial = _validate_monomial(
                raw_monomial,
                num_variables=width,
            )
            coefficient = _require_plain_int(
                raw_coefficient,
                label="integer polynomial coefficient",
            )
            value = combined.get(monomial, 0) + coefficient
            if value:
                combined[monomial] = value
            else:
                combined.pop(monomial, None)
        object.__setattr__(self, "num_variables", width)
        object.__setattr__(
            self,
            "terms",
            tuple(
                sorted(
                    combined.items(),
                    key=lambda item: monomial_order_key(item[0]),
                    reverse=True,
                )
            ),
        )

    def evaluate_modulo(
        self,
        values: Sequence[int],
        modulus: int,
    ) -> int:
        prime = _require_plain_int(modulus, label="field modulus")
        if prime <= 2:
            raise QuotientAlgebraError("field modulus must exceed two")
        if len(values) != self.num_variables:
            raise QuotientAlgebraError("evaluation point has the wrong width")
        result = 0
        for monomial, coefficient in self.terms:
            term = coefficient % prime
            for value, exponent in zip(values, monomial, strict=True):
                term = (
                    term
                    * pow(
                        _require_plain_int(value, label="evaluation value") % prime,
                        exponent,
                        prime,
                    )
                ) % prime
            result = (result + term) % prime
        return result


@dataclass(frozen=True, slots=True)
class BooleanFieldSemanticsReceipt:
    schema: str
    variable_count: int
    moduli: tuple[int, ...]
    polynomial_count: int
    assignment_count: int
    zero_set_size: int
    truth_table_sha256: str
    zero_set_sha256: str
    intended_zero_set_sha256: str | None
    gates: tuple[tuple[str, bool], ...]

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(passed for _, passed in self.gates)


def verify_boolean_field_semantics(
    num_variables: int,
    polynomials: Iterable[IntegerPolynomial],
    *,
    moduli: Sequence[int] = (257, 263),
    maximum_variables: int = 16,
    expected_zero_set: Iterable[Sequence[int]] | None = None,
) -> BooleanFieldSemanticsReceipt:
    """Prove cross-field agreement and optional intended Boolean semantics.

    ``expected_zero_set`` is independent of the polynomial encoding. A source
    compiler can enumerate intended satisfying assignments directly and
    require the algebraic encoding to match that semantic oracle exactly.
    """

    width = _validate_variable_count(num_variables)
    maximum = _require_plain_int(
        maximum_variables,
        label="maximum exhaustive Boolean variables",
    )
    if not 1 <= maximum <= 20 or width > maximum:
        raise QuotientAlgebraLimitError(
            "Boolean field-semantics enumeration exceeds its variable bound"
        )
    primes = tuple(
        _require_plain_int(modulus, label="field modulus") for modulus in moduli
    )
    if len(primes) < 2 or len(set(primes)) != len(primes):
        raise QuotientAlgebraError(
            "field semantics require at least two distinct moduli"
        )
    if any(modulus <= 2 for modulus in primes):
        raise QuotientAlgebraError("field moduli must exceed two")
    family = tuple(polynomials)
    if not family:
        raise QuotientAlgebraError(
            "field-semantics polynomial family must not be empty"
        )
    if any(
        not isinstance(polynomial, IntegerPolynomial)
        or polynomial.num_variables != width
        for polynomial in family
    ):
        raise QuotientAlgebraError(
            "field-semantics polynomial family has the wrong ring"
        )
    for polynomial in family:
        for _, coefficient in polynomial.terms:
            if any(coefficient % modulus == 0 for modulus in primes):
                raise QuotientAlgebraError(
                    "nonzero integer coefficient vanishes in a validation field"
                )
    assignments = tuple(product((0, 1), repeat=width))
    truth_tables = []
    zero_sets = []
    for modulus in primes:
        table = tuple(
            tuple(
                polynomial.evaluate_modulo(assignment, modulus) == 0
                for polynomial in family
            )
            for assignment in assignments
        )
        zero_set = tuple(
            assignment
            for assignment, row in zip(assignments, table, strict=True)
            if all(row)
        )
        truth_tables.append(table)
        zero_sets.append(zero_set)
    if any(table != truth_tables[0] for table in truth_tables[1:]):
        raise QuotientAlgebraClosureError(
            "Boolean polynomial truth tables differ across validation fields"
        )
    if any(zero_set != zero_sets[0] for zero_set in zero_sets[1:]):
        raise QuotientAlgebraClosureError(
            "Boolean zero sets differ across validation fields"
        )
    intended_zero_set: tuple[tuple[int, ...], ...] | None = None
    if expected_zero_set is not None:
        intended_rows = []
        for raw_assignment in expected_zero_set:
            assignment = tuple(
                _require_plain_int(value, label="intended Boolean value")
                for value in raw_assignment
            )
            if len(assignment) != width:
                raise QuotientAlgebraError(
                    "intended Boolean assignment has the wrong width"
                )
            if any(value not in (0, 1) for value in assignment):
                raise QuotientAlgebraError(
                    "intended Boolean assignments must contain only zero or one"
                )
            intended_rows.append(assignment)
        intended_zero_set = tuple(sorted(set(intended_rows)))
        if len(intended_zero_set) != len(intended_rows):
            raise QuotientAlgebraError(
                "intended Boolean zero set contains duplicate assignments"
            )
        if intended_zero_set != zero_sets[0]:
            raise QuotientAlgebraClosureError(
                "polynomial zero set differs from intended source semantics"
            )
    table_bytes = json.dumps(
        truth_tables[0],
        separators=(",", ":"),
    ).encode("ascii")
    zero_bytes = json.dumps(
        zero_sets[0],
        separators=(",", ":"),
    ).encode("ascii")
    gates = (
        ("coefficient_nonvanishing", True),
        *(
            (("intended_semantics", True),)
            if intended_zero_set is not None
            else ()
        ),
        ("truth_tables_identical", True),
        ("zero_sets_identical", True),
    )
    intended_bytes = (
        None
        if intended_zero_set is None
        else json.dumps(intended_zero_set, separators=(",", ":")).encode("ascii")
    )
    return BooleanFieldSemanticsReceipt(
        schema=FIELD_SEMANTICS_SCHEMA,
        variable_count=width,
        moduli=primes,
        polynomial_count=len(family),
        assignment_count=len(assignments),
        zero_set_size=len(zero_sets[0]),
        truth_table_sha256=hashlib.sha256(table_bytes).hexdigest(),
        zero_set_sha256=hashlib.sha256(zero_bytes).hexdigest(),
        intended_zero_set_sha256=(
            None
            if intended_bytes is None
            else hashlib.sha256(intended_bytes).hexdigest()
        ),
        gates=gates,
    )


def boolean_generators(
    num_variables: int,
    variables: Iterable[int] | None = None,
) -> tuple[SparsePolynomial, ...]:
    """Return ``z_i^2-z_i`` for the requested coordinates."""

    width = _validate_variable_count(num_variables)
    if variables is None:
        indices = tuple(range(width))
    else:
        indices = tuple(
            _require_plain_int(variable, label="Boolean variable index")
            for variable in variables
        )
    if len(set(indices)) != len(indices):
        raise QuotientAlgebraError("Boolean variable indices must be unique")
    result = []
    for variable in indices:
        if not 0 <= variable < width:
            raise QuotientAlgebraError("Boolean variable index is out of range")
        linear = _coordinate_monomial(width, variable)
        square = _coordinate_monomial(width, variable, 2)
        result.append(SparsePolynomial(width, {square: 1, linear: -1}))
    return tuple(result)


def one_hot_generators(
    num_variables: int,
    groups: Iterable[Iterable[int]],
) -> tuple[SparsePolynomial, ...]:
    """Return exact one-hot sum generators.

    Boolean generators must also be supplied for the group coordinates.  With
    at most 128 Boolean coordinates, ``sum(group)-1`` is equivalent to exactly
    one selected coordinate over ``F_257``.
    """

    width = _validate_variable_count(num_variables)
    result = []
    for raw_group in groups:
        group = tuple(
            _require_plain_int(variable, label="one-hot variable index")
            for variable in raw_group
        )
        if not group:
            raise QuotientAlgebraError("one-hot groups cannot be empty")
        if len(set(group)) != len(group):
            raise QuotientAlgebraError("one-hot groups cannot repeat a variable")
        if any(not 0 <= variable < width for variable in group):
            raise QuotientAlgebraError("one-hot variable index is out of range")
        terms = {_coordinate_monomial(width, variable): 1 for variable in group}
        terms[_zero_monomial(width)] = -1
        result.append(SparsePolynomial(width, terms))
    return tuple(result)


def boolean_one_hot_generators(
    num_variables: int,
    groups: Iterable[Iterable[int]],
    *,
    boolean_variables: Iterable[int] | None = None,
) -> tuple[SparsePolynomial, ...]:
    """Return the public Boolean and one-hot schema generators."""

    frozen_groups = tuple(tuple(group) for group in groups)
    if boolean_variables is None:
        variables = sorted({variable for group in frozen_groups for variable in group})
    else:
        variables = tuple(boolean_variables)
    return (
        *boolean_generators(num_variables, variables),
        *one_hot_generators(num_variables, frozen_groups),
    )


@dataclass(frozen=True, slots=True, order=True)
class MacaulaySource:
    """One canonical generator multiplied by one admitted monomial."""

    generator_index: int
    multiplier: Monomial


@dataclass(frozen=True, slots=True)
class MembershipTerm:
    """One coefficient in an exact ideal-membership witness."""

    generator_index: int
    multiplier: Monomial
    coefficient: int


@dataclass(frozen=True, slots=True)
class IdealMembershipEvidence:
    """Exact bounded-degree representation of a target in the ideal."""

    target: SparsePolynomial
    terms: tuple[MembershipTerm, ...]


@dataclass(frozen=True, slots=True)
class RrefRowCertificate:
    """One exact RREF row and its generator-multiple provenance."""

    pivot: Monomial
    coefficients: tuple[tuple[Monomial, int], ...]
    provenance: tuple[MembershipTerm, ...]


@dataclass(frozen=True, slots=True)
class QuotientAlgebraCertificate:
    """Portable evidence for the bounded quotient and its prolongation."""

    schema: str
    field_modulus: int
    variable_count: int
    degree_limit: int
    require_boolean_schema: bool
    generators: tuple[tuple[tuple[Monomial, int], ...], ...]
    admitted_monomials: tuple[Monomial, ...]
    rows: tuple[RrefRowCertificate, ...]
    prolongation_degree: int
    prolongation_monomials: tuple[Monomial, ...]
    prolongation_rows: tuple[RrefRowCertificate, ...]
    quotient_digest: str

    def canonical_bytes(self) -> bytes:
        payload = {
            "admitted_monomials": [
                list(monomial) for monomial in self.admitted_monomials
            ],
            "degree_limit": self.degree_limit,
            "field_modulus": self.field_modulus,
            "generators": [
                [
                    [list(monomial), coefficient]
                    for monomial, coefficient in generator
                ]
                for generator in self.generators
            ],
            "prolongation_degree": self.prolongation_degree,
            "prolongation_monomials": [
                list(monomial) for monomial in self.prolongation_monomials
            ],
            "prolongation_rows": [
                _rref_row_certificate_data(row) for row in self.prolongation_rows
            ],
            "quotient_digest": self.quotient_digest,
            "require_boolean_schema": self.require_boolean_schema,
            "rows": [_rref_row_certificate_data(row) for row in self.rows],
            "schema": self.schema,
            "variable_count": self.variable_count,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")

    @property
    def certificate_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class QuotientCertificateVerification:
    """Independent fail-closed verification result."""

    schema: str
    certificate_sha256: str
    quotient_digest: str
    main_rank: int
    prolongation_rank: int
    quotient_dimension: int
    gates: tuple[tuple[str, bool], ...]

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(passed for _, passed in self.gates)


@dataclass(frozen=True, slots=True)
class SSQACConsequenceOutcome:
    """Public fail-closed status for one exact consequence request."""

    schema: str
    status: str
    value: int | None
    query: SparsePolynomial | None
    allowed_values: tuple[int, ...]
    quotient_receipt: QuotientAlgebraReceipt | None
    consequence: ConsequenceReceipt | None
    certificate: QuotientAlgebraCertificate | None
    diagnostic: str | None

    def canonical_bytes(self) -> bytes:
        payload = {
            "allowed_values": list(self.allowed_values),
            "certificate_sha256": (
                None
                if self.certificate is None
                else self.certificate.certificate_sha256
            ),
            "consequence": _consequence_receipt_data(self.consequence),
            "diagnostic": self.diagnostic,
            "query": (
                None if self.query is None else self.query.canonical_data()
            ),
            "quotient_digest": (
                None
                if self.quotient_receipt is None
                else self.quotient_receipt.canonical_digest
            ),
            "schema": self.schema,
            "status": self.status,
            "value": self.value,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")

    @property
    def outcome_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class SSQACOutcomeVerification:
    schema: str
    outcome_sha256: str
    status: str
    gates: tuple[tuple[str, bool], ...]

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(passed for _, passed in self.gates)


@dataclass(frozen=True, slots=True)
class QuotientAlgebraReceipt:
    """Deterministic semantic receipt for a closed bounded quotient."""

    status: str
    field_modulus: int
    variable_count: int
    degree_limit: int
    monomial_count: int
    rank: int
    quotient_dimension: int
    prolongation_degree: int
    prolongation_monomial_count: int
    prolongation_rank: int
    boolean_schema_verified: bool
    pivot_monomials: tuple[Monomial, ...]
    basis_monomials: tuple[Monomial, ...]
    canonical_digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "basis_monomials": [list(monomial) for monomial in self.basis_monomials],
            "boolean_schema_verified": self.boolean_schema_verified,
            "canonical_digest": self.canonical_digest,
            "degree_limit": self.degree_limit,
            "field_modulus": self.field_modulus,
            "monomial_count": self.monomial_count,
            "pivot_monomials": [list(monomial) for monomial in self.pivot_monomials],
            "prolongation_degree": self.prolongation_degree,
            "prolongation_monomial_count": self.prolongation_monomial_count,
            "prolongation_rank": self.prolongation_rank,
            "quotient_dimension": self.quotient_dimension,
            "rank": self.rank,
            "status": self.status,
            "variable_count": self.variable_count,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class ConsequenceReceipt:
    """Forced, ambiguous, or inconsistent polynomial consequence."""

    status: str
    value: int | None
    normal_form: SparsePolynomial
    quotient_digest: str
    evidence: IdealMembershipEvidence | None
    domain_verified: bool


@dataclass(slots=True)
class _TrackedRow:
    coefficients: dict[int, int]
    provenance: dict[MacaulaySource, int]


def _rref_row_certificate_data(row: RrefRowCertificate) -> dict[str, object]:
    return {
        "coefficients": [
            [list(monomial), coefficient]
            for monomial, coefficient in row.coefficients
        ],
        "pivot": list(row.pivot),
        "provenance": [
            {
                "coefficient": term.coefficient,
                "generator_index": term.generator_index,
                "multiplier": list(term.multiplier),
            }
            for term in row.provenance
        ],
    }


def _membership_evidence_data(
    evidence: IdealMembershipEvidence | None,
) -> dict[str, object] | None:
    if evidence is None:
        return None
    return {
        "target": evidence.target.canonical_data(),
        "terms": [
            {
                "coefficient": term.coefficient,
                "generator_index": term.generator_index,
                "multiplier": list(term.multiplier),
            }
            for term in evidence.terms
        ],
    }


def _consequence_receipt_data(
    receipt: ConsequenceReceipt | None,
) -> dict[str, object] | None:
    if receipt is None:
        return None
    return {
        "domain_verified": receipt.domain_verified,
        "evidence": _membership_evidence_data(receipt.evidence),
        "normal_form": receipt.normal_form.canonical_data(),
        "quotient_digest": receipt.quotient_digest,
        "status": receipt.status,
        "value": receipt.value,
    }


def _enumerate_monomials(
    num_variables: int,
    degree_limit: int,
) -> tuple[Monomial, ...]:
    expected = math.comb(num_variables + degree_limit, degree_limit)
    if expected > MAX_MONOMIALS:
        raise QuotientAlgebraLimitError(
            f"admitted monomial count {expected} exceeds {MAX_MONOMIALS}"
        )
    result: list[Monomial] = []

    def append_compositions(
        variable: int,
        remaining: int,
        prefix: list[int],
    ) -> None:
        if variable == num_variables - 1:
            result.append(tuple((*prefix, remaining)))
            return
        for exponent in range(remaining, -1, -1):
            append_compositions(
                variable + 1,
                remaining - exponent,
                [*prefix, exponent],
            )

    for degree in range(degree_limit + 1):
        append_compositions(0, degree, [])
    return tuple(
        sorted(
            result,
            key=monomial_order_key,
            reverse=True,
        )
    )


def _prepare_admitted_monomials(
    num_variables: int,
    degree_limit: int,
    admitted_monomials: Iterable[Monomial] | None,
) -> tuple[Monomial, ...]:
    if admitted_monomials is None:
        return _enumerate_monomials(
            num_variables,
            degree_limit,
        )
    unique = {
        _validate_monomial(
            monomial,
            num_variables=num_variables,
            degree_limit=degree_limit,
        )
        for monomial in admitted_monomials
    }
    if len(unique) > MAX_MONOMIALS:
        raise QuotientAlgebraLimitError(
            f"admitted monomial count exceeds {MAX_MONOMIALS}"
        )
    if _zero_monomial(num_variables) not in unique:
        raise QuotientAlgebraClosureError(
            "admitted monomials must contain the constant"
        )
    for monomial in tuple(unique):
        for variable, exponent in enumerate(monomial):
            if exponent == 0:
                continue
            divisor = list(monomial)
            divisor[variable] -= 1
            if tuple(divisor) not in unique:
                raise QuotientAlgebraClosureError(
                    "admitted monomials must be divisor-closed"
                )
    return tuple(
        sorted(
            unique,
            key=monomial_order_key,
            reverse=True,
        )
    )


def _canonical_generators(
    num_variables: int,
    generators: Iterable[SparsePolynomial],
) -> tuple[SparsePolynomial, ...]:
    supplied = tuple(generators)
    if len(supplied) > MAX_GENERATORS:
        raise QuotientAlgebraLimitError(f"generator count exceeds {MAX_GENERATORS}")
    for generator in supplied:
        if not isinstance(generator, SparsePolynomial):
            raise QuotientAlgebraError("generators must be sparse polynomials")
        if generator.num_variables != num_variables:
            raise QuotientAlgebraError("generator variable counts differ")
    unique = {
        generator.terms: generator for generator in supplied if not generator.is_zero
    }
    return tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda terms: tuple(
                (monomial_order_key(monomial), coefficient)
                for monomial, coefficient in terms
            ),
            reverse=True,
        )
    )


def _row_reduce(
    generators: tuple[SparsePolynomial, ...],
    monomials: tuple[Monomial, ...],
    *,
    degree_limit: int,
) -> dict[int, _TrackedRow]:
    column_by_monomial = {monomial: column for column, monomial in enumerate(monomials)}
    admitted = set(monomials)
    pivots: dict[int, _TrackedRow] = {}

    for generator_index, generator in enumerate(generators):
        for multiplier in reversed(monomials):
            products = []
            valid = True
            for monomial, coefficient in generator.terms:
                try:
                    product = _multiply_monomials(
                        monomial,
                        multiplier,
                        degree_limit=degree_limit,
                    )
                except QuotientAlgebraLimitError:
                    valid = False
                    break
                if product not in admitted:
                    valid = False
                    break
                products.append((product, coefficient))
            if not valid:
                continue
            coefficients = {
                column_by_monomial[monomial]: coefficient
                for monomial, coefficient in products
                if coefficient
            }
            if not coefficients:
                continue
            source = MacaulaySource(
                generator_index=generator_index,
                multiplier=multiplier,
            )
            provenance = {source: 1}
            for pivot in sorted(pivots):
                factor = coefficients.get(pivot, 0)
                if factor:
                    _subtract_scaled_sparse(
                        coefficients,
                        pivots[pivot].coefficients,
                        factor,
                    )
                    _subtract_scaled_sparse(
                        provenance,
                        pivots[pivot].provenance,
                        factor,
                    )
            if not coefficients:
                continue
            pivot = min(coefficients)
            inverse = pow(
                coefficients[pivot],
                -1,
                FIELD_MODULUS,
            )
            coefficients = {
                column: coefficient * inverse % FIELD_MODULUS
                for column, coefficient in coefficients.items()
            }
            provenance = {
                item: coefficient * inverse % FIELD_MODULUS
                for item, coefficient in provenance.items()
            }
            for tracked in pivots.values():
                factor = tracked.coefficients.get(pivot, 0)
                if factor:
                    _subtract_scaled_sparse(
                        tracked.coefficients,
                        coefficients,
                        factor,
                    )
                    _subtract_scaled_sparse(
                        tracked.provenance,
                        provenance,
                        factor,
                    )
            pivots[pivot] = _TrackedRow(
                coefficients=coefficients,
                provenance=provenance,
            )
    return dict(sorted(pivots.items()))


def _verify_stable_prolongation(
    *,
    num_variables: int,
    degree_limit: int,
    generators: tuple[SparsePolynomial, ...],
    quotient_dimension: int,
) -> tuple[int, tuple[Monomial, ...], dict[int, _TrackedRow]]:
    prolongation_degree = degree_limit + 1
    monomials = _enumerate_monomials(
        num_variables,
        prolongation_degree,
    )
    pivots = _row_reduce(
        generators,
        monomials,
        degree_limit=prolongation_degree,
    )
    prolonged_dimension = len(monomials) - len(pivots)
    if prolonged_dimension != quotient_dimension:
        raise QuotientAlgebraClosureError(
            "quotient rank is not stable under complete prolongation"
        )
    return prolongation_degree, monomials, pivots


def _tracked_row_certificate(
    tracked: _TrackedRow,
    monomials: Sequence[Monomial],
) -> RrefRowCertificate:
    pivot = min(tracked.coefficients)
    return RrefRowCertificate(
        pivot=monomials[pivot],
        coefficients=tuple(
            (monomials[column], coefficient)
            for column, coefficient in sorted(tracked.coefficients.items())
        ),
        provenance=tuple(
            MembershipTerm(
                generator_index=source.generator_index,
                multiplier=source.multiplier,
                coefficient=coefficient,
            )
            for source, coefficient in sorted(
                tracked.provenance.items(),
                key=lambda item: (
                    item[0].generator_index,
                    monomial_order_key(item[0].multiplier),
                ),
            )
            if coefficient
        ),
    )


def _semantic_digest_payload(
    *,
    status: str,
    num_variables: int,
    degree_limit: int,
    monomials: tuple[Monomial, ...],
    pivots: Mapping[int, _TrackedRow],
    basis_columns: tuple[int, ...],
    prolongation_degree: int,
    prolongation_monomial_count: int,
    prolongation_rank: int,
    boolean_schema_verified: bool,
) -> dict[str, object]:
    return {
        "basis": [list(monomials[column]) for column in basis_columns],
        "boolean_schema_verified": boolean_schema_verified,
        "degree_limit": degree_limit,
        "field_modulus": FIELD_MODULUS,
        "monomials": [list(monomial) for monomial in monomials],
        "prolongation_degree": prolongation_degree,
        "prolongation_monomial_count": prolongation_monomial_count,
        "prolongation_rank": prolongation_rank,
        "rref": [
            [
                pivot,
                [
                    [column, coefficient]
                    for column, coefficient in sorted(tracked.coefficients.items())
                ],
            ]
            for pivot, tracked in sorted(pivots.items())
        ],
        "status": status,
        "variable_count": num_variables,
    }


class QuotientAlgebra:
    """Closed exact quotient with deterministic bounded normal forms."""

    __slots__ = (
        "num_variables",
        "degree_limit",
        "monomials",
        "canonical_generators",
        "_column_by_monomial",
        "_pivots",
        "_basis_columns",
        "_basis_index",
        "_multiplication_columns",
        "_prolongation_monomials",
        "_prolongation_pivots",
        "_require_boolean_schema",
        "receipt",
    )

    def __init__(
        self,
        *,
        num_variables: int,
        degree_limit: int,
        monomials: tuple[Monomial, ...],
        generators: tuple[SparsePolynomial, ...],
        pivots: dict[int, _TrackedRow],
        prolongation_degree: int,
        prolongation_monomials: tuple[Monomial, ...],
        prolongation_pivots: dict[int, _TrackedRow],
        require_boolean_schema: bool,
    ) -> None:
        self.num_variables = num_variables
        self.degree_limit = degree_limit
        self.monomials = monomials
        self.canonical_generators = generators
        self._column_by_monomial = {
            monomial: column for column, monomial in enumerate(monomials)
        }
        self._pivots = pivots
        self._prolongation_monomials = prolongation_monomials
        self._prolongation_pivots = prolongation_pivots
        self._require_boolean_schema = require_boolean_schema
        self._basis_columns = tuple(
            column for column in range(len(monomials)) if column not in pivots
        )
        self._basis_index = {
            column: index for index, column in enumerate(self._basis_columns)
        }
        constant_column = self._column_by_monomial[_zero_monomial(num_variables)]
        inconsistent = constant_column in pivots
        if not inconsistent:
            self._verify_order_ideal()
        if len(self._basis_columns) > MAX_QUOTIENT_DIMENSION:
            raise QuotientAlgebraLimitError(
                "quotient dimension "
                f"{len(self._basis_columns)} exceeds "
                f"{MAX_QUOTIENT_DIMENSION}"
            )
        status = "inconsistent" if inconsistent else "ok"
        self._multiplication_columns = (
            () if inconsistent else self._build_and_verify_multiplication()
        )
        boolean_schema_verified = False
        if not inconsistent:
            self._verify_generators_reduce_to_zero()
            boolean_schema_verified = self._verify_boolean_schema(
                required=require_boolean_schema,
            )
            if boolean_schema_verified:
                self._verify_boolean_operator_idempotence()
        prolongation_monomial_count = len(prolongation_monomials)
        prolongation_rank = len(prolongation_pivots)
        payload = _semantic_digest_payload(
            status=status,
            num_variables=num_variables,
            degree_limit=degree_limit,
            monomials=monomials,
            pivots=pivots,
            basis_columns=self._basis_columns,
            prolongation_degree=prolongation_degree,
            prolongation_monomial_count=prolongation_monomial_count,
            prolongation_rank=prolongation_rank,
            boolean_schema_verified=boolean_schema_verified,
        )
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest = hashlib.sha256(canonical).hexdigest()
        self.receipt = QuotientAlgebraReceipt(
            status=status,
            field_modulus=FIELD_MODULUS,
            variable_count=num_variables,
            degree_limit=degree_limit,
            monomial_count=len(monomials),
            rank=len(pivots),
            quotient_dimension=len(self._basis_columns),
            prolongation_degree=prolongation_degree,
            prolongation_monomial_count=prolongation_monomial_count,
            prolongation_rank=prolongation_rank,
            boolean_schema_verified=boolean_schema_verified,
            pivot_monomials=tuple(monomials[column] for column in pivots),
            basis_monomials=tuple(monomials[column] for column in self._basis_columns),
            canonical_digest=digest,
        )

    def export_certificate(self) -> QuotientAlgebraCertificate:
        """Export complete exact evidence for an independent verifier."""

        return QuotientAlgebraCertificate(
            schema=QUOTIENT_CERTIFICATE_SCHEMA,
            field_modulus=FIELD_MODULUS,
            variable_count=self.num_variables,
            degree_limit=self.degree_limit,
            require_boolean_schema=self._require_boolean_schema,
            generators=tuple(
                tuple(generator.terms) for generator in self.canonical_generators
            ),
            admitted_monomials=self.monomials,
            rows=tuple(
                _tracked_row_certificate(tracked, self.monomials)
                for _, tracked in sorted(self._pivots.items())
            ),
            prolongation_degree=self.receipt.prolongation_degree,
            prolongation_monomials=self._prolongation_monomials,
            prolongation_rows=tuple(
                _tracked_row_certificate(
                    tracked,
                    self._prolongation_monomials,
                )
                for _, tracked in sorted(self._prolongation_pivots.items())
            ),
            quotient_digest=self.receipt.canonical_digest,
        )

    @property
    def is_consistent(self) -> bool:
        return self.receipt.status == "ok"

    def _verify_order_ideal(self) -> None:
        basis = {self.monomials[column] for column in self._basis_columns}
        constant = _zero_monomial(self.num_variables)
        if constant not in basis:
            raise QuotientAlgebraClosureError(
                "quotient basis is not an order ideal connected to 1"
            )
        for monomial in basis:
            for variable, exponent in enumerate(monomial):
                if exponent == 0:
                    continue
                divisor = list(monomial)
                divisor[variable] -= 1
                if tuple(divisor) not in basis:
                    raise QuotientAlgebraClosureError(
                        "quotient basis is not divisor-closed"
                    )

    def _verify_generators_reduce_to_zero(self) -> None:
        for generator in self.canonical_generators:
            if not self.normal_form(generator).is_zero:
                raise QuotientAlgebraClosureError(
                    "source generator has a nonzero quotient normal form"
                )

    def _verify_boolean_schema(self, *, required: bool) -> bool:
        verified = True
        for variable in range(self.num_variables):
            coordinate = SparsePolynomial.variable(
                self.num_variables,
                variable,
            )
            boolean_relation = coordinate * coordinate - coordinate
            if not self.normal_form(boolean_relation).is_zero:
                verified = False
                break
        if required and not verified:
            raise QuotientAlgebraClosureError(
                "Boolean schema is not implied for every quotient variable"
            )
        return verified

    def _verify_boolean_operator_idempotence(self) -> None:
        for operator in self._multiplication_columns:
            for basis_index in range(len(self._basis_columns)):
                unit = {basis_index: 1}
                once = self._apply_sparse_operator(operator, unit)
                twice = self._apply_sparse_operator(operator, once)
                if twice != once:
                    raise QuotientAlgebraClosureError(
                        "Boolean multiplication operator is not idempotent"
                    )

    def _reduce_row(
        self,
        coefficients: dict[int, int],
        *,
        collect_evidence: bool,
    ) -> tuple[dict[int, int], dict[MacaulaySource, int]]:
        remainder = dict(coefficients)
        evidence: dict[MacaulaySource, int] = {}
        for pivot, tracked in self._pivots.items():
            factor = remainder.get(pivot, 0)
            if not factor:
                continue
            _subtract_scaled_sparse(
                remainder,
                tracked.coefficients,
                factor,
            )
            if collect_evidence:
                _add_scaled_sparse(
                    evidence,
                    tracked.provenance,
                    factor,
                )
        return remainder, evidence

    def _polynomial_row(
        self,
        polynomial: SparsePolynomial,
    ) -> dict[int, int]:
        if not isinstance(polynomial, SparsePolynomial):
            raise QuotientAlgebraError("normal-form input must be a sparse polynomial")
        if polynomial.num_variables != self.num_variables:
            raise QuotientAlgebraError("normal-form variable count differs")
        row = {}
        for monomial, coefficient in polynomial.terms:
            column = self._column_by_monomial.get(monomial)
            if column is None:
                raise QuotientAlgebraClosureError(
                    "polynomial leaves the admitted monomial workspace"
                )
            row[column] = coefficient
        return row

    def _row_polynomial(
        self,
        coefficients: Mapping[int, int],
    ) -> SparsePolynomial:
        return SparsePolynomial(
            self.num_variables,
            (
                (self.monomials[column], coefficient)
                for column, coefficient in coefficients.items()
                if coefficient
            ),
        )

    def normal_form(
        self,
        polynomial: SparsePolynomial,
    ) -> SparsePolynomial:
        remainder, _ = self._reduce_row(
            self._polynomial_row(polynomial),
            collect_evidence=False,
        )
        return self._row_polynomial(remainder)

    def membership_evidence(
        self,
        polynomial: SparsePolynomial,
    ) -> IdealMembershipEvidence | None:
        row = self._polynomial_row(polynomial)
        remainder, evidence = self._reduce_row(
            row,
            collect_evidence=True,
        )
        if remainder:
            return None
        terms = tuple(
            MembershipTerm(
                generator_index=source.generator_index,
                multiplier=source.multiplier,
                coefficient=coefficient,
            )
            for source, coefficient in sorted(
                evidence.items(),
                key=lambda item: (
                    item[0].generator_index,
                    monomial_order_key(item[0].multiplier),
                ),
            )
            if coefficient
        )
        return IdealMembershipEvidence(
            target=polynomial,
            terms=terms,
        )

    def verify_membership_evidence(
        self,
        evidence: IdealMembershipEvidence,
    ) -> bool:
        if not isinstance(evidence, IdealMembershipEvidence):
            return False
        if evidence.target.num_variables != self.num_variables:
            return False
        reconstructed = SparsePolynomial.zero(self.num_variables)
        try:
            for term in evidence.terms:
                if not (0 <= term.generator_index < len(self.canonical_generators)):
                    return False
                coefficient = _require_plain_int(
                    term.coefficient,
                    label="membership coefficient",
                )
                product = self.canonical_generators[
                    term.generator_index
                ].multiply_monomial(
                    term.multiplier,
                    degree_limit=self.degree_limit,
                )
                reconstructed = reconstructed + product.scale(coefficient)
        except QuotientAlgebraError:
            return False
        return reconstructed == evidence.target

    def decide_polynomial(
        self,
        polynomial: SparsePolynomial,
        *,
        allowed_values: Iterable[int],
    ) -> ConsequenceReceipt:
        normal = self.normal_form(polynomial)
        if not self.is_consistent:
            return ConsequenceReceipt(
                status="inconsistent",
                value=None,
                normal_form=normal,
                quotient_digest=self.receipt.canonical_digest,
                evidence=None,
                domain_verified=False,
            )
        values = tuple(
            _require_plain_int(
                value,
                label="allowed consequence value",
            )
            % FIELD_MODULUS
            for value in allowed_values
        )
        if not values or len(set(values)) != len(values):
            raise QuotientAlgebraError(
                "allowed values must be nonempty and unique modulo 257"
            )
        if not self._allowed_value_domain_verified(normal, values):
            return ConsequenceReceipt(
                status="out_of_domain",
                value=None,
                normal_form=normal,
                quotient_digest=self.receipt.canonical_digest,
                evidence=None,
                domain_verified=False,
            )
        forced = []
        for value in values:
            difference = polynomial - SparsePolynomial.constant(
                self.num_variables,
                value,
            )
            evidence = self.membership_evidence(difference)
            if evidence is not None:
                forced.append((value, evidence))
        if len(forced) == 1:
            value, evidence = forced[0]
            return ConsequenceReceipt(
                status="forced",
                value=value,
                normal_form=normal,
                quotient_digest=self.receipt.canonical_digest,
                evidence=evidence,
                domain_verified=True,
            )
        if forced:
            raise QuotientAlgebraClosureError(
                "consistent quotient forced multiple field values"
            )
        return ConsequenceReceipt(
            status="ambiguous",
            value=None,
            normal_form=normal,
            quotient_digest=self.receipt.canonical_digest,
            evidence=None,
            domain_verified=True,
        )

    def _apply_polynomial_operator(
        self,
        polynomial: SparsePolynomial,
        vector: Mapping[int, int],
    ) -> dict[int, int]:
        result: dict[int, int] = {}
        for monomial, coefficient in polynomial.terms:
            term = dict(vector)
            for variable, exponent in enumerate(monomial):
                for _ in range(exponent):
                    term = self._apply_sparse_operator(
                        self._multiplication_columns[variable],
                        term,
                    )
            _add_scaled_sparse(result, term, coefficient)
        return result

    def _allowed_value_domain_verified(
        self,
        normal_form: SparsePolynomial,
        values: Sequence[int],
    ) -> bool:
        for basis_index in range(len(self._basis_columns)):
            vector: dict[int, int] = {basis_index: 1}
            for value in values:
                transformed = self._apply_polynomial_operator(
                    normal_form,
                    vector,
                )
                _subtract_scaled_sparse(transformed, vector, value)
                vector = transformed
            if vector:
                return False
        return True

    def decide_coordinate(
        self,
        variable: int,
        *,
        allowed_values: Iterable[int] = (0, 1),
    ) -> ConsequenceReceipt:
        return self.decide_polynomial(
            SparsePolynomial.variable(
                self.num_variables,
                variable,
            ),
            allowed_values=allowed_values,
        )

    def _normal_form_vector_for_column(
        self,
        column: int,
    ) -> tuple[tuple[int, int], ...]:
        remainder, _ = self._reduce_row(
            {column: 1},
            collect_evidence=False,
        )
        if any(output_column not in self._basis_index for output_column in remainder):
            raise QuotientAlgebraClosureError("normal form retained a pivot monomial")
        return tuple(
            sorted(
                (
                    self._basis_index[output_column],
                    coefficient,
                )
                for output_column, coefficient in remainder.items()
                if coefficient
            )
        )

    @staticmethod
    def _apply_sparse_operator(
        operator: tuple[tuple[tuple[int, int], ...], ...],
        vector: Mapping[int, int],
    ) -> dict[int, int]:
        result: dict[int, int] = {}
        for input_index, input_coefficient in vector.items():
            for output_index, coefficient in operator[input_index]:
                value = (
                    result.get(output_index, 0) + input_coefficient * coefficient
                ) % FIELD_MODULUS
                if value:
                    result[output_index] = value
                else:
                    result.pop(output_index, None)
        return result

    def _build_and_verify_multiplication(
        self,
    ) -> tuple[
        tuple[tuple[tuple[int, int], ...], ...],
        ...,
    ]:
        operators = []
        for variable in range(self.num_variables):
            columns = []
            coordinate = _coordinate_monomial(
                self.num_variables,
                variable,
            )
            for basis_column in self._basis_columns:
                basis_monomial = self.monomials[basis_column]
                try:
                    product = _multiply_monomials(
                        basis_monomial,
                        coordinate,
                        degree_limit=self.degree_limit,
                    )
                except QuotientAlgebraLimitError as error:
                    raise QuotientAlgebraClosureError(
                        "quotient border exceeds the degree limit"
                    ) from error
                product_column = self._column_by_monomial.get(product)
                if product_column is None:
                    raise QuotientAlgebraClosureError(
                        "quotient border leaves admitted monomials"
                    )
                columns.append(self._normal_form_vector_for_column(product_column))
            operators.append(tuple(columns))
        frozen = tuple(operators)
        for left in range(self.num_variables):
            for right in range(left + 1, self.num_variables):
                for basis_index in range(len(self._basis_columns)):
                    unit = {basis_index: 1}
                    left_then_right = self._apply_sparse_operator(
                        frozen[right],
                        self._apply_sparse_operator(
                            frozen[left],
                            unit,
                        ),
                    )
                    right_then_left = self._apply_sparse_operator(
                        frozen[left],
                        self._apply_sparse_operator(
                            frozen[right],
                            unit,
                        ),
                    )
                    if left_then_right != right_then_left:
                        raise QuotientAlgebraClosureError(
                            "bounded multiplication operators do not commute"
                        )
        return frozen


def _compile_quotient_algebra_at_degree(
    num_variables: int,
    generators: Iterable[SparsePolynomial],
    *,
    degree_limit: int,
    admitted_monomials: Iterable[Monomial] | None = None,
    require_boolean_schema: bool = False,
) -> QuotientAlgebra:
    """Compile one exact bounded quotient workspace without retrying."""

    width = _validate_variable_count(num_variables)
    limit = _validate_degree_limit(degree_limit)
    if not isinstance(require_boolean_schema, bool):
        raise QuotientAlgebraError("require_boolean_schema must be Boolean")
    canonical_generators = _canonical_generators(
        width,
        generators,
    )
    monomials = _prepare_admitted_monomials(
        width,
        limit,
        admitted_monomials,
    )
    admitted = set(monomials)
    for generator in canonical_generators:
        if generator.degree > limit:
            raise QuotientAlgebraLimitError(f"generator degree exceeds {limit}")
        if any(monomial not in admitted for monomial, _ in generator.terms):
            raise QuotientAlgebraClosureError(
                "generator support leaves admitted monomials"
            )
    pivots = _row_reduce(
        canonical_generators,
        monomials,
        degree_limit=limit,
    )
    quotient_dimension = len(monomials) - len(pivots)
    if quotient_dimension > MAX_QUOTIENT_DIMENSION:
        raise QuotientAlgebraLimitError(
            "quotient dimension "
            f"{quotient_dimension} exceeds "
            f"{MAX_QUOTIENT_DIMENSION}"
        )
    (
        prolongation_degree,
        prolongation_monomials,
        prolongation_pivots,
    ) = _verify_stable_prolongation(
        num_variables=width,
        degree_limit=limit,
        generators=canonical_generators,
        quotient_dimension=quotient_dimension,
    )
    return QuotientAlgebra(
        num_variables=width,
        degree_limit=limit,
        monomials=monomials,
        generators=canonical_generators,
        pivots=pivots,
        prolongation_degree=prolongation_degree,
        prolongation_monomials=prolongation_monomials,
        prolongation_pivots=prolongation_pivots,
        require_boolean_schema=require_boolean_schema,
    )


def compile_quotient_algebra(
    num_variables: int,
    generators: Iterable[SparsePolynomial],
    *,
    degree_limit: int = MAX_DEGREE,
    admitted_monomials: Iterable[Monomial] | None = None,
    require_boolean_schema: bool = False,
    adaptive_closure: bool = True,
) -> QuotientAlgebra:
    """Compile a fail-closed exact quotient with bounded adaptive closure.

    ``degree_limit`` is the first complete Macaulay workspace to test.  When
    that workspace has not stabilized, the compiler increases the complete
    degree one at a time, never beyond ``MAX_CLOSURE_DEGREE`` or the fixed
    monomial/resource limits.  The returned receipt and certificate record the
    actual degree used.  Caller-supplied admitted monomials are never expanded.
    """

    if not isinstance(adaptive_closure, bool):
        raise QuotientAlgebraError("adaptive_closure must be Boolean")
    width = _validate_variable_count(num_variables)
    first_degree = _validate_degree_limit(degree_limit)
    frozen_generators = tuple(generators)
    if admitted_monomials is not None or not adaptive_closure:
        return _compile_quotient_algebra_at_degree(
            width,
            frozen_generators,
            degree_limit=first_degree,
            admitted_monomials=admitted_monomials,
            require_boolean_schema=require_boolean_schema,
        )

    last_closure_error: QuotientAlgebraClosureError | None = None
    for candidate_degree in range(first_degree, MAX_CLOSURE_DEGREE + 1):
        try:
            return _compile_quotient_algebra_at_degree(
                width,
                frozen_generators,
                degree_limit=candidate_degree,
                require_boolean_schema=require_boolean_schema,
            )
        except QuotientAlgebraClosureError as error:
            message = str(error)
            if not (
                "stable under complete prolongation" in message
                or "quotient border exceeds the degree limit" in message
                or "quotient border leaves admitted monomials" in message
            ):
                raise
            last_closure_error = error
        except QuotientAlgebraLimitError:
            raise
    if last_closure_error is None:
        raise QuotientAlgebraClosureError("adaptive quotient closure did not run")
    raise QuotientAlgebraClosureError(
        "adaptive quotient closure did not stabilize through degree "
        f"{MAX_CLOSURE_DEGREE}"
    ) from last_closure_error


def _certificate_rows_to_pivots(
    rows: Sequence[RrefRowCertificate],
    *,
    monomials: tuple[Monomial, ...],
    generators: tuple[SparsePolynomial, ...],
    degree_limit: int,
) -> dict[int, _TrackedRow]:
    column_by_monomial = {
        monomial: column for column, monomial in enumerate(monomials)
    }
    admitted = set(monomials)
    pivots: dict[int, _TrackedRow] = {}
    for row in rows:
        if not isinstance(row, RrefRowCertificate):
            raise QuotientAlgebraError("certificate row has the wrong type")
        try:
            pivot = column_by_monomial[row.pivot]
        except KeyError as error:
            raise QuotientAlgebraClosureError(
                "certificate pivot leaves the admitted workspace"
            ) from error
        coefficients: dict[int, int] = {}
        for monomial, raw_coefficient in row.coefficients:
            if monomial not in admitted:
                raise QuotientAlgebraClosureError(
                    "certificate row leaves the admitted workspace"
                )
            coefficient = _require_plain_int(
                raw_coefficient,
                label="certificate row coefficient",
            )
            if not 1 <= coefficient < FIELD_MODULUS:
                raise QuotientAlgebraError(
                    "certificate row coefficients must be canonical nonzero residues"
                )
            column = column_by_monomial[monomial]
            if column in coefficients:
                raise QuotientAlgebraError(
                    "certificate row repeats a monomial"
                )
            coefficients[column] = coefficient
        if not coefficients or pivot != min(coefficients):
            raise QuotientAlgebraClosureError(
                "certificate pivot is not the row's leading monomial"
            )
        if coefficients[pivot] != 1:
            raise QuotientAlgebraClosureError(
                "certificate pivot coefficient is not one"
            )
        if tuple(sorted(coefficients)) != tuple(
            column_by_monomial[monomial]
            for monomial, _ in row.coefficients
        ):
            raise QuotientAlgebraError(
                "certificate row coefficients are not canonical"
            )
        if pivot in pivots:
            raise QuotientAlgebraError("certificate repeats a pivot")

        provenance: dict[MacaulaySource, int] = {}
        reconstructed: dict[Monomial, int] = {}
        for term in row.provenance:
            if not isinstance(term, MembershipTerm):
                raise QuotientAlgebraError(
                    "certificate provenance term has the wrong type"
                )
            if not 0 <= term.generator_index < len(generators):
                raise QuotientAlgebraError(
                    "certificate provenance generator index is out of range"
                )
            multiplier = _validate_monomial(
                term.multiplier,
                num_variables=generators[0].num_variables,
                degree_limit=degree_limit,
            )
            coefficient = _require_plain_int(
                term.coefficient,
                label="certificate provenance coefficient",
            )
            if not 1 <= coefficient < FIELD_MODULUS:
                raise QuotientAlgebraError(
                    "certificate provenance coefficients must be canonical "
                    "nonzero residues"
                )
            source = MacaulaySource(term.generator_index, multiplier)
            if source in provenance:
                raise QuotientAlgebraError(
                    "certificate provenance repeats a generator multiple"
                )
            provenance[source] = coefficient
            generator = generators[term.generator_index]
            for generator_monomial, generator_coefficient in generator.terms:
                product = _multiply_monomials(
                    generator_monomial,
                    multiplier,
                    degree_limit=degree_limit,
                )
                if product not in admitted:
                    raise QuotientAlgebraClosureError(
                        "certificate provenance leaves the admitted workspace"
                    )
                value = (
                    reconstructed.get(product, 0)
                    + coefficient * generator_coefficient
                ) % FIELD_MODULUS
                if value:
                    reconstructed[product] = value
                else:
                    reconstructed.pop(product, None)
        certified = {
            monomials[column]: coefficient
            for column, coefficient in coefficients.items()
        }
        if reconstructed != certified:
            raise QuotientAlgebraClosureError(
                "certificate row provenance does not reconstruct the row"
            )
        pivots[pivot] = _TrackedRow(
            coefficients=coefficients,
            provenance=provenance,
        )

    for pivot, row in pivots.items():
        for other_pivot, other_row in pivots.items():
            expected = 1 if pivot == other_pivot else 0
            if row.coefficients.get(other_pivot, 0) != expected:
                raise QuotientAlgebraClosureError(
                    "certificate rows are not reduced row echelon form"
                )
    return dict(sorted(pivots.items()))


def _verify_certificate_spans_macaulay_rows(
    *,
    generators: tuple[SparsePolynomial, ...],
    monomials: tuple[Monomial, ...],
    degree_limit: int,
    pivots: Mapping[int, _TrackedRow],
) -> None:
    column_by_monomial = {
        monomial: column for column, monomial in enumerate(monomials)
    }
    admitted = set(monomials)
    for generator in generators:
        for multiplier in reversed(monomials):
            row: dict[int, int] = {}
            valid = True
            for monomial, coefficient in generator.terms:
                try:
                    product = _multiply_monomials(
                        monomial,
                        multiplier,
                        degree_limit=degree_limit,
                    )
                except QuotientAlgebraLimitError:
                    valid = False
                    break
                if product not in admitted:
                    valid = False
                    break
                row[column_by_monomial[product]] = coefficient
            if not valid or not row:
                continue
            for pivot, tracked in pivots.items():
                factor = row.get(pivot, 0)
                if factor:
                    _subtract_scaled_sparse(
                        row,
                        tracked.coefficients,
                        factor,
                    )
            if row:
                raise QuotientAlgebraClosureError(
                    "certificate RREF does not span every admitted Macaulay row"
                )


def _certificate_normal_form(
    coefficients: Mapping[int, int],
    pivots: Mapping[int, _TrackedRow],
) -> dict[int, int]:
    remainder = dict(coefficients)
    for pivot, tracked in pivots.items():
        factor = remainder.get(pivot, 0)
        if factor:
            _subtract_scaled_sparse(
                remainder,
                tracked.coefficients,
                factor,
            )
    return remainder


def _verify_certificate_quotient_structure(
    *,
    num_variables: int,
    degree_limit: int,
    monomials: tuple[Monomial, ...],
    pivots: Mapping[int, _TrackedRow],
    require_boolean_schema: bool,
) -> bool:
    column_by_monomial = {
        monomial: column for column, monomial in enumerate(monomials)
    }
    constant_column = column_by_monomial[_zero_monomial(num_variables)]
    if constant_column in pivots:
        return False
    basis_columns = tuple(
        column for column in range(len(monomials)) if column not in pivots
    )
    basis = {monomials[column] for column in basis_columns}
    if _zero_monomial(num_variables) not in basis:
        raise QuotientAlgebraClosureError(
            "certificate quotient basis is not connected to one"
        )
    for monomial in basis:
        for variable, exponent in enumerate(monomial):
            if exponent == 0:
                continue
            divisor = list(monomial)
            divisor[variable] -= 1
            if tuple(divisor) not in basis:
                raise QuotientAlgebraClosureError(
                    "certificate quotient basis is not divisor-closed"
                )
    basis_index = {
        column: index for index, column in enumerate(basis_columns)
    }
    operators: list[tuple[tuple[tuple[int, int], ...], ...]] = []
    for variable in range(num_variables):
        coordinate = _coordinate_monomial(num_variables, variable)
        columns = []
        for basis_column in basis_columns:
            try:
                product = _multiply_monomials(
                    monomials[basis_column],
                    coordinate,
                    degree_limit=degree_limit,
                )
            except QuotientAlgebraLimitError as error:
                raise QuotientAlgebraClosureError(
                    "certificate quotient border exceeds the degree limit"
                ) from error
            product_column = column_by_monomial.get(product)
            if product_column is None:
                raise QuotientAlgebraClosureError(
                    "certificate quotient border leaves admitted monomials"
                )
            remainder = _certificate_normal_form(
                {product_column: 1},
                pivots,
            )
            if any(column not in basis_index for column in remainder):
                raise QuotientAlgebraClosureError(
                    "certificate normal form retains a pivot"
                )
            columns.append(
                tuple(
                    sorted(
                        (basis_index[column], coefficient)
                        for column, coefficient in remainder.items()
                    )
                )
            )
        operators.append(tuple(columns))
    frozen = tuple(operators)
    for left in range(num_variables):
        for right in range(left, num_variables):
            for basis_coordinate in range(len(basis_columns)):
                unit = {basis_coordinate: 1}
                left_right = QuotientAlgebra._apply_sparse_operator(
                    frozen[right],
                    QuotientAlgebra._apply_sparse_operator(
                        frozen[left],
                        unit,
                    ),
                )
                right_left = QuotientAlgebra._apply_sparse_operator(
                    frozen[left],
                    QuotientAlgebra._apply_sparse_operator(
                        frozen[right],
                        unit,
                    ),
                )
                if left_right != right_left:
                    raise QuotientAlgebraClosureError(
                        "certificate multiplication operators do not commute"
                    )
    boolean_verified = True
    for variable, operator in enumerate(frozen):
        for basis_coordinate in range(len(basis_columns)):
            unit = {basis_coordinate: 1}
            once = QuotientAlgebra._apply_sparse_operator(operator, unit)
            twice = QuotientAlgebra._apply_sparse_operator(operator, once)
            if twice != once:
                boolean_verified = False
                break
        if not boolean_verified:
            break
    if require_boolean_schema and not boolean_verified:
        raise QuotientAlgebraClosureError(
            "certificate does not prove Boolean operator idempotence"
        )
    return boolean_verified


def verify_quotient_algebra_certificate(
    certificate: QuotientAlgebraCertificate,
    expected_generators: Iterable[SparsePolynomial],
) -> QuotientCertificateVerification:
    """Independently replay a portable bounded quotient certificate.

    The verifier does not trust the compiler's rank, basis, normal forms, or
    digest. It reconstructs every certified row from source generators, proves
    that the rows span all admitted Macaulay multiples, checks complete
    prolongation stability, and replays the quotient structural identities.
    """

    if not isinstance(certificate, QuotientAlgebraCertificate):
        raise QuotientAlgebraError("quotient certificate has the wrong type")
    if certificate.schema != QUOTIENT_CERTIFICATE_SCHEMA:
        raise QuotientAlgebraError("unexpected quotient certificate schema")
    if certificate.field_modulus != FIELD_MODULUS:
        raise QuotientAlgebraError("certificate field modulus differs")
    width = _validate_variable_count(certificate.variable_count)
    degree_limit = _validate_degree_limit(certificate.degree_limit)
    if certificate.prolongation_degree != degree_limit + 1:
        raise QuotientAlgebraClosureError(
            "certificate prolongation degree is not exactly one higher"
        )
    if not isinstance(certificate.require_boolean_schema, bool):
        raise QuotientAlgebraError(
            "certificate Boolean requirement has the wrong type"
        )
    generators = tuple(
        SparsePolynomial(width, generator) for generator in certificate.generators
    )
    canonical_expected = _canonical_generators(width, expected_generators)
    if generators != canonical_expected:
        raise QuotientAlgebraClosureError(
            "certificate generators differ from the independently supplied source"
        )
    monomials = _prepare_admitted_monomials(
        width,
        degree_limit,
        certificate.admitted_monomials,
    )
    if monomials != certificate.admitted_monomials:
        raise QuotientAlgebraError(
            "certificate admitted monomials are not canonical"
        )
    expected_prolongation = _enumerate_monomials(
        width,
        certificate.prolongation_degree,
    )
    if certificate.prolongation_monomials != expected_prolongation:
        raise QuotientAlgebraClosureError(
            "certificate does not contain the complete prolongation workspace"
        )
    main_pivots = _certificate_rows_to_pivots(
        certificate.rows,
        monomials=monomials,
        generators=generators,
        degree_limit=degree_limit,
    )
    prolongation_pivots = _certificate_rows_to_pivots(
        certificate.prolongation_rows,
        monomials=expected_prolongation,
        generators=generators,
        degree_limit=certificate.prolongation_degree,
    )
    _verify_certificate_spans_macaulay_rows(
        generators=generators,
        monomials=monomials,
        degree_limit=degree_limit,
        pivots=main_pivots,
    )
    _verify_certificate_spans_macaulay_rows(
        generators=generators,
        monomials=expected_prolongation,
        degree_limit=certificate.prolongation_degree,
        pivots=prolongation_pivots,
    )
    quotient_dimension = len(monomials) - len(main_pivots)
    prolonged_dimension = len(expected_prolongation) - len(prolongation_pivots)
    if quotient_dimension != prolonged_dimension:
        raise QuotientAlgebraClosureError(
            "certificate quotient dimension is not stable under prolongation"
        )
    if quotient_dimension > MAX_QUOTIENT_DIMENSION:
        raise QuotientAlgebraLimitError(
            "certificate quotient dimension exceeds the fixed limit"
        )
    boolean_schema_verified = _verify_certificate_quotient_structure(
        num_variables=width,
        degree_limit=degree_limit,
        monomials=monomials,
        pivots=main_pivots,
        require_boolean_schema=certificate.require_boolean_schema,
    )
    constant_column = monomials.index(_zero_monomial(width))
    status = "inconsistent" if constant_column in main_pivots else "ok"
    basis_columns = tuple(
        column for column in range(len(monomials)) if column not in main_pivots
    )
    payload = _semantic_digest_payload(
        status=status,
        num_variables=width,
        degree_limit=degree_limit,
        monomials=monomials,
        pivots=main_pivots,
        basis_columns=basis_columns,
        prolongation_degree=certificate.prolongation_degree,
        prolongation_monomial_count=len(expected_prolongation),
        prolongation_rank=len(prolongation_pivots),
        boolean_schema_verified=boolean_schema_verified,
    )
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    if digest != certificate.quotient_digest:
        raise QuotientAlgebraClosureError(
            "certificate quotient digest does not match independent replay"
        )
    gates = (
        (
            "boolean_and_multiplication_structure",
            boolean_schema_verified or status == "inconsistent",
        ),
        ("complete_prolongation", True),
        ("generator_binding", True),
        ("main_macaulay_span", True),
        ("prolongation_macaulay_span", True),
        ("quotient_digest", True),
        ("rref_provenance", True),
        ("stable_dimension", True),
    )
    return QuotientCertificateVerification(
        schema=QUOTIENT_VERIFICATION_SCHEMA,
        certificate_sha256=certificate.certificate_sha256,
        quotient_digest=digest,
        main_rank=len(main_pivots),
        prolongation_rank=len(prolongation_pivots),
        quotient_dimension=quotient_dimension,
        gates=gates,
    )


def certify_polynomial_consequence(
    num_variables: int,
    generators: Iterable[SparsePolynomial],
    query: SparsePolynomial,
    *,
    allowed_values: Iterable[int],
    degree_limit: int = MAX_DEGREE,
    admitted_monomials: Iterable[Monomial] | None = None,
    require_boolean_schema: bool = True,
) -> SSQACConsequenceOutcome:
    """Compile and classify one consequence without collapsing failure modes."""

    frozen_generators = tuple(generators)
    frozen_allowed_values = tuple(allowed_values)
    if require_boolean_schema is not True:
        raise QuotientAlgebraError(
            "public SSQAC consequence certification requires Boolean schema"
        )
    try:
        algebra = compile_quotient_algebra(
            num_variables,
            frozen_generators,
            degree_limit=degree_limit,
            admitted_monomials=admitted_monomials,
            require_boolean_schema=require_boolean_schema,
        )
        certificate = algebra.export_certificate()
        verification = verify_quotient_algebra_certificate(
            certificate,
            frozen_generators,
        )
        if not verification.passed:
            raise QuotientAlgebraClosureError(
                "portable quotient verification did not pass every gate"
            )
        if not algebra.is_consistent:
            return SSQACConsequenceOutcome(
                schema=SSQAC_OUTCOME_SCHEMA,
                status=STATUS_UNSAT,
                value=None,
                query=query,
                allowed_values=frozen_allowed_values,
                quotient_receipt=algebra.receipt,
                consequence=None,
                certificate=certificate,
                diagnostic=None,
            )
        consequence = algebra.decide_polynomial(
            query,
            allowed_values=frozen_allowed_values,
        )
        if consequence.status == "forced":
            status = STATUS_CERTIFIED
            value = consequence.value
        elif consequence.status == "ambiguous":
            status = STATUS_AMBIGUOUS
            value = None
        elif consequence.status == "out_of_domain":
            return SSQACConsequenceOutcome(
                schema=SSQAC_OUTCOME_SCHEMA,
                status=STATUS_INCOMPLETE,
                value=None,
                query=query,
                allowed_values=frozen_allowed_values,
                quotient_receipt=algebra.receipt,
                consequence=consequence,
                certificate=certificate,
                diagnostic="query is not confined to the declared value domain",
            )
        else:
            raise QuotientAlgebraError(
                f"unexpected consequence status {consequence.status!r}"
            )
        return SSQACConsequenceOutcome(
            schema=SSQAC_OUTCOME_SCHEMA,
            status=status,
            value=value,
            query=query,
            allowed_values=frozen_allowed_values,
            quotient_receipt=algebra.receipt,
            consequence=consequence,
            certificate=certificate,
            diagnostic=None,
        )
    except QuotientAlgebraLimitError as error:
        return SSQACConsequenceOutcome(
            schema=SSQAC_OUTCOME_SCHEMA,
            status=STATUS_RESOURCE_OVERFLOW,
            value=None,
            query=query,
            allowed_values=frozen_allowed_values,
            quotient_receipt=None,
            consequence=None,
            certificate=None,
            diagnostic=str(error),
        )
    except QuotientAlgebraClosureError as error:
        return SSQACConsequenceOutcome(
            schema=SSQAC_OUTCOME_SCHEMA,
            status=STATUS_INCOMPLETE,
            value=None,
            query=query,
            allowed_values=frozen_allowed_values,
            quotient_receipt=None,
            consequence=None,
            certificate=None,
            diagnostic=str(error),
        )


def verify_ssqac_consequence_outcome(
    outcome: SSQACConsequenceOutcome,
    expected_generators: Iterable[SparsePolynomial],
    expected_query: SparsePolynomial,
    *,
    expected_allowed_values: Iterable[int],
) -> SSQACOutcomeVerification:
    """Bind and replay the query, value, evidence, and quotient certificate."""

    if not isinstance(outcome, SSQACConsequenceOutcome):
        raise QuotientAlgebraError("SSQAC outcome has the wrong type")
    if outcome.schema != SSQAC_OUTCOME_SCHEMA:
        raise QuotientAlgebraError("unexpected SSQAC outcome schema")
    if outcome.query != expected_query:
        raise QuotientAlgebraClosureError(
            "outcome query differs from the independently supplied query"
        )
    allowed = tuple(expected_allowed_values)
    if outcome.allowed_values != allowed:
        raise QuotientAlgebraClosureError(
            "outcome value domain differs from the independently supplied domain"
        )
    generators = tuple(expected_generators)
    reproduced = certify_polynomial_consequence(
        expected_query.num_variables,
        generators,
        expected_query,
        allowed_values=allowed,
        require_boolean_schema=True,
    )
    if outcome.status != reproduced.status:
        raise QuotientAlgebraClosureError(
            "outcome status differs from independent replay"
        )
    if outcome.value != reproduced.value:
        raise QuotientAlgebraClosureError(
            "outcome value differs from independent replay"
        )
    if outcome.quotient_receipt != reproduced.quotient_receipt:
        raise QuotientAlgebraClosureError(
            "outcome quotient receipt differs from independent replay"
        )
    if outcome.consequence != reproduced.consequence:
        raise QuotientAlgebraClosureError(
            "outcome consequence evidence differs from independent replay"
        )
    if (
        outcome.certificate is None
        or reproduced.certificate is None
    ):
        if outcome.certificate != reproduced.certificate:
            raise QuotientAlgebraClosureError(
                "outcome certificate presence differs from independent replay"
            )
    else:
        verification = verify_quotient_algebra_certificate(
            outcome.certificate,
            generators,
        )
        if not verification.passed:
            raise QuotientAlgebraClosureError(
                "outcome quotient certificate failed replay"
            )
        if (
            outcome.certificate.canonical_bytes()
            != reproduced.certificate.canonical_bytes()
        ):
            raise QuotientAlgebraClosureError(
                "outcome quotient certificate differs from independent replay"
            )
    if outcome.diagnostic != reproduced.diagnostic:
        raise QuotientAlgebraClosureError(
            "outcome diagnostic differs from independent replay"
        )
    gates = (
        ("allowed_value_domain_bound", True),
        ("claimed_status_bound", True),
        ("claimed_value_bound", True),
        ("consequence_evidence_bound", True),
        ("query_bound", True),
        ("quotient_certificate_bound", True),
        ("source_generators_bound", True),
    )
    return SSQACOutcomeVerification(
        schema=SSQAC_OUTCOME_VERIFICATION_SCHEMA,
        outcome_sha256=outcome.outcome_sha256,
        status=outcome.status,
        gates=gates,
    )


def recode_generators(
    generators: Iterable[SparsePolynomial],
    permutation: Sequence[int],
) -> tuple[SparsePolynomial, ...]:
    """Apply one variable recoding to an entire generator family."""

    return tuple(generator.recode_variables(permutation) for generator in generators)
