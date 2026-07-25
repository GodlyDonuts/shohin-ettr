"""Primitive finite-field row machine for learned SSQAC controllers.

The machine intentionally does not implement Gaussian elimination, pivot
selection, branching, closure, or automatic halting. A controller must emit
every row and register operation explicitly. The VM applies those operations
over F_257 while carrying an exact provenance matrix through the same
invertible row operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable, Sequence


FIELD_MODULUS = 257
MACHINE_SCHEMA = "ssqac_primitive_field_row_machine_v1"
PROGRAM_RECEIPT_SCHEMA = "ssqac_primitive_field_row_program_receipt_v1"
MAX_ROWS = 512
MAX_COLUMNS = 512
MAX_REGISTERS = 16
MAX_INSTRUCTIONS = 1_000_000

OP_LOAD = "LOAD"
OP_INV = "INV"
OP_NEG = "NEG"
OP_SCALE = "SCALE"
OP_AXPY = "AXPY"
OP_SWAP = "SWAP"
OP_HALT = "HALT"
OPCODES = (OP_LOAD, OP_INV, OP_NEG, OP_SCALE, OP_AXPY, OP_SWAP, OP_HALT)


class AlgebraMachineError(ValueError):
    """The primitive machine contract failed closed."""


def _plain_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlgebraMachineError(f"{label} must be an integer")
    return value


def _canonical_matrix(
    rows: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    frozen = tuple(
        tuple(
            _plain_int(value, label="matrix coefficient") % FIELD_MODULUS
            for value in row
        )
        for row in rows
    )
    if not frozen:
        raise AlgebraMachineError("matrix must contain at least one row")
    width = len(frozen[0])
    if not 1 <= len(frozen) <= MAX_ROWS:
        raise AlgebraMachineError(f"matrix row count exceeds {MAX_ROWS}")
    if not 1 <= width <= MAX_COLUMNS:
        raise AlgebraMachineError(f"matrix column count exceeds {MAX_COLUMNS}")
    if any(len(row) != width for row in frozen):
        raise AlgebraMachineError("matrix rows have inconsistent widths")
    return frozen


@dataclass(frozen=True, slots=True)
class AlgebraInstruction:
    """One branch-free primitive instruction.

    Operands are interpreted by opcode:

    * ``LOAD a b c``: register ``c`` <- matrix[a,b]
    * ``INV a b``: register ``b`` <- inverse(register[a])
    * ``NEG a b``: register ``b`` <- -register[a]
    * ``SCALE a b``: row ``a`` <- register[b] * row ``a``
    * ``AXPY a b c``: row ``a`` <- row ``a`` + register[c] * row ``b``
    * ``SWAP a b``: exchange rows ``a`` and ``b``
    * ``HALT``: stop; later instructions are forbidden
    """

    opcode: str
    a: int = 0
    b: int = 0
    c: int = 0

    def __post_init__(self) -> None:
        if self.opcode not in OPCODES:
            raise AlgebraMachineError(f"unknown opcode {self.opcode!r}")
        for label, value in (("a", self.a), ("b", self.b), ("c", self.c)):
            _plain_int(value, label=f"instruction operand {label}")

    def canonical_data(self) -> list[object]:
        return [self.opcode, self.a, self.b, self.c]


@dataclass(frozen=True, slots=True)
class AlgebraMachineState:
    schema: str
    rows: tuple[tuple[int, ...], ...]
    provenance: tuple[tuple[int, ...], ...]
    registers: tuple[int, ...]
    halted: bool
    executed_instructions: int
    opcode_counts: tuple[tuple[str, int], ...]
    trace_sha256: str


@dataclass(frozen=True, slots=True)
class ReductionProgramReceipt:
    schema: str
    trace_sha256: str
    input_sha256: str
    output_sha256: str
    row_count: int
    column_count: int
    rank: int
    executed_instructions: int
    field_multiply_adds: int
    gates: tuple[tuple[str, bool], ...]

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(passed for _, passed in self.gates)


def _matrix_bytes(matrix: Sequence[Sequence[int]]) -> bytes:
    return json.dumps(
        matrix,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def execute_program(
    input_rows: Iterable[Iterable[int]],
    instructions: Iterable[AlgebraInstruction],
    *,
    register_count: int = 8,
    maximum_instructions: int = MAX_INSTRUCTIONS,
) -> AlgebraMachineState:
    """Execute a controller program without data-dependent host control flow."""

    matrix = _canonical_matrix(input_rows)
    registers_count = _plain_int(register_count, label="register count")
    instruction_limit = _plain_int(
        maximum_instructions,
        label="instruction limit",
    )
    if not 1 <= registers_count <= MAX_REGISTERS:
        raise AlgebraMachineError(
            f"register count must be in [1, {MAX_REGISTERS}]"
        )
    if not 1 <= instruction_limit <= MAX_INSTRUCTIONS:
        raise AlgebraMachineError(
            f"instruction limit must be in [1, {MAX_INSTRUCTIONS}]"
        )
    program = tuple(instructions)
    if len(program) > instruction_limit:
        raise AlgebraMachineError("controller program exceeds instruction limit")
    if any(not isinstance(instruction, AlgebraInstruction) for instruction in program):
        raise AlgebraMachineError("program contains a non-instruction value")

    row_count = len(matrix)
    column_count = len(matrix[0])
    rows = [list(row) for row in matrix]
    provenance = [
        [1 if row == source else 0 for source in range(row_count)]
        for row in range(row_count)
    ]
    registers = [0] * registers_count
    halted = False
    counts = {opcode: 0 for opcode in OPCODES}
    trace_hasher = sha256()
    trace_hasher.update(MACHINE_SCHEMA.encode("ascii"))
    trace_hasher.update(b"\0")
    trace_hasher.update(_matrix_bytes(matrix))

    def require_row(index: int) -> int:
        value = _plain_int(index, label="row index")
        if not 0 <= value < row_count:
            raise AlgebraMachineError("row index is out of range")
        return value

    def require_column(index: int) -> int:
        value = _plain_int(index, label="column index")
        if not 0 <= value < column_count:
            raise AlgebraMachineError("column index is out of range")
        return value

    def require_register(index: int) -> int:
        value = _plain_int(index, label="register index")
        if not 0 <= value < registers_count:
            raise AlgebraMachineError("register index is out of range")
        return value

    for instruction_index, instruction in enumerate(program):
        if halted:
            raise AlgebraMachineError("program contains instructions after HALT")
        counts[instruction.opcode] += 1
        trace_hasher.update(
            json.dumps(
                instruction.canonical_data(),
                separators=(",", ":"),
            ).encode("ascii")
        )
        trace_hasher.update(b"\n")
        if instruction.opcode == OP_LOAD:
            row = require_row(instruction.a)
            column = require_column(instruction.b)
            register = require_register(instruction.c)
            registers[register] = rows[row][column]
        elif instruction.opcode == OP_INV:
            source = require_register(instruction.a)
            destination = require_register(instruction.b)
            value = registers[source]
            if value == 0:
                raise AlgebraMachineError("cannot invert zero")
            registers[destination] = pow(value, -1, FIELD_MODULUS)
        elif instruction.opcode == OP_NEG:
            source = require_register(instruction.a)
            destination = require_register(instruction.b)
            registers[destination] = (-registers[source]) % FIELD_MODULUS
        elif instruction.opcode == OP_SCALE:
            row = require_row(instruction.a)
            register = require_register(instruction.b)
            factor = registers[register]
            if factor == 0:
                raise AlgebraMachineError(
                    "row scaling must remain invertible"
                )
            rows[row] = [
                factor * value % FIELD_MODULUS for value in rows[row]
            ]
            provenance[row] = [
                factor * value % FIELD_MODULUS
                for value in provenance[row]
            ]
        elif instruction.opcode == OP_AXPY:
            destination = require_row(instruction.a)
            source = require_row(instruction.b)
            register = require_register(instruction.c)
            factor = registers[register]
            rows[destination] = [
                (left + factor * right) % FIELD_MODULUS
                for left, right in zip(
                    rows[destination],
                    rows[source],
                    strict=True,
                )
            ]
            provenance[destination] = [
                (left + factor * right) % FIELD_MODULUS
                for left, right in zip(
                    provenance[destination],
                    provenance[source],
                    strict=True,
                )
            ]
        elif instruction.opcode == OP_SWAP:
            left = require_row(instruction.a)
            right = require_row(instruction.b)
            rows[left], rows[right] = rows[right], rows[left]
            provenance[left], provenance[right] = (
                provenance[right],
                provenance[left],
            )
        elif instruction.opcode == OP_HALT:
            halted = True
        else:
            raise AlgebraMachineError("unreachable opcode dispatch")
        trace_hasher.update(str(instruction_index).encode("ascii"))
        trace_hasher.update(b"\0")

    return AlgebraMachineState(
        schema=MACHINE_SCHEMA,
        rows=tuple(tuple(row) for row in rows),
        provenance=tuple(tuple(row) for row in provenance),
        registers=tuple(registers),
        halted=halted,
        executed_instructions=len(program),
        opcode_counts=tuple((opcode, counts[opcode]) for opcode in OPCODES),
        trace_sha256=trace_hasher.hexdigest(),
    )


def _rref_nonzero_rows(
    rows: Sequence[Sequence[int]],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    result = []
    previous_pivot = -1
    for row in rows:
        nonzero = tuple(index for index, value in enumerate(row) if value)
        if not nonzero:
            continue
        pivot = nonzero[0]
        if pivot <= previous_pivot:
            raise AlgebraMachineError("nonzero rows are not in pivot order")
        if row[pivot] != 1:
            raise AlgebraMachineError("pivot coefficient is not one")
        result.append((pivot, tuple(row)))
        previous_pivot = pivot
    for pivot, row in result:
        for other_pivot, other_row in result:
            expected = 1 if pivot == other_pivot else 0
            if row[other_pivot] != expected:
                raise AlgebraMachineError("rows are not reduced echelon form")
    return tuple(result)


def verify_reduction_program(
    input_rows: Iterable[Iterable[int]],
    state: AlgebraMachineState,
) -> ReductionProgramReceipt:
    """Verify a completed controller trace without trusting its schedule."""

    matrix = _canonical_matrix(input_rows)
    if not isinstance(state, AlgebraMachineState):
        raise AlgebraMachineError("machine state has the wrong type")
    if state.schema != MACHINE_SCHEMA:
        raise AlgebraMachineError("machine state schema differs")
    if not state.halted:
        raise AlgebraMachineError("controller did not emit HALT")
    if len(state.rows) != len(matrix):
        raise AlgebraMachineError("output row count differs")
    if any(len(row) != len(matrix[0]) for row in state.rows):
        raise AlgebraMachineError("output column count differs")
    if len(state.provenance) != len(matrix):
        raise AlgebraMachineError("provenance row count differs")
    if any(len(row) != len(matrix) for row in state.provenance):
        raise AlgebraMachineError("provenance column count differs")

    reconstructed = tuple(
        tuple(
            sum(
                coefficient * matrix[source][column]
                for source, coefficient in enumerate(provenance_row)
            )
            % FIELD_MODULUS
            for column in range(len(matrix[0]))
        )
        for provenance_row in state.provenance
    )
    if reconstructed != state.rows:
        raise AlgebraMachineError(
            "provenance does not reconstruct the output rows"
        )
    nonzero = _rref_nonzero_rows(state.rows)
    pivot_rows = tuple(row for _, row in nonzero)
    for input_row in matrix:
        remainder = list(input_row)
        for pivot, pivot_row in nonzero:
            factor = remainder[pivot]
            if factor:
                remainder = [
                    (left - factor * right) % FIELD_MODULUS
                    for left, right in zip(
                        remainder,
                        pivot_row,
                        strict=True,
                    )
                ]
        if any(remainder):
            raise AlgebraMachineError(
                "output rows do not span the input row space"
            )
    output_sha = sha256(_matrix_bytes(state.rows)).hexdigest()
    opcode_counts = dict(state.opcode_counts)
    field_multiply_adds = len(matrix[0]) * (
        opcode_counts.get(OP_SCALE, 0)
        + opcode_counts.get(OP_AXPY, 0)
    )
    gates = (
        ("explicit_halt", True),
        ("input_span", True),
        ("provenance_reconstruction", True),
        ("reduced_row_echelon", True),
    )
    return ReductionProgramReceipt(
        schema=PROGRAM_RECEIPT_SCHEMA,
        trace_sha256=state.trace_sha256,
        input_sha256=sha256(_matrix_bytes(matrix)).hexdigest(),
        output_sha256=output_sha,
        row_count=len(matrix),
        column_count=len(matrix[0]),
        rank=len(pivot_rows),
        executed_instructions=state.executed_instructions,
        field_multiply_adds=field_multiply_adds,
        gates=gates,
    )
