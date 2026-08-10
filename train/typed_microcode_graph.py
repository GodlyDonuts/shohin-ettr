"""Typed result-free computation graphs for natural arithmetic programs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Sequence

from learned_arithmetic_microcode import (
    OPCODE_PERMUTATION,
    DigitRational,
    LearnedArithmeticError,
    LearnedDigitMicrocode,
)
from natural_microcode_program import (
    RegisterProgram,
    _digit_fraction,
    parse_fraction,
)

SCHEMA = "shohin-typed-microcode-graph-v1"
SOURCE = "SOURCE"
STATE = "STATE"
LITERAL = "LITERAL"
OPERATIONS = ("ADD", "SUB", "MUL", "DIV", "NEG", "COPY")
ACTION_TO_OPERATION = {
    "APPLY_ADD": "ADD",
    "APPLY_SUB": "SUB",
    "APPLY_MUL": "MUL",
    "APPLY_DIV": "DIV",
    "NEGATE": "NEG",
}
OPERATION_TO_ACTION = {value: key for key, value in ACTION_TO_OPERATION.items()}
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])\$?[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?|\.\d+)(?:/\d+)?%?"
)


class TypedMicrocodeGraphError(ValueError):
    """A program cannot be represented by the typed graph contract."""


@dataclass(frozen=True, slots=True)
class NumberSpan:
    start: int
    end: int
    surface: str
    value: Fraction


@dataclass(frozen=True, slots=True)
class Operand:
    kind: str
    indices: tuple[int, ...] = ()
    literal: Fraction | None = None


@dataclass(frozen=True, slots=True)
class Instruction:
    operation: str
    left: Operand
    right: Operand | None


@dataclass(frozen=True, slots=True)
class TypedMicrocodeGraph:
    source: str
    number_spans: tuple[NumberSpan, ...]
    instructions: tuple[Instruction, ...]
    final: Operand


def source_fraction(surface: str) -> Fraction:
    normalized = surface.replace("$", "").replace(",", "")
    percentage = normalized.endswith("%")
    if percentage:
        normalized = normalized[:-1]
    try:
        value = Fraction(normalized)
    except (ValueError, ZeroDivisionError) as error:
        raise TypedMicrocodeGraphError("source numeric surface differs") from error
    return value / 100 if percentage else value


def number_spans(source: str) -> tuple[NumberSpan, ...]:
    return tuple(
        NumberSpan(
            match.start(), match.end(), match.group(), source_fraction(match.group())
        )
        for match in NUMBER_RE.finditer(source)
    )


def _source_or_literal(value: Fraction, spans: Sequence[NumberSpan]) -> Operand:
    owners = tuple(index for index, span in enumerate(spans) if span.value == value)
    if owners:
        return Operand(SOURCE, owners)
    return Operand(LITERAL, literal=value)


def compile_typed_graph(source: str, program: RegisterProgram) -> TypedMicrocodeGraph:
    spans = number_spans(source)
    instructions: list[Instruction] = []
    registers: list[Operand] = []
    for record in program.records:
        stack: list[Operand] = []
        for action in record:
            name = action.get("action")
            if name == "PUSH":
                stack.append(
                    _source_or_literal(parse_fraction(str(action["surface"])), spans)
                )
            elif name == "LOAD":
                register = action.get("register")
                if type(register) is not int or not 0 <= register < len(registers):
                    raise TypedMicrocodeGraphError("register owner is not causal")
                stack.append(registers[register])
            elif name == "NEGATE":
                if not stack:
                    raise TypedMicrocodeGraphError("NEGATE underflow")
                left = stack.pop()
                instructions.append(Instruction("NEG", left, None))
                stack.append(Operand(STATE, (len(instructions) - 1,)))
            elif isinstance(name, str) and name.startswith("APPLY_"):
                if len(stack) < 2 or name not in ACTION_TO_OPERATION:
                    raise TypedMicrocodeGraphError("APPLY differs")
                right, left = stack.pop(), stack.pop()
                instructions.append(Instruction(ACTION_TO_OPERATION[name], left, right))
                stack.append(Operand(STATE, (len(instructions) - 1,)))
            else:
                raise TypedMicrocodeGraphError("program action differs")
        if len(stack) != 1:
            raise TypedMicrocodeGraphError("record stack differs")
        registers.append(stack[0])
    if not registers or not 0 <= program.commit < len(registers):
        raise TypedMicrocodeGraphError("commit differs")
    final = registers[program.commit]
    if final.kind != STATE:
        instructions.append(Instruction("COPY", final, None))
        final = Operand(STATE, (len(instructions) - 1,))
    return TypedMicrocodeGraph(source, spans, tuple(instructions), final)


def _resolve_fraction(
    operand: Operand,
    graph: TypedMicrocodeGraph,
    states: Sequence[Fraction],
) -> Fraction:
    if operand.kind == SOURCE:
        if not operand.indices:
            raise TypedMicrocodeGraphError("source owner is empty")
        values = {graph.number_spans[index].value for index in operand.indices}
        if len(values) != 1:
            raise TypedMicrocodeGraphError("source owners disagree")
        return values.pop()
    if operand.kind == STATE:
        if len(operand.indices) != 1 or not 0 <= operand.indices[0] < len(states):
            raise TypedMicrocodeGraphError("state owner is not causal")
        return states[operand.indices[0]]
    if operand.kind == LITERAL and operand.literal is not None:
        return operand.literal
    raise TypedMicrocodeGraphError("operand differs")


def execute_fraction(graph: TypedMicrocodeGraph) -> Fraction:
    states: list[Fraction] = []
    for instruction in graph.instructions:
        left = _resolve_fraction(instruction.left, graph, states)
        if instruction.operation == "NEG":
            value = -left
        elif instruction.operation == "COPY":
            value = left
        else:
            if instruction.right is None:
                raise TypedMicrocodeGraphError("binary right operand is absent")
            right = _resolve_fraction(instruction.right, graph, states)
            if instruction.operation == "ADD":
                value = left + right
            elif instruction.operation == "SUB":
                value = left - right
            elif instruction.operation == "MUL":
                value = left * right
            elif instruction.operation == "DIV" and right:
                value = left / right
            else:
                raise TypedMicrocodeGraphError("operation differs")
        states.append(value)
    return _resolve_fraction(graph.final, graph, states)


def _resolve_digit(
    operand: Operand,
    graph: TypedMicrocodeGraph,
    states: Sequence[DigitRational],
) -> DigitRational:
    if operand.kind == SOURCE:
        if not operand.indices:
            raise LearnedArithmeticError("source owner is empty")
        surfaces = {str(graph.number_spans[index].value) for index in operand.indices}
        if len(surfaces) != 1:
            raise LearnedArithmeticError("source owners disagree")
        return _digit_fraction(surfaces.pop())
    if operand.kind == STATE:
        if len(operand.indices) != 1 or not 0 <= operand.indices[0] < len(states):
            raise LearnedArithmeticError("state owner is not causal")
        return states[operand.indices[0]]
    if operand.kind == LITERAL and operand.literal is not None:
        return _digit_fraction(str(operand.literal))
    raise LearnedArithmeticError("operand differs")


def execute_learned(
    microcode: LearnedDigitMicrocode,
    graph: TypedMicrocodeGraph,
    *,
    intervention: str = "normal",
) -> DigitRational:
    if intervention not in {"normal", "carry_reset", "opcode_permuted"}:
        raise LearnedArithmeticError("intervention differs")
    states: list[DigitRational] = []
    for instruction in graph.instructions:
        left = _resolve_digit(instruction.left, graph, states)
        if instruction.operation == "NEG":
            value = DigitRational(
                not left.negative and left.numerator != (0,),
                left.numerator,
                left.denominator,
            )
        elif instruction.operation == "COPY":
            value = left
        else:
            if instruction.right is None:
                raise LearnedArithmeticError("binary right operand is absent")
            right = _resolve_digit(instruction.right, graph, states)
            action = OPERATION_TO_ACTION[instruction.operation]
            if intervention == "opcode_permuted":
                action = OPCODE_PERMUTATION[action]
            value = microcode.apply(
                action,
                left,
                right,
                reset_carry=intervention == "carry_reset",
            )
        states.append(value)
    return _resolve_digit(graph.final, graph, states)


def operand_count(graph: TypedMicrocodeGraph, kind: str) -> int:
    operands = [
        operand
        for instruction in graph.instructions
        for operand in (instruction.left, instruction.right)
        if operand is not None
    ]
    return sum(operand.kind == kind for operand in operands)
