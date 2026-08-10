"""Lower explicit model-owned draft transactions into typed microcode."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re

from typed_microcode_graph import (
    LITERAL,
    SOURCE,
    STATE,
    Instruction,
    Operand,
    TypedMicrocodeGraph,
    number_spans,
    source_fraction,
)


SCHEMA = "shohin-dtc1-draft-transaction-compiler-v1"
TRANSACTION_RE = re.compile(r"<<([^<>]+)>>")
NUMERIC_ATOM_RE = re.compile(
    r"\$?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)%?"
)
OPERATIONS = {
    ast.Add: "ADD",
    ast.Sub: "SUB",
    ast.Mult: "MUL",
    ast.Div: "DIV",
}


class DraftTransactionError(ValueError):
    """A draft transaction violates the frozen DTC1 grammar."""


@dataclass(frozen=True, slots=True)
class CompilationReceipt:
    annotations: int
    accepted: int
    rejected: tuple[str, ...]
    state_reads: int
    source_reads: int
    literal_reads: int


def _normalize_numeric_atom(match: re.Match[str]) -> str:
    atom = match.group(0).replace("$", "").replace(",", "")
    if atom.endswith("%"):
        try:
            return format(Decimal(atom[:-1]) / Decimal(100), "f")
        except InvalidOperation as error:
            raise DraftTransactionError("percentage atom differs") from error
    return atom


def normalize_expression(expression: str) -> str:
    expression = expression.strip().replace("×", "*").replace("÷", "/")
    return NUMERIC_ATOM_RE.sub(_normalize_numeric_atom, expression)


def parse_claimed_result(surface: str) -> Fraction:
    value = surface.strip()
    if not value or any(character in value for character in "<>="):
        raise DraftTransactionError("claimed result differs")
    try:
        return source_fraction(value)
    except Exception as error:
        raise DraftTransactionError("claimed result differs") from error


def _source_or_literal(
    value: Fraction,
    spans,
    aliases: dict[Fraction, int],
) -> Operand:
    if value in aliases:
        return Operand(STATE, (aliases[value],))
    owners = tuple(index for index, span in enumerate(spans) if span.value == value)
    if owners:
        return Operand(SOURCE, owners)
    return Operand(LITERAL, literal=value)


def _constant_fraction(node: ast.Constant, expression: str) -> Fraction:
    if type(node.value) not in (int, float):
        raise DraftTransactionError("expression constant differs")
    segment = ast.get_source_segment(expression, node)
    if segment is None:
        raise DraftTransactionError("expression constant span differs")
    try:
        return Fraction(segment)
    except (ValueError, ZeroDivisionError) as error:
        raise DraftTransactionError("expression constant differs") from error


def _compile_expression(
    expression: str,
    spans,
    aliases: dict[Fraction, int],
    instructions: list[Instruction],
) -> Operand:
    normalized = normalize_expression(expression)
    try:
        tree = ast.parse(normalized, mode="eval").body
    except SyntaxError as error:
        raise DraftTransactionError("expression syntax differs") from error

    def visit(node: ast.AST) -> Operand:
        if isinstance(node, ast.Constant):
            return _source_or_literal(
                _constant_fraction(node, normalized), spans, aliases
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            instructions.append(Instruction("NEG", operand, None))
            return Operand(STATE, (len(instructions) - 1,))
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATIONS:
            left = visit(node.left)
            right = visit(node.right)
            instructions.append(Instruction(OPERATIONS[type(node.op)], left, right))
            return Operand(STATE, (len(instructions) - 1,))
        raise DraftTransactionError("expression AST differs")

    return visit(tree)


def compile_draft_transactions(
    source: str, draft: str
) -> tuple[TypedMicrocodeGraph, CompilationReceipt]:
    spans = number_spans(source)
    instructions: list[Instruction] = []
    aliases: dict[Fraction, int] = {}
    rejected: list[str] = []
    accepted = 0
    final: Operand | None = None
    matches = list(TRANSACTION_RE.finditer(draft))
    for match in matches:
        transaction = match.group(1)
        if "=" not in transaction:
            rejected.append("transaction delimiter differs")
            continue
        expression, claimed_surface = transaction.rsplit("=", 1)
        checkpoint = len(instructions)
        try:
            claimed = parse_claimed_result(claimed_surface)
            candidate = _compile_expression(
                expression, spans, aliases, instructions
            )
            if candidate.kind != STATE:
                instructions.append(Instruction("COPY", candidate, None))
                candidate = Operand(STATE, (len(instructions) - 1,))
        except DraftTransactionError as error:
            del instructions[checkpoint:]
            rejected.append(str(error))
            continue
        aliases[claimed] = candidate.indices[0]
        final = candidate
        accepted += 1
    if final is None:
        raise DraftTransactionError("draft has no accepted transaction")
    graph = TypedMicrocodeGraph(source, spans, tuple(instructions), final)
    operands = [
        operand
        for instruction in graph.instructions
        for operand in (instruction.left, instruction.right)
        if operand is not None
    ]
    counts = Counter(operand.kind for operand in operands)
    return graph, CompilationReceipt(
        annotations=len(matches),
        accepted=accepted,
        rejected=tuple(rejected),
        state_reads=counts[STATE],
        source_reads=counts[SOURCE],
        literal_reads=counts[LITERAL],
    )


def reset_state_reads(graph: TypedMicrocodeGraph) -> TypedMicrocodeGraph:
    def reset(operand: Operand | None) -> Operand | None:
        if operand is not None and operand.kind == STATE:
            return Operand(LITERAL, literal=Fraction(0))
        return operand

    return TypedMicrocodeGraph(
        graph.source,
        graph.number_spans,
        tuple(
            Instruction(
                instruction.operation,
                reset(instruction.left),
                reset(instruction.right),
            )
            for instruction in graph.instructions
        ),
        graph.final,
    )
