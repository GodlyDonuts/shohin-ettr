"""Result-free register programs executed by learned arithmetic microcode."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Sequence

from learned_arithmetic_microcode import (
    OPCODE_PERMUTATION,
    DigitRational,
    LearnedArithmeticError,
    LearnedDigitMicrocode,
    _surface_digits,
)

SCHEMA = "shohin-natural-microcode-program-v1"
OPEN = "<MICROCODE_V1>"
CLOSE = "</MICROCODE_V1>"
EQUATION_RE = re.compile(r"<<([^<>]+)=([^<>]+)>>")
FINAL_RE = re.compile(r"####\s*([^\n]+)\s*$")
OPCODE = {
    ast.Add: "APPLY_ADD",
    ast.Sub: "APPLY_SUB",
    ast.Mult: "APPLY_MUL",
    ast.Div: "APPLY_DIV",
}
TOKEN_TO_ACTION = {
    "A": "APPLY_ADD",
    "S": "APPLY_SUB",
    "M": "APPLY_MUL",
    "D": "APPLY_DIV",
    "N": "NEGATE",
}
ACTION_TO_TOKEN = {value: key for key, value in TOKEN_TO_ACTION.items()}


class NaturalMicrocodeError(ValueError):
    """A source annotation or serialized microprogram differs."""


@dataclass(frozen=True, slots=True)
class RegisterProgram:
    records: tuple[tuple[dict[str, object], ...], ...]
    commit: int


def canonical_fraction(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def parse_fraction(text: str) -> Fraction:
    normalized = text.strip().replace(",", "").replace("$", "")
    try:
        return Fraction(normalized)
    except (ValueError, ZeroDivisionError) as error:
        raise NaturalMicrocodeError("numeric surface differs") from error


def _normalize_expression(text: str) -> str:
    return text.strip().replace(",", "").replace("$", "").replace("%", "/100")


def _compile_node(
    node: ast.AST, latest_register: dict[Fraction, int]
) -> tuple[list[dict[str, object]], Fraction]:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = Fraction(str(node.value))
        if value in latest_register:
            return [{"action": "LOAD", "register": latest_register[value]}], value
        return [{"action": "PUSH", "surface": canonical_fraction(value)}], value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        actions, value = _compile_node(node.operand, latest_register)
        if isinstance(node.op, ast.USub):
            actions.append({"action": "NEGATE"})
            value = -value
        return actions, value
    if isinstance(node, ast.BinOp) and type(node.op) in OPCODE:
        left_actions, left = _compile_node(node.left, latest_register)
        right_actions, right = _compile_node(node.right, latest_register)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        else:
            if right == 0:
                raise NaturalMicrocodeError("division by zero")
            value = left / right
        return left_actions + right_actions + [{"action": OPCODE[type(node.op)]}], value
    raise NaturalMicrocodeError("expression operation differs")


def compile_gsm8k_answer(answer: str) -> tuple[RegisterProgram, Fraction]:
    equations = EQUATION_RE.findall(answer)
    final_match = FINAL_RE.search(answer)
    if not equations or final_match is None:
        raise NaturalMicrocodeError("answer lacks complete equations or final")
    latest_register: dict[Fraction, int] = {}
    records: list[tuple[dict[str, object], ...]] = []
    for index, (expression, stated_surface) in enumerate(equations):
        try:
            tree = ast.parse(_normalize_expression(expression), mode="eval").body
        except SyntaxError as error:
            raise NaturalMicrocodeError("expression syntax differs") from error
        actions, computed = _compile_node(tree, latest_register)
        stated = parse_fraction(stated_surface)
        if computed != stated:
            raise NaturalMicrocodeError("equation result is false")
        records.append(tuple(actions))
        latest_register[computed] = index
    final = parse_fraction(final_match.group(1))
    if latest_register.get(final) != len(records) - 1:
        raise NaturalMicrocodeError("final answer is not the final register")
    return RegisterProgram(tuple(records), len(records) - 1), final


def render_program(program: RegisterProgram) -> str:
    lines = [OPEN]
    for index, actions in enumerate(program.records):
        tokens = []
        for action in actions:
            name = action.get("action")
            if name == "PUSH":
                tokens.append(f"P:{action['surface']}")
            elif name == "LOAD":
                tokens.append(f"L:{action['register']}")
            elif name in ACTION_TO_TOKEN:
                tokens.append(ACTION_TO_TOKEN[str(name)])
            else:
                raise NaturalMicrocodeError("program action differs")
        lines.append(f"R{index} " + " ".join(tokens))
    lines.append(f"C:{program.commit}")
    lines.append(CLOSE)
    return "\n".join(lines)


def parse_program(text: str) -> RegisterProgram:
    lines = text.strip().splitlines()
    if len(lines) < 4 or lines[0] != OPEN or lines[-1] != CLOSE:
        raise NaturalMicrocodeError("program envelope differs")
    records: list[tuple[dict[str, object], ...]] = []
    for line in lines[1:-2]:
        prefix, separator, body = line.partition(" ")
        if not separator or prefix != f"R{len(records)}" or not body:
            raise NaturalMicrocodeError("record address differs")
        actions = []
        for token in body.split():
            if token.startswith("P:"):
                surface = canonical_fraction(parse_fraction(token[2:]))
                actions.append({"action": "PUSH", "surface": surface})
            elif token.startswith("L:") and token[2:].isdigit():
                register = int(token[2:])
                if register >= len(records):
                    raise NaturalMicrocodeError("LOAD is not causal")
                actions.append({"action": "LOAD", "register": register})
            elif token in TOKEN_TO_ACTION:
                actions.append({"action": TOKEN_TO_ACTION[token]})
            else:
                raise NaturalMicrocodeError("record token differs")
        records.append(tuple(actions))
    commit_line = lines[-2]
    if not commit_line.startswith("C:") or not commit_line[2:].isdigit():
        raise NaturalMicrocodeError("commit differs")
    commit = int(commit_line[2:])
    if not records or commit != len(records) - 1:
        raise NaturalMicrocodeError("commit is not final register")
    return RegisterProgram(tuple(records), commit)


def execute_fraction(program: RegisterProgram) -> Fraction:
    registers: list[Fraction] = []
    for record in program.records:
        stack: list[Fraction] = []
        for action in record:
            name = action["action"]
            if name == "PUSH":
                stack.append(parse_fraction(str(action["surface"])))
            elif name == "LOAD":
                register = int(action["register"])
                if register >= len(registers):
                    raise NaturalMicrocodeError("assessor LOAD differs")
                stack.append(registers[register])
            elif name == "NEGATE":
                if not stack:
                    raise NaturalMicrocodeError("assessor NEGATE underflow")
                stack[-1] = -stack[-1]
            elif isinstance(name, str) and name.startswith("APPLY_"):
                if len(stack) < 2:
                    raise NaturalMicrocodeError("assessor APPLY underflow")
                right, left = stack.pop(), stack.pop()
                if name == "APPLY_ADD":
                    stack.append(left + right)
                elif name == "APPLY_SUB":
                    stack.append(left - right)
                elif name == "APPLY_MUL":
                    stack.append(left * right)
                elif name == "APPLY_DIV" and right:
                    stack.append(left / right)
                else:
                    raise NaturalMicrocodeError("assessor operation differs")
            else:
                raise NaturalMicrocodeError("assessor action differs")
        if len(stack) != 1:
            raise NaturalMicrocodeError("assessor record stack differs")
        registers.append(stack[0])
    return registers[program.commit]


def _digit_fraction(surface: str) -> DigitRational:
    numerator, separator, denominator = surface.partition("/")
    top = _surface_digits(numerator)
    if not separator:
        return top
    bottom = _surface_digits(denominator)
    if bottom.negative or bottom.numerator == (0,):
        raise LearnedArithmeticError("PUSH denominator differs")
    return DigitRational(top.negative, top.numerator, bottom.numerator)


def execute_learned(
    microcode: LearnedDigitMicrocode,
    program: RegisterProgram,
    *,
    intervention: str = "normal",
) -> DigitRational:
    if intervention not in {"normal", "carry_reset", "opcode_permuted"}:
        raise LearnedArithmeticError("intervention differs")
    registers: list[DigitRational] = []
    for record in program.records:
        stack: list[DigitRational] = []
        for action in record:
            name = action.get("action")
            if name == "PUSH":
                stack.append(_digit_fraction(str(action.get("surface"))))
            elif name == "LOAD":
                register = action.get("register")
                if type(register) is not int or not 0 <= register < len(registers):
                    raise LearnedArithmeticError("LOAD differs")
                stack.append(registers[register])
            elif name == "NEGATE":
                if not stack:
                    raise LearnedArithmeticError("NEGATE underflow")
                value = stack.pop()
                stack.append(
                    DigitRational(
                        not value.negative and value.numerator != (0,),
                        value.numerator,
                        value.denominator,
                    )
                )
            elif isinstance(name, str) and name.startswith("APPLY_"):
                if len(stack) < 2:
                    raise LearnedArithmeticError("APPLY underflow")
                right, left = stack.pop(), stack.pop()
                operation = (
                    OPCODE_PERMUTATION[name]
                    if intervention == "opcode_permuted"
                    else name
                )
                stack.append(
                    microcode.apply(
                        operation,
                        left,
                        right,
                        reset_carry=intervention == "carry_reset",
                    )
                )
            else:
                raise LearnedArithmeticError("action differs")
        if len(stack) != 1:
            raise LearnedArithmeticError("record stack differs")
        registers.append(stack[0])
    if not registers or not 0 <= program.commit < len(registers):
        raise LearnedArithmeticError("commit differs")
    return registers[program.commit]


def result_fields_absent(serialized: str, answer: str) -> bool:
    """Confirm that serialization has no explicit record-result or answer field."""
    del answer
    return (
        all("=" not in line for line in serialized.splitlines())
        and "RESULT" not in serialized
    )


def action_count(program: RegisterProgram) -> int:
    return sum(len(record) for record in program.records)


def register_depth(program: RegisterProgram) -> int:
    return len(program.records)


def all_actions(program: RegisterProgram) -> Sequence[dict[str, object]]:
    return tuple(action for record in program.records for action in record)
