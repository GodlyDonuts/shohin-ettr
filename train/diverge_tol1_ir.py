"""Exact typed register machine for the DIVERGE-TOL1 compiler gate."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


class TOL1IRError(RuntimeError):
    """A typed program violates the frozen TOL1 machine contract."""


DIRECT_OPS = ("SET", "ADD", "SUBTRACT", "MULTIPLY")
COMPARATORS = ("EQ", "NE", "LT", "LE", "GT", "GE")
CLAUSE_OPS = (*DIRECT_OPS, "SWAP", "GUARD", "QUERY")

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{1,15}\Z")
_FRACTION = re.compile(r"-?(?:0|[1-9]\d*)(?:/[1-9]\d*)?\Z")


def parse_fraction(text: str) -> Fraction:
    if _FRACTION.fullmatch(text) is None:
        raise TOL1IRError(f"invalid rational literal: {text!r}")
    value = Fraction(text)
    if abs(value.numerator).bit_length() > 40 or value.denominator.bit_length() > 24:
        raise TOL1IRError("rational literal exceeds TOL1 bounds")
    return value


def format_fraction(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def validate_identifier(value: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise TOL1IRError(f"invalid register identifier: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Atom:
    kind: str
    value: str

    def validate(self) -> None:
        if self.kind == "REF":
            validate_identifier(self.value)
        elif self.kind == "CONST":
            parse_fraction(self.value)
        else:
            raise TOL1IRError("unknown atom kind")


@dataclass(frozen=True, slots=True)
class Action:
    operation: str
    target: str
    operand: Atom

    def validate(self) -> None:
        if self.operation not in DIRECT_OPS:
            raise TOL1IRError("unknown direct action")
        validate_identifier(self.target)
        self.operand.validate()


@dataclass(frozen=True, slots=True)
class Predicate:
    comparator: str
    left: str
    right: Atom

    def validate(self) -> None:
        if self.comparator not in COMPARATORS:
            raise TOL1IRError("unknown comparator")
        validate_identifier(self.left)
        self.right.validate()


@dataclass(frozen=True, slots=True)
class Instruction:
    operation: str
    action: Action | None = None
    swap_left: str | None = None
    swap_right: str | None = None
    predicate: Predicate | None = None
    true_action: Action | None = None
    false_action: Action | None = None
    query: str | None = None

    def validate(self) -> None:
        if self.operation not in CLAUSE_OPS:
            raise TOL1IRError("unknown clause operation")
        populated = {
            "action": self.action is not None,
            "swap": self.swap_left is not None or self.swap_right is not None,
            "guard": any(
                value is not None
                for value in (self.predicate, self.true_action, self.false_action)
            ),
            "query": self.query is not None,
        }
        if self.operation in DIRECT_OPS:
            if not populated["action"] or any(
                populated[name] for name in ("swap", "guard", "query")
            ):
                raise TOL1IRError("direct instruction fields differ")
            assert self.action is not None
            self.action.validate()
            if self.action.operation != self.operation:
                raise TOL1IRError("direct action opcode differs")
        elif self.operation == "SWAP":
            if not populated["swap"] or any(
                populated[name] for name in ("action", "guard", "query")
            ):
                raise TOL1IRError("swap instruction fields differ")
            validate_identifier(str(self.swap_left))
            validate_identifier(str(self.swap_right))
            if self.swap_left == self.swap_right:
                raise TOL1IRError("self swap is forbidden")
        elif self.operation == "GUARD":
            if not populated["guard"] or any(
                populated[name] for name in ("action", "swap", "query")
            ):
                raise TOL1IRError("guard instruction fields differ")
            if None in (self.predicate, self.true_action, self.false_action):
                raise TOL1IRError("guard is incomplete")
            assert self.predicate and self.true_action and self.false_action
            self.predicate.validate()
            self.true_action.validate()
            self.false_action.validate()
        else:
            if not populated["query"] or any(
                populated[name] for name in ("action", "swap", "guard")
            ):
                raise TOL1IRError("query instruction fields differ")
            validate_identifier(str(self.query))


def atom_record(atom: Atom) -> dict[str, str]:
    atom.validate()
    return {"kind": atom.kind, "value": atom.value}


def action_record(action: Action) -> dict[str, object]:
    action.validate()
    return {
        "operation": action.operation,
        "target": action.target,
        "operand": atom_record(action.operand),
    }


def predicate_record(predicate: Predicate) -> dict[str, object]:
    predicate.validate()
    return {
        "comparator": predicate.comparator,
        "left": predicate.left,
        "right": atom_record(predicate.right),
    }


def instruction_record(instruction: Instruction) -> dict[str, object]:
    instruction.validate()
    record: dict[str, object] = {"operation": instruction.operation}
    if instruction.action is not None:
        record["action"] = action_record(instruction.action)
    if instruction.swap_left is not None:
        record["swap_left"] = instruction.swap_left
        record["swap_right"] = instruction.swap_right
    if instruction.predicate is not None:
        record["predicate"] = predicate_record(instruction.predicate)
        assert instruction.true_action and instruction.false_action
        record["true_action"] = action_record(instruction.true_action)
        record["false_action"] = action_record(instruction.false_action)
    if instruction.query is not None:
        record["query"] = instruction.query
    return record


def _atom_from_record(record: Any) -> Atom:
    if not isinstance(record, dict) or set(record) != {"kind", "value"}:
        raise TOL1IRError("serialized atom differs")
    return Atom(str(record["kind"]), str(record["value"]))


def _action_from_record(record: Any) -> Action:
    if not isinstance(record, dict) or set(record) != {
        "operation",
        "target",
        "operand",
    }:
        raise TOL1IRError("serialized action differs")
    return Action(
        str(record["operation"]),
        str(record["target"]),
        _atom_from_record(record["operand"]),
    )


def _predicate_from_record(record: Any) -> Predicate:
    if not isinstance(record, dict) or set(record) != {
        "comparator",
        "left",
        "right",
    }:
        raise TOL1IRError("serialized predicate differs")
    return Predicate(
        str(record["comparator"]),
        str(record["left"]),
        _atom_from_record(record["right"]),
    )


def instruction_from_record(record: Any) -> Instruction:
    if not isinstance(record, dict) or "operation" not in record:
        raise TOL1IRError("serialized instruction differs")
    operation = str(record["operation"])
    if operation in DIRECT_OPS:
        expected = {"operation", "action"}
        if set(record) != expected:
            raise TOL1IRError("serialized direct instruction differs")
        instruction = Instruction(
            operation,
            action=_action_from_record(record["action"]),
        )
    elif operation == "SWAP":
        if set(record) != {"operation", "swap_left", "swap_right"}:
            raise TOL1IRError("serialized swap differs")
        instruction = Instruction(
            operation,
            swap_left=str(record["swap_left"]),
            swap_right=str(record["swap_right"]),
        )
    elif operation == "GUARD":
        if set(record) != {
            "operation",
            "predicate",
            "true_action",
            "false_action",
        }:
            raise TOL1IRError("serialized guard differs")
        instruction = Instruction(
            operation,
            predicate=_predicate_from_record(record["predicate"]),
            true_action=_action_from_record(record["true_action"]),
            false_action=_action_from_record(record["false_action"]),
        )
    elif operation == "QUERY":
        if set(record) != {"operation", "query"}:
            raise TOL1IRError("serialized query differs")
        instruction = Instruction(operation, query=str(record["query"]))
    else:
        raise TOL1IRError("serialized opcode differs")
    instruction.validate()
    return instruction


def instruction_sha256(instruction: Instruction) -> str:
    payload = json.dumps(
        instruction_record(instruction), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolve(atom: Atom, state: Mapping[str, Fraction]) -> Fraction:
    atom.validate()
    if atom.kind == "CONST":
        return parse_fraction(atom.value)
    try:
        return state[atom.value]
    except KeyError as error:
        raise TOL1IRError("read from an uninitialized register") from error


def _apply(action: Action, state: dict[str, Fraction]) -> None:
    action.validate()
    operand = _resolve(action.operand, state)
    if action.operation == "SET":
        result = operand
    else:
        try:
            current = state[action.target]
        except KeyError as error:
            raise TOL1IRError("update of an uninitialized register") from error
        if action.operation == "ADD":
            result = current + operand
        elif action.operation == "SUBTRACT":
            result = current - operand
        elif action.operation == "MULTIPLY":
            result = current * operand
        else:
            raise TOL1IRError("unknown direct action")
    if abs(result.numerator).bit_length() > 80 or result.denominator.bit_length() > 40:
        raise TOL1IRError("register value exceeds TOL1 bounds")
    state[action.target] = result


def _test(predicate: Predicate, state: Mapping[str, Fraction]) -> bool:
    predicate.validate()
    try:
        left = state[predicate.left]
    except KeyError as error:
        raise TOL1IRError("predicate reads an uninitialized register") from error
    right = _resolve(predicate.right, state)
    return {
        "EQ": left == right,
        "NE": left != right,
        "LT": left < right,
        "LE": left <= right,
        "GT": left > right,
        "GE": left >= right,
    }[predicate.comparator]


def execute_program(program: Sequence[Instruction]) -> tuple[Fraction, tuple[dict[str, str], ...]]:
    if not program or program[-1].operation != "QUERY":
        raise TOL1IRError("program must terminate in exactly one late query")
    if any(instruction.operation == "QUERY" for instruction in program[:-1]):
        raise TOL1IRError("early query is forbidden")
    state: dict[str, Fraction] = {}
    trajectory: list[dict[str, str]] = []
    for instruction in program:
        instruction.validate()
        if instruction.operation in DIRECT_OPS:
            assert instruction.action is not None
            _apply(instruction.action, state)
        elif instruction.operation == "SWAP":
            assert instruction.swap_left and instruction.swap_right
            try:
                state[instruction.swap_left], state[instruction.swap_right] = (
                    state[instruction.swap_right],
                    state[instruction.swap_left],
                )
            except KeyError as error:
                raise TOL1IRError("swap reads an uninitialized register") from error
        elif instruction.operation == "GUARD":
            assert instruction.predicate and instruction.true_action and instruction.false_action
            _apply(
                instruction.true_action if _test(instruction.predicate, state) else instruction.false_action,
                state,
            )
        else:
            assert instruction.query is not None
            try:
                answer = state[instruction.query]
            except KeyError as error:
                raise TOL1IRError("query reads an uninitialized register") from error
            trajectory.append(
                {key: format_fraction(value) for key, value in sorted(state.items())}
            )
            return answer, tuple(trajectory)
        trajectory.append(
            {key: format_fraction(value) for key, value in sorted(state.items())}
        )
    raise TOL1IRError("program did not return")


__all__ = [
    "Action",
    "Atom",
    "CLAUSE_OPS",
    "COMPARATORS",
    "DIRECT_OPS",
    "Instruction",
    "Predicate",
    "TOL1IRError",
    "execute_program",
    "format_fraction",
    "instruction_from_record",
    "instruction_record",
    "instruction_sha256",
    "parse_fraction",
    "validate_identifier",
]
