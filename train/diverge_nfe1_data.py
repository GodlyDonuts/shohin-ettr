"""Source extraction and lexical mention targets for DIVERGE-NFE1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence


SCALAR_OPERATIONS = ("add", "subtract", "multiply")
ROLE_NAMES = ("LHS", "ARGUMENT", "RHS")
ROLE_TO_ID = {name: index for index, name in enumerate(ROLE_NAMES)}
MAX_ABS_VALUE = 3_000_000

EQUATION = re.compile(
    r"(?P<lhs>-?\d+)\s*(?P<operator>[+\-*])\s*"
    r"(?P<argument>-?\d+)\s*=\s*(?P<rhs>-?\d+)"
)


class NFE1DataError(RuntimeError):
    """A source row cannot satisfy the frozen NFE1 contract."""


@dataclass(frozen=True, slots=True)
class Equation:
    text: str
    lhs: int
    operator: str
    argument: int
    rhs: int
    mention_spans: tuple[tuple[int, int], ...]

    @property
    def operation(self) -> str:
        return symbol_to_operation(self.operator)

    @property
    def exact_identity(self) -> str:
        return hashlib.sha256(self.text.encode("ascii")).hexdigest()

    def record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "text_sha256": self.exact_identity,
            "lhs": self.lhs,
            "operator": self.operator,
            "operation": self.operation,
            "argument": self.argument,
            "rhs": self.rhs,
            "mention_spans": [list(span) for span in self.mention_spans],
        }


def apply_scalar(operation: str, lhs: int, argument: int) -> int:
    if operation == "add":
        return lhs + argument
    if operation == "subtract":
        return lhs - argument
    if operation == "multiply":
        return lhs * argument
    raise NFE1DataError(f"unsupported scalar operation: {operation}")


def symbol_to_operation(symbol: str) -> str:
    try:
        return {"+": "add", "-": "subtract", "*": "multiply"}[symbol]
    except KeyError as error:
        raise NFE1DataError(f"unsupported scalar symbol: {symbol!r}") from error


def operation_to_symbol(operation: str) -> str:
    try:
        return {"add": "+", "subtract": "-", "multiply": "*"}[operation]
    except KeyError as error:
        raise NFE1DataError(f"unsupported scalar operation: {operation!r}") from error


def rotate_symbol(symbol: str) -> str:
    try:
        return {"+": "-", "-": "*", "*": "+"}[symbol]
    except KeyError as error:
        raise NFE1DataError(f"unsupported scalar symbol: {symbol!r}") from error


def _unary_minus(text: str, index: int) -> bool:
    previous = index - 1
    while previous >= 0 and text[previous].isspace():
        previous -= 1
    return previous < 0 or text[previous] in "=+-*([{,;:"


def scan_signed_integer_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return maximal signed integer mentions without treating binary '-' as a sign."""

    try:
        text.encode("ascii")
    except UnicodeEncodeError as error:
        raise NFE1DataError("equation source is not ASCII") from error
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if not text[index].isdigit():
            index += 1
            continue
        start = index
        if index > 0 and text[index - 1] == "-" and _unary_minus(text, index - 1):
            start -= 1
        end = index + 1
        while end < len(text) and text[end].isdigit():
            end += 1
        if spans and start < spans[-1][1]:
            raise NFE1DataError("numeric mention spans overlap")
        spans.append((start, end))
        index = end
    return tuple(spans)


def equation_from_match(match: re.Match[str]) -> Equation | None:
    lhs = int(match.group("lhs"))
    operator = match.group("operator")
    argument = int(match.group("argument"))
    rhs = int(match.group("rhs"))
    if apply_scalar(symbol_to_operation(operator), lhs, argument) != rhs:
        return None
    text = match.group(0)
    spans = scan_signed_integer_spans(text)
    if len(spans) != 3:
        raise NFE1DataError("verified equation does not expose three mentions")
    values = tuple(int(text[start:end]) for start, end in spans)
    if values != (lhs, argument, rhs):
        raise NFE1DataError("lexical mentions disagree with verified equation")
    return Equation(text, lhs, operator, argument, rhs, spans)


def extract_verified_equations(text: str) -> tuple[Equation, ...]:
    output: list[Equation] = []
    for match in EQUATION.finditer(text):
        equation = equation_from_match(match)
        if equation is not None:
            output.append(equation)
    return tuple(output)


def corrupt_visible_operator(equation: Equation) -> Equation:
    match = EQUATION.fullmatch(equation.text)
    if match is None:
        raise NFE1DataError("equation no longer matches its exact source span")
    start, end = match.span("operator")
    text = (
        equation.text[:start] + rotate_symbol(equation.operator) + equation.text[end:]
    )
    spans = scan_signed_integer_spans(text)
    values = tuple(int(text[left:right]) for left, right in spans)
    if len(spans) != 3 or values != (equation.lhs, equation.argument, equation.rhs):
        raise NFE1DataError("operator corruption changed numeric evidence")
    return Equation(
        text,
        equation.lhs,
        rotate_symbol(equation.operator),
        equation.argument,
        equation.rhs,
        spans,
    )


def distinct_candidate_outcomes(equation: Equation) -> bool:
    return len(
        {
            apply_scalar(operation, equation.lhs, equation.argument)
            for operation in SCALAR_OPERATIONS
        }
    ) == len(SCALAR_OPERATIONS)


def complete_verified_chain(
    row: Mapping[str, Any],
    *,
    training_texts: set[str],
) -> tuple[Equation, ...] | None:
    equations = extract_verified_equations(str(row.get("response") or ""))
    if not 2 <= len(equations) <= 5:
        return None
    if any(
        equations[index].lhs != equations[index - 1].rhs
        for index in range(1, len(equations))
    ):
        return None
    try:
        answer = int(str(row["answer"]))
    except (KeyError, TypeError, ValueError):
        return None
    if equations[-1].rhs != answer:
        return None
    if any(
        max(abs(equation.lhs), abs(equation.argument), abs(equation.rhs))
        >= MAX_ABS_VALUE
        for equation in equations
    ):
        return None
    if not all(distinct_candidate_outcomes(equation) for equation in equations):
        return None
    if any(equation.text in training_texts for equation in equations):
        return None
    return equations


def training_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[str, Equation] = {}
    for row in rows:
        if row.get("source") != "reasoning_gym_trace":
            continue
        for equation in extract_verified_equations(str(row.get("response") or "")):
            deduplicated.setdefault(equation.text, equation)
    output: list[dict[str, Any]] = []
    for text in sorted(deduplicated):
        equation = deduplicated[text]
        record = equation.record()
        record.update(
            {
                "schema": "shohin-diverge-nfe1-mention-training-v1",
                "identity_sha256": hashlib.sha256(
                    ("nfe1-mention-training\0" + text).encode("ascii")
                ).hexdigest(),
                "role_ids": list(range(len(ROLE_NAMES))),
            }
        )
        output.append(record)
    return output


def validate_training_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != "shohin-diverge-nfe1-mention-training-v1":
        raise NFE1DataError("mention-training schema differs")
    text = str(record["text"])
    spans = tuple(
        tuple(int(value) for value in span) for span in record["mention_spans"]
    )
    if spans != scan_signed_integer_spans(text) or len(spans) != 3:
        raise NFE1DataError("mention-training spans differ")
    if tuple(int(value) for value in record["role_ids"]) != (0, 1, 2):
        raise NFE1DataError("mention-training roles differ")
    values = tuple(int(text[start:end]) for start, end in spans)
    expected = (int(record["lhs"]), int(record["argument"]), int(record["rhs"]))
    if values != expected:
        raise NFE1DataError("mention-training values differ")
    if apply_scalar(str(record["operation"]), values[0], values[1]) != values[2]:
        raise NFE1DataError("mention-training arithmetic differs")


def validate_board_row(row: Mapping[str, Any]) -> None:
    if row.get("schema") != "shohin-diverge-nfe1-board-v1":
        raise NFE1DataError("NFE1 board schema differs")
    steps = row.get("steps")
    if not isinstance(steps, Sequence) or not 2 <= len(steps) <= 5:
        raise NFE1DataError("NFE1 board depth differs")
    previous_rhs: int | None = None
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or int(step["step_index"]) != index:
            raise NFE1DataError("NFE1 step index differs")
        source = str(step["source_text"])
        spans = scan_signed_integer_spans(source)
        if len(spans) != 3:
            raise NFE1DataError("NFE1 source mentions differ")
        values = tuple(int(source[start:end]) for start, end in spans)
        expected = (int(step["lhs"]), int(step["argument"]), int(step["rhs"]))
        if values != expected:
            raise NFE1DataError("NFE1 source evidence differs")
        if previous_rhs is not None and values[0] != previous_rhs:
            raise NFE1DataError("NFE1 chain continuity differs")
        if apply_scalar(str(step["gold_operation"]), values[0], values[1]) != values[2]:
            raise NFE1DataError("NFE1 gold arithmetic differs")
        if symbol_to_operation(str(step["visible_operator"])) == str(
            step["gold_operation"]
        ):
            raise NFE1DataError("NFE1 visible operation was not corrupted")
        previous_rhs = values[2]
    if int(row["answer"]) != previous_rhs or int(row["depth"]) != len(steps):
        raise NFE1DataError("NFE1 terminal answer differs")


__all__ = [
    "EQUATION",
    "Equation",
    "MAX_ABS_VALUE",
    "NFE1DataError",
    "ROLE_NAMES",
    "ROLE_TO_ID",
    "SCALAR_OPERATIONS",
    "apply_scalar",
    "complete_verified_chain",
    "corrupt_visible_operator",
    "distinct_candidate_outcomes",
    "extract_verified_equations",
    "operation_to_symbol",
    "rotate_symbol",
    "scan_signed_integer_spans",
    "symbol_to_operation",
    "training_records",
    "validate_board_row",
    "validate_training_record",
]
