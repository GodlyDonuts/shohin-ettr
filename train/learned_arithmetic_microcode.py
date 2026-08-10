"""Learned finite digit transitions composed into a rational stack machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = 10
MAX_DIGITS = 256
ADD_CLASSES = 20
MUL_CLASSES = 100
OPCODE_PERMUTATION = {
    "APPLY_ADD": "APPLY_MUL",
    "APPLY_SUB": "APPLY_ADD",
    "APPLY_MUL": "APPLY_SUB",
    "APPLY_DIV": "APPLY_MUL",
}


class LearnedArithmeticError(ValueError):
    """A microcode transition or stack state violates the frozen schema."""


@dataclass(frozen=True, slots=True)
class DigitRational:
    negative: bool
    numerator: tuple[int, ...]
    denominator: tuple[int, ...]


def _trim(value: Iterable[int]) -> tuple[int, ...]:
    digits = list(value)
    if not digits:
        return (0,)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
    if len(digits) > MAX_DIGITS or any(
        type(digit) is not int or not 0 <= digit < BASE for digit in digits
    ):
        raise LearnedArithmeticError("digit vector differs or overflows")
    return tuple(digits)


def _is_zero(value: Sequence[int]) -> bool:
    return len(value) == 1 and value[0] == 0


def _compare(left: Sequence[int], right: Sequence[int]) -> int:
    a, b = _trim(left), _trim(right)
    if len(a) != len(b):
        return -1 if len(a) < len(b) else 1
    for x, y in zip(reversed(a), reversed(b), strict=True):
        if x != y:
            return -1 if x < y else 1
    return 0


def _surface_digits(surface: str) -> DigitRational:
    value = surface.strip().replace(",", "")
    negative = value.startswith("-")
    if value[:1] in {"-", "+"}:
        value = value[1:]
    if (
        value.count(".") > 1
        or not value
        or any(character not in "0123456789." for character in value)
    ):
        raise LearnedArithmeticError("numeric surface differs")
    whole, dot, fraction = value.partition(".")
    if not whole:
        whole = "0"
    if not whole.isdigit() or (dot and (not fraction or not fraction.isdigit())):
        raise LearnedArithmeticError("numeric surface digits differ")
    numerator_text = (whole + fraction).lstrip("0") or "0"
    denominator_text = "1" + "0" * len(fraction)
    numerator = _trim(int(character) for character in reversed(numerator_text))
    denominator = _trim(int(character) for character in reversed(denominator_text))
    return DigitRational(negative and not _is_zero(numerator), numerator, denominator)


class LearnedDigitMicrocode(nn.Module):
    """Complete learned transition tables for base-10 local arithmetic."""

    def __init__(self) -> None:
        super().__init__()
        self.add_logits = nn.Parameter(torch.zeros(BASE, BASE, 2, ADD_CLASSES))
        self.sub_logits = nn.Parameter(torch.zeros(BASE, BASE, 2, ADD_CLASSES))
        self.mul_logits = nn.Parameter(torch.zeros(BASE, BASE, BASE, MUL_CLASSES))
        self._add_table: list | None = None
        self._sub_table: list | None = None
        self._mul_table: list | None = None

    @staticmethod
    def parameter_count() -> int:
        return BASE * BASE * 2 * ADD_CLASSES * 2 + BASE * BASE * BASE * MUL_CLASSES

    def transition_loss(self) -> torch.Tensor:
        device = self.add_logits.device
        a = torch.arange(BASE, device=device)[:, None, None]
        b = torch.arange(BASE, device=device)[None, :, None]
        carry2 = torch.arange(2, device=device)[None, None, :]
        add_total = a + b + carry2
        sub_total = a - b - carry2
        sub_digit = torch.remainder(sub_total, BASE)
        sub_borrow = (sub_total < 0).long()
        mul_a = torch.arange(BASE, device=device)[:, None, None]
        mul_b = torch.arange(BASE, device=device)[None, :, None]
        mul_carry = torch.arange(BASE, device=device)[None, None, :]
        product = mul_a * mul_b + mul_carry
        labels = (
            torch.remainder(add_total, BASE)
            + BASE * torch.div(add_total, BASE, rounding_mode="floor"),
            sub_digit + BASE * sub_borrow,
            torch.remainder(product, BASE)
            + BASE * torch.div(product, BASE, rounding_mode="floor"),
        )
        return sum(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                reduction="sum",
            )
            for logits, target in zip(
                (self.add_logits, self.sub_logits, self.mul_logits), labels, strict=True
            )
        )

    def transition_exact(self) -> tuple[int, int]:
        device = self.add_logits.device
        a = torch.arange(BASE, device=device)[:, None, None]
        b = torch.arange(BASE, device=device)[None, :, None]
        c2 = torch.arange(2, device=device)[None, None, :]
        c10 = torch.arange(BASE, device=device)[None, None, :]
        add = a + b + c2
        sub = a - b - c2
        product = a + 0 * b + 0 * c10
        del product
        targets = (
            torch.remainder(add, BASE)
            + BASE * torch.div(add, BASE, rounding_mode="floor"),
            torch.remainder(sub, BASE) + BASE * (sub < 0).long(),
            torch.remainder(a * b + c10, BASE)
            + BASE * torch.div(a * b + c10, BASE, rounding_mode="floor"),
        )
        exact = sum(
            int((logits.argmax(-1) == target).sum().item())
            for logits, target in zip(
                (self.add_logits, self.sub_logits, self.mul_logits),
                targets,
                strict=True,
            )
        )
        return exact, sum(target.numel() for target in targets)

    def _transition(
        self, table: torch.Tensor, discrete: list | None, a: int, b: int, carry: int
    ) -> tuple[int, int]:
        output = (
            int(table[a, b, carry].argmax().item())
            if discrete is None
            else int(discrete[a][b][carry])
        )
        return output % BASE, output // BASE

    def freeze_discrete(self) -> None:
        """Cache learned argmax tables for recurrent candidate execution."""
        self._add_table = self.add_logits.detach().argmax(-1).cpu().tolist()
        self._sub_table = self.sub_logits.detach().argmax(-1).cpu().tolist()
        self._mul_table = self.mul_logits.detach().argmax(-1).cpu().tolist()

    def add_unsigned(
        self, left: Sequence[int], right: Sequence[int], *, reset_carry: bool = False
    ) -> tuple[int, ...]:
        size = max(len(left), len(right))
        output: list[int] = []
        carry = 0
        for index in range(size):
            a = left[index] if index < len(left) else 0
            b = right[index] if index < len(right) else 0
            digit, carry = self._transition(
                self.add_logits,
                self._add_table,
                a,
                b,
                0 if reset_carry else carry,
            )
            output.append(digit)
        if carry and not reset_carry:
            output.append(carry)
        return _trim(output)

    def sub_unsigned(
        self, left: Sequence[int], right: Sequence[int], *, reset_carry: bool = False
    ) -> tuple[int, ...]:
        if _compare(left, right) < 0:
            raise LearnedArithmeticError("unsigned subtraction underflows")
        output: list[int] = []
        borrow = 0
        for index in range(len(left)):
            a = left[index]
            b = right[index] if index < len(right) else 0
            digit, borrow = self._transition(
                self.sub_logits,
                self._sub_table,
                a,
                b,
                0 if reset_carry else borrow,
            )
            output.append(digit)
        if borrow and not reset_carry:
            raise LearnedArithmeticError("subtraction leaves a borrow")
        return _trim(output)

    def mul_unsigned(
        self, left: Sequence[int], right: Sequence[int], *, reset_carry: bool = False
    ) -> tuple[int, ...]:
        if _is_zero(left) or _is_zero(right):
            return (0,)
        result: tuple[int, ...] = (0,)
        for offset, a in enumerate(left):
            partial: list[int] = [0] * offset
            carry = 0
            for b in right:
                digit, carry = self._transition(
                    self.mul_logits,
                    self._mul_table,
                    a,
                    b,
                    0 if reset_carry else carry,
                )
                partial.append(digit)
            if carry and not reset_carry:
                partial.append(carry)
            result = self.add_unsigned(result, _trim(partial), reset_carry=reset_carry)
        return _trim(result)

    def add_signed(
        self,
        left_negative: bool,
        left: Sequence[int],
        right_negative: bool,
        right: Sequence[int],
        *,
        reset_carry: bool = False,
    ) -> tuple[bool, tuple[int, ...]]:
        if left_negative == right_negative:
            value = self.add_unsigned(left, right, reset_carry=reset_carry)
            return left_negative and not _is_zero(value), value
        comparison = _compare(left, right)
        if comparison == 0:
            return False, (0,)
        if comparison > 0:
            value = self.sub_unsigned(left, right, reset_carry=reset_carry)
            return left_negative and not _is_zero(value), value
        value = self.sub_unsigned(right, left, reset_carry=reset_carry)
        return right_negative and not _is_zero(value), value

    def apply(
        self,
        operation: str,
        left: DigitRational,
        right: DigitRational,
        *,
        reset_carry: bool = False,
    ) -> DigitRational:
        if operation == "APPLY_DIV" and _is_zero(right.numerator):
            raise LearnedArithmeticError("division by zero")
        if operation in {"APPLY_ADD", "APPLY_SUB"}:
            left_num = self.mul_unsigned(
                left.numerator, right.denominator, reset_carry=reset_carry
            )
            right_num = self.mul_unsigned(
                right.numerator, left.denominator, reset_carry=reset_carry
            )
            right_negative = right.negative ^ (operation == "APPLY_SUB")
            negative, numerator = self.add_signed(
                left.negative,
                left_num,
                right_negative,
                right_num,
                reset_carry=reset_carry,
            )
            denominator = self.mul_unsigned(
                left.denominator, right.denominator, reset_carry=reset_carry
            )
        elif operation == "APPLY_MUL":
            numerator = self.mul_unsigned(
                left.numerator, right.numerator, reset_carry=reset_carry
            )
            denominator = self.mul_unsigned(
                left.denominator, right.denominator, reset_carry=reset_carry
            )
            negative = left.negative ^ right.negative
        elif operation == "APPLY_DIV":
            numerator = self.mul_unsigned(
                left.numerator, right.denominator, reset_carry=reset_carry
            )
            denominator = self.mul_unsigned(
                left.denominator, right.numerator, reset_carry=reset_carry
            )
            negative = left.negative ^ right.negative
        else:
            raise LearnedArithmeticError("operation differs")
        if _is_zero(denominator):
            raise LearnedArithmeticError("denominator is zero")
        return DigitRational(
            negative and not _is_zero(numerator), _trim(numerator), _trim(denominator)
        )


def execute_microcode(
    microcode: LearnedDigitMicrocode,
    actions: Sequence[dict[str, object]],
    *,
    intervention: str = "normal",
) -> DigitRational:
    if intervention not in {"normal", "carry_reset", "opcode_permuted"}:
        raise LearnedArithmeticError("intervention differs")
    stack: list[DigitRational] = []
    stopped = False
    for action in actions:
        name = action.get("action")
        if name == "PUSH":
            surface = action.get("surface")
            if not isinstance(surface, str):
                raise LearnedArithmeticError("PUSH surface differs")
            stack.append(_surface_digits(surface))
        elif name == "NEGATE":
            if not stack:
                raise LearnedArithmeticError("NEGATE underflow")
            value = stack.pop()
            stack.append(
                DigitRational(
                    not value.negative and not _is_zero(value.numerator),
                    value.numerator,
                    value.denominator,
                )
            )
        elif isinstance(name, str) and name.startswith("APPLY_"):
            if len(stack) < 2:
                raise LearnedArithmeticError("APPLY underflow")
            right, left = stack.pop(), stack.pop()
            operation = (
                OPCODE_PERMUTATION[name] if intervention == "opcode_permuted" else name
            )
            stack.append(
                microcode.apply(
                    operation, left, right, reset_carry=intervention == "carry_reset"
                )
            )
        elif name == "STOP":
            if stopped or len(stack) != 1:
                raise LearnedArithmeticError("STOP state differs")
            stopped = True
        else:
            raise LearnedArithmeticError("action differs")
    if not stopped or len(stack) != 1:
        raise LearnedArithmeticError("program did not commit")
    return stack[0]
