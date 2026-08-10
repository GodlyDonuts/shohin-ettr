from fractions import Fraction

import torch

from learned_arithmetic_microcode import (
    LearnedDigitMicrocode,
    execute_microcode,
)


def _fit() -> LearnedDigitMicrocode:
    model = LearnedDigitMicrocode()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model.transition_loss().backward()
        optimizer.step()
    model.freeze_discrete()
    return model


def _fraction(value) -> Fraction:
    numerator = int("".join(str(digit) for digit in reversed(value.numerator)))
    denominator = int("".join(str(digit) for digit in reversed(value.denominator)))
    result = Fraction(numerator, denominator)
    return -result if value.negative else result


def test_all_local_transitions_learn() -> None:
    model = _fit()
    assert model.parameter_count() == 108_000
    assert model.transition_exact() == (1400, 1400)


def test_recurrent_rational_program_composes() -> None:
    model = _fit()
    actions = [
        {"action": "PUSH", "surface": "12.5"},
        {"action": "PUSH", "surface": "3"},
        {"action": "APPLY_MUL"},
        {"action": "PUSH", "surface": "2"},
        {"action": "APPLY_SUB"},
        {"action": "PUSH", "surface": "7"},
        {"action": "APPLY_DIV"},
        {"action": "STOP"},
    ]
    assert _fraction(execute_microcode(model, actions)) == Fraction(71, 14)


def test_carry_reset_is_causal() -> None:
    model = _fit()
    actions = [
        {"action": "PUSH", "surface": "99"},
        {"action": "PUSH", "surface": "1"},
        {"action": "APPLY_ADD"},
        {"action": "STOP"},
    ]
    assert _fraction(execute_microcode(model, actions)) == 100
    assert (
        _fraction(execute_microcode(model, actions, intervention="carry_reset")) != 100
    )
