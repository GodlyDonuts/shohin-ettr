#!/usr/bin/env python3
"""Focused standalone checks for ATS1 compilation and algebra."""

import torch

from diverge_ats1_data import segment_target
from diverge_ats1_runtime import (
    ATS1Config,
    SourceRoleCompiler,
    compile_segment,
    execute_step,
    render_typed_state,
)


def _target(family: str, step: str, operation: object):
    row = {
        "family": family,
        "depth": 1,
        "program": [operation],
        "wrong_steps": [step],
        "correct_steps": [step],
        "identity_sha256": "b" * 64,
    }
    return segment_target(row, 0, trace_kind="wrong")


def test_scalar_compile_and_execute() -> None:
    target = _target("scalar", "Step 1: -12 * 4 = -48.", ["multiply", 4])
    packet = compile_segment(target.byte_ids, target.role_ids, target.operation_id)
    assert render_typed_state(packet.lhs) == "-12"
    assert render_typed_state(execute_step(packet.lhs, packet.operation_id, packet.arguments)) == "-48"


def test_register_compile_and_execute() -> None:
    target = _target(
        "register",
        "Step 1: subtract A from B: (A=11, B=9) -> (A=11, B=-2).",
        "B-=A",
    )
    packet = compile_segment(target.byte_ids, target.role_ids, target.operation_id)
    assert render_typed_state(execute_step(packet.lhs, packet.operation_id, ())) == "11,-2"


def test_symbol_compile_and_execute() -> None:
    target = _target(
        "symbolic",
        "Step 1: rotate left by 3: abcdef -> defabc.",
        ["rotate", 3, 0],
    )
    packet = compile_segment(target.byte_ids, target.role_ids, target.operation_id)
    assert render_typed_state(execute_step(packet.lhs, packet.operation_id, packet.arguments)) == "defabc"


def test_compiler_shapes_and_gradients() -> None:
    config = ATS1Config(width=32, layers=1, heads=4, ff_multiplier=2)
    model = SourceRoleCompiler(config)
    ids = torch.zeros(2, config.max_bytes, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    ids[:, 0] = 1
    ids[:, 1:4] = torch.tensor([67, 68, 69])
    mask[:, :4] = True
    role, operation = model(ids, mask)
    assert role.shape == (2, config.max_bytes, 9)
    assert operation.shape == (2, 11)
    (role.sum() + operation.sum()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def main() -> None:
    test_scalar_compile_and_execute()
    test_register_compile_and_execute()
    test_symbol_compile_and_execute()
    test_compiler_shapes_and_gradients()
    print("diverge ATS1 runtime tests passed")


if __name__ == "__main__":
    main()
