#!/usr/bin/env python3
"""CPU contract tests for DIVERGE-NPW1 narrative rendering."""

from __future__ import annotations

from diverge_npw1_data import (
    TRAIN_NAMES,
    augment_board,
    render_narrative,
    training_record_from_tol1,
)
from diverge_tfs1_data import StepSpec
from diverge_tol1_ir import Action, Atom, Instruction, Predicate, instruction_record


def _steps() -> tuple[StepSpec, ...]:
    return (
        StepSpec(
            "typed-set",
            fixed=Instruction("SET", action=Action("SET", "alpha", Atom("CONST", "3"))),
        ),
        StepSpec(
            "typed-ambiguous",
            options=(
                Instruction("ADD", action=Action("ADD", "alpha", Atom("REF", "beta"))),
                Instruction("SUBTRACT", action=Action("SUBTRACT", "alpha", Atom("REF", "beta"))),
            ),
            fault_index=0,
        ),
        StepSpec(
            "typed-swap",
            fixed=Instruction("SWAP", swap_left="alpha", swap_right="beta"),
        ),
        StepSpec(
            "typed-guard",
            fixed=Instruction(
                "GUARD",
                predicate=Predicate("LT", "alpha", Atom("REF", "beta")),
                true_action=Action("ADD", "alpha", Atom("CONST", "2")),
                false_action=Action("SUBTRACT", "beta", Atom("CONST", "1")),
            ),
        ),
    )


def test_narrative_has_no_instruction_lines_and_exact_spans() -> None:
    narrative = render_narrative(_steps(), seed=2026080619, confirmation=True)
    source = narrative["source_text"]
    assert "\n" not in source
    assert len(narrative["events"]) == len(_steps())
    assert {event["form"] for event in narrative["events"]} == {
        "DIRECT",
        "AMBIGUOUS",
        "SWAP",
        "GUARD",
    }
    for event in narrative["events"]:
        assert source[event["start"] : event["end"]]
        for mention in event["mentions"]:
            assert source[mention["start"] : mention["end"]] == mention["text"]


def test_augmentation_commits_semantics_and_source() -> None:
    row = {
        "identity_sha256": "a" * 64,
        "steps": [
            {
                "text": step.text,
                "fixed": None if step.fixed is None else instruction_record(step.fixed),
                "options": (
                    None
                    if step.options is None
                    else [instruction_record(value) for value in step.options]
                ),
                "fault_index": step.fault_index,
            }
            for step in _steps()
        ],
    }
    first = augment_board([row], seed=2026080619, confirmation=True)
    second = augment_board([row], seed=2026080619, confirmation=True)
    assert first == second
    assert first[0]["npw1_identity_sha256"] != row["identity_sha256"]


def test_tol1_training_conversion_renames_and_adds_ambiguity() -> None:
    instructions = (
        Instruction("SET", action=Action("SET", "alpha", Atom("CONST", "3"))),
        Instruction("SET", action=Action("SET", "beta", Atom("CONST", "4"))),
        Instruction("SET", action=Action("SET", "gamma", Atom("CONST", "5"))),
        Instruction("SET", action=Action("SET", "delta", Atom("CONST", "6"))),
        Instruction("ADD", action=Action("ADD", "alpha", Atom("REF", "beta"))),
        Instruction("SWAP", swap_left="gamma", swap_right="delta"),
        Instruction("QUERY", query="alpha"),
    )
    row = {"clauses": [{"instruction": instruction_record(value)} for value in instructions]}
    converted = training_record_from_tol1(row, index=0, seed=2026080618)
    assert set(converted["symbols"]) <= set(TRAIN_NAMES)
    assert set(converted["symbols"]).isdisjoint({"alpha", "beta", "gamma", "delta"})
    assert any(
        event["form"] == "AMBIGUOUS"
        for event in converted["natural_world"]["events"]
    )


if __name__ == "__main__":
    test_narrative_has_no_instruction_lines_and_exact_spans()
    test_augmentation_commits_semantics_and_source()
    test_tol1_training_conversion_renames_and_adds_ambiguity()
    print("diverge NPW1 data tests passed")
