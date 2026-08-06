#!/usr/bin/env python3
"""CPU mechanics tests for the DIVERGE-NPW1 narrative owner."""

from __future__ import annotations

import torch

from diverge_npw1_data import training_record_from_tol1
from diverge_npw1_runtime import (
    MAX_CANDIDATES,
    MAX_EVENTS,
    NPW1Config,
    NarrativeProgramWeaver,
    ROLE_NAMES,
    lexical_candidates,
    tensorize_records,
)
from diverge_tol1_ir import Action, Atom, Instruction, instruction_record


def _record() -> dict[str, object]:
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
    return training_record_from_tol1(row, index=0, seed=2026080618)


def test_lexical_lattice_is_source_only_and_ordered() -> None:
    source = str(_record()["natural_world"]["source_text"])
    candidates = lexical_candidates(source)
    assert candidates
    assert all(left.start < right.start for left, right in zip(candidates, candidates[1:]))
    assert all(source[value.start : value.end].lower() == value.text for value in candidates)


def test_tensorization_and_recurrent_forward() -> None:
    batch = tensorize_records([_record()], torch.device("cpu"))
    assert batch["event_count"].item() == 6
    assert (
        batch["start_targets"][batch["event_mask"]].max().item()
        < MAX_CANDIDATES
    )
    model = NarrativeProgramWeaver(NPW1Config()).eval()
    with torch.no_grad():
        output = model(
            batch["byte_ids"],
            batch["byte_mask"],
            batch["candidate_masks"],
            batch["candidate_valid"],
            batch["candidate_kind"],
            teacher_starts=batch["start_targets"],
        )
    assert output["form_logits"].shape == (1, MAX_EVENTS + 1, 5)
    assert output["start_logits"].shape == (1, MAX_EVENTS, MAX_CANDIDATES + 1)
    assert output["role_logits"].shape == (
        1,
        MAX_EVENTS,
        len(ROLE_NAMES),
        MAX_CANDIDATES + 1,
    )
    assert all(not torch.isnan(value).any() for value in output.values())


if __name__ == "__main__":
    test_lexical_lattice_is_source_only_and_ordered()
    test_tensorization_and_recurrent_forward()
    print("diverge NPW1 runtime tests passed")
