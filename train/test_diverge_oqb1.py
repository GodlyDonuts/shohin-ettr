#!/usr/bin/env python3
"""Focused mechanics and oracle tests for DIVERGE-OQB1."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from diverge_oqb1_data import (
    DEVELOPMENT_SEED,
    augment_evaluation_episode,
    build_training_record,
    canonical_to_position,
    rotate_table,
    validate_evaluation_episode,
)
from diverge_oqb1_runtime import (
    OccurrenceQuotientRegisterBinder,
    SLOT_MARKERS,
    exact_occurrence_quotient,
    tensorize_quotient_sources,
)
from eval_diverge_cab1 import _execution_score, _position_targets
from eval_diverge_jrb1 import _compile_packets, _decode_initial_states


def main() -> None:
    first = build_training_record(17)
    assert first == build_training_record(17)
    assert first != build_training_record(18)
    names = {
        value
        for serial in range(1_000)
        for value in (
            build_training_record(serial)["operation"],
            *build_training_record(serial)["register_table"],
        )
    }
    assert len(names) == 3_000
    assert sorted(first["evidence_position_targets"]) == [0, 0, 1, 1]
    assert sorted(first["initial_position_targets"]) == [0, 1]
    assert rotate_table(("rax", "reb"), 1) == ("reb", "rax")
    assert [canonical_to_position(value, 1) for value in (0, 1)] == [1, 0]

    text = "rax was 3; reb held 9; later rax became 4 and reb became 8."
    quotient, found, assigned = exact_occurrence_quotient(text, ("rax", "reb"))
    assert found == (2, 2) and assigned == (2, 2)
    assert quotient.count(SLOT_MARKERS[0]) == 2
    assert quotient.count(SLOT_MARKERS[1]) == 2
    reversed_quotient, reversed_found, _ = exact_occurrence_quotient(
        text, ("reb", "rax")
    )
    assert reversed_found == (2, 2)
    assert reversed_quotient == quotient.translate(
        str.maketrans(
            {
                "a": "a",
            }
        )
    ).replace(SLOT_MARKERS[0], "zztemporaryz").replace(
        SLOT_MARKERS[1], SLOT_MARKERS[0]
    ).replace("zztemporaryz", SLOT_MARKERS[1])
    scrubbed, scrubbed_found, _ = exact_occurrence_quotient(
        text.replace("rax", "foo").replace("reb", "bar"), ("rax", "reb")
    )
    assert scrubbed_found == (0, 0) and scrubbed == text.replace("rax", "foo").replace(
        "reb", "bar"
    )
    broken_a = exact_occurrence_quotient(
        text, ("rax", "reb"), mode="broken", salt="fixed"
    )
    broken_b = exact_occurrence_quotient(
        text, ("rax", "reb"), mode="broken", salt="fixed"
    )
    assert broken_a == broken_b

    public, assessor = augment_evaluation_episode(3, seed=DEVELOPMENT_SEED)
    assert "registers" not in public
    assert "renamed_registers" not in public
    assert "initial_state" not in public["transfer"][0]
    assert "symbols" not in public["transfer"][0]
    assert "register_index" not in public["queries"][0]
    validate_evaluation_episode(public, assessor)
    damaged = deepcopy(public)
    damaged["register_table"].reverse()
    try:
        validate_evaluation_episode(damaged, assessor)
    except Exception:
        pass
    else:
        raise AssertionError("OQB1 accepted a mutated table")

    records = [
        {
            "source": "Initially, rax was 3; meanwhile, reb held 9.",
            "registers": ["rax", "reb"],
            "serial": 0,
        },
        {
            "source": "Initially, reb was 8; meanwhile, rax held 4.",
            "registers": ["rax", "reb"],
            "serial": 1,
        },
    ]
    model = OccurrenceQuotientRegisterBinder()
    tensors = tensorize_quotient_sources(
        records,
        torch.device("cpu"),
        text_key="source",
        mention_count=2,
    )
    assert tensors[2] is not None and tensors[3] is not None
    assert torch.all(tensors[4])
    mention_logits = model.forward_mentions(
        tensors[0], tensors[1], tensors[2], tensors[3]
    )
    query_records = [
        {"source": "At the end, report rax.", "registers": ["rax", "reb"]},
        {"source": "At the end, report reb.", "registers": ["rax", "reb"]},
    ]
    query_tensors = tensorize_quotient_sources(
        query_records,
        torch.device("cpu"),
        text_key="source",
        mention_count=None,
    )
    query_logits = model.forward_query(query_tensors[0], query_tensors[1])
    assert mention_logits.shape == (2, 2, 2)
    assert query_logits.shape == (2, 2)
    loss = torch.nn.functional.cross_entropy(
        mention_logits.flatten(0, 1), torch.tensor([0, 1, 1, 0])
    ) + torch.nn.functional.cross_entropy(query_logits, torch.tensor([0, 1]))
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    board = [
        augment_evaluation_episode(index, seed=DEVELOPMENT_SEED) for index in range(2)
    ]
    visible = [value[0] for value in board]
    hidden = [value[1] for value in board]
    evidence_positions = _position_targets(
        visible,
        hidden,
        group="evidence",
        table_key="register_table",
        canonical_key="canonical_registers",
        reverse_table=False,
    )
    complete_roles = []
    cursor = 0
    for episode in hidden:
        for item in episode["evidence"]:
            positions = evidence_positions[cursor]
            complete_roles.append(
                tuple(
                    (int(role) // 2) * 2 + int(position)
                    for role, position in zip(
                        item["numeric_role_ids"], positions, strict=True
                    )
                )
            )
            cursor += 1
    packets = _compile_packets(
        visible,
        complete_roles,
        text_key="source_text",
        hash_key="source_sha256",
        owner_state_sha256="0" * 64,
    )
    initial_positions = _position_targets(
        visible,
        hidden,
        group="initial",
        table_key="register_table",
        canonical_key="canonical_registers",
        reverse_table=False,
    )
    initial_states = _decode_initial_states(
        visible, initial_positions, text_key="initial_text"
    )
    programs = [
        tuple(item["targets"])
        for episode in hidden
        for item in episode["command_targets"]
    ]
    query_positions = _position_targets(
        visible,
        hidden,
        group="query",
        table_key="register_table",
        canonical_key="canonical_registers",
        reverse_table=False,
    )
    execution = _execution_score(
        packets,
        visible,
        hidden,
        programs,
        initial_states,
        query_positions,
        state_table_key="register_table",
        canonical_key="canonical_registers",
        reverse_state_table=False,
    )
    assert execution["state_exact_rate"] == 1.0
    assert execution["answer_exact_rate"] == 1.0

    runtime_source = Path(__file__).with_name("diverge_oqb1_runtime.py").read_text()
    assert "exact_occurrence_quotient" in runtime_source
    assert "torch.logsumexp" in runtime_source
    assert "scan_register_ids" not in runtime_source
    assert "register_encoder" not in runtime_source
    print("diverge OQB1 mechanics tests passed")


if __name__ == "__main__":
    main()
