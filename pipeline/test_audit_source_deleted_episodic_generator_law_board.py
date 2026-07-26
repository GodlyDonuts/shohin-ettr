from __future__ import annotations

from audit_source_deleted_episodic_generator_law_board import (
    audit_board,
)


def test_episodic_generator_law_audit_passes() -> None:
    receipt = audit_board(
        seed=20260725,
        train_per_renderer=1,
        development_per_cell=1,
    )
    assert receipt["total_rows"] == 26
    assert receipt["exact_source_deleted"] == 26
    assert receipt["unique_target_laws"] == 52
    assert receipt["target_law_overlap"] == 0
    assert receipt["train_target_words"] == 4
    assert receipt["development_target_word_instances"] == 22
    assert receipt["development_target_word_overlap_instances"] == 14
    assert receipt["target_word_overlap"] == 4
    assert receipt["target_word_holdout_passes"] is False
    assert receipt["raw_target_map_overlap"] == 0
    assert receipt["hidden_query_rows"] == 26
    assert receipt["visible_query_steps"] == 0
    assert (
        receipt["record_removals_ambiguous"]
        == receipt["minimal_witness_records"]
    )
    assert receipt["law_swap_answer_changes"] == 26
    assert (
        receipt["action_order_answer_changes"]
        == receipt["action_order_eligible"]
    )
    assert receipt["renderer_orbits_total"] == 8
    assert receipt["renderer_orbits_packet_identical"] == 8
    assert receipt["renderer_orbits_law_identical"] == 8
    assert receipt["train_held_out_family_rows"] == 0
    assert receipt["held_out_law_joint_rows"] == 2
