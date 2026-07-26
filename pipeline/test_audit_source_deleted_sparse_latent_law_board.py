from __future__ import annotations

from audit_source_deleted_sparse_latent_law_board import audit_board


def test_sparse_latent_law_board_audit_passes_frozen_matrix() -> None:
    receipt = audit_board(
        seed=20260725,
        train_per_renderer=4,
        development_per_cell=4,
    )
    assert receipt["exact_source_deleted"] == 120
    assert receipt["hidden_query_rows"] == 120
    assert receipt["visible_query_steps"] == 0
    assert receipt["train_development_action_law_overlap"] == 0
    assert (
        receipt["record_removals_nonidentifiable"]
        == receipt["minimal_witness_records"]
    )
    assert receipt["renderer_orbits_packet_identical"] == 12
    assert receipt["hypothesis_counts"] == {"8": 78, "16": 309}
