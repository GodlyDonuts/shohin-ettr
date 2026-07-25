from __future__ import annotations

from audit_source_deleted_multifamily_machine_board import audit_board


def test_independent_multifamily_audit_passes() -> None:
    report = audit_board(
        seed=20260725,
        train_per_renderer=3,
        development_per_cell=3,
        orbit_seeds=3,
    )
    assert report["exact_source_deleted"]["correct"] == report["row_count"]
    assert report["family_label_leaks"] == 0
    assert report["role_neutral_key_passes"] == report["row_count"]
    assert report["law_swap_intervention"]["rate"] >= 0.50
    assert report["order_intervention"]["rate"] >= 0.25
    assert report["renderer_orbits"]["exact"] == 9
    assert report["renderer_orbits"]["packet_equal"] == 9
    assert len(report["payload_sha256"]) == 64
