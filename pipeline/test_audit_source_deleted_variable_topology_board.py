from __future__ import annotations

from audit_source_deleted_variable_topology_board import audit_board


def test_small_variable_topology_audit_passes() -> None:
    receipt = audit_board(
        seed=17,
        train_per_renderer=1,
        development_per_cell=1,
    )
    assert receipt["total_rows"] == 33
    assert receipt["exact_source_deleted"] == 33
    assert receipt["source_deletion_passes"] == 33
    assert receipt["family_name_leaks"] == 0
    assert receipt["collision_rows"] == 6
    assert receipt["collision_equal_incidence"] == 6
    assert receipt["incidence_type_ambiguous"] == 6
    assert receipt["incidence_type_separable"] == 27
    assert receipt["renderer_orbits_exact"] == 21
    assert receipt["renderer_orbits_packet_identical"] == 21
