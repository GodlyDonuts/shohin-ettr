from __future__ import annotations

from audit_episodic_generator_control_identifiability import (
    audit_deranged_support_identifiability,
)


def test_deranged_support_contains_one_coherent_wrong_world() -> None:
    report = audit_deranged_support_identifiability(seed=20260726)
    assert report["development_rows"] == 11
    assert report["syntactic_programs"] == 127
    assert report["classification_counts"] == {
        "ambiguous": 6,
        "contradictory": 4,
        "identified_wrong": 1,
    }
    assert report["abstract_interface_zero_seal_possible"] is False
    wrong = [
        episode
        for episode in report["episodes"]
        if episode["classification"] == "identified_wrong"
    ]
    assert len(wrong) == 1
    assert wrong[0]["family"] == "random_permutation"
    assert wrong[0]["cell"] == "law"
    assert all(
        target["matching_unique_maps"] == 1
        and target["true_map_survives"] is False
        for target in wrong[0]["targets"]
    )
