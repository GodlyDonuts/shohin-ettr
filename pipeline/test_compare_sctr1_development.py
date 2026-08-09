from __future__ import annotations

from pathlib import Path


SOURCE = (Path(__file__).resolve().parent / "compare_sctr1_development.py").read_text(
    encoding="utf-8"
)


def test_gate_is_conjunctive_and_identity_matched() -> None:
    assert "selective_beats_unchanged_by_5_points" in SOURCE
    assert "selective_at_least_always_revise" in SOURCE
    assert "selective_beats_shuffled_by_3_points" in SOURCE
    assert "all_domain_deltas_nonnegative" in SOURCE
    assert "zero_malformed_commitments" in SOURCE
    assert "identities(report) != reference_ids" in SOURCE
    assert 'independent.get("mask_internal_draft") is not True' in SOURCE
    assert "holdout_authorized" in SOURCE
