from __future__ import annotations

import torch

from run_multifamily_structural_type_smoke import run_structural_type_smoke


def test_tiny_structural_type_smoke_runs_without_candidate_oracles() -> None:
    report = run_structural_type_smoke(
        seed=123,
        steps=2,
        width=32,
        layers=1,
        learning_rate=1e-3,
        device=torch.device("cpu"),
    )
    assert report["candidate_time_oracle_calls"] == 0
    assert report["candidate_time_search_calls"] == 0
    assert report["candidate_time_verifier_calls"] == 0
    assert report["parameter_receipt"]["complete_system"] < 200_000_000
    assert report["development"]["structural_key_classes"]["total"] == 24
    assert report["development"]["structural_key_shuffle"]["total"] == 24
