from __future__ import annotations

import torch

from run_multifamily_raw_machine_smoke import run_smoke


def test_tiny_learned_smoke_runs_without_candidate_oracles() -> None:
    report = run_smoke(
        seed=123,
        steps=2,
        width=32,
        layers=1,
        learning_rate=1e-3,
        device=torch.device("cpu"),
    )
    assert report["steps"] == 2
    assert report["candidate_time_oracle_calls"] == 0
    assert report["candidate_time_search_calls"] == 0
    assert report["candidate_time_verifier_calls"] == 0
    assert report["parameter_receipt"]["complete_system"] < 200_000_000
    assert report["train"]["total"] > 0
    assert report["development"]["total"] > 0
