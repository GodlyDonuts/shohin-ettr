from __future__ import annotations

import math

import pytest
import torch

import analyze_q36_mtr_trajectory_state_geometry as module


def _state(scale: float, rotation: bool = False) -> dict[str, torch.Tensor]:
    factor_a = (
        torch.tensor([[0.0, scale], [scale, 0.0]])
        if rotation
        else torch.tensor([[scale, 0.0], [0.0, scale]])
    )
    factor_b = torch.eye(2)
    return {
        "backbone.model.layers.1.mlp.adapter_a.weight": factor_a,
        "backbone.model.layers.1.mlp.adapter_b.weight": factor_b,
    }


def test_operator_metrics_are_factorization_invariant() -> None:
    left = (torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0], [0.0]]))
    equivalent = (torch.tensor([[0.5, 0.0]]), torch.tensor([[4.0], [0.0]]))
    orthogonal = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0], [3.0]]))
    assert module._operator_inner(left, equivalent) == pytest.approx(4.0)
    assert module._operator_norm(left) == pytest.approx(2.0)
    assert module._operator_norm(equivalent) == pytest.approx(2.0)
    assert module._operator_inner(left, orthogonal) == pytest.approx(0.0)


def test_analysis_reports_pairwise_cosine_delta_and_rank() -> None:
    result = module.analyze_states(
        {
            "owner": _state(1.0),
            "revision": _state(1.1),
            "draft_hidden": _state(1.0, rotation=True),
        },
        (1,),
    )
    row = result["layers"][0]
    assert row["pairs"]["owner_vs_revision"]["operator_cosine"] == pytest.approx(1.0)
    assert row["pairs"]["owner_vs_revision"]["relative_delta_to_left"] == pytest.approx(
        0.1
    )
    assert row["pairs"]["owner_vs_draft_hidden"]["operator_cosine"] == pytest.approx(
        0.0
    )
    assert row["effective_ranks"]["owner"] == pytest.approx(2.0)
    assert (
        result["aggregate_pairs"]["revision_vs_draft_hidden"]["relative_delta_mean"]
        > 1.0
    )


def test_analysis_rejects_bad_role_order_or_nonfinite_factors() -> None:
    states = {
        "revision": _state(1.0),
        "owner": _state(1.0),
        "draft_hidden": _state(1.0),
    }
    with pytest.raises(module.Q36MTRTrajectoryGeometryError):
        module.analyze_states(states, (1,))
    states = {
        "owner": _state(1.0),
        "revision": _state(1.0),
        "draft_hidden": _state(1.0),
    }
    states["revision"]["backbone.model.layers.1.mlp.adapter_a.weight"][0, 0] = math.nan
    with pytest.raises(module.Q36MTRTrajectoryGeometryError):
        module.analyze_states(states, (1,))
