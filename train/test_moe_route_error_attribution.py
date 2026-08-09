import pytest
import torch

from moe_route_error_attribution import (
    RouteAttributionError,
    compare_route_logits,
    summarize_comparisons,
)


def test_route_comparison_identical() -> None:
    logits = torch.tensor([[4.0, 3.0, 1.0], [0.0, 2.0, 1.0]])
    report = compare_route_logits(logits, logits.clone(), top_k=2)
    assert report["top1_change_rate"] == 0.0
    assert report["topk_position_change_rate"] == 0.0
    assert report["topk_set_overlap"] == 1.0
    assert report["probability_l1_mean"] == 0.0
    assert report["route_count_l1"] == 0.0


def test_route_comparison_detects_coherent_swap() -> None:
    baseline = torch.tensor([[9.0, 8.0, 0.0], [9.0, 8.0, 0.0]])
    arm = torch.tensor([[0.0, 8.0, 9.0], [0.0, 8.0, 9.0]])
    report = compare_route_logits(baseline, arm, top_k=2)
    assert report["top1_change_rate"] == 1.0
    assert report["topk_position_change_rate"] == 0.5
    assert report["topk_set_overlap"] == 0.5
    assert report["probability_l1_mean"] > 1.0
    assert report["route_count_l1"] == 1.0


def test_route_comparison_rejects_misaligned_logits() -> None:
    with pytest.raises(RouteAttributionError):
        compare_route_logits(torch.zeros(2, 4), torch.zeros(3, 4), top_k=2)


def test_group_summary_separates_early_and_late_layers() -> None:
    comparisons = []
    for layer in range(16):
        comparisons.append(
            {
                "layer": layer,
                "top1_change_rate": float(layer >= 12),
                "topk_position_change_rate": float(layer >= 12),
                "topk_set_overlap": float(layer < 12),
                "probability_l1_mean": float(layer >= 12),
                "route_count_l1": float(layer >= 12),
            }
        )
    rows = [
        {
            "group": "corrected",
            "comparisons": {"treatment": comparisons},
        }
    ]
    summary = summarize_comparisons(rows, "treatment")
    assert summary["corrected:first_twelve"]["top1_change_rate"] == 0.0
    assert summary["corrected:last_four"]["top1_change_rate"] == 1.0
    assert summary["corrected:all_layers"]["top1_change_rate"] == 0.25
