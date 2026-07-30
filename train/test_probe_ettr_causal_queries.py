from __future__ import annotations

import torch

from ettr_objectives import ETTRCausalQueryPair
from probe_ettr_causal_queries import (
    _depth_bucket,
    _pair_rows,
    _quantile,
    _summary,
)


def test_pair_rows_measure_exact_difference_in_differences() -> None:
    pair = ETTRCausalQueryPair(
        correct_logits=torch.tensor(
            [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0]]
        ),
        foil_logits=torch.tensor(
            [[0.0, 2.0, -1.0], [0.0, 2.0, -1.0]]
        ),
        correct_target=torch.tensor([0, 1]),
        foil_target=torch.tensor([1, 1]),
    )
    rows = _pair_rows(pair)
    assert rows[0]["contrast"] is True
    assert rows[0]["correct_delta"] == 3.0
    assert rows[0]["foil_delta"] == 2.0
    assert rows[0]["difference_in_differences"] == 5.0
    assert rows[0]["correct_top1"] is True
    assert rows[0]["foil_top1"] is True
    assert rows[1]["contrast"] is False


def test_summary_stratifies_effect_rows_by_depth() -> None:
    rows = [
        {
            **row,
            "depth_bucket": bucket,
        }
        for row, bucket in zip(
            _pair_rows(
                ETTRCausalQueryPair(
                    correct_logits=torch.tensor(
                        [[3.0, 0.0], [0.5, 0.0]]
                    ),
                    foil_logits=torch.tensor(
                        [[0.0, 2.0], [0.0, 0.25]]
                    ),
                    correct_target=torch.tensor([0, 0]),
                    foil_target=torch.tensor([1, 1]),
                )
            ),
            ("1", "3-4"),
            strict=True,
        )
    ]
    summary = _summary(rows)
    assert summary["count"] == 2
    assert summary["margin_rates"]["1"] == 0.5
    assert summary["by_depth"]["1"]["margin_rates"]["1"] == 1.0
    assert summary["by_depth"]["3-4"]["margin_rates"]["1"] == 0.0


def test_quantile_and_depth_buckets_are_deterministic() -> None:
    assert _quantile([1.0, 3.0], 0.5) == 2.0
    assert [_depth_bucket(value) for value in (1, 2, 3, 4, 5, 9, 33)] == [
        "1",
        "2",
        "3-4",
        "3-4",
        "5-8",
        "9-16",
        "33-64",
    ]
