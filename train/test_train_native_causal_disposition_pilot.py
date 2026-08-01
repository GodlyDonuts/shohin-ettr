from __future__ import annotations

import pytest
import torch

from ettr_objectives import ETTRCausalQueryPair
from probe_ettr_causal_queries import _summary
from train_native_causal_disposition_pilot import (
    NativeDispositionPilotError,
    _annotate_pair_rows,
)


def _pair() -> ETTRCausalQueryPair:
    return ETTRCausalQueryPair(
        correct_logits=torch.tensor(((2.0, 0.0), (0.0, 2.0))),
        foil_logits=torch.tensor(((0.0, 2.0), (2.0, 0.0))),
        correct_target=torch.tensor((0, 1)),
        foil_target=torch.tensor((1, 0)),
    )


def test_pair_rows_carry_the_complete_shared_summary_contract() -> None:
    rows = _annotate_pair_rows(_pair(), torch.tensor((3, 17)))
    assert [row["depth"] for row in rows] == [3, 17]
    assert [row["depth_bucket"] for row in rows] == ["3-4", "17-32"]
    summary = _summary(rows)
    assert summary["count"] == 2
    assert summary["by_depth"]["3-4"]["count"] == 1
    assert summary["by_depth"]["17-32"]["count"] == 1


def test_pair_rows_reject_depth_support_mismatch() -> None:
    with pytest.raises(NativeDispositionPilotError, match="depth support"):
        _annotate_pair_rows(_pair(), torch.tensor((3,)))
