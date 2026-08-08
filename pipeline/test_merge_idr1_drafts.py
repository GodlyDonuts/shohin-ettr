from __future__ import annotations

from merge_idr1_drafts import ADAPTER_SHA256, MODEL_REVISION, SEED


def test_frozen_idr1_draft_constants() -> None:
    assert len(ADAPTER_SHA256) == 64
    assert MODEL_REVISION == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert SEED == 2026080818
