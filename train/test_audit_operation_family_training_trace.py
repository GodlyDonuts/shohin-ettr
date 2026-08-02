import json

import pytest

from audit_operation_family_training_trace import (
    OperationFamilyTraceAuditError,
    audit_trace,
    summarize_rows,
)


def _row(position: int, counts: tuple[int, int, int], loss: float) -> dict:
    return {
        "atomic_action_counts": {
            "operation_0.effect_family": {
                "0": counts[0],
                "1": counts[1],
                "2": counts[2],
            }
        },
        "loss": loss,
        "position": position,
    }


def test_summary_detects_regime_batches_and_final_shift() -> None:
    result = summarize_rows(
        (
            _row(9, (8, 8, 0), 0.7),
            _row(19, (0, 0, 16), 0.1),
            _row(29, (0, 0, 16), 0.0),
            _row(39, (12, 4, 0), 0.69),
        )
    )
    assert result["missing_at_least_one_family_rate"] == 1.0
    assert result["single_family_rate"] == 0.5
    assert result["dominant_run_max_logged_updates"] == 2
    assert result["final_logged_batch"]["histogram"] == {"0": 12, "1": 4, "2": 0}


def test_trace_receipt_is_hash_bound(tmp_path) -> None:
    trace = tmp_path / "train.jsonl"
    trace.write_text(json.dumps(_row(9, (4, 4, 4), 1.0)) + "\n", encoding="ascii")
    result = audit_trace(trace, source_label="test-arm")
    assert result["source"]["label"] == "test-arm"
    assert result["interpretation"]["release_gate"] == "pass"


def test_trace_rejects_nonmonotonic_positions() -> None:
    with pytest.raises(OperationFamilyTraceAuditError, match="positions"):
        summarize_rows((_row(9, (1, 1, 1), 1.0), _row(9, (1, 1, 1), 1.0)))


def test_trace_rejects_missing_family_schema() -> None:
    row = _row(9, (1, 1, 1), 1.0)
    del row["atomic_action_counts"]["operation_0.effect_family"]["2"]
    with pytest.raises(OperationFamilyTraceAuditError, match="counts differ"):
        summarize_rows((row,))
