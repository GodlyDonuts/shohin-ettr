import pytest

from score_vfr1_trace_quality import VFR1QualityError, summarize


def _row(index: int, *, correct: bool = True) -> dict[str, object]:
    return {
        "identity_sha256": f"{index:064x}",
        "task": "math500",
        "target_kind": "source_verified_repair",
        "parse_error": None,
        "score": {"correct": correct},
        "reference_leak": False,
        "max_token_exhausted": False,
        "fault": "Arithmetic error.",
    }


def test_quality_gate_passes_exact_boundary() -> None:
    rows = [_row(index, correct=index < 90) for index in range(100)]
    for index in range(5):
        rows[index]["parse_error"] = "bad"
    for index in range(2):
        rows[index]["reference_leak"] = True
    for index in range(10):
        rows[index]["max_token_exhausted"] = True
    result = summarize(rows, 100)
    assert result["gate_pass"] is True


def test_quality_gate_fails_below_verifier_floor() -> None:
    rows = [_row(index, correct=index < 89) for index in range(100)]
    result = summarize(rows, 100)
    assert result["gate_pass"] is False
    assert result["gates"]["verified_at_least_0_90"] is False


def test_quality_gate_rejects_wrong_row_count() -> None:
    with pytest.raises(VFR1QualityError, match="row count"):
        summarize([_row(0)], 2)
