from collections import Counter
from types import SimpleNamespace

import pytest

from audit_ettr_program_templates import (
    ProgramTemplateAuditError,
    summarize_counter,
    trace_signatures,
)


def _trace(*, value_code=(7, 9, 0), step_mask=(True, True, False)):
    return SimpleNamespace(
        opcode=(1, 6, 0),
        source=(4, 0, 0),
        target=(0, 0, 0),
        relation=(0, 0, 0),
        type_index=(0, 0, 0),
        value_code=value_code,
        step_mask=step_mask,
    )


def test_signatures_ignore_padding_and_factor_value_payload() -> None:
    left = trace_signatures(_trace())
    padding_changed = trace_signatures(_trace(value_code=(7, 9, 255)))
    payload_changed = trace_signatures(_trace(value_code=(8, 9, 0)))

    assert left == padding_changed
    assert left["exact"] != payload_changed["exact"]
    assert left["structural"] == payload_changed["structural"]
    assert left["opcode"] == payload_changed["opcode"]


def test_signatures_reject_nonprefix_mask() -> None:
    with pytest.raises(ProgramTemplateAuditError, match="not a prefix"):
        trace_signatures(_trace(step_mask=(True, False, True)))


def test_counter_summary_reports_entropy_and_frequency() -> None:
    report = summarize_counter(Counter({"a": 3, "b": 1}))
    assert report["instances"] == 4
    assert report["unique"] == 2
    assert report["entropy_bits"] == pytest.approx(0.811278, abs=1e-6)
    assert report["top"][0] == {"count": 3, "rate": 0.75, "sha256": "a"}
