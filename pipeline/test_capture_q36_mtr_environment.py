from __future__ import annotations

from pathlib import Path

import pytest

from capture_q36_mtr_environment import (
    Q36MTREnvironmentError,
    _canonical_member,
)


@pytest.mark.parametrize("value", ("member", "./member", "a/b"))
def test_q36_overlay_member_canonicalization(value: str) -> None:
    assert _canonical_member(value) == value.removeprefix("./")


@pytest.mark.parametrize("value", ("", ".", "../a", "a/../b", "/a", "a/./b"))
def test_q36_overlay_member_rejects_escape(value: str) -> None:
    with pytest.raises(Q36MTREnvironmentError):
        _canonical_member(value)


def test_environment_capture_pins_exact_qualified_overlays() -> None:
    source = Path(__file__).with_name("capture_q36_mtr_environment.py").read_text()
    assert "bitsandbytes-0.50.0-r1" in source
    assert "qwen36-fastkernels-0.4.2-r5" in source
    assert "2201774754fb2e0f" in source
    assert "dde2adf539302a32" in source
    assert '"scientific_rows_read": 0' in source
