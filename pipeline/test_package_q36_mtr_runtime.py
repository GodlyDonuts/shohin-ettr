from __future__ import annotations

from pathlib import Path

import pytest

from package_q36_mtr_runtime import Q36MTRRuntimeError, load_allowlist


def test_production_q36_allowlist_is_sorted_closed_and_exactly_one_dispatcher() -> None:
    root = Path(__file__).resolve().parents[1]
    entries = load_allowlist(root / "pipeline/q36_mtr_runtime_allowlist.txt")
    assert entries == sorted(entries)
    assert all((root / entry).is_file() for entry in entries)
    assert [entry for entry in entries if "dispatch" in entry.casefold()] == [
        "pipeline/dispatch_q36_mtr.py"
    ]
    assert not any("q35" in entry.casefold() for entry in entries)


@pytest.mark.parametrize(
    "entry",
    (
        "../train/hf_q36_mtr_train_role.py",
        "/tmp/q36.py",
        "train/ndr1_retry.py",
        "train/q35_edit_selector.py",
        "train/jobs/dispatch_pcf1.sh",
    ),
)
def test_q36_allowlist_rejects_escape_retry_and_dispatch(
    tmp_path: Path, entry: str
) -> None:
    path = tmp_path / "allowlist.txt"
    path.write_text(entry + "\n", encoding="utf-8")
    with pytest.raises(Q36MTRRuntimeError):
        load_allowlist(path)
