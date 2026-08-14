from pathlib import Path

import pytest

from hf_q36_mtr_generate_owner_preview import (
    Q36MTROwnerPreviewError,
    select_preview_rows,
)
from q36_mtr_roles import DRAFT_SHARDS


def test_selects_only_development_rows_from_canonical_shard() -> None:
    rows = [
        {
            "identity_sha256": f"{index:064x}",
            "split": "development" if index % 5 == 0 else "train",
        }
        for index in range(160)
    ]
    selected, start, end = select_preview_rows(rows, 3)
    assert start == 30
    assert end == 40
    assert [row["identity_sha256"] for row in selected] == [
        f"{index:064x}" for index in (30, 35)
    ]


def test_rejects_invalid_canonical_shard() -> None:
    with pytest.raises(Q36MTROwnerPreviewError, match="index"):
        select_preview_rows([], DRAFT_SHARDS)


def test_preview_wrapper_is_single_h100_and_nonrequeue() -> None:
    source = Path("train/jobs/q36_mtr_generate_owner_preview.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
    assert "#SBATCH --no-requeue" in source
    assert "--canonical-shard-index 0" in source
    assert "q36_require_authorization" not in source
