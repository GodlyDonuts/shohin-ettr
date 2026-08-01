from __future__ import annotations

import pytest
import torch

from ettr_objectives import ETTRCausalQueryPair
from probe_ettr_causal_queries import _summary
from train_native_causal_disposition_pilot import (
    NativeDispositionPilotError,
    _annotate_pair_rows,
    _parse_args,
    _validate_args,
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


def test_slot_addresses_require_reader_training(tmp_path) -> None:
    args = _parse_args(
        [
            "--release-root",
            str(tmp_path / "release"),
            "--release-sha256",
            "0" * 64,
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "source.json"),
            "--target-tokenizer",
            str(tmp_path / "target.json"),
            "--parent-joint-model",
            str(tmp_path / "parent.pt"),
            "--parent-joint-model-sha256",
            "1" * 64,
            "--parent-run-contract",
            str(tmp_path / "contract.json"),
            "--parent-run-contract-sha256",
            "2" * 64,
            "--initial-reader",
            str(tmp_path / "reader.safetensors"),
            "--initial-reader-sha256",
            "3" * 64,
            "--output",
            str(tmp_path / "output"),
            "--source-commit",
            "4" * 40,
            "--data-seed",
            "5",
            "--model-seed",
            "6",
            "--reader-slot-addresses",
        ]
    )
    with pytest.raises(NativeDispositionPilotError, match="arguments"):
        _validate_args(args)
