from __future__ import annotations

import argparse

import pytest

import hf_q36_mtr_train_commit_256 as module


def test_engineering_entrypoint_pins_256_updates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(updates=module.ENGINEERING_UPDATES)
    monkeypatch.setattr(module.implementation, "UPDATES", 128)
    monkeypatch.setattr(module.implementation, "parse_args", lambda: args)
    monkeypatch.setattr(
        module.implementation,
        "train",
        lambda received: {
            "status": "complete",
            "updates": received.updates,
            "pair_presentations": received.updates * 8,
            "checkpoint_sha256": "a" * 64,
        },
    )
    assert module.main() == 0
    assert '"updates": 256' in capsys.readouterr().out
    assert module.implementation.UPDATES == module.ENGINEERING_UPDATES


def test_engineering_entrypoint_rejects_other_update_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module.implementation, "UPDATES", 128)
    monkeypatch.setattr(
        module.implementation, "parse_args", lambda: argparse.Namespace(updates=128)
    )
    with pytest.raises(module.implementation.Q36MTRCommitError):
        module.main()
