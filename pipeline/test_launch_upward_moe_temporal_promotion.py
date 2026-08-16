"""Tests for the evidence-bound automatic larger-MoE temporal launch."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import launch_upward_moe_temporal_promotion as module


def _args(tmp_path: Path, *, submit: bool = True) -> SimpleNamespace:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    python = tmp_path / "python"
    python.write_text("python\n", encoding="utf-8")
    python.chmod(0o755)
    return SimpleNamespace(
        promotion=tmp_path / "promotion.json",
        runtime=runtime,
        runtime_manifest_sha256="a" * 64,
        python=python,
        run_root=tmp_path / "scientific_run",
        claim=tmp_path / "automation" / "claim.json",
        dispatch_receipt=tmp_path / "automation" / "dispatch.json",
        launch_receipt=tmp_path / "automation" / "launch.json",
        failure_receipt=tmp_path / "automation" / "terminal_failure.json",
        submit=submit,
    )


def _promotion(score_paths: list[Path], *, selected: str | None) -> dict:
    candidates = [
        {"source_path": str(path.resolve()), "source_sha256": "a" * 64}
        for path in score_paths
    ]
    if selected is None:
        return {
            "status": "no_qualifying_larger_host",
            "selected_dispatcher_host": None,
            "candidates": candidates,
        }
    return {
        "status": "promote",
        "selected_dispatcher_host": selected,
        "candidates": candidates,
    }


def _write_promotion(
    path: Path, score_paths: list[Path], *, selected: str | None
) -> dict:
    for score in score_paths:
        score.write_text("{}\n", encoding="utf-8")
    payload = _promotion(score_paths, selected=selected)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


def _patch_dispatch(monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace) -> None:
    dispatch_args = SimpleNamespace(
        host="mixtral-8x22b",
        runtime=args.runtime,
        model_manifest=args.runtime / "model.sha256",
        mechanics_report=args.runtime / "mechanics.json",
        b1=args.runtime / "b1.jsonl",
        train_source=args.runtime / "train.jsonl",
        development_source=args.runtime / "development.jsonl",
        freeze_report=args.runtime / "freeze.json",
        assessor_receipt=args.runtime / "assessor_receipt.json",
        assessors=args.runtime / "assessors.jsonl",
        overlay_manifest=None,
        run_root=args.run_root,
        receipt=args.dispatch_receipt,
        submit=args.submit,
    )
    for name in (
        "SHA256SUMS",
        "model.sha256",
        "mechanics.json",
        "b1.jsonl",
        "train.jsonl",
        "development.jsonl",
        "freeze.json",
        "assessor_receipt.json",
        "assessors.jsonl",
    ):
        (args.runtime / name).write_text(name + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "_dispatch_args", lambda *_unused: dispatch_args)
    monkeypatch.setattr(module.dispatcher, "build_graph", lambda _args: ["graph"])
    monkeypatch.setattr(module.dispatcher, "validate", lambda *_unused: None)
    monkeypatch.setattr(
        module.dispatcher,
        "submit",
        lambda *_unused: {
            "status": "submitted" if args.submit else "dry_run",
            "job_ids": {"owner": "123", "score": "456"},
            "allocation_tasks": 102,
        },
    )


def test_replay_requires_byte_equivalent_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = [tmp_path / "super.json", tmp_path / "mixtral.json"]
    receipt = tmp_path / "promotion.json"
    expected = _write_promotion(receipt, scores, selected="mixtral-8x22b")
    monkeypatch.setattr(module.promotion_selector, "select", lambda paths: expected)
    assert module.replay_promotion(receipt) == expected
    tampered = dict(expected)
    tampered["selected_dispatcher_host"] = "nemotron-super"
    receipt.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(module.UpwardMoETemporalLaunchError, match="replay"):
        module.replay_promotion(receipt)


def test_no_qualifying_host_writes_terminal_no_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.promotion.write_text("{}\n", encoding="utf-8")
    decision = _promotion([tmp_path / "a", tmp_path / "b"], selected=None)
    monkeypatch.setattr(module, "replay_promotion", lambda _path: decision)
    result = module.launch(args)
    assert result["status"] == "no_launch"
    assert result["scientific_jobs_submitted"] == 0
    assert args.launch_receipt.is_file()
    assert not args.claim.exists()


def test_qualified_host_claims_once_and_submits_exact_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.promotion.write_text("{}\n", encoding="utf-8")
    decision = _promotion(
        [tmp_path / "super", tmp_path / "mixtral"], selected="mixtral-8x22b"
    )
    monkeypatch.setattr(module, "replay_promotion", lambda _path: decision)
    _patch_dispatch(monkeypatch, args)
    result = module.launch(args)
    assert result["status"] == "submitted"
    assert result["selected_host"] == "mixtral-8x22b"
    assert result["allocation_tasks"] == 102
    assert args.claim.is_file()
    assert args.dispatch_receipt.is_file()
    assert args.launch_receipt.is_file()
    assert not args.failure_receipt.exists()
    with pytest.raises(module.UpwardMoETemporalLaunchError, match="exists"):
        module.launch(args)


def test_dry_run_creates_no_claim_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, submit=False)
    args.promotion.write_text("{}\n", encoding="utf-8")
    decision = _promotion(
        [tmp_path / "super", tmp_path / "mixtral"], selected="mixtral-8x22b"
    )
    monkeypatch.setattr(module, "replay_promotion", lambda _path: decision)
    _patch_dispatch(monkeypatch, args)
    result = module.launch(args)
    assert result["status"] == "dry_run"
    assert result["dispatch"]["allocation_tasks"] == 102
    assert not args.claim.exists()
    assert not args.run_root.exists()


def test_post_claim_submission_failure_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.promotion.write_text("{}\n", encoding="utf-8")
    decision = _promotion(
        [tmp_path / "super", tmp_path / "mixtral"], selected="nemotron-super"
    )
    monkeypatch.setattr(module, "replay_promotion", lambda _path: decision)
    _patch_dispatch(monkeypatch, args)

    def fail(*_unused):
        raise module.dispatcher.UpwardMoETemporalDispatchError("scheduler denied")

    monkeypatch.setattr(module.dispatcher, "submit", fail)
    with pytest.raises(module.dispatcher.UpwardMoETemporalDispatchError):
        module.launch(args)
    failure = json.loads(args.failure_receipt.read_text(encoding="utf-8"))
    assert failure["status"] == "terminal_infrastructure_failure_after_claim"
    assert failure["automatic_retry"] is False
    assert args.claim.is_file()


def test_campaign_constants_are_upward_only_and_source_disjoint() -> None:
    assert set(module.HOST_PATHS) == {"nemotron-super", "mixtral-8x22b"}
    assert "pcf17_ministral" in str(module.COMMON_PATHS["source_root"])
    assert "confirmation_assessors.jsonl" in str(module.COMMON_PATHS["assessors"])
    assert "qwen36-fastkernels" in str(
        module.HOST_PATHS["nemotron-super"]["causal_conv_root"]
    )
    assert module.HOST_PATHS["mixtral-8x22b"]["overlay_root"] is None
