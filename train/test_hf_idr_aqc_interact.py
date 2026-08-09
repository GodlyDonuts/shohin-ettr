#!/usr/bin/env python3
"""Tests for the complete draft/revision/commit interaction path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hf_idr_aqc_interact import (
    IDRAQCInteractionError,
    exact_revision_prompt,
    validate_commit_payload,
    verify_release,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualified_modes_use_frozen_revision_prompt() -> None:
    math = exact_revision_prompt("Find x.", "x=2", "math")
    code = exact_revision_prompt("Write f.", "def f(): pass", "code")
    assert math.startswith("Solve the original problem by checking and revising")
    assert "exact final answer in \\boxed{}" in math
    assert "only executable Python code" in code
    assert math.count("Find x.") == 2


def test_general_mode_is_explicit_fallback() -> None:
    prompt = exact_revision_prompt("Explain.", "Draft.", "general")
    assert "complete corrected answer" in prompt
    with pytest.raises(IDRAQCInteractionError, match="unsupported"):
        exact_revision_prompt("q", "d", "unknown")


def test_commit_payload_requires_qualified_exact_binding() -> None:
    draft_sha = "1" * 64
    commit_sha = "2" * 64
    metadata = {
        "arm": "antisymmetric",
        "adapter_checkpoint_sha256": draft_sha,
    }
    payload = {
        "schema": "shohin-aqc1-commit-model-v1",
        "metadata": metadata,
        "backbone_state": {"a": 1},
        "head_state": {"b": 2},
    }
    report = {
        "schema": "shohin-aqc1-commit-report-v1",
        "status": "complete",
        "holdout_gate_pass": True,
        "arm": "antisymmetric",
        "checkpoint_sha256": commit_sha,
        "adapter_checkpoint_sha256": draft_sha,
    }
    assert validate_commit_payload(payload, report, draft_sha, commit_sha) == metadata
    report["holdout_gate_pass"] = False
    with pytest.raises(IDRAQCInteractionError, match="not qualified"):
        validate_commit_payload(payload, report, draft_sha, commit_sha)


def test_release_verification_rejects_tamper(tmp_path: Path) -> None:
    release = tmp_path / "release"
    model = tmp_path / "model"
    release.mkdir()
    model.mkdir()
    config = model / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    payload = release / "payload.bin"
    payload.write_bytes(b"payload")
    manifest = {
        "schema": "shohin-idr-aqc-release-v1",
        "status": "qualified",
        "model_config_sha256": sha(config),
        "model_revision": "r",
        "inference_stages": [
            "internal_draft",
            "trained_revision",
            "unchanged_continuation",
            "whole_trajectory_commit",
        ],
        "files": {"payload.bin": sha(payload)},
    }
    manifest_path = release / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    sums = {
        "manifest.json": sha(manifest_path),
        "payload.bin": sha(payload),
    }
    (release / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
        encoding="utf-8",
    )
    assert verify_release(release, model)["model_revision"] == "r"
    payload.write_bytes(b"changed")
    with pytest.raises(IDRAQCInteractionError, match="SHA-256 differs"):
        verify_release(release, model)
