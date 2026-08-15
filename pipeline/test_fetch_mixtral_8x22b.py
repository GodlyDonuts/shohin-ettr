"""Tests for the exact Mixtral-8x22B acquisition boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fetch_mixtral_8x22b as fetch


def _info(revision: str = fetch.MODEL_REVISION) -> SimpleNamespace:
    return SimpleNamespace(
        sha=revision,
        siblings=[
            SimpleNamespace(rfilename="config.json", lfs=None),
            SimpleNamespace(
                rfilename="weight.safetensors",
                lfs=SimpleNamespace(sha256="a" * 64, size=7),
            ),
        ],
    )


def _tiny_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "EXPECTED_SIBLINGS", 2)
    monkeypatch.setattr(fetch, "SUPPORT_MEMBERS", {"config.json"})
    monkeypatch.setattr(fetch, "WEIGHT_MEMBERS", {"weight.safetensors"})
    monkeypatch.setattr(
        fetch, "EXPECTED_MEMBERS", {"config.json", "weight.safetensors"}
    )
    monkeypatch.setattr(fetch, "EXPECTED_WEIGHT_BYTES", 7)
    monkeypatch.setattr(fetch, "EXPECTED_LFS_BYTES", 7)


def test_official_model_info_binds_revision_membership_and_lfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tiny_contract(monkeypatch)
    assert fetch._sibling_receipt(_info()) == {
        "config.json": (None, None),
        "weight.safetensors": ("a" * 64, 7),
    }
    with pytest.raises(fetch.MixtralAcquisitionError):
        fetch._sibling_receipt(_info("b" * 40))


def test_seal_snapshot_builds_exact_manifest_and_atomic_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_contract(monkeypatch)
    stage = tmp_path / ".model.partial.1"
    output = tmp_path / "model"
    stage.mkdir()
    config = b'{"model_type":"mixtral"}\n'
    weight = b"weights"
    (stage / "config.json").write_bytes(config)
    (stage / "weight.safetensors").write_bytes(weight)
    monkeypatch.setattr(
        fetch, "MODEL_CONFIG_SHA256", fetch.sha256_file(stage / "config.json")
    )
    rows = {
        "config.json": (None, None),
        "weight.safetensors": (
            fetch.sha256_file(stage / "weight.safetensors"),
            len(weight),
        ),
    }
    receipt = fetch.seal_snapshot(stage, output, rows)
    assert receipt["manifest_entries"] == 3
    assert receipt["exact_membership"] is True
    assert not stage.exists()
    assert (output / "SOURCE_REVISION").read_text().strip() == fetch.MODEL_REVISION
    assert len((output / "SHA256SUMS").read_text().splitlines()) == 3


def test_fetch_job_is_cpu_only_quota_gated_and_nonrequeueing() -> None:
    source = (
        Path(__file__)
        .with_name("jobs")
        .joinpath("fetch_mixtral_8x22b.sbatch")
        .read_text()
    )
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --gres" not in source
    assert "minimum_headroom_kib=300000000" in source
    assert "lfs quota -u sa305415 /lustre/fs1" in source
    assert "fetch_mixtral_8x22b.py" in source
    assert "sha256sum -c SHA256SUMS" in source
    assert 'chmod -R a-w "$OUTPUT" "$REPORT"' in source
    assert source.index("q36_init_local_tmp") < source.index(
        '[[ -d "$SLURM_TMPDIR" && ! -L "$SLURM_TMPDIR" ]]'
    )
    assert '[[ -n "${SLURM_TMPDIR:-}"' not in source


def test_preparation_receipt_does_not_claim_a_download() -> None:
    payload = json.loads(
        Path(__file__)
        .parents[1]
        .joinpath(
            "docs/research/Q36_MIXTRAL_8X22B_UPWARD_MOE_PREPARATION_20260815.json"
        )
        .read_text()
    )
    assert payload["execution_admission"]["download_job"] is None
    assert payload["execution_admission"]["scientific_result"] is False
