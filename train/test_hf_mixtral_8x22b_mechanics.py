from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hf_mixtral_8x22b_mechanics import (
    MixtralMechanicsError,
    _router_receipt,
    _state_sha256,
    verify_model_manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_tree(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "model"
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "weights.bin").write_bytes(b"weights")
    manifest = root / "SHA256SUMS"
    manifest.write_text(
        f"{_sha(root / 'config.json')}  config.json\n"
        f"{_sha(root / 'weights.bin')}  weights.bin\n"
    )
    return root, manifest, _sha(manifest)


def test_model_manifest_requires_exact_bytes_and_membership(tmp_path: Path) -> None:
    root, manifest, digest = _model_tree(tmp_path)
    receipt = verify_model_manifest(root, manifest, digest)
    assert receipt == {
        "manifest_sha256": digest,
        "manifest_entries": 2,
        "covered_bytes": 9,
        "exact_membership": True,
    }

    (root / "extra").write_text("drift")
    with pytest.raises(MixtralMechanicsError, match="membership differs"):
        verify_model_manifest(root, manifest, digest)


def test_model_manifest_rejects_content_or_path_tamper(tmp_path: Path) -> None:
    root, manifest, digest = _model_tree(tmp_path)
    (root / "weights.bin").write_bytes(b"changed")
    with pytest.raises(MixtralMechanicsError, match="member differs"):
        verify_model_manifest(root, manifest, digest)

    root, manifest, _ = _model_tree(tmp_path / "second")
    manifest.write_text(f"{'0' * 64}  ../escape\n")
    with pytest.raises(MixtralMechanicsError, match="manifest row differs"):
        verify_model_manifest(root, manifest, _sha(manifest))


def test_receipts_hash_exact_bfloat16_storage_bytes() -> None:
    first = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    second = first.clone()
    second[1] = 3.0
    assert _state_sha256({"value": first}) != _state_sha256({"value": second})

    model = SimpleNamespace(
        blocks=[
            SimpleNamespace(base=SimpleNamespace(gate=SimpleNamespace(weight=first)))
        ]
    )
    assert len(_router_receipt(model)) == 64


def test_mechanics_worker_is_score_free_and_two_h100_bound() -> None:
    source = Path(__file__).with_name("hf_mixtral_8x22b_mechanics.py").read_text()
    assert '"score_rows_read": 0' in source
    assert '"benchmark_rows_read": 0' in source
    assert "torch.cuda.device_count() != 2" in source
    assert "set(device_map.values()) != {0, 1}" in source
    assert '"native_router_expert_trainables": 0' in source
    assert '"native_router_unchanged": True' in source
    assert 'bnb_4bit_quant_type="nf4"' in source
    assert "bnb_4bit_use_double_quant=True" in source


def test_sbatch_is_one_node_two_h100_no_requeue_and_no_data() -> None:
    source = (
        Path(__file__).parent / "jobs" / "mixtral_8x22b_mechanics.sbatch"
    ).read_text()
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:2" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --partition=normal" in source
    assert "hf_mixtral_8x22b_mechanics.py" in source
    assert "--data" not in source.casefold()
    assert "--source" not in source.casefold()
