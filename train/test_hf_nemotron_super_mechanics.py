"""CPU tests for the score-free Nemotron Super mechanics boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from hf_nemotron_super_mechanics import (
    NemotronSuperMechanicsError,
    _atomic_json,
    _modelopt_fp8_quantization_config,
    _state_sha256,
    install_triton_allocator_compatibility,
    modelopt_fp8_receipt_is_exact,
    verify_manifest,
)


def test_triton_allocator_compatibility_is_exact_and_callable(monkeypatch) -> None:
    import importlib.metadata
    import sys
    from types import ModuleType

    triton = ModuleType("triton")
    monkeypatch.setitem(sys.modules, "triton", triton)
    original = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "3.2.0" if name == "triton" else original(name),
    )
    receipt = install_triton_allocator_compatibility()
    assert receipt == {
        "triton_version": "3.2.0",
        "mode": "triton-3.2-internal-descriptor",
    }
    assert callable(triton.set_allocator)
    triton.set_allocator(lambda *_: None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_verification_binds_order_bytes_and_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a").write_bytes(b"alpha")
    (root / "b").write_bytes(b"beta")
    manifest = root / "SHA256SUMS"
    manifest.write_text(f"{_sha256(root / 'a')}  a\n{_sha256(root / 'b')}  b\n")
    receipt = verify_manifest(root, manifest, _sha256(manifest))
    assert receipt == {
        "manifest_sha256": _sha256(manifest),
        "manifest_entries": 2,
        "covered_bytes": 9,
    }
    (root / "b").write_bytes(b"changed")
    with pytest.raises(NemotronSuperMechanicsError):
        verify_manifest(root, manifest, _sha256(manifest))


def test_manifest_rejects_escape_but_accepts_hash_bound_install_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    member = root / "a"
    member.write_text("a")
    manifest = root / "SHA256SUMS"
    manifest.write_text(f"{_sha256(member)}  ../a\n")
    with pytest.raises(NemotronSuperMechanicsError):
        verify_manifest(root, manifest, _sha256(manifest))
    second = root / "b"
    second.write_text("b")
    manifest.write_text(f"{_sha256(second)}  b\n{_sha256(member)}  a\n")
    assert verify_manifest(root, manifest, _sha256(manifest)) == {
        "manifest_sha256": _sha256(manifest),
        "manifest_entries": 2,
        "covered_bytes": 2,
    }


def test_state_digest_is_order_independent_and_value_sensitive() -> None:
    first = {
        "b": torch.tensor([2.0], dtype=torch.float32),
        "a": torch.tensor([1.0], dtype=torch.float32),
    }
    second = {"a": first["a"].clone(), "b": first["b"].clone()}
    assert _state_sha256(first) == _state_sha256(second)
    second["b"].add_(1.0)
    assert _state_sha256(first) != _state_sha256(second)


def test_atomic_report_is_write_once(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    _atomic_json(output, {"status": "pass"})
    assert output.read_text() == '{\n  "status": "pass"\n}\n'
    with pytest.raises(NemotronSuperMechanicsError):
        _atomic_json(output, {"status": "changed"})


def test_modelopt_loader_config_replays_export_and_disables_exact_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exported = {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                "input_activations": {
                    "dynamic": False,
                    "num_bits": 8,
                    "type": "float",
                },
            }
        },
        "ignore": ["block.linear", "block.*"],
        "producer": {"name": "modelopt", "version": "0.41.0"},
        "quant_algo": "FP8",
        "quant_method": "modelopt",
    }
    legacy = {"producer": exported["producer"], "quantization": {"quant_algo": "FP8"}}
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": exported}))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(legacy))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "block.linear.weight": "one.safetensors",
                    "block.linear.weight_scale": "one.safetensors",
                    "block.linear.input_scale": "one.safetensors",
                }
            }
        )
    )
    import hf_nemotron_super_mechanics as mechanics

    monkeypatch.setattr(
        mechanics, "HF_QUANT_CONFIG_SHA256", _sha256(tmp_path / "hf_quant_config.json")
    )
    monkeypatch.setattr(
        mechanics,
        "MODEL_INDEX_SHA256",
        _sha256(tmp_path / "model.safetensors.index.json"),
    )
    monkeypatch.setattr(mechanics, "FP8_LINEAR_COUNT", 1)
    monkeypatch.setattr(mechanics, "MODEL_WEIGHT_MAP_ENTRIES", 3)
    monkeypatch.setattr(mechanics, "MODEL_WEIGHT_SHARDS", 1)
    monkeypatch.setattr(mechanics, "MODELOPT_IGNORE_PATTERNS", 2)

    convert_module = ModuleType("modelopt.torch.export.convert_hf_config")
    convert_module.convert_hf_quant_config_format = lambda payload: exported
    quantization_module = ModuleType("modelopt.torch.quantization")
    quantization_module.FP8_DEFAULT_CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
            "*input_quantizer": {"num_bits": (4, 3), "axis": None},
            "default": {"enable": False},
        },
        "algorithm": "max",
    }
    for name in ("modelopt", "modelopt.torch", "modelopt.torch.export"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "modelopt.torch.export.convert_hf_config", convert_module
    )
    monkeypatch.setitem(sys.modules, "modelopt.torch.quantization", quantization_module)

    config, receipt = _modelopt_fp8_quantization_config(tmp_path)
    assert config["quant_cfg"]["block.linear*"] == {"enable": False}
    assert config["quant_cfg"]["block.*"] == {"enable": False}
    assert receipt["fp8_linear_count"] == 1
    assert receipt["disabled_patterns"] == 2
    assert receipt["quant_gemm"] is True


def test_modelopt_loader_config_rejects_conversion_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exported = {
        "config_groups": {"group_0": {"targets": ["Linear"]}},
        "ignore": [],
        "producer": {"name": "modelopt", "version": "0.41.0"},
        "quant_algo": "FP8",
        "quant_method": "modelopt",
    }
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": exported}))
    (tmp_path / "hf_quant_config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    import hf_nemotron_super_mechanics as mechanics

    monkeypatch.setattr(
        mechanics, "HF_QUANT_CONFIG_SHA256", _sha256(tmp_path / "hf_quant_config.json")
    )
    monkeypatch.setattr(
        mechanics,
        "MODEL_INDEX_SHA256",
        _sha256(tmp_path / "model.safetensors.index.json"),
    )
    convert_module = ModuleType("modelopt.torch.export.convert_hf_config")
    convert_module.convert_hf_quant_config_format = lambda payload: {"changed": True}
    for name in ("modelopt", "modelopt.torch", "modelopt.torch.export"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "modelopt.torch.export.convert_hf_config", convert_module
    )
    with pytest.raises(NemotronSuperMechanicsError, match="converted"):
        _modelopt_fp8_quantization_config(tmp_path)


def test_modelopt_runtime_receipt_rejects_cpu_or_missing_fp8() -> None:
    import hf_nemotron_super_mechanics as mechanics

    payload = {
        "export": {
            "hf_quant_config_sha256": mechanics.HF_QUANT_CONFIG_SHA256,
            "model_index_sha256": mechanics.MODEL_INDEX_SHA256,
            "fp8_linear_count": mechanics.FP8_LINEAR_COUNT,
            "weight_map_entries": mechanics.MODEL_WEIGHT_MAP_ENTRIES,
            "weight_shards": mechanics.MODEL_WEIGHT_SHARDS,
            "disabled_patterns": mechanics.MODELOPT_IGNORE_PATTERNS,
            "quant_gemm": True,
        },
        "runtime": {
            "real_quant_gemm_enabled": True,
            "real_fp8_linear_count": mechanics.FP8_LINEAR_COUNT,
            "cpu_tensors": 0,
            "disk_tensors": 0,
            "meta_tensors": 0,
            "parameter_devices": {"cuda:0": 10, "cuda:1": 10},
            "buffer_devices": {"cuda:0": 2, "cuda:1": 2},
        },
    }
    assert modelopt_fp8_receipt_is_exact(payload)
    payload["runtime"]["cpu_tensors"] = 1
    assert not modelopt_fp8_receipt_is_exact(payload)
    payload["runtime"]["cpu_tensors"] = 0
    payload["runtime"]["real_fp8_linear_count"] -= 1
    assert not modelopt_fp8_receipt_is_exact(payload)
