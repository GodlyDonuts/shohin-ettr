"""CPU tests for the score-free Nemotron Super mechanics boundary."""

from __future__ import annotations

from contextlib import contextmanager
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
    _checkpoint_translation_receipt,
    _modelopt_fp8_quantization_config,
    _state_sha256,
    _translate_export_checkpoint_keys,
    install_triton_allocator_compatibility,
    load_modelopt_fp8_backbone,
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
        "remote_model": {
            "configuration_sha256": mechanics.REMOTE_CONFIGURATION_SHA256,
            "modeling_sha256": mechanics.REMOTE_MODELING_SHA256,
            "model_class": "frozen.NemotronHForCausalLM",
        },
        "checkpoint_translation": {
            "backbone_to_model": mechanics.BACKBONE_WEIGHT_MAP_ENTRIES,
            "mtp_ignored": mechanics.MTP_WEIGHT_MAP_ENTRIES,
            "lm_head_unchanged": mechanics.LM_HEAD_WEIGHT_MAP_ENTRIES,
            "source_prefix": "backbone.",
            "target_prefix": "model.",
            "mtp_policy": "ignored_not_implemented_by_remote_causal_lm",
            "input_scale_to_amax": mechanics.FP8_LINEAR_COUNT,
            "weight_scale_to_amax": mechanics.FP8_LINEAR_COUNT,
            "input_amax_placeholders": mechanics.FP8_LINEAR_COUNT,
            "weight_amax_placeholders": mechanics.FP8_LINEAR_COUNT,
            "scale_to_amax_multiplier": mechanics.FP8_SCALE_MULTIPLIER,
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


def test_checkpoint_translation_is_streamed_exact_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import accelerate.utils.modeling as accelerate_modeling
    import hf_nemotron_super_mechanics as mechanics

    sentinel = object()
    modelopt_accelerate = ModuleType("modelopt.torch.quantization.plugins.accelerate")
    modelopt_accelerate.load_checkpoint_and_dispatch = (
        lambda model, *args, **kwargs: model
    )
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch.quantization.plugins.accelerate",
        modelopt_accelerate,
    )

    def fake_load(*args, **kwargs):
        return {
            "backbone.layer.weight": sentinel,
            "backbone.layer.input_scale": torch.tensor(2.0),
            "backbone.layer.weight_scale": torch.tensor(3.0),
            "mtp.layer.weight": object(),
            "lm_head.weight": sentinel,
        }

    class Quantizer(torch.nn.Module):
        pass

    class QuantizedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_quantizer = Quantizer()
            self.weight_quantizer.register_buffer(
                "_amax", torch.empty((), dtype=torch.float32, device="meta")
            )
            self.input_quantizer = Quantizer()

    model = torch.nn.Module()
    model.linear = QuantizedLinear()

    monkeypatch.setattr(accelerate_modeling, "load_state_dict", fake_load)
    monkeypatch.setattr(mechanics, "BACKBONE_WEIGHT_MAP_ENTRIES", 3)
    monkeypatch.setattr(mechanics, "MTP_WEIGHT_MAP_ENTRIES", 1)
    monkeypatch.setattr(mechanics, "LM_HEAD_WEIGHT_MAP_ENTRIES", 1)
    monkeypatch.setattr(mechanics, "MODEL_WEIGHT_MAP_ENTRIES", 5)
    monkeypatch.setattr(mechanics, "FP8_LINEAR_COUNT", 1)
    with _translate_export_checkpoint_keys() as counts:
        assert modelopt_accelerate.load_checkpoint_and_dispatch(model) is model
        translated = accelerate_modeling.load_state_dict("one.safetensors")
        assert set(translated) == {
            "model.layer.weight",
            "model.layer.input_quantizer._amax",
            "model.layer.weight_quantizer._amax",
            "lm_head.weight",
        }
        assert translated["model.layer.weight"] is sentinel
        assert translated["lm_head.weight"] is sentinel
        assert translated["model.layer.input_quantizer._amax"].item() == 896.0
        assert translated["model.layer.weight_quantizer._amax"].item() == 1344.0
    assert accelerate_modeling.load_state_dict is fake_load
    receipt = _checkpoint_translation_receipt(counts)
    assert receipt["backbone_to_model"] == 3
    assert receipt["mtp_ignored"] == 1
    assert receipt["lm_head_unchanged"] == 1
    assert receipt["input_amax_placeholders"] == 1
    assert receipt["weight_amax_placeholders"] == 1


def test_checkpoint_translation_rejects_unknown_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import accelerate.utils.modeling as accelerate_modeling

    modelopt_accelerate = ModuleType("modelopt.torch.quantization.plugins.accelerate")
    modelopt_accelerate.load_checkpoint_and_dispatch = (
        lambda model, *args, **kwargs: model
    )
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch.quantization.plugins.accelerate",
        modelopt_accelerate,
    )

    monkeypatch.setattr(
        accelerate_modeling,
        "load_state_dict",
        lambda *args, **kwargs: {"unexpected.weight": object()},
    )
    with _translate_export_checkpoint_keys():
        with pytest.raises(NemotronSuperMechanicsError, match="namespace"):
            accelerate_modeling.load_state_dict("one.safetensors")


def test_loader_registers_the_hash_bound_remote_model_before_modelopt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hf_nemotron_super_mechanics as mechanics

    quantization = {"quant_method": "modelopt"}
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": quantization})
    )
    configuration = tmp_path / "configuration_nemotron_h.py"
    modeling = tmp_path / "modeling_nemotron_h.py"
    configuration.write_text("configuration")
    modeling.write_text("modeling")
    monkeypatch.setattr(
        mechanics, "REMOTE_CONFIGURATION_SHA256", _sha256(configuration)
    )
    monkeypatch.setattr(mechanics, "REMOTE_MODELING_SHA256", _sha256(modeling))
    monkeypatch.setattr(
        mechanics,
        "_modelopt_fp8_quantization_config",
        lambda root: ({"loader": "fp8"}, {"source": "exact"}),
    )
    monkeypatch.setattr(
        mechanics,
        "_modelopt_fp8_runtime_receipt",
        lambda model: {"runtime": "exact"},
    )

    @contextmanager
    def fake_translation():
        yield mechanics.Counter()

    monkeypatch.setattr(
        mechanics, "_translate_export_checkpoint_keys", fake_translation
    )
    monkeypatch.setattr(
        mechanics,
        "_checkpoint_translation_receipt",
        lambda counts: {"translation": "exact"},
    )

    class RemoteConfig:
        def __init__(self) -> None:
            self.quantization_config = quantization
            self.auto_map = {
                "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"
            }

    class RemoteModel:
        pass

    sentinel = object()
    events: list[object] = []

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            events.append(("config", args, kwargs))
            return RemoteConfig()

    class AutoModel:
        @classmethod
        def register(cls, config_class, model_class, *, exist_ok):
            events.append(("register", config_class, model_class, exist_ok))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            events.append(("load", args, kwargs))
            return sentinel

    @contextmanager
    def init_quantized_weights(config, *, gpu_mem_percentage, quant_gemm):
        events.append(("context", config, gpu_mem_percentage, quant_gemm))
        yield

    transformers = ModuleType("transformers")
    transformers.AutoConfig = AutoConfig
    transformers.AutoModelForCausalLM = AutoModel
    dynamic = ModuleType("transformers.dynamic_module_utils")
    dynamic.get_class_from_dynamic_module = lambda *args, **kwargs: RemoteModel
    accelerate = ModuleType("modelopt.torch.quantization.plugins.accelerate")
    accelerate.init_quantized_weights = init_quantized_weights
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", dynamic)
    for name in (
        "modelopt",
        "modelopt.torch",
        "modelopt.torch.quantization",
        "modelopt.torch.quantization.plugins",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "modelopt.torch.quantization.plugins.accelerate", accelerate
    )
    monkeypatch.setattr(mechanics.inspect, "getfile", lambda value: str(modeling))

    model, receipt = load_modelopt_fp8_backbone(tmp_path)
    assert model is sentinel
    assert events[1] == ("register", RemoteConfig, RemoteModel, True)
    assert events[2] == ("context", {"loader": "fp8"}, 0.95, True)
    assert events[3][0] == "load"
    assert events[3][2]["config"].torch_dtype == torch.bfloat16
    assert receipt["remote_model"]["model_class"].endswith(".RemoteModel")
