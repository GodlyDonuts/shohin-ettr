"""Strict import of the official SmolLM2-135M backbone into Shohin GPT.

SmolLM2-135M and Shohin share the same deep-thin transformer geometry. This
module maps the official Llama-style tensor names into Shohin's implementation
without changing tensor values. It is an experimental cross-backbone control,
not a replacement for the protected Shohin checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

from safetensors import safe_open
import torch

from model import GPT, GPTConfig


SMOLLM2_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM2_REVISION = "83212e1e2b3cfd6958f3707877bb878945dea8ee"
SMOLLM2_MODEL_SHA256 = (
    "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
)
SMOLLM2_PARAMETER_COUNT = 134_515_008
IMPORT_SCHEMA = "shohin-smollm2-135m-backbone-import-v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class SmolLM2BackboneError(ValueError):
    """The external backbone does not match the admitted control."""


@dataclass(frozen=True, slots=True)
class SmolLM2BackboneReceipt:
    schema: str
    model_id: str
    revision: str
    source_config_sha256: str
    source_model_sha256: str
    source_tokenizer_sha256: str
    rms_norm_eps: float
    base_parameters: int
    base_config: dict[str, object]
    tensor_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_config(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmolLM2BackboneError("SmolLM2 config is unreadable") from exc
    if not isinstance(value, Mapping):
        raise SmolLM2BackboneError("SmolLM2 config must be an object")
    return value


def _require_exact_smollm2_config(value: Mapping[str, object]) -> GPTConfig:
    required = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_size": 576,
        "intermediate_size": 1536,
        "max_position_embeddings": 8192,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 9,
        "num_hidden_layers": 30,
        "num_key_value_heads": 3,
        "rms_norm_eps": 1e-5,
        "rope_interleaved": False,
        "rope_scaling": None,
        "rope_theta": 100000,
        "tie_word_embeddings": True,
        "vocab_size": 49152,
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise SmolLM2BackboneError(
                f"SmolLM2 configuration field differs: {name}"
            )
    epsilon = value["rms_norm_eps"]
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (float, int))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise SmolLM2BackboneError("SmolLM2 RMSNorm epsilon differs")
    return GPTConfig(
        vocab_size=int(value["vocab_size"]),
        n_layer=int(value["num_hidden_layers"]),
        n_head=int(value["num_attention_heads"]),
        n_kv_head=int(value["num_key_value_heads"]),
        d_model=int(value["hidden_size"]),
        d_ff=int(value["intermediate_size"]),
        seq_len=int(value["max_position_embeddings"]),
        rope_theta=float(value["rope_theta"]),
        qk_norm=False,
        tie_embeddings=True,
    )


def _tensor_mapping(config: GPTConfig) -> dict[str, str]:
    mapping = {
        "tok.weight": "model.embed_tokens.weight",
        "norm.w": "model.norm.weight",
    }
    for index in range(config.n_layer):
        source = f"model.layers.{index}."
        target = f"blocks.{index}."
        mapping.update(
            {
                target + "n1.w": source + "input_layernorm.weight",
                target + "n2.w": source + "post_attention_layernorm.weight",
                target + "attn.q.weight": source + "self_attn.q_proj.weight",
                target + "attn.k.weight": source + "self_attn.k_proj.weight",
                target + "attn.v.weight": source + "self_attn.v_proj.weight",
                target + "attn.o.weight": source + "self_attn.o_proj.weight",
                target + "mlp.gate.weight": source + "mlp.gate_proj.weight",
                target + "mlp.up.weight": source + "mlp.up_proj.weight",
                target + "mlp.down.weight": source + "mlp.down_proj.weight",
            }
        )
    return mapping


def import_smollm2_135m(
    source_root: Path,
    *,
    tokenizer_sha256: str,
    expected_model_sha256: str = SMOLLM2_MODEL_SHA256,
    dtype: torch.dtype = torch.float32,
) -> tuple[GPT, SmolLM2BackboneReceipt]:
    """Load the exact admitted official checkpoint into Shohin's GPT."""

    source_root = source_root.expanduser().resolve()
    config_path = source_root / "config.json"
    model_path = source_root / "model.safetensors"
    if (
        not source_root.is_dir()
        or not config_path.is_file()
        or not model_path.is_file()
        or _HEX64.fullmatch(expected_model_sha256) is None
        or _HEX64.fullmatch(tokenizer_sha256) is None
    ):
        raise SmolLM2BackboneError("SmolLM2 import paths or hashes differ")
    observed_model_sha256 = file_sha256(model_path)
    if observed_model_sha256 != expected_model_sha256:
        raise SmolLM2BackboneError("SmolLM2 model SHA-256 differs")
    config_value = _read_config(config_path)
    config = _require_exact_smollm2_config(config_value)
    model = GPT(config).to(dtype=dtype)
    model.set_rms_norm_eps(float(config_value["rms_norm_eps"]))
    mapping = _tensor_mapping(config)
    destination = model.state_dict()
    expected_destination = set(mapping) | {"head.weight"}
    if set(destination) != expected_destination:
        raise SmolLM2BackboneError("Shohin destination tensor inventory differs")
    with safe_open(model_path, framework="pt", device="cpu") as source:
        if set(source.keys()) != set(mapping.values()):
            raise SmolLM2BackboneError("SmolLM2 source tensor inventory differs")
        with torch.no_grad():
            for target_name, source_name in mapping.items():
                source_tensor = source.get_tensor(source_name)
                target_tensor = destination[target_name]
                if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                    raise SmolLM2BackboneError(
                        f"SmolLM2 tensor shape differs: {source_name}"
                    )
                target_tensor.copy_(source_tensor.to(dtype=dtype))
    if model.head.weight.data_ptr() != model.tok.weight.data_ptr():
        raise SmolLM2BackboneError("SmolLM2 tied embedding identity differs")
    parameters = model.num_params()
    if parameters != SMOLLM2_PARAMETER_COUNT:
        raise SmolLM2BackboneError("SmolLM2 parameter count differs")
    receipt = SmolLM2BackboneReceipt(
        schema=IMPORT_SCHEMA,
        model_id=SMOLLM2_MODEL_ID,
        revision=SMOLLM2_REVISION,
        source_config_sha256=file_sha256(config_path),
        source_model_sha256=observed_model_sha256,
        source_tokenizer_sha256=tokenizer_sha256,
        rms_norm_eps=float(config_value["rms_norm_eps"]),
        base_parameters=parameters,
        base_config=asdict(config),
        tensor_count=len(mapping),
    )
    model.eval()
    return model, receipt


__all__ = [
    "IMPORT_SCHEMA",
    "SMOLLM2_MODEL_ID",
    "SMOLLM2_MODEL_SHA256",
    "SMOLLM2_PARAMETER_COUNT",
    "SMOLLM2_REVISION",
    "SmolLM2BackboneError",
    "SmolLM2BackboneReceipt",
    "file_sha256",
    "import_smollm2_135m",
]
