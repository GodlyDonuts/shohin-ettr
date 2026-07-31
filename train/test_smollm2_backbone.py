from pathlib import Path

import pytest
import torch

from model import GPT, GPTConfig, RMSNorm
from smollm2_backbone import (
    SMOLLM2_MODEL_SHA256,
    SmolLM2BackboneError,
    _tensor_mapping,
)


def test_smollm2_tensor_mapping_covers_tied_shohin_model() -> None:
    config = GPTConfig(
        vocab_size=49152,
        n_layer=30,
        n_head=9,
        n_kv_head=3,
        d_model=576,
        d_ff=1536,
        seq_len=8192,
        rope_theta=100000.0,
        qk_norm=False,
    )
    model = GPT(config)
    mapping = _tensor_mapping(config)
    assert set(model.state_dict()) == set(mapping) | {"head.weight"}
    assert len(mapping) == 272
    assert model.head.weight.data_ptr() == model.tok.weight.data_ptr()


def test_external_rms_norm_epsilon_is_explicit_and_complete() -> None:
    model = GPT(GPTConfig(n_layer=2))
    model.set_rms_norm_eps(1e-5)
    norms = [module for module in model.modules() if isinstance(module, RMSNorm)]
    assert len(norms) == 9
    assert {module.eps for module in norms} == {1e-5}
    with pytest.raises(ValueError, match="finite and positive"):
        model.set_rms_norm_eps(float("nan"))


def test_import_rejects_missing_source_before_tensor_allocation(
    tmp_path: Path,
) -> None:
    from smollm2_backbone import import_smollm2_135m

    with pytest.raises(SmolLM2BackboneError, match="paths or hashes"):
        import_smollm2_135m(
            tmp_path,
            tokenizer_sha256="a" * 64,
            expected_model_sha256=SMOLLM2_MODEL_SHA256,
            dtype=torch.bfloat16,
        )
