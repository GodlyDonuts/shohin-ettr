#!/usr/bin/env python3
"""Tests for Shohin's shared-trunk draft/revision role states."""

import copy

import pytest
import torch

from model import GPT, GPTConfig
from native_role_lora import (
    NativeRoleLoRAConfig,
    NativeRoleLoRAError,
    attach_role_lora,
    export_role_adapter,
    iter_role_lora,
    load_role_adapter,
    role_parameter_count,
    role_parameter_count_for_config,
    set_active_role,
    set_trainable_role,
)
from train import CONFIGS


def tiny_model() -> GPT:
    torch.manual_seed(7)
    return GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=3,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=16,
            zloss=0.0,
        )
    )


def test_zero_initialized_roles_preserve_shared_trunk():
    model = tiny_model().eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        baseline, _ = model(tokens)
    config = NativeRoleLoRAConfig(layers=2, rank=4, alpha=8)
    assert attach_role_lora(model, config) == 14
    for role in (None, "draft", "revision"):
        set_active_role(model, role)
        with torch.no_grad():
            observed, _ = model(tokens)
        torch.testing.assert_close(observed, baseline, rtol=0, atol=0)


def test_roles_are_isolated_and_trainable_inventory_is_exact():
    model = tiny_model().eval()
    config = NativeRoleLoRAConfig(layers=1, rank=2, alpha=4)
    attach_role_lora(model, config)
    tokens = torch.tensor([[2, 4, 6]])
    set_active_role(model, "draft")
    with torch.no_grad():
        draft_before, _ = model(tokens)
    first = next(iter(iter_role_lora(model)))[1]
    first.roles["revision"].b.data.fill_(0.1)
    set_active_role(model, "revision")
    with torch.no_grad():
        revision_after, _ = model(tokens)
    set_active_role(model, "draft")
    with torch.no_grad():
        draft_after, _ = model(tokens)
    assert not torch.equal(revision_after, draft_before)
    torch.testing.assert_close(draft_after, draft_before, rtol=0, atol=0)

    count = set_trainable_role(model, "revision")
    assert count == role_parameter_count(model, "revision")
    assert count > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (".roles.revision." in name)


def test_role_state_round_trip_and_base_binding():
    model = tiny_model()
    config = NativeRoleLoRAConfig(layers=2, rank=3, alpha=6)
    attach_role_lora(model, config)
    for _, module in iter_role_lora(model):
        module.roles["revision"].b.data.normal_()
    payload = export_role_adapter(
        model, config, "revision", base_checkpoint_sha256="a" * 64
    )

    restored = tiny_model()
    attach_role_lora(restored, config)
    load_role_adapter(
        restored,
        copy.deepcopy(payload),
        config,
        expected_role="revision",
        expected_base_checkpoint_sha256="a" * 64,
    )
    source_state = export_role_adapter(
        model, config, "revision", base_checkpoint_sha256="a" * 64
    )["state"]
    restored_state = export_role_adapter(
        restored, config, "revision", base_checkpoint_sha256="a" * 64
    )["state"]
    assert source_state.keys() == restored_state.keys()
    for key in source_state:
        torch.testing.assert_close(source_state[key], restored_state[key])

    with pytest.raises(NativeRoleLoRAError, match="base checkpoint differs"):
        load_role_adapter(
            restored,
            payload,
            config,
            expected_role="revision",
            expected_base_checkpoint_sha256="b" * 64,
        )


def test_invalid_or_duplicate_attachment_fails_closed():
    model = tiny_model()
    with pytest.raises(NativeRoleLoRAError, match="outside the trunk"):
        attach_role_lora(model, NativeRoleLoRAConfig(layers=4))
    config = NativeRoleLoRAConfig(layers=1)
    attach_role_lora(model, config)
    with pytest.raises(NativeRoleLoRAError, match="already attached"):
        attach_role_lora(model, config)


def test_scale_role_parameter_receipts_without_model_allocation():
    config = NativeRoleLoRAConfig(layers=4, rank=8, alpha=16)
    assert role_parameter_count_for_config(CONFIGS["shohin_390m"], config) == 581_632
    assert role_parameter_count_for_config(CONFIGS["shohin_920m"], config) == 892_928


def test_config_parameter_receipt_matches_attached_model():
    model = tiny_model()
    config = NativeRoleLoRAConfig(layers=2, rank=3, alpha=6)
    attach_role_lora(model, config)
    assert role_parameter_count_for_config(vars(model.cfg), config) == role_parameter_count(
        model, "draft"
    )


def test_config_parameter_receipt_fails_closed():
    config = NativeRoleLoRAConfig(layers=4)
    with pytest.raises(NativeRoleLoRAError, match="missing"):
        role_parameter_count_for_config({}, config)
    with pytest.raises(NativeRoleLoRAError, match="outside the trunk"):
        role_parameter_count_for_config(CONFIGS["shohin_390m"], config.__class__(layers=99))
