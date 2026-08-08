"""Role-switchable LoRA states for Shohin's shared draft/revision trunk."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT


ROLE_ADAPTER_SCHEMA = "shohin-native-role-lora-v1"
VALID_ROLES = ("draft", "revision")
TARGET_MODULES = (
    "attn.q",
    "attn.k",
    "attn.v",
    "attn.o",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
)


class NativeRoleLoRAError(RuntimeError):
    """Native role adapter geometry or state differs."""


@dataclass(frozen=True)
class NativeRoleLoRAConfig:
    layers: int = 4
    rank: int = 8
    alpha: float = 16.0
    roles: tuple[str, ...] = VALID_ROLES
    target_modules: tuple[str, ...] = TARGET_MODULES

    def validate(self, model: GPT) -> None:
        if self.layers <= 0 or self.layers > model.cfg.n_layer:
            raise NativeRoleLoRAError("adapter layer count is outside the trunk")
        if self.rank <= 0:
            raise NativeRoleLoRAError("adapter rank must be positive")
        if not torch.isfinite(torch.tensor(self.alpha)) or self.alpha <= 0:
            raise NativeRoleLoRAError("adapter alpha must be finite and positive")
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise NativeRoleLoRAError("adapter roles must be unique and nonempty")
        if any(not role or "." in role for role in self.roles):
            raise NativeRoleLoRAError("adapter role name is invalid")
        if set(self.target_modules) != set(TARGET_MODULES):
            raise NativeRoleLoRAError("native role target-module set differs")


class LowRankDelta(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, in_features))
        self.b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.a), self.b) * self.scaling


class RoleLoRALinear(nn.Module):
    """One frozen linear projection with disjoint episode-role deltas."""

    def __init__(self, base: nn.Linear, config: NativeRoleLoRAConfig):
        super().__init__()
        if base.bias is not None:
            raise NativeRoleLoRAError("Shohin role LoRA expects bias-free projections")
        self.base = base
        self.base.requires_grad_(False)
        self.roles = nn.ModuleDict(
            {
                role: LowRankDelta(
                    base.in_features,
                    base.out_features,
                    config.rank,
                    config.alpha,
                )
                for role in config.roles
            }
        )
        self.active_role: str | None = None

    def set_active_role(self, role: str | None) -> None:
        if role is not None and role not in self.roles:
            raise NativeRoleLoRAError(f"unknown adapter role: {role}")
        self.active_role = role

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self.active_role is not None:
            output = output + self.roles[self.active_role](x)
        return output


def _resolve_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_role_lora(model: nn.Module) -> Iterable[tuple[str, RoleLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, RoleLoRALinear):
            yield name, module


def attach_role_lora(model: GPT, config: NativeRoleLoRAConfig) -> int:
    """Attach both temporal roles after loading a normal pretrained checkpoint."""
    config.validate(model)
    if any(True for _ in iter_role_lora(model)):
        raise NativeRoleLoRAError("native role LoRA is already attached")
    model.requires_grad_(False)
    first_layer = model.cfg.n_layer - config.layers
    count = 0
    for layer in range(first_layer, model.cfg.n_layer):
        block = model.blocks[layer]
        for relative in config.target_modules:
            parent, leaf = _resolve_parent(block, relative)
            base = getattr(parent, leaf)
            if not isinstance(base, nn.Linear):
                raise NativeRoleLoRAError(
                    f"adapter target is not a linear projection: {layer}.{relative}"
                )
            setattr(parent, leaf, RoleLoRALinear(base, config))
            count += 1
    expected = config.layers * len(config.target_modules)
    if count != expected:
        raise NativeRoleLoRAError("native role adapter inventory differs")
    set_active_role(model, None)
    return count


def set_active_role(model: nn.Module, role: str | None) -> None:
    modules = list(iter_role_lora(model))
    if not modules:
        raise NativeRoleLoRAError("native role LoRA is not attached")
    for _, module in modules:
        module.set_active_role(role)


def set_trainable_role(model: nn.Module, role: str) -> int:
    """Freeze the shared trunk and the other temporal role."""
    modules = list(iter_role_lora(model))
    if not modules:
        raise NativeRoleLoRAError("native role LoRA is not attached")
    model.requires_grad_(False)
    parameter_count = 0
    for _, module in modules:
        if role not in module.roles:
            raise NativeRoleLoRAError(f"unknown adapter role: {role}")
        module.roles[role].requires_grad_(True)
        parameter_count += sum(p.numel() for p in module.roles[role].parameters())
    set_active_role(model, role)
    return parameter_count


def export_role_adapter(
    model: GPT,
    config: NativeRoleLoRAConfig,
    role: str,
    *,
    base_checkpoint_sha256: str,
) -> dict[str, Any]:
    modules = list(iter_role_lora(model))
    expected = config.layers * len(config.target_modules)
    if len(modules) != expected:
        raise NativeRoleLoRAError("native role adapter inventory differs")
    state: dict[str, torch.Tensor] = {}
    for name, module in modules:
        if role not in module.roles:
            raise NativeRoleLoRAError(f"unknown adapter role: {role}")
        for parameter_name, value in module.roles[role].state_dict().items():
            state[f"{name}.roles.{role}.{parameter_name}"] = value.detach().cpu()
    return {
        "schema": ROLE_ADAPTER_SCHEMA,
        "role": role,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "model_config": asdict(model.cfg),
        "adapter_config": asdict(config),
        "state": state,
    }


def load_role_adapter(
    model: GPT,
    payload: dict[str, Any],
    config: NativeRoleLoRAConfig,
    *,
    expected_role: str,
    expected_base_checkpoint_sha256: str,
) -> None:
    if payload.get("schema") != ROLE_ADAPTER_SCHEMA:
        raise NativeRoleLoRAError("native role adapter schema differs")
    if payload.get("role") != expected_role:
        raise NativeRoleLoRAError("native role adapter role differs")
    if payload.get("base_checkpoint_sha256") != expected_base_checkpoint_sha256:
        raise NativeRoleLoRAError("native role adapter base checkpoint differs")
    if payload.get("model_config") != asdict(model.cfg):
        raise NativeRoleLoRAError("native role adapter model config differs")
    if payload.get("adapter_config") != asdict(config):
        raise NativeRoleLoRAError("native role adapter config differs")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise NativeRoleLoRAError("native role adapter state is missing")

    modules = dict(iter_role_lora(model))
    expected_keys: set[str] = set()
    for name, module in modules.items():
        if expected_role not in module.roles:
            raise NativeRoleLoRAError("native role adapter role is absent")
        local = module.roles[expected_role]
        for parameter_name in local.state_dict():
            expected_keys.add(f"{name}.roles.{expected_role}.{parameter_name}")
    if set(state) != expected_keys:
        raise NativeRoleLoRAError("native role adapter tensor inventory differs")
    for name, module in modules.items():
        prefix = f"{name}.roles.{expected_role}."
        local_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        module.roles[expected_role].load_state_dict(local_state, strict=True)


def role_parameter_count(model: nn.Module, role: str) -> int:
    total = 0
    found = False
    for _, module in iter_role_lora(model):
        if role not in module.roles:
            raise NativeRoleLoRAError(f"unknown adapter role: {role}")
        found = True
        total += sum(parameter.numel() for parameter in module.roles[role].parameters())
    if not found:
        raise NativeRoleLoRAError("native role LoRA is not attached")
    return total


def role_parameter_count_for_config(
    model_config: dict[str, int], adapter_config: NativeRoleLoRAConfig
) -> int:
    """Return one role's exact parameter count without allocating the trunk."""
    required = ("n_layer", "n_head", "n_kv_head", "d_model", "d_ff")
    missing = [key for key in required if key not in model_config]
    if missing:
        raise NativeRoleLoRAError(
            f"model config is missing role-adapter geometry: {', '.join(missing)}"
        )
    n_layer = model_config["n_layer"]
    n_head = model_config["n_head"]
    n_kv_head = model_config["n_kv_head"]
    d_model = model_config["d_model"]
    d_ff = model_config["d_ff"]
    if adapter_config.layers <= 0 or adapter_config.layers > n_layer:
        raise NativeRoleLoRAError("adapter layer count is outside the trunk")
    if adapter_config.rank <= 0:
        raise NativeRoleLoRAError("adapter rank must be positive")
    if d_model % n_head != 0 or n_head % n_kv_head != 0:
        raise NativeRoleLoRAError("model attention geometry is invalid")
    head_dim = d_model // n_head
    per_layer = adapter_config.rank * (
        9 * d_model + 2 * n_kv_head * head_dim + 3 * d_ff
    )
    return adapter_config.layers * per_layer
