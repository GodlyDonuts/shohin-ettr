from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import save_file
import pytest
import torch
from torch import nn

from eval_ettr_component_assembly import (
    ETTRComponentAssemblyError,
    _gates,
    _validate_args,
    load_hash_bound_component,
)


def _arguments(tmp_path: Path) -> SimpleNamespace:
    paths = {
        name: (tmp_path / name).resolve()
        for name in (
            "release",
            "data",
            "tokenizer",
            "protected",
            "contract",
            "compiler",
            "reactor",
            "reader",
            "output",
        )
    }
    return SimpleNamespace(
        release_root=paths["release"],
        release_sha256="a" * 64,
        data_root=paths["data"],
        tokenizer=paths["tokenizer"],
        protected_checkpoint=paths["protected"],
        run_contract=paths["contract"],
        run_contract_sha256="b" * 64,
        compiler=paths["compiler"],
        compiler_sha256="c" * 64,
        reactor=paths["reactor"],
        reactor_sha256="d" * 64,
        query_reader=paths["reader"],
        query_reader_sha256="e" * 64,
        output=paths["output"],
        source_commit="f" * 40,
        architecture_seed=1,
        data_seed=2,
        max_batches=4,
    )


def test_component_arguments_bind_absolute_paths_and_hashes(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    _validate_args(arguments)
    arguments.compiler_sha256 = "wrong"
    with pytest.raises(
        ETTRComponentAssemblyError,
        match="arguments differ",
    ):
        _validate_args(arguments)


def test_hash_bound_component_load_is_strict_and_immutable(
    tmp_path: Path,
) -> None:
    source = nn.Linear(3, 2)
    destination = nn.Linear(3, 2)
    path = (tmp_path / "component.safetensors").resolve()
    save_file(source.state_dict(), path)
    path.chmod(0o444)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        load_hash_bound_component(
            destination,
            path,
            expected_sha256=digest,
            label="compiler",
        )
        == digest
    )
    for name, value in source.state_dict().items():
        assert torch.equal(value, destination.state_dict()[name])
    with pytest.raises(
        ETTRComponentAssemblyError,
        match="hash differs",
    ):
        load_hash_bound_component(
            destination,
            path,
            expected_sha256="0" * 64,
            label="compiler",
        )
    path.chmod(0o644)
    with pytest.raises(
        ETTRComponentAssemblyError,
        match="mutable",
    ):
        load_hash_bound_component(
            destination,
            path,
            expected_sha256=digest,
            label="compiler",
        )


def test_assembly_gate_requires_both_causal_margins_and_paired_gain() -> None:
    raw = {"query_binding_margin_rates": {"command": 0.0, "world": 0.1}}
    candidate = {
        "query_binding_margin_rates": {"command": 0.2, "world": 0.3}
    }
    paired = {
        "total": {"improved_with_upper_95_below_zero": True}
    }
    assert all(_gates(raw, candidate, paired).values())
    candidate["query_binding_margin_rates"]["world"] = 0.05
    assert not _gates(raw, candidate, paired)[
        "world_query_margin_rate_increased"
    ]
