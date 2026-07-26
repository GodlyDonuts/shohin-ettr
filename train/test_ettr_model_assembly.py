from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    SYSTEM_PARAMETER_CAP,
    TheoryReactorConfig,
)
from ettr_model_assembly import (
    ETTR_MODEL_ASSEMBLY_SCHEMA,
    ETTRModelAssemblyError,
    ETTRModelAssemblyReceipt,
)
from ettr_qualification import _model_sha256
from model import GPT, GPTConfig


def _model(seed: int = 2026072601) -> EndogenousTypedTheoryReactorGPT:
    torch.manual_seed(seed)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=32,
            zloss=0.0,
        )
    )
    return EndogenousTypedTheoryReactorGPT(
        base,
        TheoryReactorConfig(
            d_model=32,
            state_width=32,
            num_slots=6,
            num_types=3,
            num_relations=3,
            num_value_codes=64,
            max_edges=96,
            num_heads=4,
            compiler_layers=1,
            reactor_layers=1,
            query_layers=1,
            ff_multiplier=2,
            max_steps=6,
            stage_after_block=1,
            parameter_cap=1_000_000,
        ),
    ).eval()


def _write_canonical(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    path.chmod(0o444)


def _save_component(path: Path, module: torch.nn.Module) -> None:
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in module.state_dict().items()
        },
        path,
    )
    path.chmod(0o444)


def _assembly(
    tmp_path: Path,
    *,
    seed: int = 2026072601,
    prefix: str = "",
) -> tuple[EndogenousTypedTheoryReactorGPT, dict[str, Path]]:
    model = _model(seed)
    paths = {
        "config": tmp_path / f"{prefix}config.json",
        "checkpoint": tmp_path / f"{prefix}base.pt",
        "compiler": tmp_path / f"{prefix}compiler.safetensors",
        "reactor": tmp_path / f"{prefix}reactor.safetensors",
        "query_reader": tmp_path / f"{prefix}query_reader.safetensors",
    }
    _write_canonical(paths["config"], asdict(model.config))
    torch.save(
        {
            "cfg": asdict(model.base.cfg),
            "model": model.base.state_dict(),
            "step": 123,
        },
        paths["checkpoint"],
    )
    paths["checkpoint"].chmod(0o444)
    _save_component(paths["compiler"], model.compiler)
    _save_component(paths["reactor"], model.reactor)
    _save_component(paths["query_reader"], model.query_reader)
    return model, paths


def _build(paths: dict[str, Path]) -> ETTRModelAssemblyReceipt:
    return ETTRModelAssemblyReceipt.build(
        config_path=paths["config"],
        checkpoint_path=paths["checkpoint"],
        checkpoint_step=123,
        compiler_path=paths["compiler"],
        reactor_path=paths["reactor"],
        query_reader_path=paths["query_reader"],
    )


def _validate(
    receipt: ETTRModelAssemblyReceipt,
    paths: dict[str, Path],
) -> None:
    receipt.validate(
        expected_receipt_sha256=receipt.sha(),
        config_path=paths["config"],
        checkpoint_path=paths["checkpoint"],
        compiler_path=paths["compiler"],
        reactor_path=paths["reactor"],
        query_reader_path=paths["query_reader"],
    )


def test_build_recomputes_exact_complete_model_and_parameter_ledger(
    tmp_path: Path,
) -> None:
    model, paths = _assembly(tmp_path)
    receipt = _build(paths)
    parameters = model.parameter_receipt()

    assert receipt.schema == ETTR_MODEL_ASSEMBLY_SCHEMA
    assert receipt.complete_model_sha256 == _model_sha256(model)
    assert receipt.base_parameters == parameters.base_parameters
    assert (
        receipt.architecture_parameters
        == parameters.architecture_parameters
    )
    assert receipt.total_parameters == parameters.complete_system_parameters
    assert receipt.parameter_cap == parameters.parameter_cap
    assert receipt.remaining_under_cap == parameters.remaining_under_cap
    assert receipt.total_parameters < SYSTEM_PARAMETER_CAP
    assert receipt == ETTRModelAssemblyReceipt.recompute(
        config_path=paths["config"],
        checkpoint_path=paths["checkpoint"],
        checkpoint_step=123,
        compiler_path=paths["compiler"],
        reactor_path=paths["reactor"],
        query_reader_path=paths["query_reader"],
    )


def test_complete_model_identity_binds_behavioral_configuration() -> None:
    model = _model()
    original_sha256 = _model_sha256(model)
    model.config = replace(model.config, stage_after_block=0)
    assert _model_sha256(model) != original_sha256

    model = _model()
    original_sha256 = _model_sha256(model)
    assert "base.cos" not in model.state_dict()
    model.base.cos.zero_()
    assert _model_sha256(model) != original_sha256


def test_receipt_round_trip_is_canonical_deterministic_and_validated(
    tmp_path: Path,
) -> None:
    _, paths = _assembly(tmp_path)
    receipt = _build(paths)
    receipt_path = tmp_path / "assembly.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    receipt_path.chmod(0o444)

    loaded = ETTRModelAssemblyReceipt.from_path(receipt_path)
    assert loaded == receipt
    assert loaded.sha() == loaded.sha256()
    assert loaded.canonical_bytes() == receipt_path.read_bytes()
    _validate(loaded, paths)


@pytest.mark.parametrize(
    "substitution",
    ("config", "checkpoint", "compiler", "reactor", "query_reader"),
)
def test_validation_rejects_every_assembly_input_substitution(
    tmp_path: Path,
    substitution: str,
) -> None:
    _, paths = _assembly(tmp_path, prefix="original_")
    _, replacements = _assembly(
        tmp_path,
        seed=2026072602,
        prefix="replacement_",
    )
    if substitution == "config":
        replacement_config = asdict(_model(2026072602).config)
        replacement_config["max_steps"] += 1
        replacements["config"].chmod(0o644)
        _write_canonical(replacements["config"], replacement_config)
    receipt = _build(paths)
    attacked = dict(paths)
    attacked[substitution] = replacements[substitution]

    with pytest.raises(ETTRModelAssemblyError):
        _validate(receipt, attacked)


def test_checkpoint_step_and_strict_component_load_fail_closed(
    tmp_path: Path,
) -> None:
    model, paths = _assembly(tmp_path)
    with pytest.raises(ETTRModelAssemblyError):
        ETTRModelAssemblyReceipt.recompute(
            config_path=paths["config"],
            checkpoint_path=paths["checkpoint"],
            checkpoint_step=124,
            compiler_path=paths["compiler"],
            reactor_path=paths["reactor"],
            query_reader_path=paths["query_reader"],
        )

    incomplete = tmp_path / "incomplete_compiler.safetensors"
    state = model.compiler.state_dict()
    omitted = next(iter(state))
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in state.items()
            if name != omitted
        },
        incomplete,
    )
    incomplete.chmod(0o444)
    with pytest.raises(
        ETTRModelAssemblyError,
        match="compiler safetensors strict load differs",
    ):
        ETTRModelAssemblyReceipt.recompute(
            config_path=paths["config"],
            checkpoint_path=paths["checkpoint"],
            checkpoint_step=123,
            compiler_path=incomplete,
            reactor_path=paths["reactor"],
            query_reader_path=paths["query_reader"],
        )


def test_receipt_hash_parameter_cap_and_immutable_files_fail_closed(
    tmp_path: Path,
) -> None:
    _, paths = _assembly(tmp_path)
    receipt = _build(paths)

    with pytest.raises(
        ETTRModelAssemblyError,
        match="receipt hash differs",
    ):
        receipt.validate(
            expected_receipt_sha256="0" * 64,
            config_path=paths["config"],
            checkpoint_path=paths["checkpoint"],
            compiler_path=paths["compiler"],
            reactor_path=paths["reactor"],
            query_reader_path=paths["query_reader"],
        )

    over_cap = replace(
        receipt,
        total_parameters=SYSTEM_PARAMETER_CAP + 1,
        parameter_cap=SYSTEM_PARAMETER_CAP + 1,
        remaining_under_cap=0,
    )
    with pytest.raises(
        ETTRModelAssemblyError,
        match="receipt fields are invalid",
    ):
        over_cap.validate(
            expected_receipt_sha256=over_cap.sha(),
            config_path=paths["config"],
            checkpoint_path=paths["checkpoint"],
            compiler_path=paths["compiler"],
            reactor_path=paths["reactor"],
            query_reader_path=paths["query_reader"],
        )

    paths["compiler"].chmod(0o644)
    with pytest.raises(
        ETTRModelAssemblyError,
        match="not an immutable regular file",
    ):
        _validate(receipt, paths)


def test_from_path_rejects_noncanonical_and_symlink_receipts(
    tmp_path: Path,
) -> None:
    _, paths = _assembly(tmp_path)
    receipt = _build(paths)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(asdict(receipt), indent=2) + "\n",
        encoding="ascii",
    )
    noncanonical.chmod(0o444)
    with pytest.raises(
        ETTRModelAssemblyError,
        match="not canonical",
    ):
        ETTRModelAssemblyReceipt.from_path(noncanonical)

    canonical = tmp_path / "canonical.json"
    canonical.write_bytes(receipt.canonical_bytes())
    canonical.chmod(0o444)
    symlink = tmp_path / "receipt-link.json"
    symlink.symlink_to(canonical)
    with pytest.raises(
        ETTRModelAssemblyError,
        match="not an immutable regular file",
    ):
        ETTRModelAssemblyReceipt.from_path(symlink)


def test_checkpoint_load_is_explicitly_weights_only() -> None:
    source = Path(__file__).with_name("ettr_model_assembly.py").read_text(
        encoding="utf-8"
    )
    assert "weights_only=True" in source
