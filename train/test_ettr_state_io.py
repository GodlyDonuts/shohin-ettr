from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
    TheoryReactorError,
)
from ettr_state_io import (
    ETTRStateIOError,
    read_state,
    verify_state_receipt,
    write_state_once,
)
from model import GPT, GPTConfig


def _model() -> EndogenousTypedTheoryReactorGPT:
    torch.manual_seed(2026072502)
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
    config = TheoryReactorConfig(
        d_model=32,
        state_width=32,
        num_slots=6,
        num_types=3,
        num_relations=3,
        num_heads=4,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        ff_multiplier=2,
        max_steps=6,
        stage_after_block=1,
        parameter_cap=1_000_000,
    )
    return EndogenousTypedTheoryReactorGPT(base, config)


def test_state_wire_round_trips_without_source_bytes(
    tmp_path: Path,
) -> None:
    model = _model()
    source = (
        b"SOURCE_ONLY_SENTINEL_"
        b"47c9288df9b24788a1141132d1b8ec54"
    )
    state = model.compile_world(
        torch.randint(0, 64, (2, 9)),
        hard=True,
    )
    path = tmp_path / "state.safetensors"
    receipt = write_state_once(
        path,
        state,
        model.config,
        forbidden_source=source,
    )
    restored = read_state(path, model.config)
    verify_state_receipt(
        path,
        receipt,
        forbidden_source=source,
    )
    assert restored.step == state.step
    for name in (
        "value_probabilities",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
        "halted",
    ):
        assert torch.equal(getattr(restored, name), getattr(state, name))
    assert path.stat().st_mode & 0o222 == 0
    assert source not in path.read_bytes()


def test_state_wire_fails_closed_on_mutability(
    tmp_path: Path,
) -> None:
    model = _model()
    path = tmp_path / "state.safetensors"
    write_state_once(
        path,
        model.compile_world(
            torch.randint(0, 64, (1, 5)),
            hard=True,
        ),
        model.config,
    )
    path.chmod(0o644)
    with pytest.raises(
        ETTRStateIOError,
        match="mutable",
    ):
        read_state(path, model.config)


def test_state_wire_is_write_once(
    tmp_path: Path,
) -> None:
    model = _model()
    state = model.compile_world(
        torch.randint(0, 64, (1, 5)),
        hard=True,
    )
    path = tmp_path / "state.safetensors"
    write_state_once(path, state, model.config)
    with pytest.raises(
        ETTRStateIOError,
        match="already exists",
    ):
        write_state_once(path, state, model.config)


def test_state_wire_rejects_configuration_substitution(
    tmp_path: Path,
) -> None:
    model = _model()
    path = tmp_path / "state.safetensors"
    write_state_once(
        path,
        model.compile_world(
            torch.randint(0, 64, (1, 5)),
            hard=True,
        ),
        model.config,
    )
    wrong = TheoryReactorConfig(
        **{
            **asdict(model.config),
            "num_types": model.config.num_types + 1,
        }
    )
    with pytest.raises(
        ETTRStateIOError,
        match="configuration differs",
    ):
        read_state(path, wrong)


def test_state_wire_rejects_continuous_source_channel(
    tmp_path: Path,
) -> None:
    model = _model()
    soft = model.compile_world(
        torch.randint(0, 64, (1, 5)),
        hard=False,
    )
    with pytest.raises(
        TheoryReactorError,
        match="not binary",
    ):
        write_state_once(
            tmp_path / "soft-state.safetensors",
            soft,
            model.config,
        )
