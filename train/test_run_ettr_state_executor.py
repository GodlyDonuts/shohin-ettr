from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from safetensors.torch import save_file
import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from ettr_state_io import read_state, write_state_once
from model import GPT, GPTConfig


def _model() -> EndogenousTypedTheoryReactorGPT:
    torch.manual_seed(2026072503)
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
            num_heads=4,
            compiler_layers=1,
            reactor_layers=1,
            query_layers=1,
            ff_multiplier=2,
            max_steps=6,
            stage_after_block=1,
            parameter_cap=1_000_000,
        ),
    )


def _canonical_json(path: Path, value: object) -> None:
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


def test_fresh_executor_receives_only_deleted_state_and_reactor(
    tmp_path: Path,
) -> None:
    model = _model().eval()
    source = (
        b"SOURCE_ONLY_EXECUTOR_SENTINEL_"
        b"b1e7849f217b4f548ecbb1a88d2024d9"
    )
    with torch.no_grad():
        initial = model.compile_world(
            torch.randint(0, 64, (2, 8)),
            hard=True,
        )
        direct, _ = model.execute(initial, steps=3, hard=True)
    state_path = tmp_path / "state.safetensors"
    output_path = tmp_path / "terminal.safetensors"
    config_path = tmp_path / "config.json"
    reactor_path = tmp_path / "reactor.safetensors"
    write_state_once(
        state_path,
        initial,
        model.config,
        forbidden_source=source,
    )
    _canonical_json(config_path, asdict(model.config))
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.reactor.state_dict().items()
        },
        reactor_path,
    )
    reactor_path.chmod(0o444)
    runner = Path(__file__).with_name(
        "run_ettr_state_executor.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--config",
            str(config_path),
            "--state",
            str(state_path),
            "--reactor",
            str(reactor_path),
            "--output",
            str(output_path),
            "--steps",
            "3",
            "--hard",
        ],
        check=True,
        cwd=runner.parent.parent,
        capture_output=True,
        text=True,
    )
    terminal = read_state(output_path, model.config)
    for name in (
        "value_probabilities",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
        "halted",
    ):
        assert torch.equal(getattr(terminal, name), getattr(direct, name))
    assert terminal.step == 3
    assert source not in output_path.read_bytes()
    assert output_path.stat().st_mode & 0o222 == 0


def test_executor_cli_has_no_forbidden_inputs_or_model_import() -> None:
    source = Path(__file__).with_name(
        "run_ettr_state_executor.py"
    ).read_text()
    for forbidden in (
        "--source",
        "--world",
        "--query",
        "--tokenizer",
        "--assessor",
        "--checkpoint",
        "from model import",
        "EndogenousTheoryCompiler",
        "SourceDeletedQueryReader",
    ):
        assert forbidden not in source
