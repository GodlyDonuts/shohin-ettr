from __future__ import annotations

from dataclasses import asdict
import hashlib
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
from ettr_state_io import read_state
from model import GPT, GPTConfig


def _model() -> EndogenousTypedTheoryReactorGPT:
    torch.manual_seed(2026072505)
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_compiler_matches_direct_raw_token_compile(
    tmp_path: Path,
) -> None:
    model = _model()
    world = torch.randint(0, 64, (2, 9))
    mask = torch.ones_like(world, dtype=torch.bool)
    with torch.no_grad():
        direct = model.compile_world(
            world,
            attention_mask=mask,
            hard=True,
        )
    compiler_path = tmp_path / "compiler.safetensors"
    checkpoint_path = tmp_path / "base.pt"
    config_path = tmp_path / "config.json"
    world_path = tmp_path / "world.json"
    output_path = tmp_path / "state.safetensors"
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.compiler.state_dict().items()
        },
        compiler_path,
    )
    compiler_path.chmod(0o444)
    torch.save(
        {
            "cfg": asdict(model.base.cfg),
            "model": model.base.state_dict(),
            "step": 123,
        },
        checkpoint_path,
    )
    checkpoint_path.chmod(0o444)
    _canonical_json(config_path, asdict(model.config))
    _canonical_json(
        world_path,
        {
            "attention_mask": mask.int().tolist(),
            "token_ids": world.tolist(),
        },
    )
    runner = Path(__file__).with_name(
        "run_ettr_world_compiler.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--config",
            str(config_path),
            "--compiler",
            str(compiler_path),
            "--checkpoint",
            str(checkpoint_path),
            "--checkpoint-sha256",
            _sha256(checkpoint_path),
            "--expected-step",
            "123",
            "--world",
            str(world_path),
            "--output",
            str(output_path),
            "--hard",
        ],
        check=True,
        cwd=runner.parent.parent,
        capture_output=True,
        text=True,
    )
    state = read_state(output_path, model.config)
    for name in (
        "value_probabilities",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
        "halted",
    ):
        assert torch.equal(getattr(state, name), getattr(direct, name))
    assert state.step == 0
    assert world_path.read_bytes() not in output_path.read_bytes()
    assert output_path.stat().st_mode & 0o222 == 0


def test_compiler_cli_has_no_query_executor_or_assessor_input() -> None:
    source = Path(__file__).with_name(
        "run_ettr_world_compiler.py"
    ).read_text()
    for forbidden in (
        "--query",
        "--reactor",
        "--assessor",
        "GenericTransactionReactor",
        "SourceDeletedQueryReader",
    ):
        assert forbidden not in source
