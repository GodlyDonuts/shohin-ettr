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
from ettr_state_io import write_state_once
from model import GPT, GPTConfig
from run_ettr_late_query import ANSWER_SCHEMA


def _model() -> EndogenousTypedTheoryReactorGPT:
    torch.manual_seed(2026072504)
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


def test_fresh_late_query_matches_direct_state_only_read(
    tmp_path: Path,
) -> None:
    model = _model()
    world = torch.randint(0, 64, (2, 8))
    query = torch.randint(0, 64, (2, 5))
    mask = torch.ones_like(query, dtype=torch.bool)
    with torch.no_grad():
        state = model.compile_world(world, hard=True)
        state, _ = model.execute(state, steps=2, hard=True)
        direct, _ = model.answer_query(
            state,
            query,
            attention_mask=mask,
        )
    state_path = tmp_path / "terminal.safetensors"
    reader_path = tmp_path / "reader.safetensors"
    checkpoint_path = tmp_path / "base.pt"
    config_path = tmp_path / "config.json"
    query_path = tmp_path / "query.json"
    output_path = tmp_path / "answer.json"
    write_state_once(state_path, state, model.config)
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.query_reader.state_dict().items()
        },
        reader_path,
    )
    reader_path.chmod(0o444)
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
        query_path,
        {
            "attention_mask": mask.int().tolist(),
            "token_ids": query.tolist(),
        },
    )
    runner = Path(__file__).with_name(
        "run_ettr_late_query.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--config",
            str(config_path),
            "--state",
            str(state_path),
            "--reader",
            str(reader_path),
            "--checkpoint",
            str(checkpoint_path),
            "--checkpoint-sha256",
            _sha256(checkpoint_path),
            "--expected-step",
            "123",
            "--query",
            str(query_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=runner.parent.parent,
        capture_output=True,
        text=True,
    )
    answer = json.loads(output_path.read_text())
    assert answer == {
        "schema": ANSWER_SCHEMA,
        "token_ids": direct.argmax(-1).tolist(),
    }
    assert output_path.stat().st_mode & 0o222 == 0


def test_late_query_cli_has_no_world_or_compiler_interface() -> None:
    source = Path(__file__).with_name(
        "run_ettr_late_query.py"
    ).read_text()
    for forbidden in (
        "--source",
        "--world",
        "--compiler",
        "--reactor",
        "--assessor",
        "EndogenousTheoryCompiler",
        "GenericTransactionReactor",
    ):
        assert forbidden not in source
