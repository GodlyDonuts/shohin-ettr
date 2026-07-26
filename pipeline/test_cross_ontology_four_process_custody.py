from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from safetensors.torch import save_file
import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from model import GPT, GPTConfig
from run_cross_ontology_assessor import (
    ANSWER_SCHEMA,
    ASSESSMENT_SCHEMA,
    EXPECTED_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
PIPELINE = ROOT / "pipeline"


def _model() -> EndogenousTypedTheoryReactorGPT:
    torch.manual_seed(2026072506)
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


def _write_json(path: Path, value: object) -> None:
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


def _save_weights(path: Path, module: torch.nn.Module) -> None:
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in module.state_dict().items()
        },
        path,
    )
    path.chmod(0o444)


def _run(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_four_process_chain_physically_deletes_prior_inputs(
    tmp_path: Path,
) -> None:
    model = _model()
    world = torch.randint(0, 64, (2, 8))
    query = torch.randint(0, 64, (2, 5))
    world_mask = torch.ones_like(world, dtype=torch.bool)
    query_mask = torch.ones_like(query, dtype=torch.bool)
    with torch.no_grad():
        direct_state = model.compile_world(
            world,
            attention_mask=world_mask,
            hard=True,
        )
        direct_state, _ = model.execute(
            direct_state,
            steps=2,
            hard=True,
        )
        direct_logits, _ = model.answer_query(
            direct_state,
            query,
            attention_mask=query_mask,
        )
        expected_tokens = direct_logits.argmax(-1).tolist()

    base_payload = {
        "cfg": asdict(model.base.cfg),
        "model": model.base.state_dict(),
        "step": 123,
    }

    compiler_dir = tmp_path / "01_compiler"
    compiler_dir.mkdir()
    config = compiler_dir / "config.json"
    checkpoint = compiler_dir / "base.pt"
    compiler_weights = compiler_dir / "compiler.safetensors"
    world_path = compiler_dir / "world.json"
    compiled_state = compiler_dir / "state.safetensors"
    _write_json(config, asdict(model.config))
    torch.save(base_payload, checkpoint)
    checkpoint.chmod(0o444)
    _save_weights(compiler_weights, model.compiler)
    _write_json(
        world_path,
        {
            "attention_mask": world_mask.int().tolist(),
            "token_ids": world.tolist(),
        },
    )
    checkpoint_digest = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    _run(
        str(TRAIN / "run_ettr_world_compiler.py"),
        "--config",
        str(config),
        "--compiler",
        str(compiler_weights),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        checkpoint_digest,
        "--expected-step",
        "123",
        "--world",
        str(world_path),
        "--output",
        str(compiled_state),
        "--hard",
    )
    compiled_bytes = compiled_state.read_bytes()
    shutil.rmtree(compiler_dir)
    assert not compiler_dir.exists()

    executor_dir = tmp_path / "02_executor"
    executor_dir.mkdir()
    config = executor_dir / "config.json"
    state = executor_dir / "state.safetensors"
    reactor_weights = executor_dir / "reactor.safetensors"
    terminal = executor_dir / "terminal.safetensors"
    _write_json(config, asdict(model.config))
    state.write_bytes(compiled_bytes)
    state.chmod(0o444)
    _save_weights(reactor_weights, model.reactor)
    assert {path.name for path in executor_dir.iterdir()} == {
        "config.json",
        "reactor.safetensors",
        "state.safetensors",
    }
    _run(
        str(TRAIN / "run_ettr_state_executor.py"),
        "--config",
        str(config),
        "--state",
        str(state),
        "--reactor",
        str(reactor_weights),
        "--output",
        str(terminal),
        "--steps",
        "2",
        "--hard",
    )
    terminal_bytes = terminal.read_bytes()
    shutil.rmtree(executor_dir)
    assert not executor_dir.exists()

    query_dir = tmp_path / "03_query"
    query_dir.mkdir()
    config = query_dir / "config.json"
    terminal = query_dir / "terminal.safetensors"
    reader = query_dir / "reader.safetensors"
    checkpoint = query_dir / "base.pt"
    query_path = query_dir / "query.json"
    candidate = query_dir / "candidate.json"
    _write_json(config, asdict(model.config))
    terminal.write_bytes(terminal_bytes)
    terminal.chmod(0o444)
    _save_weights(reader, model.query_reader)
    torch.save(base_payload, checkpoint)
    checkpoint.chmod(0o444)
    _write_json(
        query_path,
        {
            "attention_mask": query_mask.int().tolist(),
            "token_ids": query.tolist(),
        },
    )
    assert "world.json" not in {
        path.name for path in query_dir.iterdir()
    }
    checkpoint_digest = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    _run(
        str(TRAIN / "run_ettr_late_query.py"),
        "--config",
        str(config),
        "--state",
        str(terminal),
        "--reader",
        str(reader),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        checkpoint_digest,
        "--expected-step",
        "123",
        "--query",
        str(query_path),
        "--output",
        str(candidate),
    )
    candidate_payload = json.loads(candidate.read_text())
    candidate_bytes = candidate.read_bytes()
    assert candidate_payload["schema"] == ANSWER_SCHEMA
    shutil.rmtree(query_dir)
    assert not query_dir.exists()

    assessor_dir = tmp_path / "04_assessor"
    assessor_dir.mkdir()
    candidate = assessor_dir / "candidate.json"
    expected = assessor_dir / "expected.json"
    assessment = assessor_dir / "assessment.json"
    candidate.write_bytes(candidate_bytes)
    candidate.chmod(0o444)
    assert not expected.exists()
    _write_json(
        expected,
        {
            "disposition": "singleton",
            "expected_token_ids": expected_tokens,
            "schema": EXPECTED_SCHEMA,
        },
    )
    assert {path.name for path in assessor_dir.iterdir()} == {
        "candidate.json",
        "expected.json",
    }
    _run(
        str(PIPELINE / "run_cross_ontology_assessor.py"),
        "--candidate",
        str(candidate),
        "--expected",
        str(expected),
        "--output",
        str(assessment),
    )
    assert json.loads(assessment.read_text()) == {
        "disposition": "singleton",
        "exact": True,
        "schema": ASSESSMENT_SCHEMA,
    }
