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
from ettr_factorial_custody import (
    ETTRFactorialExecutionManifest,
    ETTRStageExecutionReceipt,
    EXECUTION_MANIFEST_SCHEMA,
)
from ettr_factorial_qualification import bind_terminal_state_artifact
from ettr_factorial_qualification_board import (
    TOTAL_PACKETS,
    build_ettr_factorial_qualification_board,
)
from model import GPT, GPTConfig
from ettr_qualification import _model_sha256
from ettr_state_io import read_state
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


def _digest_tokens(payload: bytes, width: int) -> list[int]:
    digest = hashlib.shake_256(payload).digest(width)
    return [value % 64 for value in digest]


def test_four_process_chain_physically_deletes_prior_inputs(
    tmp_path: Path,
) -> None:
    model = _model()
    board = build_ettr_factorial_qualification_board()
    packet_rows = tuple(
        next(row for row in board.rows if row.packet_factor_id == packet_factor_id)
        for packet_factor_id in board.packet_factor_ids
    )
    world = torch.tensor([_digest_tokens(row.world_bytes, 8) for row in packet_rows])
    command = torch.tensor(
        [_digest_tokens(row.command_bytes, 6) for row in packet_rows]
    )
    query = torch.tensor(
        [_digest_tokens(row.query_prefix_bytes, 5) for row in packet_rows]
    )
    world_mask = torch.ones_like(world, dtype=torch.bool)
    command_mask = torch.ones_like(command, dtype=torch.bool)
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
            command_idx=command,
            command_attention_mask=command_mask,
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
    custody_dir = tmp_path / "00_custody"
    custody_dir.mkdir()
    reactor_anchor = custody_dir / "reactor.safetensors"
    command_anchor = custody_dir / "command.json"
    _save_weights(reactor_anchor, model.reactor)
    _write_json(
        command_anchor,
        {
            "attention_mask": command_mask.int().tolist(),
            "token_ids": command.tolist(),
        },
    )

    compiler_dir = tmp_path / "01_compiler"
    compiler_dir.mkdir()
    config = compiler_dir / "config.json"
    checkpoint = compiler_dir / "base.pt"
    compiler_weights = compiler_dir / "compiler.safetensors"
    world_path = compiler_dir / "world.json"
    compiled_state = compiler_dir / "state.safetensors"
    manifest_path = compiler_dir / "execution_manifest.json"
    compiler_receipt = compiler_dir / "compiler_receipt.json"
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
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = ETTRFactorialExecutionManifest(
        schema=EXECUTION_MANIFEST_SCHEMA,
        board_sha256=board.receipt.payload_sha256,
        model_sha256=_model_sha256(model),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        checkpoint_sha256=checkpoint_digest,
        checkpoint_step=123,
        compiler_sha256=hashlib.sha256(compiler_weights.read_bytes()).hexdigest(),
        reactor_sha256=hashlib.sha256(reactor_anchor.read_bytes()).hexdigest(),
        world_package_sha256=board.receipt.world_package_sha256,
        command_package_sha256=board.receipt.command_package_sha256,
        world_tokens_sha256=hashlib.sha256(world_path.read_bytes()).hexdigest(),
        command_tokens_sha256=hashlib.sha256(command_anchor.read_bytes()).hexdigest(),
        row_count=TOTAL_PACKETS,
    )
    _write_json(manifest_path, asdict(manifest))
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
        "--execution-manifest",
        str(manifest_path),
        "--execution-manifest-sha256",
        manifest.sha256(),
        "--output",
        str(compiled_state),
        "--receipt-output",
        str(compiler_receipt),
        "--hard",
    )
    compiled_bytes = compiled_state.read_bytes()
    config_bytes = config.read_bytes()
    checkpoint_bytes = checkpoint.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    compiler_receipt_bytes = compiler_receipt.read_bytes()
    compiler_receipt_record = ETTRStageExecutionReceipt.from_path(compiler_receipt)
    compiler_receipt_sha256 = hashlib.sha256(compiler_receipt_bytes).hexdigest()
    shutil.rmtree(compiler_dir)
    assert not compiler_dir.exists()

    executor_dir = tmp_path / "02_executor"
    executor_dir.mkdir()
    config = executor_dir / "config.json"
    state = executor_dir / "state.safetensors"
    reactor_weights = executor_dir / "reactor.safetensors"
    checkpoint = executor_dir / "base.pt"
    command_path = executor_dir / "command.json"
    manifest_path = executor_dir / "execution_manifest.json"
    compiler_receipt = executor_dir / "compiler_receipt.json"
    executor_receipt = executor_dir / "executor_receipt.json"
    terminal = executor_dir / "terminal.safetensors"
    config.write_bytes(config_bytes)
    config.chmod(0o444)
    state.write_bytes(compiled_bytes)
    state.chmod(0o444)
    reactor_weights.write_bytes(reactor_anchor.read_bytes())
    reactor_weights.chmod(0o444)
    checkpoint.write_bytes(checkpoint_bytes)
    checkpoint.chmod(0o444)
    command_path.write_bytes(command_anchor.read_bytes())
    command_path.chmod(0o444)
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o444)
    compiler_receipt.write_bytes(compiler_receipt_bytes)
    compiler_receipt.chmod(0o444)
    assert {path.name for path in executor_dir.iterdir()} == {
        "base.pt",
        "command.json",
        "compiler_receipt.json",
        "config.json",
        "execution_manifest.json",
        "reactor.safetensors",
        "state.safetensors",
    }
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _run(
        str(TRAIN / "run_ettr_state_executor.py"),
        "--config",
        str(config),
        "--state",
        str(state),
        "--reactor",
        str(reactor_weights),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        checkpoint_digest,
        "--expected-step",
        "123",
        "--command",
        str(command_path),
        "--execution-manifest",
        str(manifest_path),
        "--execution-manifest-sha256",
        manifest.sha256(),
        "--compiler-receipt",
        str(compiler_receipt),
        "--compiler-receipt-sha256",
        compiler_receipt_sha256,
        "--output",
        str(terminal),
        "--receipt-output",
        str(executor_receipt),
        "--steps",
        "2",
        "--hard",
    )
    terminal_bytes = terminal.read_bytes()
    executor_receipt_record = ETTRStageExecutionReceipt.from_path(executor_receipt)
    executor_receipt_sha256 = hashlib.sha256(executor_receipt.read_bytes()).hexdigest()
    terminal_state = read_state(terminal, model.config)
    artifact = bind_terminal_state_artifact(
        board,
        terminal_state,
        execution_manifest=manifest,
        compiler_receipt=compiler_receipt_record,
        executor_receipt=executor_receipt_record,
        expected_model_sha256=manifest.model_sha256,
        expected_execution_manifest_sha256=manifest.sha256(),
        expected_compiler_receipt_sha256=compiler_receipt_sha256,
        expected_executor_receipt_sha256=executor_receipt_sha256,
        config=model.config,
    )
    assert artifact.executor_receipt_sha256 == executor_receipt_sha256
    shutil.rmtree(executor_dir)
    assert not executor_dir.exists()
    shutil.rmtree(custody_dir)
    assert not custody_dir.exists()

    query_dir = tmp_path / "03_query"
    query_dir.mkdir()
    config = query_dir / "config.json"
    terminal = query_dir / "terminal.safetensors"
    reader = query_dir / "reader.safetensors"
    checkpoint = query_dir / "base.pt"
    query_path = query_dir / "query.json"
    candidate = query_dir / "candidate.json"
    config.write_bytes(config_bytes)
    config.chmod(0o444)
    terminal.write_bytes(terminal_bytes)
    terminal.chmod(0o444)
    _save_weights(reader, model.query_reader)
    checkpoint.write_bytes(checkpoint_bytes)
    checkpoint.chmod(0o444)
    _write_json(
        query_path,
        {
            "attention_mask": query_mask.int().tolist(),
            "token_ids": query.tolist(),
        },
    )
    assert "world.json" not in {path.name for path in query_dir.iterdir()}
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
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
