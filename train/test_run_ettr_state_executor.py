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
from ettr_factorial_custody import (
    ETTRFactorialExecutionManifest,
    ETTRStageExecutionReceipt,
    EXECUTION_MANIFEST_SCHEMA,
    STAGE_RECEIPT_SCHEMA,
)
from ettr_factorial_qualification_board import (
    TOTAL_PACKETS,
    build_ettr_factorial_qualification_board,
)
from ettr_state_io import read_state, write_state_once
from ettr_qualification import typed_state_sha256
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


def test_fresh_executor_receives_state_and_post_seal_command(
    tmp_path: Path,
) -> None:
    model = _model().eval()
    source = b"SOURCE_ONLY_EXECUTOR_SENTINEL_b1e7849f217b4f548ecbb1a88d2024d9"
    with torch.no_grad():
        initial = model.compile_world(
            torch.randint(0, 64, (TOTAL_PACKETS, 8)),
            hard=True,
        )
        command = torch.randint(0, 64, (TOTAL_PACKETS, 7))
        command_mask = torch.ones_like(command, dtype=torch.bool)
        direct, _ = model.execute(
            initial,
            steps=3,
            hard=True,
            command_idx=command,
            command_attention_mask=command_mask,
        )
    state_path = tmp_path / "state.safetensors"
    output_path = tmp_path / "terminal.safetensors"
    config_path = tmp_path / "config.json"
    reactor_path = tmp_path / "reactor.safetensors"
    checkpoint_path = tmp_path / "base.pt"
    command_path = tmp_path / "command.json"
    manifest_path = tmp_path / "execution_manifest.json"
    compiler_receipt_path = tmp_path / "compiler_receipt.json"
    executor_receipt_path = tmp_path / "executor_receipt.json"
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
    torch.save(
        {
            "cfg": asdict(model.base.cfg),
            "model": model.base.state_dict(),
            "step": 123,
        },
        checkpoint_path,
    )
    checkpoint_path.chmod(0o444)
    _canonical_json(
        command_path,
        {
            "attention_mask": command_mask.int().tolist(),
            "token_ids": command.tolist(),
        },
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    board = build_ettr_factorial_qualification_board()
    manifest = ETTRFactorialExecutionManifest(
        schema=EXECUTION_MANIFEST_SCHEMA,
        board_sha256=board.receipt.payload_sha256,
        model_sha256="a" * 64,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_step=123,
        compiler_sha256="b" * 64,
        reactor_sha256=hashlib.sha256(reactor_path.read_bytes()).hexdigest(),
        reader_sha256="d" * 64,
        tokenizer_sha256="e" * 64,
        tokenization_receipt_sha256="f" * 64,
        model_assembly_receipt_sha256="0" * 64,
        bootstrap_sha256="4" * 64,
        runtime_bundle_sha256="5" * 64,
        compiler_runner_sha256="2" * 64,
        executor_runner_sha256=hashlib.sha256(
            Path(__file__)
            .with_name("run_ettr_state_executor.py")
            .read_bytes()
        ).hexdigest(),
        query_runner_sha256="3" * 64,
        compiler_hard=True,
        executor_hard=True,
        executor_steps=3,
        world_package_sha256=board.receipt.world_package_sha256,
        command_package_sha256=board.receipt.command_package_sha256,
        query_package_sha256=board.receipt.query_package_sha256,
        world_tokens_sha256="c" * 64,
        command_tokens_sha256=hashlib.sha256(command_path.read_bytes()).hexdigest(),
        query_tokens_sha256="1" * 64,
        row_count=TOTAL_PACKETS,
    )
    _canonical_json(manifest_path, asdict(manifest))
    compiler_receipt = ETTRStageExecutionReceipt(
        schema=STAGE_RECEIPT_SCHEMA,
        stage="world",
        manifest_sha256=manifest.sha256(),
        parent_receipt_sha256=None,
        input_state_file_sha256=None,
        input_state_tensor_sha256=None,
        token_input_sha256=manifest.world_tokens_sha256,
        component_sha256=manifest.compiler_sha256,
        checkpoint_sha256=manifest.checkpoint_sha256,
        output_state_file_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
        output_state_tensor_sha256=typed_state_sha256(initial),
        row_count=TOTAL_PACKETS,
    )
    _canonical_json(compiler_receipt_path, asdict(compiler_receipt))
    runner = Path(__file__).with_name("run_ettr_state_executor.py")
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
            "--checkpoint",
            str(checkpoint_path),
            "--checkpoint-sha256",
            checkpoint_sha256,
            "--expected-step",
            "123",
            "--command",
            str(command_path),
            "--execution-manifest",
            str(manifest_path),
            "--execution-manifest-sha256",
            manifest.sha256(),
            "--compiler-receipt",
            str(compiler_receipt_path),
            "--compiler-receipt-sha256",
            compiler_receipt.sha256(),
            "--output",
            str(output_path),
            "--receipt-output",
            str(executor_receipt_path),
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
    assert command_path.read_bytes() not in output_path.read_bytes()
    assert output_path.stat().st_mode & 0o222 == 0
    assert executor_receipt_path.stat().st_mode & 0o222 == 0


def test_executor_cli_has_no_world_query_or_assessor_inputs() -> None:
    source = Path(__file__).with_name("run_ettr_state_executor.py").read_text()
    for forbidden in (
        "--source",
        "--world",
        "--query",
        "--tokenizer",
        "--assessor",
        "EndogenousTheoryCompiler",
        "SourceDeletedQueryReader",
    ):
        assert forbidden not in source
    assert "--command" in source
    assert "--checkpoint" in source
    assert "from model import GPT, GPTConfig" in source
