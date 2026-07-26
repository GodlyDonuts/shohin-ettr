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
from ettr_state_io import write_state_once
from ettr_qualification import typed_state_sha256
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


def test_fresh_late_query_matches_direct_state_only_read(
    tmp_path: Path,
) -> None:
    model = _model()
    world = torch.randint(0, 64, (TOTAL_PACKETS, 8))
    query = torch.randint(0, 64, (TOTAL_PACKETS, 5))
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
    manifest_path = tmp_path / "execution-manifest.json"
    executor_receipt_path = tmp_path / "executor-receipt.json"
    query_receipt_path = tmp_path / "query-receipt.json"
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
    board = build_ettr_factorial_qualification_board()
    manifest = ETTRFactorialExecutionManifest(
        schema=EXECUTION_MANIFEST_SCHEMA,
        board_sha256=board.receipt.payload_sha256,
        model_sha256="a" * 64,
        config_sha256=_sha256(config_path),
        checkpoint_sha256=_sha256(checkpoint_path),
        checkpoint_step=123,
        compiler_sha256="b" * 64,
        reactor_sha256="c" * 64,
        reader_sha256=_sha256(reader_path),
        tokenizer_sha256="5" * 64,
        tokenization_receipt_sha256="3" * 64,
        model_assembly_receipt_sha256="4" * 64,
        bootstrap_sha256="a" * 64,
        world_runtime_bundle_sha256="8" * 64,
        command_runtime_bundle_sha256="9" * 64,
        query_runtime_bundle_sha256="b" * 64,
        claim_runtime_archive_sha256="c" * 64,
        claim_runtime_archive_size=1,
        claim_runtime_inventory_sha256="d" * 64,
        external_launcher_sha256="e" * 64,
        bwrap_sha256="f" * 64,
        network_namespace_required=True,
        world_stage_policy_sha256="1" * 64,
        command_stage_policy_sha256="2" * 64,
        query_stage_policy_sha256="3" * 64,
        compiler_runner_sha256="6" * 64,
        executor_runner_sha256="7" * 64,
        query_runner_sha256=_sha256(
            Path(__file__).with_name("run_ettr_late_query.py")
        ),
        compiler_hard=True,
        executor_hard=True,
        executor_steps=2,
        world_package_sha256=board.receipt.world_package_sha256,
        command_package_sha256=board.receipt.command_package_sha256,
        query_package_sha256=board.receipt.query_package_sha256,
        world_tokens_sha256="d" * 64,
        command_tokens_sha256="e" * 64,
        query_tokens_sha256=_sha256(query_path),
        row_count=TOTAL_PACKETS,
    )
    _canonical_json(manifest_path, asdict(manifest))
    executor_receipt = ETTRStageExecutionReceipt(
        schema=STAGE_RECEIPT_SCHEMA,
        stage="command",
        manifest_sha256=manifest.sha256(),
        parent_receipt_sha256="f" * 64,
        input_state_file_sha256="1" * 64,
        input_state_tensor_sha256="2" * 64,
        token_input_sha256=manifest.command_tokens_sha256,
        component_sha256=manifest.reactor_sha256,
        checkpoint_sha256=manifest.checkpoint_sha256,
        output_state_file_sha256=_sha256(state_path),
        output_state_tensor_sha256=typed_state_sha256(state),
        row_count=TOTAL_PACKETS,
    )
    _canonical_json(executor_receipt_path, asdict(executor_receipt))
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
            "--execution-manifest",
            str(manifest_path),
            "--execution-manifest-sha256",
            manifest.sha256(),
            "--executor-receipt",
            str(executor_receipt_path),
            "--executor-receipt-sha256",
            executor_receipt.sha256(),
            "--tokenization-receipt-sha256",
            "3" * 64,
            "--model-assembly-receipt-sha256",
            "4" * 64,
            "--output",
            str(output_path),
            "--receipt-output",
            str(query_receipt_path),
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
    receipt = json.loads(query_receipt_path.read_text())
    assert receipt["execution_manifest_sha256"] == manifest.sha256()
    assert receipt["executor_receipt_sha256"] == executor_receipt.sha256()
    assert receipt["terminal_state_tensor_sha256"] == typed_state_sha256(state)
    assert receipt["query_tokens_sha256"] == _sha256(query_path)
    assert receipt["reader_sha256"] == _sha256(reader_path)
    assert query_receipt_path.stat().st_mode & 0o222 == 0


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
        "--private-key",
        "EndogenousTheoryCompiler",
        "GenericTransactionReactor",
    ):
        assert forbidden not in source
