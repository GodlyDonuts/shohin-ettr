from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from safetensors.torch import save_file
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from ettr_factorial_custody import (
    ETTRFactorialExecutionManifest,
    ETTRLateQueryExecutionReceipt,
    ETTRStageExecutionReceipt,
    EXECUTION_MANIFEST_SCHEMA,
)
from ettr_factorial_qualification import bind_terminal_state_artifact
from ettr_factorial_signed_custody import validate_primary_custody_receipts
from ettr_factorial_tokenization import (
    build_ettr_factorial_tokenization_receipt,
)
from ettr_factorial_qualification_board import (
    TOTAL_PACKETS,
    build_ettr_factorial_qualification_board,
)
from ettr_model_assembly import ETTRModelAssemblyReceipt
from model import GPT, GPTConfig
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
            vocab_size=256,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=512,
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


def _write_character_tokenizer(path: Path) -> None:
    vocabulary = {"<pad>": 0, "<unk>": 1}
    vocabulary.update({chr(code): code + 2 for code in range(128)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split("", behavior="isolated")
    tokenizer.save(str(path))
    path.chmod(0o444)


def test_candidate_stage_sources_do_not_import_assessor_authority() -> None:
    forbidden = {
        "ettr_factorial_qualification_board",
        "ettr_factorial_signed_custody",
    }
    for runner in (
        "run_ettr_world_compiler.py",
        "run_ettr_state_executor.py",
        "run_ettr_late_query.py",
    ):
        source = (TRAIN / runner).read_text(encoding="utf-8")
        imports: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.add(node.module)
        assert imports.isdisjoint(forbidden), (runner, imports & forbidden)


def test_four_process_chain_physically_deletes_prior_inputs(
    tmp_path: Path,
) -> None:
    model = _model()
    board = build_ettr_factorial_qualification_board()
    base_payload = {
        "cfg": asdict(model.base.cfg),
        "model": model.base.state_dict(),
        "step": 123,
    }
    custody_dir = tmp_path / "00_custody"
    custody_dir.mkdir()
    tokenizer_anchor = custody_dir / "tokenizer.json"
    config_anchor = custody_dir / "config.json"
    checkpoint_anchor = custody_dir / "base.pt"
    compiler_anchor = custody_dir / "compiler.safetensors"
    reactor_anchor = custody_dir / "reactor.safetensors"
    reader_anchor = custody_dir / "reader.safetensors"
    world_anchor = custody_dir / "world.json"
    command_anchor = custody_dir / "command.json"
    query_anchor = custody_dir / "query.json"
    manifest_anchor = custody_dir / "execution_manifest.json"
    _write_character_tokenizer(tokenizer_anchor)
    _write_json(config_anchor, asdict(model.config))
    torch.save(base_payload, checkpoint_anchor)
    checkpoint_anchor.chmod(0o444)
    _save_weights(compiler_anchor, model.compiler)
    _save_weights(reactor_anchor, model.reactor)
    _save_weights(reader_anchor, model.query_reader)
    tokenization_receipt = build_ettr_factorial_tokenization_receipt(
        board,
        tokenizer_anchor,
        seq_len=model.base.cfg.seq_len,
        pad_token_id=0,
    )
    _write_json(world_anchor, tokenization_receipt.world_stage_payload())
    _write_json(command_anchor, tokenization_receipt.command_stage_payload())
    _write_json(query_anchor, tokenization_receipt.query_stage_payload())
    model_assembly_receipt = ETTRModelAssemblyReceipt.build(
        config_path=config_anchor,
        checkpoint_path=checkpoint_anchor,
        checkpoint_step=123,
        compiler_path=compiler_anchor,
        reactor_path=reactor_anchor,
        query_reader_path=reader_anchor,
    )
    world_payload = tokenization_receipt.world_stage_payload()
    command_payload = tokenization_receipt.command_stage_payload()
    query_payload = tokenization_receipt.query_stage_payload()
    world = torch.tensor(world_payload["token_ids"])
    command = torch.tensor(command_payload["token_ids"])
    query = torch.tensor(query_payload["token_ids"])
    world_mask = torch.tensor(world_payload["attention_mask"], dtype=torch.bool)
    command_mask = torch.tensor(
        command_payload["attention_mask"],
        dtype=torch.bool,
    )
    query_mask = torch.tensor(query_payload["attention_mask"], dtype=torch.bool)
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
    manifest = ETTRFactorialExecutionManifest(
        schema=EXECUTION_MANIFEST_SCHEMA,
        board_sha256=board.receipt.payload_sha256,
        model_sha256=model_assembly_receipt.complete_model_sha256,
        config_sha256=model_assembly_receipt.config_sha256,
        checkpoint_sha256=model_assembly_receipt.checkpoint_sha256,
        checkpoint_step=123,
        compiler_sha256=model_assembly_receipt.compiler_sha256,
        reactor_sha256=model_assembly_receipt.reactor_sha256,
        reader_sha256=model_assembly_receipt.query_reader_sha256,
        tokenizer_sha256=tokenization_receipt.tokenizer_sha256,
        tokenization_receipt_sha256=tokenization_receipt.sha256(),
        model_assembly_receipt_sha256=model_assembly_receipt.sha256(),
        compiler_runner_sha256=hashlib.sha256(
            (TRAIN / "run_ettr_world_compiler.py").read_bytes()
        ).hexdigest(),
        executor_runner_sha256=hashlib.sha256(
            (TRAIN / "run_ettr_state_executor.py").read_bytes()
        ).hexdigest(),
        query_runner_sha256=hashlib.sha256(
            (TRAIN / "run_ettr_late_query.py").read_bytes()
        ).hexdigest(),
        compiler_hard=True,
        executor_hard=True,
        executor_steps=2,
        world_package_sha256=board.receipt.world_package_sha256,
        command_package_sha256=board.receipt.command_package_sha256,
        query_package_sha256=board.receipt.query_package_sha256,
        world_tokens_sha256=hashlib.sha256(world_anchor.read_bytes()).hexdigest(),
        command_tokens_sha256=hashlib.sha256(
            command_anchor.read_bytes()
        ).hexdigest(),
        query_tokens_sha256=hashlib.sha256(query_anchor.read_bytes()).hexdigest(),
        row_count=TOTAL_PACKETS,
    )
    _write_json(manifest_anchor, asdict(manifest))
    validate_primary_custody_receipts(
        board,
        execution_manifest=manifest,
        expected_execution_manifest_sha256=manifest.sha256(),
        tokenization_receipt=tokenization_receipt,
        tokenizer_path=tokenizer_anchor,
        model_assembly_receipt=model_assembly_receipt,
        config_path=config_anchor,
        checkpoint_path=checkpoint_anchor,
        compiler_path=compiler_anchor,
        reactor_path=reactor_anchor,
        query_reader_path=reader_anchor,
    )
    anchor_bytes = {
        "config": config_anchor.read_bytes(),
        "checkpoint": checkpoint_anchor.read_bytes(),
        "compiler": compiler_anchor.read_bytes(),
        "reactor": reactor_anchor.read_bytes(),
        "reader": reader_anchor.read_bytes(),
        "world": world_anchor.read_bytes(),
        "command": command_anchor.read_bytes(),
        "query": query_anchor.read_bytes(),
        "manifest": manifest_anchor.read_bytes(),
    }
    shutil.rmtree(custody_dir)
    assert not custody_dir.exists()

    compiler_dir = tmp_path / "01_compiler"
    compiler_dir.mkdir()
    config = compiler_dir / "config.json"
    checkpoint = compiler_dir / "base.pt"
    compiler_weights = compiler_dir / "compiler.safetensors"
    world_path = compiler_dir / "world.json"
    compiled_state = compiler_dir / "state.safetensors"
    manifest_path = compiler_dir / "execution_manifest.json"
    compiler_receipt = compiler_dir / "compiler_receipt.json"
    config.write_bytes(anchor_bytes["config"])
    config.chmod(0o444)
    checkpoint.write_bytes(anchor_bytes["checkpoint"])
    checkpoint.chmod(0o444)
    compiler_weights.write_bytes(anchor_bytes["compiler"])
    compiler_weights.chmod(0o444)
    world_path.write_bytes(anchor_bytes["world"])
    world_path.chmod(0o444)
    manifest_path.write_bytes(anchor_bytes["manifest"])
    manifest_path.chmod(0o444)
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
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
    reactor_weights.write_bytes(anchor_bytes["reactor"])
    reactor_weights.chmod(0o444)
    checkpoint.write_bytes(checkpoint_bytes)
    checkpoint.chmod(0o444)
    command_path.write_bytes(anchor_bytes["command"])
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
    executor_receipt_bytes = executor_receipt.read_bytes()
    executor_receipt_sha256 = hashlib.sha256(
        executor_receipt_bytes
    ).hexdigest()
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

    query_dir = tmp_path / "03_query"
    query_dir.mkdir()
    config = query_dir / "config.json"
    terminal = query_dir / "terminal.safetensors"
    reader = query_dir / "reader.safetensors"
    checkpoint = query_dir / "base.pt"
    query_path = query_dir / "query.json"
    manifest_path = query_dir / "execution_manifest.json"
    executor_receipt = query_dir / "executor_receipt.json"
    candidate = query_dir / "candidate.json"
    query_receipt_path = query_dir / "query_receipt.json"
    config.write_bytes(config_bytes)
    config.chmod(0o444)
    terminal.write_bytes(terminal_bytes)
    terminal.chmod(0o444)
    reader.write_bytes(anchor_bytes["reader"])
    reader.chmod(0o444)
    checkpoint.write_bytes(checkpoint_bytes)
    checkpoint.chmod(0o444)
    query_path.write_bytes(anchor_bytes["query"])
    query_path.chmod(0o444)
    manifest_path.write_bytes(anchor_bytes["manifest"])
    manifest_path.chmod(0o444)
    executor_receipt.write_bytes(executor_receipt_bytes)
    executor_receipt.chmod(0o444)
    query_names = {path.name for path in query_dir.iterdir()}
    assert "world.json" not in query_names
    assert "command.json" not in query_names
    assert "compiler.safetensors" not in query_names
    assert "reactor.safetensors" not in query_names
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
        "--execution-manifest",
        str(manifest_path),
        "--execution-manifest-sha256",
        manifest.sha256(),
        "--executor-receipt",
        str(executor_receipt),
        "--executor-receipt-sha256",
        executor_receipt_sha256,
        "--tokenization-receipt-sha256",
        tokenization_receipt.sha256(),
        "--model-assembly-receipt-sha256",
        model_assembly_receipt.sha256(),
        "--output",
        str(candidate),
        "--receipt-output",
        str(query_receipt_path),
    )
    candidate_payload = json.loads(candidate.read_text())
    candidate_bytes = candidate.read_bytes()
    assert candidate_payload["schema"] == ANSWER_SCHEMA
    query_receipt = ETTRLateQueryExecutionReceipt(
        **json.loads(query_receipt_path.read_text())
    )
    query_receipt_sha256 = hashlib.sha256(
        query_receipt_path.read_bytes()
    ).hexdigest()
    query_receipt.validate(
        expected_receipt_sha256=query_receipt_sha256,
        execution_manifest_sha256=manifest.sha256(),
        tokenization_receipt_sha256=tokenization_receipt.sha256(),
        model_assembly_receipt_sha256=model_assembly_receipt.sha256(),
        executor_receipt_sha256=executor_receipt_sha256,
        terminal_state_file_sha256=(
            executor_receipt_record.output_state_file_sha256
        ),
        terminal_state_tensor_sha256=(
            executor_receipt_record.output_state_tensor_sha256
        ),
        query_tokens_sha256=manifest.query_tokens_sha256,
        reader_sha256=manifest.reader_sha256,
        checkpoint_sha256=manifest.checkpoint_sha256,
        row_count=TOTAL_PACKETS,
    )
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
