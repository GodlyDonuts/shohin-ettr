from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

from endogenous_typed_theory_reactor import (
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
)
from ettr_factorial_qualification import (
    TERMINAL_ARTIFACT_SCHEMA,
    bind_terminal_state_artifact,
    materialize_ettr_factorial_qualification,
    materialize_signed_ettr_factorial_qualification,
)
from ettr_factorial_authority import (
    make_root_signed_ettr_custody_authority,
    write_ettr_custody_authority_once,
)
from ettr_factorial_custody import (
    ETTRFactorialExecutionManifest,
    ETTRStageExecutionReceipt,
    EXECUTION_MANIFEST_SCHEMA,
    STAGE_RECEIPT_SCHEMA,
)
from ettr_factorial_signed_custody import (
    ETTRLateQueryExecutionReceipt,
    ETTRSignedQualificationAdmission,
    QUERY_RECEIPT_SCHEMA,
    _sign_custody_chain_unchecked,
)
from ettr_factorial_tokenization import (
    build_ettr_factorial_tokenization_receipt,
)
from ettr_factorial_qualification_board import (
    TOTAL_PACKETS,
    TOTAL_ROWS,
    build_ettr_factorial_qualification_board,
)
from ettr_qualification import typed_state_row_sha256, typed_state_sha256


VOCAB_SIZE = 256


class _Encoded:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _ByteTokenizer:
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> _Encoded:
        assert not add_special_tokens
        return _Encoded(list(text.encode("ascii")))


class _OffsetTokenizer:
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> _Encoded:
        assert not add_special_tokens
        return _Encoded([(value + 1) % 128 for value in text.encode("ascii")])


def _config() -> TheoryReactorConfig:
    return TheoryReactorConfig(
        d_model=32,
        state_width=32,
        num_slots=2,
        num_types=2,
        num_relations=1,
        num_value_codes=16,
        max_edges=4,
        num_heads=4,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        ff_multiplier=2,
        max_steps=4,
        stage_after_block=1,
        parameter_cap=1_000_000,
    )


def _terminal_state() -> TypedTheoryState:
    packet = torch.arange(TOTAL_PACKETS)
    active = torch.ones(TOTAL_PACKETS, 2)
    values = torch.zeros(TOTAL_PACKETS, 2, 16)
    values[:, 0] = F.one_hot(packet + 1, 16).float()
    values[:, 1] = F.one_hot((packet % 3) + 13, 16).float()
    types = torch.zeros(TOTAL_PACKETS, 2, 2)
    types[:, 0, 0] = 1
    types[:, 1, 1] = 1
    relations = torch.zeros(TOTAL_PACKETS, 1, 2, 2)
    relations[packet.remainder(2).bool(), 0, 0, 1] = 1
    root = torch.zeros(TOTAL_PACKETS, 2)
    root[:, 0] = 1
    return TypedTheoryState(
        value_probabilities=values,
        type_probabilities=types,
        relations=relations,
        active=active,
        root=root,
        committed=torch.ones(TOTAL_PACKETS),
        halted=torch.zeros(TOTAL_PACKETS),
        step=2,
    )


def _artifact(
    *,
    tokenizer_sha256: str = "d" * 64,
    tokenization_receipt_sha256: str = "e" * 64,
    world_tokens_sha256: str = "5" * 64,
    command_tokens_sha256: str = "6" * 64,
    query_tokens_sha256: str = "0" * 64,
):
    board = build_ettr_factorial_qualification_board()
    state = _terminal_state()
    model_sha256 = "a" * 64
    manifest = ETTRFactorialExecutionManifest(
        schema=EXECUTION_MANIFEST_SCHEMA,
        board_sha256=board.receipt.payload_sha256,
        model_sha256=model_sha256,
        config_sha256="1" * 64,
        checkpoint_sha256="2" * 64,
        checkpoint_step=300_000,
        compiler_sha256="3" * 64,
        reactor_sha256="4" * 64,
        reader_sha256="c" * 64,
        tokenizer_sha256=tokenizer_sha256,
        tokenization_receipt_sha256=tokenization_receipt_sha256,
        model_assembly_receipt_sha256="f" * 64,
        bootstrap_sha256="b" * 64,
        world_runtime_bundle_sha256="0" * 64,
        command_runtime_bundle_sha256="a" * 64,
        query_runtime_bundle_sha256="b" * 64,
        claim_runtime_archive_sha256="1" * 64,
        claim_runtime_archive_size=1,
        claim_runtime_inventory_sha256="2" * 64,
        external_launcher_sha256="3" * 64,
        bwrap_sha256="4" * 64,
        network_namespace_required=True,
        world_stage_policy_sha256="5" * 64,
        command_stage_policy_sha256="6" * 64,
        query_stage_policy_sha256="7" * 64,
        compiler_runner_sha256="7" * 64,
        executor_runner_sha256="8" * 64,
        query_runner_sha256="9" * 64,
        compiler_hard=True,
        executor_hard=True,
        executor_steps=2,
        world_package_sha256=board.receipt.world_package_sha256,
        command_package_sha256=board.receipt.command_package_sha256,
        query_package_sha256=board.receipt.query_package_sha256,
        world_tokens_sha256=world_tokens_sha256,
        command_tokens_sha256=command_tokens_sha256,
        query_tokens_sha256=query_tokens_sha256,
        row_count=TOTAL_PACKETS,
    )
    manifest_sha256 = manifest.sha256()
    compiler_receipt = ETTRStageExecutionReceipt(
        schema=STAGE_RECEIPT_SCHEMA,
        stage="world",
        manifest_sha256=manifest_sha256,
        parent_receipt_sha256=None,
        input_state_file_sha256=None,
        input_state_tensor_sha256=None,
        token_input_sha256=manifest.world_tokens_sha256,
        component_sha256=manifest.compiler_sha256,
        checkpoint_sha256=manifest.checkpoint_sha256,
        output_state_file_sha256="7" * 64,
        output_state_tensor_sha256="8" * 64,
        row_count=TOTAL_PACKETS,
    )
    compiler_receipt_sha256 = compiler_receipt.sha256()
    executor_receipt = ETTRStageExecutionReceipt(
        schema=STAGE_RECEIPT_SCHEMA,
        stage="command",
        manifest_sha256=manifest_sha256,
        parent_receipt_sha256=compiler_receipt_sha256,
        input_state_file_sha256=compiler_receipt.output_state_file_sha256,
        input_state_tensor_sha256=compiler_receipt.output_state_tensor_sha256,
        token_input_sha256=manifest.command_tokens_sha256,
        component_sha256=manifest.reactor_sha256,
        checkpoint_sha256=manifest.checkpoint_sha256,
        output_state_file_sha256="9" * 64,
        output_state_tensor_sha256=typed_state_sha256(state),
        row_count=TOTAL_PACKETS,
    )
    executor_receipt_sha256 = executor_receipt.sha256()
    artifact = bind_terminal_state_artifact(
        board,
        state,
        execution_manifest=manifest,
        compiler_receipt=compiler_receipt,
        executor_receipt=executor_receipt,
        expected_model_sha256=model_sha256,
        expected_execution_manifest_sha256=manifest_sha256,
        expected_compiler_receipt_sha256=compiler_receipt_sha256,
        expected_executor_receipt_sha256=executor_receipt_sha256,
        config=_config(),
    )
    admission = {
        "expected_model_sha256": model_sha256,
        "expected_execution_manifest_sha256": manifest_sha256,
        "expected_compiler_receipt_sha256": compiler_receipt_sha256,
        "expected_executor_receipt_sha256": executor_receipt_sha256,
    }
    return board, state, artifact, admission


def _character_tokenizer(path: Path) -> Tokenizer:
    vocabulary = {chr(code): code for code in range(128)}
    vocabulary.update(
        {f"<extra-{code}>": code for code in range(128, 256)}
    )
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<extra-128>"))
    tokenizer.pre_tokenizer = Split("", behavior="isolated")
    tokenizer.save(str(path))
    path.chmod(0o444)
    return Tokenizer.from_file(str(path))


def test_terminal_artifact_binds_board_model_factors_and_packet_bytes() -> None:
    board, state, artifact, admission = _artifact()
    assert artifact.schema == TERMINAL_ARTIFACT_SCHEMA
    assert artifact.board_sha256 == board.receipt.payload_sha256
    assert artifact.model_sha256 == "a" * 64
    assert artifact.packet_factor_ids == board.packet_factor_ids
    assert artifact.packet_state_sha256s == tuple(
        typed_state_row_sha256(state, row) for row in range(TOTAL_PACKETS)
    )
    assert len(set(artifact.packet_state_sha256s)) == TOTAL_PACKETS
    artifact.validate(board, _config(), **admission)


def test_materializer_builds_real_frozen_qualification_geometry() -> None:
    board, _, artifact, admission = _artifact()
    batch = materialize_ettr_factorial_qualification(
        board,
        artifact,
        config=_config(),
        tokenizer=_ByteTokenizer(),
        tokenizer_sha256="b" * 64,
        vocab_size=VOCAB_SIZE,
        false_token_id=0,
        true_token_id=1,
        pad_token_id=255,
        **admission,
    )
    assert batch.targets.shape == (TOTAL_ROWS,)
    assert batch.terminal_state.active.shape[0] == TOTAL_ROWS
    assert len(set(batch.packet_ids)) == TOTAL_PACKETS
    assert len(set(batch.world_factor_ids)) == 6
    assert len(set(batch.command_factor_ids)) == 6
    assert len(set(batch.query_semantic_ids)) == 6
    assert len(set(batch.query_paraphrase_ids)) == 6
    assert set(batch.targets.tolist()) == {0, 1}
    batch.validate(_config(), vocab_size=VOCAB_SIZE)


def test_claim_bearing_materializer_requires_external_signed_chain(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = _character_tokenizer(tokenizer_path)
    frozen_board = build_ettr_factorial_qualification_board()
    tokenization_receipt = build_ettr_factorial_tokenization_receipt(
        frozen_board,
        tokenizer_path,
        seq_len=512,
        pad_token_id=255,
    )
    board, _, artifact, admission = _artifact(
        tokenizer_sha256=tokenization_receipt.tokenizer_sha256,
        tokenization_receipt_sha256=tokenization_receipt.sha256(),
        world_tokens_sha256=hashlib.sha256(
            tokenization_receipt.stage_payload_bytes("world")
        ).hexdigest(),
        command_tokens_sha256=hashlib.sha256(
            tokenization_receipt.stage_payload_bytes("command")
        ).hexdigest(),
        query_tokens_sha256=hashlib.sha256(
            tokenization_receipt.stage_payload_bytes("query")
        ).hexdigest(),
    )
    manifest = artifact.execution_manifest
    query_receipt = ETTRLateQueryExecutionReceipt(
        schema=QUERY_RECEIPT_SCHEMA,
        execution_manifest_sha256=manifest.sha256(),
        tokenization_receipt_sha256=manifest.tokenization_receipt_sha256,
        model_assembly_receipt_sha256=manifest.model_assembly_receipt_sha256,
        executor_receipt_sha256=artifact.executor_receipt.sha256(),
        terminal_state_file_sha256=(
            artifact.executor_receipt.output_state_file_sha256
        ),
        terminal_state_tensor_sha256=(
            artifact.executor_receipt.output_state_tensor_sha256
        ),
        query_tokens_sha256=manifest.query_tokens_sha256,
        reader_sha256=manifest.reader_sha256,
        checkpoint_sha256=manifest.checkpoint_sha256,
        answer_file_sha256="1" * 64,
        answer_token_tensor_sha256="2" * 64,
        row_count=TOTAL_PACKETS,
    )
    unsigned_batch = materialize_ettr_factorial_qualification(
        board,
        artifact,
        config=_config(),
        tokenizer=tokenizer,
        tokenizer_sha256=tokenization_receipt.tokenizer_sha256,
        vocab_size=VOCAB_SIZE,
        false_token_id=0,
        true_token_id=1,
        pad_token_id=255,
        **admission,
    )
    root_key = Ed25519PrivateKey.generate()
    private_key = Ed25519PrivateKey.generate()
    public_key_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    root_public_key_bytes = root_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    root_public_key_path = tmp_path / "custody-root.pub"
    root_public_key_path.write_bytes(root_public_key_bytes)
    root_public_key_path.chmod(0o444)
    pinned_root_public_key_sha256 = hashlib.sha256(
        root_public_key_bytes
    ).hexdigest()
    authority = make_root_signed_ettr_custody_authority(
        root_private_key=root_key,
        custody_public_key_hex=public_key_bytes.hex(),
        board_sha256=board.receipt.payload_sha256,
        execution_manifest_sha256=manifest.sha256(),
    )
    authority_path = tmp_path / "custody-authority.json"
    write_ettr_custody_authority_once(authority_path, authority)
    seal = _sign_custody_chain_unchecked(
        private_key=private_key,
        authority_record=authority,
        board_sha256=board.receipt.payload_sha256,
        model_sha256=artifact.model_sha256,
        execution_manifest_sha256=manifest.sha256(),
        tokenization_receipt_sha256=manifest.tokenization_receipt_sha256,
        model_assembly_receipt_sha256=manifest.model_assembly_receipt_sha256,
        compiler_receipt_sha256=artifact.compiler_receipt.sha256(),
        executor_receipt_sha256=artifact.executor_receipt.sha256(),
        query_receipt_sha256=query_receipt.sha256(),
        terminal_state_tensor_sha256=(
            artifact.executor_receipt.output_state_tensor_sha256
        ),
        answer_token_tensor_sha256=query_receipt.answer_token_tensor_sha256,
        qualification_batch_sha256=unsigned_batch.sha256(),
        qualification_vocab_size=VOCAB_SIZE,
        false_token_id=0,
        true_token_id=1,
        pad_token_id=255,
    )
    authority_arguments = {
        "authority_record_path": authority_path,
        "root_public_key_path": root_public_key_path,
        "pinned_root_public_key_sha256": pinned_root_public_key_sha256,
        "expected_authority_record_sha256": authority.sha256(),
    }
    batch = materialize_signed_ettr_factorial_qualification(
        board,
        artifact,
        ETTRSignedQualificationAdmission(query_receipt, seal),
        config=_config(),
        tokenizer=tokenizer,
        tokenizer_sha256=tokenization_receipt.tokenizer_sha256,
        vocab_size=VOCAB_SIZE,
        false_token_id=0,
        true_token_id=1,
        pad_token_id=255,
        tokenization_receipt=tokenization_receipt,
        tokenizer_path=tokenizer_path,
        expected_tokenization_receipt_sha256=tokenization_receipt.sha256(),
        expected_query_receipt_sha256=query_receipt.sha256(),
        expected_custody_seal_sha256=seal.sha256(),
        **authority_arguments,
        **admission,
    )
    assert batch.targets.shape == (TOTAL_ROWS,)
    with pytest.raises(TheoryReactorError):
        materialize_signed_ettr_factorial_qualification(
            board,
            artifact,
            ETTRSignedQualificationAdmission(query_receipt, seal),
            config=_config(),
            tokenizer=_OffsetTokenizer(),
            tokenizer_sha256=tokenization_receipt.tokenizer_sha256,
            vocab_size=VOCAB_SIZE,
            false_token_id=0,
            true_token_id=1,
            pad_token_id=255,
            tokenization_receipt=tokenization_receipt,
            tokenizer_path=tokenizer_path,
            expected_tokenization_receipt_sha256=(
                tokenization_receipt.sha256()
            ),
            expected_query_receipt_sha256=query_receipt.sha256(),
            expected_custody_seal_sha256=seal.sha256(),
            **authority_arguments,
            **admission,
        )
    with pytest.raises(TheoryReactorError):
        materialize_signed_ettr_factorial_qualification(
            board,
            artifact,
            ETTRSignedQualificationAdmission(query_receipt, seal),
            config=_config(),
            tokenizer=tokenizer,
            tokenizer_sha256=tokenization_receipt.tokenizer_sha256,
            vocab_size=VOCAB_SIZE,
            false_token_id=17,
            true_token_id=29,
            pad_token_id=255,
            tokenization_receipt=tokenization_receipt,
            tokenizer_path=tokenizer_path,
            expected_tokenization_receipt_sha256=(
                tokenization_receipt.sha256()
            ),
            expected_query_receipt_sha256=query_receipt.sha256(),
            expected_custody_seal_sha256=seal.sha256(),
            **authority_arguments,
            **admission,
        )
    with pytest.raises(TheoryReactorError):
        materialize_signed_ettr_factorial_qualification(
            board,
            artifact,
            ETTRSignedQualificationAdmission(
                query_receipt,
                replace(seal, answer_token_tensor_sha256="3" * 64),
            ),
            config=_config(),
            tokenizer=tokenizer,
            tokenizer_sha256=tokenization_receipt.tokenizer_sha256,
            vocab_size=VOCAB_SIZE,
            false_token_id=0,
            true_token_id=1,
            pad_token_id=255,
            tokenization_receipt=tokenization_receipt,
            tokenizer_path=tokenizer_path,
            expected_tokenization_receipt_sha256=(
                tokenization_receipt.sha256()
            ),
            expected_query_receipt_sha256=query_receipt.sha256(),
            expected_custody_seal_sha256=seal.sha256(),
            **authority_arguments,
            **admission,
        )


def test_controls_are_exact_factorial_counterfactuals() -> None:
    board, _, artifact, admission = _artifact()
    batch = materialize_ettr_factorial_qualification(
        board,
        artifact,
        config=_config(),
        tokenizer=_ByteTokenizer(),
        tokenizer_sha256="b" * 64,
        vocab_size=VOCAB_SIZE,
        false_token_id=0,
        true_token_id=1,
        pad_token_id=255,
        **admission,
    )
    row = torch.arange(TOTAL_ROWS)
    assert bool(
        (
            batch.targets
            != batch.targets.index_select(
                0,
                batch.wrong_world_state_index,
            )
        ).all()
    )
    assert bool(
        (
            batch.targets
            != batch.targets.index_select(
                0,
                batch.wrong_command_state_index,
            )
        ).all()
    )
    assert bool(
        (
            batch.targets
            != batch.targets.index_select(
                0,
                batch.query_twin_index,
            )
        ).all()
    )
    assert bool((batch.shuffled_state_index != row).all())
    assert not torch.equal(
        batch.target_derangement_index,
        batch.wrong_command_state_index,
    )


def test_provenance_tampering_fails_closed() -> None:
    board, _, artifact, admission = _artifact()
    swapped = list(artifact.packet_factor_ids)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(
        TheoryReactorError,
        match="terminal-state provenance differs",
    ):
        replace(
            artifact,
            packet_factor_ids=tuple(swapped),
        ).validate(board, _config(), **admission)

    forged_state_ids = list(artifact.packet_state_sha256s)
    forged_state_ids[0] = "f" * 64
    with pytest.raises(
        TheoryReactorError,
        match="terminal-state provenance differs",
    ):
        replace(
            artifact,
            packet_state_sha256s=tuple(forged_state_ids),
        ).validate(board, _config(), **admission)

    with pytest.raises(
        TheoryReactorError,
        match="execution manifest differs",
    ):
        artifact.validate(
            board,
            _config(),
            **{
                **admission,
                "expected_model_sha256": "f" * 64,
            },
        )

    with pytest.raises(
        TheoryReactorError,
        match="stage receipt differs",
    ):
        artifact.validate(
            board,
            _config(),
            **{
                **admission,
                "expected_executor_receipt_sha256": hashlib.sha256(
                    b"forged"
                ).hexdigest(),
            },
        )


def test_tokenizer_and_codebook_tampering_fail_closed() -> None:
    board, _, artifact, admission = _artifact()
    with pytest.raises(
        TheoryReactorError,
        match="answer codebook differs",
    ):
        materialize_ettr_factorial_qualification(
            board,
            artifact,
            config=_config(),
            tokenizer=_ByteTokenizer(),
            tokenizer_sha256="b" * 64,
            vocab_size=VOCAB_SIZE,
            false_token_id=1,
            true_token_id=1,
            pad_token_id=255,
            **admission,
        )
    with pytest.raises(
        TheoryReactorError,
        match="answer codebook differs",
    ):
        materialize_ettr_factorial_qualification(
            board,
            artifact,
            config=_config(),
            tokenizer=_ByteTokenizer(),
            tokenizer_sha256="not-a-hash",
            vocab_size=VOCAB_SIZE,
            false_token_id=0,
            true_token_id=1,
            pad_token_id=255,
            **admission,
        )
