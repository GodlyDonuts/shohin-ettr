from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import torch
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
)
from ettr_factorial_qualification import (
    TERMINAL_ARTIFACT_SCHEMA,
    bind_terminal_state_artifact,
    materialize_ettr_factorial_qualification,
)
from ettr_factorial_custody import (
    ETTRFactorialExecutionManifest,
    ETTRStageExecutionReceipt,
    EXECUTION_MANIFEST_SCHEMA,
    STAGE_RECEIPT_SCHEMA,
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


def _artifact():
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
        world_package_sha256=board.receipt.world_package_sha256,
        command_package_sha256=board.receipt.command_package_sha256,
        world_tokens_sha256="5" * 64,
        command_tokens_sha256="6" * 64,
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
