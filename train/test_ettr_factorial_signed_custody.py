from __future__ import annotations

from dataclasses import replace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_factorial_signed_custody import (
    ETTRLateQueryExecutionReceipt,
    QUERY_RECEIPT_SCHEMA,
    _sign_custody_chain_unchecked,
)


def _query_receipt() -> ETTRLateQueryExecutionReceipt:
    return ETTRLateQueryExecutionReceipt(
        schema=QUERY_RECEIPT_SCHEMA,
        execution_manifest_sha256="1" * 64,
        tokenization_receipt_sha256="2" * 64,
        model_assembly_receipt_sha256="3" * 64,
        executor_receipt_sha256="4" * 64,
        terminal_state_file_sha256="5" * 64,
        terminal_state_tensor_sha256="6" * 64,
        query_tokens_sha256="7" * 64,
        reader_sha256="8" * 64,
        checkpoint_sha256="9" * 64,
        answer_file_sha256="a" * 64,
        answer_token_tensor_sha256="b" * 64,
        row_count=12,
    )


def test_query_receipt_binds_every_late_stage_input_and_output() -> None:
    receipt = _query_receipt()
    receipt.validate(
        expected_receipt_sha256=receipt.sha256(),
        execution_manifest_sha256="1" * 64,
        tokenization_receipt_sha256="2" * 64,
        model_assembly_receipt_sha256="3" * 64,
        executor_receipt_sha256="4" * 64,
        terminal_state_file_sha256="5" * 64,
        terminal_state_tensor_sha256="6" * 64,
        query_tokens_sha256="7" * 64,
        reader_sha256="8" * 64,
        checkpoint_sha256="9" * 64,
        row_count=12,
    )
    with pytest.raises(TheoryReactorError):
        replace(receipt, reader_sha256="c" * 64).validate(
            expected_receipt_sha256=receipt.sha256(),
            execution_manifest_sha256="1" * 64,
            tokenization_receipt_sha256="2" * 64,
            model_assembly_receipt_sha256="3" * 64,
            executor_receipt_sha256="4" * 64,
            terminal_state_file_sha256="5" * 64,
            terminal_state_tensor_sha256="6" * 64,
            query_tokens_sha256="7" * 64,
            reader_sha256="8" * 64,
            checkpoint_sha256="9" * 64,
            row_count=12,
        )


def test_external_signature_rejects_self_attestation_and_wrong_key() -> None:
    private_key = Ed25519PrivateKey.generate()
    seal = _sign_custody_chain_unchecked(
        private_key=private_key,
        board_sha256="0" * 64,
        model_sha256="1" * 64,
        execution_manifest_sha256="2" * 64,
        tokenization_receipt_sha256="3" * 64,
        model_assembly_receipt_sha256="4" * 64,
        compiler_receipt_sha256="5" * 64,
        executor_receipt_sha256="6" * 64,
        query_receipt_sha256="7" * 64,
        terminal_state_tensor_sha256="8" * 64,
        answer_token_tensor_sha256="9" * 64,
        qualification_batch_sha256="a" * 64,
        qualification_vocab_size=256,
        false_token_id=17,
        true_token_id=29,
        pad_token_id=255,
    )
    public_key_hex = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    seal.verify(
        expected_seal_sha256=seal.sha256(),
        expected_public_key_hex=public_key_hex,
        expected_board_sha256="0" * 64,
        expected_model_sha256="1" * 64,
        expected_authority_preregistration_sha256=(
            seal.authority_preregistration_sha256
        ),
    )
    forged = replace(seal, terminal_state_tensor_sha256="a" * 64)
    with pytest.raises(TheoryReactorError):
        forged.verify(
            expected_seal_sha256=forged.sha256(),
            expected_public_key_hex=public_key_hex,
            expected_board_sha256="0" * 64,
            expected_model_sha256="1" * 64,
            expected_authority_preregistration_sha256=(
                seal.authority_preregistration_sha256
            ),
        )
    other_public_key_hex = Ed25519PrivateKey.generate().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    with pytest.raises(TheoryReactorError):
        seal.verify(
            expected_seal_sha256=seal.sha256(),
            expected_public_key_hex=other_public_key_hex,
            expected_board_sha256="0" * 64,
            expected_model_sha256="1" * 64,
            expected_authority_preregistration_sha256=(
                seal.authority_preregistration_sha256
            ),
        )
