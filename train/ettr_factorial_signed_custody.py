"""Externally signed custody for the complete ETTR qualification chain.

Candidate execution processes may emit hash-linked receipts, but they never
receive the assessor's private key.  The assessor signs only after validating
the complete compiler, executor, and late-query chain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
import torch

from endogenous_typed_theory_reactor import (
    TheoryReactorConfig,
    TheoryReactorError,
)
from ettr_factorial_authority import ETTRCustodyAuthorityRecord
from ettr_factorial_custody import (
    ETTRFactorialExecutionManifest,
    ETTRLateQueryExecutionReceipt,
    ETTRStageExecutionReceipt,
    QUERY_RECEIPT_SCHEMA,
    canonical_json_bytes,
    read_canonical_json,
    sha256_bytes,
    sha256_file,
    token_tensor_sha256,
)
from ettr_factorial_qualification_board import ETTRFactorialQualificationBoard
from ettr_factorial_tokenization import ETTRFactorialTokenizationReceipt
from ettr_model_assembly import ETTRModelAssemblyReceipt
from ettr_qualification import ETTRQualificationBatch
from ettr_state_io import read_state, typed_state_sha256


CUSTODY_SEAL_SCHEMA = "ettr-factorial-custody-seal-v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_32_BYTES = re.compile(r"^[0-9a-f]{64}$")
_HEX_64_BYTES = re.compile(r"^[0-9a-f]{128}$")


@dataclass(frozen=True, slots=True)
class ETTRCustodySeal:
    """Assessor signature over every qualification provenance edge."""

    schema: str
    board_sha256: str
    model_sha256: str
    execution_manifest_sha256: str
    tokenization_receipt_sha256: str
    model_assembly_receipt_sha256: str
    compiler_receipt_sha256: str
    executor_receipt_sha256: str
    query_receipt_sha256: str
    terminal_state_tensor_sha256: str
    answer_token_tensor_sha256: str
    qualification_batch_sha256: str
    qualification_vocab_size: int
    false_token_id: int
    true_token_id: int
    pad_token_id: int
    authority_record_sha256: str
    public_key_hex: str
    signature_hex: str

    def unsigned_payload(self) -> dict[str, object]:
        return asdict(replace(self, signature_hex=""))

    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(asdict(self)))

    def verify(
        self,
        *,
        authority_record: ETTRCustodyAuthorityRecord,
        expected_seal_sha256: str,
        expected_board_sha256: str,
        expected_model_sha256: str,
    ) -> None:
        hash_values = (
            expected_seal_sha256,
            self.board_sha256,
            self.model_sha256,
            self.execution_manifest_sha256,
            self.tokenization_receipt_sha256,
            self.model_assembly_receipt_sha256,
            self.compiler_receipt_sha256,
            self.executor_receipt_sha256,
            self.query_receipt_sha256,
            self.terminal_state_tensor_sha256,
            self.answer_token_tensor_sha256,
            self.qualification_batch_sha256,
            self.authority_record_sha256,
        )
        if (
            self.schema != CUSTODY_SEAL_SCHEMA
            or any(_SHA256.fullmatch(value) is None for value in hash_values)
            or _HEX_32_BYTES.fullmatch(self.public_key_hex) is None
            or _HEX_64_BYTES.fullmatch(self.signature_hex) is None
            or self.public_key_hex != authority_record.custody_public_key_hex
            or self.board_sha256 != expected_board_sha256
            or self.model_sha256 != expected_model_sha256
            or self.authority_record_sha256
            != authority_record.sha256()
            or self.board_sha256 != authority_record.board_sha256
            or self.execution_manifest_sha256
            != authority_record.execution_manifest_sha256
            or not isinstance(self.qualification_vocab_size, int)
            or isinstance(self.qualification_vocab_size, bool)
            or self.qualification_vocab_size < 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value < self.qualification_vocab_size
                for value in (
                    self.false_token_id,
                    self.true_token_id,
                    self.pad_token_id,
                )
            )
            or self.false_token_id == self.true_token_id
            or self.sha256() != expected_seal_sha256
        ):
            raise TheoryReactorError("external custody seal differs")
        try:
            public_key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(self.public_key_hex)
            )
            public_key.verify(
                bytes.fromhex(self.signature_hex),
                canonical_json_bytes(self.unsigned_payload()),
            )
        except (InvalidSignature, ValueError) as exc:
            raise TheoryReactorError("external custody signature differs") from exc


def _sign_custody_chain_unchecked(
    *,
    private_key: Ed25519PrivateKey,
    authority_record: ETTRCustodyAuthorityRecord,
    board_sha256: str,
    model_sha256: str,
    execution_manifest_sha256: str,
    tokenization_receipt_sha256: str,
    model_assembly_receipt_sha256: str,
    compiler_receipt_sha256: str,
    executor_receipt_sha256: str,
    query_receipt_sha256: str,
    terminal_state_tensor_sha256: str,
    answer_token_tensor_sha256: str,
    qualification_batch_sha256: str,
    qualification_vocab_size: int,
    false_token_id: int,
    true_token_id: int,
    pad_token_id: int,
) -> ETTRCustodySeal:
    """Sign a chain after assessor-side validation has completed."""

    public_key_hex = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    if (
        public_key_hex != authority_record.custody_public_key_hex
        or board_sha256 != authority_record.board_sha256
        or execution_manifest_sha256
        != authority_record.execution_manifest_sha256
    ):
        raise TheoryReactorError("custody signer authority differs")
    unsigned = ETTRCustodySeal(
        schema=CUSTODY_SEAL_SCHEMA,
        board_sha256=board_sha256,
        model_sha256=model_sha256,
        execution_manifest_sha256=execution_manifest_sha256,
        tokenization_receipt_sha256=tokenization_receipt_sha256,
        model_assembly_receipt_sha256=model_assembly_receipt_sha256,
        compiler_receipt_sha256=compiler_receipt_sha256,
        executor_receipt_sha256=executor_receipt_sha256,
        query_receipt_sha256=query_receipt_sha256,
        terminal_state_tensor_sha256=terminal_state_tensor_sha256,
        answer_token_tensor_sha256=answer_token_tensor_sha256,
        qualification_batch_sha256=qualification_batch_sha256,
        qualification_vocab_size=qualification_vocab_size,
        false_token_id=false_token_id,
        true_token_id=true_token_id,
        pad_token_id=pad_token_id,
        authority_record_sha256=authority_record.sha256(),
        public_key_hex=public_key_hex,
        signature_hex="",
    )
    signature = private_key.sign(
        canonical_json_bytes(unsigned.unsigned_payload())
    ).hex()
    return replace(unsigned, signature_hex=signature)


@dataclass(frozen=True, slots=True)
class ETTRSignedQualificationAdmission:
    """Public admission object whose trust root is external to candidates."""

    query_receipt: ETTRLateQueryExecutionReceipt
    custody_seal: ETTRCustodySeal

    def validate(
        self,
        *,
        execution_manifest: ETTRFactorialExecutionManifest,
        compiler_receipt: ETTRStageExecutionReceipt,
        executor_receipt: ETTRStageExecutionReceipt,
        authority_record: ETTRCustodyAuthorityRecord,
        expected_query_receipt_sha256: str,
        expected_seal_sha256: str,
        expected_board_sha256: str,
        expected_model_sha256: str,
        expected_qualification_batch_sha256: str,
        expected_qualification_vocab_size: int,
        expected_false_token_id: int,
        expected_true_token_id: int,
        expected_pad_token_id: int,
    ) -> None:
        self.query_receipt.validate(
            expected_receipt_sha256=expected_query_receipt_sha256,
            execution_manifest_sha256=execution_manifest.sha256(),
            tokenization_receipt_sha256=(
                execution_manifest.tokenization_receipt_sha256
            ),
            model_assembly_receipt_sha256=(
                execution_manifest.model_assembly_receipt_sha256
            ),
            executor_receipt_sha256=executor_receipt.sha256(),
            terminal_state_file_sha256=(
                executor_receipt.output_state_file_sha256
            ),
            terminal_state_tensor_sha256=(
                executor_receipt.output_state_tensor_sha256
            ),
            query_tokens_sha256=execution_manifest.query_tokens_sha256,
            reader_sha256=execution_manifest.reader_sha256,
            checkpoint_sha256=execution_manifest.checkpoint_sha256,
            row_count=execution_manifest.row_count,
        )
        self.custody_seal.verify(
            authority_record=authority_record,
            expected_seal_sha256=expected_seal_sha256,
            expected_board_sha256=expected_board_sha256,
            expected_model_sha256=expected_model_sha256,
        )
        if (
            self.custody_seal.execution_manifest_sha256
            != execution_manifest.sha256()
            or self.custody_seal.tokenization_receipt_sha256
            != execution_manifest.tokenization_receipt_sha256
            or self.custody_seal.model_assembly_receipt_sha256
            != execution_manifest.model_assembly_receipt_sha256
            or self.custody_seal.compiler_receipt_sha256
            != compiler_receipt.sha256()
            or self.custody_seal.executor_receipt_sha256
            != executor_receipt.sha256()
            or self.custody_seal.query_receipt_sha256
            != self.query_receipt.sha256()
            or self.custody_seal.terminal_state_tensor_sha256
            != executor_receipt.output_state_tensor_sha256
            or self.custody_seal.answer_token_tensor_sha256
            != self.query_receipt.answer_token_tensor_sha256
            or self.custody_seal.qualification_batch_sha256
            != expected_qualification_batch_sha256
            or self.custody_seal.qualification_vocab_size
            != expected_qualification_vocab_size
            or self.custody_seal.false_token_id != expected_false_token_id
            or self.custody_seal.true_token_id != expected_true_token_id
            or self.custody_seal.pad_token_id != expected_pad_token_id
        ):
            raise TheoryReactorError("signed qualification chain differs")


def validate_primary_custody_receipts(
    board: ETTRFactorialQualificationBoard,
    *,
    execution_manifest: ETTRFactorialExecutionManifest,
    expected_execution_manifest_sha256: str,
    tokenization_receipt: ETTRFactorialTokenizationReceipt,
    tokenizer_path: Path,
    model_assembly_receipt: ETTRModelAssemblyReceipt,
    config_path: Path,
    checkpoint_path: Path,
    compiler_path: Path,
    reactor_path: Path,
    query_reader_path: Path,
) -> None:
    """Recompute raw tokenization and complete model identity before signing."""

    tokenization_receipt.validate(
        board,
        tokenizer_path,
        expected_receipt_sha256=execution_manifest.tokenization_receipt_sha256,
    )
    model_assembly_receipt.validate(
        expected_receipt_sha256=execution_manifest.model_assembly_receipt_sha256,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        compiler_path=compiler_path,
        reactor_path=reactor_path,
        query_reader_path=query_reader_path,
    )
    execution_manifest.validate(
        board,
        expected_model_sha256=model_assembly_receipt.complete_model_sha256,
        expected_manifest_sha256=expected_execution_manifest_sha256,
    )
    if (
        execution_manifest.config_sha256
        != model_assembly_receipt.config_sha256
        or execution_manifest.checkpoint_sha256
        != model_assembly_receipt.checkpoint_sha256
        or execution_manifest.checkpoint_step
        != model_assembly_receipt.checkpoint_step
        or execution_manifest.compiler_sha256
        != model_assembly_receipt.compiler_sha256
        or execution_manifest.reactor_sha256
        != model_assembly_receipt.reactor_sha256
        or execution_manifest.reader_sha256
        != model_assembly_receipt.query_reader_sha256
        or execution_manifest.tokenizer_sha256
        != tokenization_receipt.tokenizer_sha256
        or execution_manifest.tokenization_receipt_sha256
        != tokenization_receipt.sha256()
        or execution_manifest.model_assembly_receipt_sha256
        != model_assembly_receipt.sha256()
        or execution_manifest.world_tokens_sha256
        != sha256_bytes(tokenization_receipt.stage_payload_bytes("world"))
        or execution_manifest.command_tokens_sha256
        != sha256_bytes(tokenization_receipt.stage_payload_bytes("command"))
        or execution_manifest.query_tokens_sha256
        != sha256_bytes(tokenization_receipt.stage_payload_bytes("query"))
    ):
        raise TheoryReactorError("primary custody receipt chain differs")


def sign_validated_custody_chain(
    board: ETTRFactorialQualificationBoard,
    *,
    private_key: Ed25519PrivateKey,
    authority_record: ETTRCustodyAuthorityRecord,
    execution_manifest: ETTRFactorialExecutionManifest,
    expected_execution_manifest_sha256: str,
    tokenization_receipt: ETTRFactorialTokenizationReceipt,
    tokenizer_path: Path,
    model_assembly_receipt: ETTRModelAssemblyReceipt,
    config: TheoryReactorConfig,
    config_path: Path,
    checkpoint_path: Path,
    compiler_path: Path,
    reactor_path: Path,
    query_reader_path: Path,
    compiler_receipt: ETTRStageExecutionReceipt,
    expected_compiler_receipt_sha256: str,
    executor_receipt: ETTRStageExecutionReceipt,
    expected_executor_receipt_sha256: str,
    query_receipt: ETTRLateQueryExecutionReceipt,
    expected_query_receipt_sha256: str,
    terminal_state_path: Path,
    query_tokens_path: Path,
    answer_path: Path,
    qualification_batch: ETTRQualificationBatch,
    qualification_vocab_size: int,
    false_token_id: int,
    true_token_id: int,
    pad_token_id: int,
) -> ETTRCustodySeal:
    """Validate every primary artifact and only then apply the external seal."""

    validate_primary_custody_receipts(
        board,
        execution_manifest=execution_manifest,
        expected_execution_manifest_sha256=expected_execution_manifest_sha256,
        tokenization_receipt=tokenization_receipt,
        tokenizer_path=tokenizer_path,
        model_assembly_receipt=model_assembly_receipt,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        compiler_path=compiler_path,
        reactor_path=reactor_path,
        query_reader_path=query_reader_path,
    )
    compiler_receipt.validate(
        execution_manifest,
        expected_receipt_sha256=expected_compiler_receipt_sha256,
    )
    executor_receipt.validate(
        execution_manifest,
        expected_receipt_sha256=expected_executor_receipt_sha256,
    )
    if (
        compiler_receipt.stage != "world"
        or executor_receipt.stage != "command"
        or executor_receipt.parent_receipt_sha256
        != expected_compiler_receipt_sha256
        or executor_receipt.input_state_file_sha256
        != compiler_receipt.output_state_file_sha256
        or executor_receipt.input_state_tensor_sha256
        != compiler_receipt.output_state_tensor_sha256
    ):
        raise TheoryReactorError("signed compiler-executor chain differs")
    query_receipt.validate(
        expected_receipt_sha256=expected_query_receipt_sha256,
        execution_manifest_sha256=expected_execution_manifest_sha256,
        tokenization_receipt_sha256=tokenization_receipt.sha256(),
        model_assembly_receipt_sha256=model_assembly_receipt.sha256(),
        executor_receipt_sha256=expected_executor_receipt_sha256,
        terminal_state_file_sha256=executor_receipt.output_state_file_sha256,
        terminal_state_tensor_sha256=executor_receipt.output_state_tensor_sha256,
        query_tokens_sha256=execution_manifest.query_tokens_sha256,
        reader_sha256=execution_manifest.reader_sha256,
        checkpoint_sha256=execution_manifest.checkpoint_sha256,
        row_count=execution_manifest.row_count,
    )
    config.validate()
    qualification_batch.validate(
        config,
        vocab_size=qualification_vocab_size,
    )
    terminal_state = read_state(terminal_state_path, config)
    answer_payload = read_canonical_json(answer_path)
    try:
        answer_tokens = torch.tensor(
            answer_payload["token_ids"],
            dtype=torch.long,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TheoryReactorError("signed answer tensor differs") from exc
    if (
        sha256_file(terminal_state_path)
        != executor_receipt.output_state_file_sha256
        or typed_state_sha256(terminal_state)
        != executor_receipt.output_state_tensor_sha256
        or sha256_file(query_tokens_path) != query_receipt.query_tokens_sha256
        or sha256_file(query_reader_path) != query_receipt.reader_sha256
        or sha256_file(answer_path) != query_receipt.answer_file_sha256
        or set(answer_payload) != {"schema", "token_ids"}
        or answer_payload["schema"] != "shohin-ettr-late-query-answer-v1"
        or answer_tokens.ndim != 2
        or answer_tokens.shape[0] != execution_manifest.row_count
        or token_tensor_sha256(answer_tokens)
        != query_receipt.answer_token_tensor_sha256
        or pad_token_id != tokenization_receipt.pad_token_id
        or set(int(value) for value in qualification_batch.targets.tolist())
        != {false_token_id, true_token_id}
    ):
        raise TheoryReactorError("signed primary output artifact differs")
    return _sign_custody_chain_unchecked(
        private_key=private_key,
        authority_record=authority_record,
        board_sha256=board.receipt.payload_sha256,
        model_sha256=model_assembly_receipt.complete_model_sha256,
        execution_manifest_sha256=expected_execution_manifest_sha256,
        tokenization_receipt_sha256=tokenization_receipt.sha256(),
        model_assembly_receipt_sha256=model_assembly_receipt.sha256(),
        compiler_receipt_sha256=expected_compiler_receipt_sha256,
        executor_receipt_sha256=expected_executor_receipt_sha256,
        query_receipt_sha256=expected_query_receipt_sha256,
        terminal_state_tensor_sha256=(
            executor_receipt.output_state_tensor_sha256
        ),
        answer_token_tensor_sha256=query_receipt.answer_token_tensor_sha256,
        qualification_batch_sha256=qualification_batch.sha256(),
        qualification_vocab_size=qualification_vocab_size,
        false_token_id=false_token_id,
        true_token_id=true_token_id,
        pad_token_id=pad_token_id,
    )


__all__ = [
    "CUSTODY_SEAL_SCHEMA",
    "ETTRCustodySeal",
    "ETTRLateQueryExecutionReceipt",
    "ETTRSignedQualificationAdmission",
    "QUERY_RECEIPT_SCHEMA",
    "sign_validated_custody_chain",
    "token_tensor_sha256",
    "validate_primary_custody_receipts",
]
