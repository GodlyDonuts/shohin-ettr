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
from ettr_claim_runtime import (
    ETTRClaimRuntimeError,
    ETTRClaimRuntimeVerificationReceipt,
)
from ettr_factorial_authority import (
    AUTHORITY_SCHEMA,
    AUTHORIZED_SEAL_SCHEMA,
    ETTRCustodyAuthorityRecord,
)
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
from ettr_deployment_contract import (
    ETTRRuntimeImageIdentity,
    ETTRStageLaunchReceipt,
    validate_stage_launch_receipt_chain,
)
from ettr_model_assembly import ETTRModelAssemblyReceipt
from ettr_qualification import ETTRQualificationBatch
from ettr_state_io import read_state, typed_state_sha256


CUSTODY_SEAL_SCHEMA = "ettr-factorial-custody-seal-v4"
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
    world_launch_receipt_sha256: str
    command_launch_receipt_sha256: str
    query_launch_receipt_sha256: str
    claim_runtime_verification_receipt_sha256: str
    launch_verifier_public_key_fingerprint: str
    launch_run_id: str
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
            self.world_launch_receipt_sha256,
            self.command_launch_receipt_sha256,
            self.query_launch_receipt_sha256,
            self.claim_runtime_verification_receipt_sha256,
            self.launch_verifier_public_key_fingerprint,
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
            or authority_record.schema != AUTHORITY_SCHEMA
            or authority_record.authorized_seal_schema != AUTHORIZED_SEAL_SCHEMA
            or AUTHORIZED_SEAL_SCHEMA != CUSTODY_SEAL_SCHEMA
            or self.board_sha256 != expected_board_sha256
            or self.model_sha256 != expected_model_sha256
            or self.authority_record_sha256 != authority_record.sha256()
            or self.board_sha256 != authority_record.board_sha256
            or self.execution_manifest_sha256
            != authority_record.execution_manifest_sha256
            or self.claim_runtime_verification_receipt_sha256
            != authority_record.claim_runtime_verification_receipt_sha256
            or self.launch_verifier_public_key_fingerprint
            != authority_record.launch_verifier_public_key_fingerprint
            or not isinstance(self.launch_run_id, str)
            or not self.launch_run_id
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
    world_launch_receipt_sha256: str,
    command_launch_receipt_sha256: str,
    query_launch_receipt_sha256: str,
    claim_runtime_verification_receipt_sha256: str,
    launch_verifier_public_key_fingerprint: str,
    launch_run_id: str,
    terminal_state_tensor_sha256: str,
    answer_token_tensor_sha256: str,
    qualification_batch_sha256: str,
    qualification_vocab_size: int,
    false_token_id: int,
    true_token_id: int,
    pad_token_id: int,
) -> ETTRCustodySeal:
    """Sign a chain after assessor-side validation has completed."""

    public_key_hex = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )
    if (
        public_key_hex != authority_record.custody_public_key_hex
        or board_sha256 != authority_record.board_sha256
        or execution_manifest_sha256 != authority_record.execution_manifest_sha256
        or claim_runtime_verification_receipt_sha256
        != authority_record.claim_runtime_verification_receipt_sha256
        or launch_verifier_public_key_fingerprint
        != authority_record.launch_verifier_public_key_fingerprint
        or not isinstance(launch_run_id, str)
        or not launch_run_id
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
        world_launch_receipt_sha256=world_launch_receipt_sha256,
        command_launch_receipt_sha256=command_launch_receipt_sha256,
        query_launch_receipt_sha256=query_launch_receipt_sha256,
        claim_runtime_verification_receipt_sha256=(
            claim_runtime_verification_receipt_sha256
        ),
        launch_verifier_public_key_fingerprint=(launch_verifier_public_key_fingerprint),
        launch_run_id=launch_run_id,
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


def _validate_root_bound_launch_chain(
    *,
    authority_record: ETTRCustodyAuthorityRecord,
    claim_runtime_verification_receipt: (ETTRClaimRuntimeVerificationReceipt),
    runtime_identity: ETTRRuntimeImageIdentity,
    world_launch_receipt: ETTRStageLaunchReceipt,
    command_launch_receipt: ETTRStageLaunchReceipt,
    query_launch_receipt: ETTRStageLaunchReceipt,
) -> str:
    """Authenticate verifier ownership, one-run lineage, and runtime bytes."""

    try:
        claim_runtime_verification_receipt.validate()
        launch_verifier_public_key_bytes = bytes.fromhex(
            authority_record.launch_verifier_public_key_hex
        )
        launch_verifier_public_key = Ed25519PublicKey.from_public_bytes(
            launch_verifier_public_key_bytes
        )
    except (ETTRClaimRuntimeError, ValueError) as exc:
        raise TheoryReactorError("root-bound launch authority differs") from exc
    launch_verifier_fingerprint = sha256_bytes(launch_verifier_public_key_bytes)
    receipts = (
        world_launch_receipt,
        command_launch_receipt,
        query_launch_receipt,
    )
    if (
        type(claim_runtime_verification_receipt)
        is not ETTRClaimRuntimeVerificationReceipt
        or any(type(receipt) is not ETTRStageLaunchReceipt for receipt in receipts)
        or authority_record.schema != AUTHORITY_SCHEMA
        or authority_record.authorized_seal_schema != CUSTODY_SEAL_SCHEMA
        or launch_verifier_fingerprint
        != authority_record.launch_verifier_public_key_fingerprint
        or claim_runtime_verification_receipt.sha256()
        != authority_record.claim_runtime_verification_receipt_sha256
        or claim_runtime_verification_receipt.archive_sha256
        != runtime_identity.archive_sha256
        or claim_runtime_verification_receipt.archive_size
        != runtime_identity.archive_size
        or claim_runtime_verification_receipt.inventory_sha256
        != runtime_identity.inventory_sha256
        or claim_runtime_verification_receipt.python_sha256
        != runtime_identity.python_sha256
        or claim_runtime_verification_receipt.bootstrap_sha256
        != runtime_identity.bootstrap_sha256
        or any(
            getattr(
                receipt,
                "launch_verifier_public_key_fingerprint",
                getattr(receipt, "verifier_public_key_sha256", None),
            )
            != launch_verifier_fingerprint
            for receipt in receipts
        )
    ):
        raise TheoryReactorError("root-bound launch authority differs")
    try:
        for receipt in receipts:
            if hasattr(receipt, "verify_verifier_signature"):
                signature_result = receipt.verify_verifier_signature(
                    launch_verifier_public_key_bytes
                )
            else:
                signature_result = receipt.validate_signature(
                    launch_verifier_public_key
                )
            if signature_result is False:
                raise TheoryReactorError("stage launch receipt signature differs")
    except (AttributeError, InvalidSignature, TypeError, ValueError) as exc:
        raise TheoryReactorError("stage launch receipt signature differs") from exc
    run_id = world_launch_receipt.run_id
    if (
        not isinstance(run_id, str)
        or not run_id
        or command_launch_receipt.run_id != run_id
        or query_launch_receipt.run_id != run_id
        or world_launch_receipt.parent_launch_receipt_sha256 is not None
        or command_launch_receipt.parent_launch_receipt_sha256
        != world_launch_receipt.sha256()
        or query_launch_receipt.parent_launch_receipt_sha256
        != command_launch_receipt.sha256()
    ):
        raise TheoryReactorError("stage launch receipt lineage differs")
    return run_id


def _validate_launch_artifact_bindings(
    *,
    execution_manifest: ETTRFactorialExecutionManifest,
    compiler_receipt: ETTRStageExecutionReceipt,
    executor_receipt: ETTRStageExecutionReceipt,
    query_receipt: ETTRLateQueryExecutionReceipt,
    claim_runtime_verification_receipt: (ETTRClaimRuntimeVerificationReceipt),
    runtime_identity: ETTRRuntimeImageIdentity,
    world_launch_receipt: ETTRStageLaunchReceipt,
    command_launch_receipt: ETTRStageLaunchReceipt,
    query_launch_receipt: ETTRStageLaunchReceipt,
) -> None:
    reconstructed_identity = ETTRRuntimeImageIdentity.from_manifest(
        asdict(execution_manifest),
        python_sha256=claim_runtime_verification_receipt.python_sha256,
    )
    canonical_policy_hashes = {
        stage: receipt.stage_policy_sha256
        for stage, receipt in (
            ("world", world_launch_receipt),
            ("command", command_launch_receipt),
            ("query", query_launch_receipt),
        )
    }
    manifest_policy_hashes = {
        "world": execution_manifest.world_stage_policy_sha256,
        "command": execution_manifest.command_stage_policy_sha256,
        "query": execution_manifest.query_stage_policy_sha256,
    }
    expected_inputs = {
        "world": {
            "application_bundle": (execution_manifest.world_runtime_bundle_sha256),
            "checkpoint": execution_manifest.checkpoint_sha256,
            "compiler_weights": execution_manifest.compiler_sha256,
            "configuration": execution_manifest.config_sha256,
            "execution_manifest": execution_manifest.sha256(),
            "runtime_bundle_receipt": (execution_manifest.world_runtime_bundle_sha256),
            "runtime_image": execution_manifest.claim_runtime_archive_sha256,
            "world_tokens": execution_manifest.world_tokens_sha256,
        },
        "command": {
            "application_bundle": (execution_manifest.command_runtime_bundle_sha256),
            "checkpoint": execution_manifest.checkpoint_sha256,
            "command_tokens": execution_manifest.command_tokens_sha256,
            "compiled_state": compiler_receipt.output_state_file_sha256,
            "compiler_receipt": compiler_receipt.sha256(),
            "configuration": execution_manifest.config_sha256,
            "execution_manifest": execution_manifest.sha256(),
            "reactor_weights": execution_manifest.reactor_sha256,
            "runtime_bundle_receipt": (
                execution_manifest.command_runtime_bundle_sha256
            ),
            "runtime_image": execution_manifest.claim_runtime_archive_sha256,
        },
        "query": {
            "application_bundle": (execution_manifest.query_runtime_bundle_sha256),
            "checkpoint": execution_manifest.checkpoint_sha256,
            "configuration": execution_manifest.config_sha256,
            "execution_manifest": execution_manifest.sha256(),
            "executor_receipt": executor_receipt.sha256(),
            "query_reader_weights": execution_manifest.reader_sha256,
            "query_tokens": execution_manifest.query_tokens_sha256,
            "runtime_bundle_receipt": (execution_manifest.query_runtime_bundle_sha256),
            "runtime_image": execution_manifest.claim_runtime_archive_sha256,
            "terminal_state": executor_receipt.output_state_file_sha256,
        },
    }
    expected_outputs = {
        "world": {
            "compiled_state_output": (compiler_receipt.output_state_file_sha256),
            "compiler_receipt_output": compiler_receipt.sha256(),
        },
        "command": {
            "executor_receipt_output": executor_receipt.sha256(),
            "terminal_state_output": executor_receipt.output_state_file_sha256,
        },
        "query": {
            "answer_output": query_receipt.answer_file_sha256,
            "query_receipt_output": query_receipt.sha256(),
        },
    }
    actual_receipts = {
        "world": world_launch_receipt,
        "command": command_launch_receipt,
        "query": query_launch_receipt,
    }
    if (
        reconstructed_identity != runtime_identity
        or canonical_policy_hashes != manifest_policy_hashes
        or any(
            dict(actual_receipts[stage].input_role_sha256s) != expected_inputs[stage]
            or dict(actual_receipts[stage].output_role_sha256s)
            != expected_outputs[stage]
            for stage in ("world", "command", "query")
        )
    ):
        raise TheoryReactorError("measured launch artifact chain differs")


@dataclass(frozen=True, slots=True)
class ETTRSignedQualificationAdmission:
    """Public admission object whose trust root is external to candidates."""

    query_receipt: ETTRLateQueryExecutionReceipt
    world_launch_receipt: ETTRStageLaunchReceipt
    command_launch_receipt: ETTRStageLaunchReceipt
    query_launch_receipt: ETTRStageLaunchReceipt
    custody_seal: ETTRCustodySeal

    def validate(
        self,
        *,
        execution_manifest: ETTRFactorialExecutionManifest,
        compiler_receipt: ETTRStageExecutionReceipt,
        executor_receipt: ETTRStageExecutionReceipt,
        claim_runtime_verification_receipt: (ETTRClaimRuntimeVerificationReceipt),
        runtime_identity: ETTRRuntimeImageIdentity,
        authority_record: ETTRCustodyAuthorityRecord,
        expected_query_receipt_sha256: str,
        expected_world_launch_receipt_sha256: str,
        expected_command_launch_receipt_sha256: str,
        expected_query_launch_receipt_sha256: str,
        expected_seal_sha256: str,
        expected_board_sha256: str,
        expected_model_sha256: str,
        expected_qualification_batch_sha256: str,
        expected_qualification_vocab_size: int,
        expected_false_token_id: int,
        expected_true_token_id: int,
        expected_pad_token_id: int,
    ) -> None:
        if (
            type(self) is not ETTRSignedQualificationAdmission
            or type(self.query_receipt) is not ETTRLateQueryExecutionReceipt
            or type(self.custody_seal) is not ETTRCustodySeal
        ):
            raise TheoryReactorError("signed admission type differs")
        launch_run_id = _validate_root_bound_launch_chain(
            authority_record=authority_record,
            claim_runtime_verification_receipt=(claim_runtime_verification_receipt),
            runtime_identity=runtime_identity,
            world_launch_receipt=self.world_launch_receipt,
            command_launch_receipt=self.command_launch_receipt,
            query_launch_receipt=self.query_launch_receipt,
        )
        launch_sha256s = validate_stage_launch_receipt_chain(
            receipts={
                "world": self.world_launch_receipt,
                "command": self.command_launch_receipt,
                "query": self.query_launch_receipt,
            },
            runtime_identity=runtime_identity,
            expected_execution_manifest_sha256=execution_manifest.sha256(),
            expected_verifier_public_key=bytes.fromhex(
                authority_record.launch_verifier_public_key_hex
            ),
        )
        if launch_sha256s != (
            expected_world_launch_receipt_sha256,
            expected_command_launch_receipt_sha256,
            expected_query_launch_receipt_sha256,
        ):
            raise TheoryReactorError("stage launch receipt chain differs")
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
            terminal_state_file_sha256=(executor_receipt.output_state_file_sha256),
            terminal_state_tensor_sha256=(executor_receipt.output_state_tensor_sha256),
            query_tokens_sha256=execution_manifest.query_tokens_sha256,
            reader_sha256=execution_manifest.reader_sha256,
            checkpoint_sha256=execution_manifest.checkpoint_sha256,
            row_count=execution_manifest.row_count,
        )
        _validate_launch_artifact_bindings(
            execution_manifest=execution_manifest,
            compiler_receipt=compiler_receipt,
            executor_receipt=executor_receipt,
            query_receipt=self.query_receipt,
            claim_runtime_verification_receipt=(claim_runtime_verification_receipt),
            runtime_identity=runtime_identity,
            world_launch_receipt=self.world_launch_receipt,
            command_launch_receipt=self.command_launch_receipt,
            query_launch_receipt=self.query_launch_receipt,
        )
        self.custody_seal.verify(
            authority_record=authority_record,
            expected_seal_sha256=expected_seal_sha256,
            expected_board_sha256=expected_board_sha256,
            expected_model_sha256=expected_model_sha256,
        )
        if (
            self.custody_seal.execution_manifest_sha256 != execution_manifest.sha256()
            or self.custody_seal.tokenization_receipt_sha256
            != execution_manifest.tokenization_receipt_sha256
            or self.custody_seal.model_assembly_receipt_sha256
            != execution_manifest.model_assembly_receipt_sha256
            or self.custody_seal.compiler_receipt_sha256 != compiler_receipt.sha256()
            or self.custody_seal.executor_receipt_sha256 != executor_receipt.sha256()
            or self.custody_seal.query_receipt_sha256 != self.query_receipt.sha256()
            or self.custody_seal.world_launch_receipt_sha256
            != self.world_launch_receipt.sha256()
            or self.custody_seal.command_launch_receipt_sha256
            != self.command_launch_receipt.sha256()
            or self.custody_seal.query_launch_receipt_sha256
            != self.query_launch_receipt.sha256()
            or self.custody_seal.claim_runtime_verification_receipt_sha256
            != claim_runtime_verification_receipt.sha256()
            or self.custody_seal.launch_verifier_public_key_fingerprint
            != authority_record.launch_verifier_public_key_fingerprint
            or self.custody_seal.launch_run_id != launch_run_id
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
        execution_manifest.config_sha256 != model_assembly_receipt.config_sha256
        or execution_manifest.checkpoint_sha256
        != model_assembly_receipt.checkpoint_sha256
        or execution_manifest.checkpoint_step != model_assembly_receipt.checkpoint_step
        or execution_manifest.compiler_sha256 != model_assembly_receipt.compiler_sha256
        or execution_manifest.reactor_sha256 != model_assembly_receipt.reactor_sha256
        or execution_manifest.reader_sha256
        != model_assembly_receipt.query_reader_sha256
        or execution_manifest.tokenizer_sha256 != tokenization_receipt.tokenizer_sha256
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
    claim_runtime_verification_receipt: (ETTRClaimRuntimeVerificationReceipt),
    runtime_identity: ETTRRuntimeImageIdentity,
    world_launch_receipt: ETTRStageLaunchReceipt,
    expected_world_launch_receipt_sha256: str,
    command_launch_receipt: ETTRStageLaunchReceipt,
    expected_command_launch_receipt_sha256: str,
    query_launch_receipt: ETTRStageLaunchReceipt,
    expected_query_launch_receipt_sha256: str,
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
        or executor_receipt.parent_receipt_sha256 != expected_compiler_receipt_sha256
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
    launch_run_id = _validate_root_bound_launch_chain(
        authority_record=authority_record,
        claim_runtime_verification_receipt=(claim_runtime_verification_receipt),
        runtime_identity=runtime_identity,
        world_launch_receipt=world_launch_receipt,
        command_launch_receipt=command_launch_receipt,
        query_launch_receipt=query_launch_receipt,
    )
    launch_sha256s = validate_stage_launch_receipt_chain(
        receipts={
            "world": world_launch_receipt,
            "command": command_launch_receipt,
            "query": query_launch_receipt,
        },
        runtime_identity=runtime_identity,
        expected_execution_manifest_sha256=(expected_execution_manifest_sha256),
        expected_verifier_public_key=bytes.fromhex(
            authority_record.launch_verifier_public_key_hex
        ),
    )
    if launch_sha256s != (
        expected_world_launch_receipt_sha256,
        expected_command_launch_receipt_sha256,
        expected_query_launch_receipt_sha256,
    ):
        raise TheoryReactorError("stage launch receipt chain differs")
    _validate_launch_artifact_bindings(
        execution_manifest=execution_manifest,
        compiler_receipt=compiler_receipt,
        executor_receipt=executor_receipt,
        query_receipt=query_receipt,
        claim_runtime_verification_receipt=(claim_runtime_verification_receipt),
        runtime_identity=runtime_identity,
        world_launch_receipt=world_launch_receipt,
        command_launch_receipt=command_launch_receipt,
        query_launch_receipt=query_launch_receipt,
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
        sha256_file(terminal_state_path) != executor_receipt.output_state_file_sha256
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
        world_launch_receipt_sha256=(expected_world_launch_receipt_sha256),
        command_launch_receipt_sha256=(expected_command_launch_receipt_sha256),
        query_launch_receipt_sha256=(expected_query_launch_receipt_sha256),
        claim_runtime_verification_receipt_sha256=(
            claim_runtime_verification_receipt.sha256()
        ),
        launch_verifier_public_key_fingerprint=(
            authority_record.launch_verifier_public_key_fingerprint
        ),
        launch_run_id=launch_run_id,
        terminal_state_tensor_sha256=(executor_receipt.output_state_tensor_sha256),
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
