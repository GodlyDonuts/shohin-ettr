"""Bind the frozen factorial board to source-deleted ETTR terminal packets.

This module does not compile WORLD, execute COMMAND, train weights, or call an
ontology oracle.  It accepts a terminal-state artifact produced under external
process custody, binds it to the already frozen board, tokenizes only the late
query prefixes, and materializes the immutable qualification batch consumed by
``ETTRQualificationHarness``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Protocol

import torch

from endogenous_typed_theory_reactor import (
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
    validate_deployed_state,
)
from ettr_factorial_custody import (
    ETTRFactorialExecutionManifest,
    ETTRStageExecutionReceipt,
)
from ettr_factorial_signed_custody import ETTRSignedQualificationAdmission
from ettr_factorial_tokenization import ETTRFactorialTokenizationReceipt
from ettr_qualification import (
    ETTRQualificationBatch,
    ETTRQualificationManifest,
    ETTR_QUALIFICATION_MANIFEST_SCHEMA,
    _prefix_bytes,
    typed_state_sha256,
    typed_state_row_sha256,
)
from ettr_factorial_qualification_board import (
    ETTRFactorialQualificationBoard,
    TOTAL_PACKETS,
    TOTAL_ROWS,
)


TERMINAL_ARTIFACT_SCHEMA = "ettr-factorial-terminal-state-artifact-v1"
MATERIALIZATION_SCHEMA = "ettr-factorial-query-materialization-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _Encoded(Protocol):
    ids: list[int]


class QueryTokenizer(Protocol):
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> _Encoded: ...


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _index_state(
    state: TypedTheoryState,
    index: torch.Tensor,
) -> TypedTheoryState:
    return TypedTheoryState(
        value_probabilities=state.value_probabilities.index_select(0, index),
        type_probabilities=state.type_probabilities.index_select(0, index),
        relations=state.relations.index_select(0, index),
        active=state.active.index_select(0, index),
        root=state.root.index_select(0, index),
        committed=state.committed.index_select(0, index),
        halted=state.halted.index_select(0, index),
        step=state.step,
    )


def _packet_rows(
    board: ETTRFactorialQualificationBoard,
) -> tuple[Any, ...]:
    packets = []
    seen: set[str] = set()
    for row in board.rows:
        if row.packet_factor_id in seen:
            continue
        seen.add(row.packet_factor_id)
        packets.append(row)
    if len(packets) != TOTAL_PACKETS:
        raise TheoryReactorError("factorial board packet geometry differs")
    return tuple(packets)


@dataclass(frozen=True, slots=True)
class ETTRTerminalStateArtifact:
    """Hash-bound packet output from the external WORLD/COMMAND processes."""

    schema: str
    board_sha256: str
    model_sha256: str
    execution_manifest_sha256: str
    compiler_receipt_sha256: str
    executor_receipt_sha256: str
    execution_manifest: ETTRFactorialExecutionManifest
    compiler_receipt: ETTRStageExecutionReceipt
    executor_receipt: ETTRStageExecutionReceipt
    packet_factor_ids: tuple[str, ...]
    world_factor_ids: tuple[str, ...]
    command_factor_ids: tuple[str, ...]
    packet_state_sha256s: tuple[str, ...]
    terminal_state: TypedTheoryState

    def sha256(self) -> str:
        payload = {
            "schema": self.schema,
            "board_sha256": self.board_sha256,
            "model_sha256": self.model_sha256,
            "execution_manifest_sha256": self.execution_manifest_sha256,
            "compiler_receipt_sha256": self.compiler_receipt_sha256,
            "executor_receipt_sha256": self.executor_receipt_sha256,
            "packet_factor_ids": list(self.packet_factor_ids),
            "world_factor_ids": list(self.world_factor_ids),
            "command_factor_ids": list(self.command_factor_ids),
            "packet_state_sha256s": list(self.packet_state_sha256s),
            "step": self.terminal_state.step,
        }
        return _sha256(payload)

    def validate(
        self,
        board: ETTRFactorialQualificationBoard,
        config: TheoryReactorConfig,
        *,
        expected_model_sha256: str,
        expected_execution_manifest_sha256: str,
        expected_compiler_receipt_sha256: str,
        expected_executor_receipt_sha256: str,
    ) -> None:
        self.execution_manifest.validate(
            board,
            expected_model_sha256=expected_model_sha256,
            expected_manifest_sha256=expected_execution_manifest_sha256,
        )
        self.compiler_receipt.validate(
            self.execution_manifest,
            expected_receipt_sha256=expected_compiler_receipt_sha256,
        )
        self.executor_receipt.validate(
            self.execution_manifest,
            expected_receipt_sha256=expected_executor_receipt_sha256,
        )
        validate_deployed_state(self.terminal_state, config)
        packets = _packet_rows(board)
        expected_packet_ids = tuple(row.packet_factor_id for row in packets)
        expected_world_ids = tuple(row.world_factor_id for row in packets)
        expected_command_ids = tuple(row.command_factor_id for row in packets)
        observed_state_ids = tuple(
            typed_state_row_sha256(self.terminal_state, row)
            for row in range(self.terminal_state.active.shape[0])
        )
        if (
            self.schema != TERMINAL_ARTIFACT_SCHEMA
            or self.board_sha256 != board.receipt.payload_sha256
            or self.model_sha256 != expected_model_sha256
            or self.execution_manifest_sha256 != expected_execution_manifest_sha256
            or self.compiler_receipt_sha256 != expected_compiler_receipt_sha256
            or self.executor_receipt_sha256 != expected_executor_receipt_sha256
            or self.execution_manifest.sha256() != self.execution_manifest_sha256
            or self.compiler_receipt.sha256() != self.compiler_receipt_sha256
            or self.executor_receipt.sha256() != self.executor_receipt_sha256
            or self.compiler_receipt.stage != "world"
            or self.executor_receipt.stage != "command"
            or self.executor_receipt.parent_receipt_sha256
            != self.compiler_receipt_sha256
            or self.executor_receipt.input_state_file_sha256
            != self.compiler_receipt.output_state_file_sha256
            or self.executor_receipt.input_state_tensor_sha256
            != self.compiler_receipt.output_state_tensor_sha256
            or self.executor_receipt.output_state_tensor_sha256
            != typed_state_sha256(self.terminal_state)
            or self.terminal_state.active.shape[0] != TOTAL_PACKETS
            or self.packet_factor_ids != expected_packet_ids
            or self.world_factor_ids != expected_world_ids
            or self.command_factor_ids != expected_command_ids
            or self.packet_state_sha256s != observed_state_ids
        ):
            raise TheoryReactorError("factorial terminal-state provenance differs")


def bind_terminal_state_artifact(
    board: ETTRFactorialQualificationBoard,
    terminal_state: TypedTheoryState,
    *,
    execution_manifest: ETTRFactorialExecutionManifest,
    compiler_receipt: ETTRStageExecutionReceipt,
    executor_receipt: ETTRStageExecutionReceipt,
    expected_model_sha256: str,
    expected_execution_manifest_sha256: str,
    expected_compiler_receipt_sha256: str,
    expected_executor_receipt_sha256: str,
    config: TheoryReactorConfig,
) -> ETTRTerminalStateArtifact:
    """Freeze packet order and bytes after externally custodied execution."""

    packets = _packet_rows(board)
    cloned = terminal_state.detached_clone()
    artifact = ETTRTerminalStateArtifact(
        schema=TERMINAL_ARTIFACT_SCHEMA,
        board_sha256=board.receipt.payload_sha256,
        model_sha256=expected_model_sha256,
        execution_manifest_sha256=expected_execution_manifest_sha256,
        compiler_receipt_sha256=expected_compiler_receipt_sha256,
        executor_receipt_sha256=expected_executor_receipt_sha256,
        execution_manifest=execution_manifest,
        compiler_receipt=compiler_receipt,
        executor_receipt=executor_receipt,
        packet_factor_ids=tuple(row.packet_factor_id for row in packets),
        world_factor_ids=tuple(row.world_factor_id for row in packets),
        command_factor_ids=tuple(row.command_factor_id for row in packets),
        packet_state_sha256s=tuple(
            typed_state_row_sha256(cloned, row) for row in range(cloned.active.shape[0])
        ),
        terminal_state=cloned,
    )
    artifact.validate(
        board,
        config,
        expected_model_sha256=expected_model_sha256,
        expected_execution_manifest_sha256=expected_execution_manifest_sha256,
        expected_compiler_receipt_sha256=expected_compiler_receipt_sha256,
        expected_executor_receipt_sha256=expected_executor_receipt_sha256,
    )
    return artifact


def _control_indices(
    board: ETTRFactorialQualificationBoard,
) -> tuple[torch.Tensor, ...]:
    lookup = {
        (
            row.fold,
            row.world_index,
            row.command_index,
            row.semantic_index,
            row.paraphrase_index,
        ): index
        for index, row in enumerate(board.rows)
    }

    def donor(
        row: Any,
        *,
        world_flip: bool = False,
        command_flip: bool = False,
        semantic_flip: bool = False,
        paraphrase_flip: bool = False,
    ) -> int:
        return lookup[
            (
                row.fold,
                row.world_index ^ int(world_flip),
                row.command_index ^ int(command_flip),
                row.semantic_index ^ int(semantic_flip),
                row.paraphrase_index ^ int(paraphrase_flip),
            )
        ]

    shuffled = []
    wrong_world = []
    wrong_command = []
    query_twin = []
    target_derangement = []
    shuffled_packet = {
        (0, 0): (0, 1),
        (0, 1): (1, 1),
        (1, 1): (1, 0),
        (1, 0): (0, 0),
    }
    for row in board.rows:
        shuffled_world, shuffled_command = shuffled_packet[
            (row.world_index, row.command_index)
        ]
        shuffled.append(
            lookup[
                (
                    row.fold,
                    shuffled_world,
                    shuffled_command,
                    row.semantic_index,
                    row.paraphrase_index,
                )
            ]
        )
        wrong_world.append(donor(row, world_flip=True))
        wrong_command.append(donor(row, command_flip=True))
        query_twin.append(donor(row, semantic_flip=True))
        target_derangement.append(
            donor(
                row,
                command_flip=True,
                paraphrase_flip=True,
            )
        )
    return tuple(
        torch.tensor(values, dtype=torch.long)
        for values in (
            shuffled,
            wrong_world,
            wrong_command,
            query_twin,
            target_derangement,
        )
    )


def materialize_ettr_factorial_qualification(
    board: ETTRFactorialQualificationBoard,
    artifact: ETTRTerminalStateArtifact,
    *,
    config: TheoryReactorConfig,
    tokenizer: QueryTokenizer,
    tokenizer_sha256: str,
    vocab_size: int,
    false_token_id: int,
    true_token_id: int,
    pad_token_id: int,
    expected_model_sha256: str,
    expected_execution_manifest_sha256: str,
    expected_compiler_receipt_sha256: str,
    expected_executor_receipt_sha256: str,
) -> ETTRQualificationBatch:
    """Create the frozen late-query batch without exposing targets to forwards."""

    if (
        _SHA256.fullmatch(tokenizer_sha256) is None
        or not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size < 2
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < vocab_size
            for value in (
                false_token_id,
                true_token_id,
                pad_token_id,
            )
        )
        or false_token_id == true_token_id
    ):
        raise TheoryReactorError("factorial tokenizer or answer codebook differs")
    if len(board.rows) != TOTAL_ROWS:
        raise TheoryReactorError("factorial board row count differs")
    artifact.validate(
        board,
        config,
        expected_model_sha256=expected_model_sha256,
        expected_execution_manifest_sha256=expected_execution_manifest_sha256,
        expected_compiler_receipt_sha256=expected_compiler_receipt_sha256,
        expected_executor_receipt_sha256=expected_executor_receipt_sha256,
    )

    packet_lookup = {
        packet_id: index for index, packet_id in enumerate(artifact.packet_factor_ids)
    }
    state_index = torch.tensor(
        [packet_lookup[row.packet_factor_id] for row in board.rows],
        dtype=torch.long,
        device=artifact.terminal_state.active.device,
    )
    terminal_state = _index_state(
        artifact.terminal_state,
        state_index,
    )

    encoded_prefixes: list[list[int]] = []
    for row in board.rows:
        try:
            text = row.query_prefix_bytes.decode("ascii")
            encoded = tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            token_ids = list(encoded.ids)
        except (AttributeError, TypeError, UnicodeDecodeError) as error:
            raise TheoryReactorError(
                "factorial query tokenizer contract differs"
            ) from error
        if not token_ids or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < vocab_size
            for value in token_ids
        ):
            raise TheoryReactorError("factorial query prefix tokenization differs")
        encoded_prefixes.append(token_ids)

    prefix_width = max(len(prefix) for prefix in encoded_prefixes)
    maximum = prefix_width + 1
    device = artifact.terminal_state.active.device
    query_tokens = torch.full(
        (TOTAL_ROWS, maximum),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    query_attention_mask = torch.ones(
        (TOTAL_ROWS, maximum),
        dtype=torch.bool,
        device=device,
    )
    query_read_index = torch.empty(
        TOTAL_ROWS,
        dtype=torch.long,
        device=device,
    )
    targets = torch.empty(
        TOTAL_ROWS,
        dtype=torch.long,
        device=device,
    )
    for index, (row, prefix) in enumerate(
        zip(board.rows, encoded_prefixes, strict=True)
    ):
        target = true_token_id if row.target else false_token_id
        values = torch.tensor(
            prefix,
            dtype=torch.long,
            device=device,
        )
        offset = prefix_width - values.numel()
        query_tokens[index, offset:prefix_width] = values
        query_tokens[index, prefix_width] = target
        query_read_index[index] = prefix_width - 1
        targets[index] = target

    packet_ids = tuple(
        typed_state_row_sha256(terminal_state, row) for row in range(TOTAL_ROWS)
    )
    world_factor_ids = tuple(row.world_factor_id for row in board.rows)
    command_factor_ids = tuple(row.command_factor_id for row in board.rows)
    query_semantic_ids = tuple(row.query_semantic_id for row in board.rows)
    query_paraphrase_ids = tuple(row.query_paraphrase_id for row in board.rows)
    dataset_sha256 = _sha256(
        {
            "schema": MATERIALIZATION_SCHEMA,
            "board_sha256": board.receipt.payload_sha256,
            "terminal_artifact_sha256": artifact.sha256(),
            "tokenizer_sha256": tokenizer_sha256,
            "vocab_size": vocab_size,
            "false_token_id": false_token_id,
            "true_token_id": true_token_id,
            "pad_token_id": pad_token_id,
        }
    )
    manifest = ETTRQualificationManifest(
        schema=ETTR_QUALIFICATION_MANIFEST_SCHEMA,
        dataset_sha256=dataset_sha256,
        producer_model_sha256=artifact.model_sha256,
        row_ids=tuple(row.row_id for row in board.rows),
        packet_ids=packet_ids,
        world_factor_ids=world_factor_ids,
        command_factor_ids=command_factor_ids,
        query_semantic_ids=query_semantic_ids,
        query_paraphrase_ids=query_paraphrase_ids,
        query_prefix_sha256s=tuple(
            hashlib.sha256(
                _prefix_bytes(
                    query_tokens,
                    query_attention_mask,
                    query_read_index,
                    row,
                )
            ).hexdigest()
            for row in range(TOTAL_ROWS)
        ),
        target_token_ids=tuple(int(value) for value in targets),
    )
    (
        shuffled,
        wrong_world,
        wrong_command,
        query_twin,
        target_derangement,
    ) = (index.to(device=device) for index in _control_indices(board))
    batch = ETTRQualificationBatch(
        terminal_state=terminal_state,
        manifest=manifest,
        query_tokens=query_tokens,
        query_attention_mask=query_attention_mask,
        query_read_index=query_read_index,
        targets=targets,
        packet_ids=packet_ids,
        world_factor_ids=world_factor_ids,
        command_factor_ids=command_factor_ids,
        query_semantic_ids=query_semantic_ids,
        query_paraphrase_ids=query_paraphrase_ids,
        shuffled_state_index=shuffled,
        wrong_world_state_index=wrong_world,
        wrong_command_state_index=wrong_command,
        query_twin_index=query_twin,
        target_derangement_index=target_derangement,
    )
    batch.validate(config, vocab_size=vocab_size)
    return batch


def materialize_signed_ettr_factorial_qualification(
    board: ETTRFactorialQualificationBoard,
    artifact: ETTRTerminalStateArtifact,
    signed_admission: ETTRSignedQualificationAdmission,
    *,
    config: TheoryReactorConfig,
    tokenizer: QueryTokenizer,
    tokenizer_sha256: str,
    vocab_size: int,
    false_token_id: int,
    true_token_id: int,
    pad_token_id: int,
    expected_model_sha256: str,
    expected_execution_manifest_sha256: str,
    expected_compiler_receipt_sha256: str,
    expected_executor_receipt_sha256: str,
    tokenization_receipt: ETTRFactorialTokenizationReceipt,
    tokenizer_path: Path,
    expected_tokenization_receipt_sha256: str,
    expected_query_receipt_sha256: str,
    expected_custody_seal_sha256: str,
    expected_custody_public_key_hex: str,
    expected_authority_preregistration_sha256: str,
) -> ETTRQualificationBatch:
    """Claim-bearing materialization gated by an external Ed25519 trust root."""

    tokenization_receipt.validate(
        board,
        tokenizer_path,
        expected_receipt_sha256=expected_tokenization_receipt_sha256,
    )
    if (
        expected_tokenization_receipt_sha256
        != artifact.execution_manifest.tokenization_receipt_sha256
        or tokenization_receipt.sha256()
        != artifact.execution_manifest.tokenization_receipt_sha256
        or tokenization_receipt.tokenizer_sha256
        != artifact.execution_manifest.tokenizer_sha256
        or tokenizer_sha256 != tokenization_receipt.tokenizer_sha256
    ):
        raise TheoryReactorError("signed tokenizer admission differs")
    artifact.validate(
        board,
        config,
        expected_model_sha256=expected_model_sha256,
        expected_execution_manifest_sha256=expected_execution_manifest_sha256,
        expected_compiler_receipt_sha256=expected_compiler_receipt_sha256,
        expected_executor_receipt_sha256=expected_executor_receipt_sha256,
    )
    batch = materialize_ettr_factorial_qualification(
        board,
        artifact,
        config=config,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        vocab_size=vocab_size,
        false_token_id=false_token_id,
        true_token_id=true_token_id,
        pad_token_id=pad_token_id,
        expected_model_sha256=expected_model_sha256,
        expected_execution_manifest_sha256=(
            expected_execution_manifest_sha256
        ),
        expected_compiler_receipt_sha256=expected_compiler_receipt_sha256,
        expected_executor_receipt_sha256=expected_executor_receipt_sha256,
    )
    if len(tokenization_receipt.qualification_query_rows) != TOTAL_ROWS:
        raise TheoryReactorError("signed query token row count differs")
    for row_index, receipt_row in enumerate(
        tokenization_receipt.qualification_query_rows
    ):
        prefix = receipt_row.token_ids[: receipt_row.unpadded_length]
        read_index = int(batch.query_read_index[row_index])
        start = read_index + 1 - len(prefix)
        if (
            receipt_row.source_row_id != board.rows[row_index].row_id
            or start < 0
            or tuple(
                int(value)
                for value in batch.query_tokens[
                    row_index,
                    start : read_index + 1,
                ]
            )
            != prefix
            or any(
                int(value) != pad_token_id
                for value in batch.query_tokens[row_index, :start]
            )
        ):
            raise TheoryReactorError(
                "signed qualification query tokenization differs"
            )
    signed_admission.validate(
        execution_manifest=artifact.execution_manifest,
        compiler_receipt=artifact.compiler_receipt,
        executor_receipt=artifact.executor_receipt,
        expected_query_receipt_sha256=expected_query_receipt_sha256,
        expected_seal_sha256=expected_custody_seal_sha256,
        expected_public_key_hex=expected_custody_public_key_hex,
        expected_board_sha256=board.receipt.payload_sha256,
        expected_model_sha256=expected_model_sha256,
        expected_qualification_batch_sha256=batch.sha256(),
        expected_qualification_vocab_size=vocab_size,
        expected_false_token_id=false_token_id,
        expected_true_token_id=true_token_id,
        expected_pad_token_id=pad_token_id,
        expected_authority_preregistration_sha256=(
            expected_authority_preregistration_sha256
        ),
    )
    return batch


__all__ = [
    "ETTRTerminalStateArtifact",
    "MATERIALIZATION_SCHEMA",
    "QueryTokenizer",
    "TERMINAL_ARTIFACT_SCHEMA",
    "bind_terminal_state_artifact",
    "materialize_ettr_factorial_qualification",
    "materialize_signed_ettr_factorial_qualification",
]
