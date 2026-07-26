from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from ettr_factorial_qualification_board import (
    TOTAL_PACKETS,
    TOTAL_ROWS,
    build_ettr_factorial_qualification_board,
)
from ettr_factorial_tokenization import (
    ENCODING_SCHEMA,
    ETTRFactorialTokenizationError,
    ETTRFactorialTokenizationReceipt,
    TOKENIZATION_RECEIPT_SCHEMA,
    build_ettr_factorial_tokenization_receipt,
    canonical_json_bytes,
    sha256_bytes,
)


SEQ_LEN = 320
PAD_TOKEN_ID = 0


def _tokenizer(path: Path, *, reverse: bool = False) -> Path:
    alphabet = sorted(ByteLevel.alphabet(), reverse=reverse)
    vocab = {"<pad>": PAD_TOKEN_ID, "<unk>": 1}
    vocab.update(
        {
            token: index
            for index, token in enumerate(alphabet, start=2)
        }
    )
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    tokenizer.save(str(path))
    path.chmod(0o444)
    return path


@pytest.fixture
def frozen_tokenizer(tmp_path: Path) -> Path:
    return _tokenizer(tmp_path / "tokenizer.json")


def _receipt(tokenizer_path: Path) -> ETTRFactorialTokenizationReceipt:
    return build_ettr_factorial_tokenization_receipt(
        build_ettr_factorial_qualification_board(),
        tokenizer_path,
        seq_len=SEQ_LEN,
        pad_token_id=PAD_TOKEN_ID,
    )


def test_receipt_binds_all_raw_stages_and_emits_runner_payloads(
    frozen_tokenizer: Path,
) -> None:
    board = build_ettr_factorial_qualification_board()
    receipt = _receipt(frozen_tokenizer)

    assert receipt.schema == TOKENIZATION_RECEIPT_SCHEMA
    assert receipt.encoding_schema == ENCODING_SCHEMA
    assert receipt.board_sha256 == board.receipt.payload_sha256
    assert receipt.world_package_sha256 == board.receipt.world_package_sha256
    assert receipt.command_package_sha256 == board.receipt.command_package_sha256
    assert receipt.query_package_sha256 == board.receipt.query_package_sha256
    assert receipt.packet_factor_ids == board.packet_factor_ids
    assert len(receipt.world_rows) == TOTAL_PACKETS
    assert len(receipt.command_rows) == TOTAL_PACKETS
    assert len(receipt.query_rows) == TOTAL_PACKETS
    assert len(receipt.qualification_query_rows) == TOTAL_ROWS
    assert tuple(
        row.source_row_id for row in receipt.qualification_query_rows
    ) == tuple(row.row_id for row in board.rows)

    selected = tuple(
        row
        for row in board.rows
        if row.semantic_index == 0 and row.paraphrase_index == 0
    )
    for index, source in enumerate(selected):
        world = receipt.world_rows[index]
        command = receipt.command_rows[index]
        query = receipt.query_rows[index]
        assert world.packet_index == command.packet_index == query.packet_index == index
        assert world.packet_factor_id == source.packet_factor_id
        assert bytes.fromhex(world.raw_hex) == source.world_bytes
        assert bytes.fromhex(command.raw_hex) == source.command_bytes
        assert bytes.fromhex(query.raw_hex) == source.query_prefix_bytes
        assert world.raw_sha256 == sha256_bytes(source.world_bytes)
        assert command.raw_sha256 == sha256_bytes(source.command_bytes)
        assert query.raw_sha256 == sha256_bytes(source.query_prefix_bytes)

    for stage in ("world", "command", "query"):
        payload = receipt.stage_payload(stage)
        assert set(payload) == {"attention_mask", "token_ids"}
        assert len(payload["token_ids"]) == TOTAL_PACKETS
        assert len(payload["attention_mask"]) == TOTAL_PACKETS
        assert all(len(row) == SEQ_LEN for row in payload["token_ids"])
        assert all(len(row) == SEQ_LEN for row in payload["attention_mask"])
        for ids, mask in zip(
            payload["token_ids"],
            payload["attention_mask"],
            strict=True,
        ):
            boundary = sum(mask)
            assert mask == [1] * boundary + [0] * (SEQ_LEN - boundary)
            assert ids[boundary:] == [PAD_TOKEN_ID] * (SEQ_LEN - boundary)
        assert receipt.stage_payload_bytes(stage) == canonical_json_bytes(payload)


def test_receipt_is_canonical_round_trips_and_recomputes(
    frozen_tokenizer: Path,
    tmp_path: Path,
) -> None:
    board = build_ettr_factorial_qualification_board()
    receipt = _receipt(frozen_tokenizer)
    expected_sha256 = receipt.sha256()
    receipt.validate(
        board,
        frozen_tokenizer,
        expected_receipt_sha256=expected_sha256,
    )

    path = tmp_path / "receipt.json"
    assert receipt.write_once(path) == expected_sha256
    assert path.read_bytes() == canonical_json_bytes(json.loads(path.read_text()))
    loaded = ETTRFactorialTokenizationReceipt.from_path(path)
    assert loaded == receipt
    assert loaded.sha256() == expected_sha256
    loaded.validate(
        board,
        frozen_tokenizer,
        expected_receipt_sha256=expected_sha256,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "token",
        "mask",
        "row_order",
        "raw",
        "seq_len",
        "pad_token_id",
        "query_package",
        "selected_query_config",
    ),
)
def test_receipt_tampering_fails_closed(
    frozen_tokenizer: Path,
    mutation: str,
) -> None:
    board = build_ettr_factorial_qualification_board()
    receipt = _receipt(frozen_tokenizer)
    expected_sha256 = receipt.sha256()

    if mutation == "token":
        rows = list(receipt.world_rows)
        ids = list(rows[0].token_ids)
        ids[0] = ids[0] + 1
        rows[0] = replace(rows[0], token_ids=tuple(ids))
        tampered = replace(receipt, world_rows=tuple(rows))
    elif mutation == "mask":
        rows = list(receipt.command_rows)
        mask = list(rows[0].attention_mask)
        mask[0] = 0
        rows[0] = replace(rows[0], attention_mask=tuple(mask))
        tampered = replace(receipt, command_rows=tuple(rows))
    elif mutation == "row_order":
        rows = list(receipt.query_rows)
        rows[0], rows[1] = rows[1], rows[0]
        tampered = replace(receipt, query_rows=tuple(rows))
    elif mutation == "raw":
        rows = list(receipt.query_rows)
        rows[0] = replace(rows[0], raw_hex="00" + rows[0].raw_hex[2:])
        tampered = replace(receipt, query_rows=tuple(rows))
    elif mutation == "seq_len":
        tampered = replace(receipt, seq_len=receipt.seq_len + 1)
    elif mutation == "pad_token_id":
        tampered = replace(receipt, pad_token_id=1)
    elif mutation == "query_package":
        tampered = replace(receipt, query_package_sha256="f" * 64)
    elif mutation == "selected_query_config":
        tampered = replace(receipt, selected_query_semantic_index=1)
    else:
        raise AssertionError(mutation)

    with pytest.raises(
        ETTRFactorialTokenizationError,
        match="tokenization receipt",
    ):
        tampered.validate(
            board,
            frozen_tokenizer,
            expected_receipt_sha256=expected_sha256,
        )


def test_altered_tokenizer_and_noncanonical_receipt_fail_closed(
    frozen_tokenizer: Path,
    tmp_path: Path,
) -> None:
    board = build_ettr_factorial_qualification_board()
    receipt = _receipt(frozen_tokenizer)
    expected_sha256 = receipt.sha256()
    altered = _tokenizer(tmp_path / "altered-tokenizer.json", reverse=True)

    with pytest.raises(
        ETTRFactorialTokenizationError,
        match="tokenization receipt differs",
    ):
        receipt.validate(
            board,
            altered,
            expected_receipt_sha256=expected_sha256,
        )

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(receipt.to_dict(), indent=2),
        encoding="ascii",
    )
    noncanonical.chmod(0o444)
    with pytest.raises(
        ETTRFactorialTokenizationError,
        match="not canonical",
    ):
        ETTRFactorialTokenizationReceipt.from_path(noncanonical)


def test_tokenizer_must_be_immutable_regular_file(
    frozen_tokenizer: Path,
    tmp_path: Path,
) -> None:
    writable = tmp_path / "writable.json"
    writable.write_bytes(frozen_tokenizer.read_bytes())
    writable.chmod(0o644)
    with pytest.raises(
        ETTRFactorialTokenizationError,
        match="not immutable",
    ):
        _receipt(writable)

    link = tmp_path / "tokenizer-link.json"
    link.symlink_to(frozen_tokenizer)
    with pytest.raises(
        ETTRFactorialTokenizationError,
        match="not immutable",
    ):
        _receipt(link)


def test_encoding_configuration_fails_closed(
    frozen_tokenizer: Path,
) -> None:
    board = build_ettr_factorial_qualification_board()
    for seq_len, pad_token_id in (
        (0, PAD_TOKEN_ID),
        (SEQ_LEN, -1),
        (SEQ_LEN, 10_000),
        (8, PAD_TOKEN_ID),
    ):
        with pytest.raises(ETTRFactorialTokenizationError):
            build_ettr_factorial_tokenization_receipt(
                board,
                frozen_tokenizer,
                seq_len=seq_len,
                pad_token_id=pad_token_id,
            )
