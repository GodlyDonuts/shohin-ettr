from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from ettr_factorial_qualification_board import (
    BOARD_SCHEMA,
    CLAIM_BOUNDARY,
    FOLD_ORDER,
    TOTAL_PACKETS,
    TOTAL_ROWS,
    build_ettr_factorial_qualification_board,
)


FROZEN_PAYLOAD_SHA256 = (
    "18686ff7f0476b5a4432830f2a301f693833cf867656d3997a010cf17bb0149a"
)
FROZEN_PACKAGE_SHA256S = (
    "88ed46c84c7a6721d5cf45ef56b31170b7c180e6bb243ab64bc5b5fbd56b6f17",
    "d74182ceac823ab0448de2371a5716c68f776e81b4f885d701bc23bac86d98cf",
    "781d9dde0faf466b224acd32ed982d9b39b083bf99cddecb0f4562d7f486e6e9",
    "097c0c769950a79eef9846e22e6f46d867e8debcf40322045bf4bb70597fb209",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_board_freezes_exact_staged_geometry_and_receipts() -> None:
    board = build_ettr_factorial_qualification_board()
    receipt = board.receipt
    assert receipt.schema == BOARD_SCHEMA
    assert receipt.fold_count == len(FOLD_ORDER) == 3
    assert receipt.world_count == 6
    assert receipt.command_count == 6
    assert receipt.packet_count == TOTAL_PACKETS == 12
    assert receipt.row_count == TOTAL_ROWS == 48
    assert receipt.independent_oracle_agreement_count == 12
    assert receipt.world_edge_change_count == 24
    assert receipt.command_edge_change_count == 24
    assert receipt.within_packet_target_contrast_count == 12
    assert receipt.candidate_label_leak_count == 0
    assert receipt.unique_row_count == 48
    assert receipt.payload_sha256 == FROZEN_PAYLOAD_SHA256
    assert receipt.claim_boundary == CLAIM_BOUNDARY
    assert receipt.all_contracts_pass

    assert Counter(row.fold for row in board.rows) == {fold: 16 for fold in FOLD_ORDER}
    assert len(board.packet_factor_ids) == 12
    assert len(board.world_factor_ids) == 6
    assert len(board.command_factor_ids) == 6


def test_each_packet_has_two_semantics_two_paraphrases_and_contrast() -> None:
    board = build_ettr_factorial_qualification_board()
    by_packet = {
        packet: [row for row in board.rows if row.packet_factor_id == packet]
        for packet in board.packet_factor_ids
    }
    for rows in by_packet.values():
        assert len(rows) == 4
        assert {row.semantic_index for row in rows} == {0, 1}
        assert {row.paraphrase_index for row in rows} == {0, 1}
        by_semantic = {
            semantic: {row.target for row in rows if row.semantic_index == semantic}
            for semantic in (0, 1)
        }
        assert all(len(targets) == 1 for targets in by_semantic.values())
        assert next(iter(by_semantic[0])) != next(iter(by_semantic[1]))


def test_factorial_edges_change_every_late_query_target() -> None:
    board = build_ettr_factorial_qualification_board()
    lookup = {
        (
            row.fold,
            row.world_index,
            row.command_index,
            row.semantic_index,
            row.paraphrase_index,
        ): row.target
        for row in board.rows
    }
    for row in board.rows:
        key = (
            row.fold,
            row.world_index,
            row.command_index,
            row.semantic_index,
            row.paraphrase_index,
        )
        wrong_world = (
            row.fold,
            row.world_index ^ 1,
            row.command_index,
            row.semantic_index,
            row.paraphrase_index,
        )
        wrong_command = (
            row.fold,
            row.world_index,
            row.command_index ^ 1,
            row.semantic_index,
            row.paraphrase_index,
        )
        assert lookup[key] != lookup[wrong_world]
        assert lookup[key] != lookup[wrong_command]


def test_stage_packages_are_hash_frozen_and_capability_separated() -> None:
    board = build_ettr_factorial_qualification_board()
    packages = (
        board.world_package_bytes(),
        board.command_package_bytes(),
        board.query_package_bytes(),
        board.assessor_package_bytes(),
    )
    assert tuple(_sha256(payload) for payload in packages) == (FROZEN_PACKAGE_SHA256S)
    decoded = tuple(json.loads(payload) for payload in packages)
    assert tuple(package["stage"] for package in decoded) == (1, 2, 3, 4)

    world_text = packages[0].lower()
    command_text = packages[1].lower()
    query_text = packages[2].lower()
    assessor_text = packages[3].lower()
    assert b"command_hex" not in world_text
    assert b"query_hex" not in world_text
    assert b'"target"' not in world_text
    assert b"world_hex" not in command_text
    assert b"query_hex" not in command_text
    assert b'"target"' not in command_text
    assert b"world_hex" not in query_text
    assert b"command_hex" not in query_text
    assert b'"target"' not in query_text
    assert b"world_hex" not in assessor_text
    assert b"command_hex" not in assessor_text
    assert b"query_hex" not in assessor_text


def test_physical_stage_deletion_leaves_only_the_next_capability(
    tmp_path: Path,
) -> None:
    board = build_ettr_factorial_qualification_board()
    packages = (
        board.world_package_bytes(),
        board.command_package_bytes(),
        board.query_package_bytes(),
        board.assessor_package_bytes(),
    )
    previous: Path | None = None
    for stage, payload in enumerate(packages, start=1):
        if previous is not None:
            shutil.rmtree(previous)
            assert not previous.exists()
        current = tmp_path / f"{stage:02d}"
        current.mkdir()
        artifact = current / "package.json"
        artifact.write_bytes(payload)
        assert json.loads(artifact.read_bytes())["stage"] == stage
        assert tuple(tmp_path.iterdir()) == (current,)
        previous = current


def test_candidate_surfaces_have_no_assessor_or_family_labels() -> None:
    board = build_ettr_factorial_qualification_board()
    forbidden = (
        b"answer",
        b"expected",
        b"family",
        b"horn",
        b"label",
        b"oracle",
        b"resource",
        b"rewrite",
        b"target",
        b"theory_index",
    )
    for row in board.rows:
        candidate = (
            row.world_bytes + row.command_bytes + row.query_prefix_bytes
        ).lower()
        assert candidate.isascii()
        assert all(token not in candidate for token in forbidden)
        assert row.fold.value.encode("ascii") not in candidate


def test_claim_boundary_rejects_capability_overclaim() -> None:
    boundary = CLAIM_BOUNDARY.lower()
    assert "evaluation" in boundary
    assert "not training" in boundary
    assert "not" in boundary
    assert "native reasoning" in boundary
    assert "general-reasoning claim" in boundary
