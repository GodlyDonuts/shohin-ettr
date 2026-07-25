from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from source_deleted_multifamily_machine_board import generate_episode  # noqa: E402
from multifamily_raw_machine_compiler import (  # noqa: E402
    collate_queries,
    collate_sources,
    scan_query,
    scan_source,
)
from multifamily_shohin_feature_adapter import (  # noqa: E402
    ConnectedFeatureReceipt,
    PROTECTED_SHOHIN_PARAMETERS,
    PROTECTED_SHOHIN_SHA256,
    anonymous_payload,
    build_trunk_batch,
    extract_query_unit_features,
    extract_source_unit_features,
)
from multifamily_raw_machine_compiler import MultiFamilyCompilerError  # noqa: E402


@dataclass
class _Encoding:
    ids: list[int]
    offsets: list[tuple[int, int]]


class _ByteTokenizer:
    def encode(self, text: str) -> _Encoding:
        payload = text.encode("ascii")
        return _Encoding(
            ids=[int(value) for value in payload],
            offsets=[(index, index + 1) for index in range(len(payload))],
        )


@dataclass
class _Features:
    values: torch.Tensor


class _FakeTrunk:
    feature_width = 4

    def encode_batch(self, batch):
        rows = []
        for payload in batch.payloads:
            values = torch.arange(
                len(payload) * self.feature_width,
                dtype=torch.float32,
                device=batch.token_ids.device,
            ).reshape(len(payload), self.feature_width)
            rows.append(values)
        maximum = max(row.shape[0] for row in rows)
        output = torch.zeros(
            (len(rows), maximum, self.feature_width),
            dtype=torch.float32,
            device=batch.token_ids.device,
        )
        for index, row in enumerate(rows):
            output[index, : row.shape[0]] = row
        return _Features(output)

    def flatten_byte_features(self, features: _Features) -> torch.Tensor:
        return features.values


def test_anonymous_payload_is_role_neutral_and_width_preserving() -> None:
    episode = generate_episode(
        seed=8,
        split="train",
        family="affine_modular",
        renderer=0,
        cell="fit",
    )
    payload = episode.candidate.source.splitlines()[0].encode("ascii")
    anonymous = anonymous_payload(payload)
    assert len(anonymous) == len(payload)
    assert b"h00000000000000000000" in anonymous
    assert payload != anonymous


def test_frozen_features_align_to_source_and_query_units() -> None:
    episode = generate_episode(
        seed=9,
        split="train",
        family="permutation",
        renderer=1,
        cell="fit",
    )
    scanned_source = scan_source(episode.candidate.source.encode("ascii"))
    scanned_query = scan_query(episode.candidate.query.encode("ascii"))
    source_batch = collate_sources((scanned_source,))
    query_batch = collate_queries((scanned_query,))
    trunk = _FakeTrunk()
    tokenizer = _ByteTokenizer()
    source_features, source_count, source_manifest = (
        extract_source_unit_features(
            trunk=trunk,
            tokenizer=tokenizer,
            scanned=(scanned_source,),
            batch=source_batch,
            chunk_size=5,
        )
    )
    query_features, query_count, query_manifest = (
        extract_query_unit_features(
            trunk=trunk,
            tokenizer=tokenizer,
            scanned=(scanned_query,),
            batch=query_batch,
            chunk_size=5,
        )
    )
    assert source_features.shape == (*source_batch.unit_ids.shape, 4)
    assert query_features.shape == (*query_batch.unit_ids.shape, 4)
    assert source_count == len(scanned_source.records)
    assert query_count == 1
    assert len(source_manifest) == len(query_manifest) == 64


def test_trunk_batch_offsets_partition_bytes() -> None:
    payloads = (b"abc", b"defgh")
    batch = build_trunk_batch(
        payloads,
        tokenizer=_ByteTokenizer(),
        device=torch.device("cpu"),
    )
    assert batch.payloads == payloads
    assert batch.token_valid.sum(1).tolist() == [3, 5]


def test_connected_receipt_fails_closed_when_checkpoint_is_not_verified() -> None:
    try:
        ConnectedFeatureReceipt(
            checkpoint_sha256=PROTECTED_SHOHIN_SHA256,
            checkpoint_verified=False,
            protected_parameters=PROTECTED_SHOHIN_PARAMETERS,
            frozen_feature_width=4,
            source_payload_count=1,
            query_payload_count=1,
            anonymous_source_manifest_sha256="0" * 64,
            anonymous_query_manifest_sha256="1" * 64,
        )
    except MultiFamilyCompilerError:
        pass
    else:
        raise AssertionError("unverified checkpoint receipt did not fail closed")
