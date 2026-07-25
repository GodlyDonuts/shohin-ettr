"""Hash-bound frozen-Shohin features for the multi-family raw compiler.

Every opaque key is replaced by the same width-preserving role-neutral token
before the frozen language model is evaluated. Literal key bytes remain only
in the exact equality partition and sealed key table. Source and query
residuals are projected to masked byte units, consumed by the learned role
compiler, and forbidden from the deployed machine wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Protocol, Sequence

import torch

from episode_functor_shohin_trunk import ShohinTrunkBatch
from multifamily_raw_machine_compiler import (
    MultiFamilyCompilerError,
    QueryTensorBatch,
    ScannedQuery,
    ScannedSource,
    SourceTensorBatch,
    project_byte_features_to_units,
)


PROTECTED_SHOHIN_SHA256 = (
    "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6"
)
PROTECTED_SHOHIN_PARAMETERS = 125_081_664
_OPAQUE_KEY = re.compile(rb"(?<![A-Za-z0-9])h[0-9a-f]{20}(?![A-Za-z0-9])")
_ANONYMOUS_KEY = b"h00000000000000000000"


class TokenizerProtocol(Protocol):
    def encode(self, text: str): ...


@dataclass(frozen=True, slots=True)
class ConnectedFeatureReceipt:
    checkpoint_sha256: str
    checkpoint_verified: bool
    protected_parameters: int
    frozen_feature_width: int
    source_payload_count: int
    query_payload_count: int
    anonymous_source_manifest_sha256: str
    anonymous_query_manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            self.checkpoint_sha256 != PROTECTED_SHOHIN_SHA256
            or type(self.checkpoint_verified) is not bool
            or self.protected_parameters != PROTECTED_SHOHIN_PARAMETERS
            or self.frozen_feature_width < 1
            or self.source_payload_count < 1
            or self.query_payload_count < 1
        ):
            raise MultiFamilyCompilerError("connected feature receipt differs")
        for digest in (
            self.anonymous_source_manifest_sha256,
            self.anonymous_query_manifest_sha256,
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise MultiFamilyCompilerError(
                    "anonymous feature manifest digest differs"
                )


def anonymous_payload(payload: bytes) -> bytes:
    """Erase literal key identity while preserving every byte offset."""

    if not isinstance(payload, bytes) or not payload:
        raise MultiFamilyCompilerError("anonymous payload input differs")
    output = _OPAQUE_KEY.sub(_ANONYMOUS_KEY, payload)
    if len(output) != len(payload):
        raise MultiFamilyCompilerError("anonymous payload changed byte length")
    if _OPAQUE_KEY.search(output) is None:
        raise MultiFamilyCompilerError("anonymous payload contains no key slots")
    return output


def _manifest_sha256(payloads: Sequence[bytes]) -> str:
    digest = sha256(b"MULTIFAMILY-ANONYMOUS-PAYLOAD-MANIFEST-V1\0")
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_trunk_batch(
    payloads: Sequence[bytes],
    *,
    tokenizer: TokenizerProtocol,
    device: torch.device,
) -> ShohinTrunkBatch:
    """Tokenize ASCII payloads with exact byte-partition validation."""

    payload_tuple = tuple(payloads)
    if not payload_tuple:
        raise MultiFamilyCompilerError("frozen trunk payload batch is empty")
    encoded: list[tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = []
    for payload in payload_tuple:
        try:
            result = tokenizer.encode(payload.decode("ascii"))
            ids = tuple(int(value) for value in result.ids)
            offsets = tuple(
                (int(start), int(end)) for start, end in result.offsets
            )
        except (
            AttributeError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            raise MultiFamilyCompilerError(
                "frozen trunk tokenization failed"
            ) from exc
        coverage = [0] * len(payload)
        for start, end in offsets:
            if not 0 <= start < end <= len(payload):
                raise MultiFamilyCompilerError(
                    "frozen tokenizer offset leaves payload"
                )
            for index in range(start, end):
                coverage[index] += 1
        if (
            not ids
            or len(ids) != len(offsets)
            or any(value < 0 for value in ids)
            or any(value != 1 for value in coverage)
        ):
            raise MultiFamilyCompilerError(
                "frozen tokenizer does not partition payload bytes"
            )
        encoded.append((ids, offsets))
    maximum = max(len(ids) for ids, _ in encoded)
    token_ids = torch.zeros(
        (len(encoded), maximum),
        dtype=torch.long,
        device=device,
    )
    token_valid = torch.zeros_like(token_ids, dtype=torch.bool)
    token_bounds = torch.zeros(
        (len(encoded), maximum, 2),
        dtype=torch.int32,
        device=device,
    )
    for row, (ids, offsets) in enumerate(encoded):
        count = len(ids)
        token_ids[row, :count] = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )
        token_valid[row, :count] = True
        token_bounds[row, :count] = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=device,
        )
    return ShohinTrunkBatch(
        payloads=payload_tuple,
        token_ids=token_ids,
        token_valid=token_valid,
        token_byte_bounds=token_bounds,
    )


def _extract_flat_byte_features(
    *,
    trunk,
    tokenizer: TokenizerProtocol,
    payloads: tuple[bytes, ...],
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if chunk_size < 1:
        raise MultiFamilyCompilerError("frozen feature chunk size differs")
    output: list[torch.Tensor] = []
    for start in range(0, len(payloads), chunk_size):
        chunk = payloads[start : start + chunk_size]
        trunk_batch = build_trunk_batch(
            chunk,
            tokenizer=tokenizer,
            device=device,
        )
        with torch.no_grad():
            encoded = trunk.encode_batch(trunk_batch)
            flattened = trunk.flatten_byte_features(encoded)
        for row, payload in enumerate(chunk):
            output.append(flattened[row, : len(payload)].detach())
    return tuple(output)


def extract_source_unit_features(
    *,
    trunk,
    tokenizer: TokenizerProtocol,
    scanned: Sequence[ScannedSource],
    batch: SourceTensorBatch,
    chunk_size: int,
) -> tuple[torch.Tensor, int, str]:
    """Extract and align frozen features for every valid source record."""

    if len(scanned) != batch.unit_ids.shape[0]:
        raise MultiFamilyCompilerError("source feature batch differs")
    payloads = tuple(
        anonymous_payload(record.payload)
        for source in scanned
        for record in source.records
    )
    device = batch.unit_ids.device
    flat = _extract_flat_byte_features(
        trunk=trunk,
        tokenizer=tokenizer,
        payloads=payloads,
        chunk_size=chunk_size,
        device=device,
    )
    if not flat:
        raise MultiFamilyCompilerError("source frozen features are empty")
    width = int(flat[0].shape[-1])
    output = torch.zeros(
        (*batch.unit_ids.shape, width),
        dtype=flat[0].dtype,
        device=device,
    )
    cursor = 0
    for row, source in enumerate(scanned):
        for record_index, record in enumerate(source.records):
            projected = project_byte_features_to_units(
                unit_byte_bounds=record.unit_byte_bounds,
                byte_features=flat[cursor],
            )
            output[row, record_index, : projected.shape[0]] = projected
            cursor += 1
    if cursor != len(flat):
        raise MultiFamilyCompilerError("source feature alignment count differs")
    return output, len(payloads), _manifest_sha256(payloads)


def extract_query_unit_features(
    *,
    trunk,
    tokenizer: TokenizerProtocol,
    scanned: Sequence[ScannedQuery],
    batch: QueryTensorBatch,
    chunk_size: int,
) -> tuple[torch.Tensor, int, str]:
    """Extract and align frozen features for every late query."""

    if len(scanned) != batch.unit_ids.shape[0]:
        raise MultiFamilyCompilerError("query feature batch differs")
    payloads = tuple(anonymous_payload(query.payload) for query in scanned)
    device = batch.unit_ids.device
    flat = _extract_flat_byte_features(
        trunk=trunk,
        tokenizer=tokenizer,
        payloads=payloads,
        chunk_size=chunk_size,
        device=device,
    )
    width = int(flat[0].shape[-1])
    output = torch.zeros(
        (*batch.unit_ids.shape, width),
        dtype=flat[0].dtype,
        device=device,
    )
    for row, (query, features) in enumerate(zip(scanned, flat, strict=True)):
        projected = project_byte_features_to_units(
            unit_byte_bounds=query.unit_byte_bounds,
            byte_features=features,
        )
        output[row, : projected.shape[0]] = projected
    return output, len(payloads), _manifest_sha256(payloads)


def connected_feature_receipt(
    *,
    trunk,
    source_payload_count: int,
    query_payload_count: int,
    anonymous_source_manifest_sha256: str,
    anonymous_query_manifest_sha256: str,
) -> ConnectedFeatureReceipt:
    receipt = trunk.parameter_receipt()
    return ConnectedFeatureReceipt(
        checkpoint_sha256=receipt.checkpoint_sha256,
        checkpoint_verified=receipt.checkpoint_verified,
        protected_parameters=receipt.parent_unique_parameters,
        frozen_feature_width=trunk.feature_width,
        source_payload_count=source_payload_count,
        query_payload_count=query_payload_count,
        anonymous_source_manifest_sha256=anonymous_source_manifest_sha256,
        anonymous_query_manifest_sha256=anonymous_query_manifest_sha256,
    )


__all__ = [
    "ConnectedFeatureReceipt",
    "PROTECTED_SHOHIN_PARAMETERS",
    "PROTECTED_SHOHIN_SHA256",
    "anonymous_payload",
    "build_trunk_batch",
    "connected_feature_receipt",
    "extract_query_unit_features",
    "extract_source_unit_features",
]
