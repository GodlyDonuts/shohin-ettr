"""Matched token-native ETTR transcoding for cross-backbone controls.

The ETTR surface codec assigns semantics to public codebook indices, not to a
particular tokenizer's vocabulary IDs. A cross-backbone control can therefore
map each source codebook index to the same target codebook index while leaving
all packet, transaction, rectangle, and causal-answer labels unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from ettr_data_contract import ETTRContinuationBatch
from ettr_episode import ETTREpisodeBatch, ETTREpisodeSegment


class ETTRTokenTranscodeError(ValueError):
    """A matched tokenizer transcode cannot be proven."""


@dataclass(frozen=True, slots=True)
class ETTRTokenTranscodeReceipt:
    schema: str
    source_tokenizer_sha256: str
    target_tokenizer_sha256: str
    source_codebook_sha256: str
    target_codebook_sha256: str
    mapped_codewords: int
    mapped_tail_tokens: int
    token_id_map_sha256: str

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transcode_segment(
    segment: ETTREpisodeSegment,
    token_id_map: torch.Tensor,
) -> ETTREpisodeSegment:
    """Map one fixed-width segment and reconstruct exact shifted targets."""

    segment.validate()
    if (
        not torch.is_tensor(token_id_map)
        or token_id_map.ndim != 1
        or token_id_map.dtype != torch.long
        or token_id_map.device != segment.tokens.device
        or token_id_map.numel() <= int(segment.tokens.max())
    ):
        raise ETTRTokenTranscodeError("token-id map geometry differs")
    mapped = token_id_map.index_select(0, segment.tokens.flatten()).reshape_as(
        segment.tokens
    )
    if bool(mapped.lt(0).any()):
        raise ETTRTokenTranscodeError("ETTR segment contains an unmapped token")
    result = ETTREpisodeSegment.from_tokens(
        mapped,
        attention_mask=segment.attention_mask.clone(),
    )
    if not torch.equal(result.targets.ne(-1), segment.targets.ne(-1)):
        raise ETTRTokenTranscodeError(
            "ETTR target supervision mask changed during transcode"
        )
    return result


class TokenNativeETTRTranscoder:
    """One-to-one logical-code transcoder between two tokenizer artifacts."""

    SCHEMA = "shohin-ettr-token-native-transcode-v1"
    _TAILS = ("\nR=", "\nR=0\n", "\nR=1\n")

    def __init__(
        self,
        source_tokenizer: Path,
        target_tokenizer: Path,
    ) -> None:
        # Keep the production codec import outside module import so this helper
        # remains usable from the immutable training runtime layout.
        from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec

        source_tokenizer = source_tokenizer.expanduser().resolve()
        target_tokenizer = target_tokenizer.expanduser().resolve()
        if (
            not source_tokenizer.is_file()
            or not target_tokenizer.is_file()
            or source_tokenizer == target_tokenizer
        ):
            raise ETTRTokenTranscodeError("tokenizer paths differ")
        self.source = TokenNativeSurfaceCodec(
            source_tokenizer,
            required_tokenizer_sha256=None,
        )
        self.target = TokenNativeSurfaceCodec(
            target_tokenizer,
            required_tokenizer_sha256=None,
        )
        if len(self.target.codebook.token_ids) < len(
            self.source.codebook.token_ids
        ):
            raise ETTRTokenTranscodeError(
                "target token-native codebook is too small"
            )
        mapping: dict[int, int] = {
            source_id: self.target.codebook.token_ids[index]
            for index, source_id in enumerate(
                self.source.codebook.token_ids
            )
        }
        tail_tokens = 0
        for text in self._TAILS:
            source_ids = self.source.tokenizer.encode(
                text,
                add_special_tokens=False,
            ).ids
            target_ids = self.target.tokenizer.encode(
                text,
                add_special_tokens=False,
            ).ids
            if len(source_ids) != len(target_ids):
                raise ETTRTokenTranscodeError(
                    "query framing token width differs"
                )
            for source_id, target_id in zip(
                source_ids,
                target_ids,
                strict=True,
            ):
                existing = mapping.get(source_id)
                if existing is not None and existing != target_id:
                    raise ETTRTokenTranscodeError(
                        "query framing collides with codebook mapping"
                    )
                mapping[source_id] = target_id
                tail_tokens += existing is None
        self._mapping = mapping
        self._lookup = torch.full(
            (self.source.tokenizer.get_vocab_size(),),
            -1,
            dtype=torch.long,
        )
        for source_id, target_id in mapping.items():
            self._lookup[source_id] = target_id
        mapping_payload = json.dumps(
            sorted(mapping.items()),
            separators=(",", ":"),
        ).encode("ascii")
        self.receipt = ETTRTokenTranscodeReceipt(
            schema=self.SCHEMA,
            source_tokenizer_sha256=_file_sha256(source_tokenizer),
            target_tokenizer_sha256=_file_sha256(target_tokenizer),
            source_codebook_sha256=self.source.codebook_sha256,
            target_codebook_sha256=self.target.codebook_sha256,
            mapped_codewords=len(self.source.codebook.token_ids),
            mapped_tail_tokens=tail_tokens,
            token_id_map_sha256=hashlib.sha256(mapping_payload).hexdigest(),
        )

    @property
    def target_vocab_size(self) -> int:
        return self.target.tokenizer.get_vocab_size()

    def transcode_batch(
        self,
        batch: ETTRContinuationBatch,
    ) -> ETTRContinuationBatch:
        """Transcode only model-visible tokens in one admitted batch."""

        lookup = self._lookup.to(batch.episodes.world.tokens.device)
        episodes = ETTREpisodeBatch(
            episode_ids=batch.episodes.episode_ids,
            reset_mask=batch.episodes.reset_mask.clone(),
            query_read_index=batch.episodes.query_read_index.clone(),
            world=transcode_segment(batch.episodes.world, lookup),
            command=transcode_segment(batch.episodes.command, lookup),
            query=transcode_segment(batch.episodes.query, lookup),
        )
        receipt_sha256 = self.receipt.sha256()

        def derived(label: str, value: str) -> str:
            return hashlib.sha256(
                (
                    f"{self.SCHEMA}\x1f{label}\x1f{value}\x1f"
                    f"{receipt_sha256}"
                ).encode("ascii")
            ).hexdigest()

        result = replace(
            batch,
            manifest_sha256=derived("manifest", batch.manifest_sha256),
            dataset_sha256=derived("dataset", batch.dataset_sha256),
            episodes=episodes,
        )
        result.episodes.validate()
        return result


def receipt_value(
    receipt: ETTRTokenTranscodeReceipt,
) -> Mapping[str, object]:
    return {
        name: getattr(receipt, name)
        for name in receipt.__dataclass_fields__
    }


__all__ = [
    "ETTRTokenTranscodeError",
    "ETTRTokenTranscodeReceipt",
    "TokenNativeETTRTranscoder",
    "receipt_value",
    "transcode_segment",
]
