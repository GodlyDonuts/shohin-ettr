"""Frozen continuation-data and runner-to-objective contract for ETTR.

Candidate-visible tensors contain token segments and generic categorical
supervision only.  No family identifier, semantic executor, parser product,
host callback, answer verifier, or continuous source payload is admitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import re
from typing import Protocol, Sequence, runtime_checkable

import torch

from endogenous_typed_theory_reactor import (
    TheoryReactorConfig,
    TheoryReactorError,
)
from ettr_episode import (
    ETTREpisodeBatch,
    ETTREpisodeOutput,
    ETTREpisodeSegment,
    ETTRInterventionOutput,
)
from ettr_objectives import (
    ETTRCausalQueryPair,
    ETTRObjectiveBatch,
    ETTRObjectiveConfig,
    ETTRPacketTargets,
    ETTRTokenTargets,
    ETTRTransactionPredictions,
    ETTRTransactionTargets,
    ETTRVariantAlignment,
)


ETTR_CONTINUATION_SCHEMA = "shohin-ettr-continuation-data-v1"
ETTR_PACKET_SUFFICIENCY_SCHEMA = "shohin-ettr-packet-sufficiency-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _canonical_payload(value: object) -> object:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(
            memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
        )
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": digest.hexdigest(),
        }
    if hasattr(value, "__dataclass_fields__"):
        return {
            field.name: _canonical_payload(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_canonical_payload(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TheoryReactorError(
        "ETTR continuation payload contains an unsupported value"
    )


def continuation_batch_payload_sha256(
    batch: ETTRContinuationBatch,
) -> str:
    if not isinstance(batch, ETTRContinuationBatch):
        raise TheoryReactorError("ETTR continuation payload type differs")
    payload = {
        field.name: _canonical_payload(getattr(batch, field.name))
        for field in fields(batch)
        if field.name not in {"manifest_sha256", "dataset_sha256"}
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def select_continuation_rows(
    batch: ETTRContinuationBatch,
    indices: torch.Tensor,
) -> ETTRContinuationBatch:
    """Select complete causal rectangles and remap every row reference."""

    if not isinstance(batch, ETTRContinuationBatch):
        raise TheoryReactorError("ETTR continuation selection type differs")
    rows = batch.episodes.world.tokens.shape[0]
    if (
        not torch.is_tensor(indices)
        or indices.ndim != 1
        or indices.dtype != torch.long
        or indices.device != batch.episodes.world.tokens.device
        or indices.numel() < 4
        or indices.unique().numel() != indices.numel()
        or bool((indices < 0).any())
        or bool((indices >= rows).any())
    ):
        raise TheoryReactorError("ETTR continuation row selection differs")
    inverse = torch.full(
        (rows,),
        -1,
        dtype=torch.long,
        device=indices.device,
    )
    inverse[indices] = torch.arange(indices.numel(), device=indices.device)

    def tensor_dataclass(value: object) -> object:
        return type(value)(
            **{
                field.name: getattr(value, field.name).index_select(0, indices)
                for field in fields(value)
            }
        )

    def segment(value: ETTREpisodeSegment) -> ETTREpisodeSegment:
        return ETTREpisodeSegment(
            tokens=value.tokens.index_select(0, indices),
            targets=value.targets.index_select(0, indices),
            attention_mask=value.attention_mask.index_select(0, indices),
        )

    rectangle_rows = batch.causal_rectangles.rows
    rectangle_mask = inverse.index_select(
        0,
        rectangle_rows.flatten(),
    ).reshape_as(rectangle_rows).ge(0).all(dim=(1, 2))
    selected_rectangles = rectangle_rows[rectangle_mask]
    if (
        selected_rectangles.numel() != indices.numel()
        or selected_rectangles.shape[0] * 4 != indices.numel()
    ):
        raise TheoryReactorError(
            "ETTR continuation selection splits a causal rectangle"
        )
    causal = ETTRCausalRectangle(
        rows=inverse.index_select(
            0,
            selected_rectangles.flatten(),
        ).reshape_as(selected_rectangles)
    )

    alignment = None
    if batch.equivariance is not None:
        value = batch.equivariance
        pair_mask = (
            inverse.index_select(0, value.left_index).ge(0)
            & inverse.index_select(0, value.right_index).ge(0)
        )
        if not bool(pair_mask.any()):
            raise TheoryReactorError(
                "ETTR continuation selection loses equivariance support"
            )
        alignment = ETTRVariantAlignment(
            left_index=inverse.index_select(
                0,
                value.left_index[pair_mask],
            ),
            right_index=inverse.index_select(
                0,
                value.right_index[pair_mask],
            ),
            **{
                field.name: getattr(value, field.name)[pair_mask]
                for field in fields(value)
                if field.name not in {"left_index", "right_index"}
            },
        )

    selected = ETTRContinuationBatch(
        manifest_sha256=batch.manifest_sha256,
        dataset_sha256=batch.dataset_sha256,
        episodes=ETTREpisodeBatch(
            episode_ids=tuple(
                batch.episodes.episode_ids[int(index)]
                for index in indices.tolist()
            ),
            reset_mask=batch.episodes.reset_mask.index_select(0, indices),
            query_read_index=batch.episodes.query_read_index.index_select(
                0,
                indices,
            ),
            world=segment(batch.episodes.world),
            command=segment(batch.episodes.command),
            query=segment(batch.episodes.query),
        ),
        packet_targets=tensor_dataclass(batch.packet_targets),
        terminal_packet_targets=tensor_dataclass(
            batch.terminal_packet_targets
        ),
        causal_rectangles=causal,
        transaction_targets=tensor_dataclass(
            batch.transaction_targets
        ),
        initial_committed=batch.initial_committed.index_select(0, indices),
        initial_halted=batch.initial_halted.index_select(0, indices),
        equivariance=alignment,
    )
    return selected


def terminal_packet_query_context(
    batch: ETTRContinuationBatch,
    row: int,
) -> tuple[dict[str, object], int]:
    """Build one assessor-only sufficient-statistic record.

    Unsupported packet cells and masked query tokens are canonicalized to
    zero. The returned target is used only for admission and is never included
    as a candidate-visible field.
    """

    packet = batch.terminal_packet_targets
    read_index = int(batch.episodes.query_read_index.detach().cpu()[row])
    slot_mask = packet.slot_mask.detach().cpu()[row].bool()
    relation_mask = packet.relation_mask.detach().cpu()[row].bool()
    if not bool(slot_mask.all()) or not bool(relation_mask.all()):
        raise TheoryReactorError(
            "ETTR packet sufficiency requires full deployed-state supervision"
        )
    active = packet.active.detach().cpu()[row].bool()
    categorical_mask = active
    values = packet.value_code.detach().cpu()[row]
    types = packet.type_index.detach().cpu()[row]
    root = packet.root.detach().cpu()[row].bool()
    relations = packet.relations.detach().cpu()[row].bool()
    query_tokens = batch.episodes.query.tokens.detach().cpu()[row, : read_index + 1]
    query_mask = batch.episodes.query.attention_mask.detach().cpu()[
        row,
        : read_index + 1,
    ].bool()
    context = {
        "packet": {
            "active": active.tolist(),
            "committed": bool(packet.committed.detach().cpu()[row]),
            "halted": bool(packet.halted.detach().cpu()[row]),
            "relations": relations.tolist(),
            "root": root.tolist(),
            "type_index": torch.where(
                categorical_mask,
                types,
                torch.zeros_like(types),
            ).tolist(),
            "value_code": torch.where(
                categorical_mask,
                values,
                torch.zeros_like(values),
            ).tolist(),
        },
        "query": {
            "mask": query_mask.tolist(),
            "read_index": read_index,
            "tokens": torch.where(
                query_mask,
                query_tokens,
                torch.zeros_like(query_tokens),
            ).tolist(),
        },
    }
    target = int(
        batch.episodes.query.targets.detach().cpu()[
            row,
            read_index,
        ]
    )
    return context, target


@dataclass(frozen=True, slots=True)
class ETTRPacketSufficiencyReceipt:
    """Target-bound admission receipt with no raw targets in its interface."""

    schema: str
    batches: int
    rows: int
    unique_contexts: int
    context_sha256: str
    target_bound_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema != ETTR_PACKET_SUFFICIENCY_SCHEMA
            or self.batches < 1
            or self.rows < 1
            or self.unique_contexts < 1
            or self.unique_contexts > self.rows
            or _SHA256.fullmatch(self.context_sha256) is None
            or _SHA256.fullmatch(self.target_bound_sha256) is None
        ):
            raise TheoryReactorError(
                "ETTR terminal-packet sufficiency receipt differs"
            )


@runtime_checkable
class ETTRPacketSufficiencyVerifier(Protocol):
    """Runtime contract shared by in-memory and disk-backed indexes."""

    receipt: ETTRPacketSufficiencyReceipt
    train_batches: int
    validation_batches: int
    train_rows: int
    validation_rows: int
    train_payload_sha256: str
    validation_payload_sha256: str

    @property
    def train_contexts(self) -> int: ...

    @property
    def validation_contexts(self) -> int: ...

    def verify_train(
        self,
        batches: Sequence[ETTRContinuationBatch],
    ) -> None: ...

    def verify_validation(
        self,
        batches: Sequence[ETTRContinuationBatch],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ETTRPacketSufficiencyIndex:
    """Immutable global packet/query-to-target admission index."""

    _train_entries: tuple[tuple[bytes, int], ...]
    _validation_entries: tuple[tuple[bytes, int], ...]
    _train_batch_digests: tuple[bytes, ...]
    _validation_batch_digests: tuple[bytes, ...]
    train_batches: int
    validation_batches: int
    train_rows: int
    validation_rows: int
    train_payload_sha256: str
    validation_payload_sha256: str
    receipt: ETTRPacketSufficiencyReceipt
    _sealed_train_entries: frozenset[tuple[bytes, int]] = field(
        init=False,
        repr=False,
    )
    _sealed_validation_entries: frozenset[tuple[bytes, int]] = field(
        init=False,
        repr=False,
    )
    _sealed_train_batch_digests: frozenset[bytes] = field(
        init=False,
        repr=False,
    )
    _sealed_validation_batch_digests: frozenset[bytes] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        train = self._validated_entries(self._train_entries, "train")
        validation = self._validated_entries(
            self._validation_entries,
            "validation",
        )
        if set(train) & set(validation):
            raise TheoryReactorError(
                "ETTR packet sufficiency train/validation contexts overlap"
            )
        train_batch_digests = self._validated_digests(
            self._train_batch_digests,
            "train",
        )
        validation_batch_digests = self._validated_digests(
            self._validation_batch_digests,
            "validation",
        )
        if set(train_batch_digests) & set(validation_batch_digests):
            raise TheoryReactorError(
                "ETTR packet sufficiency train/validation batches overlap"
            )
        expected_train_payload = hashlib.sha256(
            _canonical_json_bytes(
                [digest.hex() for digest in train_batch_digests]
            )
        ).hexdigest()
        expected_validation_payload = hashlib.sha256(
            _canonical_json_bytes(
                [digest.hex() for digest in validation_batch_digests]
            )
        ).hexdigest()
        combined = {**train, **validation}
        context_hashes = sorted(digest.hex() for digest in combined)
        target_commitments = sorted(
            hashlib.sha256(
                _canonical_json_bytes(
                    {
                        "context_sha256": digest.hex(),
                        "target": target,
                    }
                )
            ).hexdigest()
            for digest, target in combined.items()
        )
        expected = ETTRPacketSufficiencyReceipt(
            schema=ETTR_PACKET_SUFFICIENCY_SCHEMA,
            batches=self.train_batches + self.validation_batches,
            rows=self.train_rows + self.validation_rows,
            unique_contexts=len(combined),
            context_sha256=hashlib.sha256(
                _canonical_json_bytes(context_hashes)
            ).hexdigest(),
            target_bound_sha256=hashlib.sha256(
                _canonical_json_bytes(target_commitments)
            ).hexdigest(),
        )
        if (
            self.train_batches < 1
            or self.train_rows < 1
            or self.validation_batches < 0
            or self.validation_rows < 0
            or bool(self.validation_batches) != bool(self.validation_rows)
            or self.train_batches != len(train_batch_digests)
            or self.validation_batches != len(validation_batch_digests)
            or self.train_payload_sha256 != expected_train_payload
            or self.validation_payload_sha256 != expected_validation_payload
            or expected != self.receipt
        ):
            raise TheoryReactorError(
                "ETTR packet sufficiency index receipt differs"
            )
        object.__setattr__(
            self,
            "_sealed_train_entries",
            frozenset(train.items()),
        )
        object.__setattr__(
            self,
            "_sealed_validation_entries",
            frozenset(validation.items()),
        )
        object.__setattr__(
            self,
            "_sealed_train_batch_digests",
            frozenset(train_batch_digests),
        )
        object.__setattr__(
            self,
            "_sealed_validation_batch_digests",
            frozenset(validation_batch_digests),
        )

    @staticmethod
    def _validated_entries(
        entries: tuple[tuple[bytes, int], ...],
        split: str,
    ) -> dict[bytes, int]:
        if not isinstance(entries, tuple):
            raise TheoryReactorError(
                f"ETTR packet sufficiency {split} entries differ"
            )
        mapping: dict[bytes, int] = {}
        for entry in entries:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], bytes)
                or len(entry[0]) != hashlib.sha256().digest_size
                or not isinstance(entry[1], int)
                or isinstance(entry[1], bool)
                or entry[1] < 0
                or entry[0] in mapping
            ):
                raise TheoryReactorError(
                    f"ETTR packet sufficiency {split} entries differ"
                )
            mapping[entry[0]] = entry[1]
        if tuple(sorted(mapping.items())) != entries:
            raise TheoryReactorError(
                f"ETTR packet sufficiency {split} entries are not canonical"
            )
        return mapping

    @staticmethod
    def _validated_digests(
        digests: tuple[bytes, ...],
        split: str,
    ) -> tuple[bytes, ...]:
        if (
            not isinstance(digests, tuple)
            or any(
                not isinstance(digest, bytes)
                or len(digest) != hashlib.sha256().digest_size
                for digest in digests
            )
            or tuple(sorted(set(digests))) != digests
        ):
            raise TheoryReactorError(
                f"ETTR packet sufficiency {split} batch digests differ"
            )
        return digests

    @classmethod
    def from_batches(
        cls,
        batches: Sequence[ETTRContinuationBatch],
    ) -> ETTRPacketSufficiencyIndex:
        return cls.from_splits(batches, ())

    @classmethod
    def from_splits(
        cls,
        train_batches: Sequence[ETTRContinuationBatch],
        validation_batches: Sequence[ETTRContinuationBatch],
    ) -> ETTRPacketSufficiencyIndex:
        frozen_train = tuple(train_batches)
        frozen_validation = tuple(validation_batches)
        frozen = frozen_train + frozen_validation
        if (
            not frozen_train
            or any(
                not isinstance(batch, ETTRContinuationBatch)
                for batch in frozen
            )
        ):
            raise TheoryReactorError(
                "ETTR terminal-packet sufficiency sequence differs"
            )
        targets: dict[bytes, int] = {}
        split_entries: dict[str, list[tuple[bytes, int]]] = {
            "train": [],
            "validation": [],
        }
        split_rows = {"train": 0, "validation": 0}
        for split, batches_for_split in (
            ("train", frozen_train),
            ("validation", frozen_validation),
        ):
            for batch in batches_for_split:
                batch_rows = batch.episodes.world.tokens.shape[0]
                if batch_rows < 1:
                    raise TheoryReactorError(
                        "ETTR terminal-packet sufficiency batch is empty"
                    )
                for row in range(batch_rows):
                    context, target = terminal_packet_query_context(batch, row)
                    context_bytes = _canonical_json_bytes(context)
                    context_digest = hashlib.sha256(context_bytes).digest()
                    prior = targets.setdefault(context_digest, target)
                    if prior != target:
                        raise TheoryReactorError(
                            "ETTR terminal packet and query prefix map to "
                            "multiple factual next-token targets"
                        )
                    split_entries[split].append((context_digest, target))
                    split_rows[split] += 1
        canonical_train = tuple(sorted(set(split_entries["train"])))
        canonical_validation = tuple(
            sorted(set(split_entries["validation"]))
        )
        train_batch_digests = tuple(
            sorted(
                bytes.fromhex(continuation_batch_payload_sha256(batch))
                for batch in frozen_train
            )
        )
        validation_batch_digests = tuple(
            sorted(
                bytes.fromhex(continuation_batch_payload_sha256(batch))
                for batch in frozen_validation
            )
        )
        if {digest for digest, _ in canonical_train} & {
            digest for digest, _ in canonical_validation
        }:
            raise TheoryReactorError(
                "ETTR packet sufficiency train/validation contexts overlap"
            )
        combined = dict(canonical_train + canonical_validation)
        context_hashes = sorted(digest.hex() for digest in combined)
        target_commitments = sorted(
            hashlib.sha256(
                _canonical_json_bytes(
                    {
                        "context_sha256": digest.hex(),
                        "target": target,
                    }
                )
            ).hexdigest()
            for digest, target in combined.items()
        )
        receipt = ETTRPacketSufficiencyReceipt(
            schema=ETTR_PACKET_SUFFICIENCY_SCHEMA,
            batches=len(frozen),
            rows=split_rows["train"] + split_rows["validation"],
            unique_contexts=len(combined),
            context_sha256=hashlib.sha256(
                _canonical_json_bytes(context_hashes)
            ).hexdigest(),
            target_bound_sha256=hashlib.sha256(
                _canonical_json_bytes(target_commitments)
            ).hexdigest(),
        )
        return cls(
            _train_entries=canonical_train,
            _validation_entries=canonical_validation,
            _train_batch_digests=train_batch_digests,
            _validation_batch_digests=validation_batch_digests,
            train_batches=len(frozen_train),
            validation_batches=len(frozen_validation),
            train_rows=split_rows["train"],
            validation_rows=split_rows["validation"],
            train_payload_sha256=hashlib.sha256(
                _canonical_json_bytes(
                    [digest.hex() for digest in train_batch_digests]
                )
            ).hexdigest(),
            validation_payload_sha256=hashlib.sha256(
                _canonical_json_bytes(
                    [digest.hex() for digest in validation_batch_digests]
                )
            ).hexdigest(),
            receipt=receipt,
        )

    def verify_train(
        self,
        batches: Sequence[ETTRContinuationBatch],
    ) -> None:
        self._verify(
            batches,
            self._sealed_train_entries,
            "train",
            self._sealed_train_batch_digests,
        )

    @property
    def train_contexts(self) -> int:
        return len(self._sealed_train_entries)

    @property
    def validation_contexts(self) -> int:
        return len(self._sealed_validation_entries)

    def verify_validation(
        self,
        batches: Sequence[ETTRContinuationBatch],
    ) -> None:
        self._verify(
            batches,
            self._sealed_validation_entries,
            "validation",
            self._sealed_validation_batch_digests,
        )

    @staticmethod
    def _verify(
        batches: Sequence[ETTRContinuationBatch],
        entries: frozenset[tuple[bytes, int]],
        split: str,
        batch_digests: frozenset[bytes] | None = None,
    ) -> None:
        frozen = tuple(batches)
        if not frozen or any(
            not isinstance(batch, ETTRContinuationBatch) for batch in frozen
        ):
            raise TheoryReactorError(
                "ETTR terminal-packet sufficiency sequence differs"
            )
        targets = dict(entries)
        if batch_digests is None:
            raise TheoryReactorError(
                "ETTR packet sufficiency batch digest index is missing"
            )
        admitted_batches = set(batch_digests)
        for batch in frozen:
            payload_digest = bytes.fromhex(
                continuation_batch_payload_sha256(batch)
            )
            if payload_digest not in admitted_batches:
                raise TheoryReactorError(
                    f"ETTR batch is absent from the frozen {split} "
                    "payload index"
                )
            batch_rows = batch.episodes.world.tokens.shape[0]
            if batch_rows < 1:
                raise TheoryReactorError(
                    "ETTR terminal-packet sufficiency batch is empty"
                )
            for row in range(batch_rows):
                context, target = terminal_packet_query_context(batch, row)
                digest = hashlib.sha256(
                    _canonical_json_bytes(context)
                ).digest()
                if targets.get(digest) != target:
                    raise TheoryReactorError(
                        f"ETTR batch is absent from the frozen {split} "
                        "packet sufficiency index"
                    )


def terminal_packet_sufficiency_receipt(
    batches: Sequence[ETTRContinuationBatch],
) -> ETTRPacketSufficiencyReceipt:
    """Reject non-functional packet/query-to-target mappings.

    This admission helper is intentionally outside the candidate forward path.
    It detects diagonal, cross-rectangle, and cross-batch collisions over a
    frozen batch sequence while returning only hashes and support counts.
    """

    return ETTRPacketSufficiencyIndex.from_batches(batches).receipt


def _packet_target_rows_differ(
    targets: ETTRPacketTargets,
    left_index: torch.Tensor,
    right_index: torch.Tensor,
) -> torch.Tensor:
    left_active = targets.active.index_select(0, left_index)
    right_active = targets.active.index_select(0, right_index)
    left_slot_mask = targets.slot_mask.index_select(0, left_index)
    right_slot_mask = targets.slot_mask.index_select(0, right_index)
    slot_support = left_slot_mask & right_slot_mask
    categorical_support = slot_support & (left_active | right_active)
    relation_support = (
        targets.relation_mask.index_select(0, left_index)
        & targets.relation_mask.index_select(0, right_index)
    )
    return (
        ((left_active != right_active) & slot_support).any(dim=1)
        | (
            (
                targets.root.index_select(0, left_index)
                != targets.root.index_select(0, right_index)
            )
            & slot_support
        ).any(dim=1)
        | (
            (
                targets.value_code.index_select(0, left_index)
                != targets.value_code.index_select(0, right_index)
            )
            & categorical_support
        ).any(dim=1)
        | (
            (
                targets.type_index.index_select(0, left_index)
                != targets.type_index.index_select(0, right_index)
            )
            & categorical_support
        ).any(dim=1)
        | (
            (
                targets.relations.index_select(0, left_index)
                != targets.relations.index_select(0, right_index)
            )
            & relation_support
        ).flatten(1).any(dim=1)
        | (
            targets.committed.index_select(0, left_index)
            != targets.committed.index_select(0, right_index)
        )
        | (
            targets.halted.index_select(0, left_index)
            != targets.halted.index_select(0, right_index)
        )
    )


def _require_same_packet_target(
    targets: ETTRPacketTargets,
    left: torch.Tensor,
    right: torch.Tensor,
    name: str,
) -> None:
    torch._assert_async(
        torch.stack(
            tuple(
                getattr(targets, field.name)
                .index_select(0, left)
                .eq(getattr(targets, field.name).index_select(0, right))
                .all()
                for field in fields(targets)
            )
        ).all(),
        f"ETTR causal rectangle {name} packet target differs",
    )


@dataclass(frozen=True, slots=True)
class ETTRCausalRectangle:
    """Immutable factorial rows ``[rectangle, world, command]``."""

    rows: torch.Tensor

    def validate(
        self,
        episodes: ETTREpisodeBatch,
        factual_initial_targets: ETTRPacketTargets,
        factual_terminal_targets: ETTRPacketTargets,
    ) -> None:
        batch = episodes.world.tokens.shape[0]
        if (
            self.rows.ndim != 3
            or self.rows.shape[1:] != (2, 2)
            or self.rows.dtype != torch.long
            or self.rows.device != episodes.world.tokens.device
            or self.rows.shape[0] < 1
        ):
            raise TheoryReactorError("ETTR causal rectangle geometry differs")
        flat = self.rows.flatten()
        expected = torch.arange(
            batch,
            device=episodes.world.tokens.device,
        )
        if flat.numel() != batch:
            raise TheoryReactorError(
                "ETTR causal rectangles must partition the batch"
            )
        torch._assert_async(
            flat.sort().values.eq(expected).all(),
            "ETTR causal rectangles must partition the batch",
        )
        r00 = self.rows[:, 0, 0]
        r01 = self.rows[:, 0, 1]
        r10 = self.rows[:, 1, 0]
        r11 = self.rows[:, 1, 1]
        _require_same_packet_target(
            factual_initial_targets,
            r00,
            r01,
            "WORLD W0",
        )
        _require_same_packet_target(
            factual_initial_targets,
            r10,
            r11,
            "WORLD W1",
        )
        torch._assert_async(
            _packet_target_rows_differ(
                factual_initial_targets,
                r00,
                r10,
            ).all(),
            "ETTR causal rectangle WORLD factors have identical packet targets",
        )
        for segment, left, right, name in (
            (episodes.world, r00, r01, "WORLD W0"),
            (episodes.world, r10, r11, "WORLD W1"),
            (episodes.command, r00, r10, "COMMAND C0"),
            (episodes.command, r01, r11, "COMMAND C1"),
            (episodes.world, r00, r10, "WORLD factors"),
            (episodes.command, r00, r01, "COMMAND factors"),
        ):
            _require_distinct_source(segment, left, right, name)
        for field_name in ("slot_mask", "relation_mask"):
            reference = getattr(factual_terminal_targets, field_name).index_select(
                0,
                r00,
            )
            for index in (r01, r10, r11):
                torch._assert_async(
                    getattr(
                        factual_terminal_targets,
                        field_name,
                    ).index_select(0, index).eq(reference).all(),
                    "ETTR causal rectangle packet support differs",
                )
        for left, right, name in (
            (r00, r10, "WORLD/C0"),
            (r01, r11, "WORLD/C1"),
            (r00, r01, "COMMAND/W0"),
            (r10, r11, "COMMAND/W1"),
        ):
            torch._assert_async(
                _packet_target_rows_differ(
                    factual_terminal_targets,
                    left,
                    right,
                ).all(),
                f"ETTR causal rectangle {name} has no terminal consequence",
            )
        read_indices = torch.stack(
            tuple(
                episodes.query_read_index.index_select(0, index)
                for index in (r00, r01, r10, r11)
            ),
            dim=1,
        )
        torch._assert_async(
            read_indices.eq(read_indices[:, :1]).all(),
            "ETTR causal rectangle query read indices differ",
        )
        positions = torch.arange(
            episodes.query.tokens.shape[1],
            device=episodes.query.tokens.device,
        )[None, :]
        prefix_mask = positions.le(read_indices[:, :1])
        reference_tokens = episodes.query.tokens.index_select(0, r00)
        reference_mask = episodes.query.attention_mask.index_select(0, r00)
        for index in (r01, r10, r11):
            candidate_tokens = episodes.query.tokens.index_select(0, index)
            candidate_mask = episodes.query.attention_mask.index_select(0, index)
            torch._assert_async(
                (
                    (~prefix_mask | candidate_tokens.eq(reference_tokens))
                    & (~prefix_mask | candidate_mask.eq(reference_mask))
                ).all(),
                "ETTR causal rectangle query prefixes differ",
            )
    def intervention_indices(
        self,
    ) -> tuple[torch.Tensor, ...]:
        r00 = self.rows[:, 0, 0]
        r01 = self.rows[:, 0, 1]
        r10 = self.rows[:, 1, 0]
        r11 = self.rows[:, 1, 1]
        world_packet = torch.cat((r11, r10, r01, r00))
        world_command = torch.cat((r00, r01, r10, r11))
        world_target = torch.cat((r10, r11, r00, r01))
        command_packet = torch.cat((r00, r01, r10, r11))
        command_command = torch.cat((r11, r10, r01, r00))
        command_target = torch.cat((r01, r00, r11, r10))
        return (
            world_packet,
            world_command,
            world_target,
            command_packet,
            command_command,
            command_target,
        )


def _require_distinct_source(
    segment: ETTREpisodeSegment,
    left: torch.Tensor,
    right: torch.Tensor,
    name: str,
) -> None:
    left_tokens = segment.tokens.index_select(0, left)
    right_tokens = segment.tokens.index_select(0, right)
    left_mask = segment.attention_mask.index_select(0, left)
    right_mask = segment.attention_mask.index_select(0, right)
    differs = (
        (left_tokens.ne(right_tokens) & (left_mask | right_mask))
        | left_mask.ne(right_mask)
    ).any(dim=1)
    torch._assert_async(
        differs.all(),
        f"ETTR causal rectangle {name} raw renderings are identical",
    )


def _index_packet_targets(
    targets: ETTRPacketTargets,
    index: torch.Tensor,
) -> ETTRPacketTargets:
    return ETTRPacketTargets(
        **{
            field.name: getattr(targets, field.name).index_select(0, index)
            for field in fields(targets)
        }
    )


def _index_transaction_targets(
    targets: ETTRTransactionTargets,
    index: torch.Tensor,
) -> ETTRTransactionTargets:
    return ETTRTransactionTargets(
        **{
            field.name: getattr(targets, field.name).index_select(0, index)
            for field in fields(targets)
        }
    )


def _gather_query_rows(
    values: torch.Tensor,
    row_index: torch.Tensor,
    read_index: torch.Tensor,
) -> torch.Tensor:
    selected = values.index_select(0, row_index)
    selected_read = read_index.index_select(0, row_index)
    if selected.ndim == 3:
        return selected.gather(
            1,
            selected_read[:, None, None].expand(
                -1,
                1,
                selected.shape[-1],
            ),
        ).squeeze(1)
    if selected.ndim == 2:
        return selected.gather(1, selected_read[:, None]).squeeze(1)
    raise TheoryReactorError("ETTR causal query source rank differs")


def _validate_target_trajectory(
    initial: ETTRPacketTargets,
    transactions: ETTRTransactionTargets,
    terminal: ETTRPacketTargets,
    reactor_config: TheoryReactorConfig,
) -> None:
    """Replay generic labeled transactions without any ontology semantics."""

    batch, slots = initial.active.shape
    rows = torch.arange(batch, device=initial.active.device)
    active = initial.active.clone()
    root = initial.root.clone()
    values = initial.value_code.clone()
    types = initial.type_index.clone()
    relations = initial.relations.clone()
    committed = initial.committed.clone()
    halted = initial.halted.clone()
    for step in range(transactions.opcode.shape[1]):
        valid = transactions.step_mask[:, step]
        open_state = valid & ~committed & ~halted
        opcode = transactions.opcode[:, step]
        source = transactions.source[:, step]
        target = transactions.target[:, step]
        relation = transactions.relation[:, step]
        source_mask = torch.nn.functional.one_hot(
            source,
            slots,
        ).bool()
        alloc = source_mask & (
            open_state
            & opcode.eq(0)
            & ~active[rows, source]
        )[:, None]
        write = source_mask & (
            open_state
            & opcode.eq(1)
            & active[rows, source]
        )[:, None]
        clear = source_mask & (
            open_state
            & opcode.eq(2)
            & active[rows, source]
        )[:, None]
        active = (active | alloc) & ~clear
        values = torch.where(
            (alloc | write),
            transactions.value_code[:, step, None],
            values,
        )
        types = torch.where(
            alloc,
            transactions.type_index[:, step, None],
            types,
        )
        values = torch.where(clear, torch.zeros_like(values), values)
        types = torch.where(clear, torch.zeros_like(types), types)

        pair = (
            torch.nn.functional.one_hot(
                relation,
                reactor_config.num_relations,
            ).bool()[:, :, None, None]
            & source_mask[:, None, :, None]
            & torch.nn.functional.one_hot(
                target,
                slots,
            ).bool()[:, None, None, :]
        )
        link = (
            open_state
            & opcode.eq(3)
        )[:, None, None, None]
        unlink = (
            open_state
            & opcode.eq(4)
        )[:, None, None, None]
        relations = (relations | (link & pair)) & ~(unlink & pair)
        clear_pair = clear[:, None, :, None] | clear[:, None, None, :]
        relations = relations & ~clear_pair
        relations = (
            relations
            & active[:, None, :, None]
            & active[:, None, None, :]
        )
        torch._assert_async(
            relations.flatten(1).sum(-1).le(reactor_config.max_edges).all(),
            "ETTR target trajectory exceeds the relation-edge budget",
        )

        set_root = open_state & opcode.eq(5)
        requested_root = source_mask & active
        root = torch.where(set_root[:, None], requested_root, root)
        root = root & active
        committed = committed | (
            open_state & (opcode.eq(6) | opcode.eq(8))
        )
        halted = halted | (
            open_state & (opcode.eq(7) | opcode.eq(8))
        )
        torch._assert_async(
            (
                ~valid
                | (
                    transactions.committed[:, step].eq(committed)
                    & transactions.halted[:, step].eq(halted)
                )
            ).all(),
            "ETTR transaction disposition disagrees with labeled recurrence",
        )

    slot_mask = terminal.slot_mask
    categorical_mask = slot_mask & terminal.active
    relation_mask = terminal.relation_mask
    consistent = (
        ((active == terminal.active) | ~slot_mask).all()
        & ((root == terminal.root) | ~slot_mask).all()
        & ((values == terminal.value_code) | ~categorical_mask).all()
        & ((types == terminal.type_index) | ~categorical_mask).all()
        & ((relations == terminal.relations) | ~relation_mask).all()
        & committed.eq(terminal.committed).all()
        & halted.eq(terminal.halted).all()
    )
    torch._assert_async(
        consistent,
        "ETTR transaction labels do not realize the terminal packet target",
    )


@dataclass(frozen=True, slots=True)
class ETTRContinuationBatch:
    """One architecture batch plus its generic offline supervision."""

    manifest_sha256: str
    dataset_sha256: str
    episodes: ETTREpisodeBatch
    packet_targets: ETTRPacketTargets
    terminal_packet_targets: ETTRPacketTargets
    causal_rectangles: ETTRCausalRectangle
    transaction_targets: ETTRTransactionTargets
    initial_committed: torch.Tensor
    initial_halted: torch.Tensor
    equivariance: ETTRVariantAlignment | None

    def validate(
        self,
        reactor_config: TheoryReactorConfig,
        objective_config: ETTRObjectiveConfig,
    ) -> None:
        if (
            _SHA256.fullmatch(self.manifest_sha256) is None
            or _SHA256.fullmatch(self.dataset_sha256) is None
        ):
            raise TheoryReactorError("ETTR continuation snapshot receipt differs")
        self.episodes.validate()
        ETTRPacketTargets(
            **{
                field.name: getattr(self.packet_targets, field.name)
                for field in fields(self.packet_targets)
            }
        )
        ETTRPacketTargets(
            **{
                field.name: getattr(self.terminal_packet_targets, field.name)
                for field in fields(self.terminal_packet_targets)
            }
        )
        if not isinstance(self.causal_rectangles, ETTRCausalRectangle):
            raise TheoryReactorError("ETTR causal rectangle type differs")
        self.causal_rectangles.validate(
            self.episodes,
            self.packet_targets,
            self.terminal_packet_targets,
        )
        terminal_packet_sufficiency_receipt((self,))
        ETTRTransactionTargets(
            **{
                field.name: getattr(self.transaction_targets, field.name)
                for field in fields(self.transaction_targets)
            }
        )
        if self.equivariance is not None:
            ETTRVariantAlignment(
                **{
                    field.name: getattr(self.equivariance, field.name)
                    for field in fields(self.equivariance)
                }
            )
        batch = self.episodes.world.tokens.shape[0]
        steps = self.transaction_targets.opcode.shape[1]
        if (
            objective_config.num_slots != reactor_config.num_slots
            or objective_config.num_types != reactor_config.num_types
            or objective_config.num_relations != reactor_config.num_relations
            or objective_config.num_value_codes != reactor_config.num_value_codes
            or objective_config.relation_edge_budget != reactor_config.max_edges
            or steps > reactor_config.max_steps
            or self.packet_targets.active.shape != (batch, reactor_config.num_slots)
            or self.terminal_packet_targets.active.shape
            != (batch, reactor_config.num_slots)
            or self.transaction_targets.opcode.shape[0] != batch
            or self.initial_committed.shape != (batch,)
            or self.initial_halted.shape != (batch,)
            or self.initial_committed.dtype != torch.bool
            or self.initial_halted.dtype != torch.bool
        ):
            raise TheoryReactorError("ETTR continuation/objective geometry differs")
        torch._assert_async(
            self.packet_targets.committed.eq(self.initial_committed).all()
            & self.packet_targets.halted.eq(self.initial_halted).all()
            & ~self.initial_committed.any()
            & ~self.initial_halted.any(),
            "ETTR initial packet disposition differs from compiler reset state",
        )
        final_valid = self.transaction_targets.step_mask.sum(-1) - 1
        final_committed = self.transaction_targets.committed.gather(
            1,
            final_valid[:, None],
        ).squeeze(1)
        final_halted = self.transaction_targets.halted.gather(
            1,
            final_valid[:, None],
        ).squeeze(1)
        padded = ~self.transaction_targets.step_mask.all(dim=1)
        torch._assert_async(
            (~padded | final_committed | final_halted).all(),
            "ETTR padded transaction row remains open at its supervision boundary",
        )
        _validate_target_trajectory(
            self.packet_targets,
            self.transaction_targets,
            self.terminal_packet_targets,
            reactor_config,
        )
        devices = {
            self.episodes.world.tokens.device,
            self.packet_targets.active.device,
            self.terminal_packet_targets.active.device,
            self.causal_rectangles.rows.device,
            self.transaction_targets.opcode.device,
            self.initial_committed.device,
            self.initial_halted.device,
        }
        if self.equivariance is not None:
            devices.add(self.equivariance.left_index.device)
        if len(devices) != 1:
            raise TheoryReactorError("ETTR continuation tensors must share one device")
        declared = {
            field.name
            for value in (
                self,
                self.packet_targets,
                self.terminal_packet_targets,
                self.causal_rectangles,
                self.transaction_targets,
            )
            for field in fields(value)
        }
        if any("family" in name or "ontology" in name for name in declared):
            raise TheoryReactorError(
                "ETTR continuation contract exposes a family label"
            )

    def objective_batch(
        self,
        output: ETTREpisodeOutput,
        interventions: ETTRInterventionOutput,
    ) -> ETTRObjectiveBatch:
        """Join reset segment logits without creating boundary targets."""

        if not isinstance(output, ETTREpisodeOutput):
            raise TheoryReactorError("ETTR episode output type differs")
        if not isinstance(interventions, ETTRInterventionOutput):
            raise TheoryReactorError("ETTR intervention output type differs")
        segments = (
            self.episodes.world,
            self.episodes.command,
            self.episodes.query,
        )
        logits = torch.cat(
            (
                output.world_logits,
                output.command_logits,
                output.query_logits,
            ),
            dim=1,
        )
        token_ids = torch.cat(
            tuple(segment.tokens for segment in segments),
            dim=1,
        )
        mask = torch.cat(
            tuple(segment.attention_mask for segment in segments),
            dim=1,
        ).bool()
        reset_mask = torch.zeros_like(mask)
        cursor = 0
        for segment in segments:
            reset_mask[:, cursor] = True
            cursor += segment.tokens.shape[1]
        (
            _world_packet,
            world_command,
            world_target,
            command_packet,
            _command_command,
            command_target,
        ) = self.causal_rectangles.intervention_indices()
        world_query_binding = ETTRCausalQueryPair(
            correct_logits=interventions.world_query_logits,
            foil_logits=_gather_query_rows(
                output.query_logits,
                world_command,
                self.episodes.query_read_index,
            ),
            correct_target=_gather_query_rows(
                self.episodes.query.targets,
                world_target,
                self.episodes.query_read_index,
            ),
            foil_target=_gather_query_rows(
                self.episodes.query.targets,
                world_command,
                self.episodes.query_read_index,
            ),
        )
        command_query_binding = ETTRCausalQueryPair(
            correct_logits=interventions.command_query_logits,
            foil_logits=_gather_query_rows(
                output.query_logits,
                command_packet,
                self.episodes.query_read_index,
            ),
            correct_target=_gather_query_rows(
                self.episodes.query.targets,
                command_target,
                self.episodes.query_read_index,
            ),
            foil_target=_gather_query_rows(
                self.episodes.query.targets,
                command_packet,
                self.episodes.query_read_index,
            ),
        )
        return ETTRObjectiveBatch(
            token_logits=logits,
            token_targets=ETTRTokenTargets(
                token_ids=token_ids,
                mask=mask,
                reset_mask=reset_mask,
            ),
            packet_prediction=output.initial_state,
            packet_targets=self.packet_targets,
            terminal_packet_prediction=output.terminal_state,
            terminal_packet_targets=self.terminal_packet_targets,
            world_intervention_prediction=(
                interventions.world_terminal_state
            ),
            world_intervention_targets=_index_packet_targets(
                self.terminal_packet_targets,
                world_target,
            ),
            world_intervention_transactions=(
                ETTRTransactionPredictions.from_reactor_trace(
                    interventions.world_trace
                )
            ),
            world_intervention_transaction_targets=(
                _index_transaction_targets(
                    self.transaction_targets,
                    world_target,
                )
            ),
            command_intervention_prediction=(
                interventions.command_terminal_state
            ),
            command_intervention_targets=_index_packet_targets(
                self.terminal_packet_targets,
                command_target,
            ),
            command_intervention_transactions=(
                ETTRTransactionPredictions.from_reactor_trace(
                    interventions.command_trace
                )
            ),
            command_intervention_transaction_targets=(
                _index_transaction_targets(
                    self.transaction_targets,
                    command_target,
                )
            ),
            world_query_binding=world_query_binding,
            command_query_binding=command_query_binding,
            transactions=(ETTRTransactionPredictions.from_reactor_trace(output.trace)),
            transaction_targets=self.transaction_targets,
            initial_committed=self.initial_committed,
            initial_halted=self.initial_halted,
            equivariance=self.equivariance,
        )


@dataclass(frozen=True, slots=True)
class ETTRContinuationManifest:
    schema: str
    protected_checkpoint_sha256: str
    tokenizer_sha256: str
    qualification_payload_sha256: str
    hybrid_payload_sha256: str
    train_rows: int
    validation_rows: int
    train_payload_sha256: str
    validation_payload_sha256: str
    dataset_sha256: str
    packet_sufficiency_train_batches: int
    packet_sufficiency_validation_batches: int
    packet_sufficiency_rows: int
    packet_sufficiency_unique_contexts: int
    packet_sufficiency_train_contexts: int
    packet_sufficiency_validation_contexts: int
    packet_sufficiency_context_sha256: str
    packet_sufficiency_target_bound_sha256: str
    source_deleted: bool
    immutable_snapshot: bool
    live_writer_input: bool
    family_label_fields: tuple[str, ...]

    @staticmethod
    def combined_dataset_sha256(
        train_payload_sha256: str,
        validation_payload_sha256: str,
    ) -> str:
        if (
            _SHA256.fullmatch(train_payload_sha256) is None
            or _SHA256.fullmatch(validation_payload_sha256) is None
        ):
            raise TheoryReactorError(
                "ETTR continuation payload hash differs"
            )
        return hashlib.sha256(
            _canonical_json_bytes(
                {
                    "train_payload_sha256": train_payload_sha256,
                    "validation_payload_sha256": (
                        validation_payload_sha256
                    ),
                }
            )
        ).hexdigest()

    def validate(self) -> None:
        if self.schema != ETTR_CONTINUATION_SCHEMA:
            raise TheoryReactorError("ETTR continuation manifest schema differs")
        for name in (
            "protected_checkpoint_sha256",
            "tokenizer_sha256",
            "qualification_payload_sha256",
            "hybrid_payload_sha256",
            "train_payload_sha256",
            "validation_payload_sha256",
            "dataset_sha256",
            "packet_sufficiency_context_sha256",
            "packet_sufficiency_target_bound_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise TheoryReactorError(f"ETTR continuation manifest {name} differs")
        expected_dataset_sha256 = self.combined_dataset_sha256(
            self.train_payload_sha256,
            self.validation_payload_sha256,
        )
        if (
            self.train_rows <= 0
            or self.validation_rows <= 0
            or self.packet_sufficiency_train_batches <= 0
            or self.packet_sufficiency_validation_batches <= 0
            or self.packet_sufficiency_rows
            != self.train_rows + self.validation_rows
            or self.packet_sufficiency_unique_contexts <= 0
            or self.packet_sufficiency_unique_contexts
            > self.packet_sufficiency_rows
            or self.packet_sufficiency_train_contexts <= 0
            or self.packet_sufficiency_train_contexts > self.train_rows
            or self.packet_sufficiency_validation_contexts <= 0
            or self.packet_sufficiency_validation_contexts
            > self.validation_rows
            or self.packet_sufficiency_train_contexts
            + self.packet_sufficiency_validation_contexts
            != self.packet_sufficiency_unique_contexts
            or not self.source_deleted
            or not self.immutable_snapshot
            or self.live_writer_input
            or self.family_label_fields
            or self.train_payload_sha256 == self.validation_payload_sha256
            or self.dataset_sha256 != expected_dataset_sha256
        ):
            raise TheoryReactorError("ETTR continuation data custody differs")

    def packet_sufficiency_receipt(self) -> ETTRPacketSufficiencyReceipt:
        self.validate()
        return ETTRPacketSufficiencyReceipt(
            schema=ETTR_PACKET_SUFFICIENCY_SCHEMA,
            batches=(
                self.packet_sufficiency_train_batches
                + self.packet_sufficiency_validation_batches
            ),
            rows=self.packet_sufficiency_rows,
            unique_contexts=self.packet_sufficiency_unique_contexts,
            context_sha256=self.packet_sufficiency_context_sha256,
            target_bound_sha256=self.packet_sufficiency_target_bound_sha256,
        )

    def sha256(self) -> str:
        self.validate()
        return hashlib.sha256(
            _canonical_json_bytes(asdict(self))
        ).hexdigest()


__all__ = [
    "ETTR_CONTINUATION_SCHEMA",
    "ETTR_PACKET_SUFFICIENCY_SCHEMA",
    "ETTRCausalRectangle",
    "ETTRContinuationBatch",
    "ETTRContinuationManifest",
    "ETTRPacketSufficiencyIndex",
    "ETTRPacketSufficiencyReceipt",
    "ETTRPacketSufficiencyVerifier",
    "continuation_batch_payload_sha256",
    "select_continuation_rows",
    "terminal_packet_query_context",
    "terminal_packet_sufficiency_receipt",
]
