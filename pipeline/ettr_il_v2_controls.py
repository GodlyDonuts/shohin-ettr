"""Deterministic causal-control construction for R12-ETTR-IL-v2."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable


PROTOCOL = "R12-ETTR-IL-v2"
DERANGEMENT_SCHEMA = "r12-ettr-il-v2-binding-derangement-v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class ControlError(ValueError):
    """A matched causal control cannot be constructed exactly."""


@dataclass(frozen=True, order=True, slots=True)
class BindingKey:
    ontology: str
    depth: int
    renderer: int
    presentation: str
    query_semantic_pair_signature: str
    paraphrase_pair_signature: str
    initial_support_shape: str
    terminal_support_shape: str
    transaction_mask: str

    def validate(self) -> None:
        if self.ontology not in {"horn", "rewrite", "resource"}:
            raise ControlError("binding key ontology differs")
        if self.depth not in range(1, 7):
            raise ControlError("binding key depth differs")
        if self.renderer not in range(4):
            raise ControlError("binding key renderer differs")
        if not self.presentation:
            raise ControlError("binding key presentation is empty")
        for name in (
            "query_semantic_pair_signature",
            "paraphrase_pair_signature",
            "initial_support_shape",
            "terminal_support_shape",
            "transaction_mask",
        ):
            if _HEX64.fullmatch(getattr(self, name)) is None:
                raise ControlError(f"binding key {name} differs")

    def as_dict(self) -> dict[str, int | str]:
        return {
            "depth": self.depth,
            "initial_support_shape": self.initial_support_shape,
            "ontology": self.ontology,
            "paraphrase_pair_signature": self.paraphrase_pair_signature,
            "presentation": self.presentation,
            "query_semantic_pair_signature": (
                self.query_semantic_pair_signature
            ),
            "renderer": self.renderer,
            "terminal_support_shape": self.terminal_support_shape,
            "transaction_mask": self.transaction_mask,
        }


@dataclass(frozen=True, slots=True)
class TargetBundleDescriptor:
    semantic_rectangle_id: str
    key: BindingKey
    terminal_packet_sha256s: tuple[str, str, str, str]
    transaction_sha256s: tuple[str, str, str, str]
    answer_labels: tuple[int, ...]

    def validate(self) -> None:
        if _HEX64.fullmatch(self.semantic_rectangle_id) is None:
            raise ControlError("semantic rectangle ID differs")
        self.key.validate()
        if len(self.terminal_packet_sha256s) != 4:
            raise ControlError("terminal packet digest count differs")
        if len(self.transaction_sha256s) != 4:
            raise ControlError("transaction digest count differs")
        for name, values in (
            ("terminal packet", self.terminal_packet_sha256s),
            ("transaction", self.transaction_sha256s),
        ):
            if any(_HEX64.fullmatch(value) is None for value in values):
                raise ControlError(f"{name} digest differs")
        if (
            len(self.answer_labels) != 16
            or any(value not in (0, 1) for value in self.answer_labels)
        ):
            raise ControlError("answer label vector differs")


@dataclass(frozen=True, slots=True)
class DerangementAssignment:
    recipient_id: str
    donor_id: str
    donor_rank: int
    donor_digest: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "donor_digest": self.donor_digest,
            "donor_id": self.donor_id,
            "donor_rank": self.donor_rank,
            "recipient_id": self.recipient_id,
        }


@dataclass(frozen=True, slots=True)
class BindingDerangement:
    fold: int
    assignments: tuple[DerangementAssignment, ...]
    assignment_sha256: str

    def receipt(self) -> dict[str, int | str]:
        return {
            "assignment_count": len(self.assignments),
            "assignment_sha256": self.assignment_sha256,
            "fold": self.fold,
            "fixed_points": sum(
                value.recipient_id == value.donor_id
                for value in self.assignments
            ),
            "protocol": PROTOCOL,
            "schema": DERANGEMENT_SCHEMA,
        }


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _donor_digest(fold: int, recipient_id: str, donor_id: str) -> bytes:
    return hashlib.sha256(
        (
            f"{PROTOCOL}|binding-derangement|{fold}|"
            f"{recipient_id}|{donor_id}"
        ).encode("ascii")
    ).digest()


def _admissible(
    recipient: TargetBundleDescriptor,
    donor: TargetBundleDescriptor,
) -> bool:
    return (
        recipient.semantic_rectangle_id != donor.semantic_rectangle_id
        and recipient.key == donor.key
        and all(
            left != right
            for left, right in zip(
                recipient.terminal_packet_sha256s,
                donor.terminal_packet_sha256s,
                strict=True,
            )
        )
        and all(
            left != right
            for left, right in zip(
                recipient.transaction_sha256s,
                donor.transaction_sha256s,
                strict=True,
            )
        )
        and all(
            left != right
            for left, right in zip(
                recipient.answer_labels,
                donor.answer_labels,
                strict=True,
            )
        )
    )


def _has_completion(
    recipients: tuple[str, ...],
    candidate_ids: dict[str, tuple[str, ...]],
    occupied: frozenset[str],
) -> bool:
    donor_owner: dict[str, str] = {}

    def augment(recipient_id: str, seen: set[str]) -> bool:
        for donor_id in candidate_ids[recipient_id]:
            if donor_id in occupied or donor_id in seen:
                continue
            seen.add(donor_id)
            owner = donor_owner.get(donor_id)
            if owner is None or augment(owner, seen):
                donor_owner[donor_id] = recipient_id
                return True
        return False

    return all(augment(recipient_id, set()) for recipient_id in recipients)


def _solve_group(
    records: tuple[TargetBundleDescriptor, ...],
    *,
    fold: int,
) -> tuple[DerangementAssignment, ...]:
    by_id = {
        value.semantic_rectangle_id: value
        for value in records
    }
    recipient_ids = tuple(sorted(by_id))
    if len(by_id) != len(records):
        raise ControlError("semantic rectangle IDs are not unique")
    candidates: dict[str, tuple[str, ...]] = {}
    ranks: dict[tuple[str, str], tuple[int, str]] = {}
    for recipient_id in recipient_ids:
        recipient = by_id[recipient_id]
        ordered: list[tuple[bytes, str]] = []
        for donor_id in recipient_ids:
            donor = by_id[donor_id]
            if _admissible(recipient, donor):
                digest = _donor_digest(fold, recipient_id, donor_id)
                ordered.append((digest, donor_id))
        ordered.sort()
        candidates[recipient_id] = tuple(donor_id for _, donor_id in ordered)
        for rank, (digest, donor_id) in enumerate(ordered):
            ranks[(recipient_id, donor_id)] = (rank, digest.hex())
        if not ordered:
            raise ControlError(
                f"binding group has no donor for {recipient_id}"
            )

    if not _has_completion(recipient_ids, candidates, frozenset()):
        raise ControlError("binding group has no perfect matching")

    occupied: set[str] = set()
    chosen: list[DerangementAssignment] = []
    for index, recipient_id in enumerate(recipient_ids):
        remaining = recipient_ids[index + 1 :]
        selected: str | None = None
        for donor_id in candidates[recipient_id]:
            if donor_id in occupied:
                continue
            proposed = frozenset((*occupied, donor_id))
            if _has_completion(remaining, candidates, proposed):
                selected = donor_id
                break
        if selected is None:
            raise AssertionError("lexicographic perfect matching disappeared")
        occupied.add(selected)
        rank, digest = ranks[(recipient_id, selected)]
        chosen.append(
            DerangementAssignment(
                recipient_id=recipient_id,
                donor_id=selected,
                donor_rank=rank,
                donor_digest=digest,
            )
        )
    if len(occupied) != len(recipient_ids):
        raise AssertionError("perfect matching donor cardinality differs")
    return tuple(chosen)


def build_binding_derangement(
    records: Iterable[TargetBundleDescriptor],
    *,
    fold: int,
) -> BindingDerangement:
    if fold not in (0, 1, 2):
        raise ControlError("fold differs")
    values = tuple(records)
    if not values:
        raise ControlError("binding population is empty")
    for value in values:
        if not isinstance(value, TargetBundleDescriptor):
            raise ControlError("target bundle type differs")
        value.validate()
    if len({value.semantic_rectangle_id for value in values}) != len(values):
        raise ControlError("semantic rectangle IDs are not unique")

    groups: dict[BindingKey, list[TargetBundleDescriptor]] = defaultdict(list)
    for value in values:
        groups[value.key].append(value)
    assignments = tuple(
        assignment
        for key in sorted(groups)
        for assignment in _solve_group(tuple(groups[key]), fold=fold)
    )
    payload = canonical_json_bytes(
        [value.as_dict() for value in assignments]
    )
    result = BindingDerangement(
        fold=fold,
        assignments=assignments,
        assignment_sha256=hashlib.sha256(payload).hexdigest(),
    )
    if result.receipt()["fixed_points"] != 0:
        raise AssertionError("binding derangement contains a fixed point")
    return result


__all__ = [
    "BindingDerangement",
    "BindingKey",
    "ControlError",
    "DerangementAssignment",
    "TargetBundleDescriptor",
    "build_binding_derangement",
    "canonical_json_bytes",
]
