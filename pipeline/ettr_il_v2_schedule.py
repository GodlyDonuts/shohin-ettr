"""Exact invariant-pair training schedule for R12-ETTR-IL-v2.

The schedule is CPU-only. It consumes already frozen invariant-pair records
and emits the sole admitted 6,000-update exposure stream. It does not load a
model, optimizer, checkpoint, or dataset row.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable


PROTOCOL = "R12-ETTR-IL-v2"
SCHEDULE_SCHEMA = "r12-ettr-il-v2-pair-schedule-v1"
FOLDS = (0, 1, 2)
MODEL_SEEDS = (
    827771697280926998,
    9160563446168054265,
    5619173084519213573,
    2431337583064323711,
    8750822315343322697,
)
FIT_ONTOLOGIES = {
    0: ("rewrite", "resource"),
    1: ("horn", "resource"),
    2: ("horn", "rewrite"),
}
FIT_DEPTHS = (1, 2, 3)
CORES_PER_ONTOLOGY_DEPTH = 96
PAIRS_PER_CORE = 2
PAIRS_PER_ONTOLOGY_DEPTH = CORES_PER_ONTOLOGY_DEPTH * PAIRS_PER_CORE
PAIRS_PER_ONTOLOGY = PAIRS_PER_ONTOLOGY_DEPTH * len(FIT_DEPTHS)
PAIRS_PER_FOLD = PAIRS_PER_ONTOLOGY * 2
PAIRS_PER_MICROSTEP = 1
MICROSTEPS_PER_UPDATE = 4
UPDATES = 6000
PAIR_EXPOSURES = UPDATES * MICROSTEPS_PER_UPDATE
COMPLETE_EPOCHS = PAIR_EXPOSURES // PAIRS_PER_FOLD
TAIL_PAIRS = PAIR_EXPOSURES % PAIRS_PER_FOLD

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ONTOLOGIES = frozenset(("horn", "rewrite", "resource"))


class ScheduleError(ValueError):
    """The frozen pair population or requested schedule differs."""


@dataclass(frozen=True, order=True, slots=True)
class InvariantPairRecord:
    pair_id: str
    semantic_core_id: str
    ontology: str
    depth: int
    left_semantic_rectangle_id: str
    right_semantic_rectangle_id: str

    def validate(self) -> None:
        for name in (
            "pair_id",
            "semantic_core_id",
            "left_semantic_rectangle_id",
            "right_semantic_rectangle_id",
        ):
            if _HEX64.fullmatch(getattr(self, name)) is None:
                raise ScheduleError(f"{name} differs")
        if self.ontology not in _ONTOLOGIES:
            raise ScheduleError("pair ontology differs")
        if self.depth not in FIT_DEPTHS:
            raise ScheduleError("pair depth differs")
        if (
            self.left_semantic_rectangle_id
            == self.right_semantic_rectangle_id
        ):
            raise ScheduleError("pair rectangle identities are equal")


@dataclass(frozen=True, slots=True)
class PairExposure:
    update: int
    microstep: int
    epoch: int
    pair_id: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "epoch": self.epoch,
            "microstep": self.microstep,
            "pair_id": self.pair_id,
            "update": self.update,
        }


@dataclass(frozen=True, slots=True)
class PairSchedule:
    fold: int
    seed: int
    population_sha256: str
    schedule_sha256: str
    exposures: tuple[PairExposure, ...]

    def receipt(self) -> dict[str, int | str | list[int]]:
        counts = Counter(value.pair_id for value in self.exposures)
        multiplicities = Counter(counts.values())
        return {
            "complete_epochs": COMPLETE_EPOCHS,
            "exposure_multiplicities": [
                multiplicities.get(COMPLETE_EPOCHS, 0),
                multiplicities.get(COMPLETE_EPOCHS + 1, 0),
            ],
            "fold": self.fold,
            "microsteps_per_update": MICROSTEPS_PER_UPDATE,
            "pair_exposures": len(self.exposures),
            "pair_population": len(counts),
            "population_sha256": self.population_sha256,
            "protocol": PROTOCOL,
            "schedule_sha256": self.schedule_sha256,
            "schema": SCHEDULE_SCHEMA,
            "seed": self.seed,
            "tail_pairs": TAIL_PAIRS,
            "updates": UPDATES,
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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_population(
    records: Iterable[InvariantPairRecord],
    *,
    fold: int,
) -> tuple[InvariantPairRecord, ...]:
    if fold not in FOLDS:
        raise ScheduleError("fold differs")
    values = tuple(records)
    if len(values) != PAIRS_PER_FOLD:
        raise ScheduleError("pair population count differs")
    for value in values:
        if not isinstance(value, InvariantPairRecord):
            raise ScheduleError("pair record type differs")
        value.validate()
    if len({value.pair_id for value in values}) != len(values):
        raise ScheduleError("pair IDs are not unique")
    rectangles = tuple(
        rectangle
        for value in values
        for rectangle in (
            value.left_semantic_rectangle_id,
            value.right_semantic_rectangle_id,
        )
    )
    if len(set(rectangles)) != len(rectangles):
        raise ScheduleError("semantic rectangle occurs in multiple pairs")

    expected_ontologies = FIT_ONTOLOGIES[fold]
    counts = Counter((value.ontology, value.depth) for value in values)
    expected = {
        (ontology, depth): PAIRS_PER_ONTOLOGY_DEPTH
        for ontology in expected_ontologies
        for depth in FIT_DEPTHS
    }
    if dict(counts) != expected:
        raise ScheduleError("pair ontology/depth geometry differs")

    core_pairs: dict[str, list[InvariantPairRecord]] = defaultdict(list)
    for value in values:
        core_pairs[value.semantic_core_id].append(value)
    if len(core_pairs) != PAIRS_PER_FOLD // PAIRS_PER_CORE:
        raise ScheduleError("semantic core population count differs")
    for core_id, pairs in core_pairs.items():
        if len(pairs) != PAIRS_PER_CORE:
            raise ScheduleError(f"semantic core pair count differs: {core_id}")
        if len({(pair.ontology, pair.depth) for pair in pairs}) != 1:
            raise ScheduleError(f"semantic core metadata differs: {core_id}")

    return tuple(sorted(values, key=lambda value: value.pair_id))


def _population_sha256(values: tuple[InvariantPairRecord, ...]) -> str:
    rows = [
        {
            "depth": value.depth,
            "left_semantic_rectangle_id": value.left_semantic_rectangle_id,
            "ontology": value.ontology,
            "pair_id": value.pair_id,
            "right_semantic_rectangle_id": value.right_semantic_rectangle_id,
            "semantic_core_id": value.semantic_core_id,
        }
        for value in values
    ]
    return _sha256(canonical_json_bytes(rows))


def _epoch_order(
    values: tuple[InvariantPairRecord, ...],
    *,
    fold: int,
    seed: int,
    epoch: int,
) -> tuple[InvariantPairRecord, ...]:
    def key(value: InvariantPairRecord) -> tuple[bytes, str]:
        preimage = (
            f"{PROTOCOL}|schedule|{fold}|{seed}|{epoch}|{value.pair_id}"
        ).encode("ascii")
        return hashlib.sha256(preimage).digest(), value.pair_id

    return tuple(sorted(values, key=key))


def build_pair_schedule(
    records: Iterable[InvariantPairRecord],
    *,
    fold: int,
    seed: int,
) -> PairSchedule:
    if seed not in MODEL_SEEDS:
        raise ScheduleError("model seed differs")
    values = _validate_population(records, fold=fold)
    ordered: list[tuple[int, InvariantPairRecord]] = []
    for epoch in range(COMPLETE_EPOCHS):
        ordered.extend(
            (epoch, value)
            for value in _epoch_order(
                values,
                fold=fold,
                seed=seed,
                epoch=epoch,
            )
        )
    tail_epoch = COMPLETE_EPOCHS
    ordered.extend(
        (tail_epoch, value)
        for value in _epoch_order(
            values,
            fold=fold,
            seed=seed,
            epoch=tail_epoch,
        )[:TAIL_PAIRS]
    )
    if len(ordered) != PAIR_EXPOSURES:
        raise AssertionError("pair exposure arithmetic differs")
    exposures = tuple(
        PairExposure(
            update=index // MICROSTEPS_PER_UPDATE,
            microstep=index % MICROSTEPS_PER_UPDATE,
            epoch=epoch,
            pair_id=value.pair_id,
        )
        for index, (epoch, value) in enumerate(ordered)
    )
    schedule_payload = canonical_json_bytes(
        [value.as_dict() for value in exposures]
    )
    result = PairSchedule(
        fold=fold,
        seed=seed,
        population_sha256=_population_sha256(values),
        schedule_sha256=_sha256(schedule_payload),
        exposures=exposures,
    )
    receipt = result.receipt()
    if receipt["exposure_multiplicities"] != [
        PAIRS_PER_FOLD - TAIL_PAIRS,
        TAIL_PAIRS,
    ]:
        raise AssertionError("pair multiplicity arithmetic differs")
    return result


__all__ = [
    "COMPLETE_EPOCHS",
    "FIT_ONTOLOGIES",
    "InvariantPairRecord",
    "MICROSTEPS_PER_UPDATE",
    "MODEL_SEEDS",
    "PAIR_EXPOSURES",
    "PAIRS_PER_FOLD",
    "PairExposure",
    "PairSchedule",
    "ScheduleError",
    "TAIL_PAIRS",
    "UPDATES",
    "build_pair_schedule",
    "canonical_json_bytes",
]
