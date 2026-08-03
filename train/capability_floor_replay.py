"""Rectangle-atomic stratified replay for capability-floor optimization.

One optimizer update is four semantic microbatches of sixteen rows.  Causal
rectangles are indivisible four-row units, so each microbatch contains four
rectangles and every update contains sixteen distinct rectangles.  Required
causal strata are covered inside every accumulated update, preventing the
single-family regime erasure measured in v19 and v20.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Sequence


REPLAY_SCHEMA = "shohin-ettr-component-stratified-replay-v1"
REPLAY_MATRIX_SCHEMA = "shohin-ettr-candidate-replay-matrix-v1"
CORPUS_INDEX_SCHEMA = "shohin-ettr-capability-floor-core-index-v2"


class CapabilityFloorReplayError(ValueError):
    """The replay schedule is incomplete, non-atomic, or has drifted."""


@dataclass(frozen=True, slots=True)
class ReplayRectangle:
    rectangle_id: str
    strata: tuple[str, ...]
    row_count: int = 4
    charged_positions: int = 0

    def validate(self) -> None:
        if not self.rectangle_id:
            raise CapabilityFloorReplayError("rectangle identity is required")
        if self.row_count != 4:
            raise CapabilityFloorReplayError("causal rectangle must contain four rows")
        if self.charged_positions < 0:
            raise CapabilityFloorReplayError("charged positions differ")
        if not self.strata or any(not value for value in self.strata):
            raise CapabilityFloorReplayError("rectangle strata are required")
        if len(set(self.strata)) != len(self.strata):
            raise CapabilityFloorReplayError("rectangle strata are duplicated")


@dataclass(frozen=True, slots=True)
class ReplayScheduleConfig:
    component: str
    required_strata: tuple[str, ...]
    updates: int
    seed: int
    dataset_sha256: str
    semantic_microbatch_size: int = 16
    semantic_microbatches_per_update: int = 4
    rectangle_rows: int = 4

    def validate(self) -> None:
        if not self.component or not self.required_strata:
            raise CapabilityFloorReplayError("component replay strata are required")
        if len(set(self.required_strata)) != len(self.required_strata):
            raise CapabilityFloorReplayError("required strata are duplicated")
        if self.updates <= 0 or self.semantic_microbatches_per_update != 4:
            raise CapabilityFloorReplayError("optimizer update geometry differs")
        if self.semantic_microbatch_size != 16 or self.rectangle_rows != 4:
            raise CapabilityFloorReplayError("semantic microbatch geometry differs")
        if len(self.dataset_sha256) != 64:
            raise CapabilityFloorReplayError("dataset digest differs")


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def _rectangle_manifest_sha256(rectangles: Sequence[ReplayRectangle]) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                value.rectangle_id: {
                    "charged_positions": value.charged_positions,
                    "row_count": value.row_count,
                    "strata": list(value.strata),
                }
                for value in sorted(rectangles, key=lambda item: item.rectangle_id)
            }
        )
    ).hexdigest()


def _next_distinct(
    order: Sequence[str],
    cursor: int,
    selected: set[str],
) -> tuple[str, int]:
    for offset in range(len(order)):
        position = (cursor + offset) % len(order)
        candidate = order[position]
        if candidate not in selected:
            return candidate, (position + 1) % len(order)
    raise CapabilityFloorReplayError("not enough distinct rectangles per update")


def build_replay_schedule(
    rectangles: Sequence[ReplayRectangle],
    config: ReplayScheduleConfig,
) -> dict[str, object]:
    config.validate()
    if not rectangles:
        raise CapabilityFloorReplayError("replay corpus is empty")
    by_id: dict[str, ReplayRectangle] = {}
    for rectangle in rectangles:
        rectangle.validate()
        if rectangle.rectangle_id in by_id:
            raise CapabilityFloorReplayError("rectangle identity is duplicated")
        by_id[rectangle.rectangle_id] = rectangle
    rectangles_per_microbatch = config.semantic_microbatch_size // config.rectangle_rows
    rectangles_per_update = rectangles_per_microbatch * config.semantic_microbatches_per_update
    if len(by_id) < rectangles_per_update:
        raise CapabilityFloorReplayError("not enough rectangles for a distinct update")

    generator = random.Random(config.seed)
    global_order = sorted(by_id)
    generator.shuffle(global_order)
    stratum_orders: dict[str, list[str]] = {}
    for stratum in config.required_strata:
        candidates = sorted(
            rectangle.rectangle_id
            for rectangle in rectangles
            if stratum in rectangle.strata
        )
        if not candidates:
            raise CapabilityFloorReplayError(f"required stratum {stratum} is empty")
        generator.shuffle(candidates)
        stratum_orders[stratum] = candidates
    stratum_cursors = {stratum: 0 for stratum in config.required_strata}
    global_cursor = 0
    update_receipts: list[dict[str, object]] = []
    for update_index in range(config.updates):
        selected: list[str] = []
        selected_set: set[str] = set()
        for stratum in config.required_strata:
            candidate, cursor = _next_distinct(
                stratum_orders[stratum],
                stratum_cursors[stratum],
                selected_set,
            )
            stratum_cursors[stratum] = cursor
            selected.append(candidate)
            selected_set.add(candidate)
        while len(selected) < rectangles_per_update:
            candidate, global_cursor = _next_distinct(
                global_order,
                global_cursor,
                selected_set,
            )
            selected.append(candidate)
            selected_set.add(candidate)
        generator.shuffle(selected)
        microbatches = [
            selected[offset : offset + rectangles_per_microbatch]
            for offset in range(0, rectangles_per_update, rectangles_per_microbatch)
        ]
        covered = sorted(
            {
                stratum
                for rectangle_id in selected
                for stratum in by_id[rectangle_id].strata
                if stratum in config.required_strata
            }
        )
        charged_positions = sum(by_id[value].charged_positions for value in selected)
        update_receipts.append(
            {
                "charged_positions": charged_positions,
                "covered_strata": covered,
                "microbatches": microbatches,
                "update": update_index,
            }
        )
    payload: dict[str, object] = {
        "config": asdict(config),
        "global_loss_normalization": "one-denominator-over-four-microbatches",
        "rectangle_count": len(rectangles),
        "rectangle_manifest_sha256": _rectangle_manifest_sha256(rectangles),
        "schema": REPLAY_SCHEMA,
        "shared_between": ["ettr", "favorable-dense-recurrent-control"],
        "updates": update_receipts,
    }
    validate_replay_schedule(payload, rectangles)
    return payload


def validate_replay_schedule(
    payload: Mapping[str, object],
    rectangles: Sequence[ReplayRectangle],
) -> None:
    if (
        payload.get("schema") != REPLAY_SCHEMA
        or payload.get("global_loss_normalization")
        != "one-denominator-over-four-microbatches"
        or payload.get("shared_between")
        != ["ettr", "favorable-dense-recurrent-control"]
    ):
        raise CapabilityFloorReplayError("replay custody differs")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise CapabilityFloorReplayError("replay config differs")
    try:
        config = ReplayScheduleConfig(**dict(config_payload))
    except TypeError as error:
        raise CapabilityFloorReplayError("replay config differs") from error
    config.validate()
    by_id = {item.rectangle_id: item for item in rectangles}
    for rectangle in rectangles:
        rectangle.validate()
    if (
        len(by_id) != len(rectangles)
        or payload.get("rectangle_count") != len(rectangles)
        or payload.get("rectangle_manifest_sha256")
        != _rectangle_manifest_sha256(rectangles)
    ):
        raise CapabilityFloorReplayError("replay rectangle corpus differs")
    updates = payload.get("updates")
    if not isinstance(updates, list) or len(updates) != config.updates:
        raise CapabilityFloorReplayError("replay update count differs")
    required = set(config.required_strata)
    expected_rectangles_per_batch = config.semantic_microbatch_size // config.rectangle_rows
    for update_index, update in enumerate(updates):
        if not isinstance(update, Mapping) or update.get("update") != update_index:
            raise CapabilityFloorReplayError("replay update identity differs")
        microbatches = update.get("microbatches")
        if (
            not isinstance(microbatches, list)
            or len(microbatches) != config.semantic_microbatches_per_update
        ):
            raise CapabilityFloorReplayError("replay microbatch count differs")
        flattened: list[str] = []
        for microbatch in microbatches:
            if (
                not isinstance(microbatch, list)
                or len(microbatch) != expected_rectangles_per_batch
            ):
                raise CapabilityFloorReplayError("replay microbatch geometry differs")
            flattened.extend(microbatch)
        if len(set(flattened)) != len(flattened):
            raise CapabilityFloorReplayError("rectangle repeats inside optimizer update")
        if any(value not in by_id for value in flattened):
            raise CapabilityFloorReplayError("unknown replay rectangle")
        covered = {
            stratum
            for rectangle_id in flattened
            for stratum in by_id[rectangle_id].strata
            if stratum in required
        }
        if covered != required or update.get("covered_strata") != sorted(required):
            raise CapabilityFloorReplayError("optimizer update omits required stratum")
        charged = sum(by_id[value].charged_positions for value in flattened)
        if update.get("charged_positions") != charged:
            raise CapabilityFloorReplayError("charged-position receipt differs")


def replay_schedule_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_replay_rectangles(
    index_path: Path,
    *,
    expected_sha256: str,
    candidates: Sequence[str],
    split: str,
) -> dict[str, tuple[ReplayRectangle, ...]]:
    """Load one audited cohort index with candidate-specific token charges.

    Rectangle identities and strata are shared across every candidate.  Only
    charged token positions may differ because tokenizers differ.  Parsing the
    index once prevents candidate-specific row selection from entering later
    schedules.
    """

    selected = tuple(candidates)
    if (
        len(expected_sha256) != 64
        or not selected
        or any(not candidate for candidate in selected)
        or len(set(selected)) != len(selected)
        or split not in {"train", "development"}
    ):
        raise CapabilityFloorReplayError("cohort replay inventory arguments differ")
    if _sha256_file(index_path) != expected_sha256:
        raise CapabilityFloorReplayError("cohort index SHA-256 differs")
    values: dict[str, list[ReplayRectangle]] = {candidate: [] for candidate in selected}
    seen: set[str] = set()
    with index_path.open("r", encoding="ascii") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CapabilityFloorReplayError(
                    f"cohort index row {line_number} is unreadable"
                ) from error
            if (
                not isinstance(row, Mapping)
                or row.get("index_schema") != CORPUS_INDEX_SCHEMA
                or row.get("accepted") is not True
                or row.get("assessor_fields_in_model_input") is not False
                or row.get("split") not in {"train", "development"}
            ):
                raise CapabilityFloorReplayError("cohort index custody differs")
            if row["split"] != split:
                continue
            rectangles = row.get("rectangles")
            if not isinstance(rectangles, list) or not rectangles:
                raise CapabilityFloorReplayError("cohort index rectangle set differs")
            for rectangle in rectangles:
                if not isinstance(rectangle, Mapping):
                    raise CapabilityFloorReplayError("cohort index rectangle differs")
                rectangle_id = rectangle.get("rectangle_id")
                strata = rectangle.get("strata")
                charged = rectangle.get("charged_positions")
                if (
                    not isinstance(rectangle_id, str)
                    or not rectangle_id
                    or rectangle_id in seen
                    or not isinstance(strata, list)
                    or not strata
                    or any(not isinstance(value, str) or not value for value in strata)
                    or not isinstance(charged, Mapping)
                    or any(candidate not in charged for candidate in selected)
                ):
                    raise CapabilityFloorReplayError("cohort index rectangle differs")
                seen.add(rectangle_id)
                for candidate in selected:
                    positions = charged[candidate]
                    if (
                        not isinstance(positions, int)
                        or isinstance(positions, bool)
                        or positions <= 0
                    ):
                        raise CapabilityFloorReplayError(
                            "candidate charged positions differ"
                        )
                    values[candidate].append(
                        ReplayRectangle(
                            rectangle_id=rectangle_id,
                            strata=tuple(strata),
                            charged_positions=positions,
                        )
                    )
    if not seen or any(len(rows) != len(seen) for rows in values.values()):
        raise CapabilityFloorReplayError("cohort replay inventory is incomplete")
    return {
        candidate: tuple(rows)
        for candidate, rows in values.items()
    }


def _shared_update_manifest(schedule: Mapping[str, object]) -> list[dict[str, object]]:
    updates = schedule.get("updates")
    if not isinstance(updates, list):
        raise CapabilityFloorReplayError("candidate replay updates differ")
    return [
        {
            "covered_strata": update["covered_strata"],
            "microbatches": update["microbatches"],
            "update": update["update"],
        }
        for update in updates
    ]


def build_candidate_replay_matrix(
    rectangles: Mapping[str, Sequence[ReplayRectangle]],
    config: ReplayScheduleConfig,
) -> dict[str, object]:
    """Build candidate schedules with one invariant rectangle order.

    The complete ETTR and dense-control schedule is byte-identical within a
    candidate.  Across tokenizers, update identities remain identical while
    charged positions are candidate-specific and explicit.
    """

    if not rectangles:
        raise CapabilityFloorReplayError("candidate replay matrix is empty")
    schedules = {
        candidate: build_replay_schedule(tuple(rows), config)
        for candidate, rows in sorted(rectangles.items())
    }
    first_candidate = next(iter(schedules))
    shared = _shared_update_manifest(schedules[first_candidate])
    if any(
        _shared_update_manifest(schedule) != shared
        for schedule in schedules.values()
    ):
        raise CapabilityFloorReplayError("candidate rectangle schedules diverge")
    payload: dict[str, object] = {
        "arm_schedule_sha256": {
            candidate: {
                "dense": replay_schedule_sha256(schedule),
                "ettr": replay_schedule_sha256(schedule),
            }
            for candidate, schedule in schedules.items()
        },
        "candidate_schedules": schedules,
        "charged_positions": "candidate-tokenizer-specific",
        "config": asdict(config),
        "ettr_dense_schedule_identity": "byte-identical-within-candidate",
        "schema": REPLAY_MATRIX_SCHEMA,
        "shared_rectangle_order_across_candidates": True,
        "shared_update_manifest_sha256": hashlib.sha256(
            _canonical_bytes({"updates": shared})
        ).hexdigest(),
    }
    validate_candidate_replay_matrix(payload, rectangles)
    return payload


def validate_candidate_replay_matrix(
    payload: Mapping[str, object],
    rectangles: Mapping[str, Sequence[ReplayRectangle]],
) -> None:
    if (
        payload.get("schema") != REPLAY_MATRIX_SCHEMA
        or payload.get("charged_positions") != "candidate-tokenizer-specific"
        or payload.get("ettr_dense_schedule_identity")
        != "byte-identical-within-candidate"
        or payload.get("shared_rectangle_order_across_candidates") is not True
    ):
        raise CapabilityFloorReplayError("candidate replay matrix custody differs")
    config_payload = payload.get("config")
    schedules = payload.get("candidate_schedules")
    arm_hashes = payload.get("arm_schedule_sha256")
    if (
        not isinstance(config_payload, Mapping)
        or not isinstance(schedules, Mapping)
        or not isinstance(arm_hashes, Mapping)
    ):
        raise CapabilityFloorReplayError("candidate replay matrix structure differs")
    try:
        config = ReplayScheduleConfig(**dict(config_payload))
    except TypeError as error:
        raise CapabilityFloorReplayError("candidate replay config differs") from error
    config.validate()
    if set(schedules) != set(rectangles):
        raise CapabilityFloorReplayError("candidate replay set differs")
    if set(arm_hashes) != set(rectangles):
        raise CapabilityFloorReplayError("candidate replay arm set differs")
    shared = None
    for candidate in sorted(schedules):
        schedule = schedules[candidate]
        if not isinstance(schedule, Mapping):
            raise CapabilityFloorReplayError("candidate replay schedule differs")
        validate_replay_schedule(schedule, rectangles[candidate])
        schedule_hash = replay_schedule_sha256(schedule)
        if arm_hashes[candidate] != {
            "dense": schedule_hash,
            "ettr": schedule_hash,
        }:
            raise CapabilityFloorReplayError("ETTR and dense schedules diverge")
        candidate_shared = _shared_update_manifest(schedule)
        if shared is None:
            shared = candidate_shared
        elif candidate_shared != shared:
            raise CapabilityFloorReplayError("candidate rectangle schedules diverge")
    assert shared is not None
    expected = hashlib.sha256(_canonical_bytes({"updates": shared})).hexdigest()
    if payload.get("shared_update_manifest_sha256") != expected:
        raise CapabilityFloorReplayError("shared update manifest digest differs")


def candidate_replay_matrix_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()
