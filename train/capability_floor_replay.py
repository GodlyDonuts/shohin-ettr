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
import random
from typing import Mapping, Sequence


REPLAY_SCHEMA = "shohin-ettr-component-stratified-replay-v1"


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
