from copy import deepcopy

import pytest

from capability_floor_replay import (
    CapabilityFloorReplayError,
    ReplayRectangle,
    ReplayScheduleConfig,
    build_replay_schedule,
    replay_schedule_sha256,
    validate_replay_schedule,
)


def _rectangles() -> list[ReplayRectangle]:
    families = ("NONE", "WRITE", "LINK")
    return [
        ReplayRectangle(
            rectangle_id=f"rectangle-{index:03d}",
            strata=(families[index % len(families)],),
            charged_positions=100 + index,
        )
        for index in range(48)
    ]


def _config() -> ReplayScheduleConfig:
    return ReplayScheduleConfig(
        component="oracle-program-executor",
        required_strata=("NONE", "WRITE", "LINK"),
        updates=5,
        seed=31,
        dataset_sha256="b" * 64,
    )


def test_schedule_is_deterministic_atomic_and_stratified() -> None:
    rectangles = _rectangles()
    first = build_replay_schedule(rectangles, _config())
    second = build_replay_schedule(rectangles, _config())
    assert first == second
    assert replay_schedule_sha256(first) == replay_schedule_sha256(second)
    for update in first["updates"]:
        assert update["covered_strata"] == ["LINK", "NONE", "WRITE"]
        assert len(update["microbatches"]) == 4
        flattened = [value for batch in update["microbatches"] for value in batch]
        assert len(flattened) == 16
        assert len(set(flattened)) == 16


def test_schedule_changes_with_seed_but_not_contract() -> None:
    rectangles = _rectangles()
    first = build_replay_schedule(rectangles, _config())
    second_config = ReplayScheduleConfig(
        component=_config().component,
        required_strata=_config().required_strata,
        updates=_config().updates,
        seed=32,
        dataset_sha256=_config().dataset_sha256,
    )
    second = build_replay_schedule(rectangles, second_config)
    assert first["updates"] != second["updates"]
    validate_replay_schedule(second, rectangles)


def test_schedule_rejects_missing_stratum() -> None:
    rectangles = [
        ReplayRectangle(
            rectangle_id=f"rectangle-{index:03d}",
            strata=("NONE",),
        )
        for index in range(16)
    ]
    with pytest.raises(CapabilityFloorReplayError, match="WRITE is empty"):
        build_replay_schedule(rectangles, _config())


def test_schedule_rejects_non_atomic_or_duplicate_rectangles() -> None:
    rectangles = _rectangles()
    rectangles[0] = ReplayRectangle(
        rectangle_id=rectangles[0].rectangle_id,
        strata=rectangles[0].strata,
        row_count=3,
    )
    with pytest.raises(CapabilityFloorReplayError, match="four rows"):
        build_replay_schedule(rectangles, _config())


def test_validation_rejects_within_update_repetition_and_coverage_loss() -> None:
    rectangles = _rectangles()
    payload = build_replay_schedule(rectangles, _config())
    duplicate = deepcopy(payload)
    duplicate["updates"][0]["microbatches"][0][1] = duplicate["updates"][0][
        "microbatches"
    ][0][0]
    with pytest.raises(CapabilityFloorReplayError, match="repeats"):
        validate_replay_schedule(duplicate, rectangles)

    no_link = deepcopy(payload)
    non_link = [item.rectangle_id for item in rectangles if "LINK" not in item.strata]
    replacement = iter(non_link)
    used: set[str] = set()
    for microbatch in no_link["updates"][0]["microbatches"]:
        for index in range(len(microbatch)):
            while True:
                candidate = next(replacement)
                if candidate not in used:
                    break
            microbatch[index] = candidate
            used.add(candidate)
    no_link["updates"][0]["covered_strata"] = ["NONE", "WRITE"]
    with pytest.raises(CapabilityFloorReplayError, match="omits"):
        validate_replay_schedule(no_link, rectangles)
