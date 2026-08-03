from copy import deepcopy
import hashlib
import json

import pytest

from capability_floor_replay import (
    CapabilityFloorReplayError,
    ReplayRectangle,
    ReplayScheduleConfig,
    build_candidate_replay_matrix,
    build_replay_schedule,
    candidate_replay_matrix_sha256,
    load_candidate_replay_rectangles,
    replay_schedule_sha256,
    validate_candidate_replay_matrix,
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


def test_cohort_index_builds_shared_candidate_schedule_with_exact_charges(
    tmp_path,
) -> None:
    candidates = ("candidate-a", "candidate-b")
    rows = []
    for group in range(3):
        rectangles = []
        for offset in range(16):
            index = 16 * group + offset
            rectangles.append(
                {
                    "charged_positions": {
                        "candidate-a": 100 + index,
                        "candidate-b": 200 + 2 * index,
                    },
                    "rectangle_id": f"rectangle-{index:03d}",
                    "strata": [("NONE", "WRITE", "LINK")[index % 3]],
                }
            )
        rows.append(
            {
                "accepted": True,
                "assessor_fields_in_model_input": False,
                "index_schema": "shohin-ettr-capability-floor-core-index-v2",
                "rectangles": rectangles,
                "split": "train" if group < 2 else "development",
            }
        )
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
        for row in rows
    )
    path = tmp_path / "cohort-index.jsonl"
    path.write_bytes(payload)
    inventories = load_candidate_replay_rectangles(
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        candidates=candidates,
        split="train",
    )
    assert len(inventories["candidate-a"]) == 32
    matrix = build_candidate_replay_matrix(inventories, _config())
    validate_candidate_replay_matrix(matrix, inventories)
    schedules = matrix["candidate_schedules"]
    assert [
        update["microbatches"] for update in schedules["candidate-a"]["updates"]
    ] == [
        update["microbatches"] for update in schedules["candidate-b"]["updates"]
    ]
    assert [
        update["charged_positions"]
        for update in schedules["candidate-a"]["updates"]
    ] != [
        update["charged_positions"]
        for update in schedules["candidate-b"]["updates"]
    ]
    assert len(candidate_replay_matrix_sha256(matrix)) == 64


def test_cohort_replay_rejects_hash_or_missing_candidate(tmp_path) -> None:
    payload = (
        json.dumps(
            {
                "accepted": True,
                "assessor_fields_in_model_input": False,
                "index_schema": "shohin-ettr-capability-floor-core-index-v2",
                "rectangles": [
                    {
                        "charged_positions": {"candidate-a": 10},
                        "rectangle_id": "r0",
                        "strata": ["NONE"],
                    }
                ],
                "split": "train",
            }
        )
        + "\n"
    ).encode("ascii")
    path = tmp_path / "cohort-index.jsonl"
    path.write_bytes(payload)
    with pytest.raises(CapabilityFloorReplayError, match="SHA-256"):
        load_candidate_replay_rectangles(
            path,
            expected_sha256="0" * 64,
            candidates=("candidate-a",),
            split="train",
        )
    with pytest.raises(CapabilityFloorReplayError, match="rectangle differs"):
        load_candidate_replay_rectangles(
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            candidates=("candidate-a", "candidate-b"),
            split="train",
        )
