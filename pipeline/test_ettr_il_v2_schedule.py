from __future__ import annotations

import hashlib

import pytest

from ettr_il_v2_schedule import (
    COMPLETE_EPOCHS,
    FIT_ONTOLOGIES,
    InvariantPairRecord,
    MODEL_SEEDS,
    PAIR_EXPOSURES,
    PAIRS_PER_FOLD,
    ScheduleError,
    TAIL_PAIRS,
    UPDATES,
    build_pair_schedule,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _population(fold: int) -> list[InvariantPairRecord]:
    values: list[InvariantPairRecord] = []
    for ontology in FIT_ONTOLOGIES[fold]:
        for depth in (1, 2, 3):
            for core_index in range(96):
                core = _digest(f"core|{fold}|{ontology}|{depth}|{core_index}")
                for pair_index in range(2):
                    values.append(
                        InvariantPairRecord(
                            pair_id=_digest(f"pair|{core}|{pair_index}"),
                            semantic_core_id=core,
                            ontology=ontology,
                            depth=depth,
                            left_semantic_rectangle_id=_digest(
                                f"left|{core}|{pair_index}"
                            ),
                            right_semantic_rectangle_id=_digest(
                                f"right|{core}|{pair_index}"
                            ),
                        )
                    )
    return values


def test_exact_pair_schedule_geometry_and_multiplicity() -> None:
    schedule = build_pair_schedule(
        _population(0),
        fold=0,
        seed=MODEL_SEEDS[0],
    )
    receipt = schedule.receipt()
    assert len(schedule.exposures) == PAIR_EXPOSURES == UPDATES * 4
    assert receipt["pair_population"] == PAIRS_PER_FOLD
    assert receipt["complete_epochs"] == COMPLETE_EPOCHS == 20
    assert receipt["tail_pairs"] == TAIL_PAIRS == 960
    assert receipt["exposure_multiplicities"] == [192, 960]
    assert schedule.exposures[0].update == 0
    assert schedule.exposures[3].microstep == 3
    assert schedule.exposures[4].update == 1
    assert schedule.exposures[-1].update == 5999


def test_schedule_is_deterministic_and_seed_specific() -> None:
    population = _population(1)
    first = build_pair_schedule(
        population,
        fold=1,
        seed=MODEL_SEEDS[0],
    )
    replay = build_pair_schedule(
        reversed(population),
        fold=1,
        seed=MODEL_SEEDS[0],
    )
    second_seed = build_pair_schedule(
        population,
        fold=1,
        seed=MODEL_SEEDS[1],
    )
    assert first.population_sha256 == replay.population_sha256
    assert first.schedule_sha256 == replay.schedule_sha256
    assert first.exposures == replay.exposures
    assert first.schedule_sha256 != second_seed.schedule_sha256


def test_rejects_unpaired_rectangle_reuse() -> None:
    population = _population(2)
    first = population[0]
    second = population[1]
    population[1] = InvariantPairRecord(
        pair_id=second.pair_id,
        semantic_core_id=second.semantic_core_id,
        ontology=second.ontology,
        depth=second.depth,
        left_semantic_rectangle_id=first.left_semantic_rectangle_id,
        right_semantic_rectangle_id=second.right_semantic_rectangle_id,
    )
    with pytest.raises(ScheduleError, match="multiple pairs"):
        build_pair_schedule(
            population,
            fold=2,
            seed=MODEL_SEEDS[0],
        )


def test_rejects_wrong_population_geometry_and_seed() -> None:
    population = _population(0)
    with pytest.raises(ScheduleError, match="population count"):
        build_pair_schedule(
            population[:-1],
            fold=0,
            seed=MODEL_SEEDS[0],
        )
    with pytest.raises(ScheduleError, match="model seed"):
        build_pair_schedule(population, fold=0, seed=1)
