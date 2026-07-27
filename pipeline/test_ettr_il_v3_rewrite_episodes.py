"""Tests for ETTR-IL-v3 local-rewrite curriculum episodes."""

from __future__ import annotations

import pytest

from ettr_il_v3_horn_resource import CurriculumStage
from ettr_il_v3_rewrite import (
    Direction,
    LocalOperation,
    RewriteCommand,
    RewriteWorld,
)
from ettr_il_v3_rewrite_episodes import (
    RewriteEpisodeError,
    build_rewrite_episode,
    counterfactual_bundle,
    generate_rewrite_episodes,
    minimal_query_counterfactual,
    population_sha256,
)


def test_atomic_episode_binds_independent_execution_and_queries() -> None:
    episode = build_rewrite_episode(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        world=RewriteWorld(0, (0, 0, 0, 0, 0, 0)),
        command=RewriteCommand((LocalOperation(0, 0, Direction.FORWARD),)),
    )
    assert episode.primary == episode.replay
    assert episode.primary.terminal == (1, 1, 0, 0, 0, 0)
    assert episode.answers == (False, True)
    assert episode.capacity.encoded_trace_steps <= 64
    assert len(episode.episode_id) == 64


def test_dependent_stage_rejects_non_dependent_path() -> None:
    with pytest.raises(RewriteEpisodeError, match="prefix-dependent"):
        build_rewrite_episode(
            stage=CurriculumStage.DEPENDENT_COMPOSITION,
            world=RewriteWorld(0, (0, 0, 0, 0, 0, 0)),
            command=RewriteCommand(
                (
                    LocalOperation(0, 0, Direction.FORWARD),
                    LocalOperation(0, 2, Direction.FORWARD),
                )
            ),
        )


def test_all_stage_generators_are_deterministic_and_nonempty() -> None:
    depths = {
        CurriculumStage.COMPILER_GROUNDING: 1,
        CurriculumStage.ATOMIC_TRANSITIONS: 1,
        CurriculumStage.DEPENDENT_COMPOSITION: 2,
        CurriculumStage.QUERY_COUNTERFACTUAL_GROUNDING: 1,
        CurriculumStage.CLOSED_LOOP: 2,
    }
    for stage, depth in depths.items():
        first = generate_rewrite_episodes(
            stage=stage,
            theory_index=0,
            depth=depth,
            limit=3,
            beam_width=16,
        )
        second = generate_rewrite_episodes(
            stage=stage,
            theory_index=0,
            depth=depth,
            limit=3,
            beam_width=16,
        )
        assert first
        assert tuple(item.episode_id for item in first) == tuple(
            item.episode_id for item in second
        )


def test_bucket_partition_is_disjoint() -> None:
    left = generate_rewrite_episodes(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        theory_index=1,
        limit=25,
        bucket_index=0,
        bucket_count=2,
    )
    right = generate_rewrite_episodes(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        theory_index=1,
        limit=25,
        bucket_index=1,
        bucket_count=2,
    )
    assert left and right
    assert {item.world.index for item in left}.isdisjoint(
        item.world.index for item in right
    )


def test_counterfactual_query_and_closed_bundle_exist() -> None:
    query_episode = generate_rewrite_episodes(
        stage=CurriculumStage.QUERY_COUNTERFACTUAL_GROUNDING,
        theory_index=2,
        depth=1,
        limit=1,
    )[0]
    assert any(
        minimal_query_counterfactual(query_episode, query_index=index) is not None
        for index in range(2)
    )

    closed = generate_rewrite_episodes(
        stage=CurriculumStage.CLOSED_LOOP,
        theory_index=0,
        depth=2,
        limit=1,
        beam_width=64,
    )[0]
    bundle = counterfactual_bundle(closed)
    assert bundle is not None
    assert {
        bundle.world.axis,
        bundle.command.axis,
        bundle.query.axis,
    } == {"world", "command", "query"}


def test_population_hash_is_order_independent_and_rejects_duplicates() -> None:
    episodes = generate_rewrite_episodes(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        theory_index=3,
        limit=4,
    )
    assert population_sha256(episodes) == population_sha256(reversed(episodes))
    with pytest.raises(RewriteEpisodeError, match="repeats"):
        population_sha256((episodes[0], episodes[0]))
