from __future__ import annotations

import re

import pytest

from cross_ontology_horn_board import GroundAtom
from cross_ontology_resource_board import Marking, ProcessStatus
from ettr_il_v2_semantics import (
    HornCommand,
    HornPolicy,
    HornWorld,
    QueryOp,
    ResourceCommand,
    ResourcePolicy,
    ResourceWorld,
    SemanticQuery,
    StepOutcome,
)
from ettr_il_v3_horn_resource import (
    MAX_PACKET_EDGES,
    MAX_PACKET_SLOTS,
    MAX_TRACE_STEPS,
    CounterfactualAxis,
    CurriculumStage,
    EpisodeRecord,
    V3EpisodeError,
    admitted_commands,
    build_episode,
    counterfactual_bundle,
    domain_cardinality_receipt,
    full_resource_markings,
    generate_horn_episodes,
    generate_resource_episodes,
    horn_worlds,
    minimal_command_counterfactual,
    minimal_query_counterfactual,
    minimal_world_counterfactual,
    population_sha256,
    resource_worlds,
)


EVIDENCE_ID = "0" * 64
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
HORN_OPERATIONS = (
    GroundAtom(0, (1,)),
    GroundAtom(0, (2,)),
    GroundAtom(2, (3,)),
    GroundAtom(2, (4,)),
    GroundAtom(2, (5,)),
    GroundAtom(3, (0, 3)),
)
RESOURCE_OPERATIONS = (2, 1, 2, 0, 0, 1)


def _horn_world() -> HornWorld:
    return HornWorld(
        EVIDENCE_ID,
        0,
        (GroundAtom(0, (0,)),),
        HornPolicy.PERSISTENT,
    )


def _resource_world() -> ResourceWorld:
    return ResourceWorld(
        EVIDENCE_ID,
        0,
        Marking((0, 1, 0, 2)),
        ResourcePolicy.ATOMIC_DEADLOCK,
    )


def _counterfactual_source() -> EpisodeRecord:
    queries = (
        SemanticQuery(QueryOp.RESOURCE_PLACE_GE, (1, 1)),
        SemanticQuery(QueryOp.RESOURCE_PLACE_GE, (1, 2)),
    )
    return build_episode(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        world=ResourceWorld(
            EVIDENCE_ID,
            0,
            Marking((1, 0, 1, 0)),
            ResourcePolicy.ATOMIC_DEADLOCK,
        ),
        command=ResourceCommand(1, (0,)),
        queries=queries,
    )


def test_static_domain_cardinality_receipt_is_hash_bound() -> None:
    receipt = domain_cardinality_receipt()
    assert receipt == {
        "horn": {
            "atomic_operations": 27,
            "raw_atomic_pairs_per_policy": 204_120,
            "theories": 20,
            "worlds_per_theory_policy": 378,
        },
        "protocol": "R12-ETTR-IL-v3-initializer",
        "resource": {
            "atomic_operations": 3,
            "raw_atomic_pairs_per_policy": 46_080,
            "theories": 60,
            "worlds_per_theory_policy": 256,
        },
        "schema": "r12-ettr-il-v3-horn-resource-domain-receipt-v1",
        "receipt_sha256": (
            "2d95101897c35eccac15ed88857b4322b72125c784faaec3586de816a6099588"
        ),
    }


def test_world_domains_are_canonical_and_resource_uses_all_4_pow_4_states() -> None:
    markings = full_resource_markings()
    assert len(markings) == 4**4 == 256
    assert len(set(markings)) == 256
    assert markings[0] == Marking((0, 0, 0, 0))
    assert markings[-1] == Marking((3, 3, 3, 3))
    assert len(horn_worlds(0)) == 378
    assert len(resource_worlds(0)) == 256
    assert len({world.evidence_id for world in horn_worlds(0)}) == 1
    assert len({world.evidence_id for world in resource_worlds(0)}) == 1


@pytest.mark.parametrize("depth", range(2, 7))
def test_horn_dependent_composition_depths_two_through_six(
    depth: int,
) -> None:
    episode = build_episode(
        stage=CurriculumStage.DEPENDENT_COMPOSITION,
        world=_horn_world(),
        command=HornCommand(depth, HORN_OPERATIONS[:depth]),
    )
    assert episode.primary == episode.replay
    assert episode.command.depth == depth
    assert len(episode.primary.steps) == depth
    assert all(step.outcome is StepOutcome.APPLIED for step in episode.primary.steps)
    assert all(step.prefix_dependent for step in episode.primary.steps)
    assert episode.answers == (False, True)
    assert episode.coverage.prefix_dependent_steps == depth
    assert HEX_64.fullmatch(episode.episode_id)
    episode.capacity.validate()
    assert episode.capacity.initial_active_slots <= MAX_PACKET_SLOTS
    assert episode.capacity.terminal_edges <= MAX_PACKET_EDGES
    assert episode.capacity.encoded_trace_steps <= MAX_TRACE_STEPS


@pytest.mark.parametrize("depth", range(2, 7))
def test_resource_dependent_composition_depths_two_through_six(
    depth: int,
) -> None:
    episode = build_episode(
        stage=CurriculumStage.DEPENDENT_COMPOSITION,
        world=_resource_world(),
        command=ResourceCommand(depth, RESOURCE_OPERATIONS[:depth]),
    )
    assert episode.primary == episode.replay
    assert episode.command.depth == depth
    assert len(episode.primary.steps) == depth
    assert all(step.outcome is StepOutcome.APPLIED for step in episode.primary.steps)
    assert all(step.prefix_dependent for step in episode.primary.steps)
    assert episode.answers == (False, True)
    assert episode.coverage.prefix_dependent_steps == depth
    assert episode.capacity.initial_active_slots == 24
    assert episode.capacity.terminal_active_slots == 32
    assert episode.capacity.terminal_edges == 27
    episode.capacity.validate()


def test_all_non_composition_stage_records_are_broad_and_deterministic() -> None:
    world = ResourceWorld(
        EVIDENCE_ID,
        0,
        Marking((1, 0, 1, 0)),
        ResourcePolicy.ATOMIC_DEADLOCK,
    )
    command = ResourceCommand(1, (0,))
    records = tuple(
        build_episode(stage=stage, world=world, command=command)
        for stage in (
            CurriculumStage.COMPILER_GROUNDING,
            CurriculumStage.ATOMIC_TRANSITIONS,
            CurriculumStage.QUERY_COUNTERFACTUAL_GROUNDING,
        )
    )
    replay = tuple(
        build_episode(stage=stage, world=world, command=command)
        for stage in (
            CurriculumStage.COMPILER_GROUNDING,
            CurriculumStage.ATOMIC_TRANSITIONS,
            CurriculumStage.QUERY_COUNTERFACTUAL_GROUNDING,
        )
    )
    assert records == replay
    assert len({record.episode_id for record in records}) == 3
    assert all(record.primary == record.replay for record in records)
    assert all(record.answers == (False, True) for record in records)
    assert tuple(record.coverage.stage for record in records) == tuple(
        stage.value
        for stage in (
            CurriculumStage.COMPILER_GROUNDING,
            CurriculumStage.ATOMIC_TRANSITIONS,
            CurriculumStage.QUERY_COUNTERFACTUAL_GROUNDING,
        )
    )


def test_resource_atomic_generation_is_deterministic_without_xor_admission() -> None:
    first = generate_resource_episodes(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        theory_index=0,
        limit=3,
    )
    replay = generate_resource_episodes(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        theory_index=0,
        limit=3,
    )
    assert first == replay
    assert len(first) == 3
    assert tuple(record.episode_id for record in first) == (
        "9a0915e2c81c43a765aebf4172d52dd02d3fb340aafeefdef35b11e6b761de52",
        "fcae6101ddf86c8b0501a3b080effb2d40f7b27f221d1b22b280049b43bb860d",
        "cfd038c5eb8dcf91d8866902dc9f458fb227643a2db478ca9924e1a17b881a49",
    )
    assert all(record.answers == (False, True) for record in first)


def test_horn_compiler_generation_maximizes_distinct_worlds() -> None:
    records = generate_horn_episodes(
        stage=CurriculumStage.COMPILER_GROUNDING,
        theory_index=0,
        limit=4,
    )
    replay = generate_horn_episodes(
        stage=CurriculumStage.COMPILER_GROUNDING,
        theory_index=0,
        limit=4,
    )
    assert records == replay
    assert len(records) == 4
    assert len({record.world for record in records}) == 4
    assert all(record.command.depth == 1 for record in records)


def test_deterministic_world_buckets_are_disjoint() -> None:
    left = generate_resource_episodes(
        stage=CurriculumStage.COMPILER_GROUNDING,
        theory_index=0,
        limit=8,
        bucket_index=0,
        bucket_count=2,
    )
    right = generate_resource_episodes(
        stage=CurriculumStage.COMPILER_GROUNDING,
        theory_index=0,
        limit=8,
        bucket_index=1,
        bucket_count=2,
    )
    assert {record.world for record in left}.isdisjoint(
        {record.world for record in right}
    )
    assert len(left) == len(right) == 8


def test_minimal_world_command_and_query_counterfactuals_flip_exactly() -> None:
    episode = _counterfactual_source()
    assert episode.answers == (True, False)
    functions = (
        (CounterfactualAxis.WORLD, minimal_world_counterfactual),
        (CounterfactualAxis.COMMAND, minimal_command_counterfactual),
        (CounterfactualAxis.QUERY, minimal_query_counterfactual),
    )
    records = []
    for axis, helper in functions:
        counterfactual = helper(episode, query_index=0)
        assert counterfactual is not None
        assert counterfactual.axis is axis
        assert counterfactual.primary == counterfactual.replay
        assert counterfactual.answer_before is True
        assert counterfactual.answer_after is False
        assert counterfactual.semantic_distance == 1
        assert HEX_64.fullmatch(counterfactual.counterfactual_id)
        counterfactual.capacity.validate()
        records.append(counterfactual)
    assert len({record.counterfactual_id for record in records}) == 3


def test_closed_loop_generation_requires_all_three_counterfactual_axes() -> None:
    records = generate_resource_episodes(
        stage=CurriculumStage.CLOSED_LOOP,
        theory_index=0,
        depth=2,
        beam_width=9,
        limit=1,
    )
    assert len(records) == 1
    episode = records[0]
    bundle = counterfactual_bundle(episode)
    assert bundle is not None
    assert bundle.query_index in (0, 1)
    assert {
        bundle.world.axis,
        bundle.command.axis,
        bundle.query.axis,
    } == set(CounterfactualAxis)
    assert all(
        record.answer_before is not record.answer_after
        for record in (bundle.world, bundle.command, bundle.query)
    )


def test_admitted_command_beams_are_replay_equal_and_hash_stable() -> None:
    world = ResourceWorld(
        EVIDENCE_ID,
        0,
        Marking((0, 1, 0, 2)),
        ResourcePolicy.ATOMIC_DEADLOCK,
    )
    first = admitted_commands(
        world,
        depth=3,
        beam_width=9,
        require_dependent=True,
    )
    replay = admitted_commands(
        world,
        depth=3,
        beam_width=9,
        require_dependent=True,
    )
    assert first == replay
    assert first
    for command in first:
        episode = build_episode(
            stage=CurriculumStage.DEPENDENT_COMPOSITION,
            world=world,
            command=command,
        )
        assert episode.primary == episode.replay


def test_population_hash_is_order_independent_and_rejects_duplicates() -> None:
    records = generate_resource_episodes(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        theory_index=0,
        limit=3,
    )
    digest = population_sha256(records)
    assert digest == population_sha256(reversed(records))
    assert HEX_64.fullmatch(digest)
    with pytest.raises(V3EpisodeError, match="repeats"):
        population_sha256((*records, records[0]))


def test_fail_closed_on_stage_depth_or_bucket_mismatch() -> None:
    with pytest.raises(V3EpisodeError, match="depth one"):
        build_episode(
            stage=CurriculumStage.ATOMIC_TRANSITIONS,
            world=_horn_world(),
            command=HornCommand(2, HORN_OPERATIONS[:2]),
        )
    deadlocking = build_episode(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        world=ResourceWorld(
            EVIDENCE_ID,
            0,
            Marking((0, 0, 0, 0)),
            ResourcePolicy.ATOMIC_DEADLOCK,
        ),
        command=ResourceCommand(1, (0,)),
    )
    assert deadlocking.primary.status is ProcessStatus.DEADLOCK  # type: ignore[union-attr]
    assert deadlocking.coverage.outcome_histogram == (("deadlock", 1),)
    with pytest.raises(V3EpisodeError, match="bucket"):
        generate_resource_episodes(
            stage=CurriculumStage.COMPILER_GROUNDING,
            theory_index=0,
            limit=1,
            bucket_index=2,
            bucket_count=2,
        )
