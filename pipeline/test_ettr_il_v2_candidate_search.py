from __future__ import annotations

import pytest

from ettr_il_v2_candidate_search import (
    CandidateSearchError,
    beam_commands,
    find_first_depth1_checkerboard,
    owned_worlds,
    scan_admissible_candidates,
    semantic_core_id,
    semantic_world_id,
    terminal_observation_value,
    terminal_witness_universe,
    world_owner,
)
from ettr_il_v2_semantics import (
    CHECKERBOARD_PATTERNS,
    Ontology,
    execute_semantics,
    replay_semantics,
)


@pytest.mark.parametrize("ontology", tuple(Ontology))
def test_all_three_ontologies_have_a_strict_depth1_checkerboard(
    ontology: Ontology,
) -> None:
    candidate = find_first_depth1_checkerboard(ontology)
    assert candidate.ontology is ontology
    assert candidate.depth == 1
    assert candidate.queries.slot_0_labels in CHECKERBOARD_PATTERNS
    assert candidate.queries.slot_1_labels in CHECKERBOARD_PATTERNS
    assert candidate.queries.slot_0_denotation != (
        candidate.queries.slot_1_denotation
    )
    assert candidate.queries.slot_0_denotation != tuple(
        not value for value in candidate.queries.slot_1_denotation
    )
    assert candidate.worlds[0].evidence_id == candidate.worlds[1].evidence_id
    assert len(semantic_core_id(candidate)) == 64
    for world in candidate.worlds:
        for command in candidate.commands:
            assert execute_semantics(
                world,
                command,
                require_dependent=False,
            ) == replay_semantics(
                world,
                command,
                require_dependent=False,
            )


@pytest.mark.parametrize("ontology", tuple(Ontology))
def test_terminal_witness_universe_is_sorted_unique_and_deterministic(
    ontology: Ontology,
) -> None:
    first = terminal_witness_universe(ontology)
    replay = terminal_witness_universe(ontology)
    assert first is replay
    values = tuple(
        repr(terminal_observation_value(execution))
        for execution in first
    )
    assert len(values) == len(set(values))


def test_invalid_candidate_search_inputs_fail_closed() -> None:
    with pytest.raises(CandidateSearchError, match="ontology"):
        find_first_depth1_checkerboard("horn")  # type: ignore[arg-type]
    with pytest.raises(CandidateSearchError, match="universe"):
        find_first_depth1_checkerboard(
            Ontology.HORN,
            bounded_terminal_universe=(),
        )


@pytest.mark.parametrize("ontology", tuple(Ontology))
def test_world_ownership_is_a_disjoint_total_partition(
    ontology: Ontology,
) -> None:
    split_worlds = {
        split: owned_worlds(
            ontology,
            theory_index=0,
            fold=1,
            split=split,
        )
        for split in ("train", "development", "confirmation")
    }
    ids = {
        split: {semantic_world_id(world) for world in worlds}
        for split, worlds in split_worlds.items()
    }
    assert all(ids[left].isdisjoint(ids[right]) for left, right in (
        ("train", "development"),
        ("train", "confirmation"),
        ("development", "confirmation"),
    ))
    assert sum(map(len, ids.values())) == {
        Ontology.HORN: 378,
        Ontology.REWRITE: 46,
        Ontology.RESOURCE: 81,
    }[ontology]
    for split_index, split in enumerate(
        ("train", "development", "confirmation")
    ):
        assert all(
            world_owner(world, fold=1, ontology=ontology) == split_index
            for world in split_worlds[split]
        )


def test_depth_two_resource_beam_and_candidate_prefix_are_deterministic() -> None:
    worlds = owned_worlds(
        Ontology.RESOURCE,
        theory_index=0,
        fold=0,
        split="development",
    )
    commands, receipt = beam_commands(
        fold=0,
        split="development",
        ontology=Ontology.RESOURCE,
        theory_index=0,
        depth=2,
        worlds=(worlds[0], worlds[1]),
    )
    replay_commands, replay_receipt = beam_commands(
        fold=0,
        split="development",
        ontology=Ontology.RESOURCE,
        theory_index=0,
        depth=2,
        worlds=(worlds[0], worlds[1]),
    )
    assert commands == replay_commands
    assert receipt == replay_receipt
    assert receipt.raw_template_count == 9
    assert receipt.operation_alphabet_size == 3
    assert receipt.final_command_count <= 9

    candidates, scan = scan_admissible_candidates(
        fold=0,
        split="development",
        ontology=Ontology.RESOURCE,
        theory_index=0,
        depth=2,
        stop_after=1,
    )
    replay_candidates, replay_scan = scan_admissible_candidates(
        fold=0,
        split="development",
        ontology=Ontology.RESOURCE,
        theory_index=0,
        depth=2,
        stop_after=1,
    )
    assert candidates == replay_candidates
    assert scan == replay_scan
    assert scan.unique_core_count == len(candidates)
    assert scan.scanned_world_pairs <= scan.owned_world_pair_count
