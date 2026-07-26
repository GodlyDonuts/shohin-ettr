from __future__ import annotations

from dataclasses import replace
import json

import pytest

import ettr_il_v2_semantics as semantics
from cross_ontology_horn_board import (
    THEORIES as HORN_THEORIES,
    GroundAtom,
    challenge_initials,
)
from cross_ontology_resource_board import (
    THEORIES as RESOURCE_THEORIES,
    Marking,
    ProcessStatus,
    input_markings,
)
from cross_ontology_rewrite_board import (
    THEORIES as REWRITE_THEORIES,
    GroundTerm,
    challenge_terms,
)
from ettr_il_v2_semantics import (
    CHECKERBOARD_PATTERNS,
    HornCommand,
    HornPolicy,
    HornWorld,
    Ontology,
    QueryOp,
    ResourceCommand,
    ResourcePolicy,
    ResourceWorld,
    RewriteCommand,
    RewritePolicy,
    RewriteWorld,
    SemanticAdmissionError,
    SemanticError,
    SemanticQuery,
    SemanticRectangle,
    StepOutcome,
    TerminalDisposition,
    admissible_checkerboard_queries,
    checkerboard_labels,
    enumerate_queries,
    evaluate_query,
    execute_horn,
    execute_resource,
    execute_rewrite,
    query_surface_value,
    replay_horn,
    replay_resource,
    replay_rewrite,
    select_queries,
)


EVIDENCE_ID = "0" * 64


def _dependent_horn_case(depth: int = 3) -> tuple[HornWorld, HornCommand]:
    for theory_index in range(len(HORN_THEORIES)):
        for initial in challenge_initials():
            for operations in __import__("itertools").product(
                semantics.all_ground_atoms(),
                repeat=depth,
            ):
                world = HornWorld(
                    EVIDENCE_ID,
                    theory_index,
                    initial,
                    HornPolicy.PERSISTENT,
                )
                command = HornCommand(depth, tuple(operations))
                try:
                    execute_horn(world, command)
                except SemanticAdmissionError:
                    continue
                return world, command
    raise AssertionError("bounded Horn board has no dependent case")


def _dependent_rewrite_case(
    depth: int = 3,
    *,
    policy: RewritePolicy = RewritePolicy.CONTEXTUAL,
) -> tuple[RewriteWorld, RewriteCommand]:
    for theory_index in range(len(REWRITE_THEORIES)):
        for initial in challenge_terms():
            if initial.type_index != 0:
                continue
            for operations in __import__("itertools").product((0, 1), repeat=depth):
                world = RewriteWorld(
                    EVIDENCE_ID,
                    theory_index,
                    initial,
                    policy,
                )
                command = RewriteCommand(depth, tuple(operations))
                try:
                    result = execute_rewrite(world, command)
                except SemanticAdmissionError:
                    continue
                if result.disposition == TerminalDisposition.ANSWER:
                    return world, command
    raise AssertionError("bounded rewrite board has no dependent case")


def _dependent_resource_case(
    depth: int = 3,
    *,
    policy: ResourcePolicy = ResourcePolicy.ATOMIC_DEADLOCK,
) -> tuple[ResourceWorld, ResourceCommand]:
    for theory_index in range(len(RESOURCE_THEORIES)):
        for initial in input_markings():
            for operations in __import__("itertools").product((0, 1, 2), repeat=depth):
                world = ResourceWorld(
                    EVIDENCE_ID,
                    theory_index,
                    initial,
                    policy,
                )
                command = ResourceCommand(depth, tuple(operations))
                try:
                    result = execute_resource(world, command)
                except SemanticAdmissionError:
                    continue
                if all(
                    step.outcome == StepOutcome.APPLIED
                    for step in result.steps
                ):
                    return world, command
    raise AssertionError("bounded resource board has no dependent case")


def _horn_checkerboard() -> SemanticRectangle:
    worlds = (
        HornWorld(
            EVIDENCE_ID,
            0,
            (
                GroundAtom(3, (1, 3)),
                GroundAtom(4, (3, 2)),
            ),
            HornPolicy.PERSISTENT,
        ),
        HornWorld(
            EVIDENCE_ID,
            0,
            (
                GroundAtom(3, (0, 3)),
                GroundAtom(4, (3, 2)),
            ),
            HornPolicy.PERSISTENT,
        ),
    )
    commands = (
        HornCommand(1, (GroundAtom(0, (0,)),)),
        HornCommand(1, (GroundAtom(0, (1,)),)),
    )
    return SemanticRectangle(
        tuple(
            execute_horn(
                world,
                command,
                require_dependent=False,
            )
            for world in worlds
            for command in commands
        )
    )


def test_strict_world_and_command_dataclasses_reject_coercions() -> None:
    atom = GroundAtom(0, (0,))
    with pytest.raises(SemanticError, match="theory index"):
        HornWorld(EVIDENCE_ID, True, (atom,), HornPolicy.PERSISTENT)
    with pytest.raises(SemanticError, match="sorted unique"):
        HornWorld(
            EVIDENCE_ID,
            0,
            (atom, atom),
            HornPolicy.PERSISTENT,
        )
    with pytest.raises(SemanticError, match="policy"):
        HornWorld(EVIDENCE_ID, 0, (atom,), "persistent")  # type: ignore[arg-type]
    with pytest.raises(SemanticError, match="operation count"):
        HornCommand(2, (atom,))
    with pytest.raises(SemanticError, match="bounded initial"):
        RewriteWorld(
            EVIDENCE_ID,
            0,
            GroundTerm(
                0,
                5,
                (
                    GroundTerm(0, 5, (GroundTerm(0, 0), GroundTerm(0, 1))),
                    GroundTerm(0, 5, (GroundTerm(0, 0), GroundTerm(0, 1))),
                ),
            ),
            RewritePolicy.CONTEXTUAL,
        )
    with pytest.raises(SemanticError, match="depth"):
        ResourceCommand(7, (0, 0, 0, 0, 0, 0, 0))


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6])
def test_resource_v2_depth_guard_accepts_one_through_six(depth: int) -> None:
    command = ResourceCommand(depth, (0,) * depth)
    assert command.depth == depth


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6])
def test_all_ontologies_execute_dependent_prefixes_through_depth_six(
    depth: int,
) -> None:
    horn_operations = (
        GroundAtom(0, (1,)),
        GroundAtom(0, (2,)),
        GroundAtom(2, (3,)),
        GroundAtom(2, (4,)),
        GroundAtom(2, (5,)),
        GroundAtom(3, (0, 3)),
    )
    horn_world = HornWorld(
        EVIDENCE_ID,
        0,
        (GroundAtom(0, (0,)),),
        HornPolicy.PERSISTENT,
    )
    horn_command = HornCommand(depth, horn_operations[:depth])
    horn = execute_horn(horn_world, horn_command)
    assert horn == replay_horn(horn_world, horn_command)
    assert len(horn.snapshots) == depth + 1
    assert all(step.prefix_dependent for step in horn.steps)

    rewrite_world = RewriteWorld(
        EVIDENCE_ID,
        0,
        GroundTerm(0, 0),
        RewritePolicy.CONTEXTUAL,
    )
    rewrite_command = RewriteCommand(depth, (0,) * depth)
    rewrite = execute_rewrite(rewrite_world, rewrite_command)
    assert rewrite == replay_rewrite(rewrite_world, rewrite_command)
    assert len(rewrite.snapshots) == depth + 1
    assert all(step.prefix_dependent for step in rewrite.steps)

    resource_operations = (2, 1, 2, 0, 0, 1)
    resource_world = ResourceWorld(
        EVIDENCE_ID,
        0,
        Marking((0, 1, 0, 2)),
        ResourcePolicy.ATOMIC_DEADLOCK,
    )
    resource_command = ResourceCommand(
        depth,
        resource_operations[:depth],
    )
    resource = execute_resource(resource_world, resource_command)
    assert resource == replay_resource(resource_world, resource_command)
    assert len(resource.snapshots) == depth + 1
    assert all(step.prefix_dependent for step in resource.steps)


def test_horn_dependent_execution_replay_and_snapshots_agree() -> None:
    world, command = _dependent_horn_case()
    primary = execute_horn(world, command)
    replay = replay_horn(world, command)
    assert primary == replay
    assert len(primary.snapshots) == command.depth + 1
    assert len(primary.steps) == command.depth
    assert all(step.prefix_dependent for step in primary.steps)
    assert all(step.outcome == StepOutcome.APPLIED for step in primary.steps)


def test_horn_derived_only_policy_projects_asserted_facts() -> None:
    world, command = _dependent_horn_case(depth=1)
    derived = execute_horn(
        replace(world, policy=HornPolicy.DERIVED_ONLY),
        command,
    )
    assert command.operations[0] not in derived.terminal
    assert replay_horn(
        replace(world, policy=HornPolicy.DERIVED_ONLY),
        command,
    ) == derived


@pytest.mark.parametrize(
    "policy",
    [RewritePolicy.CONTEXTUAL, RewritePolicy.ROOT_ONLY],
)
def test_rewrite_execution_replay_and_snapshots_agree(
    policy: RewritePolicy,
) -> None:
    world, command = _dependent_rewrite_case(policy=policy)
    primary = execute_rewrite(world, command)
    replay = replay_rewrite(world, command)
    assert primary == replay
    assert len(primary.snapshots) == command.depth + 1
    assert len(primary.steps) == command.depth
    assert all(step.prefix_dependent for step in primary.steps)
    assert primary.disposition == TerminalDisposition.ANSWER


def test_rewrite_ambiguous_normal_forms_are_not_boolean_answers() -> None:
    found = None
    for theory_index in range(len(REWRITE_THEORIES)):
        for initial in challenge_terms():
            if initial.type_index != 0:
                continue
            for operation in (0, 1):
                world = RewriteWorld(
                    EVIDENCE_ID,
                    theory_index,
                    initial,
                    RewritePolicy.CONTEXTUAL,
                )
                command = RewriteCommand(1, (operation,))
                result = execute_rewrite(
                    world,
                    command,
                    require_dependent=False,
                )
                if result.disposition == TerminalDisposition.ABSTAIN:
                    found = result
                    break
            if found is not None:
                break
        if found is not None:
            break
    if found is None:
        pytest.skip("finite board has no ambiguous wrapped normal form")
    with pytest.raises(SemanticAdmissionError, match="non-ANSWER"):
        evaluate_query(
            SemanticQuery(QueryOp.REWRITE_ROOT_IS, (0,)),
            found,
        )


def test_resource_dependent_execution_replay_and_snapshots_agree() -> None:
    world, command = _dependent_resource_case()
    primary = execute_resource(world, command)
    replay = replay_resource(world, command)
    assert primary == replay
    assert len(primary.snapshots) == command.depth + 1
    assert len(primary.steps) == command.depth
    assert primary.cursor == command.depth
    assert primary.status == ProcessStatus.HALT
    assert all(step.prefix_dependent for step in primary.steps)


def test_resource_atomic_deadlock_and_skip_blocked_are_explicit() -> None:
    world = ResourceWorld(
        EVIDENCE_ID,
        0,
        Marking((0, 0, 0, 0)),
        ResourcePolicy.ATOMIC_DEADLOCK,
    )
    command = ResourceCommand(2, (0, 0))
    deadlock = execute_resource(
        world,
        command,
        require_dependent=False,
    )
    assert deadlock.status == ProcessStatus.DEADLOCK
    assert deadlock.cursor == 0
    assert deadlock.steps[0].outcome == StepOutcome.DEADLOCK
    skipped = execute_resource(
        replace(world, policy=ResourcePolicy.SKIP_BLOCKED),
        command,
        require_dependent=False,
    )
    assert skipped.status == ProcessStatus.HALT
    assert skipped.cursor == 2
    assert tuple(step.outcome for step in skipped.steps) == (
        StepOutcome.SKIPPED,
        StepOutcome.SKIPPED,
    )
    assert replay_resource(world, command, require_dependent=False) == deadlock


def test_replay_paths_do_not_call_primary_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    horn_world, horn_command = _dependent_horn_case(depth=1)
    rewrite_world, rewrite_command = _dependent_rewrite_case(depth=1)
    resource_world, resource_command = _dependent_resource_case(depth=1)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("primary helper was called")

    monkeypatch.setattr(semantics, "_horn_closure_primary", forbidden)
    monkeypatch.setattr(semantics, "_execute_horn_with", forbidden)
    monkeypatch.setattr(
        semantics,
        "_rewrite_normal_forms_primary",
        forbidden,
    )
    monkeypatch.setattr(semantics, "_execute_rewrite_with", forbidden)
    monkeypatch.setattr(
        semantics,
        "_resource_transition_primary",
        forbidden,
    )
    monkeypatch.setattr(semantics, "_execute_resource_with", forbidden)
    replay_horn(horn_world, horn_command)
    replay_rewrite(rewrite_world, rewrite_command)
    replay_resource(resource_world, resource_command)


def test_query_enumeration_is_exact_finite_and_ordered() -> None:
    horn = enumerate_queries(Ontology.HORN)
    rewrite = enumerate_queries(Ontology.REWRITE)
    resource = enumerate_queries(Ontology.RESOURCE)
    assert len(horn) == 54
    assert len(rewrite) == 48
    assert len(resource) == 19
    assert horn[0].op == QueryOp.HORN_HAS
    assert horn[-1] == SemanticQuery(QueryOp.HORN_COUNT_GE, (27,))
    assert rewrite[:8] == tuple(
        SemanticQuery(QueryOp.REWRITE_ROOT_IS, (index,))
        for index in range(8)
    )
    assert resource[-1] == SemanticQuery(QueryOp.RESOURCE_HALT, ())


def test_strict_checkerboard_admission_and_query_selection() -> None:
    rectangle = _horn_checkerboard()
    admitted = admissible_checkerboard_queries(rectangle)
    assert admitted
    for query in admitted:
        assert checkerboard_labels(query, rectangle) in CHECKERBOARD_PATTERNS
    rejected = next(
        query
        for query in enumerate_queries(Ontology.HORN)
        if query not in admitted
    )
    with pytest.raises(SemanticAdmissionError, match="strict checkerboard"):
        checkerboard_labels(rejected, rectangle)

    denotation_witness = execute_horn(
        HornWorld(
            EVIDENCE_ID,
            0,
            (GroundAtom(0, (0,)),),
            HornPolicy.PERSISTENT,
        ),
        HornCommand(1, (GroundAtom(0, (2,)),)),
        require_dependent=False,
    )
    universe = (*rectangle.cells, denotation_witness)
    selected = select_queries(
        rectangle,
        bounded_terminal_universe=universe,
    )
    assert selected.slot_0 != selected.slot_1
    assert selected.slot_0_labels in CHECKERBOARD_PATTERNS
    assert selected.slot_1_labels in CHECKERBOARD_PATTERNS
    assert selected.slot_1_denotation != selected.slot_0_denotation
    assert selected.slot_1_denotation != tuple(
        not value for value in selected.slot_0_denotation
    )
    assert selected == select_queries(
        rectangle,
        bounded_terminal_universe=universe,
    )


def test_checkerboard_rejects_non_answer_and_mixed_ontology_cells() -> None:
    rectangle = _horn_checkerboard()
    with pytest.raises(SemanticAdmissionError, match="non-ANSWER"):
        SemanticRectangle(
            (
                replace(
                    rectangle.cells[0],
                    disposition=TerminalDisposition.REJECT,
                ),
                *rectangle.cells[1:],
            )
        )
    resource_world, resource_command = _dependent_resource_case(depth=1)
    with pytest.raises(SemanticError, match="mixes ontologies"):
        SemanticRectangle(
            (
                execute_resource(resource_world, resource_command),
                *rectangle.cells[1:],
            )
        )


def test_candidate_query_surface_contains_no_ontology_tag() -> None:
    query = SemanticQuery(QueryOp.HORN_HAS, (0, 0))
    first = query_surface_value(
        query,
        operator_symbol="x0123456789abcdef",
        paraphrase=0,
    )
    second = query_surface_value(
        query,
        operator_symbol="x0123456789abcdef",
        paraphrase=1,
    )
    assert first != second
    for value in (first, second):
        encoded = json.dumps(value, sort_keys=True).lower()
        assert set(value) == {"a", "h"}
        assert not any(
            forbidden in encoded
            for forbidden in (
                "horn",
                "rewrite",
                "resource",
                "ontology",
                "theory",
                "target",
                "answer",
            )
        )
        assert query.op.value not in encoded
    with pytest.raises(SemanticError, match="not opaque"):
        query_surface_value(
            query,
            operator_symbol="horn_has",
            paraphrase=0,
        )
