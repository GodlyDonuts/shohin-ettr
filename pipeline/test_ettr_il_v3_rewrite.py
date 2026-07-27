from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations

import pytest

import ettr_il_v3_rewrite as rewrite
from ettr_il_v3_rewrite import (
    Direction,
    LOCAL_LAWS,
    LocalOperation,
    MAX_DEPTH,
    PRIMITIVE_OPERATION_COUNT,
    QueryOp,
    REGISTER_COUNT,
    REGISTER_TYPES,
    RejectReason,
    RewriteAdmissionError,
    RewriteCommand,
    RewriteSemanticError,
    RewriteWorld,
    StepOutcome,
    StructuralQuery,
    THEORIES,
    THEORY_COUNT,
    TerminalDisposition,
    VALID_PRIMITIVE_OPERATION_COUNT,
    WORLD_COUNT,
    canonical_json_bytes,
    canonical_sha256,
    evaluate_query_primary,
    evaluate_query_replay,
    execute_primary,
    execute_replay,
    exhaustive_primitive_audit,
    iter_worlds,
    primitive_operations,
    structural_queries,
    world_from_index,
    world_index,
)


def _operation(
    law_slot: int,
    site: int,
    direction: Direction = Direction.FORWARD,
) -> LocalOperation:
    return LocalOperation(law_slot, site, direction)


def _command(*operations: LocalOperation) -> RewriteCommand:
    return RewriteCommand(tuple(operations))


def test_frozen_widths_and_domain_cardinalities() -> None:
    assert REGISTER_COUNT == 6
    assert REGISTER_TYPES == (0, 1, 0, 1, 0, 1)
    assert rewrite.TYPE_SYMBOLS == ((0, 1, 2, 3), (0, 1, 2, 3))
    assert WORLD_COUNT == 4**6 == 4096
    assert len(LOCAL_LAWS) == 6
    assert THEORY_COUNT == len(THEORIES) == 15
    assert PRIMITIVE_OPERATION_COUNT == 4 * 8 * 2 == 64
    assert VALID_PRIMITIVE_OPERATION_COUNT == 2 * 5 * 2 == 20
    assert MAX_DEPTH == 6
    assert REGISTER_COUNT <= rewrite.RUNTIME_SLOT_LIMIT == 16
    assert MAX_DEPTH <= rewrite.TRANSACTION_LIMIT == 64


def test_world_enumeration_is_exact_unique_and_indexed() -> None:
    worlds = tuple(iter_worlds())
    assert len(worlds) == WORLD_COUNT
    assert worlds[0].registers == (0, 0, 0, 0, 0, 0)
    assert worlds[-1].registers == (3, 3, 3, 3, 3, 3)
    assert len({world.registers for world in worlds}) == WORLD_COUNT
    assert tuple(world.index for world in worlds) == tuple(range(WORLD_COUNT))
    assert all(
        world_from_index(index).registers == world.registers
        for index, world in enumerate(worlds)
    )
    assert world_from_index(1234, 14).theory_index == 14
    assert world_index((1, 0, 3, 2, 1, 0)) == int("103210", 4)


def test_laws_are_reversible_size_preserving_and_pair_unique() -> None:
    sources = {law.forward_source for law in LOCAL_LAWS}
    targets = {law.forward_target for law in LOCAL_LAWS}
    assert len(sources) == len(targets) == len(LOCAL_LAWS)
    for expected_index, law in enumerate(LOCAL_LAWS):
        assert law.index == expected_index
        assert len(law.forward_source) == len(law.forward_target) == 2
        assert law.forward_source != law.forward_target
        assert all(0 <= symbol < 4 for symbol in law.forward_source)
        assert all(0 <= symbol < 4 for symbol in law.forward_target)

        theory_index = next(
            theory.index
            for theory in THEORIES
            if law.index in theory.law_indices
        )
        law_slot = THEORIES[theory_index].law_indices.index(law.index)
        registers = (
            *law.forward_source,
            0,
            0,
            0,
            0,
        )
        world = RewriteWorld(theory_index, registers)
        forward = execute_primary(
            world,
            _command(_operation(law_slot, 0)),
        )
        reverse = execute_primary(
            replace(world, registers=forward.terminal),
            _command(
                _operation(
                    law_slot,
                    0,
                    Direction.REVERSE,
                )
            ),
        )
        assert forward.terminal[:2] == law.forward_target
        assert forward.terminal[2:] == world.registers[2:]
        assert reverse.terminal == world.registers


def test_theories_are_all_two_law_combinations_with_balanced_incidence() -> None:
    expected = tuple(combinations(range(6), 2))
    assert tuple(theory.law_indices for theory in THEORIES) == expected
    assert tuple(theory.index for theory in THEORIES) == tuple(range(15))
    incidence = Counter(
        law_index
        for theory in THEORIES
        for law_index in theory.law_indices
    )
    assert incidence == Counter({law_index: 5 for law_index in range(6)})


def test_primitive_word_domains_are_canonical_and_exact() -> None:
    complete = primitive_operations()
    valid = primitive_operations(False)
    assert len(complete) == PRIMITIVE_OPERATION_COUNT
    assert len(valid) == VALID_PRIMITIVE_OPERATION_COUNT
    assert len(set(complete)) == len(complete)
    assert complete[0] == _operation(0, 0, Direction.FORWARD)
    assert complete[-1] == _operation(3, 7, Direction.REVERSE)
    assert all(operation.semantically_valid for operation in valid)
    assert sum(operation.semantically_valid for operation in complete) == 20


def test_apply_block_and_reject_outcomes_are_deterministic() -> None:
    world = RewriteWorld(0, (0, 0, 0, 0, 0, 0))
    applied_command = _command(_operation(0, 0))
    blocked_command = _command(_operation(1, 0))
    invalid_slot_command = _command(_operation(2, 0))
    invalid_site_command = _command(_operation(0, 5))

    applied = execute_primary(world, applied_command)
    assert applied == execute_replay(world, applied_command)
    assert applied.steps[0].outcome is StepOutcome.APPLIED
    assert applied.steps[0].resolved_law_index == 0
    assert applied.steps[0].reject_reason is RejectReason.NONE
    assert applied.terminal == (1, 1, 0, 0, 0, 0)
    assert applied.disposition is TerminalDisposition.ANSWER

    blocked = execute_primary(world, blocked_command)
    assert blocked == execute_replay(world, blocked_command)
    assert blocked.steps[0].outcome is StepOutcome.BLOCKED
    assert blocked.steps[0].resolved_law_index == 1
    assert blocked.terminal == world.registers
    assert blocked.disposition is TerminalDisposition.ANSWER

    invalid_slot = execute_primary(world, invalid_slot_command)
    assert invalid_slot == execute_replay(world, invalid_slot_command)
    assert invalid_slot.steps[0].outcome is StepOutcome.REJECTED
    assert (
        invalid_slot.steps[0].reject_reason
        is RejectReason.OPAQUE_LAW_SLOT
    )
    assert invalid_slot.steps[0].resolved_law_index == -1
    assert invalid_slot.disposition is TerminalDisposition.REJECT

    invalid_site = execute_primary(world, invalid_site_command)
    assert invalid_site == execute_replay(world, invalid_site_command)
    assert invalid_site.steps[0].outcome is StepOutcome.REJECTED
    assert invalid_site.steps[0].reject_reason is RejectReason.LOCAL_SITE
    assert invalid_site.disposition is TerminalDisposition.REJECT


@pytest.mark.parametrize("depth", range(1, 7))
def test_primary_and_replay_agree_at_every_supported_depth(depth: int) -> None:
    alternating = (
        _operation(0, 0, Direction.FORWARD),
        _operation(0, 0, Direction.REVERSE),
        _operation(1, 1, Direction.FORWARD),
        _operation(1, 1, Direction.REVERSE),
        _operation(0, 4, Direction.FORWARD),
        _operation(0, 4, Direction.REVERSE),
    )
    command = RewriteCommand(alternating[:depth])
    for theory_index in range(THEORY_COUNT):
        for index in (0, 1, 341, 1365, 2730, 4095):
            world = world_from_index(index, theory_index)
            primary = execute_primary(world, command)
            replay = execute_replay(world, command)
            assert primary == replay
            assert len(primary.steps) == depth
            assert len(primary.snapshots) == depth + 1
            assert primary.disposition is TerminalDisposition.ANSWER
            assert all(
                step.outcome in (StepOutcome.APPLIED, StepOutcome.BLOCKED)
                for step in primary.steps
            )


def test_rejection_halts_before_later_transactions() -> None:
    world = RewriteWorld(0, (0, 0, 0, 0, 0, 0))
    command = _command(
        _operation(0, 0),
        _operation(3, 0),
        _operation(0, 0, Direction.REVERSE),
    )
    execution = execute_primary(world, command)
    assert execution == execute_replay(world, command)
    assert len(execution.steps) == 2
    assert len(execution.snapshots) == 3
    assert execution.steps[0].outcome is StepOutcome.APPLIED
    assert execution.steps[1].outcome is StepOutcome.REJECTED
    assert execution.terminal == (1, 1, 0, 0, 0, 0)


def test_replay_does_not_call_primary_transition_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = RewriteWorld(0, (0, 0, 0, 0, 0, 0))
    command = _command(_operation(0, 0))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("primary transition helper was called")

    monkeypatch.setattr(rewrite, "_primary_step", forbidden)
    monkeypatch.setattr(rewrite, "execute_primary", forbidden)
    replay = execute_replay(world, command)
    assert replay.terminal == (1, 1, 0, 0, 0, 0)


def test_structural_query_grammar_is_exact_and_primary_replay_agree() -> None:
    queries = structural_queries()
    counts = Counter(query.op for query in queries)
    assert len(queries) == 156
    assert counts == Counter(
        {
            QueryOp.SLOT_IS: 24,
            QueryOp.TYPE_COUNT_GE: 24,
            QueryOp.ADJACENT_IS: 80,
            QueryOp.PATTERN_EXISTS: 16,
            QueryOp.SAME_TYPE_SLOTS_EQUAL: 6,
            QueryOp.SLOT_CHANGED: 6,
        }
    )
    assert len({query.canonical_bytes() for query in queries}) == len(queries)

    command = _command(_operation(0, 0))
    for world in iter_worlds(0):
        execution = execute_primary(world, command)
        assert execution == execute_replay(world, command)
        for query in queries:
            assert evaluate_query_primary(
                execution,
                query,
            ) is evaluate_query_replay(execution, query)


def test_every_state_only_query_has_both_boolean_denotations() -> None:
    queries = tuple(
        query
        for query in structural_queries()
        if query.op is not QueryOp.SLOT_CHANGED
    )
    labels = {query: set() for query in queries}
    for world in iter_worlds(0):
        blocked_operation = next(
            operation
            for operation in (
                _operation(0, 0, Direction.FORWARD),
                _operation(0, 0, Direction.REVERSE),
                _operation(1, 0, Direction.FORWARD),
                _operation(1, 0, Direction.REVERSE),
            )
            if world.registers[:2]
            != (
                LOCAL_LAWS[
                    THEORIES[world.theory_index].law_indices[
                        operation.law_slot
                    ]
                ].forward_source
                if operation.direction is Direction.FORWARD
                else LOCAL_LAWS[
                    THEORIES[world.theory_index].law_indices[
                        operation.law_slot
                    ]
                ].forward_target
            )
        )
        execution = execute_primary(
            world,
            _command(blocked_operation),
        )
        assert execution.steps[0].outcome is StepOutcome.BLOCKED
        assert execution.terminal == world.registers
        for query in queries:
            labels[query].add(evaluate_query_primary(execution, query))
    assert all(values == {False, True} for values in labels.values())


def test_slot_changed_queries_detect_locality() -> None:
    world = RewriteWorld(0, (0, 0, 2, 2, 3, 3))
    execution = execute_primary(world, _command(_operation(0, 0)))
    labels = tuple(
        evaluate_query_primary(
            execution,
            StructuralQuery(QueryOp.SLOT_CHANGED, (slot,)),
        )
        for slot in range(REGISTER_COUNT)
    )
    assert labels == (True, True, False, False, False, False)


def test_rejected_execution_has_no_boolean_query_answer() -> None:
    world = RewriteWorld(0, (0, 0, 0, 0, 0, 0))
    execution = execute_primary(world, _command(_operation(3, 0)))
    query = StructuralQuery(QueryOp.SLOT_IS, (0, 0))
    with pytest.raises(RewriteAdmissionError, match="no Boolean"):
        evaluate_query_primary(execution, query)
    with pytest.raises(RewriteAdmissionError, match="no Boolean"):
        evaluate_query_replay(execution, query)


def test_canonical_json_and_hash_helpers_are_strict_and_stable() -> None:
    value = {"b": 1, "a": [True, "x"]}
    assert canonical_json_bytes(value) == b'{"a":[true,"x"],"b":1}\n'
    assert canonical_sha256(value) == canonical_sha256(
        {"a": [True, "x"], "b": 1}
    )
    assert len(canonical_sha256(value)) == 64
    with pytest.raises(RewriteSemanticError, match="noncanonical"):
        canonical_json_bytes(("tuple",))
    with pytest.raises(RewriteSemanticError, match="key"):
        canonical_json_bytes({1: "not-string"})


def test_semantic_objects_have_deterministic_distinct_identities() -> None:
    first = RewriteWorld(0, (0, 0, 0, 0, 0, 0))
    same = RewriteWorld(0, (0, 0, 0, 0, 0, 0))
    other_state = RewriteWorld(0, (0, 0, 0, 0, 0, 1))
    other_theory = RewriteWorld(1, (0, 0, 0, 0, 0, 0))
    assert first.canonical_bytes() == same.canonical_bytes()
    assert first.sha256() == same.sha256()
    assert len(
        {
            first.sha256(),
            other_state.sha256(),
            other_theory.sha256(),
        }
    ) == 3

    command = _command(_operation(0, 0))
    primary = execute_primary(first, command)
    replay = execute_replay(first, command)
    assert primary.canonical_bytes() == replay.canonical_bytes()
    assert primary.sha256() == replay.sha256()


def test_strict_dataclasses_reject_coercions_and_noncanonical_queries() -> None:
    with pytest.raises(RewriteSemanticError, match="theory index"):
        RewriteWorld(True, (0, 0, 0, 0, 0, 0))
    with pytest.raises(RewriteSemanticError, match="width"):
        RewriteWorld(0, (0, 0, 0, 0, 0))  # type: ignore[arg-type]
    with pytest.raises(RewriteSemanticError, match="symbol"):
        RewriteWorld(0, (0, 0, 0, 0, 0, 4))
    with pytest.raises(RewriteSemanticError, match="law slot"):
        LocalOperation(True, 0, Direction.FORWARD)
    with pytest.raises(RewriteSemanticError, match="direction"):
        LocalOperation(0, 0, "forward")  # type: ignore[arg-type]
    with pytest.raises(RewriteSemanticError, match="depth"):
        RewriteCommand(())
    with pytest.raises(RewriteSemanticError, match="depth"):
        RewriteCommand((_operation(0, 0),) * 7)
    with pytest.raises(RewriteSemanticError, match="threshold"):
        StructuralQuery(QueryOp.TYPE_COUNT_GE, (0, 0, 0))
    with pytest.raises(RewriteSemanticError, match="same-type"):
        StructuralQuery(QueryOp.SAME_TYPE_SLOTS_EQUAL, (0, 1))
    with pytest.raises(RewriteSemanticError, match="canonical"):
        StructuralQuery(QueryOp.SAME_TYPE_SLOTS_EQUAL, (2, 0))


def test_exhaustive_primitive_primary_replay_audit() -> None:
    receipt = exhaustive_primitive_audit()
    assert receipt.world_count == 4096
    assert receipt.theory_count == 15
    assert receipt.theory_world_count == 61_440
    assert receipt.primitive_operation_count == 64
    assert receipt.valid_primitive_operation_count == 20
    assert receipt.case_count == 3_932_160
    assert receipt.applied_count == 76_800
    assert receipt.blocked_count == 1_152_000
    assert receipt.rejected_count == 2_703_360
    assert receipt.invalid_law_slot_rejected_count == 1_966_080
    assert receipt.invalid_site_rejected_count == 737_280
    assert receipt.primary_replay_mismatch_count == 0
    assert receipt.minimum_applied_support_per_valid_theory_operation == 256
    assert receipt.maximum_applied_support_per_valid_theory_operation == 256
    assert receipt.structural_query_count == 156
    assert receipt.max_transactions == 1
    assert (
        receipt.applied_count
        + receipt.blocked_count
        + receipt.rejected_count
        == receipt.case_count
    )
    assert len(receipt.sha256()) == 64
    assert exhaustive_primitive_audit() is receipt
