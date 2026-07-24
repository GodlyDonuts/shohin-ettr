from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest

from pipeline.episode_functor_law_collision_board import (
    COLLISION_RECEIPT_SCHEMA,
    DEFAULT_SEED,
    MINIMALITY_RECEIPT_SCHEMA,
    PAIR_SCHEMA,
    QUADRUPLE_RECEIPT_SCHEMA,
    QUADRUPLE_SCHEMA,
    SOURCE_SCHEMA,
    VERSION_SPACE_RECEIPT_SCHEMA,
    LawCollisionBoardError,
    LawCollisionPair,
    PathObservationClause,
    VisibleObservationClause,
    add_redundant_visible_clause,
    audit_collision_pair,
    audit_collision_quadruple,
    audit_minimality_in_fixture_family,
    audit_version_space,
    behavior_multiset,
    behavior_table,
    build_minimal_collision_pair,
    build_minimal_collision_quadruple,
    delete_clause,
    delete_law,
    deterministic_key_recode,
    encode_non_law,
    encode_source,
    low_order_collision_signature,
    parse_source,
    recode_source,
    shortest_distinguishing_query,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


@pytest.fixture(scope="module")
def pair() -> LawCollisionPair:
    return build_minimal_collision_pair()


def test_fixture_and_receipts_are_deterministic_and_canonical(
    pair: LawCollisionPair,
) -> None:
    rebuilt = build_minimal_collision_pair(DEFAULT_SEED)
    assert rebuilt == pair
    assert pair.schema == PAIR_SCHEMA
    assert parse_source(pair.left_source) == parse_source(rebuilt.left_source)
    assert encode_source(parse_source(pair.left_source)) == pair.left_source
    assert encode_source(parse_source(pair.right_source)) == pair.right_source

    first = audit_collision_pair(pair)
    second = audit_collision_pair(rebuilt)
    assert first.receipt == second.receipt
    assert first.receipt.schema == COLLISION_RECEIPT_SCHEMA
    assert first.left.receipt.schema == VERSION_SPACE_RECEIPT_SCHEMA
    assert first.receipt.canonical_bytes().endswith(b"\n")
    assert json.loads(first.receipt.canonical_bytes())["schema"] == (
        COLLISION_RECEIPT_SCHEMA
    )
    assert len(first.receipt.receipt_sha256) == 64


def test_minimal_fixture_is_exhaustively_minimal_in_declared_family() -> None:
    receipt = audit_minimality_in_fixture_family()
    assert receipt.schema == MINIMALITY_RECEIPT_SCHEMA
    assert receipt.searched_smaller_state_counts == (2, 3)
    assert receipt.tested_smaller_candidates > 0
    assert receipt.smaller_witness_count == 0
    assert receipt.witness_state_count == 4
    assert receipt.witness_first_distinguishing_depth == 3
    assert len(receipt.receipt_sha256) == 64


def test_twins_have_identical_non_law_bytes_and_opposite_laws(
    pair: LawCollisionPair,
) -> None:
    left = parse_source(pair.left_source)
    right = parse_source(pair.right_source)
    assert encode_non_law(left.evidence) == encode_non_law(right.evidence)
    assert left.evidence == right.evidence

    left_path = next(
        clause for clause in left.clauses if isinstance(clause, PathObservationClause)
    )
    right_path = next(
        clause for clause in right.clauses if isinstance(clause, PathObservationClause)
    )
    assert left_path.degree == right_path.degree == 3
    assert left_path.start == right_path.start
    assert left_path.actions == right_path.actions
    assert left_path.observer == right_path.observer
    assert left_path.expected == right_path.alternate
    assert left_path.alternate == right_path.expected
    assert len(pair.left_source) == len(pair.right_source)


def test_exhaustive_version_space_is_ambiguous_then_uniquely_resolved(
    pair: LawCollisionPair,
) -> None:
    left = audit_version_space(pair.left_source)
    right = audit_version_space(pair.right_source)
    assert left.receipt.direct_completion_count == 2
    assert right.receipt.direct_completion_count == 2
    assert left.receipt.direct_behavior_class_count == 2
    assert right.receipt.direct_behavior_class_count == 2
    assert left.receipt.direct_completion_sha256s == (
        right.receipt.direct_completion_sha256s
    )
    assert left.receipt.resolution == "unique-completion"
    assert right.receipt.resolution == "unique-completion"
    assert left.receipt.law_completion_count == 1
    assert right.receipt.law_completion_count == 1
    assert left.receipt.selected_completion_sha256 != (
        right.receipt.selected_completion_sha256
    )
    assert {
        left.receipt.selected_completion_sha256,
        right.receipt.selected_completion_sha256,
    } == set(left.receipt.direct_completion_sha256s)


def test_selected_completions_first_diverge_at_exactly_depth_three(
    pair: LawCollisionPair,
) -> None:
    left = audit_version_space(pair.left_source).selected_completion
    right = audit_version_space(pair.right_source).selected_completion
    assert left is not None
    assert right is not None
    witness = shortest_distinguishing_query(left, right)
    assert witness is not None
    assert witness.depth == 3
    assert behavior_table(left, maximum_depth=2) == behavior_table(
        right,
        maximum_depth=2,
    )
    assert behavior_table(left, maximum_depth=3) != behavior_table(
        right,
        maximum_depth=3,
    )
    for depth in range(3):
        assert behavior_multiset(left, maximum_depth=depth) == behavior_multiset(
            right,
            maximum_depth=depth,
        )


def test_all_feasible_low_order_collision_signatures_match(
    pair: LawCollisionPair,
) -> None:
    left = audit_version_space(pair.left_source)
    right = audit_version_space(pair.right_source)
    left_signature = low_order_collision_signature(left)
    right_signature = low_order_collision_signature(right)
    assert left_signature == right_signature
    assert left_signature["source_length"] == len(pair.left_source)
    assert left_signature["candidate_completion_cardinality"] == 2
    assert left_signature["cardinalities"] == {
        "actions": 1,
        "answers": 2,
        "observers": 1,
        "states": 4,
    }
    assert left_signature["key_occurrence_counts"]
    assert left_signature["row_and_unary_marginals"]
    assert left_signature["renderer_statistics"]
    assert left_signature["record_count"] == 13


@pytest.mark.parametrize("side", ("left", "right"))
def test_law_and_determining_clause_deletion_restore_ambiguity(
    pair: LawCollisionPair,
    side: str,
) -> None:
    payload = pair.left_source if side == "left" else pair.right_source
    for deleted in (
        delete_law(payload),
        delete_clause(payload, "determining-path3"),
    ):
        audit = audit_version_space(deleted)
        assert audit.receipt.resolution == "ambiguous"
        assert audit.receipt.law_completion_count == 2
        assert audit.receipt.law_behavior_class_count == 2
        assert audit.receipt.selected_completion_sha256 is None


@pytest.mark.parametrize("side", ("left", "right"))
def test_redundant_clause_deletion_and_insertion_change_nothing(
    pair: LawCollisionPair,
    side: str,
) -> None:
    payload = pair.left_source if side == "left" else pair.right_source
    baseline = audit_version_space(payload)
    deleted = audit_version_space(delete_clause(payload, "redundant-visible"))
    inserted = audit_version_space(add_redundant_visible_clause(payload))
    assert deleted.receipt.resolution == "unique-completion"
    assert inserted.receipt.resolution == "unique-completion"
    assert deleted.receipt.selected_completion_sha256 == (
        baseline.receipt.selected_completion_sha256
    )
    assert inserted.receipt.selected_completion_sha256 == (
        baseline.receipt.selected_completion_sha256
    )


def test_key_recode_is_an_exact_gauge_conjugacy(pair: LawCollisionPair) -> None:
    recoded_left, mapping = deterministic_key_recode(pair.left_source)
    recoded_right = recode_source(pair.right_source, mapping)
    original_left = audit_version_space(pair.left_source)
    original_right = audit_version_space(pair.right_source)
    gauge_left = audit_version_space(recoded_left)
    gauge_right = audit_version_space(recoded_right)
    for original, gauge in (
        (original_left, gauge_left),
        (original_right, gauge_right),
    ):
        assert original.receipt.source_sha256 != gauge.receipt.source_sha256
        assert original.receipt.semantic_source_sha256 == (
            gauge.receipt.semantic_source_sha256
        )
        assert original.receipt.direct_completion_sha256s == (
            gauge.receipt.direct_completion_sha256s
        )
        assert original.receipt.law_completion_sha256s == (
            gauge.receipt.law_completion_sha256s
        )
        assert original.receipt.resolution == gauge.receipt.resolution
    assert low_order_collision_signature(gauge_left) == (
        low_order_collision_signature(gauge_right)
    )


def test_complete_pair_audit_passes_every_gate(pair: LawCollisionPair) -> None:
    audit = audit_collision_pair(pair)
    assert audit.gates
    assert all(audit.gates.values())
    assert audit.receipt.first_distinguishing_query is not None
    assert audit.receipt.first_distinguishing_query.depth == 3
    assert audit.low_order_signature


def test_balanced_counterfactual_quadruple_passes_every_gate() -> None:
    quadruple = build_minimal_collision_quadruple()
    assert quadruple.schema == QUADRUPLE_SCHEMA
    audit = audit_collision_quadruple(quadruple)
    assert audit.receipt.schema == QUADRUPLE_RECEIPT_SCHEMA
    assert audit.gates
    assert all(audit.gates.values())
    assert audit.receipt.late_answer_indices in ((1, 0, 0, 1), (0, 1, 1, 0))
    assert audit.receipt.classification == "minimal_mechanics_fixture_only"
    assert "xor" in audit.receipt.known_shortcut
    assert not audit.receipt.promotion_eligible
    assert len(set(audit.receipt.selected_completion_sha256s[:2])) == 2
    assert len(set(audit.receipt.selected_completion_sha256s[2:])) == 2
    assert len(audit.receipt.receipt_sha256) == 64


def test_quadruple_reuses_laws_but_changes_facts() -> None:
    quadruple = build_minimal_collision_quadruple()
    f0_l0 = json.loads(quadruple.f0_l0_source)
    f0_l1 = json.loads(quadruple.f0_l1_source)
    f1_l0 = json.loads(quadruple.f1_l0_source)
    f1_l1 = json.loads(quadruple.f1_l1_source)
    assert f0_l0["law"] == f1_l0["law"]
    assert f0_l1["law"] == f1_l1["law"]
    assert f0_l0["machine"] != f1_l0["machine"]
    assert f0_l1["machine"] != f1_l1["machine"]
    assert f0_l0["machine"] == f0_l1["machine"]
    assert f1_l0["machine"] == f1_l1["machine"]


def test_quadruple_late_query_requires_both_facts_and_law() -> None:
    audit = audit_collision_quadruple(build_minimal_collision_quadruple())
    answers = audit.receipt.late_answer_indices
    assert answers[0] != answers[1]
    assert answers[2] != answers[3]
    assert answers[0] != answers[2]
    assert answers[1] != answers[3]
    assert answers[0] == answers[3]
    assert answers[1] == answers[2]


def test_quadruple_fact_or_law_collapse_fails_closed() -> None:
    quadruple = build_minimal_collision_quadruple()
    same_fact = replace(
        quadruple,
        f1_l0_source=quadruple.f0_l0_source,
        f1_l1_source=quadruple.f0_l1_source,
    )
    report = audit_collision_quadruple(same_fact, require_all_gates=False)
    assert not report.gates["counterfactual_facts_differ"]
    assert not report.gates["law_alone_is_uninformative"]
    with pytest.raises(LawCollisionBoardError, match="quadruple failed gates"):
        audit_collision_quadruple(same_fact)

    same_law = replace(
        quadruple,
        f0_l1_source=quadruple.f0_l0_source,
        f1_l1_source=quadruple.f1_l0_source,
    )
    report = audit_collision_quadruple(same_law, require_all_gates=False)
    assert not report.gates["facts_alone_are_uninformative"]
    assert not report.gates["both_fact_pairs_pass_collision_gates"]
    with pytest.raises(LawCollisionBoardError, match="quadruple failed gates"):
        audit_collision_quadruple(same_law)


def test_parser_rejects_duplicate_unknown_and_noncanonical_fields(
    pair: LawCollisionPair,
) -> None:
    duplicate = pair.left_source.replace(
        b'"machine":{',
        b'"machine":{},"machine":{',
        1,
    )
    with pytest.raises(LawCollisionBoardError, match="duplicate JSON field"):
        parse_source(duplicate)

    value = json.loads(pair.left_source)
    value["unexpected"] = 1
    with pytest.raises(LawCollisionBoardError, match="source fields differ"):
        parse_source(_canonical(value))

    pretty = json.dumps(
        json.loads(pair.left_source),
        indent=2,
        sort_keys=True,
    ).encode("ascii")
    with pytest.raises(LawCollisionBoardError, match="not the canonical"):
        parse_source(pretty)


def test_direct_fact_and_clause_mutations_fail_closed(
    pair: LawCollisionPair,
) -> None:
    missing_observation = json.loads(pair.left_source)
    missing_observation["machine"]["observations"].pop()
    with pytest.raises(LawCollisionBoardError, match="complete observer table"):
        parse_source(_canonical(missing_observation))

    duplicate_destination = json.loads(pair.left_source)
    duplicate_destination["machine"]["transitions"][1]["destination"] = (
        duplicate_destination["machine"]["transitions"][0]["destination"]
    )
    with pytest.raises(LawCollisionBoardError, match="permutation row"):
        parse_source(_canonical(duplicate_destination))

    invalid_answer = json.loads(pair.left_source)
    invalid_answer["law"]["clauses"][0]["expected"] = "k_0000000000000000"
    with pytest.raises(LawCollisionBoardError, match="unknown opaque key"):
        parse_source(_canonical(invalid_answer))

    contradictory_redundant = json.loads(pair.left_source)
    clauses = contradictory_redundant["law"]["clauses"]
    clauses[1]["answer"] = clauses[0]["expected"]
    with pytest.raises(LawCollisionBoardError, match="not redundant"):
        parse_source(_canonical(contradictory_redundant))


def test_degree_two_law_mutation_cannot_pass_collision_audit(
    pair: LawCollisionPair,
) -> None:
    shortened = json.loads(pair.left_source)
    shortened["law"]["clauses"][0]["actions"].pop()
    mutated_pair = replace(pair, left_source=_canonical(shortened))
    report = audit_collision_pair(mutated_pair, require_all_gates=False)
    assert not report.gates["degree_three_first_behavior_collision"]
    assert not report.gates["laws_select_exactly_one"]
    with pytest.raises(LawCollisionBoardError, match="failed gates"):
        audit_collision_pair(mutated_pair)


def test_identical_law_negative_control_is_rejected(
    pair: LawCollisionPair,
) -> None:
    invalid = replace(pair, right_source=pair.left_source)
    report = audit_collision_pair(invalid, require_all_gates=False)
    assert not report.gates["law_twins_opposite"]
    assert not report.gates["opposite_selected_completions"]
    with pytest.raises(LawCollisionBoardError, match="failed gates"):
        audit_collision_pair(invalid)


def test_recode_rejects_partial_nonbijective_and_nonopaque_maps(
    pair: LawCollisionPair,
) -> None:
    source = parse_source(pair.left_source)
    keys = (
        *source.evidence.states,
        *source.evidence.actions,
        *source.evidence.observers,
        *source.evidence.answers,
    )
    partial = {key: key for key in keys[:-1]}
    with pytest.raises(LawCollisionBoardError, match="domain"):
        recode_source(pair.left_source, partial)

    duplicate = {key: "k_0000000000000000" for key in keys}
    with pytest.raises(LawCollisionBoardError, match="unique"):
        recode_source(pair.left_source, duplicate)

    malformed = {key: f"bad-{index}" for index, key in enumerate(keys)}
    with pytest.raises(LawCollisionBoardError, match="must match"):
        recode_source(pair.left_source, malformed)


def test_clause_mutators_reject_unknown_or_duplicate_ids(
    pair: LawCollisionPair,
) -> None:
    with pytest.raises(LawCollisionBoardError, match="exactly one"):
        delete_clause(pair.left_source, "does-not-exist")
    with pytest.raises(LawCollisionBoardError, match="already exists"):
        add_redundant_visible_clause(
            pair.left_source,
            clause_id="redundant-visible",
        )


def test_module_has_no_neural_or_external_solver_dependency() -> None:
    module_path = Path(__file__).with_name("episode_functor_law_collision_board.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported_roots.isdisjoint(
        {"numpy", "sage", "scipy", "sympy", "torch", "transformers"}
    )


def test_source_schema_is_explicit_and_redundant_clause_is_typed(
    pair: LawCollisionPair,
) -> None:
    value = json.loads(pair.left_source)
    assert value["schema"] == SOURCE_SCHEMA
    source = parse_source(pair.left_source)
    assert any(
        isinstance(clause, VisibleObservationClause) for clause in source.clauses
    )
