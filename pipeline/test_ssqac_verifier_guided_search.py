from __future__ import annotations

from dataclasses import replace

import pytest

from episode_functor_algebra_machine import (
    OP_HALT,
    AlgebraInstruction,
    execute_program,
)
from pipeline import ssqac_verifier_guided_search as search_module
from pipeline.ssqac_verifier_guided_search import (
    BENCHMARK_SCHEMA,
    CANDIDATE_PACKET_SCHEMA,
    GUIDANCE_RANDOM_RELABEL,
    GUIDANCE_STRUCTURAL,
    STATUS,
    CandidateSearchResult,
    SearchBudget,
    SearchFalsifierError,
    SealedAlgebraPacket,
    WeakLocalScorer,
    _program_sha256,
    assess_candidate_program,
    bounded_beam_candidate_search,
    enumerate_legal_repair_actions,
    generate_geometry_cases,
    greedy_candidate_search,
    load_candidate_packet,
    run_falsifier_benchmark,
    structural_potential,
)


SMALL_SEARCH_BUDGET = SearchBudget(
    max_nodes_expanded=512,
    max_edges_considered=8_000,
    max_depth=24,
    max_frontier=64,
    beam_width=64,
    max_program_instructions=96,
)


def _packet() -> SealedAlgebraPacket:
    return SealedAlgebraPacket.from_rows(
        (
            (0, 2, 1, 0),
            (3, 1, 0, 4),
            (0, 0, 1, 5),
        )
    )


def test_candidate_packet_round_trips_exact_keys() -> None:
    packet = _packet()
    loaded = load_candidate_packet(packet.canonical_data())
    assert loaded == packet
    assert loaded.schema == CANDIDATE_PACKET_SCHEMA
    assert set(loaded.canonical_data()) == {
        "field_modulus",
        "register_count",
        "rows",
        "schema",
        "seal_sha256",
    }


def test_candidate_packet_rejects_tamper_and_metadata_channels() -> None:
    packet = _packet()
    with pytest.raises(SearchFalsifierError, match="seal differs"):
        replace(packet, rows=((1, 0), (0, 1))).validate()
    for forbidden in ("source", "query", "workspace", "oracle_answer"):
        value = packet.canonical_data()
        value[forbidden] = "not allowed"
        with pytest.raises(
            SearchFalsifierError,
            match="forbidden fields",
        ):
            load_candidate_packet(value)


def test_candidate_packet_rejects_extra_and_noncanonical_fields() -> None:
    value = _packet().canonical_data()
    value["comment"] = "side channel"
    with pytest.raises(SearchFalsifierError, match="keys differ"):
        load_candidate_packet(value)
    value = _packet().canonical_data()
    value["rows"][0][0] = True
    with pytest.raises(SearchFalsifierError, match="plain integer"):
        load_candidate_packet(value)


def test_structural_potential_is_zero_exactly_on_sample_rref() -> None:
    solved = ((1, 0, 2, 3), (0, 1, 4, 5), (0, 0, 0, 0))
    reverse = ((0, 1, 0), (1, 0, 0), (0, 0, 0))
    nonunit = ((2, 0, 0), (0, 1, 0), (0, 0, 0))
    extra = ((1, 5, 0), (0, 1, 0), (0, 0, 0))
    assert structural_potential(solved).solved
    assert not structural_potential(reverse).solved
    assert structural_potential(reverse).pivot_order_pairs == 1
    assert structural_potential(nonunit).nonunit_pivots == 1
    assert structural_potential(extra).pivot_column_extras == 1


def test_every_enumerated_repair_replays_as_legal_primitives() -> None:
    packet = _packet()
    state = execute_program(
        packet.rows,
        (),
        register_count=packet.register_count,
    )
    actions = enumerate_legal_repair_actions(state)
    assert {action.kind for action in actions} == {
        "ELIMINATE",
        "NORMALIZE",
        "SWAP",
    }
    for action in actions:
        replay = execute_program(
            packet.rows,
            action.instructions,
            register_count=packet.register_count,
        )
        assert replay.executed_instructions == len(action.instructions)
        assert not replay.halted


def test_candidate_search_never_invokes_the_assessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _packet()

    def forbidden_verifier(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("candidate called the assessor")

    with monkeypatch.context() as context:
        context.setattr(
            search_module,
            "verify_reduction_program",
            forbidden_verifier,
        )
        result = bounded_beam_candidate_search(
            packet,
            WeakLocalScorer(seed=7),
            SMALL_SEARCH_BUDGET,
        )
    assert result.receipt.completed
    assert result.program is not None
    assert assess_candidate_program(packet, result).passed


def test_separate_assessor_certifies_only_the_final_program() -> None:
    packet = _packet()
    result = bounded_beam_candidate_search(
        packet,
        WeakLocalScorer(seed=11),
        SMALL_SEARCH_BUDGET,
    )
    assessment = assess_candidate_program(packet, result)
    assert assessment.passed
    assert assessment.reason == "independent_verifier_accepted"
    assert assessment.verifier_gates
    assert result.program is not None
    assert result.program[-1].opcode == OP_HALT


def test_separate_assessor_rejects_premature_halt() -> None:
    packet = _packet()
    good = bounded_beam_candidate_search(
        packet,
        WeakLocalScorer(seed=13),
        SMALL_SEARCH_BUDGET,
    )
    bad_program = (AlgebraInstruction(OP_HALT),)
    bad_receipt = replace(
        good.receipt,
        completed=True,
        termination="candidate_goal",
        final_program_instructions=1,
        program_sha256=_program_sha256(bad_program),
    )
    bad = CandidateSearchResult(
        program=bad_program,
        receipt=bad_receipt,
    )
    assessment = assess_candidate_program(packet, bad)
    assert not assessment.passed
    assert assessment.reason.startswith("independent_verifier_rejected:")


def test_assessor_rejects_cross_packet_receipt_reuse() -> None:
    packet = _packet()
    result = bounded_beam_candidate_search(
        packet,
        WeakLocalScorer(seed=17),
        SMALL_SEARCH_BUDGET,
    )
    other = SealedAlgebraPacket.from_rows(((1, 0), (0, 1)))
    with pytest.raises(SearchFalsifierError, match="different packet"):
        assess_candidate_program(other, result)


def test_structural_search_is_byte_deterministic() -> None:
    packet = _packet()
    policy = WeakLocalScorer(seed=19)
    left = bounded_beam_candidate_search(
        packet,
        policy,
        SMALL_SEARCH_BUDGET,
    )
    right = bounded_beam_candidate_search(
        packet,
        policy,
        SMALL_SEARCH_BUDGET,
    )
    assert left.program == right.program
    assert left.receipt.canonical_bytes() == right.receipt.canonical_bytes()


def test_random_relabel_control_is_matched_but_distinct() -> None:
    packet = _packet()
    policy = WeakLocalScorer(seed=23)
    control_policy = WeakLocalScorer(
        seed=23,
        relabel_action_kinds=True,
    )
    treatment = bounded_beam_candidate_search(
        packet,
        policy,
        SMALL_SEARCH_BUDGET,
        guidance=GUIDANCE_STRUCTURAL,
        guidance_seed=23,
    )
    control = bounded_beam_candidate_search(
        packet,
        control_policy,
        SMALL_SEARCH_BUDGET,
        guidance=GUIDANCE_RANDOM_RELABEL,
        guidance_seed=23,
    )
    assert treatment.receipt.guidance == GUIDANCE_STRUCTURAL
    assert control.receipt.guidance == GUIDANCE_RANDOM_RELABEL
    assert treatment.receipt.max_nodes_expanded == control.receipt.max_nodes_expanded
    assert (
        treatment.receipt.max_edges_considered == control.receipt.max_edges_considered
    )
    assert treatment.receipt.policy_sha256 != control.receipt.policy_sha256
    assert treatment.receipt.search_trace_sha256 != control.receipt.search_trace_sha256


def test_greedy_and_search_receipts_obey_hard_budgets() -> None:
    packet = _packet()
    budget = SearchBudget(
        max_nodes_expanded=1,
        max_edges_considered=2,
        max_depth=1,
        max_frontier=1,
        beam_width=1,
        max_program_instructions=4,
    )
    for result in (
        greedy_candidate_search(packet, WeakLocalScorer(seed=29), budget),
        bounded_beam_candidate_search(
            packet,
            WeakLocalScorer(seed=29),
            budget,
        ),
    ):
        receipt = result.receipt
        assert receipt.hard_budgets_respected
        assert receipt.nodes_expanded <= 1
        assert receipt.edges_considered <= 2
        assert receipt.maximum_depth_reached <= 1
        assert receipt.peak_frontier <= 1
        assert receipt.final_program_instructions <= 4
        assert receipt.nodes_generated == receipt.edges_legal


def test_instruction_budget_fails_closed() -> None:
    packet = _packet()
    budget = SearchBudget(
        max_nodes_expanded=32,
        max_edges_considered=256,
        max_depth=8,
        max_frontier=8,
        beam_width=8,
        max_program_instructions=1,
    )
    result = bounded_beam_candidate_search(
        packet,
        WeakLocalScorer(seed=31),
        budget,
    )
    assert not result.receipt.completed
    assert result.program is None
    assert result.receipt.hard_budgets_respected
    assert not assess_candidate_program(packet, result).passed


def test_solved_packet_needs_only_explicit_halt() -> None:
    packet = SealedAlgebraPacket.from_rows(((1, 0, 3), (0, 1, 4), (0, 0, 0)))
    budget = SearchBudget(
        max_nodes_expanded=1,
        max_edges_considered=1,
        max_depth=1,
        max_frontier=1,
        beam_width=1,
        max_program_instructions=1,
    )
    result = bounded_beam_candidate_search(
        packet,
        WeakLocalScorer(seed=37),
        budget,
    )
    assert result.program == (AlgebraInstruction(OP_HALT),)
    assert result.receipt.nodes_expanded == 0
    assert result.receipt.edges_considered == 0
    assert assess_candidate_program(packet, result).passed


def test_generated_geometry_cases_are_deterministic_and_matrix_only() -> None:
    left = generate_geometry_cases(
        seed=41,
        count=6,
        minimum_rows=4,
        maximum_rows=4,
        minimum_columns=5,
        maximum_columns=6,
    )
    right = generate_geometry_cases(
        seed=41,
        count=6,
        minimum_rows=4,
        maximum_rows=4,
        minimum_columns=5,
        maximum_columns=6,
    )
    assert left == right
    assert len({case.matrix_sha256 for case in left}) == 6
    assert all(len(case.packet.rows) == 4 for case in left)
    assert all(5 <= len(case.packet.rows[0]) <= 6 for case in left)


def test_benchmark_rejects_non_strict_geometry() -> None:
    with pytest.raises(SearchFalsifierError, match="strictly larger"):
        run_falsifier_benchmark(
            seed=43,
            development_count=1,
            evaluation_count=1,
            development_maximum_rows=4,
            development_maximum_columns=5,
            evaluation_minimum_rows=4,
            evaluation_minimum_columns=5,
            evaluation_maximum_rows=4,
            evaluation_maximum_columns=5,
            greedy_budget=SMALL_SEARCH_BUDGET,
            search_budget=SMALL_SEARCH_BUDGET,
        )


def test_strict_larger_geometry_falsifier_reports_all_controls() -> None:
    greedy_budget = SearchBudget(
        max_nodes_expanded=48,
        max_edges_considered=2_000,
        max_depth=36,
        max_frontier=128,
        beam_width=1,
        max_program_instructions=160,
    )
    search_budget = SearchBudget(
        max_nodes_expanded=2_048,
        max_edges_considered=50_000,
        max_depth=36,
        max_frontier=96,
        beam_width=96,
        max_program_instructions=160,
    )
    report = run_falsifier_benchmark(
        seed=20260724,
        development_count=2,
        evaluation_count=4,
        development_maximum_rows=3,
        development_maximum_columns=4,
        evaluation_minimum_rows=4,
        evaluation_minimum_columns=5,
        evaluation_maximum_rows=4,
        evaluation_maximum_columns=6,
        greedy_budget=greedy_budget,
        search_budget=search_budget,
    )
    assert report.schema == BENCHMARK_SCHEMA
    assert report.status == STATUS
    assert not report.reasoning_claim_authorized
    assert report.candidate_oracle_calls == 0
    assert report.strict_geometry_holdout
    assert report.hard_resource_gate
    assert report.search_certified == report.evaluation_cases
    assert report.search_certified > report.greedy_certified
    assert report.search_certified > report.random_relabel_certified
    assert report.mechanics_promotion_gate
    assert report.candidate_packet_fields == (
        "field_modulus",
        "register_count",
        "rows",
        "schema",
        "seal_sha256",
    )
    assert "source" in report.forbidden_candidate_fields
    assert "query" in report.forbidden_candidate_fields
    assert all(case.search_passed for case in report.cases)


def test_even_a_passing_mechanics_gate_never_authorizes_reasoning() -> None:
    assert STATUS.endswith("not_reasoning")
    assert "reasoning" not in CANDIDATE_PACKET_SCHEMA
