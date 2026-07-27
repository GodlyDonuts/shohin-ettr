from __future__ import annotations

from copy import deepcopy
import re

import pytest
import torch

from assess_er_dual_stream_opcode_canary import (
    DECISIVE_FIELDS,
    independent_gates,
    score_predictions,
    verify_path_evidence,
    verify_query_logit_evidence,
)
from build_er_dual_stream_fresh_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
    build_board,
)
from er_relation_tensor_training import loss_batch, parse_row
from pilot_er_dual_stream_opcode_canary import (
    compute_gates,
    evaluate_coherent_routes,
    identity_deranged_row,
    query_only_alpha_row,
    query_target_counterfactual_row,
    relocation_consistency,
    renderer_relocation_rows,
)
from pilot_er_dual_stream_train_canary import score_train_row
from pilot_er_dual_stream_relation_adapter import EXPECTED_PARAMETERS
from er_dual_stream_relation_adapter import DualStreamRelationCompiler


def test_relocation_rerenders_only_training_semantics() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 3, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    raw = splits[TRAIN_SPLIT]
    families = {str(row["family_id"]) for row in raw[:8]}
    relocated = renderer_relocation_rows(raw, families, seed=701)
    assert len(relocated) == 4 * len(families)
    assert {row.split for row in relocated} == {TRAIN_SPLIT}
    assert {row.family_id for row in relocated} == families
    assert all(row.final_state is None and row.answer_role is None for row in relocated)
    assert {row.renderer for row in relocated} == {
        "er-ds-d0w0e0q0-v2",
        "er-ds-d0w0e0q1-v2",
        "er-ds-d0w1e0q0-v2",
        "er-ds-d0w1e0q1-v2",
    }
    by_family = {}
    for row in relocated:
        by_family.setdefault(row.family_id, []).append(row)
    for rows in by_family.values():
        assert len({row.program_bytes for row in rows if "w0" in row.renderer}) == 1
        assert len({row.program_bytes for row in rows if "w1" in row.renderer}) == 1


def test_renderer_consistency_requires_all_eight_views() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 2, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    raw = splits[TRAIN_SPLIT]
    canonical = [parse_row(row, TRAIN_SPLIT) for row in raw]
    families = {row.family_id for row in canonical}
    relocated = renderer_relocation_rows(raw, families, seed=702)
    rows = sorted(canonical + relocated, key=lambda row: (row.family_id, row.row_id))
    predictions = {}
    for key, shape in (
        ("cardinality", (len(rows),)),
        ("initial", (len(rows), 6)),
        ("relations", (len(rows), 4, 6)),
        ("rule_active", (len(rows), 4)),
        ("events", (len(rows), 13)),
        ("halt", (len(rows), 13)),
        ("query", (len(rows),)),
    ):
        predictions[key] = torch.zeros(shape, dtype=torch.int16)
    result = relocation_consistency(rows, predictions)
    assert result == {"exact": 2, "families": 2, "rate": 1.0}
    predictions["query"][0] = 1
    assert relocation_consistency(rows, predictions)["exact"] == 1


def test_query_only_recode_preserves_program_and_target_span() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 1, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    row = parse_row(splits[TRAIN_SPLIT][0], TRAIN_SPLIT)
    recoded = query_only_alpha_row(row, "unit-query-only")
    assert recoded.program_bytes == row.program_bytes
    assert recoded.query_range == row.query_range
    assert recoded.query_bytes != row.query_bytes


def test_query_target_counterfactual_changes_only_target_and_answer() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 1, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    row = score_train_row(parse_row(splits[TRAIN_SPLIT][0], TRAIN_SPLIT))
    changed = query_target_counterfactual_row(row, 1)
    assert changed.program_bytes == row.program_bytes
    assert changed.query_range == row.query_range
    assert changed.query_position != row.query_position
    assert changed.answer_role == changed.final_state[changed.query_position]
    assert changed.query_bytes != row.query_bytes


def test_identity_derangement_preserves_shape_and_unique_candidates() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 1, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    row = parse_row(splits[TRAIN_SPLIT][0], TRAIN_SPLIT)
    changed = identity_deranged_row(row, "unit-identity")
    assert len(changed.program_bytes) == len(row.program_bytes)
    assert changed.line_ranges == row.line_ranges
    tokens = re.findall(
        rb"(?<!\S)z[0-9a-z]{5}(?!\S)", bytes(changed.program_bytes)
    )
    assert len(tokens) == len(set(tokens))


def test_coherent_decoder_emits_complete_train_only_evidence() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 1, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    rows = [
        score_train_row(parse_row(row, TRAIN_SPLIT))
        for row in splits[TRAIN_SPLIT]
    ]
    model = DualStreamRelationCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=1024,
        fingerprint_width=16,
        orbit_width=32,
        orbit_heads=4,
        orbit_layers=1,
        orbit_ff=64,
        native_slot_layers=1,
        native_slot_heads=4,
        native_slot_ff=64,
        record_width=32,
        record_heads=4,
        record_layers=1,
        record_set_layers=1,
        record_ff=64,
        max_line_bytes=96,
        sinkhorn_steps=4,
        occurrence_ff=64,
        equality_width=16,
    ).eval()
    metrics, evidence = evaluate_coherent_routes(
        model, rows, opcode_weight=1.0, batch_size=4
    )
    assert metrics["overall"]["joint"]["rows"] == 4
    assert evidence["path_scores"].shape == (4, 4, 4, 13)
    assert evidence["candidate_positions"].shape == (4, 4, 13)
    assert evidence["target_exclusion"].shape == (4, 4)
    assert evidence["pred_relations"].shape == (4, 4, 6)
    query_check = verify_query_logit_evidence({"branch": evidence})
    assert query_check["all_query_semantic_argmax_exact"] is True
    assert query_check["all_query_pointer_argmax_exact"] is True
    for row_index, row in enumerate(rows):
        cardinality = row.cardinality
        cardinality_index = cardinality - 3
        slots = tuple(range(cardinality)) + tuple(range(6, 6 + cardinality))
        for rule in range(row.rule_count):
            width = 2 * cardinality + 1
            recomputed_scores = []
            for excluded in range(width):
                score = evidence["candidate_opcode_logits"][
                    row_index, rule, excluded
                ]
                for ordinal, slot in enumerate(slots):
                    rank = ordinal + int(ordinal >= excluded)
                    score = score + evidence["candidate_witness_logits"][
                        row_index, rule, slot, rank
                    ]
                recomputed_scores.append(score)
            assert torch.allclose(
                evidence["path_scores"][
                    row_index, rule, cardinality_index, :width
                ],
                torch.stack(recomputed_scores),
            )
            recomputed_marginal = torch.zeros((12, 13))
            candidate_count = int(
                evidence["candidate_positions"][row_index, rule].ge(0).sum()
            )
            for route_cardinality in range(3, 7):
                route_index = route_cardinality - 3
                route_slots = tuple(range(route_cardinality)) + tuple(
                    range(6, 6 + route_cardinality)
                )
                cardinality_probability = evidence["cardinality_probability"][
                    row_index, route_index
                ]
                if candidate_count == 2 * route_cardinality:
                    for ordinal, slot in enumerate(route_slots):
                        recomputed_marginal[slot, ordinal] += cardinality_probability
                elif candidate_count == 2 * route_cardinality + 1:
                    path_probability = evidence["path_probability"][
                        row_index, rule, route_index, :candidate_count
                    ]
                    for excluded in range(candidate_count):
                        for ordinal, slot in enumerate(route_slots):
                            rank = ordinal + int(ordinal >= excluded)
                            recomputed_marginal[slot, rank] += (
                                cardinality_probability
                                * path_probability[excluded]
                            )
            assert torch.allclose(
                evidence["candidate_marginal_probability"][
                    row_index, rule
                ],
                recomputed_marginal,
            )
    rebuilt = score_predictions(
        rows,
        evidence,
        {
            "candidate_positions": evidence["candidate_positions"],
            "target_exclusion": evidence["target_exclusion"],
        },
    )
    for field in DECISIVE_FIELDS:
        assert rebuilt["overall"][field] == metrics["overall"][field]
    model.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        evaluate_coherent_routes(model, rows, opcode_weight=1.0, batch_size=4)


def test_structured_route_objective_is_finite_and_reaches_both_queries() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 1, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    rows = [parse_row(row, TRAIN_SPLIT) for row in splits[TRAIN_SPLIT]]
    model = DualStreamRelationCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=1024,
        fingerprint_width=16,
        orbit_width=32,
        orbit_heads=4,
        orbit_layers=1,
        orbit_ff=64,
        native_slot_layers=1,
        native_slot_heads=4,
        native_slot_ff=64,
        record_width=32,
        record_heads=4,
        record_layers=1,
        record_set_layers=1,
        record_ff=64,
        max_line_bytes=96,
        sinkhorn_steps=4,
        occurrence_ff=64,
        equality_width=16,
    ).train()
    model.structured_route_objective = True
    loss, pieces = loss_batch(model, [rows], torch.device("cpu"))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in pieces.values())
    loss.backward()
    assert model.er_ds_witness_queries.grad is not None
    assert model.er_ds_rule_opcode_query.grad is not None


def test_independent_assessor_verifies_coherent_candidate_partition() -> None:
    rows = 8_000
    predicted_cardinality = torch.zeros(rows, dtype=torch.int16)
    target_cardinality = torch.zeros(rows, dtype=torch.int16)
    target_rule_count = torch.zeros(rows, dtype=torch.int16)
    target_exclusion = torch.full((rows, 4), -1, dtype=torch.int16)
    candidates = torch.full((rows, 4, 13), -1, dtype=torch.int32)
    predicted_cardinality[0] = target_cardinality[0] = 3
    target_rule_count[0] = 1
    target_exclusion[0, 0] = 3
    candidates[0, 0, :7] = torch.arange(7)

    def branch(weight: float, selected: int) -> dict[str, torch.Tensor]:
        witness_logits = torch.zeros((rows, 4, 12, 13), dtype=torch.float16)
        opcode_logits = torch.zeros((rows, 4, 13), dtype=torch.float16)
        opcode_logits[0, 0, selected] = 5
        scores = torch.zeros((rows, 4, 4, 13), dtype=torch.float16)
        scores[0, 0, 0, :7] = weight * opcode_logits[0, 0, :7]
        probability = torch.zeros_like(scores)
        probability[0, 0, 0, :7] = scores[0, 0, 0, :7].softmax(-1)
        map_exclusion = torch.full((rows, 4), -1, dtype=torch.int16)
        map_exclusion[0, 0] = selected
        witness = torch.full((rows, 4, 12), -1, dtype=torch.int32)
        expected = torch.cat(
            (torch.arange(selected), torch.arange(selected + 1, 7))
        ).to(torch.int32)
        slots = [0, 1, 2, 6, 7, 8]
        witness[0, 0, slots] = expected
        opcode = torch.full((rows, 4), -1, dtype=torch.int32)
        opcode[0, 0] = selected
        marginal = torch.zeros_like(witness_logits)
        for excluded in range(7):
            for ordinal, slot in enumerate(slots):
                rank = ordinal + int(ordinal >= excluded)
                marginal[0, 0, slot, rank] += probability[0, 0, 0, excluded]
        cardinality_probability = torch.zeros((rows, 4), dtype=torch.float16)
        cardinality_probability[0, 0] = 1
        return {
            "path_scores": scores,
            "path_probability": probability,
            "candidate_witness_logits": witness_logits,
            "candidate_opcode_logits": opcode_logits,
            "candidate_marginal_probability": marginal,
            "cardinality_probability": cardinality_probability,
            "pred_cardinality": predicted_cardinality,
            "target_cardinality": target_cardinality,
            "target_rule_count": target_rule_count,
            "map_exclusion": map_exclusion,
            "target_exclusion": target_exclusion,
            "candidate_positions": candidates,
            "pred_witness_pointer": witness,
            "pred_rule_opcode_pointer": opcode,
            "rule_opcode_pointer": opcode,
        }

    def independent(branch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = deepcopy(branch)
        slots = [0, 1, 2, 6, 7, 8]
        ranks = result["candidate_marginal_probability"][
            0, 0, slots, :7
        ].argmax(-1)
        positions = candidates[0, 0, ranks]
        witness = torch.full((rows, 4, 12), -1, dtype=torch.int32)
        opcode = torch.full((rows, 4), -1, dtype=torch.int32)
        if len(set(map(int, positions.tolist()))) == 6:
            witness[0, 0, slots] = positions
            opcode[0, 0] = next(
                iter(set(range(7)) - set(map(int, positions.tolist())))
            )
        result["pred_witness_pointer"] = witness
        result["pred_rule_opcode_pointer"] = opcode
        return result

    s0 = branch(0.0, 0)
    s1 = branch(1.0, 3)
    s0_independent = independent(s0)
    s1_independent = independent(s1)
    s1_rotated = branch(1.0, 4)
    s1_rotated["candidate_opcode_logits"] = s1[
        "candidate_opcode_logits"
    ].roll(1, dims=-1)
    evidence = {
        "arms": {
            "unit": {
                "modes": {
                    "s0_qraw": {
                        "relocated": {
                            "coherent": s0,
                            "independent": s0_independent,
                        },
                        "opcode_shuffled": None,
                        "witness_shuffled": s0,
                    },
                    "s1_qraw": {
                        "relocated": {
                            "coherent": s1,
                            "independent": s1_independent,
                        },
                        "opcode_shuffled": s1_rotated,
                        "witness_shuffled": s1,
                    },
                    "s0_qstruct": {
                        "relocated": {
                            "coherent": s0,
                            "independent": s0_independent,
                        },
                        "opcode_shuffled": None,
                        "witness_shuffled": s0,
                    },
                    "s1_qstruct": {
                        "relocated": {
                            "coherent": s1,
                            "independent": s1_independent,
                        },
                        "opcode_shuffled": s1_rotated,
                        "witness_shuffled": s1,
                    },
                }
            }
        }
    }
    result = verify_path_evidence(evidence)
    assert result["active_routes_checked"] == 4
    assert result["all_route_coverage_exact"] is True
    assert result["all_map_argmax_exact"] is True
    assert result["all_ordered_complements_exact"] is True
    assert result["all_probabilities_normalized"] is True
    assert result["all_probabilities_match_softmax"] is True
    assert result["all_path_scores_recompute"] is True
    assert result["all_marginal_probabilities_recompute"] is True
    assert result["all_independent_decodes_recompute"] is True
    assert result["all_opcode_rotations_recompute"] is True
    assert result["all_witness_rotations_recompute"] is True
    corrupted_marginal = deepcopy(evidence)
    corrupted_marginal["arms"]["unit"]["modes"]["s1_qraw"]["relocated"][
        "coherent"
    ]["candidate_marginal_probability"][0, 0, 0, 0] += 0.5
    assert (
        verify_path_evidence(corrupted_marginal)[
            "all_marginal_probabilities_recompute"
        ]
        is False
    )
    s1["pred_witness_pointer"][0, 0, 0], s1["pred_witness_pointer"][0, 0, 1] = (
        s1["pred_witness_pointer"][0, 0, 1].clone(),
        s1["pred_witness_pointer"][0, 0, 0].clone(),
    )
    assert verify_path_evidence(evidence)["all_ordered_complements_exact"] is False


def _metric(rate: float) -> dict[str, object]:
    fields = {
        name: {"correct": int(8_000 * rate), "rows": 8_000, "rate": rate}
        for name in (
            "packet",
            "state",
            "answer",
            "joint",
            "route_joint",
            "relation_rows",
            "witness_pointer",
            "rule_opcode_pointer",
            "event_opcode_pointer",
            "query",
            "query_pointer",
        )
    }
    grouped = {
        "x": {
            name: fields[name]
            for name in (
                "route_joint",
                "witness_pointer",
                "rule_opcode_pointer",
                "event_opcode_pointer",
                "query",
                "query_pointer",
            )
        }
    }
    return {
        "overall": fields,
        "by_cardinality": grouped,
        "by_renderer": grouped,
        "by_renderer_cardinality": grouped,
    }


def test_gate_requires_causal_advantage_over_legacy() -> None:
    exact = {"exact": 8_000, "rows": 8_000, "rate": 1.0}
    fit = {"frozen_parent_unchanged": True, "updates": 2_500}

    def mode(
        coherent: float,
        independent: float,
        opcode_shuffled: float | None = None,
        witness_shuffled: float = 0.7,
    ):
        return {
            "canonical": {
                "coherent": _metric(coherent),
                "independent": _metric(independent),
            },
            "relocated": {
                "coherent": _metric(coherent),
                "independent": _metric(independent),
            },
            "relocation_consistency": {"exact": 2_000, "families": 2_000},
            "alpha": {"route": exact, "query": exact},
            "distractor": {"route": exact, "query": exact},
            "identity_deranged": _metric(0.0),
            "opcode_shuffled": None
            if opcode_shuffled is None
            else _metric(opcode_shuffled),
            "witness_shuffled": _metric(witness_shuffled),
        }

    query = {
        name: {
            "recode_a": {"query": exact},
            "recode_b": {"query": exact},
            "target_a": _metric(1.0),
            "target_b": _metric(1.0),
        }
        for name in ("qraw", "qstruct")
    }

    def modes(s0: float, s1: float, independent: float, shuffled: float | None):
        return {
            "s0_qraw": mode(s0, independent),
            "s1_qraw": mode(s1, independent, shuffled),
            "s0_qstruct": mode(s0, independent),
            "s1_qstruct": mode(s1, independent, shuffled),
        }

    arms = {
        "zero_update": {
            "modes": modes(0.7, 0.7, 0.7, 0.7),
            "query_modes": query,
            "fit": {"frozen_parent_unchanged": True, "updates": 0},
        },
        "opcode_coupled": {
            "modes": modes(0.7, 1.0, 0.7, 0.7),
            "query_modes": query,
            "fit": fit,
        },
        "legacy_uncoupled": {
            "modes": modes(0.7, 0.7, 0.7, 0.7),
            "query_modes": query,
            "fit": fit,
        },
        "structured_route": {
            "modes": modes(0.7, 1.0, 0.7, 0.7),
            "query_modes": query,
            "fit": fit,
        },
    }
    gates, diagnosis = compute_gates(
        arms,
        parameters={
            **__import__(
                "pilot_er_dual_stream_relation_adapter"
            ).EXPECTED_PARAMETERS
        },
        shared_initialization=True,
        development_accesses=0,
        confirmation_accesses=0,
    )
    assert all(gates.values())
    assessed_gates, assessed_diagnosis = independent_gates(
        arms, True, EXPECTED_PARAMETERS, 0, 0
    )
    assert assessed_gates == gates
    assert assessed_diagnosis == diagnosis
    assert diagnosis["selected"] == {
        "arm": "opcode_coupled",
        "mode": "s1_qraw",
        "mechanism": "learned_opcode_coupling",
    }
    legacy_interaction = deepcopy(arms)
    for query_mode in ("qraw", "qstruct"):
        for view in ("canonical", "relocated"):
            legacy_interaction["legacy_uncoupled"]["modes"][f"s1_{query_mode}"][
                view
            ]["coherent"] = _metric(1.0)
            legacy_interaction["opcode_coupled"]["modes"][f"s1_{query_mode}"][
                view
            ]["coherent"] = _metric(0.7)
    gates, diagnosis = compute_gates(
        legacy_interaction,
        parameters=EXPECTED_PARAMETERS,
        shared_initialization=True,
        development_accesses=0,
        confirmation_accesses=0,
    )
    assert all(gates.values())
    assessed_gates, assessed_diagnosis = independent_gates(
        legacy_interaction, True, EXPECTED_PARAMETERS, 0, 0
    )
    assert assessed_gates == gates
    assert assessed_diagnosis == diagnosis
    assert diagnosis["selected"] == {
        "arm": "legacy_uncoupled",
        "mode": "s1_qraw",
        "mechanism": "legacy_training_plus_acute_opcode",
    }
    trained_legacy = deepcopy(arms)
    for mode_name in ("s0_qraw", "s1_qraw", "s0_qstruct", "s1_qstruct"):
        for view in ("canonical", "relocated"):
            trained_legacy["legacy_uncoupled"]["modes"][mode_name][view][
                "coherent"
            ] = _metric(1.0)
            trained_legacy["opcode_coupled"]["modes"][mode_name][view][
                "coherent"
            ] = _metric(0.7)
            trained_legacy["structured_route"]["modes"][mode_name][view][
                "coherent"
            ] = _metric(0.7)
    gates, diagnosis = compute_gates(
        trained_legacy,
        parameters={
            **__import__(
                "pilot_er_dual_stream_relation_adapter"
            ).EXPECTED_PARAMETERS
        },
        shared_initialization=True,
        development_accesses=0,
        confirmation_accesses=0,
    )
    assert all(gates.values())
    assessed_gates, assessed_diagnosis = independent_gates(
        trained_legacy, True, EXPECTED_PARAMETERS, 0, 0
    )
    assert assessed_gates == gates
    assert assessed_diagnosis == diagnosis
    assert diagnosis["selected"] == {
        "arm": "legacy_uncoupled",
        "mode": "s0_qraw",
        "mechanism": "additional_marginal_training",
    }
    arms["opcode_coupled"]["modes"]["s1_qraw"]["opcode_shuffled"] = _metric(
        0.9
    )
    gates, _ = compute_gates(
        arms,
        parameters={
            **__import__(
                "pilot_er_dual_stream_relation_adapter"
            ).EXPECTED_PARAMETERS
        },
        shared_initialization=True,
        development_accesses=0,
        confirmation_accesses=0,
    )
    assert not gates["opcode_score_is_causal_when_selected"]
