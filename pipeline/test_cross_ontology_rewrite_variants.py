from __future__ import annotations

import ast
import json
from pathlib import Path

from cross_ontology_rewrite_board import (
    HELDOUT_THEORY_INDICES,
    identifying_evidence,
)
from cross_ontology_rewrite_variants import (
    CHALLENGES_PER_VARIANT,
    VARIANT_ORDER,
    ExecutionSemantics,
    GraphEdgeKind,
    GraphNodeKind,
    QualificationDecision,
    RewriteVariantKind,
    VariantExpectation,
    audit_rewrite_variant_family,
    build_rewrite_variant_family,
    canonical_rule_signature,
    execute_variant_challenge,
)


def test_all_seven_variants_are_deterministic_and_materially_distinct() -> None:
    for theory_index in HELDOUT_THEORY_INDICES:
        first = build_rewrite_variant_family(theory_index)
        second = build_rewrite_variant_family(theory_index)
        assert first == second
        assert tuple(variant.kind for variant in first) == VARIANT_ORDER
        assert {
            variant.material_sha256()
            for variant in first
        } == {
            variant.material_sha256()
            for variant in second
        }
        assert len(
            {
                variant.material_sha256()
                for variant in first
            }
        ) == 7
        assert all(
            len(variant.challenges) == CHALLENGES_PER_VARIANT
            for variant in first
        )
        receipt = audit_rewrite_variant_family(first)
        assert receipt.all_contracts_pass
        assert receipt.variant_count == 7
        assert receipt.exact_invariance_cases == 48


def test_structural_equivalents_are_exactly_invariant_after_alignment() -> None:
    for theory_index in HELDOUT_THEORY_INDICES:
        family = build_rewrite_variant_family(theory_index)
        base, alpha, alias, reified = family[:4]
        assert all(
            variant.expectation == VariantExpectation.EXACT_INVARIANCE
            for variant in (alpha, alias, reified)
        )
        assert all(
            canonical_rule_signature(variant)
            == canonical_rule_signature(base)
            for variant in (alpha, alias, reified)
        )
        assert all(
            variant.oracle.aligned_outputs
            == base.oracle.aligned_outputs
            for variant in (alpha, alias, reified)
        )
        for variant in (base, alpha, alias, reified):
            assert tuple(
                execute_variant_challenge(variant, offset)
                for offset in range(CHALLENGES_PER_VARIANT)
            ) == variant.oracle.aligned_outputs


def test_alpha_alias_and_reification_are_real_graph_transforms() -> None:
    for theory_index in HELDOUT_THEORY_INDICES:
        base, alpha, alias, reified, *_ = (
            build_rewrite_variant_family(theory_index)
        )

        assert tuple(
            base_rule
            for _, base_rule in alpha.alignment.rule_to_base
        ) == tuple(
            reversed(
                tuple(
                    base_rule
                    for _, base_rule in base.alignment.rule_to_base
                )
            )
        )
        base_variables = {
            node.variable_index
            for rule in base.rules
            for graph in (rule.lhs, rule.rhs)
            for node in graph.nodes
            if node.kind == GraphNodeKind.VARIABLE
        }
        alpha_variables = {
            node.variable_index
            for rule in alpha.rules
            for graph in (rule.lhs, rule.rhs)
            for node in graph.nodes
            if node.kind == GraphNodeKind.VARIABLE
        }
        assert base_variables
        assert alpha_variables
        assert base_variables.isdisjoint(alpha_variables)
        assert min(
            node.node_index
            for rule in alpha.rules
            for graph in (rule.lhs, rule.rhs)
            for node in graph.nodes
        ) >= 1000

        alias_classes = {}
        for constructor in alias.constructors:
            alias_classes.setdefault(
                constructor.equivalence_class,
                [],
            ).append(constructor.symbol_index)
        assert any(
            len(symbols) == 2
            for symbols in alias_classes.values()
        )
        material_graphs = [
            graph
            for rule in alias.rules
            for graph in (rule.lhs, rule.rhs)
        ]
        for demonstration in alias.evidence:
            material_graphs.append(demonstration.initial)
            material_graphs.extend(demonstration.normal_forms)
        material_graphs.extend(alias.challenges)
        used_alias_symbols = {
            node.symbol_index
            for graph in material_graphs
            for node in graph.nodes
            if node.kind == GraphNodeKind.CONSTRUCTOR
        }
        split_symbols = next(
            symbols
            for symbols in alias_classes.values()
            if len(symbols) == 2
        )
        assert set(split_symbols) <= used_alias_symbols

        all_reified_graphs = [
            graph
            for rule in reified.rules
            for graph in (rule.lhs, rule.rhs)
        ]
        assert all(graph.reified_relations for graph in all_reified_graphs)
        assert all(
            any(
                node.kind == GraphNodeKind.RELATION
                for node in graph.nodes
            )
            for graph in all_reified_graphs
            if len(graph.nodes) > 1
        )
        assert all(
            edge.kind != GraphEdgeKind.CHILD
            for graph in all_reified_graphs
            for edge in graph.edges
        )
        assert any(
            edge.kind == GraphEdgeKind.CHILD
            for rule in base.rules
            for graph in (rule.lhs, rule.rhs)
            for edge in graph.edges
        )


def test_type_and_execution_twins_have_exact_behavioral_witnesses() -> None:
    for theory_index in HELDOUT_THEORY_INDICES:
        family = build_rewrite_variant_family(theory_index)
        base = family[0]
        type_twin = family[4]
        execution_twin = family[5]

        assert type_twin.kind == RewriteVariantKind.TYPE_TWIN
        assert type_twin.expectation == (
            VariantExpectation.EXACT_TYPE_SEPARATION
        )
        assert any(
            constructor.result_type == 2
            or 2 in constructor.argument_types
            for constructor in type_twin.constructors
        )
        assert canonical_rule_signature(type_twin) != (
            canonical_rule_signature(base)
        )
        type_witness = type_twin.oracle.witness
        assert type_witness is not None
        assert type_witness.base_outcome not in (
            type_witness.variant_outcomes
        )
        assert execute_variant_challenge(
            type_twin,
            type_witness.challenge_offset,
        ) in type_witness.variant_outcomes

        assert execution_twin.execution_semantics == (
            ExecutionSemantics.ROOT_ONLY
        )
        assert canonical_rule_signature(execution_twin) == (
            canonical_rule_signature(base)
        )
        execution_witness = execution_twin.oracle.witness
        assert execution_witness is not None
        assert execution_witness.base_outcome not in (
            execution_witness.variant_outcomes
        )
        assert execute_variant_challenge(
            execution_twin,
            execution_witness.challenge_offset,
        ) in execution_witness.variant_outcomes


def test_ambiguity_twin_deletes_evidence_and_requires_abstention() -> None:
    for theory_index in HELDOUT_THEORY_INDICES:
        family = build_rewrite_variant_family(theory_index)
        base = family[0]
        ambiguity = family[6]
        assert ambiguity.kind == (
            RewriteVariantKind.AMBIGUITY_DELETED_TWIN
        )
        assert len(ambiguity.evidence) == (
            len(identifying_evidence(theory_index)) - 1
        )
        assert ambiguity.rules == base.rules
        assert ambiguity.execution_semantics == base.execution_semantics
        assert ambiguity.oracle.decision == (
            QualificationDecision.ABSTAIN_AMBIGUOUS
        )
        assert ambiguity.oracle.behavioral_class_count >= 2
        witness = ambiguity.oracle.witness
        assert witness is not None
        assert witness.base_decision == QualificationDecision.EXECUTE
        assert witness.variant_decision == (
            QualificationDecision.ABSTAIN_AMBIGUOUS
        )
        assert len(witness.variant_outcomes) >= 2
        assert witness.base_outcome in witness.variant_outcomes


def test_source_deleted_packets_exclude_oracle_and_runtime_channels() -> None:
    forbidden = {
        "alignment",
        "answer",
        "assessor",
        "callback",
        "challenge",
        "evidence",
        "normal_forms",
        "oracle",
        "renderer",
        "source",
        "theory_index",
        "token",
        "variant",
    }
    for theory_index in HELDOUT_THEORY_INDICES:
        family = build_rewrite_variant_family(theory_index)
        for variant in family:
            payload = variant.source_deleted_theory_bytes()
            assert payload == variant.source_deleted_theory_bytes()
            value = json.loads(payload)
            assert value["schema"] == (
                "cross_ontology_rewrite_theory_v1"
            )
            lowered = payload.decode("ascii").lower()
            assert all(word not in lowered for word in forbidden)


def test_compiler_surface_contains_evidence_without_latent_rules() -> None:
    for theory_index in HELDOUT_THEORY_INDICES:
        for variant in build_rewrite_variant_family(theory_index):
            source = json.loads(variant.compiler_source_bytes())
            assert source["schema"] == (
                "ettr-anonymous-evidence-v1"
            )
            assert (
                source["demonstrations"]
                or variant.kind
                == RewriteVariantKind.AMBIGUITY_DELETED_TWIN
            )
            assert "rules" not in source
            assert "execution_semantics" not in source
            for offset in range(len(variant.challenges)):
                challenge = json.loads(
                    variant.late_challenge_bytes(offset)
                )
                assert challenge["schema"] == (
                    "ettr-anonymous-challenge-v1"
                )
                assert set(challenge) == {"challenge", "schema"}


def test_module_has_no_candidate_or_assessor_runtime_import() -> None:
    module_path = Path(__file__).with_name(
        "cross_ontology_rewrite_variants.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".")[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "cross_ontology_rewrite_board",
            "dataclasses",
            "enum",
            "functools",
            "hashlib",
        "json",
        "typing",
    }
