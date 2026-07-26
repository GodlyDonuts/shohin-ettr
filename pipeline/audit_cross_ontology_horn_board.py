"""Independent exact audit for the cross-ontology Horn board."""

from __future__ import annotations

from itertools import product

from cross_ontology_horn_board import (
    Demonstration,
    GroundAtom,
    OBJECT_TYPES,
    PREDICATES,
    RULE_LIBRARY,
    THEORIES,
    HornTheory,
)


def independent_closure(
    theory: HornTheory,
    initial: tuple[GroundAtom, ...],
) -> tuple[GroundAtom, ...]:
    """Ground every implication first, then saturate propositionally."""

    implications: list[
        tuple[frozenset[GroundAtom], GroundAtom]
    ] = []
    for rule_index in theory.rule_indices:
        rule = RULE_LIBRARY[rule_index]
        variable_types: dict[int, int] = {}
        for atom in (*rule.premises, rule.conclusion):
            spec = PREDICATES[atom.predicate]
            for variable, type_index in zip(
                atom.variables,
                spec.argument_types,
                strict=True,
            ):
                if variable in variable_types:
                    assert variable_types[variable] == type_index
                variable_types[variable] = type_index
        variables = tuple(sorted(variable_types))
        domains = tuple(
            tuple(
                slot
                for slot, object_type in enumerate(OBJECT_TYPES)
                if object_type == variable_types[variable]
            )
            for variable in variables
        )
        for values in product(*domains):
            assignment = dict(zip(variables, values, strict=True))

            def grounded(pattern) -> GroundAtom:
                return GroundAtom(
                    pattern.predicate,
                    tuple(
                        assignment[variable]
                        for variable in pattern.variables
                    ),
                )

            implications.append(
                (
                    frozenset(
                        grounded(premise)
                        for premise in rule.premises
                    ),
                    grounded(rule.conclusion),
                )
            )
    facts = set(initial)
    pending = list(implications)
    changed = True
    while changed:
        changed = False
        retained = []
        for premises, conclusion in pending:
            if premises <= facts:
                if conclusion not in facts:
                    facts.add(conclusion)
                    changed = True
            else:
                retained.append((premises, conclusion))
        pending = retained
    return tuple(sorted(facts))


def independent_consistent_theories(
    evidence: tuple[Demonstration, ...],
) -> tuple[int, ...]:
    return tuple(
        theory_index
        for theory_index, theory in enumerate(THEORIES)
        if all(
            independent_closure(theory, demo.initial)
            == demo.terminal
            for demo in evidence
        )
    )


__all__ = [
    "independent_closure",
    "independent_consistent_theories",
]
