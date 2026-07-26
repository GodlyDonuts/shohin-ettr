"""Independent exact auditor for the cross-ontology rewrite board.

The board implementation explores explicit occurrence paths breadth first.
This auditor instead derives root substitutions with an iterative matcher,
recursively lifts child reducts, and computes terminal sets by memoized
depth-first descent.  It never calls the board's matching, reduction, normal
form, consistency, or version-space functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from cross_ontology_rewrite_board import (
    CONSTRUCTORS,
    HELDOUT_THEORY_INDICES,
    RULE_LIBRARY,
    TRAIN_THEORY_INDICES,
    THEORIES,
    Demonstration,
    GroundTerm,
    PatternTerm,
    RewriteTheory,
    challenge_terms,
)


@dataclass(frozen=True, slots=True)
class RewriteAuditReceipt:
    term_count: int
    theory_count: int
    training_combination_count: int
    heldout_combination_count: int
    exhaustive_oracle_cases: int
    distinct_behavior_classes: int
    renderer_count: int
    version_space_episode_count: int


def _independent_root_reduct(
    pattern: PatternTerm,
    replacement: PatternTerm,
    term: GroundTerm,
) -> GroundTerm | None:
    stack = [(pattern, term)]
    bindings: dict[int, GroundTerm] = {}
    while stack:
        expected, observed = stack.pop()
        if expected.type_index != observed.type_index:
            return None
        if expected.variable_index is not None:
            previous = bindings.get(expected.variable_index)
            if previous is not None and previous != observed:
                return None
            bindings[expected.variable_index] = observed
            continue
        if (
            expected.constructor_index != observed.constructor_index
            or len(expected.children) != len(observed.children)
        ):
            return None
        stack.extend(zip(expected.children, observed.children, strict=True))

    def materialize(node: PatternTerm) -> GroundTerm:
        if node.variable_index is not None:
            return bindings[node.variable_index]
        assert node.constructor_index is not None
        return GroundTerm(
            node.type_index,
            node.constructor_index,
            tuple(materialize(child) for child in node.children),
        )

    return materialize(replacement)


def independent_direct_reducts(
    theory: RewriteTheory,
    term: GroundTerm,
) -> tuple[GroundTerm, ...]:
    """Derive root and child reductions without occurrence path enumeration."""

    reducts: set[GroundTerm] = set()
    for rule_index in theory.rule_indices:
        rule = RULE_LIBRARY[rule_index]
        root_reduct = _independent_root_reduct(
            rule.lhs,
            rule.rhs,
            term,
        )
        if root_reduct is not None:
            reducts.add(root_reduct)
    for child_index, child in enumerate(term.children):
        for child_reduct in independent_direct_reducts(theory, child):
            children = list(term.children)
            children[child_index] = child_reduct
            reducts.add(
                GroundTerm(
                    term.type_index,
                    term.constructor_index,
                    tuple(children),
                )
            )
    return tuple(sorted(reducts))


@lru_cache(maxsize=None)
def independent_normal_forms(
    theory_index: int,
    initial: GroundTerm,
) -> tuple[GroundTerm, ...]:
    """Compute exact terminal descendants by decreasing-size recursion."""

    reducts = independent_direct_reducts(
        THEORIES[theory_index],
        initial,
    )
    if not reducts:
        return (initial,)
    if any(reduct.node_count >= initial.node_count for reduct in reducts):
        raise ValueError("independent rewrite is not decreasing")
    terminals: set[GroundTerm] = set()
    for reduct in reducts:
        terminals.update(independent_normal_forms(theory_index, reduct))
    return tuple(sorted(terminals))


@lru_cache(maxsize=None)
def independent_behavior_signature(
    theory_index: int,
) -> tuple[tuple[GroundTerm, ...], ...]:
    return tuple(
        independent_normal_forms(theory_index, term)
        for term in challenge_terms()
    )


def independent_consistent_theories(
    evidence: tuple[Demonstration, ...],
) -> tuple[int, ...]:
    return tuple(
        theory_index
        for theory_index in range(len(THEORIES))
        if all(
            independent_normal_forms(theory_index, demo.initial)
            == demo.normal_forms
            for demo in evidence
        )
    )


def independent_behavioral_class_count(
    theory_indices: tuple[int, ...],
) -> int:
    return len(
        {
            independent_behavior_signature(theory_index)
            for theory_index in theory_indices
        }
    )


def audit_static_board() -> RewriteAuditReceipt:
    """Return exact finite-board cardinalities after structural checks."""

    constructor_indices = {item.index for item in CONSTRUCTORS}
    if constructor_indices != set(range(len(CONSTRUCTORS))):
        raise ValueError("constructor catalog is not contiguous")
    train_pairs = {
        THEORIES[index].rule_indices
        for index in TRAIN_THEORY_INDICES
    }
    heldout_pairs = {
        THEORIES[index].rule_indices
        for index in HELDOUT_THEORY_INDICES
    }
    if train_pairs & heldout_pairs:
        raise ValueError("rule-combination split overlaps")
    train_primitives = {
        rule_index
        for pair in train_pairs
        for rule_index in pair
    }
    if train_primitives != set(range(len(RULE_LIBRARY))):
        raise ValueError("training split omits a primitive rule")
    signatures = {
        independent_behavior_signature(theory_index)
        for theory_index in range(len(THEORIES))
    }
    return RewriteAuditReceipt(
        term_count=len(challenge_terms()),
        theory_count=len(THEORIES),
        training_combination_count=len(TRAIN_THEORY_INDICES),
        heldout_combination_count=len(HELDOUT_THEORY_INDICES),
        exhaustive_oracle_cases=len(challenge_terms()) * len(THEORIES),
        distinct_behavior_classes=len(signatures),
        renderer_count=4,
        version_space_episode_count=(
            len(HELDOUT_THEORY_INDICES) * 4
        ),
    )


__all__ = [
    "RewriteAuditReceipt",
    "audit_static_board",
    "independent_behavior_signature",
    "independent_behavioral_class_count",
    "independent_consistent_theories",
    "independent_direct_reducts",
    "independent_normal_forms",
]
