#!/usr/bin/env python3
"""Exact multi-valued decision-DAG execution for DIVERGE-ULC1.

The arena represents disjoint sets of complete record choices without assigning
one bit to every Cartesian world.  State groups may merge only when their full
typed states are equal; their lineage expressions remain available for later
evidence filtering.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

from version_space_accounting import canonical_json_bytes

from diverge_v0 import (
    ABSTAIN,
    ANSWER,
    REJECT,
    DivergeContractError,
    Query,
    QueryDecision,
    TypedState,
    TypedTransaction,
    apply_transaction,
    read_query,
    validate_commitment,
)

SCHEMA = "shohin-diverge-ulc1-mdd-v1"


@dataclass(frozen=True)
class RuntimeChoice:
    record_index: int
    domain_value: int
    mass: int
    transactions: tuple[TypedTransaction, ...]
    witness_code: int
    semantic_key: str
    provenance: str
    parse_record: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("record index", self.record_index),
            ("domain value", self.domain_value),
            ("mass", self.mass),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DivergeContractError(f"{name} must be a nonnegative integer")
        if self.mass <= 0:
            raise DivergeContractError("choice mass must be positive")
        if isinstance(self.witness_code, bool) or not isinstance(
            self.witness_code, int
        ):
            raise DivergeContractError("witness code must be an exact integer")
        if not isinstance(self.semantic_key, str) or not self.semantic_key:
            raise DivergeContractError("choice semantic key must be nonempty")
        object.__setattr__(
            self,
            "provenance",
            validate_commitment(self.provenance, "choice provenance"),
        )
        record = tuple(sorted(self.parse_record))
        if any(not isinstance(key, str) for key, _ in record):
            raise DivergeContractError("parse record keys must be strings")
        canonical_json_bytes(dict(record))
        object.__setattr__(self, "parse_record", record)

    def record(self) -> dict[str, object]:
        return {
            "record_index": self.record_index,
            "domain_value": self.domain_value,
            "mass": self.mass,
            "transactions": [item.record() for item in self.transactions],
            "witness_code": self.witness_code,
            "semantic_key": self.semantic_key,
            "provenance": self.provenance,
            "parse": dict(self.parse_record),
        }


@dataclass(frozen=True)
class ExpressionNode:
    kind: str
    parent: int | None = None
    variable: int | None = None
    choice: int | None = None
    mass: int = 1
    children: tuple[int, ...] = ()

    def record(self) -> dict[str, object]:
        if self.kind == "base":
            return {"kind": "base"}
        if self.kind == "extend":
            return {
                "kind": "extend",
                "parent": self.parent,
                "variable": self.variable,
                "choice": self.choice,
                "mass": self.mass,
            }
        if self.kind == "union":
            return {"kind": "union", "children": list(self.children)}
        raise AssertionError("unknown expression node")


class ExpressionArena:
    """Hash-consed exact semiring expression over disjoint assignments."""

    def __init__(self) -> None:
        self.nodes: list[ExpressionNode] = [ExpressionNode("base")]
        self._intern: dict[ExpressionNode, int] = {self.nodes[0]: 0}

    @property
    def base(self) -> int:
        return 0

    def _add(self, node: ExpressionNode) -> int:
        existing = self._intern.get(node)
        if existing is not None:
            return existing
        index = len(self.nodes)
        self.nodes.append(node)
        self._intern[node] = index
        self._clear_caches()
        return index

    def _clear_caches(self) -> None:
        self.assignment_count.cache_clear()
        self.total_mass.cache_clear()

    def extend(self, parent: int, variable: int, choice: int, mass: int) -> int:
        if not 0 <= parent < len(self.nodes):
            raise DivergeContractError("expression parent is absent")
        if min(variable, choice) < 0 or mass <= 0:
            raise DivergeContractError("invalid expression extension")
        return self._add(ExpressionNode("extend", parent, variable, choice, mass))

    def union(self, roots: Iterable[int]) -> int:
        children = []
        for root in roots:
            if not 0 <= root < len(self.nodes):
                raise DivergeContractError("expression union child is absent")
            node = self.nodes[root]
            children.extend(node.children if node.kind == "union" else (root,))
        children = sorted(set(children))
        if not children:
            raise DivergeContractError("expression union cannot be empty")
        if len(children) == 1:
            return children[0]
        return self._add(ExpressionNode("union", children=tuple(children)))

    @lru_cache(maxsize=None)
    def assignment_count(self, root: int) -> int:
        node = self.nodes[root]
        if node.kind == "base":
            return 1
        if node.kind == "extend":
            assert node.parent is not None
            return self.assignment_count(node.parent)
        return sum(self.assignment_count(child) for child in node.children)

    @lru_cache(maxsize=None)
    def total_mass(self, root: int) -> int:
        node = self.nodes[root]
        if node.kind == "base":
            return 1
        if node.kind == "extend":
            assert node.parent is not None
            return node.mass * self.total_mass(node.parent)
        return sum(self.total_mass(child) for child in node.children)

    def constrained_mass(
        self,
        root: int,
        allowed: Mapping[int, frozenset[int]],
    ) -> int:
        memo: dict[int, int] = {}

        def visit(index: int) -> int:
            cached = memo.get(index)
            if cached is not None:
                return cached
            node = self.nodes[index]
            if node.kind == "base":
                value = 1
            elif node.kind == "extend":
                assert node.parent is not None and node.variable is not None
                assert node.choice is not None
                permit = allowed.get(node.variable)
                value = (
                    0
                    if permit is not None and node.choice not in permit
                    else node.mass * visit(node.parent)
                )
            else:
                value = sum(visit(child) for child in node.children)
            memo[index] = value
            return value

        return visit(root)

    def accepts(self, root: int, assignment: Sequence[int]) -> bool:
        memo: dict[int, bool] = {}

        def visit(index: int) -> bool:
            cached = memo.get(index)
            if cached is not None:
                return cached
            node = self.nodes[index]
            if node.kind == "base":
                value = True
            elif node.kind == "extend":
                assert node.parent is not None and node.variable is not None
                assert node.choice is not None
                value = (
                    node.variable < len(assignment)
                    and assignment[node.variable] == node.choice
                    and visit(node.parent)
                )
            else:
                value = any(visit(child) for child in node.children)
            memo[index] = value
            return value

        return visit(root)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes([item.record() for item in self.nodes])


@dataclass(frozen=True)
class MDDStateGroup:
    state: TypedState | None
    expression: int

    @property
    def contradiction(self) -> bool:
        return self.state is None

    def record(self) -> dict[str, object]:
        return {
            "state": None if self.state is None else self.state.record(),
            "expression": self.expression,
        }


@dataclass(frozen=True)
class MDDExecution:
    arena: ExpressionArena
    groups: tuple[MDDStateGroup, ...]
    choices: tuple[tuple[RuntimeChoice, ...], ...]
    represented_worlds: int
    unique_transaction_applications: int
    logical_transaction_applications: int
    peak_groups: int
    overflow: bool = False

    @property
    def shared_transaction_applications(self) -> int:
        return (
            self.logical_transaction_applications - self.unique_transaction_applications
        )

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "groups": [item.record() for item in self.groups],
            "choices": [[item.record() for item in row] for row in self.choices],
            "represented_worlds": self.represented_worlds,
            "unique_transaction_applications": self.unique_transaction_applications,
            "logical_transaction_applications": self.logical_transaction_applications,
            "peak_groups": self.peak_groups,
            "overflow": self.overflow,
            "arena": [item.record() for item in self.arena.nodes],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.record())


def _state_key(state: TypedState | None) -> bytes:
    return canonical_json_bytes(None if state is None else state.record())


def _validate_choices(
    choices: Sequence[Sequence[RuntimeChoice]],
) -> tuple[tuple[RuntimeChoice, ...], ...]:
    output = []
    for record_index, row in enumerate(choices):
        row = tuple(sorted(row, key=lambda item: item.domain_value))
        if not row or any(item.record_index != record_index for item in row):
            raise DivergeContractError("runtime choices are not record aligned")
        if tuple(item.domain_value for item in row) != tuple(range(len(row))):
            raise DivergeContractError("runtime choice domains must be contiguous")
        if len({item.semantic_key for item in row}) != len(row):
            raise DivergeContractError("runtime choices contain duplicate semantics")
        output.append(row)
    if not output:
        raise DivergeContractError("MDD execution needs at least one record")
    return tuple(output)


def execute_mdd(
    initial_state: TypedState,
    choices: Sequence[Sequence[RuntimeChoice]],
    *,
    max_nodes: int = 1_000_000,
    max_groups: int = 100_000,
) -> MDDExecution:
    choices = _validate_choices(choices)
    arena = ExpressionArena()
    groups = (MDDStateGroup(initial_state, arena.base),)
    unique = 0
    logical = 0
    peak_groups = 1
    represented = 1
    for variable, row in enumerate(choices):
        next_groups: dict[bytes, tuple[TypedState | None, list[int]]] = {}
        for group in groups:
            prefixes = arena.assignment_count(group.expression)
            for choice in row:
                state = group.state
                for transaction in choice.transactions:
                    logical += prefixes
                    unique += 1
                    if state is not None:
                        try:
                            state = apply_transaction(state, transaction)
                        except DivergeContractError:
                            state = None
                expression = arena.extend(
                    group.expression,
                    variable,
                    choice.domain_value,
                    choice.mass,
                )
                key = _state_key(state)
                if key not in next_groups:
                    next_groups[key] = (state, [])
                next_groups[key][1].append(expression)
        groups = tuple(
            MDDStateGroup(state, arena.union(expressions))
            for _, (state, expressions) in sorted(next_groups.items())
        )
        represented *= len(row)
        peak_groups = max(peak_groups, len(groups))
        if len(arena.nodes) > max_nodes or len(groups) > max_groups:
            return MDDExecution(
                arena, (), choices, 0, unique, logical, peak_groups, True
            )
    if sum(arena.assignment_count(group.expression) for group in groups) != represented:
        raise AssertionError("MDD support accounting lost or duplicated assignments")
    return MDDExecution(
        arena,
        groups,
        choices,
        represented,
        unique,
        logical,
        peak_groups,
        False,
    )


def query_mdd(
    execution: MDDExecution,
    query: Query,
    *,
    allowed: Mapping[int, frozenset[int]] | None = None,
) -> QueryDecision:
    if execution.overflow:
        return QueryDecision("OVERFLOW", None, (), 0)
    allowed = allowed or {}
    answers: dict[int, int] = {}
    for group in execution.groups:
        mass = execution.arena.constrained_mass(group.expression, allowed)
        if mass == 0:
            continue
        if group.state is None:
            return QueryDecision(REJECT, None, (), 0)
        answer = read_query(group.state, query)
        answers[answer] = answers.get(answer, 0) + mass
    marginal = tuple(sorted(answers.items()))
    total = sum(answers.values())
    if not marginal:
        return QueryDecision(REJECT, None, (), 0)
    if len(marginal) == 1:
        return QueryDecision(ANSWER, marginal[0][0], marginal, total)
    return QueryDecision(ABSTAIN, None, marginal, total)


def support_contains(
    execution: MDDExecution,
    assignment: Sequence[int],
) -> bool:
    return any(
        execution.arena.accepts(group.expression, assignment)
        for group in execution.groups
    )


def execute_choice_path(
    initial_state: TypedState,
    choices: Sequence[Sequence[RuntimeChoice]],
    assignment: Sequence[int],
) -> TypedState | None:
    if len(choices) != len(assignment):
        raise DivergeContractError("choice path has the wrong width")
    state: TypedState | None = initial_state
    for row, value in zip(choices, assignment, strict=True):
        if value < 0 or value >= len(row):
            raise DivergeContractError("choice path leaves a record domain")
        for transaction in row[value].transactions:
            if state is None:
                break
            try:
                state = apply_transaction(state, transaction)
            except DivergeContractError:
                state = None
    return state


def k_best_product_paths(
    choices: Sequence[Sequence[RuntimeChoice]],
    limit: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate product assignments in descending exact mass without full product."""

    if limit <= 0:
        return ()
    rows = [
        tuple(sorted(row, key=lambda item: (-item.mass, item.domain_value)))
        for row in choices
    ]
    if not rows or any(not row for row in rows):
        return ()
    start = tuple(0 for _ in rows)

    def score(indices: tuple[int, ...]) -> int:
        return math.prod(
            row[index].mass for row, index in zip(rows, indices, strict=True)
        )

    heap = [(-score(start), start)]
    visited = {start}
    output = []
    while heap and len(output) < limit:
        _, indices = heapq.heappop(heap)
        output.append(
            tuple(
                rows[variable][index].domain_value
                for variable, index in enumerate(indices)
            )
        )
        for variable in range(len(rows)):
            if indices[variable] + 1 >= len(rows[variable]):
                continue
            candidate = list(indices)
            candidate[variable] += 1
            key = tuple(candidate)
            if key in visited:
                continue
            visited.add(key)
            heapq.heappush(heap, (-score(key), key))
    return tuple(output)
