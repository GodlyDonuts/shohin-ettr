"""Ontology-neutral immutable packets and typed transaction mechanics.

This module owns structural integrity only.  It contains no task-family
labels, semantic operators, rule matching, scheduling, search, repair,
answering, or assessor callback.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import json
from typing import Iterable


MAX_TYPES = 8
MAX_RELATIONS = 8
MAX_ARITY = 3
MAX_OBJECTS = 32
MAX_EDGES = 96
MAX_STEPS = 256


class CrossOntologySchemaError(ValueError):
    """A packet or transaction violates structural custody."""


class TransactionOpcode(IntEnum):
    ALLOC = 0
    WRITE = 1
    CLEAR = 2
    LINK = 3
    UNLINK = 4
    SET_ROOT = 5
    COMMIT = 6
    HALT = 7


@dataclass(frozen=True, slots=True)
class RelationSpec:
    index: int
    argument_types: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not 0 <= self.index < MAX_RELATIONS
            or not 1 <= len(self.argument_types) <= MAX_ARITY
            or any(
                not 0 <= type_index < MAX_TYPES
                for type_index in self.argument_types
            )
        ):
            raise CrossOntologySchemaError(
                "relation specification differs"
            )


@dataclass(frozen=True, slots=True)
class ObjectCell:
    slot: int
    type_index: int
    value: int = 0

    def __post_init__(self) -> None:
        if (
            not 0 <= self.slot < MAX_OBJECTS
            or not 0 <= self.type_index < MAX_TYPES
            or not -(2**31) <= self.value < 2**31
        ):
            raise CrossOntologySchemaError("object cell differs")


@dataclass(frozen=True, slots=True)
class RelationEdge:
    relation_index: int
    arguments: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not 0 <= self.relation_index < MAX_RELATIONS
            or not 1 <= len(self.arguments) <= MAX_ARITY
            or any(
                not 0 <= slot < MAX_OBJECTS
                for slot in self.arguments
            )
        ):
            raise CrossOntologySchemaError("relation edge differs")


@dataclass(frozen=True, slots=True)
class ReactorState:
    capacity: int
    type_count: int
    relation_specs: tuple[RelationSpec, ...]
    cells: tuple[ObjectCell, ...] = ()
    edges: tuple[RelationEdge, ...] = ()
    root: int | None = None
    committed_steps: int = 0
    halted: bool = False

    def __post_init__(self) -> None:
        if (
            not 1 <= self.capacity <= MAX_OBJECTS
            or not 1 <= self.type_count <= MAX_TYPES
            or not 0 <= self.committed_steps <= MAX_STEPS
            or len(self.cells) > self.capacity
            or len(self.edges) > MAX_EDGES
        ):
            raise CrossOntologySchemaError("reactor geometry differs")
        relation_indices = [
            relation.index for relation in self.relation_specs
        ]
        if (
            len(relation_indices) != len(set(relation_indices))
            or len(relation_indices) > MAX_RELATIONS
            or any(
                type_index >= self.type_count
                for relation in self.relation_specs
                for type_index in relation.argument_types
            )
        ):
            raise CrossOntologySchemaError(
                "reactor relation schema differs"
            )
        slots = [cell.slot for cell in self.cells]
        if (
            len(slots) != len(set(slots))
            or any(cell.slot >= self.capacity for cell in self.cells)
            or any(
                cell.type_index >= self.type_count
                for cell in self.cells
            )
        ):
            raise CrossOntologySchemaError("reactor cells differ")
        cell_by_slot = {
            cell.slot: cell
            for cell in self.cells
        }
        relation_by_index = {
            relation.index: relation
            for relation in self.relation_specs
        }
        if len(self.edges) != len(set(self.edges)):
            raise CrossOntologySchemaError(
                "reactor edges are not unique"
            )
        for edge in self.edges:
            relation = relation_by_index.get(edge.relation_index)
            if (
                relation is None
                or len(edge.arguments)
                != len(relation.argument_types)
            ):
                raise CrossOntologySchemaError(
                    "edge relation geometry differs"
                )
            for slot, type_index in zip(
                edge.arguments,
                relation.argument_types,
                strict=True,
            ):
                cell = cell_by_slot.get(slot)
                if cell is None or cell.type_index != type_index:
                    raise CrossOntologySchemaError(
                        "edge argument typing differs"
                    )
        if self.root is not None and self.root not in cell_by_slot:
            raise CrossOntologySchemaError("reactor root differs")

    def deployed_wire(self) -> bytes:
        payload = {
            "capacity": self.capacity,
            "cells": [
                [cell.slot, cell.type_index, cell.value]
                for cell in self.cells
            ],
            "committed_steps": self.committed_steps,
            "edges": [
                [edge.relation_index, *edge.arguments]
                for edge in self.edges
            ],
            "halted": self.halted,
            "relation_specs": [
                [relation.index, *relation.argument_types]
                for relation in self.relation_specs
            ],
            "root": self.root,
            "type_count": self.type_count,
            "version": 1,
        }
        return (
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")

    @classmethod
    def from_deployed_wire(cls, payload: bytes) -> ReactorState:
        try:
            value = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CrossOntologySchemaError(
                "reactor wire differs"
            ) from exc
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "capacity",
                "cells",
                "committed_steps",
                "edges",
                "halted",
                "relation_specs",
                "root",
                "type_count",
                "version",
            }
            or value["version"] != 1
        ):
            raise CrossOntologySchemaError(
                "reactor wire schema differs"
            )
        try:
            relations = tuple(
                RelationSpec(
                    index=int(item[0]),
                    argument_types=tuple(
                        int(argument)
                        for argument in item[1:]
                    ),
                )
                for item in value["relation_specs"]
            )
            cells = tuple(
                ObjectCell(
                    slot=int(item[0]),
                    type_index=int(item[1]),
                    value=int(item[2]),
                )
                for item in value["cells"]
            )
            relation_by_index = {
                relation.index: relation
                for relation in relations
            }
            edges = tuple(
                RelationEdge(
                    relation_index=int(item[0]),
                    arguments=tuple(
                        int(argument)
                        for argument in item[1:]
                    ),
                )
                for item in value["edges"]
            )
            if any(
                len(edge.arguments)
                != len(
                    relation_by_index[edge.relation_index]
                    .argument_types
                )
                for edge in edges
            ):
                raise KeyError
            return cls(
                capacity=int(value["capacity"]),
                type_count=int(value["type_count"]),
                relation_specs=relations,
                cells=cells,
                edges=edges,
                root=(
                    None
                    if value["root"] is None
                    else int(value["root"])
                ),
                committed_steps=int(value["committed_steps"]),
                halted=bool(value["halted"]),
            )
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise CrossOntologySchemaError(
                "reactor wire payload differs"
            ) from exc


@dataclass(frozen=True, slots=True)
class Transaction:
    opcode: TransactionOpcode
    operands: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.opcode, TransactionOpcode)
            or any(not isinstance(value, int) for value in self.operands)
        ):
            raise CrossOntologySchemaError("transaction differs")


def _sorted_cells(
    cells: Iterable[ObjectCell],
) -> tuple[ObjectCell, ...]:
    return tuple(sorted(cells, key=lambda cell: cell.slot))


def _sorted_edges(
    edges: Iterable[RelationEdge],
) -> tuple[RelationEdge, ...]:
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.relation_index,
                edge.arguments,
            ),
        )
    )


def apply_transaction(
    state: ReactorState,
    transaction: Transaction,
) -> ReactorState:
    """Apply one structurally checked transaction atomically."""

    if state.halted:
        raise CrossOntologySchemaError(
            "transaction follows halt"
        )
    opcode = transaction.opcode
    operands = transaction.operands
    cells = {
        cell.slot: cell
        for cell in state.cells
    }
    edges = set(state.edges)
    root = state.root

    if opcode == TransactionOpcode.ALLOC:
        if len(operands) != 2:
            raise CrossOntologySchemaError("ALLOC arity differs")
        slot, type_index = operands
        if (
            not 0 <= slot < state.capacity
            or not 0 <= type_index < state.type_count
            or slot in cells
        ):
            raise CrossOntologySchemaError("ALLOC operands differ")
        cells[slot] = ObjectCell(slot, type_index)
    elif opcode == TransactionOpcode.WRITE:
        if len(operands) != 2 or operands[0] not in cells:
            raise CrossOntologySchemaError("WRITE operands differ")
        slot, value = operands
        cells[slot] = replace(cells[slot], value=value)
    elif opcode == TransactionOpcode.CLEAR:
        if len(operands) != 1 or operands[0] not in cells:
            raise CrossOntologySchemaError("CLEAR operands differ")
        slot = operands[0]
        if root == slot or any(slot in edge.arguments for edge in edges):
            raise CrossOntologySchemaError(
                "CLEAR references live structure"
            )
        del cells[slot]
    elif opcode in {
        TransactionOpcode.LINK,
        TransactionOpcode.UNLINK,
    }:
        if not operands:
            raise CrossOntologySchemaError(
                "relation transaction arity differs"
            )
        relation_index, *arguments = operands
        relation = next(
            (
                item
                for item in state.relation_specs
                if item.index == relation_index
            ),
            None,
        )
        if (
            relation is None
            or len(arguments) != len(relation.argument_types)
        ):
            raise CrossOntologySchemaError(
                "relation transaction schema differs"
            )
        for slot, type_index in zip(
            arguments,
            relation.argument_types,
            strict=True,
        ):
            if slot not in cells or cells[slot].type_index != type_index:
                raise CrossOntologySchemaError(
                    "relation transaction typing differs"
                )
        edge = RelationEdge(
            relation_index=relation_index,
            arguments=tuple(arguments),
        )
        if opcode == TransactionOpcode.LINK:
            if edge in edges or len(edges) >= MAX_EDGES:
                raise CrossOntologySchemaError(
                    "LINK edge differs"
                )
            edges.add(edge)
        else:
            if edge not in edges:
                raise CrossOntologySchemaError(
                    "UNLINK edge differs"
                )
            edges.remove(edge)
    elif opcode == TransactionOpcode.SET_ROOT:
        if len(operands) != 1 or operands[0] not in cells:
            raise CrossOntologySchemaError(
                "SET_ROOT operands differ"
            )
        root = operands[0]
    elif opcode == TransactionOpcode.COMMIT:
        if operands:
            raise CrossOntologySchemaError("COMMIT operands differ")
    elif opcode == TransactionOpcode.HALT:
        if operands:
            raise CrossOntologySchemaError("HALT operands differ")
    else:
        raise CrossOntologySchemaError("transaction opcode differs")

    return ReactorState(
        capacity=state.capacity,
        type_count=state.type_count,
        relation_specs=state.relation_specs,
        cells=_sorted_cells(cells.values()),
        edges=_sorted_edges(edges),
        root=root,
        committed_steps=state.committed_steps + 1,
        halted=opcode == TransactionOpcode.HALT,
    )


def apply_transactions(
    state: ReactorState,
    transactions: Iterable[Transaction],
) -> ReactorState:
    result = state
    for transaction in transactions:
        result = apply_transaction(result, transaction)
    return result


__all__ = [
    "CrossOntologySchemaError",
    "MAX_ARITY",
    "MAX_EDGES",
    "MAX_OBJECTS",
    "MAX_RELATIONS",
    "MAX_STEPS",
    "MAX_TYPES",
    "ObjectCell",
    "ReactorState",
    "RelationEdge",
    "RelationSpec",
    "Transaction",
    "TransactionOpcode",
    "apply_transaction",
    "apply_transactions",
]
