"""Deterministic bounded typed local rewriting semantics for ETTR-IL v3.

This module is a standalone, assessor-side finite semantic family.  It has no
model, tensor, filesystem, network, training, or accelerator dependency.

The state is six fixed registers with alternating anonymous types.  Each type
has four symbols, so the state domain has exactly ``4**6 == 4096`` worlds.
Six reversible, size-preserving local laws act on adjacent register pairs.
The fifteen theories are all unordered two-law combinations.  Commands name
only an opaque theory-local law slot, a local site, and a direction.

Primary execution interprets the law catalog directly.  Replay execution uses
an independently encoded transition table and a separate functional update
path.  :func:`exhaustive_primitive_audit` compares both implementations over
every theory, world, and fixed-width primitive command word.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
from typing import Any


SCHEMA = "r12-ettr-il-v3-bounded-typed-local-rewrite-v1"
REGISTER_COUNT = 6
REGISTER_TYPES = (0, 1, 0, 1, 0, 1)
TYPE_COUNT = 2
SYMBOLS_PER_TYPE = 4
TYPE_SYMBOLS = tuple(
    tuple(range(SYMBOLS_PER_TYPE))
    for _ in range(TYPE_COUNT)
)
WORLD_COUNT = SYMBOLS_PER_TYPE**REGISTER_COUNT
RUNTIME_SLOT_LIMIT = 16
TRANSACTION_LIMIT = 64
MIN_DEPTH = 1
MAX_DEPTH = 6

LAW_COUNT = 6
THEORY_LAW_COUNT = 2
THEORY_COUNT = 15
VALID_LOCAL_SITE_COUNT = REGISTER_COUNT - 1

# The command representation reserves two bits for each opaque law slot and
# three bits for each local site.  Values outside the semantic sub-domain are
# retained as deterministic rejection controls.
OPAQUE_LAW_SLOT_CARDINALITY = 4
LOCAL_SITE_CARDINALITY = 8
DIRECTION_CARDINALITY = 2
PRIMITIVE_OPERATION_COUNT = (
    OPAQUE_LAW_SLOT_CARDINALITY
    * LOCAL_SITE_CARDINALITY
    * DIRECTION_CARDINALITY
)
VALID_PRIMITIVE_OPERATION_COUNT = (
    THEORY_LAW_COUNT
    * VALID_LOCAL_SITE_COUNT
    * DIRECTION_CARDINALITY
)

if REGISTER_COUNT > RUNTIME_SLOT_LIMIT:
    raise RuntimeError("v3 rewrite registers exceed the frozen runtime slots")
if MAX_DEPTH > TRANSACTION_LIMIT:
    raise RuntimeError("v3 rewrite depth exceeds the frozen transaction limit")
if WORLD_COUNT != 4096:
    raise RuntimeError("v3 rewrite world cardinality differs")
if THEORY_COUNT != 15:
    raise RuntimeError("v3 rewrite theory cardinality differs")


class RewriteSemanticError(ValueError):
    """A v3 rewrite object or query is malformed."""


class RewriteAdmissionError(RewriteSemanticError):
    """A well-formed object has no admissible Boolean denotation."""


class PrimitiveAuditError(RuntimeError):
    """Primary and replay semantics disagree during exhaustive audit."""


class Direction(StrEnum):
    FORWARD = "forward"
    REVERSE = "reverse"


class StepOutcome(StrEnum):
    APPLIED = "applied"
    BLOCKED = "blocked"
    REJECTED = "rejected"


class TerminalDisposition(StrEnum):
    ANSWER = "answer"
    REJECT = "reject"


class RejectReason(StrEnum):
    NONE = "none"
    OPAQUE_LAW_SLOT = "opaque_law_slot"
    LOCAL_SITE = "local_site"


class QueryOp(StrEnum):
    SLOT_IS = "slot_is"
    TYPE_COUNT_GE = "type_count_ge"
    ADJACENT_IS = "adjacent_is"
    PATTERN_EXISTS = "pattern_exists"
    SAME_TYPE_SLOTS_EQUAL = "same_type_slots_equal"
    SLOT_CHANGED = "slot_changed"


def _require_exact_int(
    value: object,
    name: str,
    lower: int,
    upper: int,
) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise RewriteSemanticError(f"{name} differs")
    return value


def _require_exact_tuple(value: object, name: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise RewriteSemanticError(f"{name} is not an exact tuple")
    return value


def _validate_registers(
    registers: object,
    name: str,
) -> tuple[int, ...]:
    values = _require_exact_tuple(registers, name)
    if len(values) != REGISTER_COUNT:
        raise RewriteSemanticError(f"{name} width differs")
    for index, value in enumerate(values):
        type_index = REGISTER_TYPES[index]
        if value not in TYPE_SYMBOLS[type_index] or type(value) is not int:
            raise RewriteSemanticError(f"{name} symbol differs")
    return values


def _validate_pair(pair: object, name: str) -> tuple[int, int]:
    values = _require_exact_tuple(pair, name)
    if len(values) != 2:
        raise RewriteSemanticError(f"{name} width differs")
    left = _require_exact_int(
        values[0],
        f"{name} left symbol",
        0,
        SYMBOLS_PER_TYPE - 1,
    )
    right = _require_exact_int(
        values[1],
        f"{name} right symbol",
        0,
        SYMBOLS_PER_TYPE - 1,
    )
    return left, right


def _strict_json_value(value: object, name: str = "canonical JSON") -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is list:
        for item in value:
            _strict_json_value(item, name)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise RewriteSemanticError(f"{name} key is not a string")
            _strict_json_value(item, name)
        return
    raise RewriteSemanticError(f"{name} contains a noncanonical value")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one strict JSON value with deterministic bytes and final LF."""

    _strict_json_value(value)
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RewriteSemanticError("canonical JSON rendering failed") from exc
    return rendered.encode("ascii") + b"\n"


def canonical_sha256(value: object) -> str:
    """Return lowercase SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, order=True, slots=True)
class LocalLaw:
    """One reversible, type-preserving rewrite over adjacent register values."""

    index: int
    forward_source: tuple[int, int]
    forward_target: tuple[int, int]

    def __post_init__(self) -> None:
        _require_exact_int(self.index, "law index", 0, LAW_COUNT - 1)
        source = _validate_pair(self.forward_source, "law source")
        target = _validate_pair(self.forward_target, "law target")
        if source == target:
            raise RewriteSemanticError("law does not change local state")

    def to_value(self) -> dict[str, object]:
        return {
            "forward_source": list(self.forward_source),
            "forward_target": list(self.forward_target),
            "index": self.index,
            "schema": f"{SCHEMA}/law",
        }


# No pair appears twice on either side.  Each rule is polymorphic over the two
# alternating site orientations: values change, while register types and
# register positions remain fixed.
LOCAL_LAWS = (
    LocalLaw(0, (0, 0), (1, 1)),
    LocalLaw(1, (0, 1), (2, 3)),
    LocalLaw(2, (0, 2), (3, 0)),
    LocalLaw(3, (1, 0), (2, 2)),
    LocalLaw(4, (1, 2), (3, 3)),
    LocalLaw(5, (2, 1), (3, 2)),
)


@dataclass(frozen=True, order=True, slots=True)
class RewriteTheory:
    """A theory binds two global laws to opaque local slots zero and one."""

    index: int
    law_indices: tuple[int, int]

    def __post_init__(self) -> None:
        _require_exact_int(self.index, "theory index", 0, THEORY_COUNT - 1)
        law_indices = _require_exact_tuple(
            self.law_indices,
            "theory law indices",
        )
        if len(law_indices) != THEORY_LAW_COUNT:
            raise RewriteSemanticError("theory law count differs")
        for law_index in law_indices:
            _require_exact_int(
                law_index,
                "theory law index",
                0,
                LAW_COUNT - 1,
            )
        if tuple(sorted(set(law_indices))) != law_indices:
            raise RewriteSemanticError(
                "theory law indices are not sorted unique"
            )

    def to_value(self) -> dict[str, object]:
        return {
            "index": self.index,
            "law_indices": list(self.law_indices),
            "schema": f"{SCHEMA}/theory",
        }


THEORIES = tuple(
    RewriteTheory(index, pair)
    for index, pair in enumerate(combinations(range(LAW_COUNT), 2))
)
if len(THEORIES) != THEORY_COUNT:
    raise RuntimeError("v3 rewrite theory construction differs")


@dataclass(frozen=True, order=True, slots=True)
class RewriteWorld:
    theory_index: int
    registers: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        _require_exact_int(
            self.theory_index,
            "world theory index",
            0,
            THEORY_COUNT - 1,
        )
        _validate_registers(self.registers, "world registers")

    @property
    def index(self) -> int:
        return world_index(self.registers)

    def to_value(self) -> dict[str, object]:
        return {
            "registers": list(self.registers),
            "schema": f"{SCHEMA}/world",
            "theory_index": self.theory_index,
            "world_index": self.index,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_value())

    def sha256(self) -> str:
        return canonical_sha256(self.to_value())


@dataclass(frozen=True, order=True, slots=True)
class LocalOperation:
    """One fixed-width command word.

    Opaque law slots two and three and local sites five through seven are
    syntactically representable but semantically invalid.  The executors turn
    them into deterministic rejection outcomes instead of coercing them.
    """

    law_slot: int
    site: int
    direction: Direction

    def __post_init__(self) -> None:
        _require_exact_int(
            self.law_slot,
            "operation law slot",
            0,
            OPAQUE_LAW_SLOT_CARDINALITY - 1,
        )
        _require_exact_int(
            self.site,
            "operation site",
            0,
            LOCAL_SITE_CARDINALITY - 1,
        )
        if type(self.direction) is not Direction:
            raise RewriteSemanticError("operation direction differs")

    @property
    def semantically_valid(self) -> bool:
        return (
            self.law_slot < THEORY_LAW_COUNT
            and self.site < VALID_LOCAL_SITE_COUNT
        )

    def to_value(self) -> dict[str, object]:
        return {
            "direction": self.direction.value,
            "law_slot": self.law_slot,
            "schema": f"{SCHEMA}/operation",
            "site": self.site,
        }


@dataclass(frozen=True, slots=True)
class RewriteCommand:
    operations: tuple[LocalOperation, ...]

    def __post_init__(self) -> None:
        operations = _require_exact_tuple(
            self.operations,
            "command operations",
        )
        if not MIN_DEPTH <= len(operations) <= MAX_DEPTH:
            raise RewriteSemanticError("command depth differs")
        if any(type(operation) is not LocalOperation for operation in operations):
            raise RewriteSemanticError("command operation type differs")

    @property
    def depth(self) -> int:
        return len(self.operations)

    def to_value(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "operations": [
                operation.to_value()
                for operation in self.operations
            ],
            "schema": f"{SCHEMA}/command",
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_value())

    def sha256(self) -> str:
        return canonical_sha256(self.to_value())


@dataclass(frozen=True, slots=True)
class RewriteStep:
    index: int
    operation: LocalOperation
    resolved_law_index: int
    before: tuple[int, int, int, int, int, int]
    after: tuple[int, int, int, int, int, int]
    outcome: StepOutcome
    reject_reason: RejectReason

    def to_value(self) -> dict[str, object]:
        return {
            "after": list(self.after),
            "before": list(self.before),
            "index": self.index,
            "operation": self.operation.to_value(),
            "outcome": self.outcome.value,
            "reject_reason": self.reject_reason.value,
            "resolved_law_index": self.resolved_law_index,
            "schema": f"{SCHEMA}/step",
        }


@dataclass(frozen=True, slots=True)
class RewriteExecution:
    world: RewriteWorld
    command: RewriteCommand
    snapshots: tuple[
        tuple[int, int, int, int, int, int],
        ...,
    ]
    steps: tuple[RewriteStep, ...]
    disposition: TerminalDisposition

    @property
    def terminal(self) -> tuple[int, int, int, int, int, int]:
        return self.snapshots[-1]

    @property
    def applied_count(self) -> int:
        return sum(
            step.outcome is StepOutcome.APPLIED
            for step in self.steps
        )

    @property
    def blocked_count(self) -> int:
        return sum(
            step.outcome is StepOutcome.BLOCKED
            for step in self.steps
        )

    def to_value(self) -> dict[str, object]:
        return {
            "command": self.command.to_value(),
            "disposition": self.disposition.value,
            "schema": f"{SCHEMA}/execution",
            "snapshots": [list(snapshot) for snapshot in self.snapshots],
            "steps": [step.to_value() for step in self.steps],
            "world": self.world.to_value(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_value())

    def sha256(self) -> str:
        return canonical_sha256(self.to_value())


def world_index(registers: tuple[int, ...]) -> int:
    """Return the canonical base-four index for one register state."""

    values = _validate_registers(registers, "registers")
    result = 0
    for value in values:
        result = result * SYMBOLS_PER_TYPE + value
    return result


def world_from_index(index: int, theory_index: int = 0) -> RewriteWorld:
    """Decode one canonical world index for a selected theory."""

    remaining = _require_exact_int(
        index,
        "world index",
        0,
        WORLD_COUNT - 1,
    )
    _require_exact_int(
        theory_index,
        "world theory index",
        0,
        THEORY_COUNT - 1,
    )
    digits = [0] * REGISTER_COUNT
    for position in range(REGISTER_COUNT - 1, -1, -1):
        digits[position] = remaining % SYMBOLS_PER_TYPE
        remaining //= SYMBOLS_PER_TYPE
    return RewriteWorld(theory_index, tuple(digits))  # type: ignore[arg-type]


def iter_worlds(theory_index: int = 0):
    """Yield all 4,096 worlds in canonical base-four order."""

    _require_exact_int(
        theory_index,
        "world theory index",
        0,
        THEORY_COUNT - 1,
    )
    for registers in product(
        range(SYMBOLS_PER_TYPE),
        repeat=REGISTER_COUNT,
    ):
        yield RewriteWorld(theory_index, registers)


@lru_cache(maxsize=2)
def primitive_operations(
    include_reject_controls: bool = True,
) -> tuple[LocalOperation, ...]:
    """Return primitive words in canonical slot/site/direction order."""

    if type(include_reject_controls) is not bool:
        raise RewriteSemanticError("reject-control flag differs")
    law_slots = (
        range(OPAQUE_LAW_SLOT_CARDINALITY)
        if include_reject_controls
        else range(THEORY_LAW_COUNT)
    )
    sites = (
        range(LOCAL_SITE_CARDINALITY)
        if include_reject_controls
        else range(VALID_LOCAL_SITE_COUNT)
    )
    directions = (Direction.FORWARD, Direction.REVERSE)
    return tuple(
        LocalOperation(law_slot, site, direction)
        for law_slot, site, direction in product(
            law_slots,
            sites,
            directions,
        )
    )


def _primary_step(
    theory: RewriteTheory,
    state: list[int],
    operation: LocalOperation,
    index: int,
) -> RewriteStep:
    before = tuple(state)
    if operation.law_slot >= THEORY_LAW_COUNT:
        return RewriteStep(
            index,
            operation,
            -1,
            before,
            before,
            StepOutcome.REJECTED,
            RejectReason.OPAQUE_LAW_SLOT,
        )
    if operation.site >= VALID_LOCAL_SITE_COUNT:
        return RewriteStep(
            index,
            operation,
            -1,
            before,
            before,
            StepOutcome.REJECTED,
            RejectReason.LOCAL_SITE,
        )

    law_index = theory.law_indices[operation.law_slot]
    law = LOCAL_LAWS[law_index]
    if operation.direction is Direction.FORWARD:
        source = law.forward_source
        target = law.forward_target
    else:
        source = law.forward_target
        target = law.forward_source
    observed = (state[operation.site], state[operation.site + 1])
    if observed != source:
        return RewriteStep(
            index,
            operation,
            law_index,
            before,
            before,
            StepOutcome.BLOCKED,
            RejectReason.NONE,
        )
    state[operation.site] = target[0]
    state[operation.site + 1] = target[1]
    return RewriteStep(
        index,
        operation,
        law_index,
        before,
        tuple(state),
        StepOutcome.APPLIED,
        RejectReason.NONE,
    )


def execute_primary(
    world: RewriteWorld,
    command: RewriteCommand,
) -> RewriteExecution:
    """Execute by directly interpreting the local-law objects."""

    if type(world) is not RewriteWorld or type(command) is not RewriteCommand:
        raise RewriteSemanticError("primary world or command type differs")
    theory = THEORIES[world.theory_index]
    state = list(world.registers)
    snapshots = [world.registers]
    steps: list[RewriteStep] = []
    disposition = TerminalDisposition.ANSWER
    for index, operation in enumerate(command.operations, start=1):
        step = _primary_step(theory, state, operation, index)
        steps.append(step)
        snapshots.append(step.after)
        if step.outcome is StepOutcome.REJECTED:
            disposition = TerminalDisposition.REJECT
            break
    return RewriteExecution(
        world,
        command,
        tuple(snapshots),
        tuple(steps),
        disposition,
    )


# Replay owns a separately encoded law table.  It does not call LocalLaw,
# _primary_step, execute_primary, or any primary transition helper.
_REPLAY_FORWARD = {
    0: ((0, 0), (1, 1)),
    1: ((0, 1), (2, 3)),
    2: ((0, 2), (3, 0)),
    3: ((1, 0), (2, 2)),
    4: ((1, 2), (3, 3)),
    5: ((2, 1), (3, 2)),
}
_REPLAY_REVERSE = {
    law_index: (target, source)
    for law_index, (source, target) in _REPLAY_FORWARD.items()
}


def execute_replay(
    world: RewriteWorld,
    command: RewriteCommand,
) -> RewriteExecution:
    """Independently replay with lookup transitions and tuple reconstruction."""

    if type(world) is not RewriteWorld or type(command) is not RewriteCommand:
        raise RewriteSemanticError("replay world or command type differs")
    state = world.registers
    snapshots = [state]
    steps: list[RewriteStep] = []
    disposition = TerminalDisposition.ANSWER
    law_slots = THEORIES[world.theory_index].law_indices
    for index, operation in enumerate(command.operations, start=1):
        before = state
        if operation.law_slot not in (0, 1):
            step = RewriteStep(
                index,
                operation,
                -1,
                before,
                before,
                StepOutcome.REJECTED,
                RejectReason.OPAQUE_LAW_SLOT,
            )
        elif operation.site not in (0, 1, 2, 3, 4):
            step = RewriteStep(
                index,
                operation,
                -1,
                before,
                before,
                StepOutcome.REJECTED,
                RejectReason.LOCAL_SITE,
            )
        else:
            law_index = law_slots[operation.law_slot]
            table = (
                _REPLAY_FORWARD
                if operation.direction == Direction.FORWARD
                else _REPLAY_REVERSE
            )
            source, target = table[law_index]
            observed = state[operation.site : operation.site + 2]
            if observed == source:
                state = (
                    state[: operation.site]
                    + target
                    + state[operation.site + 2 :]
                )
                outcome = StepOutcome.APPLIED
            else:
                outcome = StepOutcome.BLOCKED
            step = RewriteStep(
                index,
                operation,
                law_index,
                before,
                state,
                outcome,
                RejectReason.NONE,
            )
        steps.append(step)
        snapshots.append(step.after)
        state = step.after
        if step.outcome == StepOutcome.REJECTED:
            disposition = TerminalDisposition.REJECT
            break
    return RewriteExecution(
        world,
        command,
        tuple(snapshots),
        tuple(steps),
        disposition,
    )


@dataclass(frozen=True, order=True, slots=True)
class StructuralQuery:
    op: QueryOp
    arguments: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.op) is not QueryOp:
            raise RewriteSemanticError("query operation differs")
        arguments = _require_exact_tuple(
            self.arguments,
            "query arguments",
        )
        if self.op is QueryOp.SLOT_IS:
            if len(arguments) != 2:
                raise RewriteSemanticError("slot-is query arity differs")
            _require_exact_int(
                arguments[0],
                "slot-is slot",
                0,
                REGISTER_COUNT - 1,
            )
            _require_exact_int(
                arguments[1],
                "slot-is symbol",
                0,
                SYMBOLS_PER_TYPE - 1,
            )
        elif self.op is QueryOp.TYPE_COUNT_GE:
            if len(arguments) != 3:
                raise RewriteSemanticError("type-count query arity differs")
            _require_exact_int(
                arguments[0],
                "type-count type",
                0,
                TYPE_COUNT - 1,
            )
            _require_exact_int(
                arguments[1],
                "type-count symbol",
                0,
                SYMBOLS_PER_TYPE - 1,
            )
            _require_exact_int(
                arguments[2],
                "type-count threshold",
                1,
                REGISTER_COUNT // TYPE_COUNT,
            )
        elif self.op is QueryOp.ADJACENT_IS:
            if len(arguments) != 3:
                raise RewriteSemanticError("adjacent query arity differs")
            _require_exact_int(
                arguments[0],
                "adjacent site",
                0,
                VALID_LOCAL_SITE_COUNT - 1,
            )
            _require_exact_int(
                arguments[1],
                "adjacent left symbol",
                0,
                SYMBOLS_PER_TYPE - 1,
            )
            _require_exact_int(
                arguments[2],
                "adjacent right symbol",
                0,
                SYMBOLS_PER_TYPE - 1,
            )
        elif self.op is QueryOp.PATTERN_EXISTS:
            if len(arguments) != 2:
                raise RewriteSemanticError("pattern query arity differs")
            _require_exact_int(
                arguments[0],
                "pattern left symbol",
                0,
                SYMBOLS_PER_TYPE - 1,
            )
            _require_exact_int(
                arguments[1],
                "pattern right symbol",
                0,
                SYMBOLS_PER_TYPE - 1,
            )
        elif self.op is QueryOp.SAME_TYPE_SLOTS_EQUAL:
            if len(arguments) != 2:
                raise RewriteSemanticError("slot-equality query arity differs")
            left = _require_exact_int(
                arguments[0],
                "slot-equality left slot",
                0,
                REGISTER_COUNT - 1,
            )
            right = _require_exact_int(
                arguments[1],
                "slot-equality right slot",
                0,
                REGISTER_COUNT - 1,
            )
            if left >= right or REGISTER_TYPES[left] != REGISTER_TYPES[right]:
                raise RewriteSemanticError(
                    "slot-equality query is not canonical same-type order"
                )
        elif self.op is QueryOp.SLOT_CHANGED:
            if len(arguments) != 1:
                raise RewriteSemanticError("slot-changed query arity differs")
            _require_exact_int(
                arguments[0],
                "slot-changed slot",
                0,
                REGISTER_COUNT - 1,
            )
        else:
            raise RewriteSemanticError("query operation is unsupported")

    def to_value(self) -> dict[str, object]:
        return {
            "arguments": list(self.arguments),
            "op": self.op.value,
            "schema": f"{SCHEMA}/query",
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_value())

    def sha256(self) -> str:
        return canonical_sha256(self.to_value())


@lru_cache(maxsize=1)
def structural_queries() -> tuple[StructuralQuery, ...]:
    """Enumerate the finite, nontrivially thresholded Boolean grammar."""

    queries: list[StructuralQuery] = []
    queries.extend(
        StructuralQuery(QueryOp.SLOT_IS, (slot, symbol))
        for slot, symbol in product(
            range(REGISTER_COUNT),
            range(SYMBOLS_PER_TYPE),
        )
    )
    queries.extend(
        StructuralQuery(QueryOp.TYPE_COUNT_GE, (type_index, symbol, threshold))
        for type_index, symbol, threshold in product(
            range(TYPE_COUNT),
            range(SYMBOLS_PER_TYPE),
            range(1, REGISTER_COUNT // TYPE_COUNT + 1),
        )
    )
    queries.extend(
        StructuralQuery(QueryOp.ADJACENT_IS, (site, left, right))
        for site, left, right in product(
            range(VALID_LOCAL_SITE_COUNT),
            range(SYMBOLS_PER_TYPE),
            range(SYMBOLS_PER_TYPE),
        )
    )
    queries.extend(
        StructuralQuery(QueryOp.PATTERN_EXISTS, (left, right))
        for left, right in product(
            range(SYMBOLS_PER_TYPE),
            range(SYMBOLS_PER_TYPE),
        )
    )
    for type_index in range(TYPE_COUNT):
        slots = tuple(
            slot
            for slot, register_type in enumerate(REGISTER_TYPES)
            if register_type == type_index
        )
        queries.extend(
            StructuralQuery(QueryOp.SAME_TYPE_SLOTS_EQUAL, pair)
            for pair in combinations(slots, 2)
        )
    queries.extend(
        StructuralQuery(QueryOp.SLOT_CHANGED, (slot,))
        for slot in range(REGISTER_COUNT)
    )
    return tuple(queries)


def _require_answer_execution(
    execution: RewriteExecution,
) -> RewriteExecution:
    if type(execution) is not RewriteExecution:
        raise RewriteSemanticError("query execution type differs")
    if execution.disposition is not TerminalDisposition.ANSWER:
        raise RewriteAdmissionError(
            "rejected execution has no Boolean query answer"
        )
    return execution


def evaluate_query_primary(
    execution: RewriteExecution,
    query: StructuralQuery,
) -> bool:
    """Evaluate one query with direct indexed state inspection."""

    item = _require_answer_execution(execution)
    if type(query) is not StructuralQuery:
        raise RewriteSemanticError("query type differs")
    terminal = item.terminal
    arguments = query.arguments
    if query.op is QueryOp.SLOT_IS:
        return terminal[arguments[0]] == arguments[1]
    if query.op is QueryOp.TYPE_COUNT_GE:
        type_index, symbol, threshold = arguments
        count = sum(
            terminal[slot] == symbol
            for slot in range(REGISTER_COUNT)
            if REGISTER_TYPES[slot] == type_index
        )
        return count >= threshold
    if query.op is QueryOp.ADJACENT_IS:
        site, left, right = arguments
        return terminal[site] == left and terminal[site + 1] == right
    if query.op is QueryOp.PATTERN_EXISTS:
        left, right = arguments
        return any(
            terminal[site] == left and terminal[site + 1] == right
            for site in range(VALID_LOCAL_SITE_COUNT)
        )
    if query.op is QueryOp.SAME_TYPE_SLOTS_EQUAL:
        return terminal[arguments[0]] == terminal[arguments[1]]
    if query.op is QueryOp.SLOT_CHANGED:
        slot = arguments[0]
        return terminal[slot] != item.world.registers[slot]
    raise RewriteSemanticError("query operation is unsupported")


def evaluate_query_replay(
    execution: RewriteExecution,
    query: StructuralQuery,
) -> bool:
    """Independently evaluate with tuple projections and pair membership."""

    item = _require_answer_execution(execution)
    if type(query) is not StructuralQuery:
        raise RewriteSemanticError("query type differs")
    terminal = item.terminal
    arguments = query.arguments
    if query.op == QueryOp.SLOT_IS:
        projected = tuple(enumerate(terminal))
        return (arguments[0], arguments[1]) in projected
    if query.op == QueryOp.TYPE_COUNT_GE:
        type_index, symbol, threshold = arguments
        projected = tuple(
            value
            for register_type, value in zip(
                REGISTER_TYPES,
                terminal,
                strict=True,
            )
            if register_type == type_index
        )
        return projected.count(symbol) >= threshold
    if query.op == QueryOp.ADJACENT_IS:
        site, left, right = arguments
        return terminal[site : site + 2] == (left, right)
    if query.op == QueryOp.PATTERN_EXISTS:
        pairs = tuple(
            zip(
                terminal[:-1],
                terminal[1:],
                strict=True,
            )
        )
        return arguments in pairs
    if query.op == QueryOp.SAME_TYPE_SLOTS_EQUAL:
        selected = tuple(terminal[slot] for slot in arguments)
        return len(set(selected)) == 1
    if query.op == QueryOp.SLOT_CHANGED:
        changes = tuple(
            before != after
            for before, after in zip(
                item.world.registers,
                terminal,
                strict=True,
            )
        )
        return changes[arguments[0]]
    raise RewriteSemanticError("query operation is unsupported")


@dataclass(frozen=True, slots=True)
class PrimitiveAuditReceipt:
    schema: str
    world_count: int
    theory_count: int
    theory_world_count: int
    primitive_operation_count: int
    valid_primitive_operation_count: int
    case_count: int
    applied_count: int
    blocked_count: int
    rejected_count: int
    invalid_law_slot_rejected_count: int
    invalid_site_rejected_count: int
    primary_replay_mismatch_count: int
    minimum_applied_support_per_valid_theory_operation: int
    maximum_applied_support_per_valid_theory_operation: int
    structural_query_count: int
    max_transactions: int

    def to_value(self) -> dict[str, object]:
        return {
            "applied_count": self.applied_count,
            "blocked_count": self.blocked_count,
            "case_count": self.case_count,
            "invalid_law_slot_rejected_count": (
                self.invalid_law_slot_rejected_count
            ),
            "invalid_site_rejected_count": (
                self.invalid_site_rejected_count
            ),
            "max_transactions": self.max_transactions,
            "maximum_applied_support_per_valid_theory_operation": (
                self.maximum_applied_support_per_valid_theory_operation
            ),
            "minimum_applied_support_per_valid_theory_operation": (
                self.minimum_applied_support_per_valid_theory_operation
            ),
            "primary_replay_mismatch_count": (
                self.primary_replay_mismatch_count
            ),
            "primitive_operation_count": self.primitive_operation_count,
            "rejected_count": self.rejected_count,
            "schema": self.schema,
            "structural_query_count": self.structural_query_count,
            "theory_count": self.theory_count,
            "theory_world_count": self.theory_world_count,
            "valid_primitive_operation_count": (
                self.valid_primitive_operation_count
            ),
            "world_count": self.world_count,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_value())

    def sha256(self) -> str:
        return canonical_sha256(self.to_value())


@lru_cache(maxsize=1)
def exhaustive_primitive_audit() -> PrimitiveAuditReceipt:
    """Compare both executors over all 3,932,160 primitive cases.

    The audit spans all 15 theories, all 4,096 worlds, and all 64 fixed-width
    command words.  It therefore covers applied, blocked, invalid-law-slot,
    and invalid-site behavior exhaustively.
    """

    valid_ranks = {
        operation: rank
        for rank, operation in enumerate(primitive_operations(False))
    }
    commands = tuple(
        (
            RewriteCommand((operation,)),
            valid_ranks.get(operation, -1),
        )
        for operation in primitive_operations()
    )
    applied = 0
    blocked = 0
    rejected = 0
    invalid_law_slot_rejected = 0
    invalid_site_rejected = 0
    cases = 0
    max_transactions = 0
    applied_support = [
        [0] * VALID_PRIMITIVE_OPERATION_COUNT
        for _ in range(THEORY_COUNT)
    ]
    for theory_index in range(THEORY_COUNT):
        for world in iter_worlds(theory_index):
            for command, valid_rank in commands:
                primary = execute_primary(world, command)
                replay = execute_replay(world, command)
                cases += 1
                if primary != replay:
                    raise PrimitiveAuditError(
                        "primary/replay mismatch at "
                        f"theory={theory_index},world={world.index},"
                        f"command={command.sha256()}"
                    )
                max_transactions = max(
                    max_transactions,
                    len(primary.steps),
                )
                outcome = primary.steps[0].outcome
                if outcome is StepOutcome.APPLIED:
                    applied += 1
                    if valid_rank < 0:
                        raise PrimitiveAuditError(
                            "reject-control command applied"
                        )
                    applied_support[theory_index][valid_rank] += 1
                elif outcome is StepOutcome.BLOCKED:
                    blocked += 1
                elif outcome is StepOutcome.REJECTED:
                    rejected += 1
                    reason = primary.steps[0].reject_reason
                    if reason is RejectReason.OPAQUE_LAW_SLOT:
                        invalid_law_slot_rejected += 1
                    elif reason is RejectReason.LOCAL_SITE:
                        invalid_site_rejected += 1
                    else:
                        raise PrimitiveAuditError(
                            "primitive rejection reason differs"
                        )
                else:
                    raise PrimitiveAuditError("unknown primitive outcome")

    expected_cases = THEORY_COUNT * WORLD_COUNT * PRIMITIVE_OPERATION_COUNT
    if cases != expected_cases:
        raise PrimitiveAuditError("primitive audit case count differs")
    if applied + blocked + rejected != cases:
        raise PrimitiveAuditError("primitive audit outcome count differs")
    support_values = tuple(
        support
        for theory_support in applied_support
        for support in theory_support
    )
    if len(support_values) != (
        THEORY_COUNT * VALID_PRIMITIVE_OPERATION_COUNT
    ):
        raise PrimitiveAuditError("primitive support cell count differs")
    if invalid_law_slot_rejected + invalid_site_rejected != rejected:
        raise PrimitiveAuditError("primitive rejection partition differs")
    return PrimitiveAuditReceipt(
        schema=f"{SCHEMA}/primitive-audit",
        world_count=WORLD_COUNT,
        theory_count=THEORY_COUNT,
        theory_world_count=THEORY_COUNT * WORLD_COUNT,
        primitive_operation_count=PRIMITIVE_OPERATION_COUNT,
        valid_primitive_operation_count=VALID_PRIMITIVE_OPERATION_COUNT,
        case_count=cases,
        applied_count=applied,
        blocked_count=blocked,
        rejected_count=rejected,
        invalid_law_slot_rejected_count=invalid_law_slot_rejected,
        invalid_site_rejected_count=invalid_site_rejected,
        primary_replay_mismatch_count=0,
        minimum_applied_support_per_valid_theory_operation=min(
            support_values
        ),
        maximum_applied_support_per_valid_theory_operation=max(
            support_values
        ),
        structural_query_count=len(structural_queries()),
        max_transactions=max_transactions,
    )


__all__ = [
    "DIRECTION_CARDINALITY",
    "Direction",
    "LAW_COUNT",
    "LOCAL_LAWS",
    "LOCAL_SITE_CARDINALITY",
    "LocalLaw",
    "LocalOperation",
    "MAX_DEPTH",
    "MIN_DEPTH",
    "OPAQUE_LAW_SLOT_CARDINALITY",
    "PRIMITIVE_OPERATION_COUNT",
    "PrimitiveAuditError",
    "PrimitiveAuditReceipt",
    "QueryOp",
    "REGISTER_COUNT",
    "REGISTER_TYPES",
    "RUNTIME_SLOT_LIMIT",
    "RejectReason",
    "RewriteAdmissionError",
    "RewriteCommand",
    "RewriteExecution",
    "RewriteSemanticError",
    "RewriteStep",
    "RewriteTheory",
    "RewriteWorld",
    "SCHEMA",
    "SYMBOLS_PER_TYPE",
    "StepOutcome",
    "StructuralQuery",
    "THEORIES",
    "THEORY_COUNT",
    "THEORY_LAW_COUNT",
    "TRANSACTION_LIMIT",
    "TYPE_COUNT",
    "TYPE_SYMBOLS",
    "TerminalDisposition",
    "VALID_LOCAL_SITE_COUNT",
    "VALID_PRIMITIVE_OPERATION_COUNT",
    "WORLD_COUNT",
    "canonical_json_bytes",
    "canonical_sha256",
    "evaluate_query_primary",
    "evaluate_query_replay",
    "exhaustive_primitive_audit",
    "execute_primary",
    "execute_replay",
    "iter_worlds",
    "primitive_operations",
    "structural_queries",
    "world_from_index",
    "world_index",
]
