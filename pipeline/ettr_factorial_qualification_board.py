"""Frozen staged qualification board for the ETTR architecture.

The board is evaluation-only.  It materializes three exact
``WORLD -> COMMAND -> QUERY`` rectangles without training a model:

* WORLD contains laws and an initial state.
* COMMAND is disclosed only after WORLD has been compiled and deleted.
* QUERY is disclosed only after COMMAND execution has committed.

Every ontology contributes one 2x2 WORLD/COMMAND rectangle.  Each of the
twelve terminal packets receives two semantic queries through two distinct
raw paraphrases, producing 48 scored rows.  Primary and independently
implemented oracles must agree before a row is admitted.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from functools import lru_cache
import hashlib
import json
from typing import Any

from audit_cross_ontology_horn_board import independent_closure
from audit_cross_ontology_resource_board import independent_execute_sequence
from audit_cross_ontology_rewrite_board import independent_normal_forms
from cross_ontology_horn_board import (
    OBJECT_TYPES,
    PREDICATES,
    RULE_LIBRARY as HORN_RULE_LIBRARY,
    THEORIES as HORN_THEORIES,
    GroundAtom,
    execute_closure,
)
from cross_ontology_resource_board import (
    OPERATOR_LIBRARY,
    PLACE_SPECS,
    THEORIES as RESOURCE_THEORIES,
    execute_sequence,
    heldout_programs,
    input_markings,
)
from cross_ontology_rewrite_board import (
    CONSTRUCTORS,
    RULE_LIBRARY as REWRITE_RULE_LIBRARY,
    THEORIES as REWRITE_THEORIES,
    GroundTerm,
    PatternTerm,
    challenge_terms,
    execute_normal_forms,
)


BOARD_SCHEMA = "ettr-factorial-qualification-board-v1"
WORLD_SCHEMA = "ettr-factorial-world-v1"
COMMAND_SCHEMA = "ettr-factorial-command-v1"
QUERY_SCHEMA = "ettr-factorial-late-query-v1"
PACKAGE_SCHEMA = "ettr-factorial-stage-package-v1"
CLAIM_BOUNDARY = (
    "Frozen evaluation mechanics and source-deleted qualification geometry "
    "only; this is not training, learned capability, native reasoning, or a "
    "general-reasoning claim."
)
FOLDS = 3
WORLDS_PER_FOLD = 2
COMMANDS_PER_FOLD = 2
PACKETS_PER_FOLD = WORLDS_PER_FOLD * COMMANDS_PER_FOLD
SEMANTICS_PER_PACKET = 2
PARAPHRASES_PER_SEMANTIC = 2
TOTAL_PACKETS = FOLDS * PACKETS_PER_FOLD
TOTAL_ROWS = TOTAL_PACKETS * SEMANTICS_PER_PACKET * PARAPHRASES_PER_SEMANTIC

_FORBIDDEN_CANDIDATE_TOKENS = (
    b"answer",
    b"expected",
    b"family",
    b"horn",
    b"label",
    b"oracle",
    b"resource",
    b"rewrite",
    b"target",
    b"theory_index",
)


class QualificationFold(StrEnum):
    """Assessor-only names; candidate payloads use numeric domain codes."""

    HORN_CLOSURE = "horn_closure"
    TYPED_REWRITE = "typed_rewrite"
    GUARDED_RESOURCE = "guarded_resource"


FOLD_ORDER = tuple(QualificationFold)


@dataclass(frozen=True, slots=True)
class SemanticProbe:
    """One late, assessor-interpreted predicate over a terminal result."""

    code: int
    arguments: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QualificationRow:
    fold: QualificationFold
    world_index: int
    command_index: int
    semantic_index: int
    paraphrase_index: int
    world_bytes: bytes
    command_bytes: bytes
    query_prefix_bytes: bytes
    target: bool
    world_factor_id: str
    command_factor_id: str
    packet_factor_id: str
    query_semantic_id: str
    query_paraphrase_id: str
    row_id: str


@dataclass(frozen=True, slots=True)
class FactorialQualificationReceipt:
    schema: str
    fold_count: int
    world_count: int
    command_count: int
    packet_count: int
    row_count: int
    independent_oracle_agreement_count: int
    world_edge_change_count: int
    command_edge_change_count: int
    within_packet_target_contrast_count: int
    candidate_label_leak_count: int
    unique_row_count: int
    world_package_sha256: str
    command_package_sha256: str
    query_package_sha256: str
    assessor_package_sha256: str
    payload_sha256: str
    claim_boundary: str
    all_contracts_pass: bool


@dataclass(frozen=True, slots=True)
class ETTRFactorialQualificationBoard:
    rows: tuple[QualificationRow, ...]
    receipt: FactorialQualificationReceipt

    @property
    def packet_factor_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.packet_factor_id for row in self.rows))

    @property
    def world_factor_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.world_factor_id for row in self.rows))

    @property
    def command_factor_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.command_factor_id for row in self.rows))

    def world_package_bytes(self) -> bytes:
        entries = []
        seen: set[str] = set()
        for row in self.rows:
            if row.world_factor_id in seen:
                continue
            seen.add(row.world_factor_id)
            entries.append(
                {
                    "world_factor_id": row.world_factor_id,
                    "world_sha256": _sha256(row.world_bytes),
                    "world_hex": row.world_bytes.hex(),
                }
            )
        return _canonical_bytes(
            {
                "schema": PACKAGE_SCHEMA,
                "stage": 1,
                "board_sha256": self.receipt.payload_sha256,
                "entries": entries,
            }
        )

    def command_package_bytes(self) -> bytes:
        entries = []
        seen: set[str] = set()
        for row in self.rows:
            if row.packet_factor_id in seen:
                continue
            seen.add(row.packet_factor_id)
            entries.append(
                {
                    "packet_factor_id": row.packet_factor_id,
                    "world_factor_id": row.world_factor_id,
                    "command_factor_id": row.command_factor_id,
                    "command_sha256": _sha256(row.command_bytes),
                    "command_hex": row.command_bytes.hex(),
                }
            )
        return _canonical_bytes(
            {
                "schema": PACKAGE_SCHEMA,
                "stage": 2,
                "board_sha256": self.receipt.payload_sha256,
                "entries": entries,
            }
        )

    def query_package_bytes(self) -> bytes:
        return _canonical_bytes(
            {
                "schema": PACKAGE_SCHEMA,
                "stage": 3,
                "board_sha256": self.receipt.payload_sha256,
                "entries": [
                    {
                        "row_id": row.row_id,
                        "packet_factor_id": row.packet_factor_id,
                        "query_sha256": _sha256(row.query_prefix_bytes),
                        "query_hex": row.query_prefix_bytes.hex(),
                    }
                    for row in self.rows
                ],
            }
        )

    def assessor_package_bytes(self) -> bytes:
        return _canonical_bytes(
            {
                "schema": PACKAGE_SCHEMA,
                "stage": 4,
                "board_sha256": self.receipt.payload_sha256,
                "entries": [
                    {
                        "row_id": row.row_id,
                        "query_semantic_id": row.query_semantic_id,
                        "query_paraphrase_id": row.query_paraphrase_id,
                        "target": int(row.target),
                    }
                    for row in self.rows
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class _FoldDefinition:
    fold: QualificationFold
    worlds: tuple[Any, Any]
    commands: tuple[Any, Any]
    probes: tuple[SemanticProbe, SemanticProbe]


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _canonicalize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    if isinstance(value, tuple | list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None or isinstance(value, bool | int | str):
        return value
    raise TypeError(f"unsupported qualification value {type(value)!r}")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonicalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _atom_payload(atom: GroundAtom) -> list[Any]:
    return [atom.predicate, list(atom.arguments)]


def _pattern_payload(pattern: PatternTerm) -> list[Any]:
    if pattern.variable_index is not None:
        return [pattern.type_index, "v", pattern.variable_index]
    return [
        pattern.type_index,
        "c",
        pattern.constructor_index,
        [_pattern_payload(child) for child in pattern.children],
    ]


def _term_payload(term: GroundTerm) -> list[Any]:
    return [
        term.type_index,
        term.constructor_index,
        [_term_payload(child) for child in term.children],
    ]


def _quantity_payload(quantities: tuple[Any, ...]) -> list[Any]:
    return [[item.place, item.resource_kind, item.multiplicity] for item in quantities]


def _world_bytes(fold: QualificationFold, world: Any) -> bytes:
    if fold == QualificationFold.HORN_CLOSURE:
        theory_index, initial = world
        theory = HORN_THEORIES[theory_index]
        body = {
            "d": 0,
            "objects": list(OBJECT_TYPES),
            "predicates": [
                [item.index, list(item.argument_types)] for item in PREDICATES
            ],
            "laws": [
                [
                    [_atom_pattern_payload(item) for item in rule.premises],
                    _atom_pattern_payload(rule.conclusion),
                ]
                for rule_index in theory.rule_indices
                for rule in (HORN_RULE_LIBRARY[rule_index],)
            ],
            "state": [_atom_payload(atom) for atom in initial],
        }
    elif fold == QualificationFold.TYPED_REWRITE:
        theory_index, initial = world
        theory = REWRITE_THEORIES[theory_index]
        body = {
            "d": 1,
            "constructors": [
                [
                    item.index,
                    item.result_type,
                    list(item.argument_types),
                ]
                for item in CONSTRUCTORS
            ],
            "laws": [
                [
                    _pattern_payload(rule.lhs),
                    _pattern_payload(rule.rhs),
                ]
                for rule_index in theory.rule_indices
                for rule in (REWRITE_RULE_LIBRARY[rule_index],)
            ],
            "state": _term_payload(initial),
        }
    else:
        theory_index, initial = world
        theory = RESOURCE_THEORIES[theory_index]
        body = {
            "d": 2,
            "places": [
                [item.index, item.resource_kind, item.capacity] for item in PLACE_SPECS
            ],
            "laws": [
                [
                    _quantity_payload(operator.guards),
                    _quantity_payload(operator.consumes),
                    _quantity_payload(operator.produces),
                ]
                for operator_index in theory.operator_indices
                for operator in (OPERATOR_LIBRARY[operator_index],)
            ],
            "state": list(initial.multiplicities),
        }
    return _canonical_bytes({"schema": WORLD_SCHEMA, "world": body})


def _atom_pattern_payload(pattern: Any) -> list[Any]:
    return [pattern.predicate, list(pattern.variables)]


def _command_bytes(fold: QualificationFold, command: Any) -> bytes:
    if fold == QualificationFold.HORN_CLOSURE:
        body = {"d": 0, "op": 0, "args": _atom_payload(command)}
    elif fold == QualificationFold.TYPED_REWRITE:
        body = {"d": 1, "op": 1, "args": [5, command]}
    else:
        body = {"d": 2, "op": 2, "args": list(command)}
    return _canonical_bytes({"schema": COMMAND_SCHEMA, "command": body})


def _query_prefix_bytes(
    fold: QualificationFold,
    probe: SemanticProbe,
    paraphrase: int,
) -> bytes:
    domain_code = FOLD_ORDER.index(fold)
    if paraphrase == 0:
        body = {
            "d": domain_code,
            "probe": {
                "code": probe.code,
                "args": list(probe.arguments),
            },
            "codebook": [0, 1],
        }
    elif paraphrase == 1:
        body = {
            "d": domain_code,
            "read": [
                probe.code,
                list(probe.arguments),
            ],
            "values": {"false": 0, "true": 1},
        }
    else:
        raise ValueError("qualification paraphrase differs")
    return _canonical_bytes({"schema": QUERY_SCHEMA, "query": body}) + b"R="


def _execute_primary(
    fold: QualificationFold,
    world: Any,
    command: Any,
) -> Any:
    if fold == QualificationFold.HORN_CLOSURE:
        theory_index, initial = world
        return execute_closure(
            HORN_THEORIES[theory_index],
            tuple((*initial, command)),
        )
    if fold == QualificationFold.TYPED_REWRITE:
        theory_index, initial = world
        commanded = GroundTerm(
            type_index=0,
            constructor_index=5,
            children=(
                initial,
                GroundTerm(type_index=0, constructor_index=command),
            ),
        )
        normal_forms = execute_normal_forms(theory_index, commanded)
        if len(normal_forms) != 1:
            raise ValueError("qualification rewrite command is nonconfluent")
        return normal_forms[0]
    theory_index, initial = world
    return execute_sequence(
        RESOURCE_THEORIES[theory_index],
        initial,
        command,
    )


def _execute_independent(
    fold: QualificationFold,
    world: Any,
    command: Any,
) -> Any:
    if fold == QualificationFold.HORN_CLOSURE:
        theory_index, initial = world
        return independent_closure(
            HORN_THEORIES[theory_index],
            tuple((*initial, command)),
        )
    if fold == QualificationFold.TYPED_REWRITE:
        theory_index, initial = world
        commanded = GroundTerm(
            type_index=0,
            constructor_index=5,
            children=(
                initial,
                GroundTerm(type_index=0, constructor_index=command),
            ),
        )
        normal_forms = independent_normal_forms(theory_index, commanded)
        if len(normal_forms) != 1:
            raise ValueError(
                "independent qualification rewrite command is nonconfluent"
            )
        return normal_forms[0]
    theory_index, initial = world
    return independent_execute_sequence(
        RESOURCE_THEORIES[theory_index],
        initial,
        command,
    )


def _probe_result(
    fold: QualificationFold,
    result: Any,
    probe: SemanticProbe,
) -> bool:
    if fold == QualificationFold.HORN_CLOSURE:
        if probe.code == 0:
            atom = GroundAtom(probe.arguments[0], probe.arguments[1:])
            return atom in result
        if probe.code == 1:
            return len(result) < probe.arguments[0]
    elif fold == QualificationFold.TYPED_REWRITE:
        if probe.code == 0:
            return result.constructor_index == probe.arguments[0]
        if probe.code == 1:
            return len(result.children) < probe.arguments[0]
    else:
        if probe.code == 0:
            return result.cursor >= probe.arguments[0]
        if probe.code == 1:
            place, threshold = probe.arguments
            return result.marking.multiplicities[place] >= threshold
    raise ValueError("qualification semantic probe differs")


def _definitions() -> tuple[_FoldDefinition, ...]:
    return (
        _FoldDefinition(
            fold=QualificationFold.HORN_CLOSURE,
            worlds=(
                (1, ()),
                (2, ()),
            ),
            commands=(
                GroundAtom(3, (0, 3)),
                GroundAtom(4, (3, 0)),
            ),
            probes=(
                SemanticProbe(0, (2, 3)),
                SemanticProbe(1, (2,)),
            ),
        ),
        _FoldDefinition(
            fold=QualificationFold.TYPED_REWRITE,
            worlds=(
                (1, challenge_terms()[0]),
                (1, challenge_terms()[1]),
            ),
            commands=(0, 1),
            probes=(
                SemanticProbe(0, (5,)),
                SemanticProbe(1, (2,)),
            ),
        ),
        _FoldDefinition(
            fold=QualificationFold.GUARDED_RESOURCE,
            worlds=(
                (12, input_markings()[12]),
                (12, input_markings()[30]),
            ),
            commands=(
                heldout_programs()[0],
                heldout_programs()[3],
            ),
            probes=(
                SemanticProbe(0, (1,)),
                SemanticProbe(1, (2, 1)),
            ),
        ),
    )


def _stage_package_bytes(
    stage: int,
    board_sha256: str,
    entries: list[dict[str, Any]],
) -> bytes:
    return _canonical_bytes(
        {
            "schema": PACKAGE_SCHEMA,
            "stage": stage,
            "board_sha256": board_sha256,
            "entries": entries,
        }
    )


def _package_payloads(
    rows: tuple[QualificationRow, ...],
    board_sha256: str,
) -> tuple[bytes, bytes, bytes, bytes]:
    worlds: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    seen_worlds: set[str] = set()
    seen_packets: set[str] = set()
    for row in rows:
        if row.world_factor_id not in seen_worlds:
            seen_worlds.add(row.world_factor_id)
            worlds.append(
                {
                    "world_factor_id": row.world_factor_id,
                    "world_sha256": _sha256(row.world_bytes),
                    "world_hex": row.world_bytes.hex(),
                }
            )
        if row.packet_factor_id not in seen_packets:
            seen_packets.add(row.packet_factor_id)
            commands.append(
                {
                    "packet_factor_id": row.packet_factor_id,
                    "world_factor_id": row.world_factor_id,
                    "command_factor_id": row.command_factor_id,
                    "command_sha256": _sha256(row.command_bytes),
                    "command_hex": row.command_bytes.hex(),
                }
            )
    queries = [
        {
            "row_id": row.row_id,
            "packet_factor_id": row.packet_factor_id,
            "query_sha256": _sha256(row.query_prefix_bytes),
            "query_hex": row.query_prefix_bytes.hex(),
        }
        for row in rows
    ]
    assessor = [
        {
            "row_id": row.row_id,
            "query_semantic_id": row.query_semantic_id,
            "query_paraphrase_id": row.query_paraphrase_id,
            "target": int(row.target),
        }
        for row in rows
    ]
    return (
        _stage_package_bytes(1, board_sha256, worlds),
        _stage_package_bytes(2, board_sha256, commands),
        _stage_package_bytes(3, board_sha256, queries),
        _stage_package_bytes(4, board_sha256, assessor),
    )


def _audit_rows(
    rows: tuple[QualificationRow, ...],
    *,
    independent_agreement: int,
) -> FactorialQualificationReceipt:
    if len(rows) != TOTAL_ROWS:
        raise ValueError("qualification row count differs")
    if len({row.row_id for row in rows}) != TOTAL_ROWS:
        raise ValueError("qualification row identities repeat")
    packets = tuple(dict.fromkeys(row.packet_factor_id for row in rows))
    worlds = tuple(dict.fromkeys(row.world_factor_id for row in rows))
    commands = tuple(dict.fromkeys(row.command_factor_id for row in rows))
    if (
        len(packets) != TOTAL_PACKETS
        or len(worlds) != FOLDS * WORLDS_PER_FOLD
        or len(commands) != FOLDS * COMMANDS_PER_FOLD
    ):
        raise ValueError("qualification factor geometry differs")

    world_changes = 0
    command_changes = 0
    target_contrasts = 0
    for fold in FOLD_ORDER:
        subset = [row for row in rows if row.fold == fold]
        for semantic in range(SEMANTICS_PER_PACKET):
            for paraphrase in range(PARAPHRASES_PER_SEMANTIC):
                targets = {
                    (row.world_index, row.command_index): row.target
                    for row in subset
                    if row.semantic_index == semantic
                    and row.paraphrase_index == paraphrase
                }
                for command in range(COMMANDS_PER_FOLD):
                    world_changes += targets[(0, command)] != targets[(1, command)]
                for world in range(WORLDS_PER_FOLD):
                    command_changes += targets[(world, 0)] != targets[(world, 1)]
        for world in range(WORLDS_PER_FOLD):
            for command in range(COMMANDS_PER_FOLD):
                per_semantic = {
                    semantic: {
                        row.target
                        for row in subset
                        if row.world_index == world
                        and row.command_index == command
                        and row.semantic_index == semantic
                    }
                    for semantic in range(SEMANTICS_PER_PACKET)
                }
                if any(len(values) != 1 for values in per_semantic.values()):
                    raise ValueError("qualification paraphrase target is not invariant")
                target_contrasts += next(iter(per_semantic[0])) != next(
                    iter(per_semantic[1])
                )

    expected_edge_changes = FOLDS * SEMANTICS_PER_PACKET * PARAPHRASES_PER_SEMANTIC * 2
    if (
        world_changes != expected_edge_changes
        or command_changes != expected_edge_changes
        or target_contrasts != TOTAL_PACKETS
    ):
        raise ValueError("qualification rectangle has a target collision")

    leak_count = 0
    for row in rows:
        candidate = (
            row.world_bytes + row.command_bytes + row.query_prefix_bytes
        ).lower()
        leak_count += sum(token in candidate for token in _FORBIDDEN_CANDIDATE_TOKENS)
        if row.fold.value.encode("ascii") in candidate:
            leak_count += 1
    if leak_count:
        raise ValueError("qualification candidate surface leaks assessor labels")

    row_payloads = [
        {
            item.name: getattr(row, item.name)
            for item in fields(row)
            if item.name != "row_id"
        }
        for row in rows
    ]
    board_sha256 = _digest(
        {
            "schema": BOARD_SCHEMA,
            "rows": row_payloads,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    packages = _package_payloads(rows, board_sha256)
    return FactorialQualificationReceipt(
        schema=BOARD_SCHEMA,
        fold_count=FOLDS,
        world_count=len(worlds),
        command_count=len(commands),
        packet_count=len(packets),
        row_count=len(rows),
        independent_oracle_agreement_count=independent_agreement,
        world_edge_change_count=world_changes,
        command_edge_change_count=command_changes,
        within_packet_target_contrast_count=target_contrasts,
        candidate_label_leak_count=leak_count,
        unique_row_count=len({row.row_id for row in rows}),
        world_package_sha256=_sha256(packages[0]),
        command_package_sha256=_sha256(packages[1]),
        query_package_sha256=_sha256(packages[2]),
        assessor_package_sha256=_sha256(packages[3]),
        payload_sha256=board_sha256,
        claim_boundary=CLAIM_BOUNDARY,
        all_contracts_pass=True,
    )


@lru_cache(maxsize=1)
def build_ettr_factorial_qualification_board() -> ETTRFactorialQualificationBoard:
    rows: list[QualificationRow] = []
    independent_agreement = 0
    for definition in _definitions():
        outputs: dict[tuple[int, int], Any] = {}
        for world_index, world in enumerate(definition.worlds):
            for command_index, command in enumerate(definition.commands):
                primary = _execute_primary(
                    definition.fold,
                    world,
                    command,
                )
                independent = _execute_independent(
                    definition.fold,
                    world,
                    command,
                )
                if primary != independent:
                    raise ValueError("qualification independent oracle disagrees")
                independent_agreement += 1
                outputs[(world_index, command_index)] = primary

        world_payloads = tuple(
            _world_bytes(definition.fold, world) for world in definition.worlds
        )
        command_payloads = tuple(
            _command_bytes(definition.fold, command) for command in definition.commands
        )
        world_ids = tuple(
            _digest(
                {
                    "fold": definition.fold,
                    "world_index": index,
                    "payload_sha256": _sha256(payload),
                }
            )
            for index, payload in enumerate(world_payloads)
        )
        command_ids = tuple(
            _digest(
                {
                    "fold": definition.fold,
                    "command_index": index,
                    "payload_sha256": _sha256(payload),
                }
            )
            for index, payload in enumerate(command_payloads)
        )
        semantic_ids = tuple(
            _digest(
                {
                    "fold": definition.fold,
                    "semantic_index": index,
                    "probe": probe,
                }
            )
            for index, probe in enumerate(definition.probes)
        )
        paraphrase_ids = tuple(
            _digest(
                {
                    "fold": definition.fold,
                    "paraphrase_index": index,
                }
            )
            for index in range(PARAPHRASES_PER_SEMANTIC)
        )

        for world_index in range(WORLDS_PER_FOLD):
            for command_index in range(COMMANDS_PER_FOLD):
                packet_id = _digest(
                    {
                        "fold": definition.fold,
                        "world_factor_id": world_ids[world_index],
                        "command_factor_id": command_ids[command_index],
                    }
                )
                result = outputs[(world_index, command_index)]
                for semantic_index, probe in enumerate(definition.probes):
                    target = _probe_result(
                        definition.fold,
                        result,
                        probe,
                    )
                    for paraphrase_index in range(PARAPHRASES_PER_SEMANTIC):
                        query = _query_prefix_bytes(
                            definition.fold,
                            probe,
                            paraphrase_index,
                        )
                        material = {
                            "fold": definition.fold,
                            "world_index": world_index,
                            "command_index": command_index,
                            "semantic_index": semantic_index,
                            "paraphrase_index": paraphrase_index,
                            "world_sha256": _sha256(world_payloads[world_index]),
                            "command_sha256": _sha256(command_payloads[command_index]),
                            "query_sha256": _sha256(query),
                            "target": target,
                            "world_factor_id": world_ids[world_index],
                            "command_factor_id": command_ids[command_index],
                            "packet_factor_id": packet_id,
                            "query_semantic_id": semantic_ids[semantic_index],
                            "query_paraphrase_id": paraphrase_ids[paraphrase_index],
                        }
                        rows.append(
                            QualificationRow(
                                fold=definition.fold,
                                world_index=world_index,
                                command_index=command_index,
                                semantic_index=semantic_index,
                                paraphrase_index=paraphrase_index,
                                world_bytes=world_payloads[world_index],
                                command_bytes=command_payloads[command_index],
                                query_prefix_bytes=query,
                                target=target,
                                world_factor_id=world_ids[world_index],
                                command_factor_id=command_ids[command_index],
                                packet_factor_id=packet_id,
                                query_semantic_id=semantic_ids[semantic_index],
                                query_paraphrase_id=paraphrase_ids[paraphrase_index],
                                row_id=_digest(material),
                            )
                        )

    frozen_rows = tuple(rows)
    receipt = _audit_rows(
        frozen_rows,
        independent_agreement=independent_agreement,
    )
    board = ETTRFactorialQualificationBoard(frozen_rows, receipt)
    packages = (
        board.world_package_bytes(),
        board.command_package_bytes(),
        board.query_package_bytes(),
        board.assessor_package_bytes(),
    )
    observed_hashes = tuple(_sha256(package) for package in packages)
    expected_hashes = (
        receipt.world_package_sha256,
        receipt.command_package_sha256,
        receipt.query_package_sha256,
        receipt.assessor_package_sha256,
    )
    if observed_hashes != expected_hashes:
        raise ValueError("qualification package receipt differs")
    return board


__all__ = [
    "BOARD_SCHEMA",
    "CLAIM_BOUNDARY",
    "COMMANDS_PER_FOLD",
    "ETTRFactorialQualificationBoard",
    "FOLD_ORDER",
    "FOLDS",
    "FactorialQualificationReceipt",
    "PACKETS_PER_FOLD",
    "PARAPHRASES_PER_SEMANTIC",
    "QualificationFold",
    "QualificationRow",
    "SEMANTICS_PER_PACKET",
    "TOTAL_PACKETS",
    "TOTAL_ROWS",
    "WORLDS_PER_FOLD",
    "build_ettr_factorial_qualification_board",
]
