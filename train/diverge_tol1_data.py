"""Deterministic source board and supervision for DIVERGE-TOL1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import random
import re
from typing import Iterable, Sequence

from diverge_tol1_ir import (
    Action,
    Atom,
    CLAUSE_OPS,
    COMPARATORS,
    DIRECT_OPS,
    Instruction,
    Predicate,
    TOL1IRError,
    execute_program,
    format_fraction,
    instruction_record,
    instruction_sha256,
)


class TOL1DataError(RuntimeError):
    """A generated source row violates the frozen TOL1 board contract."""


SCHEMA = "shohin-diverge-tol1-board-v1"
PAD_ID = 0
CLS_ID = 1
BYTE_OFFSET = 2
BYTE_VOCAB_SIZE = 130
MAX_CLAUSE_BYTES = 192

ROLE_NAMES = (
    "NONE",
    "TARGET",
    "OPERAND",
    "PRED_LEFT",
    "PRED_RIGHT",
    "TRUE_TARGET",
    "TRUE_OPERAND",
    "FALSE_TARGET",
    "FALSE_OPERAND",
    "QUERY_REF",
)
ROLE_TO_ID = {name: index for index, name in enumerate(ROLE_NAMES)}
OP_TO_ID = {name: index for index, name in enumerate(CLAUSE_OPS)}
COMPARATOR_NAMES = ("NONE", *COMPARATORS)
COMPARATOR_TO_ID = {name: index for index, name in enumerate(COMPARATOR_NAMES)}
ACTION_NAMES = ("NONE", *DIRECT_OPS)
ACTION_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}

TRAIN_NAMES = (
    "amber", "birch", "cedar", "delta", "ember", "fable", "grove", "hazel",
    "ivory", "jade", "kestrel", "lilac", "maple", "nova", "onyx", "pearl",
    "quartz", "raven", "sable", "topaz", "umber", "violet", "willow", "xenon",
    "yarrow", "zephyr", "acorn", "brook", "clover", "dune", "elm", "frost",
)
OOD_NAMES = (
    "atlas", "blaze", "coral", "drift", "echo", "fjord", "glint", "harbor",
    "iris", "juniper", "knoll", "lotus", "mirth", "nimbus", "opal", "plume",
    "ridge", "spruce", "thistle", "upland", "vale", "wren", "yonder", "zenith",
)

_CANDIDATE = re.compile(r"-?(?:0|[1-9]\d*)(?:/[1-9]\d*)?|[a-z][a-z0-9_]{1,15}")


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    text: str
    start: int
    end: int
    kind: str
    role_id: int


@dataclass(frozen=True, slots=True)
class ClauseTarget:
    text: str
    byte_ids: tuple[int, ...]
    candidates: tuple[SourceCandidate, ...]
    operation_id: int
    comparator_id: int
    true_action_id: int
    false_action_id: int
    instruction: Instruction
    instruction_sha256: str


def encode_bytes(text: str) -> tuple[int, ...]:
    try:
        payload = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise TOL1DataError("TOL1 source must be ASCII") from error
    if not payload or len(payload) + 1 > MAX_CLAUSE_BYTES:
        raise TOL1DataError("TOL1 clause length differs")
    return (CLS_ID, *(value + BYTE_OFFSET for value in payload))


def source_candidates(
    text: str,
    labelled: Sequence[tuple[int, int, str]] = (),
) -> tuple[SourceCandidate, ...]:
    labels = {(int(start), int(end)): role for start, end, role in labelled}
    output = []
    covered = set()
    for match in _CANDIDATE.finditer(text):
        value = match.group(0)
        key = (match.start(), match.end())
        role = labels.get(key, "NONE")
        if role not in ROLE_TO_ID:
            raise TOL1DataError("unknown source role")
        kind = "NUMBER" if value[0].isdigit() or value[0] == "-" else "WORD"
        output.append(SourceCandidate(value, *key, kind, ROLE_TO_ID[role]))
        if role != "NONE":
            covered.add(key)
    if covered != set(labels):
        raise TOL1DataError("labelled span is absent from candidate lattice")
    return tuple(output)


def _assemble(parts: Sequence[tuple[str, str | None]]) -> tuple[str, tuple[tuple[int, int, str], ...]]:
    text = ""
    spans = []
    for value, role in parts:
        start = len(text)
        text += value
        if role is not None:
            spans.append((start, len(text), role))
    return text, tuple(spans)


def _atom(atom: Atom) -> str:
    atom.validate()
    return atom.value


def _action_parts(action: Action, target_role: str, operand_role: str, variant: int):
    action.validate()
    target = (action.target, target_role)
    operand = (_atom(action.operand), operand_role)
    if action.operation == "SET":
        options = (
            (("set ", None), target, (" to ", None), operand),
            (("assign ", None), operand, (" into ", None), target),
            (("into ", None), target, (", assign ", None), operand),
        )
    elif action.operation == "ADD":
        options = (
            (("add ", None), operand, (" to ", None), target),
            (("increase ", None), target, (" by ", None), operand),
            (("to ", None), target, (", add ", None), operand),
        )
    elif action.operation == "SUBTRACT":
        options = (
            (("subtract ", None), operand, (" from ", None), target),
            (("decrease ", None), target, (" by ", None), operand),
            (("from ", None), target, (", subtract ", None), operand),
        )
    elif action.operation == "MULTIPLY":
        options = (
            (("multiply ", None), target, (" by ", None), operand),
            (("scale ", None), target, (" by ", None), operand),
            (("by ", None), operand, (", multiply ", None), target),
        )
    else:
        raise TOL1DataError("unknown rendered action")
    return options[variant % len(options)]


_COMPARATOR_TEXT = {
    "EQ": ("equals", "is equal to", "=="),
    "NE": ("differs from", "is not equal to", "!="),
    "LT": ("is less than", "falls below", "<"),
    "LE": ("is at most", "does not exceed", "<="),
    "GT": ("is greater than", "exceeds", ">"),
    "GE": ("is at least", "is not below", ">="),
}


def render_instruction(
    instruction: Instruction,
    *,
    renderer: int,
    ood: bool,
) -> ClauseTarget:
    instruction.validate()
    variant = 2 if ood else renderer % 2
    if instruction.operation in DIRECT_OPS:
        assert instruction.action is not None
        parts = (*_action_parts(instruction.action, "TARGET", "OPERAND", variant), (".", None))
    elif instruction.operation == "SWAP":
        assert instruction.swap_left and instruction.swap_right
        left = (instruction.swap_left, "TARGET")
        right = (instruction.swap_right, "OPERAND")
        options = (
            (("swap ", None), left, (" with ", None), right, (".", None)),
            (("exchange ", None), left, (" and ", None), right, (".", None)),
            (("with ", None), right, (", swap ", None), left, (".", None)),
        )
        parts = options[variant]
    elif instruction.operation == "GUARD":
        assert instruction.predicate and instruction.true_action and instruction.false_action
        predicate = instruction.predicate
        comparison = _COMPARATOR_TEXT[predicate.comparator][renderer % 3]
        left = (predicate.left, "PRED_LEFT")
        right = (_atom(predicate.right), "PRED_RIGHT")
        branch_variant = renderer % 2
        true_parts = _action_parts(
            instruction.true_action, "TRUE_TARGET", "TRUE_OPERAND", branch_variant
        )
        false_parts = _action_parts(
            instruction.false_action,
            "FALSE_TARGET",
            "FALSE_OPERAND",
            branch_variant + 1,
        )
        options = (
            (
                ("if ", None), left, (f" {comparison} ", None), right,
                (", then ", None), *true_parts, ("; otherwise, ", None),
                *false_parts, (".", None),
            ),
            (
                ("when ", None), left, (f" {comparison} ", None), right,
                (": ", None), *true_parts, ("; else ", None), *false_parts,
                (".", None),
            ),
            (
                *true_parts, (" if ", None), left, (f" {comparison} ", None),
                right, ("; otherwise ", None), *false_parts, (".", None),
            ),
        )
        parts = options[variant]
    else:
        assert instruction.query is not None
        query = (instruction.query, "QUERY_REF")
        options = (
            (("report ", None), query, (".", None)),
            (("return ", None), query, (".", None)),
            (("report the value in ", None), query, (".", None)),
        )
        parts = options[variant]
    text, spans = _assemble(parts)
    candidates = source_candidates(text, spans)
    return ClauseTarget(
        text=text,
        byte_ids=encode_bytes(text),
        candidates=candidates,
        operation_id=OP_TO_ID[instruction.operation],
        comparator_id=COMPARATOR_TO_ID[
            instruction.predicate.comparator if instruction.predicate else "NONE"
        ],
        true_action_id=ACTION_TO_ID[
            instruction.true_action.operation if instruction.true_action else "NONE"
        ],
        false_action_id=ACTION_TO_ID[
            instruction.false_action.operation if instruction.false_action else "NONE"
        ],
        instruction=instruction,
        instruction_sha256=instruction_sha256(instruction),
    )


def _random_fraction(rng: random.Random, *, nonzero: bool = False) -> Atom:
    while True:
        denominator = rng.choice((1, 1, 1, 2, 3, 4, 5))
        numerator = rng.randint(-9, 9)
        if nonzero and numerator == 0:
            continue
        value = Fraction(numerator, denominator)
        return Atom("CONST", format_fraction(value))


def _random_atom(
    rng: random.Random,
    names: Sequence[str],
    *,
    prefer_reference: bool = False,
    nonzero: bool = False,
) -> Atom:
    if prefer_reference or rng.random() < 0.5:
        return Atom("REF", rng.choice(tuple(names)))
    return _random_fraction(rng, nonzero=nonzero)


def _action_for(
    rng: random.Random,
    operation: str,
    names: Sequence[str],
) -> Action:
    target = rng.choice(tuple(names))
    operand = _random_atom(
        rng,
        tuple(name for name in names if name != target),
        prefer_reference=rng.random() < 0.45,
        nonzero=operation == "MULTIPLY",
    )
    return Action(operation, target, operand)


def _body_program(
    rng: random.Random,
    names: Sequence[str],
    depth: int,
    *,
    ood: bool,
) -> tuple[Instruction, ...]:
    if depth < 4:
        raise TOL1DataError("TOL1 body depth is too small")
    direct = list(DIRECT_OPS)
    if ood:
        kinds = ["GUARD", "SWAP", "MULTIPLY"]
    else:
        kinds = ["GUARD", "SWAP"]
        rng.shuffle(kinds)
    while len(kinds) < depth:
        kinds.append(rng.choice((*direct, "GUARD", "SWAP")))
    if not ood:
        rng.shuffle(kinds)
        for index in range(len(kinds) - 1):
            if (kinds[index], kinds[index + 1]) in {
                ("GUARD", "SWAP"),
                ("SWAP", "MULTIPLY"),
            }:
                kinds[index + 1] = rng.choice(("SET", "ADD", "SUBTRACT"))
    output = []
    for kind in kinds:
        if kind in DIRECT_OPS:
            output.append(Instruction(kind, action=_action_for(rng, kind, names)))
        elif kind == "SWAP":
            left, right = rng.sample(tuple(names), 2)
            output.append(Instruction("SWAP", swap_left=left, swap_right=right))
        else:
            left = rng.choice(tuple(names))
            predicate = Predicate(
                rng.choice(COMPARATORS),
                left,
                _random_atom(rng, tuple(name for name in names if name != left)),
            )
            true_operation = rng.choice(DIRECT_OPS)
            false_operation = rng.choice(DIRECT_OPS)
            output.append(
                Instruction(
                    "GUARD",
                    predicate=predicate,
                    true_action=_action_for(rng, true_operation, names),
                    false_action=_action_for(rng, false_operation, names),
                )
            )
    return tuple(output)


def generate_row(
    rng: random.Random,
    *,
    split: str,
    index: int,
) -> dict[str, object]:
    if split not in {"train", "development", "ood"}:
        raise TOL1DataError("unknown TOL1 split")
    ood = split == "ood"
    name_bank = OOD_NAMES if ood else TRAIN_NAMES
    depth = rng.randint(9, 14) if ood else rng.randint(4, 8)
    for _ in range(100):
        names = tuple(rng.sample(name_bank, 4))
        initial = tuple(
            Instruction(
                "SET",
                action=Action("SET", name, _random_fraction(rng)),
            )
            for name in names
        )
        body = _body_program(rng, names, depth, ood=ood)
        query = Instruction("QUERY", query=rng.choice(names))
        program = (*initial, *body, query)
        try:
            answer, trajectory = execute_program(program)
        except TOL1IRError:
            continue
        clauses = [
            render_instruction(
                instruction,
                renderer=(index * 17 + clause_index * 7) % 3,
                ood=ood,
            )
            for clause_index, instruction in enumerate(program)
        ]
        features = {
            "guard": any(value.operation == "GUARD" for value in body),
            "swap": any(value.operation == "SWAP" for value in body),
            "register_operand": any(
                (
                    value.action is not None
                    and value.action.operand.kind == "REF"
                )
                or (
                    value.predicate is not None
                    and value.predicate.right.kind == "REF"
                )
                for value in body
            ),
            "rational": any("/" in clause.text for clause in clauses),
        }
        if not all(features.values()):
            continue
        source = "Typed state program:\n" + "\n".join(
            clause.text for clause in clauses
        ) + "\nEnd program."
        identity = hashlib.sha256(
            json.dumps(
                [instruction_record(value) for value in program],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return {
            "schema": SCHEMA,
            "split": split,
            "id": f"tol1-{split}-{index:07d}-{identity[:12]}",
            "identity_sha256": identity,
            "source": source,
            "clauses": [clause_record(value) for value in clauses],
            "answer": format_fraction(answer),
            "trajectory_sha256": hashlib.sha256(
                json.dumps(trajectory, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "body_depth": depth,
            "features": features,
        }
    raise TOL1DataError("could not generate bounded TOL1 program")


def clause_record(clause: ClauseTarget) -> dict[str, object]:
    return {
        "text": clause.text,
        "roles": [
            {
                "start": candidate.start,
                "end": candidate.end,
                "role": ROLE_NAMES[candidate.role_id],
            }
            for candidate in clause.candidates
            if candidate.role_id != ROLE_TO_ID["NONE"]
        ],
        "operation": CLAUSE_OPS[clause.operation_id],
        "comparator": COMPARATOR_NAMES[clause.comparator_id],
        "true_action": ACTION_NAMES[clause.true_action_id],
        "false_action": ACTION_NAMES[clause.false_action_id],
        "instruction": instruction_record(clause.instruction),
        "instruction_sha256": clause.instruction_sha256,
    }


def clause_from_record(record: dict[str, object]) -> ClauseTarget:
    from diverge_tol1_ir import instruction_from_record

    text = str(record["text"])
    labelled = tuple(
        (int(value["start"]), int(value["end"]), str(value["role"]))
        for value in record["roles"]
    )
    instruction = instruction_from_record(record["instruction"])
    if instruction_sha256(instruction) != record["instruction_sha256"]:
        raise TOL1DataError("serialized instruction hash differs")
    return ClauseTarget(
        text=text,
        byte_ids=encode_bytes(text),
        candidates=source_candidates(text, labelled),
        operation_id=OP_TO_ID[str(record["operation"])],
        comparator_id=COMPARATOR_TO_ID[str(record["comparator"])],
        true_action_id=ACTION_TO_ID[str(record["true_action"])],
        false_action_id=ACTION_TO_ID[str(record["false_action"])],
        instruction=instruction,
        instruction_sha256=str(record["instruction_sha256"]),
    )


def validate_row(row: dict[str, object], expected_split: str) -> None:
    if row.get("schema") != SCHEMA or row.get("split") != expected_split:
        raise TOL1DataError("TOL1 row schema or split differs")
    clauses = tuple(clause_from_record(value) for value in row["clauses"])
    expected_source = "Typed state program:\n" + "\n".join(
        value.text for value in clauses
    ) + "\nEnd program."
    if row.get("source") != expected_source:
        raise TOL1DataError("TOL1 document source differs")
    answer, trajectory = execute_program(tuple(value.instruction for value in clauses))
    if format_fraction(answer) != row.get("answer"):
        raise TOL1DataError("TOL1 answer differs")
    trajectory_sha = hashlib.sha256(
        json.dumps(trajectory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if trajectory_sha != row.get("trajectory_sha256"):
        raise TOL1DataError("TOL1 trajectory hash differs")


def generate_split(split: str, count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = [generate_row(rng, split=split, index=index) for index in range(count)]
    identities = [str(row["identity_sha256"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise TOL1DataError("duplicate TOL1 program identity")
    for row in rows:
        validate_row(row, split)
    return rows


def split_report(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    materialized = list(rows)
    clauses = [clause for row in materialized for clause in row["clauses"]]
    operation_counts = Counter(str(clause["operation"]) for clause in clauses)
    return {
        "rows": len(materialized),
        "clauses": len(clauses),
        "operation_counts": dict(sorted(operation_counts.items())),
        "depth_counts": dict(
            sorted(Counter(int(row["body_depth"]) for row in materialized).items())
        ),
        "feature_counts": {
            name: sum(bool(row["features"][name]) for row in materialized)
            for name in ("guard", "swap", "register_operand", "rational")
        },
    }


__all__ = [
    "ACTION_NAMES",
    "ACTION_TO_ID",
    "BYTE_OFFSET",
    "BYTE_VOCAB_SIZE",
    "CLAUSE_OPS",
    "CLS_ID",
    "COMPARATOR_NAMES",
    "COMPARATOR_TO_ID",
    "ClauseTarget",
    "MAX_CLAUSE_BYTES",
    "OP_TO_ID",
    "PAD_ID",
    "ROLE_NAMES",
    "ROLE_TO_ID",
    "SCHEMA",
    "SourceCandidate",
    "TOL1DataError",
    "clause_from_record",
    "clause_record",
    "encode_bytes",
    "generate_row",
    "generate_split",
    "render_instruction",
    "source_candidates",
    "split_report",
    "validate_row",
]
