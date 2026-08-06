"""Fresh held-out renderer and board for DIVERGE-TOL3 confirmation."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Sequence

from diverge_tol1_data import (
    ACTION_TO_ID,
    COMPARATOR_TO_ID,
    OP_TO_ID,
    SCHEMA,
    ClauseTarget,
    TOL1DataError,
    _body_program,
    _random_fraction,
    clause_record,
    encode_bytes,
    source_candidates,
    validate_row,
)
from diverge_tol1_ir import (
    Action,
    DIRECT_OPS,
    Instruction,
    TOL1IRError,
    execute_program,
    format_fraction,
    instruction_record,
    instruction_sha256,
)


CONFIRMATION_NAMES = (
    "alpine",
    "aurora",
    "basil",
    "brisk",
    "cipher",
    "cobalt",
    "denim",
    "dragon",
    "egret",
    "estuary",
    "flint",
    "forest",
    "garnet",
    "helios",
    "indigo",
    "jigsaw",
    "krypton",
    "lagoon",
    "meteor",
    "nectar",
    "orbit",
    "pollen",
    "quiver",
    "rocket",
    "saffron",
    "tundra",
    "utopia",
    "velvet",
    "whisper",
    "xylem",
    "yukon",
    "zodiac",
)

_COMPARATOR_TEXT = {
    "EQ": ("equals", "is equal to", "=="),
    "NE": ("differs from", "is not equal to", "!="),
    "LT": ("is less than", "falls below", "<"),
    "LE": ("is at most", "does not exceed", "<="),
    "GT": ("is greater than", "exceeds", ">"),
    "GE": ("is at least", "is not below", ">="),
}


def _assemble(
    parts: Sequence[tuple[str, str | None]],
) -> tuple[str, tuple[tuple[int, int, str], ...]]:
    text = ""
    spans = []
    for value, role in parts:
        start = len(text)
        text += value
        if role is not None:
            spans.append((start, len(text), role))
    return text, tuple(spans)


def _atom_text(action: Action) -> str:
    action.validate()
    return action.operand.value


def _action_parts(action: Action, target_role: str, operand_role: str):
    action.validate()
    target = (action.target, target_role)
    operand = (_atom_text(action), operand_role)
    if action.operation == "SET":
        return (("into ", None), target, (", set ", None), operand)
    if action.operation == "ADD":
        return (("to ", None), target, (", increase by ", None), operand)
    if action.operation == "SUBTRACT":
        return (("from ", None), target, (", decrease by ", None), operand)
    if action.operation == "MULTIPLY":
        return (("multiply ", None), target, (" with ", None), operand)
    raise TOL1DataError("unknown TOL3 confirmation action")


def render_confirmation_instruction(
    instruction: Instruction,
    *,
    comparator_variant: int,
) -> ClauseTarget:
    instruction.validate()
    if instruction.operation in DIRECT_OPS:
        assert instruction.action is not None
        parts = (*_action_parts(instruction.action, "TARGET", "OPERAND"), (".", None))
    elif instruction.operation == "SWAP":
        assert instruction.swap_left and instruction.swap_right
        parts = (
            ("with ", None),
            (instruction.swap_right, "OPERAND"),
            (", exchange ", None),
            (instruction.swap_left, "TARGET"),
            (".", None),
        )
    elif instruction.operation == "GUARD":
        assert instruction.predicate
        assert instruction.true_action
        assert instruction.false_action
        predicate = instruction.predicate
        comparison = _COMPARATOR_TEXT[predicate.comparator][comparator_variant % 3]
        parts = (
            ("otherwise ", None),
            *_action_parts(
                instruction.false_action, "FALSE_TARGET", "FALSE_OPERAND"
            ),
            ("; if ", None),
            (predicate.left, "PRED_LEFT"),
            (f" {comparison} ", None),
            (predicate.right.value, "PRED_RIGHT"),
            (", then ", None),
            *_action_parts(
                instruction.true_action, "TRUE_TARGET", "TRUE_OPERAND"
            ),
            (".", None),
        )
    else:
        assert instruction.query is not None
        parts = (
            ("with ", None),
            (instruction.query, "QUERY_REF"),
            (", return.", None),
        )
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


def generate_confirmation_row(
    rng: random.Random,
    *,
    index: int,
) -> dict[str, object]:
    depth = rng.randint(15, 20)
    for _ in range(100):
        names = tuple(rng.sample(CONFIRMATION_NAMES, 4))
        initial = tuple(
            Instruction(
                "SET",
                action=Action("SET", name, _random_fraction(rng)),
            )
            for name in names
        )
        body = _body_program(rng, names, depth, ood=True)
        query = Instruction("QUERY", query=rng.choice(names))
        program = (*initial, *body, query)
        try:
            answer, trajectory = execute_program(program)
        except TOL1IRError:
            continue
        clauses = [
            render_confirmation_instruction(
                instruction,
                comparator_variant=(index * 19 + clause_index * 11) % 3,
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
            ).encode("ascii")
        ).hexdigest()
        row = {
            "schema": SCHEMA,
            "split": "ood",
            "id": f"tol3-confirmation-{index:07d}-{identity[:12]}",
            "identity_sha256": identity,
            "source": source,
            "clauses": [clause_record(value) for value in clauses],
            "answer": format_fraction(answer),
            "trajectory_sha256": hashlib.sha256(
                json.dumps(
                    trajectory, sort_keys=True, separators=(",", ":")
                ).encode("ascii")
            ).hexdigest(),
            "body_depth": depth,
            "features": features,
        }
        validate_row(row, "ood")
        return row
    raise TOL1DataError("could not generate bounded TOL3 confirmation program")


def generate_confirmation_split(count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = [generate_confirmation_row(rng, index=index) for index in range(count)]
    identities = [str(row["identity_sha256"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise TOL1DataError("duplicate TOL3 confirmation identity")
    return rows


__all__ = [
    "CONFIRMATION_NAMES",
    "generate_confirmation_row",
    "generate_confirmation_split",
    "render_confirmation_instruction",
]
