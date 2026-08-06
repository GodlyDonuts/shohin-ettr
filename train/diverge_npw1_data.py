"""Source-disjoint narrative rendering for the DIVERGE-NPW1 WORLD gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Mapping, Sequence

from diverge_tfs1_data import StepSpec, steps_from_record
from diverge_tol1_ir import (
    DIRECT_OPS,
    Action,
    Atom,
    Instruction,
    Predicate,
    instruction_from_record,
    instruction_record,
)


SCHEMA = "shohin-diverge-npw1-world-v1"
TRAIN_SEED = 2026080618
CONFIRMATION_SEED = 2026080619
CONFIRMATION_ROWS = 256

TRAIN_NAMES = (
    "aster", "banyan", "cirrus", "dovetail", "eider", "fennel", "ginkgo",
    "harvest", "impala", "jetstream", "koala", "lattice", "monsoon",
    "nautilus", "orchid", "pulsar", "quince", "rivulet", "sequoia",
    "tempest", "uplink", "verdant", "watershed", "yearling", "zircon",
    "acacia", "bluejay", "canyon", "drizzle", "evergreen", "firefly",
    "glacier",
)

CONFIRMATION_NAMES = (
    "albatross", "bramble", "citrine", "dogwood", "eclipse", "foxglove",
    "geyser", "hemlock", "inlet", "jackpine", "kingfisher", "lupine",
    "mangrove", "nightfall", "obsidian", "porpoise", "redwood", "starling",
    "thunder", "undertow", "volcano", "wildflower", "yewtree", "zeppelin",
    "anemone", "buckeye", "cormorant", "dewdrop", "firebrand", "goldfinch",
    "highland", "ironwood",
)

_COMPARATORS = {
    "EQ": ("equals", "has the same value as"),
    "NE": ("differs from", "does not have the same value as"),
    "LT": ("is less than", "falls below"),
    "LE": ("is at most", "does not exceed"),
    "GT": ("is greater than", "exceeds"),
    "GE": ("is at least", "is not below"),
}

_TRAIN_HEADERS = (
    "A ledger changes through the following account. ",
    "Read this description of a changing register system. ",
    "The quantities evolve in the narrated order. ",
)
_CONFIRMATION_HEADERS = (
    "Consider this uninterrupted history of several named quantities. ",
    "Follow the events in this prose record from beginning to end. ",
    "This paragraph recounts how a collection of values changes over time. ",
)
_TRAIN_CONNECTORS = (
    "First, ", "Next, ", "After that, ", "Then, ", "Subsequently, ",
)
_CONFIRMATION_CONNECTORS = (
    "To begin, ", "In the following event, ", "Thereafter, ",
    "Continuing the history, ", "At the next moment, ",
)
_TRAIN_JOINERS = ("; then ", "; afterward ", ", and next ")
_CONFIRMATION_JOINERS = (
    "; in the event that follows, ",
    ", after which ",
    "; the narrative then says to ",
)


class NPW1DataError(RuntimeError):
    """An NPW1 narrative or span ledger violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class RenderedEvent:
    text: str
    form: str
    mentions: tuple[dict[str, object], ...]


def _assemble(parts: Sequence[tuple[str, str | None]]) -> tuple[str, tuple[dict[str, object], ...]]:
    text = ""
    mentions = []
    for value, role in parts:
        start = len(text)
        text += value
        if role is not None:
            mentions.append(
                {"role": role, "text": value, "start": start, "end": len(text)}
            )
    return text, tuple(mentions)


def _action_parts(action: Action, prefix: str, expanded: bool):
    action.validate()
    operation = (action.operation.lower(), f"{prefix}OPERATION")
    target = (action.target, f"{prefix}TARGET")
    operand = (action.operand.value, f"{prefix}OPERAND")
    relation = "to" if action.operation == "SET" else "by"
    if expanded:
        return (
            operation,
            (" the value held in ", None),
            target,
            (f" {relation} ", None),
            operand,
        )
    return (operation, (" ", None), target, (f" {relation} ", None), operand)


def render_event(step: StepSpec, *, renderer: int, confirmation: bool) -> RenderedEvent:
    expanded = bool((renderer + int(confirmation)) % 2)
    if step.options is not None:
        left, right = step.options
        assert left.action is not None and right.action is not None
        if left.action.target != right.action.target or left.action.operand != right.action.operand:
            raise NPW1DataError("ambiguous event arguments differ")
        parts = (
            ("apply either ", None),
            (left.operation.lower(), "OPTION_A_OPERATION"),
            (" or ", None),
            (right.operation.lower(), "OPTION_B_OPERATION"),
            (" to ", None),
            (left.action.target, "TARGET"),
            (", using ", None),
            (left.action.operand.value, "OPERAND"),
            (" as the operand", None),
        )
        text, mentions = _assemble(parts)
        return RenderedEvent(text, "AMBIGUOUS", mentions)

    assert step.fixed is not None
    instruction = step.fixed
    if instruction.operation in {"SET", "ADD", "SUBTRACT", "MULTIPLY"}:
        assert instruction.action is not None
        text, mentions = _assemble(_action_parts(instruction.action, "", expanded))
        return RenderedEvent(text, "DIRECT", mentions)
    if instruction.operation == "SWAP":
        assert instruction.swap_left and instruction.swap_right
        parts = (
            ("exchange", "OPERATION"),
            (" the contents of ", None) if expanded else (" ", None),
            (instruction.swap_left, "LEFT"),
            (" and ", None) if expanded else (" with ", None),
            (instruction.swap_right, "RIGHT"),
        )
        text, mentions = _assemble(parts)
        return RenderedEvent(text, "SWAP", mentions)
    if instruction.operation == "GUARD":
        assert instruction.predicate is not None
        assert instruction.true_action is not None
        assert instruction.false_action is not None
        predicate = instruction.predicate
        comparator = _COMPARATORS[predicate.comparator][renderer % 2]
        parts = (
            ("if ", None),
            (predicate.left, "PRED_LEFT"),
            (" ", None),
            (comparator, "COMPARATOR"),
            (" ", None),
            (predicate.right.value, "PRED_RIGHT"),
            (", ", None),
            *_action_parts(instruction.true_action, "TRUE_", expanded),
            ("; otherwise, ", None),
            *_action_parts(instruction.false_action, "FALSE_", not expanded),
        )
        text, mentions = _assemble(parts)
        return RenderedEvent(text, "GUARD", mentions)
    raise NPW1DataError(f"unsupported NPW1 event {instruction.operation}")


def render_narrative(
    steps: Sequence[StepSpec],
    *,
    seed: int,
    confirmation: bool,
) -> dict[str, object]:
    if not steps:
        raise NPW1DataError("empty NPW1 program")
    rng = random.Random(seed)
    headers = _CONFIRMATION_HEADERS if confirmation else _TRAIN_HEADERS
    connectors = _CONFIRMATION_CONNECTORS if confirmation else _TRAIN_CONNECTORS
    joiners = _CONFIRMATION_JOINERS if confirmation else _TRAIN_JOINERS
    source = headers[seed % len(headers)]
    event_rows = []
    cursor = 0
    while cursor < len(steps):
        group_size = min(1 + rng.randrange(3), len(steps) - cursor)
        for offset in range(group_size):
            event_index = cursor + offset
            if offset == 0:
                source += connectors[(seed + event_index) % len(connectors)]
            else:
                source += joiners[(seed + event_index) % len(joiners)]
            rendered = render_event(
                steps[event_index],
                renderer=(seed * 17 + event_index * 11) % 8,
                confirmation=confirmation,
            )
            start = len(source)
            source += rendered.text
            mentions = [
                {
                    **mention,
                    "start": start + int(mention["start"]),
                    "end": start + int(mention["end"]),
                }
                for mention in rendered.mentions
            ]
            event_rows.append(
                {
                    "index": event_index,
                    "start": start,
                    "end": len(source),
                    "form": rendered.form,
                    "fault_index": steps[event_index].fault_index,
                    "mentions": mentions,
                }
            )
        source += ". "
        cursor += group_size
    source += (
        "Use the final state only after every event has been applied."
        if confirmation
        else "The final state follows only after the complete account."
    )
    try:
        source.encode("ascii")
    except UnicodeEncodeError as error:
        raise NPW1DataError("NPW1 source is not ASCII") from error
    return {
        "schema": SCHEMA,
        "source_text": source,
        "source_sha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
        "renderer_split": "confirmation" if confirmation else "training",
        "events": event_rows,
    }


def augment_board(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    confirmation: bool,
) -> list[dict[str, object]]:
    output = []
    for index, row in enumerate(rows):
        steps = steps_from_record(row["steps"])  # type: ignore[arg-type]
        natural_world = render_narrative(
            steps,
            seed=seed + index * 104729,
            confirmation=confirmation,
        )
        identity = hashlib.sha256(
            json.dumps(
                {
                    "semantic_identity": row["identity_sha256"],
                    "source_sha256": natural_world["source_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        augmented = dict(row)
        augmented["npw1_identity_sha256"] = identity
        augmented["natural_world"] = natural_world
        validate_augmented_row(augmented, confirmation=confirmation)
        output.append(augmented)
    return output


def _rename_atom(atom: Atom, mapping: Mapping[str, str]) -> Atom:
    atom.validate()
    if atom.kind == "CONST":
        return atom
    return Atom("REF", mapping[atom.value])


def _rename_action(action: Action, mapping: Mapping[str, str]) -> Action:
    action.validate()
    return Action(
        action.operation,
        mapping[action.target],
        _rename_atom(action.operand, mapping),
    )


def rename_instruction(
    instruction: Instruction,
    mapping: Mapping[str, str],
) -> Instruction:
    instruction.validate()
    if instruction.operation in DIRECT_OPS:
        assert instruction.action is not None
        return Instruction(
            instruction.operation,
            action=_rename_action(instruction.action, mapping),
        )
    if instruction.operation == "SWAP":
        assert instruction.swap_left and instruction.swap_right
        return Instruction(
            "SWAP",
            swap_left=mapping[instruction.swap_left],
            swap_right=mapping[instruction.swap_right],
        )
    if instruction.operation == "GUARD":
        assert instruction.predicate is not None
        assert instruction.true_action is not None
        assert instruction.false_action is not None
        predicate = instruction.predicate
        return Instruction(
            "GUARD",
            predicate=Predicate(
                predicate.comparator,
                mapping[predicate.left],
                _rename_atom(predicate.right, mapping),
            ),
            true_action=_rename_action(instruction.true_action, mapping),
            false_action=_rename_action(instruction.false_action, mapping),
        )
    if instruction.operation == "QUERY":
        assert instruction.query is not None
        return Instruction("QUERY", query=mapping[instruction.query])
    raise NPW1DataError("unknown instruction during NPW1 renaming")


def step_record(step: StepSpec) -> dict[str, object]:
    return {
        "text": step.text,
        "fixed": None if step.fixed is None else instruction_record(step.fixed),
        "options": (
            None
            if step.options is None
            else [instruction_record(value) for value in step.options]
        ),
        "fault_index": step.fault_index,
    }


def training_record_from_tol1(
    row: Mapping[str, object],
    *,
    index: int,
    seed: int,
) -> dict[str, object]:
    clauses = row.get("clauses")
    if not isinstance(clauses, list) or not clauses:
        raise NPW1DataError("NPW1 training source lacks clauses")
    original = [instruction_from_record(value["instruction"]) for value in clauses]
    declarations = [
        value.action.target
        for value in original
        if value.operation == "SET" and value.action is not None
    ]
    symbols = tuple(dict.fromkeys(declarations))
    if len(symbols) != 4:
        raise NPW1DataError("NPW1 training symbol table differs")
    rng = random.Random(seed + index * 65537)
    renamed = tuple(rng.sample(TRAIN_NAMES, len(symbols)))
    mapping = dict(zip(symbols, renamed, strict=True))
    instructions = [rename_instruction(value, mapping) for value in original]
    steps = []
    fault_index = 0
    for step_index, instruction in enumerate(instructions):
        if instruction.operation == "QUERY":
            continue
        if (
            instruction.operation in DIRECT_OPS
            and instruction.action is not None
            and (index + step_index) % 4 == 0
        ):
            alternatives = tuple(
                value for value in DIRECT_OPS if value != instruction.operation
            )
            alternate = alternatives[(index + step_index) % len(alternatives)]
            options = (
                instruction,
                Instruction(
                    alternate,
                    action=Action(
                        alternate,
                        instruction.action.target,
                        instruction.action.operand,
                    ),
                ),
            )
            steps.append(
                StepSpec(
                    f"npw1-training-ambiguous-{step_index}",
                    options=options,
                    fault_index=fault_index,
                )
            )
            fault_index += 1
        else:
            steps.append(
                StepSpec(f"npw1-training-fixed-{step_index}", fixed=instruction)
            )
    records = [step_record(value) for value in steps]
    semantic_identity = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    record: dict[str, object] = {
        "schema": "shohin-diverge-npw1-training-v1",
        "id": f"npw1-train-{index:07d}-{semantic_identity[:12]}",
        "identity_sha256": semantic_identity,
        "symbols": list(renamed),
        "steps": records,
    }
    return augment_board(
        [record],
        seed=seed + index * 104729,
        confirmation=False,
    )[0]


def validate_augmented_row(row: Mapping[str, object], *, confirmation: bool) -> None:
    steps = steps_from_record(row["steps"])  # type: ignore[arg-type]
    world = row["natural_world"]
    if not isinstance(world, Mapping) or world.get("schema") != SCHEMA:
        raise NPW1DataError("NPW1 world schema differs")
    expected_split = "confirmation" if confirmation else "training"
    if world.get("renderer_split") != expected_split:
        raise NPW1DataError("NPW1 renderer split differs")
    source = str(world["source_text"])
    if "\n" in source or hashlib.sha256(source.encode("ascii")).hexdigest() != world.get("source_sha256"):
        raise NPW1DataError("NPW1 source commitment differs")
    events = world["events"]
    if not isinstance(events, list) or len(events) != len(steps):
        raise NPW1DataError("NPW1 event count differs")
    previous_end = -1
    for index, event in enumerate(events):
        if int(event["index"]) != index:
            raise NPW1DataError("NPW1 event order differs")
        start, end = int(event["start"]), int(event["end"])
        if not (previous_end < start < end <= len(source)):
            raise NPW1DataError("NPW1 event spans are not monotone")
        previous_end = end
        mentions = event["mentions"]
        if not isinstance(mentions, list) or not mentions:
            raise NPW1DataError("NPW1 event has no mention ledger")
        for mention in mentions:
            left, right = int(mention["start"]), int(mention["end"])
            if not (start <= left < right <= end):
                raise NPW1DataError("NPW1 mention lies outside its event")
            if source[left:right] != mention["text"]:
                raise NPW1DataError("NPW1 mention text differs")
    identity = hashlib.sha256(
        json.dumps(
            {
                "semantic_identity": row["identity_sha256"],
                "source_sha256": world["source_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    if identity != row.get("npw1_identity_sha256"):
        raise NPW1DataError("NPW1 identity commitment differs")


__all__ = [
    "CONFIRMATION_NAMES",
    "CONFIRMATION_ROWS",
    "CONFIRMATION_SEED",
    "NPW1DataError",
    "SCHEMA",
    "TRAIN_NAMES",
    "TRAIN_SEED",
    "augment_board",
    "render_event",
    "render_narrative",
    "rename_instruction",
    "step_record",
    "training_record_from_tol1",
    "validate_augmented_row",
]
