"""Independent board and exact assessor for DIVERGE-TFS1."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import operator
import random
from typing import Mapping, Sequence

from diverge_tol1_ir import (
    Action,
    Atom,
    COMPARATORS,
    DIRECT_OPS,
    Instruction,
    Predicate,
    TOL1IRError,
    format_fraction,
    instruction_from_record,
    instruction_record,
    parse_fraction,
)
from diverge_tol3_confirmation_data import render_confirmation_instruction


SCHEMA = "shohin-diverge-tfs1-board-v1"
FAULT_LINES = 12
WORLDS = 1 << FAULT_LINES
REGISTER_COUNT = 5
ACTIVE_REGISTERS = 4

TFS1_NAMES = (
    "apricot",
    "beacon",
    "canvas",
    "dahlia",
    "equinox",
    "falcon",
    "galaxy",
    "horizon",
    "isotope",
    "jasmine",
    "kernel",
    "lantern",
    "marble",
    "neutron",
    "oasis",
    "prairie",
    "quasar",
    "ripple",
    "sonnet",
    "timber",
    "urchin",
    "vector",
    "walnut",
    "yonderly",
    "zenlike",
    "artifact",
    "boulder",
    "crimson",
    "diagram",
    "emerald",
    "furnace",
    "granite",
)

OPERATION_WORD = {
    "SET": "set",
    "ADD": "increase",
    "SUBTRACT": "decrease",
    "MULTIPLY": "multiply",
}

OPERATION_PAIRS = (
    ("ADD", "SUBTRACT"),
    ("SET", "ADD"),
    ("MULTIPLY", "ADD"),
    ("SET", "MULTIPLY"),
    ("SUBTRACT", "MULTIPLY"),
    ("ADD", "MULTIPLY"),
)

_COMPARISON = {
    "EQ": operator.eq,
    "NE": operator.ne,
    "LT": operator.lt,
    "LE": operator.le,
    "GT": operator.gt,
    "GE": operator.ge,
}


class TFS1DataError(RuntimeError):
    """A TFS1 board or assessor contract is invalid."""


State = tuple[tuple[str, Fraction], ...]


@dataclass(frozen=True, slots=True)
class StepSpec:
    text: str
    fixed: Instruction | None = None
    options: tuple[Instruction, Instruction] | None = None
    fault_index: int | None = None

    def __post_init__(self) -> None:
        if (self.fixed is None) == (self.options is None):
            raise TFS1DataError("step must be fixed or ambiguous")
        if self.options is not None:
            if self.fault_index is None or len(self.options) != 2:
                raise TFS1DataError("ambiguous step geometry differs")
            for option in self.options:
                option.validate()
                if option.operation not in DIRECT_OPS:
                    raise TFS1DataError("fault-line option must be direct")
        elif self.fault_index is not None:
            raise TFS1DataError("fixed step carries a fault index")
        if self.fixed is not None:
            self.fixed.validate()


def _state_dict(state: State) -> dict[str, Fraction]:
    return dict(state)


def _freeze_state(state: Mapping[str, Fraction]) -> State:
    return tuple(sorted((name, Fraction(value)) for name, value in state.items()))


def state_record(state: State) -> dict[str, str]:
    return {name: format_fraction(value) for name, value in state}


def state_from_record(record: Mapping[str, str]) -> State:
    return _freeze_state(
        {name: parse_fraction(value) for name, value in record.items()}
    )


def _resolve(atom: Atom, state: Mapping[str, Fraction]) -> Fraction:
    atom.validate()
    if atom.kind == "CONST":
        return parse_fraction(atom.value)
    try:
        return state[atom.value]
    except KeyError as error:
        raise TOL1IRError("action reads an uninitialized register") from error


def _apply_action(action: Action, state: dict[str, Fraction]) -> None:
    action.validate()
    value = _resolve(action.operand, state)
    if action.operation == "SET":
        state[action.target] = value
        return
    if action.target not in state:
        raise TOL1IRError("action updates an uninitialized register")
    if action.operation == "ADD":
        state[action.target] += value
    elif action.operation == "SUBTRACT":
        state[action.target] -= value
    elif action.operation == "MULTIPLY":
        state[action.target] *= value
    else:
        raise TOL1IRError("unknown direct action")


def apply_instruction(state: State, instruction: Instruction) -> State:
    instruction.validate()
    mutable = _state_dict(state)
    if instruction.operation in DIRECT_OPS:
        assert instruction.action is not None
        _apply_action(instruction.action, mutable)
    elif instruction.operation == "SWAP":
        assert instruction.swap_left and instruction.swap_right
        try:
            mutable[instruction.swap_left], mutable[instruction.swap_right] = (
                mutable[instruction.swap_right],
                mutable[instruction.swap_left],
            )
        except KeyError as error:
            raise TOL1IRError("swap reads an uninitialized register") from error
    elif instruction.operation == "GUARD":
        assert instruction.predicate
        assert instruction.true_action
        assert instruction.false_action
        predicate = instruction.predicate
        try:
            left = mutable[predicate.left]
        except KeyError as error:
            raise TOL1IRError("predicate reads an uninitialized register") from error
        right = _resolve(predicate.right, mutable)
        action = (
            instruction.true_action
            if _COMPARISON[predicate.comparator](left, right)
            else instruction.false_action
        )
        _apply_action(action, mutable)
    else:
        raise TOL1IRError("TFS1 execution forbids embedded query")
    return _freeze_state(mutable)


def execute_steps(
    steps: Sequence[StepSpec],
    assignment: Sequence[int],
) -> tuple[State, tuple[State, ...]]:
    if len(assignment) != FAULT_LINES or any(
        value not in (0, 1) for value in assignment
    ):
        raise TFS1DataError("assignment geometry differs")
    state: State = ()
    trajectory = []
    for step in steps:
        if step.fixed is not None:
            instruction = step.fixed
        else:
            assert step.options is not None
            assert step.fault_index is not None
            instruction = step.options[assignment[step.fault_index]]
        state = apply_instruction(state, instruction)
        trajectory.append(state)
    return state, tuple(trajectory)


def _bounded(state: State) -> bool:
    return all(
        abs(value.numerator).bit_length() <= 72 and value.denominator.bit_length() <= 36
        for _, value in state
    )


def _random_nonzero_fraction(rng: random.Random) -> Atom:
    while True:
        denominator = rng.choice((1, 1, 2, 3, 4, 5))
        numerator = rng.randint(-5, 5)
        if numerator:
            return Atom("CONST", format_fraction(Fraction(numerator, denominator)))


def _random_operand(
    rng: random.Random,
    names: Sequence[str],
    target: str,
) -> Atom:
    others = tuple(name for name in names if name != target)
    if rng.random() < 0.35:
        return Atom("REF", rng.choice(others))
    return _random_nonzero_fraction(rng)


def _ambiguous_text(options: tuple[Instruction, Instruction]) -> str:
    left, right = options
    assert left.action and right.action
    if (
        left.action.target != right.action.target
        or left.action.operand != right.action.operand
    ):
        raise TFS1DataError("fault-line options do not share arguments")
    return (
        f"{OPERATION_WORD[left.operation]} / {OPERATION_WORD[right.operation]} "
        f"{left.action.target} with {left.action.operand.value}."
    )


def _instruction_record(instruction: Instruction) -> dict[str, object]:
    return instruction_record(instruction)


def _step_record(step: StepSpec) -> dict[str, object]:
    return {
        "text": step.text,
        "fixed": None if step.fixed is None else _instruction_record(step.fixed),
        "options": (
            None
            if step.options is None
            else [_instruction_record(value) for value in step.options]
        ),
        "fault_index": step.fault_index,
    }


def steps_from_record(records: Sequence[Mapping[str, object]]) -> tuple[StepSpec, ...]:
    output = []
    for record in records:
        fixed_record = record.get("fixed")
        option_records = record.get("options")
        fixed = (
            None if fixed_record is None else instruction_from_record(fixed_record)  # type: ignore[arg-type]
        )
        options = (
            None
            if option_records is None
            else tuple(
                instruction_from_record(value)
                for value in option_records  # type: ignore[union-attr]
            )
        )
        output.append(
            StepSpec(
                str(record["text"]),
                fixed,
                options,  # type: ignore[arg-type]
                None
                if record.get("fault_index") is None
                else int(record["fault_index"]),
            )
        )
    return tuple(output)


def _evidence_commitment(
    source_commitment: str,
    index: int,
    step_index: int,
    register: str,
    value: str,
) -> str:
    payload = {
        "source_commitment": source_commitment,
        "index": index,
        "step_index": step_index,
        "register": register,
        "value": value,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _query_text(register: str) -> str:
    return f"with {register}, return."


def _value(state: State, register: str) -> Fraction:
    try:
        return dict(state)[register]
    except KeyError as error:
        raise TFS1DataError("query register is absent") from error


def _generate_candidate(
    rng: random.Random,
    *,
    index: int,
) -> dict[str, object] | None:
    names = tuple(rng.sample(TFS1_NAMES, REGISTER_COUNT))
    active = names[:ACTIVE_REGISTERS]
    sentinel = names[-1]
    gold = [0] * (FAULT_LINES // 2) + [1] * (FAULT_LINES // 2)
    rng.shuffle(gold)
    steps: list[StepSpec] = []
    gold_state: State = ()

    for name in names:
        atom = _random_nonzero_fraction(rng)
        instruction = Instruction("SET", action=Action("SET", name, atom))
        text = render_confirmation_instruction(
            instruction, comparator_variant=index % 3
        ).text
        step = StepSpec(text, fixed=instruction)
        steps.append(step)
        gold_state = apply_instruction(gold_state, instruction)

    evidence_specs = []
    for fault_index in range(FAULT_LINES):
        pair = OPERATION_PAIRS[(index + fault_index) % len(OPERATION_PAIRS)]
        accepted = None
        for _ in range(100):
            target = rng.choice(active)
            operand = _random_operand(rng, active, target)
            options = tuple(
                Instruction(operation, action=Action(operation, target, operand))
                for operation in pair
            )
            alternatives = tuple(
                apply_instruction(gold_state, instruction) for instruction in options
            )
            if _value(alternatives[0], target) != _value(
                alternatives[1], target
            ) and all(_bounded(value) for value in alternatives):
                accepted = options, alternatives, target
                break
        if accepted is None:
            return None
        options, alternatives, target = accepted
        step = StepSpec(
            _ambiguous_text(options),
            options=options,
            fault_index=fault_index,
        )
        steps.append(step)
        gold_state = alternatives[gold[fault_index]]
        evidence_specs.append((len(steps) - 1, target, _value(gold_state, target)))

        if fault_index % 3 == 2:
            left, right = rng.sample(active, 2)
            swap = Instruction("SWAP", swap_left=left, swap_right=right)
            steps.append(
                StepSpec(
                    render_confirmation_instruction(
                        swap, comparator_variant=(index + fault_index) % 3
                    ).text,
                    fixed=swap,
                )
            )
            gold_state = apply_instruction(gold_state, swap)

            predicate_left = rng.choice(active)
            predicate = Predicate(
                rng.choice(COMPARATORS),
                predicate_left,
                _random_operand(rng, active, predicate_left),
            )
            true_target = rng.choice(active)
            false_target = rng.choice(active)
            guard = Instruction(
                "GUARD",
                predicate=predicate,
                true_action=Action("ADD", true_target, _random_nonzero_fraction(rng)),
                false_action=Action(
                    "SUBTRACT", false_target, _random_nonzero_fraction(rng)
                ),
            )
            try:
                next_state = apply_instruction(gold_state, guard)
            except TOL1IRError:
                return None
            if not _bounded(next_state):
                return None
            steps.append(
                StepSpec(
                    render_confirmation_instruction(
                        guard, comparator_variant=(index * 7 + fault_index) % 3
                    ).text,
                    fixed=guard,
                )
            )
            gold_state = next_state

    source = (
        "Typed ambiguous state program:\n"
        + "\n".join(step.text for step in steps)
        + "\nEnd program."
    )
    source_commitment = hashlib.sha256(source.encode("ascii")).hexdigest()
    evidence = [
        {
            "source_commitment": source_commitment,
            "index": evidence_index,
            "step_index": step_index,
            "register": register,
            "value": format_fraction(value),
            "commitment": _evidence_commitment(
                source_commitment,
                evidence_index,
                step_index,
                register,
                format_fraction(value),
            ),
        }
        for evidence_index, (step_index, register, value) in enumerate(evidence_specs)
    ]

    terminal_by_assignment = {}
    signature_by_assignment = {}
    digest_rows = []
    for assignment in itertools.product((0, 1), repeat=FAULT_LINES):
        try:
            terminal, trajectory = execute_steps(steps, assignment)
        except TOL1IRError:
            return None
        if not _bounded(terminal):
            return None
        signature = tuple(
            format_fraction(_value(trajectory[item["step_index"]], item["register"]))
            for item in evidence
        )
        terminal_by_assignment[assignment] = terminal
        signature_by_assignment[assignment] = signature
        digest_rows.append(
            {
                "assignment": list(assignment),
                "terminal": state_record(terminal),
                "signature": list(signature),
            }
        )

    gold_assignment = tuple(gold)
    gold_signature = tuple(item["value"] for item in evidence)
    full_survivors = tuple(
        assignment
        for assignment, signature in signature_by_assignment.items()
        if signature == gold_signature
    )
    if full_survivors != (gold_assignment,):
        return None
    partial_signature = gold_signature[:-1]
    partial_survivors = tuple(
        assignment
        for assignment, signature in signature_by_assignment.items()
        if signature[:-1] == partial_signature
    )
    if len(partial_survivors) < 2:
        return None

    sensitivity = []
    for register in active:
        values = {
            _value(terminal_by_assignment[assignment], register)
            for assignment in terminal_by_assignment
        }
        partial_values = {
            _value(terminal_by_assignment[assignment], register)
            for assignment in partial_survivors
        }
        sensitivity.append((len(values), len(partial_values), register))
    under_candidates = [value for value in sensitivity if value[1] > 1]
    if not under_candidates:
        return None
    _, _, under_register = max(under_candidates)
    sensitive_candidates = [value for value in sensitivity if value[0] > 1]
    if not sensitive_candidates:
        return None
    _, _, sensitive_register = max(sensitive_candidates)
    gold_terminal = terminal_by_assignment[gold_assignment]
    gold_answer = format_fraction(_value(gold_terminal, sensitive_register))
    identity = hashlib.sha256(
        json.dumps(
            {
                "source_commitment": source_commitment,
                "gold_assignment": gold,
                "evidence": evidence,
                "queries": [sensitive_register, sentinel, under_register],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "id": f"tfs1-{index:06d}-{identity[:12]}",
        "identity_sha256": identity,
        "source": source,
        "source_commitment": source_commitment,
        "steps": [_step_record(step) for step in steps],
        "symbols": list(names),
        "active_symbols": list(active),
        "sentinel": sentinel,
        "gold_assignment": gold,
        "gold_terminal": state_record(gold_terminal),
        "gold_answer": gold_answer,
        "evidence": evidence,
        "queries": {
            "sensitive": _query_text(sensitive_register),
            "invariant": _query_text(sentinel),
            "underdetermined": _query_text(under_register),
        },
        "query_registers": {
            "sensitive": sensitive_register,
            "invariant": sentinel,
            "underdetermined": under_register,
        },
        "partial_survivors": len(partial_survivors),
        "enumeration_sha256": hashlib.sha256(
            json.dumps(digest_rows, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest(),
        "represented_worlds": WORLDS,
    }


def generate_row(rng: random.Random, *, index: int) -> dict[str, object]:
    for _ in range(100):
        row = _generate_candidate(rng, index=index)
        if row is not None:
            validate_row(row)
            return row
    raise TFS1DataError("could not generate a bounded TFS1 episode")


def generate_board(count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = [generate_row(rng, index=index) for index in range(count)]
    identities = [str(row["identity_sha256"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise TFS1DataError("duplicate TFS1 episode identity")
    return rows


def validate_row(row: Mapping[str, object]) -> None:
    if row.get("schema") != SCHEMA:
        raise TFS1DataError("TFS1 row schema differs")
    source = str(row["source"])
    if hashlib.sha256(source.encode("ascii")).hexdigest() != row["source_commitment"]:
        raise TFS1DataError("TFS1 source commitment differs")
    steps = steps_from_record(row["steps"])  # type: ignore[arg-type]
    expected_source = (
        "Typed ambiguous state program:\n"
        + "\n".join(step.text for step in steps)
        + "\nEnd program."
    )
    if source != expected_source:
        raise TFS1DataError("TFS1 document source differs")
    fault_indices = sorted(
        step.fault_index for step in steps if step.fault_index is not None
    )
    if fault_indices != list(range(FAULT_LINES)):
        raise TFS1DataError("TFS1 fault-line indices differ")
    assignment = tuple(int(value) for value in row["gold_assignment"])  # type: ignore[arg-type]
    if len(assignment) != FAULT_LINES or sum(assignment) != FAULT_LINES // 2:
        raise TFS1DataError("TFS1 gold assignment is not balanced")
    terminal, trajectory = execute_steps(steps, assignment)
    if state_record(terminal) != row["gold_terminal"]:
        raise TFS1DataError("TFS1 gold terminal differs")
    evidence = row["evidence"]
    if len(evidence) != FAULT_LINES:  # type: ignore[arg-type]
        raise TFS1DataError("TFS1 evidence count differs")
    for index, item in enumerate(evidence):  # type: ignore[union-attr]
        if item["source_commitment"] != row["source_commitment"]:
            raise TFS1DataError("TFS1 evidence source differs")
        if int(item["index"]) != index:
            raise TFS1DataError("TFS1 evidence order differs")
        step_index = int(item["step_index"])
        register = str(item["register"])
        value = format_fraction(_value(trajectory[step_index], register))
        if value != item["value"]:
            raise TFS1DataError("TFS1 evidence value differs")
        if item["commitment"] != _evidence_commitment(
            str(row["source_commitment"]), index, step_index, register, value
        ):
            raise TFS1DataError("TFS1 evidence commitment differs")
    sensitive = str(row["query_registers"]["sensitive"])  # type: ignore[index]
    if format_fraction(_value(terminal, sensitive)) != row["gold_answer"]:
        raise TFS1DataError("TFS1 gold answer differs")
    if int(row["represented_worlds"]) != WORLDS:
        raise TFS1DataError("TFS1 world count differs")

    signatures: dict[tuple[int, ...], tuple[str, ...]] = {}
    terminals: dict[tuple[int, ...], State] = {}
    digest_rows = []
    for candidate in itertools.product((0, 1), repeat=FAULT_LINES):
        candidate_terminal, candidate_trajectory = execute_steps(steps, candidate)
        signature = tuple(
            format_fraction(
                _value(
                    candidate_trajectory[int(item["step_index"])],
                    str(item["register"]),
                )
            )
            for item in evidence  # type: ignore[union-attr]
        )
        signatures[candidate] = signature
        terminals[candidate] = candidate_terminal
        digest_rows.append(
            {
                "assignment": list(candidate),
                "terminal": state_record(candidate_terminal),
                "signature": list(signature),
            }
        )

    gold_signature = tuple(str(item["value"]) for item in evidence)  # type: ignore[union-attr]
    full_survivors = tuple(
        candidate
        for candidate, signature in signatures.items()
        if signature == gold_signature
    )
    if full_survivors != (assignment,):
        raise TFS1DataError("TFS1 full evidence does not isolate gold")
    partial_survivors = tuple(
        candidate
        for candidate, signature in signatures.items()
        if signature[:-1] == gold_signature[:-1]
    )
    if len(partial_survivors) != int(row["partial_survivors"]):
        raise TFS1DataError("TFS1 partial survivor count differs")
    if len(partial_survivors) < 2:
        raise TFS1DataError("TFS1 partial evidence is not underdetermined")

    query_registers = row["query_registers"]  # type: ignore[assignment]
    invariant = str(query_registers["invariant"])  # type: ignore[index]
    underdetermined = str(query_registers["underdetermined"])  # type: ignore[index]
    if len({_value(state, sensitive) for state in terminals.values()}) < 2:
        raise TFS1DataError("TFS1 sensitive query is invariant")
    if len({_value(state, invariant) for state in terminals.values()}) != 1:
        raise TFS1DataError("TFS1 invariant query varies")
    if (
        len(
            {
                _value(terminals[candidate], underdetermined)
                for candidate in partial_survivors
            }
        )
        < 2
    ):
        raise TFS1DataError("TFS1 partial query is falsely invariant")

    enumeration_sha256 = hashlib.sha256(
        json.dumps(digest_rows, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if enumeration_sha256 != row["enumeration_sha256"]:
        raise TFS1DataError("TFS1 enumeration commitment differs")
    identity_sha256 = hashlib.sha256(
        json.dumps(
            {
                "source_commitment": row["source_commitment"],
                "gold_assignment": list(assignment),
                "evidence": evidence,
                "queries": [sensitive, invariant, underdetermined],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    if identity_sha256 != row["identity_sha256"]:
        raise TFS1DataError("TFS1 identity commitment differs")


__all__ = [
    "ACTIVE_REGISTERS",
    "FAULT_LINES",
    "OPERATION_PAIRS",
    "OPERATION_WORD",
    "REGISTER_COUNT",
    "SCHEMA",
    "StepSpec",
    "TFS1DataError",
    "TFS1_NAMES",
    "WORLDS",
    "State",
    "apply_instruction",
    "execute_steps",
    "generate_board",
    "generate_row",
    "state_from_record",
    "state_record",
    "steps_from_record",
    "validate_row",
]
