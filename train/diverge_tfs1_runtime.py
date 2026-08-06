"""Source-sealed compiler and exact factorized runtime for DIVERGE-TFS1."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import torch

from diverge_tfs1_data import (
    FAULT_LINES,
    REGISTER_COUNT,
    State,
    apply_instruction,
    state_record,
)
from diverge_tol1_data import source_candidates
from diverge_tol1_ir import (
    Action,
    Atom,
    DIRECT_OPS,
    Instruction,
    TOL1IRError,
    format_fraction,
    instruction_record,
    parse_fraction,
)
from diverge_tol2_anchor_decoder import (
    TOL2DecodeError,
    decode_predicate,
    decode_query,
    decode_swap,
    split_guard,
)
from diverge_tol3_semantic_anchor import (
    COMPARATOR_NAMES,
    LocalSemanticAnchor,
    TOL3AnchorError,
    decode_direct_action_from_anchor,
    runtime_comparator_phrase,
    select_operation_anchor,
    tensorize_texts,
)
from version_space_accounting import canonical_json_bytes


SCHEMA = "shohin-diverge-tfs1-packet-v1"
QUERY_SCHEMA = "shohin-diverge-tfs1-query-v1"
RECEIPT_SCHEMA = "shohin-diverge-tfs1-factorized-receipt-v1"
ANSWER = "ANSWER"
ABSTAIN = "ABSTAIN"
REJECT = "REJECT"


class TFS1RuntimeError(RuntimeError):
    """A compiler, packet, evidence, or execution contract is invalid."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_record(payload: object) -> str:
    return _sha256_bytes(canonical_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class AnchorProvenance:
    clause_sha256: str
    start: int
    end: int
    margin: float

    def record(self) -> dict[str, object]:
        return {
            "clause_sha256": self.clause_sha256,
            "start": self.start,
            "end": self.end,
            "margin_hex": self.margin.hex(),
        }


@dataclass(frozen=True, slots=True)
class FaultLine:
    index: int
    options: tuple[Instruction, Instruction]
    provenance: tuple[AnchorProvenance, AnchorProvenance]

    def __post_init__(self) -> None:
        if self.index < 0 or len(self.options) != 2 or len(self.provenance) != 2:
            raise TFS1RuntimeError("fault-line geometry differs")
        for option in self.options:
            option.validate()
            if option.operation not in DIRECT_OPS:
                raise TFS1RuntimeError("fault-line option is not direct")
        left = self.options[0].action
        right = self.options[1].action
        assert left and right
        if left.target != right.target or left.operand != right.operand:
            raise TFS1RuntimeError("fault-line options do not share arguments")
        if self.options[0].operation == self.options[1].operation:
            raise TFS1RuntimeError("fault-line options are not distinct")
        if any(
            item.margin <= 0.0 or not math.isfinite(item.margin)
            for item in self.provenance
        ):
            raise TFS1RuntimeError("fault-line support is not positive and finite")

    def record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "options": [instruction_record(value) for value in self.options],
            "provenance": [value.record() for value in self.provenance],
        }


@dataclass(frozen=True, slots=True)
class PacketStep:
    clause_sha256: str
    fixed: Instruction | None = None
    fault: FaultLine | None = None

    def __post_init__(self) -> None:
        if (self.fixed is None) == (self.fault is None):
            raise TFS1RuntimeError("packet step must be fixed or ambiguous")
        if self.fixed is not None:
            self.fixed.validate()
            if self.fixed.operation == "QUERY":
                raise TFS1RuntimeError("query cannot enter the source packet")

    def record(self) -> dict[str, object]:
        return {
            "clause_sha256": self.clause_sha256,
            "fixed": None if self.fixed is None else instruction_record(self.fixed),
            "fault": None if self.fault is None else self.fault.record(),
        }


@dataclass(frozen=True, slots=True)
class CompiledPacket:
    source_commitment: str
    compiler_commitment: str
    symbols: tuple[str, ...]
    steps: tuple[PacketStep, ...]

    def __post_init__(self) -> None:
        if len(self.source_commitment) != 64 or len(self.compiler_commitment) != 64:
            raise TFS1RuntimeError("packet commitment width differs")
        if (
            len(self.symbols) != REGISTER_COUNT
            or len(set(self.symbols)) != REGISTER_COUNT
        ):
            raise TFS1RuntimeError("packet symbol table differs")
        faults = [step.fault.index for step in self.steps if step.fault is not None]
        if faults != list(range(FAULT_LINES)):
            raise TFS1RuntimeError("packet fault-line indices differ")

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "source_commitment": self.source_commitment,
            "compiler_commitment": self.compiler_commitment,
            "symbols": list(self.symbols),
            "steps": [value.record() for value in self.steps],
        }

    @property
    def commitment(self) -> str:
        return _sha256_record(self.record())

    @property
    def static_bytes(self) -> int:
        return len(canonical_json_bytes(self.record()))


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    packet_commitment: str
    query_sha256: str
    register: str
    provenance: AnchorProvenance

    def record(self) -> dict[str, object]:
        return {
            "schema": QUERY_SCHEMA,
            "packet_commitment": self.packet_commitment,
            "query_sha256": self.query_sha256,
            "register": self.register,
            "provenance": self.provenance.record(),
        }


@dataclass(frozen=True, slots=True)
class EvidenceReceipt:
    index: int
    step_index: int
    register: str
    value: Fraction
    commitment: str


@dataclass(frozen=True, slots=True)
class StateGroup:
    support_mask: int
    state: State

    def __post_init__(self) -> None:
        if self.support_mask <= 0:
            raise TFS1RuntimeError("state group has empty support")

    def record(self) -> dict[str, object]:
        return {
            "support_mask": format(self.support_mask, "x"),
            "state": state_record(self.state),
        }


@dataclass(frozen=True, slots=True)
class FactorizedReceipt:
    groups: tuple[StateGroup, ...]
    represented_worlds: int
    unique_instruction_applications: int
    logical_instruction_applications: int
    peak_groups: int
    peak_group_bytes: int
    evidence_items: int
    rejected: bool
    rejection_reason: str | None

    def record(self) -> dict[str, object]:
        return {
            "schema": RECEIPT_SCHEMA,
            "groups": [value.record() for value in self.groups],
            "represented_worlds": self.represented_worlds,
            "unique_instruction_applications": self.unique_instruction_applications,
            "logical_instruction_applications": self.logical_instruction_applications,
            "peak_groups": self.peak_groups,
            "peak_group_bytes": self.peak_group_bytes,
            "evidence_items": self.evidence_items,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True, slots=True)
class QueryDecision:
    disposition: str
    answer: str | None
    represented_worlds: int


class LocalScorer:
    """Batched TOL3 inference without supervisor anchor dictionaries."""

    def __init__(self, model: LocalSemanticAnchor, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.operation_scores: dict[str, tuple[float, ...]] = {}
        self.comparator_scores: dict[str, tuple[float, ...]] = {}

    def _score(self, texts: Iterable[str]) -> None:
        missing = tuple(sorted(set(texts) - set(self.operation_scores)))
        if not missing:
            return
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(missing), 512):
                batch = missing[start : start + 512]
                ids, mask = tensorize_texts(batch, self.device)
                operations, comparators = self.model(ids, mask)
                for text, operation, comparator in zip(
                    batch,
                    operations.detach().float().cpu(),
                    comparators.detach().float().cpu(),
                    strict=True,
                ):
                    self.operation_scores[text] = tuple(
                        float(value) for value in operation
                    )
                    self.comparator_scores[text] = tuple(
                        float(value) for value in comparator
                    )

    def operation(self, text: str):
        words = [
            candidate.text
            for candidate in source_candidates(text)
            if candidate.kind == "WORD"
        ]
        self._score(words)
        return select_operation_anchor(text, self.operation_scores)

    def comparator(self, phrase: str) -> str:
        self._score((phrase,))
        scores = self.comparator_scores[phrase]
        return COMPARATOR_NAMES[max(range(len(scores)), key=scores.__getitem__)]


def _anchor_provenance(text: str, prediction) -> AnchorProvenance:
    return AnchorProvenance(
        _sha256_bytes(text.encode("ascii")),
        prediction.start,
        prediction.end,
        prediction.margin,
    )


def _decode_ambiguous_action(
    text: str,
    symbols: Sequence[str],
    scorer: LocalScorer,
    index: int,
) -> FaultLine:
    if text.count(" / ") != 1:
        raise TFS1RuntimeError("ambiguous clause delimiter differs")
    left_text, right_text = text.split(" / ", 1)
    left = scorer.operation(left_text)
    right = scorer.operation(right_text)
    if left.operation not in DIRECT_OPS or right.operation not in DIRECT_OPS:
        raise TFS1RuntimeError("ambiguous anchor is not a direct operation")
    if left.operation == right.operation:
        raise TFS1RuntimeError("ambiguous anchors collapse to one operation")

    candidates = tuple(
        value
        for value in source_candidates(right_text)
        if value.kind == "NUMBER" or value.text in set(symbols)
    )
    if len(candidates) != 2 or candidates[0].kind != "WORD":
        raise TFS1RuntimeError("ambiguous clause arguments differ")
    target, operand = candidates
    atom = Atom("CONST" if operand.kind == "NUMBER" else "REF", operand.text)
    options = tuple(
        Instruction(operation, action=Action(operation, target.text, atom))
        for operation in (left.operation, right.operation)
    )
    right_offset = len(left_text) + len(" / ")
    right_provenance = AnchorProvenance(
        _sha256_bytes(text.encode("ascii")),
        right.start + right_offset,
        right.end + right_offset,
        right.margin,
    )
    return FaultLine(
        index,
        options,  # type: ignore[arg-type]
        (
            _anchor_provenance(text, left),
            right_provenance,
        ),
    )


def _compile_guard(
    text: str, symbols: Sequence[str], scorer: LocalScorer
) -> Instruction:
    regions = split_guard(text)
    true_anchor = scorer.operation(regions.true_action)
    false_anchor = scorer.operation(regions.false_action)
    if (
        true_anchor.operation not in DIRECT_OPS
        or false_anchor.operation not in DIRECT_OPS
    ):
        raise TFS1RuntimeError("guard action is not direct")
    phrase = runtime_comparator_phrase(regions.predicate, symbols)
    comparator = scorer.comparator(phrase)
    instruction = Instruction(
        "GUARD",
        predicate=decode_predicate(regions.predicate, comparator, symbols),
        true_action=decode_direct_action_from_anchor(
            regions.true_action,
            true_anchor.operation,
            symbols,
            true_anchor.end,
        ),
        false_action=decode_direct_action_from_anchor(
            regions.false_action,
            false_anchor.operation,
            symbols,
            false_anchor.end,
        ),
    )
    instruction.validate()
    return instruction


def compile_source(
    model: LocalSemanticAnchor,
    source: str,
    *,
    expected_source_commitment: str,
    compiler_commitment: str,
    device: torch.device,
) -> tuple[CompiledPacket, LocalScorer]:
    try:
        source_bytes = source.encode("ascii")
    except UnicodeEncodeError as error:
        raise TFS1RuntimeError("source is not ASCII") from error
    if _sha256_bytes(source_bytes) != expected_source_commitment:
        raise TFS1RuntimeError("source commitment differs")
    prefix = "Typed ambiguous state program:\n"
    suffix = "\nEnd program."
    if not source.startswith(prefix) or not source.endswith(suffix):
        raise TFS1RuntimeError("source envelope differs")
    lines = source[len(prefix) : -len(suffix)].splitlines()
    if len(lines) != REGISTER_COUNT + FAULT_LINES + 8:
        raise TFS1RuntimeError("source line count differs")

    scorer = LocalScorer(model, device)
    scorer._score(
        candidate.text
        for line in lines
        for candidate in source_candidates(line)
        if candidate.kind == "WORD"
    )
    symbols: list[str] = []
    steps: list[PacketStep] = []
    fault_index = 0
    try:
        for line_index, line in enumerate(lines):
            clause_sha256 = _sha256_bytes(line.encode("ascii"))
            if line_index < REGISTER_COUNT:
                anchor = scorer.operation(line)
                if anchor.operation != "SET":
                    raise TFS1RuntimeError("declaration operation differs")
                action = decode_direct_action_from_anchor(line, "SET", None, anchor.end)
                if action.operand.kind != "CONST" or action.target in symbols:
                    raise TFS1RuntimeError("declaration binding differs")
                symbols.append(action.target)
                steps.append(
                    PacketStep(
                        clause_sha256,
                        fixed=Instruction("SET", action=action),
                    )
                )
                continue
            if " / " in line:
                fault = _decode_ambiguous_action(line, symbols, scorer, fault_index)
                steps.append(PacketStep(clause_sha256, fault=fault))
                fault_index += 1
                continue
            try:
                instruction = _compile_guard(line, symbols, scorer)
            except TOL2DecodeError:
                anchor = scorer.operation(line)
                if anchor.operation != "SWAP":
                    raise TFS1RuntimeError("fixed body operation differs")
                instruction = decode_swap(line, symbols)
            steps.append(PacketStep(clause_sha256, fixed=instruction))
    except (KeyError, TOL1IRError, TOL2DecodeError, TOL3AnchorError) as error:
        raise TFS1RuntimeError("source compilation failed") from error

    packet = CompiledPacket(
        expected_source_commitment,
        compiler_commitment,
        tuple(symbols),
        tuple(steps),
    )
    return packet, scorer


def compile_query(
    packet: CompiledPacket,
    scorer: LocalScorer,
    text: str,
) -> CompiledQuery:
    try:
        anchor = scorer.operation(text)
        if anchor.operation != "QUERY":
            raise TFS1RuntimeError("late query operation differs")
        instruction = decode_query(text, packet.symbols)
    except (TOL2DecodeError, TOL3AnchorError) as error:
        raise TFS1RuntimeError("late query compilation failed") from error
    assert instruction.query is not None
    return CompiledQuery(
        packet.commitment,
        _sha256_bytes(text.encode("ascii")),
        instruction.query,
        _anchor_provenance(text, anchor),
    )


def _evidence_commitment(
    source_commitment: str,
    index: int,
    step_index: int,
    register: str,
    value: str,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "source_commitment": source_commitment,
                "index": index,
                "step_index": step_index,
                "register": register,
                "value": value,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def verify_evidence(
    packet: CompiledPacket,
    records: Sequence[Mapping[str, object]],
) -> tuple[EvidenceReceipt, ...]:
    if len(records) > FAULT_LINES:
        raise TFS1RuntimeError("evidence count exceeds fault-line count")
    fault_steps = {
        step.fault.index: step_index
        for step_index, step in enumerate(packet.steps)
        if step.fault is not None
    }
    output = []
    for index, record in enumerate(records):
        if record.get("source_commitment") != packet.source_commitment:
            raise TFS1RuntimeError("evidence source commitment differs")
        if int(record.get("index", -1)) != index:
            raise TFS1RuntimeError("evidence order differs")
        step_index = int(record.get("step_index", -1))
        if fault_steps.get(index) != step_index:
            raise TFS1RuntimeError("evidence step provenance differs")
        fault = packet.steps[step_index].fault
        assert fault is not None
        action = fault.options[0].action
        assert action is not None
        register = str(record.get("register", ""))
        if register != action.target:
            raise TFS1RuntimeError("evidence register differs from fault target")
        value_text = str(record.get("value", ""))
        try:
            value = parse_fraction(value_text)
        except TOL1IRError as error:
            raise TFS1RuntimeError("evidence value is invalid") from error
        commitment = _evidence_commitment(
            packet.source_commitment,
            index,
            step_index,
            register,
            value_text,
        )
        if record.get("commitment") != commitment:
            raise TFS1RuntimeError("evidence receipt commitment differs")
        output.append(EvidenceReceipt(index, step_index, register, value, commitment))
    return tuple(output)


def assignment_from_index(index: int) -> tuple[int, ...]:
    if index < 0 or index >= 1 << FAULT_LINES:
        raise TFS1RuntimeError("assignment index is out of range")
    return tuple(
        (index >> (FAULT_LINES - 1 - fault_index)) & 1
        for fault_index in range(FAULT_LINES)
    )


def assignment_index(assignment: Sequence[int]) -> int:
    if len(assignment) != FAULT_LINES or any(
        value not in (0, 1) for value in assignment
    ):
        raise TFS1RuntimeError("assignment geometry differs")
    result = 0
    for value in assignment:
        result = (result << 1) | value
    return result


def _option_mask(fault_index: int, option: int) -> int:
    if fault_index < 0 or fault_index >= FAULT_LINES or option not in (0, 1):
        raise TFS1RuntimeError("literal is outside assignment geometry")
    stride = 1 << (FAULT_LINES - 1 - fault_index)
    period = 2 * stride
    mask = 0
    for start in range(option * stride, 1 << FAULT_LINES, period):
        mask |= ((1 << stride) - 1) << start
    return mask


def _groups_bytes(groups: Sequence[StateGroup]) -> int:
    return len(canonical_json_bytes([value.record() for value in groups]))


def _merge_groups(groups: Iterable[StateGroup]) -> tuple[StateGroup, ...]:
    support_by_state: dict[State, int] = {}
    for group in groups:
        support_by_state[group.state] = (
            support_by_state.get(group.state, 0) | group.support_mask
        )
    return tuple(
        StateGroup(mask, state)
        for state, mask in sorted(support_by_state.items(), key=lambda item: item[0])
    )


_SHIFTED_OPERATION = {
    "SET": "ADD",
    "ADD": "SUBTRACT",
    "SUBTRACT": "MULTIPLY",
    "MULTIPLY": "SET",
}


def _shift_direct(instruction: Instruction) -> Instruction:
    if instruction.operation not in DIRECT_OPS or instruction.action is None:
        raise TFS1RuntimeError("operation shift received a non-direct instruction")
    operation = _SHIFTED_OPERATION[instruction.operation]
    return Instruction(
        operation,
        action=Action(operation, instruction.action.target, instruction.action.operand),
    )


def _rejected_receipt(reason: str) -> FactorizedReceipt:
    return FactorizedReceipt((), 0, 0, 0, 0, 0, 0, True, reason)


def execute_factorized(
    packet: CompiledPacket,
    evidence_records: Sequence[Mapping[str, object]] = (),
    *,
    reset_after_declarations: bool = False,
    shift_fault_operations: bool = False,
) -> FactorizedReceipt:
    """Execute exact state groups, refining only from verified evidence."""

    try:
        evidence = verify_evidence(packet, evidence_records)
    except TFS1RuntimeError as error:
        return _rejected_receipt(str(error))
    evidence_by_step = {value.step_index: value for value in evidence}
    groups = (StateGroup((1 << (1 << FAULT_LINES)) - 1, ()),)
    peak_groups = 1
    peak_bytes = _groups_bytes(groups)
    unique = 0
    logical = 0

    for step_index, step in enumerate(packet.steps):
        if reset_after_declarations and step_index == REGISTER_COUNT:
            zero_state = tuple(
                (symbol, Fraction(0)) for symbol in sorted(packet.symbols)
            )
            groups = (
                StateGroup(sum(value.support_mask for value in groups), zero_state),
            )
        updated = []
        if step.fixed is not None:
            for group in groups:
                unique += 1
                logical += group.support_mask.bit_count()
                try:
                    next_state = apply_instruction(group.state, step.fixed)
                except TOL1IRError as error:
                    return _rejected_receipt(f"fixed execution failed: {error}")
                updated.append(StateGroup(group.support_mask, next_state))
        else:
            assert step.fault is not None
            for group in groups:
                for option_index, option in enumerate(step.fault.options):
                    active = group.support_mask & _option_mask(
                        step.fault.index, option_index
                    )
                    if not active:
                        continue
                    unique += 1
                    logical += active.bit_count()
                    instruction = (
                        _shift_direct(option) if shift_fault_operations else option
                    )
                    try:
                        next_state = apply_instruction(group.state, instruction)
                    except TOL1IRError as error:
                        return _rejected_receipt(f"fault execution failed: {error}")
                    updated.append(StateGroup(active, next_state))
        groups = _merge_groups(updated)
        peak_groups = max(peak_groups, len(groups))
        peak_bytes = max(peak_bytes, _groups_bytes(groups))

        observation = evidence_by_step.get(step_index)
        if observation is not None:
            groups = tuple(
                group
                for group in groups
                if dict(group.state).get(observation.register) == observation.value
            )
            if not groups:
                return _rejected_receipt("verified evidence leaves empty support")
            groups = _merge_groups(groups)

    represented = sum(value.support_mask.bit_count() for value in groups)
    return FactorizedReceipt(
        groups,
        represented,
        unique,
        logical,
        peak_groups,
        peak_bytes,
        len(evidence),
        False,
        None,
    )


def query_receipt(
    packet: CompiledPacket,
    receipt: FactorizedReceipt,
    query: CompiledQuery,
) -> QueryDecision:
    if query.packet_commitment != packet.commitment:
        return QueryDecision(REJECT, None, 0)
    if receipt.rejected or not receipt.groups:
        return QueryDecision(REJECT, None, 0)
    values = {dict(group.state).get(query.register) for group in receipt.groups}
    if None in values:
        return QueryDecision(REJECT, None, receipt.represented_worlds)
    if len(values) != 1:
        return QueryDecision(ABSTAIN, None, receipt.represented_worlds)
    value = values.pop()
    assert isinstance(value, Fraction)
    return QueryDecision(
        ANSWER,
        format_fraction(value),
        receipt.represented_worlds,
    )


def instantiate_instructions(
    packet: CompiledPacket,
    assignment: Sequence[int],
    *,
    shift_fault_operations: bool = False,
) -> tuple[Instruction, ...]:
    if len(assignment) != FAULT_LINES or any(
        value not in (0, 1) for value in assignment
    ):
        raise TFS1RuntimeError("assignment geometry differs")
    output = []
    for step in packet.steps:
        if step.fixed is not None:
            output.append(step.fixed)
            continue
        assert step.fault is not None
        instruction = step.fault.options[assignment[step.fault.index]]
        output.append(
            _shift_direct(instruction) if shift_fault_operations else instruction
        )
    return tuple(output)


def execute_assignment(
    packet: CompiledPacket,
    assignment: Sequence[int],
    *,
    reset_after_declarations: bool = False,
    shift_fault_operations: bool = False,
) -> tuple[State, tuple[State, ...]]:
    instructions = instantiate_instructions(
        packet, assignment, shift_fault_operations=shift_fault_operations
    )
    state: State = ()
    trajectory = []
    for index, instruction in enumerate(instructions):
        if reset_after_declarations and index == REGISTER_COUNT:
            state = tuple((symbol, Fraction(0)) for symbol in sorted(packet.symbols))
        try:
            state = apply_instruction(state, instruction)
        except TOL1IRError as error:
            raise TFS1RuntimeError("independent execution failed") from error
        trajectory.append(state)
    return state, tuple(trajectory)


def enumerate_packet(packet: CompiledPacket) -> dict[tuple[int, ...], State]:
    return {
        assignment: execute_assignment(packet, assignment)[0]
        for assignment in (
            assignment_from_index(index) for index in range(1 << FAULT_LINES)
        )
    }


def receipt_extensional_map(
    receipt: FactorizedReceipt,
) -> dict[tuple[int, ...], State]:
    if receipt.rejected:
        return {}
    output = {}
    for group in receipt.groups:
        active = group.support_mask
        while active:
            bit = active & -active
            index = bit.bit_length() - 1
            output[assignment_from_index(index)] = group.state
            active ^= bit
    return output


def assignment_score(packet: CompiledPacket, assignment: Sequence[int]) -> float:
    if len(assignment) != FAULT_LINES:
        raise TFS1RuntimeError("assignment score geometry differs")
    score = 0.0
    for step in packet.steps:
        if step.fault is not None:
            score += step.fault.provenance[assignment[step.fault.index]].margin
    return score


def ranked_assignments(packet: CompiledPacket) -> tuple[tuple[int, ...], ...]:
    assignments = tuple(
        assignment_from_index(index) for index in range(1 << FAULT_LINES)
    )
    return tuple(
        sorted(
            assignments,
            key=lambda assignment: (
                -assignment_score(packet, assignment),
                assignment,
            ),
        )
    )


def assignment_matches_evidence(
    packet: CompiledPacket,
    assignment: Sequence[int],
    evidence: Sequence[EvidenceReceipt],
    *,
    reset_after_declarations: bool = False,
    shift_fault_operations: bool = False,
) -> tuple[bool, State]:
    terminal, trajectory = execute_assignment(
        packet,
        assignment,
        reset_after_declarations=reset_after_declarations,
        shift_fault_operations=shift_fault_operations,
    )
    matches = all(
        dict(trajectory[item.step_index]).get(item.register) == item.value
        for item in evidence
    )
    return matches, terminal


def query_particles(
    packet: CompiledPacket,
    query: CompiledQuery,
    assignments: Sequence[Sequence[int]],
    evidence_records: Sequence[Mapping[str, object]],
) -> QueryDecision:
    if query.packet_commitment != packet.commitment:
        return QueryDecision(REJECT, None, 0)
    try:
        evidence = verify_evidence(packet, evidence_records)
    except TFS1RuntimeError:
        return QueryDecision(REJECT, None, 0)
    values = set()
    represented = 0
    for assignment in assignments:
        matches, terminal = assignment_matches_evidence(packet, assignment, evidence)
        if matches:
            represented += 1
            values.add(dict(terminal).get(query.register))
    if not represented or None in values:
        return QueryDecision(REJECT, None, represented)
    if len(values) != 1:
        return QueryDecision(ABSTAIN, None, represented)
    value = values.pop()
    assert isinstance(value, Fraction)
    return QueryDecision(ANSWER, format_fraction(value), represented)


def query_soft_answers(
    packet: CompiledPacket,
    query: CompiledQuery,
    evidence_records: Sequence[Mapping[str, object]],
) -> QueryDecision:
    if query.packet_commitment != packet.commitment:
        return QueryDecision(REJECT, None, 0)
    receipt = execute_factorized(packet, evidence_records)
    if receipt.rejected or not receipt.groups:
        return QueryDecision(REJECT, None, 0)
    weighted: dict[Fraction, float] = {}
    worlds = receipt_extensional_map(receipt)
    maximum = max(assignment_score(packet, assignment) for assignment in worlds)
    for assignment, terminal in worlds.items():
        value = dict(terminal).get(query.register)
        if value is None:
            return QueryDecision(REJECT, None, receipt.represented_worlds)
        weighted[value] = weighted.get(value, 0.0) + math.exp(
            assignment_score(packet, assignment) - maximum
        )
    answer = max(weighted, key=lambda value: (weighted[value], format_fraction(value)))
    return QueryDecision(
        ANSWER,
        format_fraction(answer),
        receipt.represented_worlds,
    )


def evidence_bytes(records: Sequence[Mapping[str, object]]) -> int:
    return len(canonical_json_bytes(list(records)))


def factorized_total_bytes(
    packet: CompiledPacket,
    receipt: FactorizedReceipt,
    evidence_records: Sequence[Mapping[str, object]],
) -> int:
    return (
        packet.static_bytes
        + receipt.peak_group_bytes
        + evidence_bytes(evidence_records)
    )


def whole_particle_record(
    packet: CompiledPacket,
    assignment: Sequence[int],
    evidence_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    terminal, _ = execute_assignment(packet, assignment)
    instructions = instantiate_instructions(packet, assignment)
    provenance = [
        step.fault.provenance[assignment[step.fault.index]].record()
        for step in packet.steps
        if step.fault is not None
    ]
    return {
        "source_commitment": packet.source_commitment,
        "compiler_commitment": packet.compiler_commitment,
        "symbols": list(packet.symbols),
        "assignment": list(assignment),
        "program": [instruction_record(value) for value in instructions],
        "terminal": state_record(terminal),
        "provenance": provenance,
        "support_score_hex": assignment_score(packet, assignment).hex(),
        "evidence": list(evidence_records),
    }


def whole_particle_bytes(
    packet: CompiledPacket,
    assignment: Sequence[int],
    evidence_records: Sequence[Mapping[str, object]],
) -> int:
    return len(
        canonical_json_bytes(
            whole_particle_record(packet, assignment, evidence_records)
        )
    )


def particle_capacity_for_bytes(
    packet: CompiledPacket,
    ranked: Sequence[Sequence[int]],
    evidence_records: Sequence[Mapping[str, object]],
    byte_budget: int,
) -> tuple[int, int]:
    used = 0
    count = 0
    for assignment in ranked:
        charge = whole_particle_bytes(packet, assignment, evidence_records)
        if used + charge > byte_budget:
            break
        used += charge
        count += 1
    return count, used


def all_particle_bytes(
    packet: CompiledPacket,
    evidence_records: Sequence[Mapping[str, object]],
) -> int:
    return sum(
        whole_particle_bytes(packet, assignment_from_index(index), evidence_records)
        for index in range(1 << FAULT_LINES)
    )


__all__ = [
    "ABSTAIN",
    "ANSWER",
    "REJECT",
    "AnchorProvenance",
    "CompiledPacket",
    "CompiledQuery",
    "FactorizedReceipt",
    "FaultLine",
    "LocalScorer",
    "PacketStep",
    "QueryDecision",
    "StateGroup",
    "TFS1RuntimeError",
    "all_particle_bytes",
    "assignment_from_index",
    "assignment_index",
    "assignment_matches_evidence",
    "assignment_score",
    "compile_query",
    "compile_source",
    "enumerate_packet",
    "evidence_bytes",
    "execute_assignment",
    "execute_factorized",
    "factorized_total_bytes",
    "instantiate_instructions",
    "particle_capacity_for_bytes",
    "query_particles",
    "query_receipt",
    "query_soft_answers",
    "ranked_assignments",
    "receipt_extensional_map",
    "verify_evidence",
    "whole_particle_bytes",
]
