#!/usr/bin/env python3
"""On-policy search-distillation escalation for autonomous SSQAC control.

This lane starts from the frozen search-distillation mechanics and adds a
discarded DAgger collector.  Bounded structural search is allowed only while
constructing labels.  Search programs, receipts, and all preparation source
matrices are hash-accounted and deleted before final candidate evaluation.

Four final policies are reinitialized from identical weights and receive
exactly equal example, parameter, batch-schedule, and optimizer-update budgets:

* ``on_policy_search_dagger``: search labels on collector-reached states.
* ``no_dagger_search_distillation``: an equal number of fresh off-policy
  search-trajectory states.
* ``ordinary_oracle_imitation``: ordinary scheduler labels on the treatment's
  exact observations.
* ``randomized_labels``: seeded legal alternatives on those observations.

Autonomous inference is the baseline's unmasked greedy action scorer plus the
primitive VM.  It cannot call search, an oracle, or a verifier.  Strict
``verify_reduction_program`` certification occurs afterward in the separate
assessor.  This remains a mechanics falsifier, not a native-reasoning claim.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import torch
from episode_functor_algebra_machine import (
    AlgebraInstruction,
    AlgebraMachineError,
    execute_program,
)
import ssqac_search_distillation_pilot as base


SCHEMA = "ssqac_on_policy_search_dagger_v1"
STATUS = "mechanics_falsifier_only_not_reasoning"
ARM_DAGGER = "on_policy_search_dagger"
ARM_NO_DAGGER = "no_dagger_search_distillation"
ARM_ORACLE = "ordinary_oracle_imitation"
ARM_RANDOM = "randomized_labels"
ARMS = (ARM_DAGGER, ARM_NO_DAGGER, ARM_ORACLE, ARM_RANDOM)
REFERENCE_BASELINE_CERTIFIED = 28
REFERENCE_BASELINE_TOTAL = 768
REFERENCE_EVALUATION_SEED_XOR = 0xE7A1


class OnPolicySearchDAggerError(ValueError):
    """The isolated on-policy escalation contract failed closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise OnPolicySearchDAggerError(
            "value is not canonical ASCII JSON data"
        ) from error


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OnPolicySearchDAggerError(f"{label} must be a positive integer")
    return value


def _matrix_manifest(
    matrices: Sequence[Iterable[Iterable[int]]],
) -> str:
    return _digest([base.matrix_sha256(matrix) for matrix in matrices])


def _state_manifest(states: Sequence[base.LabeledPolicyState]) -> str:
    return _digest(
        [
            {
                "label": state.label_sha256,
                "observation": state.observation_sha256,
            }
            for state in states
        ]
    )


def _observation_manifest(
    states: Sequence[base.LabeledPolicyState],
) -> str:
    return _digest([state.observation_sha256 for state in states])


def _merge_states(
    existing: Sequence[base.LabeledPolicyState],
    additions: Sequence[base.LabeledPolicyState],
) -> tuple[base.LabeledPolicyState, ...]:
    merged = {state.observation_sha256: state for state in existing}
    for state in additions:
        prior = merged.get(state.observation_sha256)
        if prior is not None and prior != state:
            raise OnPolicySearchDAggerError("aggregate contains conflicting labels")
        merged[state.observation_sha256] = state
    return tuple(merged[key] for key in sorted(merged))


def _randomized_alternative(
    rows: tuple[tuple[int, ...], ...],
    target: base.PolicyAction,
    *,
    seed: int,
) -> base.PolicyAction:
    legal = base.enumerate_policy_actions(rows)
    alternatives = tuple(action for action in legal if action != target)
    if not alternatives:
        return target
    material = f"{seed}:{base.matrix_sha256(rows)}".encode("ascii")
    index = int.from_bytes(sha256(material).digest()[:8], "big") % len(alternatives)
    return alternatives[index]


@dataclass(slots=True)
class SearchResources:
    search_calls: int = 0
    search_successes: int = 0
    search_failures: int = 0
    ordinary_oracle_calls: int = 0
    nodes_expanded: int = 0
    edges_considered: int = 0
    maximum_depth_reached: int = 0
    peak_frontier: int = 0
    trace_files_written: int = 0
    receipt_hashes: list[str] | None = None
    trace_hashes: list[str] | None = None

    def __post_init__(self) -> None:
        if self.receipt_hashes is None:
            self.receipt_hashes = []
        if self.trace_hashes is None:
            self.trace_hashes = []

    def absorb(self, other: "SearchResources") -> None:
        self.search_calls += other.search_calls
        self.search_successes += other.search_successes
        self.search_failures += other.search_failures
        self.ordinary_oracle_calls += other.ordinary_oracle_calls
        self.nodes_expanded += other.nodes_expanded
        self.edges_considered += other.edges_considered
        self.maximum_depth_reached = max(
            self.maximum_depth_reached,
            other.maximum_depth_reached,
        )
        self.peak_frontier = max(self.peak_frontier, other.peak_frontier)
        self.trace_files_written += other.trace_files_written
        assert self.receipt_hashes is not None
        assert self.trace_hashes is not None
        self.receipt_hashes.extend(other.receipt_hashes or ())
        self.trace_hashes.extend(other.trace_hashes or ())

    def canonical_data(self) -> dict[str, object]:
        return {
            "edges_considered": self.edges_considered,
            "maximum_depth_reached": self.maximum_depth_reached,
            "nodes_expanded": self.nodes_expanded,
            "ordinary_oracle_calls": self.ordinary_oracle_calls,
            "peak_frontier": self.peak_frontier,
            "receipt_manifest_sha256": _digest(self.receipt_hashes or ()),
            "search_calls": self.search_calls,
            "search_failures": self.search_failures,
            "search_successes": self.search_successes,
            "trace_files_written_then_deleted": self.trace_files_written,
            "trace_manifest_sha256": _digest(self.trace_hashes or ()),
        }


@dataclass(frozen=True, slots=True)
class SearchLabelBatch:
    search_states: tuple[base.LabeledPolicyState, ...]
    ordinary_states: tuple[base.LabeledPolicyState, ...]
    random_states: tuple[base.LabeledPolicyState, ...]
    resources: Mapping[str, object]
    trace_directory_deleted: bool
    retained_trace_files: int


@dataclass(frozen=True, slots=True)
class CollectedObservation:
    rows: tuple[tuple[int, ...], ...]
    pre_error: bool

    @property
    def sha256(self) -> str:
        return base.matrix_sha256(self.rows)


@dataclass(frozen=True, slots=True)
class CollectionReceipt:
    matrices: int
    reached_states: int
    unique_states: int
    retained_states: int
    pre_error_states: int
    halted: int
    vm_errors: int
    cycles: int
    macro_budget: int
    instruction_budget: int
    model_decisions: int
    vm_calls: int
    final_search_calls: int
    final_oracle_calls: int
    final_verifier_calls: int
    state_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class OffPolicyBatch:
    labels: SearchLabelBatch
    source_manifest_sha256: str


def _make_search_runtime(
    config: base.SearchPreparationConfig,
) -> tuple[object, object, object, object]:
    from ssqac_verifier_guided_search import (
        SearchBudget,
        SealedAlgebraPacket,
        WeakLocalScorer,
        bounded_beam_candidate_search,
    )

    budget = SearchBudget(
        max_nodes_expanded=config.max_nodes_expanded,
        max_edges_considered=config.max_edges_considered,
        max_depth=config.max_depth,
        max_frontier=config.max_frontier,
        beam_width=config.beam_width,
        max_program_instructions=config.max_program_instructions,
    )
    return (
        SealedAlgebraPacket,
        WeakLocalScorer,
        bounded_beam_candidate_search,
        budget,
    )


def _write_trace(
    trace_directory: Path,
    *,
    index: int,
    rows: tuple[tuple[int, ...], ...],
    result: object,
) -> tuple[str, str]:
    receipt = result.receipt
    program = result.program
    payload = {
        "index": index,
        "matrix_sha256": base.matrix_sha256(rows),
        "program": None
        if program is None
        else [instruction.canonical_data() for instruction in program],
        "receipt": asdict(receipt),
    }
    trace_bytes = _canonical_bytes(payload) + b"\n"
    path = trace_directory / f"trace_{index:08d}.json"
    path.write_bytes(trace_bytes)
    return (
        sha256(trace_bytes).hexdigest(),
        sha256(receipt.canonical_bytes()).hexdigest(),
    )


def _record_search_result(
    resources: SearchResources,
    result: object,
    *,
    trace_sha256: str,
    receipt_sha256: str,
) -> None:
    receipt = result.receipt
    resources.search_calls += 1
    resources.nodes_expanded += receipt.nodes_expanded
    resources.edges_considered += receipt.edges_considered
    resources.maximum_depth_reached = max(
        resources.maximum_depth_reached,
        receipt.maximum_depth_reached,
    )
    resources.peak_frontier = max(
        resources.peak_frontier,
        receipt.peak_frontier,
    )
    resources.trace_files_written += 1
    assert resources.trace_hashes is not None
    assert resources.receipt_hashes is not None
    resources.trace_hashes.append(trace_sha256)
    resources.receipt_hashes.append(receipt_sha256)
    if result.program is None:
        resources.search_failures += 1
    else:
        resources.search_successes += 1


def label_observations_with_search(
    observations: Sequence[CollectedObservation],
    *,
    seed: int,
    search_config: base.SearchPreparationConfig,
    scratch_root: Path | None,
) -> SearchLabelBatch:
    """Search-label reached states and delete every raw trace before return."""

    from ssqac_controller_trace_pilot import compile_reference_program

    (
        packet_type,
        scorer_type,
        search_function,
        budget,
    ) = _make_search_runtime(search_config)
    root = None if scratch_root is None else str(scratch_root)
    trace_directory = Path(tempfile.mkdtemp(prefix="ssqac-op-dagger-traces-", dir=root))
    resources = SearchResources()
    labeled: dict[
        str,
        tuple[
            tuple[tuple[int, ...], ...],
            base.PolicyAction,
            base.PolicyAction,
            base.PolicyAction,
        ],
    ] = {}
    try:
        for index, observation in enumerate(observations):
            rows = base.canonical_matrix(observation.rows)
            state_seed = seed ^ int(base.matrix_sha256(rows)[:16], 16)
            packet = packet_type.from_rows(rows, register_count=4)
            scorer = scorer_type(
                seed=state_seed,
                noise_scale=search_config.policy_noise_scale,
            )
            result = search_function(packet, scorer, budget)
            trace_sha, receipt_sha = _write_trace(
                trace_directory,
                index=index,
                rows=rows,
                result=result,
            )
            _record_search_result(
                resources,
                result,
                trace_sha256=trace_sha,
                receipt_sha256=receipt_sha,
            )
            if result.program is None:
                continue
            search_target = base.decode_macro_program(
                rows,
                result.program,
            )[0]
            ordinary_target = base.decode_macro_program(
                rows,
                compile_reference_program(rows),
            )[0]
            resources.ordinary_oracle_calls += 1
            random_target = _randomized_alternative(
                rows,
                search_target,
                seed=seed ^ 0xA11CE,
            )
            key = base.matrix_sha256(rows)
            value = (
                rows,
                search_target,
                ordinary_target,
                random_target,
            )
            prior = labeled.get(key)
            if prior is not None and prior != value:
                raise OnPolicySearchDAggerError("on-policy search labels conflict")
            labeled[key] = value
            del result
    finally:
        shutil.rmtree(trace_directory, ignore_errors=False)
    trace_deleted = not trace_directory.exists()
    if not trace_deleted:
        raise RuntimeError("on-policy search trace directory survived")
    ordered = tuple(labeled[key] for key in sorted(labeled))
    return SearchLabelBatch(
        search_states=tuple(
            base.LabeledPolicyState(rows=rows, target=search_target)
            for rows, search_target, _, _ in ordered
        ),
        ordinary_states=tuple(
            base.LabeledPolicyState(rows=rows, target=ordinary_target)
            for rows, _, ordinary_target, _ in ordered
        ),
        random_states=tuple(
            base.LabeledPolicyState(rows=rows, target=random_target)
            for rows, _, _, random_target in ordered
        ),
        resources=resources.canonical_data(),
        trace_directory_deleted=trace_deleted,
        retained_trace_files=0,
    )


def prepare_off_policy_batch(
    matrices: Sequence[Iterable[Iterable[int]]],
    *,
    seed: int,
    states_per_case: int,
    search_config: base.SearchPreparationConfig,
    scratch_root: Path | None,
) -> OffPolicyBatch:
    """Collect bounded search paths from fresh initial matrices."""

    _positive_int(states_per_case, label="states_per_case")
    from ssqac_controller_trace_pilot import compile_reference_program

    (
        packet_type,
        scorer_type,
        search_function,
        budget,
    ) = _make_search_runtime(search_config)
    frozen = tuple(base.canonical_matrix(matrix) for matrix in matrices)
    root = None if scratch_root is None else str(scratch_root)
    trace_directory = Path(
        tempfile.mkdtemp(prefix="ssqac-off-policy-traces-", dir=root)
    )
    resources = SearchResources()
    labeled: dict[
        str,
        tuple[
            tuple[tuple[int, ...], ...],
            base.PolicyAction,
            base.PolicyAction,
            base.PolicyAction,
        ],
    ] = {}
    try:
        for index, matrix in enumerate(frozen):
            state_seed = seed ^ int(base.matrix_sha256(matrix)[:16], 16)
            packet = packet_type.from_rows(matrix, register_count=4)
            scorer = scorer_type(
                seed=state_seed,
                noise_scale=search_config.policy_noise_scale,
            )
            result = search_function(packet, scorer, budget)
            trace_sha, receipt_sha = _write_trace(
                trace_directory,
                index=index,
                rows=matrix,
                result=result,
            )
            _record_search_result(
                resources,
                result,
                trace_sha256=trace_sha,
                receipt_sha256=receipt_sha,
            )
            if result.program is None:
                continue
            actions = base.decode_macro_program(matrix, result.program)
            rows = matrix
            for search_target in actions[:states_per_case]:
                ordinary_target = base.decode_macro_program(
                    rows,
                    compile_reference_program(rows),
                )[0]
                resources.ordinary_oracle_calls += 1
                random_target = _randomized_alternative(
                    rows,
                    search_target,
                    seed=seed ^ 0xA11CE,
                )
                key = base.matrix_sha256(rows)
                value = (
                    rows,
                    search_target,
                    ordinary_target,
                    random_target,
                )
                prior = labeled.get(key)
                if prior is not None and prior != value:
                    raise OnPolicySearchDAggerError("off-policy labels conflict")
                labeled[key] = value
                rows = base.apply_policy_action(rows, search_target)
            del actions, result
    finally:
        shutil.rmtree(trace_directory, ignore_errors=False)
    trace_deleted = not trace_directory.exists()
    if not trace_deleted:
        raise RuntimeError("off-policy search trace directory survived")
    ordered = tuple(labeled[key] for key in sorted(labeled))
    labels = SearchLabelBatch(
        search_states=tuple(
            base.LabeledPolicyState(rows=rows, target=search_target)
            for rows, search_target, _, _ in ordered
        ),
        ordinary_states=tuple(
            base.LabeledPolicyState(rows=rows, target=ordinary_target)
            for rows, _, ordinary_target, _ in ordered
        ),
        random_states=tuple(
            base.LabeledPolicyState(rows=rows, target=random_target)
            for rows, _, _, random_target in ordered
        ),
        resources=resources.canonical_data(),
        trace_directory_deleted=trace_deleted,
        retained_trace_files=0,
    )
    return OffPolicyBatch(
        labels=labels,
        source_manifest_sha256=_matrix_manifest(frozen),
    )


def _predict_action(
    model: base.EquivariantActionPolicy,
    rows: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
    amp_bfloat16: bool,
) -> base.PolicyAction:
    inputs, actions = base._single_observation_batch(rows, model.config)
    resident = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.no_grad(), base._autocast(device, amp_bfloat16):
        logits = model(**resident)
    selected = int(logits[0].argmax().detach().cpu().item())
    return actions[selected]


def collect_on_policy_observations(
    model: base.EquivariantActionPolicy,
    matrices: Sequence[Iterable[Iterable[int]]],
    *,
    device: torch.device,
    amp_bfloat16: bool,
    maximum_macros: int,
    maximum_instructions: int,
    states_per_case: int,
    state_budget: int,
) -> tuple[tuple[CollectedObservation, ...], CollectionReceipt]:
    """Collect model-reached states without search, oracle, or verification."""

    for name, value in (
        ("maximum_macros", maximum_macros),
        ("maximum_instructions", maximum_instructions),
        ("states_per_case", states_per_case),
        ("state_budget", state_budget),
    ):
        _positive_int(value, label=name)
    reached: list[CollectedObservation] = []
    halted = 0
    vm_errors = 0
    cycles = 0
    macro_budget = 0
    instruction_budget = 0
    model_decisions = 0
    vm_calls = 0
    model.eval()
    for raw_matrix in matrices:
        source = base.canonical_matrix(raw_matrix)
        program: list[AlgebraInstruction] = []
        visited: set[tuple[tuple[int, ...], ...]] = set()
        trajectory: list[tuple[tuple[int, ...], ...]] = []
        termination = "macro_budget"
        for _ in range(maximum_macros):
            try:
                state = execute_program(
                    source,
                    program,
                    maximum_instructions=maximum_instructions,
                )
            except AlgebraMachineError:
                vm_errors += 1
                termination = "vm_error"
                break
            vm_calls += 1
            rows = state.rows
            if rows in visited:
                cycles += 1
                termination = "cycle"
                break
            visited.add(rows)
            trajectory.append(rows)
            action = _predict_action(
                model,
                rows,
                device=device,
                amp_bfloat16=amp_bfloat16,
            )
            model_decisions += 1
            primitives = base.compile_candidate_action(rows, action)
            if len(program) + len(primitives) > maximum_instructions:
                instruction_budget += 1
                termination = "instruction_budget"
                break
            program.extend(primitives)
            if action.kind == base.ACTION_HALT:
                try:
                    execute_program(
                        source,
                        program,
                        maximum_instructions=maximum_instructions,
                    )
                except AlgebraMachineError:
                    vm_errors += 1
                    termination = "vm_error"
                else:
                    vm_calls += 1
                    halted += 1
                    termination = "halted"
                break
        else:
            macro_budget += 1
        if not trajectory:
            continue
        retained = trajectory[: max(0, states_per_case - 1)]
        if trajectory[-1] not in retained:
            retained.append(trajectory[-1])
        pre_error = termination in {
            "cycle",
            "instruction_budget",
            "macro_budget",
            "vm_error",
        }
        for index, rows in enumerate(retained):
            reached.append(
                CollectedObservation(
                    rows=rows,
                    pre_error=pre_error and index == len(retained) - 1,
                )
            )
    unique: dict[str, CollectedObservation] = {}
    for observation in reached:
        prior = unique.get(observation.sha256)
        if prior is None or observation.pre_error:
            unique[observation.sha256] = observation
    ordered = sorted(
        unique.values(),
        key=lambda item: (not item.pre_error, item.sha256),
    )
    retained_observations = tuple(ordered[:state_budget])
    receipt = CollectionReceipt(
        matrices=len(matrices),
        reached_states=len(reached),
        unique_states=len(unique),
        retained_states=len(retained_observations),
        pre_error_states=sum(
            observation.pre_error for observation in retained_observations
        ),
        halted=halted,
        vm_errors=vm_errors,
        cycles=cycles,
        macro_budget=macro_budget,
        instruction_budget=instruction_budget,
        model_decisions=model_decisions,
        vm_calls=vm_calls,
        final_search_calls=0,
        final_oracle_calls=0,
        final_verifier_calls=0,
        state_manifest_sha256=_digest(
            [observation.sha256 for observation in retained_observations]
        ),
    )
    return retained_observations, receipt


@dataclass(frozen=True, slots=True)
class DAggerRoundReport:
    round_index: int
    aggregate_examples_before: int
    collector_optimizer_updates: int
    collector_label_accuracy: float
    collector_model_sha256: str
    collection: CollectionReceipt
    on_policy_search_labeled_states: int
    no_dagger_off_policy_states: int
    aggregate_examples_after: int
    on_policy_search_resources: Mapping[str, object]
    no_dagger_search_resources: Mapping[str, object]
    all_search_traces_deleted: bool


@dataclass(frozen=True, slots=True)
class PreparationCustody:
    source_manifest_sha256: str
    source_files_written_then_deleted: int
    source_directory_deleted: bool
    retained_source_files: int
    all_search_trace_directories_deleted: bool
    retained_search_trace_files: int
    total_search_calls: int
    total_search_successes: int
    total_search_failures: int
    total_ordinary_oracle_calls: int
    total_nodes_expanded: int
    total_edges_considered: int
    search_receipt_manifest_sha256: str
    search_trace_manifest_sha256: str
    preparation_wall_seconds: float


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    filename: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class FinalArmReport:
    name: str
    examples: int
    parameters: int
    optimizer_updates: int
    initial_model_sha256: str
    final_model_sha256: str
    observation_manifest_sha256: str
    label_manifest_sha256: str
    tensor_manifest_sha256: str
    batch_schedule_sha256: str
    mean_loss: float
    final_loss: float
    label_accuracy: float
    certified: int
    invalid: int
    overlong: int
    certification_rate: float
    final_search_calls: int
    final_oracle_calls: int
    inference_verifier_calls: int
    assessor_verifier_calls: int
    model_decisions: int
    vm_calls: int
    evaluation_cases_sha256: str
    artifact: ModelArtifact | None


@dataclass(frozen=True, slots=True)
class EscalationReport:
    schema: str
    status: str
    reasoning_claim_authorized: bool
    seed: int
    slurm_job_id: str
    node: str
    device: str
    gpu_name: str
    slurm_cpus: int
    peak_cuda_memory_bytes: int
    reference_baseline_certified: int
    reference_baseline_total: int
    reference_baseline_rate: float
    reference_baseline_evaluation_seed_xor: int
    paired_reference_evaluation_cases: bool
    candidate_runtime: str
    assessor_boundary: str
    fit_maximum_rows: int
    fit_maximum_columns: int
    evaluation_minimum_rows: int
    evaluation_minimum_columns: int
    evaluation_maximum_rows: int
    evaluation_maximum_columns: int
    strict_geometry_disjoint: bool
    dagger_rounds: int
    controller_parameters: int
    policy_config_sha256: str
    search_config_sha256: str
    initial_source_manifest_sha256: str
    collection_source_manifest_sha256: str
    evaluation_manifest_sha256: str
    matched_examples: bool
    matched_parameters: bool
    matched_updates: bool
    matched_batch_schedule: bool
    identical_initial_weights: bool
    final_search_calls: int
    final_oracle_calls: int
    inference_verifier_calls: int
    preparation: PreparationCustody
    rounds: tuple[DAggerRoundReport, ...]
    arms: tuple[FinalArmReport, ...]
    training_wall_seconds: float
    evaluation_wall_seconds: float
    total_wall_seconds: float
    material_minimum_evaluation_cases: int
    material_minimum_certification_rate: float
    dagger_beats_no_dagger_margin: float
    dagger_beats_oracle_margin: float
    dagger_beats_random_margin: float
    material_mechanics_gate_passed: bool

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(asdict(self)) + b"\n"


def _resource_from_mapping(mapping: Mapping[str, object]) -> SearchResources:
    resource = SearchResources()
    resource.search_calls = int(mapping["search_calls"])
    resource.search_successes = int(mapping["search_successes"])
    resource.search_failures = int(mapping["search_failures"])
    resource.ordinary_oracle_calls = int(mapping["ordinary_oracle_calls"])
    resource.nodes_expanded = int(mapping["nodes_expanded"])
    resource.edges_considered = int(mapping["edges_considered"])
    resource.maximum_depth_reached = int(mapping["maximum_depth_reached"])
    resource.peak_frontier = int(mapping["peak_frontier"])
    resource.trace_files_written = int(mapping["trace_files_written_then_deleted"])
    resource.receipt_hashes = [str(mapping["receipt_manifest_sha256"])]
    resource.trace_hashes = [str(mapping["trace_manifest_sha256"])]
    return resource


def _write_source_batch(
    source_directory: Path,
    *,
    name: str,
    matrices: Sequence[Iterable[Iterable[int]]],
) -> tuple[str, str]:
    frozen = tuple(base.canonical_matrix(matrix) for matrix in matrices)
    payload = {
        "manifest_sha256": _matrix_manifest(frozen),
        "matrices": [[list(row) for row in matrix] for matrix in frozen],
        "name": name,
    }
    path = source_directory / f"{name}.json"
    path.write_bytes(_canonical_bytes(payload) + b"\n")
    return path.name, _file_sha256(path)


def _fresh_off_policy_additions(
    *,
    required: int,
    existing_observations: set[str],
    seed: int,
    fit_maximum_rows: int,
    fit_maximum_columns: int,
    states_per_case: int,
    search_config: base.SearchPreparationConfig,
    scratch_root: Path | None,
    source_directory: Path,
    round_index: int,
) -> tuple[
    tuple[base.LabeledPolicyState, ...],
    SearchResources,
    tuple[str, ...],
    bool,
]:
    if required == 0:
        return (), SearchResources(), (), True
    additions: dict[str, base.LabeledPolicyState] = {}
    resources = SearchResources()
    source_hashes: list[str] = []
    all_trace_directories_deleted = True
    attempt = 0
    while len(additions) < required:
        remaining = required - len(additions)
        case_count = max(
            4,
            (remaining + states_per_case - 1) // states_per_case + 2,
        )
        matrices = base.generate_matrix_cases(
            seed=seed ^ (round_index << 20) ^ attempt,
            count=case_count,
            minimum_rows=2,
            maximum_rows=fit_maximum_rows,
            minimum_columns=2,
            maximum_columns=fit_maximum_columns,
        )
        filename, file_hash = _write_source_batch(
            source_directory,
            name=f"no_dagger_round_{round_index:02d}_{attempt:03d}",
            matrices=matrices,
        )
        source_hashes.append(_digest([filename, file_hash]))
        batch = prepare_off_policy_batch(
            matrices,
            seed=seed ^ 0x0FFD00 ^ attempt,
            states_per_case=states_per_case,
            search_config=search_config,
            scratch_root=scratch_root,
        )
        resources.absorb(_resource_from_mapping(batch.labels.resources))
        all_trace_directories_deleted &= (
            batch.labels.trace_directory_deleted
            and batch.labels.retained_trace_files == 0
        )
        for state in batch.labels.search_states:
            key = state.observation_sha256
            if key in existing_observations or key in additions:
                continue
            additions[key] = state
            if len(additions) == required:
                break
        attempt += 1
        if attempt > 64:
            raise RuntimeError("could not fill matched no-DAgger off-policy budget")
    return (
        tuple(additions[key] for key in sorted(additions)),
        resources,
        tuple(source_hashes),
        all_trace_directories_deleted,
    )


def _save_model_artifact(
    model: base.EquivariantActionPolicy,
    *,
    arm: str,
    directory: Path,
    seed: int,
) -> ModelArtifact:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{arm}_seed{seed}.pt"
    if path.exists():
        raise OnPolicySearchDAggerError(f"model artifact already exists: {path}")
    payload = {
        "arm": arm,
        "model_schema": base.MODEL_SCHEMA,
        "policy_config": asdict(model.config),
        "seed": seed,
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
    }
    torch.save(payload, path)
    return ModelArtifact(
        filename=path.name,
        bytes=path.stat().st_size,
        sha256=_file_sha256(path),
    )


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise OnPolicySearchDAggerError("CUDA requested but unavailable")
    return device


def _validate_args(args: argparse.Namespace) -> None:
    if not (
        2
        <= args.fit_maximum_rows
        < args.evaluation_minimum_rows
        <= args.evaluation_maximum_rows
        <= args.maximum_rows
    ):
        raise OnPolicySearchDAggerError("row geometry holdout is not strict")
    if not (
        2
        <= args.fit_maximum_columns
        < args.evaluation_minimum_columns
        <= args.evaluation_maximum_columns
        <= args.maximum_columns
    ):
        raise OnPolicySearchDAggerError("column geometry holdout is not strict")
    for name in (
        "initial_cases",
        "initial_states_per_case",
        "dagger_rounds",
        "collection_cases",
        "collection_states_per_case",
        "on_policy_state_budget",
        "collector_epochs",
        "final_epochs",
        "batch_size",
        "maximum_rollout_macros",
        "maximum_rollout_instructions",
    ):
        _positive_int(getattr(args, name), label=name)


def run_pilot(args: argparse.Namespace) -> EscalationReport:
    _validate_args(args)
    total_start = perf_counter()
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    policy_config = base.PolicyConfig(
        maximum_rows=args.maximum_rows,
        maximum_columns=args.maximum_columns,
        width=args.width,
        blocks=args.blocks,
        feedforward=args.feedforward,
        dropout=0.0,
    )
    search_config = base.SearchPreparationConfig(
        max_nodes_expanded=args.search_max_nodes,
        max_edges_considered=args.search_max_edges,
        max_depth=args.search_max_depth,
        max_frontier=args.search_max_frontier,
        beam_width=args.search_beam_width,
        max_program_instructions=args.search_max_instructions,
        policy_noise_scale=float(args.search_policy_noise),
    )
    root = None if args.scratch_root is None else str(args.scratch_root)
    source_directory = Path(
        tempfile.mkdtemp(prefix="ssqac-op-dagger-source-", dir=root)
    )
    source_file_hashes: list[str] = []
    aggregate_resources = SearchResources()
    all_trace_dirs_deleted = True
    preparation_start = perf_counter()

    initial_matrices = base.generate_matrix_cases(
        seed=args.seed,
        count=args.initial_cases,
        minimum_rows=2,
        maximum_rows=args.fit_maximum_rows,
        minimum_columns=2,
        maximum_columns=args.fit_maximum_columns,
    )
    initial_source_manifest = _matrix_manifest(initial_matrices)
    source_name, source_hash = _write_source_batch(
        source_directory,
        name="initial_sources",
        matrices=initial_matrices,
    )
    source_file_hashes.append(_digest([source_name, source_hash]))
    initial_batch = prepare_off_policy_batch(
        initial_matrices,
        seed=args.seed,
        states_per_case=args.initial_states_per_case,
        search_config=search_config,
        scratch_root=args.scratch_root,
    )
    aggregate_resources.absorb(_resource_from_mapping(initial_batch.labels.resources))
    all_trace_dirs_deleted &= initial_batch.labels.trace_directory_deleted
    treatment = initial_batch.labels.search_states
    no_dagger = initial_batch.labels.search_states
    ordinary = initial_batch.labels.ordinary_states
    randomized = initial_batch.labels.random_states
    if not treatment:
        raise RuntimeError("initial search distillation produced no states")
    if not (len(treatment) == len(no_dagger) == len(ordinary) == len(randomized)):
        raise RuntimeError("initial arm data budgets differ")

    torch.manual_seed(args.seed)
    template = base.EquivariantActionPolicy(policy_config)
    initial_state = deepcopy(template.state_dict())
    initial_model_hash = base.model_state_sha256(template)
    round_reports: list[DAggerRoundReport] = []
    collection_source_hashes: list[str] = []
    for round_index in range(1, args.dagger_rounds + 1):
        collector = base.EquivariantActionPolicy(policy_config)
        collector.load_state_dict(initial_state)
        collector_dataset = base.tensorize_labeled_states(
            treatment,
            policy_config,
        )
        collector_schedule = base.build_batch_schedule(
            examples=collector_dataset.examples,
            epochs=args.collector_epochs,
            batch_size=args.batch_size,
            seed=args.seed ^ (round_index << 16) ^ 0xC011EC7,
        )
        collector_training = base.train_policy(
            collector,
            collector_dataset,
            schedule=collector_schedule,
            config=base.TrainingConfig(
                epochs=args.collector_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                amp_bfloat16=args.amp_bfloat16,
                torch_compile=args.collector_torch_compile,
            ),
            device=device,
        )
        collector_accuracy = base.label_accuracy(
            collector,
            collector_dataset,
            device=device,
            amp_bfloat16=args.amp_bfloat16,
        )
        collection_matrices = base.generate_matrix_cases(
            seed=args.seed ^ (round_index << 24) ^ 0xDA66E2,
            count=args.collection_cases,
            minimum_rows=2,
            maximum_rows=args.fit_maximum_rows,
            minimum_columns=2,
            maximum_columns=args.fit_maximum_columns,
        )
        collection_manifest = _matrix_manifest(collection_matrices)
        collection_source_hashes.append(collection_manifest)
        source_name, source_hash = _write_source_batch(
            source_directory,
            name=f"collection_round_{round_index:02d}",
            matrices=collection_matrices,
        )
        source_file_hashes.append(_digest([source_name, source_hash]))
        observations, collection_receipt = collect_on_policy_observations(
            collector,
            collection_matrices,
            device=device,
            amp_bfloat16=args.amp_bfloat16,
            maximum_macros=args.maximum_rollout_macros,
            maximum_instructions=args.maximum_rollout_instructions,
            states_per_case=args.collection_states_per_case,
            state_budget=args.on_policy_state_budget,
        )
        existing = {state.observation_sha256 for state in treatment}
        novel_observations = tuple(
            observation
            for observation in observations
            if observation.sha256 not in existing
        )
        on_policy_labels = label_observations_with_search(
            novel_observations,
            seed=args.seed ^ (round_index << 12) ^ 0x5EA2C4,
            search_config=search_config,
            scratch_root=args.scratch_root,
        )
        all_trace_dirs_deleted &= (
            on_policy_labels.trace_directory_deleted
            and on_policy_labels.retained_trace_files == 0
        )
        on_policy_resources = _resource_from_mapping(on_policy_labels.resources)
        aggregate_resources.absorb(on_policy_resources)
        additions = len(on_policy_labels.search_states)
        (
            no_dagger_additions,
            no_dagger_resources,
            no_dagger_sources,
            no_dagger_traces_deleted,
        ) = _fresh_off_policy_additions(
            required=additions,
            existing_observations={state.observation_sha256 for state in no_dagger},
            seed=args.seed ^ 0xBADC0DE,
            fit_maximum_rows=args.fit_maximum_rows,
            fit_maximum_columns=args.fit_maximum_columns,
            states_per_case=args.initial_states_per_case,
            search_config=search_config,
            scratch_root=args.scratch_root,
            source_directory=source_directory,
            round_index=round_index,
        )
        all_trace_dirs_deleted &= no_dagger_traces_deleted
        source_file_hashes.extend(no_dagger_sources)
        aggregate_resources.absorb(no_dagger_resources)
        before = len(treatment)
        treatment = _merge_states(
            treatment,
            on_policy_labels.search_states,
        )
        ordinary = _merge_states(
            ordinary,
            on_policy_labels.ordinary_states,
        )
        randomized = _merge_states(
            randomized,
            on_policy_labels.random_states,
        )
        no_dagger = _merge_states(no_dagger, no_dagger_additions)
        if not (len(treatment) == len(no_dagger) == len(ordinary) == len(randomized)):
            raise RuntimeError("post-round arm example budgets differ")
        round_reports.append(
            DAggerRoundReport(
                round_index=round_index,
                aggregate_examples_before=before,
                collector_optimizer_updates=(collector_training.optimizer_updates),
                collector_label_accuracy=collector_accuracy,
                collector_model_sha256=base.model_state_sha256(collector),
                collection=collection_receipt,
                on_policy_search_labeled_states=additions,
                no_dagger_off_policy_states=len(no_dagger_additions),
                aggregate_examples_after=len(treatment),
                on_policy_search_resources=on_policy_labels.resources,
                no_dagger_search_resources=(no_dagger_resources.canonical_data()),
                all_search_traces_deleted=(
                    on_policy_labels.trace_directory_deleted
                    and on_policy_labels.retained_trace_files == 0
                    and no_dagger_traces_deleted
                ),
            )
        )
        del (
            collector,
            collector_dataset,
            collector_schedule,
            collection_matrices,
            observations,
            novel_observations,
            on_policy_labels,
        )
        gc.collect()

    preparation_wall_seconds = perf_counter() - preparation_start
    final_aggregates = {
        ARM_DAGGER: treatment,
        ARM_NO_DAGGER: no_dagger,
        ARM_ORACLE: ordinary,
        ARM_RANDOM: randomized,
    }
    example_counts = {len(states) for states in final_aggregates.values()}
    if len(example_counts) != 1:
        raise RuntimeError("final arm example budgets differ")
    datasets = {
        arm: base.tensorize_labeled_states(states, policy_config)
        for arm, states in final_aggregates.items()
    }
    final_schedule = base.build_batch_schedule(
        examples=next(iter(example_counts)),
        epochs=args.final_epochs,
        batch_size=args.batch_size,
        seed=args.seed ^ 0xF1A1,
    )
    models = {arm: base.EquivariantActionPolicy(policy_config) for arm in ARMS}
    for model in models.values():
        model.load_state_dict(initial_state)
    initial_hashes = {
        arm: base.model_state_sha256(model) for arm, model in models.items()
    }
    if set(initial_hashes.values()) != {initial_model_hash}:
        raise RuntimeError("final models do not share initial weights")
    training_start = perf_counter()
    training_receipts: dict[str, base.TrainingReceipt] = {}
    label_accuracies: dict[str, float] = {}
    final_hashes: dict[str, str] = {}
    for arm in ARMS:
        training_receipts[arm] = base.train_policy(
            models[arm],
            datasets[arm],
            schedule=final_schedule,
            config=base.TrainingConfig(
                epochs=args.final_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                amp_bfloat16=args.amp_bfloat16,
                torch_compile=args.final_torch_compile,
            ),
            device=device,
        )
        label_accuracies[arm] = base.label_accuracy(
            models[arm],
            datasets[arm],
            device=device,
            amp_bfloat16=args.amp_bfloat16,
        )
        final_hashes[arm] = base.model_state_sha256(models[arm])
    training_wall_seconds = perf_counter() - training_start

    source_files = tuple(sorted(source_directory.iterdir()))
    source_manifest = _digest(
        [
            {
                "name": path.name,
                "sha256": _file_sha256(path),
            }
            for path in source_files
        ]
    )
    shutil.rmtree(source_directory, ignore_errors=False)
    source_deleted = not source_directory.exists()
    if not source_deleted:
        raise RuntimeError("preparation source directory survived deletion")
    dataset_receipts = {
        arm: {
            "examples": dataset.examples,
            "label_manifest": dataset.label_manifest_sha256,
            "observation_manifest": dataset.observation_manifest_sha256,
            "tensor_manifest": dataset.tensor_manifest_sha256,
        }
        for arm, dataset in datasets.items()
    }
    del (
        datasets,
        final_aggregates,
        treatment,
        no_dagger,
        ordinary,
        randomized,
        initial_matrices,
        initial_batch,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    evaluation_matrices = base.generate_matrix_cases(
        seed=args.seed ^ REFERENCE_EVALUATION_SEED_XOR,
        count=args.evaluation_cases,
        minimum_rows=args.evaluation_minimum_rows,
        maximum_rows=args.evaluation_maximum_rows,
        minimum_columns=args.evaluation_minimum_columns,
        maximum_columns=args.evaluation_maximum_columns,
    )
    evaluation_manifest = _matrix_manifest(evaluation_matrices)
    evaluation_start = perf_counter()
    evaluations = {
        arm: base.evaluate_autonomous_policy(
            models[arm],
            evaluation_matrices,
            device=device,
            amp_bfloat16=args.amp_bfloat16,
            maximum_macros=args.maximum_rollout_macros,
            maximum_instructions=args.maximum_rollout_instructions,
        )
        for arm in ARMS
    }
    evaluation_wall_seconds = perf_counter() - evaluation_start
    artifacts: dict[str, ModelArtifact | None] = {arm: None for arm in ARMS}
    if args.model_output_directory is not None:
        for arm in ARMS:
            artifacts[arm] = _save_model_artifact(
                models[arm],
                arm=arm,
                directory=args.model_output_directory,
                seed=args.seed,
            )
    arm_reports: list[FinalArmReport] = []
    for arm in ARMS:
        evaluation = evaluations[arm]
        training = training_receipts[arm]
        receipt = dataset_receipts[arm]
        arm_reports.append(
            FinalArmReport(
                name=arm,
                examples=int(receipt["examples"]),
                parameters=models[arm].parameter_count,
                optimizer_updates=training.optimizer_updates,
                initial_model_sha256=initial_hashes[arm],
                final_model_sha256=final_hashes[arm],
                observation_manifest_sha256=str(receipt["observation_manifest"]),
                label_manifest_sha256=str(receipt["label_manifest"]),
                tensor_manifest_sha256=str(receipt["tensor_manifest"]),
                batch_schedule_sha256=training.batch_schedule_sha256,
                mean_loss=training.mean_loss,
                final_loss=training.final_loss,
                label_accuracy=label_accuracies[arm],
                certified=evaluation.certified,
                invalid=evaluation.invalid,
                overlong=evaluation.overlong,
                certification_rate=evaluation.certification_rate,
                final_search_calls=evaluation.final_search_calls,
                final_oracle_calls=evaluation.final_oracle_calls,
                inference_verifier_calls=0,
                assessor_verifier_calls=evaluation.assessor_verifier_calls,
                model_decisions=evaluation.model_decisions,
                vm_calls=evaluation.vm_calls,
                evaluation_cases_sha256=_digest(
                    [asdict(case) for case in evaluation.cases]
                ),
                artifact=artifacts[arm],
            )
        )
    updates = {report.optimizer_updates for report in arm_reports}
    parameters = {report.parameters for report in arm_reports}
    schedules = {report.batch_schedule_sha256 for report in arm_reports}
    dagger_rate = evaluations[ARM_DAGGER].certification_rate
    no_dagger_rate = evaluations[ARM_NO_DAGGER].certification_rate
    oracle_rate = evaluations[ARM_ORACLE].certification_rate
    random_rate = evaluations[ARM_RANDOM].certification_rate
    material_gate = (
        args.evaluation_cases >= args.minimum_material_evaluation_cases
        and dagger_rate >= args.minimum_material_certification_rate
        and dagger_rate - no_dagger_rate >= args.dagger_beats_no_dagger_margin
        and dagger_rate - oracle_rate >= args.dagger_beats_oracle_margin
        and dagger_rate - random_rate >= args.dagger_beats_random_margin
        and all(
            report.final_search_calls == 0
            and report.final_oracle_calls == 0
            and report.inference_verifier_calls == 0
            for report in arm_reports
        )
    )
    peak_cuda = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    preparation = PreparationCustody(
        source_manifest_sha256=source_manifest,
        source_files_written_then_deleted=len(source_files),
        source_directory_deleted=source_deleted,
        retained_source_files=0,
        all_search_trace_directories_deleted=all_trace_dirs_deleted,
        retained_search_trace_files=0,
        total_search_calls=aggregate_resources.search_calls,
        total_search_successes=aggregate_resources.search_successes,
        total_search_failures=aggregate_resources.search_failures,
        total_ordinary_oracle_calls=(aggregate_resources.ordinary_oracle_calls),
        total_nodes_expanded=aggregate_resources.nodes_expanded,
        total_edges_considered=aggregate_resources.edges_considered,
        search_receipt_manifest_sha256=_digest(
            aggregate_resources.receipt_hashes or ()
        ),
        search_trace_manifest_sha256=_digest(aggregate_resources.trace_hashes or ()),
        preparation_wall_seconds=preparation_wall_seconds,
    )
    return EscalationReport(
        schema=SCHEMA,
        status=STATUS,
        reasoning_claim_authorized=False,
        seed=args.seed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        node=os.environ.get("SLURMD_NODENAME", ""),
        device=str(device),
        gpu_name=(torch.cuda.get_device_name(device) if device.type == "cuda" else ""),
        slurm_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0),
        peak_cuda_memory_bytes=peak_cuda,
        reference_baseline_certified=REFERENCE_BASELINE_CERTIFIED,
        reference_baseline_total=REFERENCE_BASELINE_TOTAL,
        reference_baseline_rate=(
            REFERENCE_BASELINE_CERTIFIED / REFERENCE_BASELINE_TOTAL
        ),
        reference_baseline_evaluation_seed_xor=(REFERENCE_EVALUATION_SEED_XOR),
        paired_reference_evaluation_cases=True,
        candidate_runtime="unmasked_greedy_model_plus_primitive_vm_only",
        assessor_boundary=(
            "strict_verify_reduction_program_after_candidate_termination"
        ),
        fit_maximum_rows=args.fit_maximum_rows,
        fit_maximum_columns=args.fit_maximum_columns,
        evaluation_minimum_rows=args.evaluation_minimum_rows,
        evaluation_minimum_columns=args.evaluation_minimum_columns,
        evaluation_maximum_rows=args.evaluation_maximum_rows,
        evaluation_maximum_columns=args.evaluation_maximum_columns,
        strict_geometry_disjoint=True,
        dagger_rounds=args.dagger_rounds,
        controller_parameters=next(iter(parameters)),
        policy_config_sha256=policy_config.sha256,
        search_config_sha256=_digest(asdict(search_config)),
        initial_source_manifest_sha256=initial_source_manifest,
        collection_source_manifest_sha256=_digest(collection_source_hashes),
        evaluation_manifest_sha256=evaluation_manifest,
        matched_examples=len({report.examples for report in arm_reports}) == 1,
        matched_parameters=len(parameters) == 1,
        matched_updates=len(updates) == 1,
        matched_batch_schedule=len(schedules) == 1,
        identical_initial_weights=len(set(initial_hashes.values())) == 1,
        final_search_calls=sum(report.final_search_calls for report in arm_reports),
        final_oracle_calls=sum(report.final_oracle_calls for report in arm_reports),
        inference_verifier_calls=sum(
            report.inference_verifier_calls for report in arm_reports
        ),
        preparation=preparation,
        rounds=tuple(round_reports),
        arms=tuple(arm_reports),
        training_wall_seconds=training_wall_seconds,
        evaluation_wall_seconds=evaluation_wall_seconds,
        total_wall_seconds=perf_counter() - total_start,
        material_minimum_evaluation_cases=(args.minimum_material_evaluation_cases),
        material_minimum_certification_rate=(args.minimum_material_certification_rate),
        dagger_beats_no_dagger_margin=(args.dagger_beats_no_dagger_margin),
        dagger_beats_oracle_margin=args.dagger_beats_oracle_margin,
        dagger_beats_random_margin=args.dagger_beats_random_margin,
        material_mechanics_gate_passed=material_gate,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--initial-cases", type=int, default=256)
    parser.add_argument("--initial-states-per-case", type=int, default=24)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--collection-cases", type=int, default=96)
    parser.add_argument("--collection-states-per-case", type=int, default=8)
    parser.add_argument("--on-policy-state-budget", type=int, default=512)
    parser.add_argument("--fit-maximum-rows", type=int, default=4)
    parser.add_argument("--fit-maximum-columns", type=int, default=6)
    parser.add_argument("--evaluation-minimum-rows", type=int, default=5)
    parser.add_argument("--evaluation-minimum-columns", type=int, default=7)
    parser.add_argument("--evaluation-maximum-rows", type=int, default=5)
    parser.add_argument("--evaluation-maximum-columns", type=int, default=8)
    parser.add_argument("--evaluation-cases", type=int, default=256)
    parser.add_argument("--maximum-rows", type=int, default=6)
    parser.add_argument("--maximum-columns", type=int, default=8)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--feedforward", type=int, default=1536)
    parser.add_argument("--collector-epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--amp-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--collector-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--final-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--search-max-nodes", type=int, default=8_192)
    parser.add_argument("--search-max-edges", type=int, default=200_000)
    parser.add_argument("--search-max-depth", type=int, default=48)
    parser.add_argument("--search-max-frontier", type=int, default=128)
    parser.add_argument("--search-beam-width", type=int, default=128)
    parser.add_argument("--search-max-instructions", type=int, default=256)
    parser.add_argument("--search-policy-noise", type=float, default=8.0)
    parser.add_argument("--maximum-rollout-macros", type=int, default=192)
    parser.add_argument(
        "--maximum-rollout-instructions",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--minimum-material-evaluation-cases",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--minimum-material-certification-rate",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--dagger-beats-no-dagger-margin",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--dagger-beats-oracle-margin",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--dagger-beats-random-margin",
        type=float,
        default=0.08,
    )
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--model-output-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.device = "cpu"
        args.initial_cases = 4
        args.initial_states_per_case = 4
        args.dagger_rounds = 1
        args.collection_cases = 4
        args.collection_states_per_case = 3
        args.on_policy_state_budget = 4
        args.width = 32
        args.blocks = 1
        args.feedforward = 64
        args.collector_epochs = 1
        args.final_epochs = 1
        args.batch_size = 8
        args.amp_bfloat16 = False
        args.collector_torch_compile = False
        args.final_torch_compile = False
        args.search_max_nodes = 512
        args.search_max_edges = 10_000
        args.search_max_depth = 24
        args.search_max_frontier = 32
        args.search_beam_width = 32
        args.search_max_instructions = 96
        args.maximum_rollout_macros = 48
        args.maximum_rollout_instructions = 160
        args.evaluation_minimum_rows = 4
        args.evaluation_minimum_columns = 5
        args.evaluation_maximum_rows = 4
        args.evaluation_maximum_columns = 6
        args.fit_maximum_rows = 3
        args.fit_maximum_columns = 4
        args.evaluation_cases = 8
        args.minimum_material_evaluation_cases = 8
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_pilot(args)
    payload = report.canonical_bytes()
    if args.output is None:
        print(payload.decode("ascii"), end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise OnPolicySearchDAggerError(f"report already exists: {args.output}")
        args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
