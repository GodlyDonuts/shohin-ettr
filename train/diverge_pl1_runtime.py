#!/usr/bin/env python3
"""Small CPU reference for DIVERGE-PL1 policy-state mechanics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Literal

from diverge_pl1_data import Episode, OP_NAMES, Program, execute_mapping, verify_trace


Arm = Literal[
    "STATIC",
    "CONTEXT_ONLY",
    "DIVERGE_ONLY",
    "FAST_WEIGHT",
    "TRANSIENT_GRAD",
    "PL1",
]
CreditControl = Literal["normal", "shuffled", "wrong_branch", "no_eligibility"]
SIZE = len(OP_NAMES)


def _canonical_hash(domain: str, value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256()
    for part in (domain.encode("ascii"), payload):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def zero_matrix() -> list[list[float]]:
    return [[0.0 for _ in range(SIZE)] for _ in range(SIZE)]


def matrix_hash(matrix: list[list[float]]) -> str:
    rounded = [[round(value, 12) for value in row] for row in matrix]
    return _canonical_hash("diverge-pl1-policy", rounded)


def _gumbel(rng: random.Random) -> float:
    uniform = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
    return -math.log(-math.log(uniform))


def maximum_assignment(scores: list[list[float]]) -> tuple[int, ...]:
    """Return the exact maximum-weight bijection using bitmask dynamic programming."""

    if len(scores) != SIZE or any(len(row) != SIZE for row in scores):
        raise ValueError("assignment matrix shape differs")
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for symbol in range(SIZE):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (total, mapping) in states.items():
            for operation in range(SIZE):
                bit = 1 << operation
                if mask & bit:
                    continue
                candidate = (total + scores[symbol][operation], mapping + (operation,))
                new_mask = mask | bit
                incumbent = next_states.get(new_mask)
                if incumbent is None or candidate[0] > incumbent[0] + 1e-12:
                    next_states[new_mask] = candidate
                elif abs(candidate[0] - incumbent[0]) <= 1e-12 and candidate[1] < incumbent[1]:
                    next_states[new_mask] = candidate
        states = next_states
    return states[(1 << SIZE) - 1][1]


def sample_assignment(
    scores: list[list[float]], rng: random.Random, noise_scale: float = 1.0
) -> tuple[int, ...]:
    noisy = [
        [scores[symbol][operation] + noise_scale * _gumbel(rng) for operation in range(SIZE)]
        for symbol in range(SIZE)
    ]
    return maximum_assignment(noisy)


def _frob(matrix: list[list[float]]) -> float:
    return math.sqrt(sum(value * value for row in matrix for value in row))


def _project_write(update: list[list[float]], budget: float) -> tuple[list[list[float]], float]:
    norm = _frob(update)
    if norm <= budget or norm == 0.0:
        return update, norm
    scale = budget / norm
    return [[value * scale for value in row] for row in update], budget


@dataclass(frozen=True)
class BranchReceipt:
    branch: int
    mapping: tuple[int, ...]
    passed: bool
    correct_prefix: int
    program_depth: int
    receipt: str


@dataclass(frozen=True)
class WriteReceipt:
    attempt: int
    pre_hash: str
    post_hash: str
    update_norm: float
    cumulative_write: float
    protected_hash: str
    rejected_credits: int


@dataclass(frozen=True)
class EpisodeResult:
    arm: Arm
    episode_id: str
    selected_mapping: tuple[int, ...]
    mapping_exact: bool
    transfer_exact: int
    transfer_total: int
    attempt_passes: tuple[int, ...]
    probe_transfer_exact: tuple[int, ...]
    policy_hash: str
    policy_state: tuple[tuple[float, ...], ...]
    write_receipts: tuple[WriteReceipt, ...]


@dataclass(frozen=True)
class RollbackProbe:
    pre_hash: str
    poisoned_hash: str
    rollback_hash: str
    pre_outputs: tuple[tuple[int, int], ...]
    poisoned_outputs: tuple[tuple[int, int], ...]
    rollback_outputs: tuple[tuple[int, int], ...]

    @property
    def exact(self) -> bool:
        return self.pre_hash == self.rollback_hash and self.pre_outputs == self.rollback_outputs


def thaw_policy(policy_state: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    if len(policy_state) != SIZE or any(len(row) != SIZE for row in policy_state):
        raise ValueError("policy state shape differs")
    return [list(row) for row in policy_state]


def freeze_policy(policy_state: list[list[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(row) for row in policy_state)


def evaluate_policy_state(
    episode: Episode, policy_state: tuple[tuple[float, ...], ...]
) -> tuple[tuple[int, int], ...]:
    mapping = maximum_assignment(thaw_policy(policy_state))
    return tuple(execute_mapping(mapping, program)[-1] for program in episode.transfer)


def poison_and_rollback_probe(episode: Episode, result: EpisodeResult) -> RollbackProbe:
    before = thaw_policy(result.policy_state)
    pre_outputs = evaluate_policy_state(episode, freeze_policy(before))
    poisoned = [row[:] for row in before]
    wrong_mapping = list(episode.symbol_to_operation[1:]) + [episode.symbol_to_operation[0]]
    for symbol, operation in enumerate(wrong_mapping):
        poisoned[symbol][operation] += 64.0
    poisoned_state = freeze_policy(poisoned)
    poisoned_outputs = evaluate_policy_state(episode, poisoned_state)
    rollback = [row[:] for row in before]
    rollback_state = freeze_policy(rollback)
    return RollbackProbe(
        pre_hash=matrix_hash(before),
        poisoned_hash=matrix_hash(poisoned),
        rollback_hash=matrix_hash(rollback),
        pre_outputs=pre_outputs,
        poisoned_outputs=poisoned_outputs,
        rollback_outputs=evaluate_policy_state(episode, rollback_state),
    )


def _branch_receipts(
    *,
    episode: Episode,
    program: Program,
    mappings: tuple[tuple[int, ...], ...],
) -> tuple[BranchReceipt, ...]:
    receipts = []
    for branch, mapping in enumerate(mappings):
        trace = execute_mapping(mapping, program)
        verification = verify_trace(episode, program, trace)
        prefix = len(program.symbols) if verification.passed else int(verification.first_error) - 1
        receipts.append(
            BranchReceipt(
                branch=branch,
                mapping=mapping,
                passed=verification.passed,
                correct_prefix=prefix,
                program_depth=len(program.symbols),
                receipt=verification.receipt,
            )
        )
    return tuple(receipts)


def _credit_receipts(
    receipts: tuple[BranchReceipt, ...], control: CreditControl, rng: random.Random
) -> tuple[BranchReceipt, ...]:
    if control in {"normal", "no_eligibility"}:
        return receipts
    indices = list(range(len(receipts)))
    if control == "shuffled":
        rng.shuffle(indices)
    elif control == "wrong_branch":
        indices = indices[1:] + indices[:1]
    else:
        raise ValueError(f"unknown credit control {control}")
    reassigned = []
    for branch, source_index in enumerate(indices):
        branch_receipt = receipts[branch]
        credit = receipts[source_index]
        reassigned.append(
            BranchReceipt(
                branch=branch,
                mapping=branch_receipt.mapping,
                passed=credit.passed,
                correct_prefix=credit.correct_prefix,
                program_depth=branch_receipt.program_depth,
                receipt=credit.receipt,
            )
        )
    return tuple(reassigned)


def _pl1_update(
    episode: Episode,
    program: Program,
    receipts: tuple[BranchReceipt, ...],
    *,
    localized: bool,
) -> tuple[list[list[float]], int]:
    update = zero_matrix()
    rejected = 0
    for receipt in receipts:
        expected = verify_trace(episode, program, execute_mapping(receipt.mapping, program))
        if expected.receipt != receipt.receipt:
            rejected += 1
            continue
        if localized:
            for position in range(receipt.correct_prefix):
                symbol = program.symbols[position]
                update[symbol][receipt.mapping[symbol]] += 1.0
            if not receipt.passed and receipt.correct_prefix < len(program.symbols):
                symbol = program.symbols[receipt.correct_prefix]
                update[symbol][receipt.mapping[symbol]] -= 1.0
        else:
            reward = receipt.correct_prefix / receipt.program_depth
            reward = 1.0 if receipt.passed else 2.0 * reward - 1.0
            for symbol, operation in enumerate(receipt.mapping):
                update[symbol][operation] += reward
    scale = 1.0 / max(1, len(receipts))
    return [[value * scale for value in row] for row in update], rejected


def _fast_weight_update(receipts: tuple[BranchReceipt, ...]) -> list[list[float]]:
    update = zero_matrix()
    for receipt in receipts:
        reward = 1.0 if receipt.passed else -1.0
        for symbol, operation in enumerate(receipt.mapping):
            update[symbol][operation] += reward
    scale = 1.0 / max(1, len(receipts))
    return [[value * scale for value in row] for row in update]


def _transient_gradient_update(receipts: tuple[BranchReceipt, ...]) -> list[list[float]]:
    rewards = [receipt.correct_prefix / receipt.program_depth for receipt in receipts]
    baseline = sum(rewards) / len(rewards)
    update = zero_matrix()
    for receipt, reward in zip(receipts, rewards, strict=True):
        advantage = reward - baseline
        for symbol, operation in enumerate(receipt.mapping):
            update[symbol][operation] += advantage
    return [[value / len(receipts) for value in row] for row in update]


def run_episode(
    episode: Episode,
    *,
    arm: Arm,
    seed: int,
    branches: int = 8,
    write_budget: float = 4.0,
    credit_control: CreditControl = "normal",
    reset_before_transfer: bool = False,
    homeostatic: bool = True,
    inject_protected_mutation: bool = False,
) -> EpisodeResult:
    if branches <= 0:
        raise ValueError("branches must be positive")
    rng = random.Random(_canonical_hash("diverge-pl1-run", [episode.episode_id, arm, seed]))
    scores = zero_matrix()
    protected_manifest = {
        "world_owner": "immutable-world-owner-v1",
        "evidence_owner": "immutable-evidence-owner-v1",
        "referent_owner": "REFERENT_ORACLE",
        "executor": "exact-z97-executor-v1",
    }
    protected_hash = _canonical_hash("diverge-pl1-protected", protected_manifest)
    cumulative_write = 0.0
    writes = []
    attempt_passes = []
    probe_transfer_exact = []
    best_mapping = tuple(range(SIZE))
    best_prefix = -1

    for attempt, program in enumerate(episode.acquisition):
        mappings = tuple(sample_assignment(scores, rng) for _ in range(branches))
        receipts = _branch_receipts(episode=episode, program=program, mappings=mappings)
        attempt_passes.append(sum(receipt.passed for receipt in receipts))
        for receipt in receipts:
            if receipt.correct_prefix > best_prefix:
                best_mapping = receipt.mapping
                best_prefix = receipt.correct_prefix

        if arm in {"STATIC", "DIVERGE_ONLY"}:
            # Charge the same receipt validation and local update work, then discard it.
            matched_update, _ = _pl1_update(episode, program, receipts, localized=True)
            _project_write(matched_update, write_budget)
            probe_mapping = best_mapping if arm == "DIVERGE_ONLY" else maximum_assignment(scores)
            probe_transfer_exact.append(
                sum(
                    execute_mapping(probe_mapping, transfer)[-1] == transfer.terminal_state
                    for transfer in episode.transfer
                )
            )
            continue
        credited = _credit_receipts(receipts, credit_control, rng)
        if arm in {"CONTEXT_ONLY", "PL1"}:
            update, rejected_credits = _pl1_update(
                episode,
                program,
                credited,
                localized=credit_control != "no_eligibility",
            )
        elif arm == "FAST_WEIGHT":
            update = _fast_weight_update(credited)
            rejected_credits = 0
        elif arm == "TRANSIENT_GRAD":
            update = _transient_gradient_update(credited)
            rejected_credits = 0
        else:
            raise ValueError(f"unknown arm {arm}")

        pre_hash = matrix_hash(scores)
        if homeostatic:
            update, update_norm = _project_write(update, write_budget)
        else:
            update_norm = _frob(update)
        cumulative_write += update_norm
        if homeostatic:
            scores = [
                [
                    max(-8.0, min(8.0, scores[row][column] + update[row][column]))
                    for column in range(SIZE)
                ]
                for row in range(SIZE)
            ]
        else:
            scores = [
                [scores[row][column] + update[row][column] for column in range(SIZE)]
                for row in range(SIZE)
            ]
        if inject_protected_mutation and attempt == 0:
            protected_manifest["referent_owner"] = "MUTATED"
        if _canonical_hash("diverge-pl1-protected", protected_manifest) != protected_hash:
            raise RuntimeError("protected owner changed during plastic commit")
        writes.append(
            WriteReceipt(
                attempt=attempt,
                pre_hash=pre_hash,
                post_hash=matrix_hash(scores),
                update_norm=update_norm,
                cumulative_write=cumulative_write,
                protected_hash=protected_hash,
                rejected_credits=rejected_credits,
            )
        )
        probe_mapping = maximum_assignment(scores)
        probe_transfer_exact.append(
            sum(
                execute_mapping(probe_mapping, transfer)[-1] == transfer.terminal_state
                for transfer in episode.transfer
            )
        )

    if arm == "CONTEXT_ONLY" or reset_before_transfer:
        scores = zero_matrix()
    if arm == "DIVERGE_ONLY":
        selected = best_mapping
    else:
        selected = maximum_assignment(scores)
    transfer_exact = sum(
        execute_mapping(selected, program)[-1] == program.terminal_state
        for program in episode.transfer
    )
    return EpisodeResult(
        arm=arm,
        episode_id=episode.episode_id,
        selected_mapping=selected,
        mapping_exact=selected == episode.symbol_to_operation,
        transfer_exact=transfer_exact,
        transfer_total=len(episode.transfer),
        attempt_passes=tuple(attempt_passes),
        probe_transfer_exact=tuple(probe_transfer_exact),
        policy_hash=matrix_hash(scores),
        policy_state=freeze_policy(scores),
        write_receipts=tuple(writes),
    )
