"""Rejected gauge-equivariant counterfactual repair mechanics.

This prototype is retained to reproduce a closed negative architecture audit.
Its machine-shaped evidence tensors can carry the exact target machine, so its
counterfactual features do not establish source-derived repair.  It must not be
integrated, fitted, or counted in the complete-system parameter receipt.

See ``R12_EFC_COUNTERFACTUAL_MACHINE_REPAIR_LATTICE.md`` for the closure
decision and successor requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from episode_functor_runtime_constants import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)


COUNTERFACTUAL_REPAIR_MODES: Final = frozenset(
    {
        "causal",
        "observational-twin",
        "candidate-averaged",
        "one-step-myope",
        "commutative-counterfactual",
        "unsigned-repair",
        "fixed-cycle",
    }
)
COUNTERFACTUAL_FEATURE_WIDTH: Final = 128
ROW_FEATURE_WIDTH: Final = 64
DEFAULT_CONTROLLER_WIDTH: Final = 704
DEFAULT_MEMORY_WIDTH: Final = 384
DEFAULT_PARAMETER_COUNT: Final = 9_618_567
COUNTERFACTUAL_REPAIR_ADMITTED: Final = False


class CounterfactualRepairError(ValueError):
    """Counterfactual repair geometry, values, or configuration failed."""


@dataclass(frozen=True, slots=True)
class CounterfactualRepairResult:
    """Completed-cycle repair tensors and adaptive mixture diagnostics."""

    transition_probabilities: torch.Tensor
    observer_probabilities: torch.Tensor
    cycle_transition_probabilities: tuple[torch.Tensor, ...]
    cycle_observer_probabilities: tuple[torch.Tensor, ...]
    cycle_halt_probabilities: torch.Tensor
    cycle_mixture_weights: torch.Tensor
    mode: str

    def __post_init__(self) -> None:
        batch = int(self.transition_probabilities.shape[0])
        cycles = len(self.cycle_transition_probabilities)
        if (
            self.transition_probabilities.shape
            != (
                batch,
                PRIMARY_ACTIONS,
                PRIMARY_STATES,
                PRIMARY_STATES,
            )
            or self.observer_probabilities.shape
            != (
                batch,
                PRIMARY_OBSERVERS,
                PRIMARY_STATES,
                PRIMARY_ANSWERS,
            )
            or cycles < 1
            or len(self.cycle_observer_probabilities) != cycles
            or self.cycle_halt_probabilities.shape != (batch, cycles)
            or self.cycle_mixture_weights.shape != (batch, cycles)
            or self.mode not in COUNTERFACTUAL_REPAIR_MODES
        ):
            raise CounterfactualRepairError(
                "counterfactual repair result geometry differs"
            )
        values = (
            self.transition_probabilities,
            self.observer_probabilities,
            self.cycle_halt_probabilities,
            self.cycle_mixture_weights,
            *self.cycle_transition_probabilities,
            *self.cycle_observer_probabilities,
        )
        if any(
            not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
            for value in values
        ):
            raise CounterfactualRepairError(
                "counterfactual repair result values differ"
            )


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.down(F.silu(self.up(self.norm(values))))


def _validate_and_normalize(
    name: str,
    values: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if (
        values.shape != shape
        or not values.is_floating_point()
        or not bool(torch.isfinite(values).all())
        or bool(values.lt(0).any())
    ):
        raise CounterfactualRepairError(f"{name} differs")
    total = values.sum(-1, keepdim=True)
    if bool(total.le(0).any()):
        raise CounterfactualRepairError(f"{name} has empty rows")
    normalized = values / total
    return normalized / normalized.sum(-1, keepdim=True)


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(probabilities.dtype).tiny
    return -(
        probabilities
        * probabilities.clamp_min(tiny).log()
    ).sum(-1)


def _reduce_mean(values: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, values.ndim))
    if not dimensions:
        return values
    return values.mean(dimensions)


def _behavior_signatures(
    transition: torch.Tensor,
    observer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    depth_zero = observer
    depth_one = torch.einsum(
        "bast,boty->baosy",
        transition,
        observer,
    )
    depth_two = torch.einsum(
        "bast,bctu,bouy->bacosy",
        transition,
        transition,
        observer,
    )
    depth_three = torch.einsum(
        "bast,bctu,bduv,bovy->bacdosy",
        transition,
        transition,
        transition,
        observer,
    )
    return depth_zero, depth_one, depth_two, depth_three


def _commutative_signature(
    signature: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    if depth < 2:
        return signature
    action_axes = tuple(range(1, depth + 1))
    permutations: tuple[tuple[int, ...], ...]
    if depth == 2:
        permutations = ((0, 1), (1, 0))
    elif depth == 3:
        permutations = (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    else:
        raise CounterfactualRepairError(
            "unsupported commutative signature depth"
        )
    outputs = []
    suffix = tuple(range(depth + 1, signature.ndim))
    for order in permutations:
        outputs.append(
            signature.permute(
                (0,)
                + tuple(action_axes[index] for index in order)
                + suffix
            )
        )
    return torch.stack(outputs).mean(0)


def _signature_errors(
    transition: torch.Tensor,
    observer: torch.Tensor,
    target_transition: torch.Tensor,
    target_observer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = _behavior_signatures(transition, observer)
    targets = _behavior_signatures(target_transition, target_observer)
    ordered = []
    commutative = []
    for depth, (prediction, target) in enumerate(
        zip(predictions, targets, strict=True)
    ):
        ordered.append(_reduce_mean((prediction - target).square()))
        commutative.append(
            _reduce_mean(
                (
                    _commutative_signature(prediction, depth)
                    - _commutative_signature(target, depth)
                ).square()
            )
        )
    return torch.stack(ordered, -1), torch.stack(commutative, -1)


def _expand_scalar_bank(values: torch.Tensor) -> torch.Tensor:
    absolute = values.abs()
    expanded = torch.stack(
        (
            values,
            absolute,
            values.square(),
            values.tanh(),
            values / (1.0 + absolute),
            F.relu(values),
            F.relu(-values),
            torch.exp(-absolute),
        ),
        dim=-1,
    )
    return expanded.flatten(-2)


def _mode_features(
    scalar_bank: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "one-step-myope":
        scalar_bank = scalar_bank.clone()
        scalar_bank[..., 6:8] = 0.0
        scalar_bank[..., 10:12] = 0.0
    elif mode == "commutative-counterfactual":
        scalar_bank = scalar_bank.clone()
        scalar_bank[..., 4:8] = scalar_bank[..., 8:12]
    elif mode == "unsigned-repair":
        scalar_bank = scalar_bank.abs()
    features = _expand_scalar_bank(scalar_bank)
    if mode == "observational-twin":
        features = features.mean(-2, keepdim=True).expand_as(features)
    elif mode == "candidate-averaged":
        candidates = int(features.shape[-2])
        features = (
            features.sum(-2, keepdim=True) - features
        ) / float(candidates - 1)
    return features


def _candidate_scalar_bank(
    current_row: torch.Tensor,
    target_row: torch.Tensor,
    candidate_ordered_error: torch.Tensor,
    candidate_commutative_error: torch.Tensor,
    current_ordered_error: torch.Tensor,
    current_commutative_error: torch.Tensor,
    global_machine_delta: torch.Tensor,
) -> torch.Tensor:
    batch, categories = current_row.shape
    identity = torch.eye(
        categories,
        dtype=current_row.dtype,
        device=current_row.device,
    )[None].expand(batch, -1, -1)
    current_at_candidate = current_row
    target_at_candidate = target_row
    local_base = (current_row - target_row).square().mean(-1, keepdim=True)
    local_candidate = (
        identity - target_row[:, None]
    ).square().mean(-1)
    local_delta = local_candidate - local_base
    ordered_delta = (
        candidate_ordered_error
        - current_ordered_error[:, None]
    )
    commutative_delta = (
        candidate_commutative_error
        - current_commutative_error[:, None]
    )
    entropy_current = _entropy(current_row)[:, None].expand(
        -1,
        categories,
    )
    entropy_target = _entropy(target_row)[:, None].expand(
        -1,
        categories,
    )
    alternative_margin = (
        current_row.amax(-1, keepdim=True) - current_row
    )
    return torch.cat(
        (
            current_at_candidate[..., None],
            target_at_candidate[..., None],
            (target_at_candidate - current_at_candidate)[..., None],
            local_delta[..., None],
            ordered_delta,
            commutative_delta,
            global_machine_delta[..., None],
            entropy_current[..., None],
            entropy_target[..., None],
            alternative_margin[..., None],
        ),
        dim=-1,
    )


class CounterfactualMachineRepair(nn.Module):
    """Shared finite-intervention repair controller with adaptive mixing."""

    def __init__(
        self,
        *,
        width: int = DEFAULT_CONTROLLER_WIDTH,
        memory_width: int = DEFAULT_MEMORY_WIDTH,
        cycles: int = 4,
    ) -> None:
        super().__init__()
        if (
            width < 16
            or memory_width < 8
            or not 1 <= cycles <= 8
        ):
            raise CounterfactualRepairError(
                "counterfactual repair configuration differs"
            )
        self.width = int(width)
        self.memory_width = int(memory_width)
        self.cycles = int(cycles)

        self.candidate_stem = nn.Linear(
            COUNTERFACTUAL_FEATURE_WIDTH,
            self.width,
        )
        self.row_stem = nn.Linear(ROW_FEATURE_WIDTH, self.width)
        self.memory_initializer = nn.Linear(
            32,
            self.memory_width,
        )
        self.residual_blocks = nn.ModuleList(
            (_ResidualBlock(self.width), _ResidualBlock(self.width))
        )
        self.conflict_memory = nn.GRUCell(
            self.width,
            self.memory_width,
        )
        self.memory_projection = nn.Linear(
            self.memory_width,
            self.width,
            bias=False,
        )
        self.candidate_norm = nn.LayerNorm(self.width)
        self.candidate_readout = nn.Linear(self.width, 1)
        self.row_gate = nn.Linear(
            self.width + self.memory_width,
            1,
        )
        self.halt_head = nn.Linear(self.memory_width, 1)
        self.log_scales = nn.Parameter(torch.zeros(4))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _row_features(
        rows: torch.Tensor,
        target_rows: torch.Tensor,
        transition: torch.Tensor,
        observer: torch.Tensor,
        target_transition: torch.Tensor,
        target_observer: torch.Tensor,
    ) -> torch.Tensor:
        batch, row_count, _ = rows.shape
        transition_error = (
            transition - target_transition
        ).square().mean((1, 2, 3))
        observer_error = (
            observer - target_observer
        ).square().mean((1, 2, 3))
        scalars = torch.stack(
            (
                _entropy(rows),
                _entropy(target_rows),
                (rows - target_rows).square().mean(-1),
                rows.amax(-1),
                target_rows.amax(-1),
                transition_error[:, None].expand(-1, row_count),
                observer_error[:, None].expand(-1, row_count),
                torch.ones(
                    (batch, row_count),
                    dtype=rows.dtype,
                    device=rows.device,
                ),
            ),
            dim=-1,
        )
        return _expand_scalar_bank(scalars)

    @staticmethod
    def _intervention_features(
        transition: torch.Tensor,
        observer: torch.Tensor,
        target_transition: torch.Tensor,
        target_observer: torch.Tensor,
        *,
        row_kind: str,
        mode: str,
    ) -> torch.Tensor:
        batch = int(transition.shape[0])
        current_ordered, current_commutative = _signature_errors(
            transition,
            observer,
            target_transition,
            target_observer,
        )
        feature_rows = []
        if row_kind == "transition":
            rows = transition.reshape(
                batch,
                PRIMARY_ACTIONS * PRIMARY_STATES,
                PRIMARY_STATES,
            )
            targets = target_transition.reshape_as(rows)
            categories = PRIMARY_STATES
        elif row_kind == "observer":
            rows = observer.reshape(
                batch,
                PRIMARY_OBSERVERS * PRIMARY_STATES,
                PRIMARY_ANSWERS,
            )
            targets = target_observer.reshape_as(rows)
            categories = PRIMARY_ANSWERS
        else:
            raise CounterfactualRepairError(
                "counterfactual row kind differs"
            )
        identity = torch.eye(
            categories,
            dtype=rows.dtype,
            device=rows.device,
        )
        current_machine_error = (
            transition - target_transition
        ).square().mean((1, 2, 3)) + (
            observer - target_observer
        ).square().mean((1, 2, 3))
        for row_index in range(int(rows.shape[1])):
            candidate_transition = transition[:, None].expand(
                -1,
                categories,
                -1,
                -1,
                -1,
            )
            candidate_observer = observer[:, None].expand(
                -1,
                categories,
                -1,
                -1,
                -1,
            )
            if row_kind == "transition":
                flat = candidate_transition.reshape(
                    batch,
                    categories,
                    PRIMARY_ACTIONS * PRIMARY_STATES,
                    PRIMARY_STATES,
                ).clone()
                flat[:, :, row_index] = identity[None]
                candidate_transition = flat.reshape(
                    batch * categories,
                    PRIMARY_ACTIONS,
                    PRIMARY_STATES,
                    PRIMARY_STATES,
                )
                candidate_observer = candidate_observer.reshape(
                    batch * categories,
                    PRIMARY_OBSERVERS,
                    PRIMARY_STATES,
                    PRIMARY_ANSWERS,
                )
            else:
                flat = candidate_observer.reshape(
                    batch,
                    categories,
                    PRIMARY_OBSERVERS * PRIMARY_STATES,
                    PRIMARY_ANSWERS,
                ).clone()
                flat[:, :, row_index] = identity[None]
                candidate_observer = flat.reshape(
                    batch * categories,
                    PRIMARY_OBSERVERS,
                    PRIMARY_STATES,
                    PRIMARY_ANSWERS,
                )
                candidate_transition = candidate_transition.reshape(
                    batch * categories,
                    PRIMARY_ACTIONS,
                    PRIMARY_STATES,
                    PRIMARY_STATES,
                )
            repeated_target_transition = target_transition.repeat_interleave(
                categories,
                dim=0,
            )
            repeated_target_observer = target_observer.repeat_interleave(
                categories,
                dim=0,
            )
            candidate_ordered, candidate_commutative = _signature_errors(
                candidate_transition,
                candidate_observer,
                repeated_target_transition,
                repeated_target_observer,
            )
            candidate_ordered = candidate_ordered.reshape(
                batch,
                categories,
                4,
            )
            candidate_commutative = candidate_commutative.reshape(
                batch,
                categories,
                4,
            )
            candidate_machine_error = (
                candidate_transition - repeated_target_transition
            ).square().mean((1, 2, 3)) + (
                candidate_observer - repeated_target_observer
            ).square().mean((1, 2, 3))
            global_delta = candidate_machine_error.reshape(
                batch,
                categories,
            ) - current_machine_error[:, None]
            feature_rows.append(
                _mode_features(
                    _candidate_scalar_bank(
                        rows[:, row_index],
                        targets[:, row_index],
                        candidate_ordered,
                        candidate_commutative,
                        current_ordered,
                        current_commutative,
                        global_delta,
                    ),
                    mode,
                )
            )
        return torch.stack(feature_rows, dim=1)

    def _repair_rows(
        self,
        rows: torch.Tensor,
        target_rows: torch.Tensor,
        candidate_features: torch.Tensor,
        row_features: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, row_count, categories = rows.shape
        if (
            candidate_features.shape
            != (
                batch,
                row_count,
                categories,
                COUNTERFACTUAL_FEATURE_WIDTH,
            )
            or row_features.shape
            != (batch, row_count, ROW_FEATURE_WIDTH)
            or memory.shape
            != (batch, row_count, self.memory_width)
        ):
            raise CounterfactualRepairError(
                "counterfactual controller geometry differs"
            )
        scales = F.softplus(self.log_scales) + 0.1
        hidden = (
            self.candidate_stem(candidate_features)
            + self.row_stem(row_features)[:, :, None]
            + scales[1]
            * self.memory_projection(memory)[:, :, None]
        )
        for block in self.residual_blocks:
            hidden = block(hidden)
        scores = self.candidate_readout(
            self.candidate_norm(hidden)
        ).squeeze(-1)
        candidate_probability = (
            scores / scales[0]
        ).softmax(-1)
        pooled = (
            hidden * candidate_probability[..., None]
        ).sum(-2)
        flat_memory = memory.reshape(-1, self.memory_width)
        revised_memory = self.conflict_memory(
            pooled.reshape(-1, self.width),
            flat_memory,
        ).reshape_as(memory)
        gate = torch.sigmoid(
            scales[2]
            * self.row_gate(
                torch.cat((pooled, revised_memory), dim=-1)
            )
        )
        revised = (
            (1.0 - gate) * rows + gate * candidate_probability
        )
        revised = revised / revised.sum(-1, keepdim=True)
        if revised.shape != target_rows.shape:
            raise CounterfactualRepairError(
                "counterfactual repaired rows differ"
            )
        return revised, revised_memory

    def forward(
        self,
        transition_probabilities: torch.Tensor,
        observer_probabilities: torch.Tensor,
        transition_evidence: torch.Tensor,
        observer_evidence: torch.Tensor,
        *,
        mode: str = "causal",
    ) -> CounterfactualRepairResult:
        """Run every repair cycle and adaptively mix completed machines."""

        if mode not in COUNTERFACTUAL_REPAIR_MODES:
            raise CounterfactualRepairError(
                f"unknown counterfactual repair mode: {mode}"
            )
        batch = int(transition_probabilities.shape[0])
        transition_shape = (
            batch,
            PRIMARY_ACTIONS,
            PRIMARY_STATES,
            PRIMARY_STATES,
        )
        observer_shape = (
            batch,
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            PRIMARY_ANSWERS,
        )
        transition = _validate_and_normalize(
            "transition probabilities",
            transition_probabilities,
            transition_shape,
        )
        observer = _validate_and_normalize(
            "observer probabilities",
            observer_probabilities,
            observer_shape,
        )
        target_transition = _validate_and_normalize(
            "transition evidence",
            transition_evidence,
            transition_shape,
        )
        target_observer = _validate_and_normalize(
            "observer evidence",
            observer_evidence,
            observer_shape,
        )

        transition_rows = transition.reshape(
            batch,
            PRIMARY_ACTIONS * PRIMARY_STATES,
            PRIMARY_STATES,
        )
        target_transition_rows = target_transition.reshape_as(
            transition_rows
        )
        observer_rows = observer.reshape(
            batch,
            PRIMARY_OBSERVERS * PRIMARY_STATES,
            PRIMARY_ANSWERS,
        )
        target_observer_rows = target_observer.reshape_as(observer_rows)
        transition_row_features = self._row_features(
            transition_rows,
            target_transition_rows,
            transition,
            observer,
            target_transition,
            target_observer,
        )
        observer_row_features = self._row_features(
            observer_rows,
            target_observer_rows,
            transition,
            observer,
            target_transition,
            target_observer,
        )
        transition_memory = self.memory_initializer(
            transition_row_features[..., :32]
        )
        observer_memory = self.memory_initializer(
            observer_row_features[..., :32]
        )
        transition_cycles = []
        observer_cycles = []
        halt_cycles = []

        for _ in range(self.cycles):
            transition = transition_rows.reshape(transition_shape)
            observer = observer_rows.reshape(observer_shape)
            transition_features = self._intervention_features(
                transition,
                observer,
                target_transition,
                target_observer,
                row_kind="transition",
                mode=mode,
            )
            observer_features = self._intervention_features(
                transition,
                observer,
                target_transition,
                target_observer,
                row_kind="observer",
                mode=mode,
            )
            transition_row_features = self._row_features(
                transition_rows,
                target_transition_rows,
                transition,
                observer,
                target_transition,
                target_observer,
            )
            observer_row_features = self._row_features(
                observer_rows,
                target_observer_rows,
                transition,
                observer,
                target_transition,
                target_observer,
            )
            transition_rows, transition_memory = self._repair_rows(
                transition_rows,
                target_transition_rows,
                transition_features,
                transition_row_features,
                transition_memory,
            )
            observer_rows, observer_memory = self._repair_rows(
                observer_rows,
                target_observer_rows,
                observer_features,
                observer_row_features,
                observer_memory,
            )
            transition = transition_rows.reshape(transition_shape)
            observer = observer_rows.reshape(observer_shape)
            combined_memory = torch.cat(
                (transition_memory, observer_memory),
                dim=1,
            ).mean(1)
            halt = torch.sigmoid(
                (F.softplus(self.log_scales[3]) + 0.1)
                * self.halt_head(combined_memory).squeeze(-1)
            )
            transition_cycles.append(transition)
            observer_cycles.append(observer)
            halt_cycles.append(halt)

        halt_probabilities = torch.stack(halt_cycles, -1)
        if mode == "fixed-cycle":
            halt_probabilities = torch.zeros_like(halt_probabilities)
            halt_probabilities[:, -1] = 1.0
        remaining = torch.ones(
            batch,
            dtype=transition.dtype,
            device=transition.device,
        )
        mixture_weights = []
        for cycle, halt in enumerate(halt_probabilities.unbind(-1)):
            weight = remaining if cycle == self.cycles - 1 else remaining * halt
            mixture_weights.append(weight)
            remaining = remaining - weight
        mixture = torch.stack(mixture_weights, -1)
        transition = sum(
            weight[:, None, None, None] * value
            for weight, value in zip(
                mixture.unbind(-1),
                transition_cycles,
                strict=True,
            )
        )
        observer = sum(
            weight[:, None, None, None] * value
            for weight, value in zip(
                mixture.unbind(-1),
                observer_cycles,
                strict=True,
            )
        )
        transition = transition / transition.sum(-1, keepdim=True)
        observer = observer / observer.sum(-1, keepdim=True)
        return CounterfactualRepairResult(
            transition_probabilities=transition,
            observer_probabilities=observer,
            cycle_transition_probabilities=tuple(transition_cycles),
            cycle_observer_probabilities=tuple(observer_cycles),
            cycle_halt_probabilities=halt_probabilities,
            cycle_mixture_weights=mixture,
            mode=mode,
        )


__all__ = [
    "COUNTERFACTUAL_FEATURE_WIDTH",
    "COUNTERFACTUAL_REPAIR_ADMITTED",
    "COUNTERFACTUAL_REPAIR_MODES",
    "CounterfactualMachineRepair",
    "CounterfactualRepairError",
    "CounterfactualRepairResult",
    "DEFAULT_CONTROLLER_WIDTH",
    "DEFAULT_MEMORY_WIDTH",
    "DEFAULT_PARAMETER_COUNT",
    "ROW_FEATURE_WIDTH",
]
