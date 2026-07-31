"""Deterministic joint language/ETTR training primitives.

The joint scheduler charges supervised positions rather than optimizer
updates because a language batch and an ETTR batch have different sizes.
The language step shares the exact ETTR optimizer bundle so the base model
can co-adapt to both streams without applying fabricated gradients to the
ETTR-only modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Callable, Literal, Sequence

import torch
import torch.nn as nn

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorError,
)
from ettr_optimization import ETTROptimizerBundle


JointStreamName = Literal["general", "ettr"]
TriStreamName = Literal["general", "instruction", "ettr"]


@dataclass(frozen=True, slots=True)
class ETTRJointScheduleConfig:
    """Exact target mix expressed as an integer charged-position ratio."""

    general_position_weight: int
    ettr_position_weight: int

    def validate(self) -> None:
        values = (
            self.general_position_weight,
            self.ettr_position_weight,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in values
        ):
            raise TheoryReactorError(
                "joint-stream position weights must be positive integers"
            )

    @property
    def ettr_fraction(self) -> Fraction:
        self.validate()
        return Fraction(
            self.ettr_position_weight,
            self.general_position_weight + self.ettr_position_weight,
        )


@dataclass(frozen=True, slots=True)
class ETTRJointScheduleReceipt:
    general_positions: int
    ettr_positions: int
    general_updates: int
    ettr_updates: int

    @property
    def total_positions(self) -> int:
        return self.general_positions + self.ettr_positions


class ETTRJointPositionScheduler:
    """Choose the next stream by closest cumulative target-position error."""

    def __init__(
        self,
        config: ETTRJointScheduleConfig,
        *,
        receipt: ETTRJointScheduleReceipt | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._receipt = (
            ETTRJointScheduleReceipt(0, 0, 0, 0)
            if receipt is None
            else receipt
        )
        self._validate_receipt(self._receipt)
        self._pending: tuple[JointStreamName, int] | None = None

    @property
    def receipt(self) -> ETTRJointScheduleReceipt:
        return self._receipt

    def select(
        self,
        *,
        general_positions: int,
        ettr_positions: int,
    ) -> JointStreamName:
        if self._pending is not None:
            raise TheoryReactorError(
                "joint-stream choice must be recorded before selecting again"
            )
        self._validate_charge(general_positions)
        self._validate_charge(ettr_positions)
        general_error = self._candidate_error(
            stream="general",
            positions=general_positions,
        )
        ettr_error = self._candidate_error(
            stream="ettr",
            positions=ettr_positions,
        )
        if general_error < ettr_error:
            stream: JointStreamName = "general"
            positions = general_positions
        elif ettr_error < general_error:
            stream = "ettr"
            positions = ettr_positions
        else:
            target = self.config.ettr_fraction
            current = (
                Fraction(
                    self._receipt.ettr_positions,
                    self._receipt.total_positions,
                )
                if self._receipt.total_positions
                else Fraction(0, 1)
            )
            stream = "ettr" if current < target else "general"
            positions = (
                ettr_positions if stream == "ettr" else general_positions
            )
        self._pending = (stream, positions)
        return stream

    def record(
        self,
        *,
        stream: JointStreamName,
        positions: int,
    ) -> ETTRJointScheduleReceipt:
        self._validate_charge(positions)
        if self._pending != (stream, positions):
            raise TheoryReactorError(
                "joint-stream charge differs from the selected update"
            )
        current = self._receipt
        if stream == "general":
            receipt = ETTRJointScheduleReceipt(
                general_positions=current.general_positions + positions,
                ettr_positions=current.ettr_positions,
                general_updates=current.general_updates + 1,
                ettr_updates=current.ettr_updates,
            )
        elif stream == "ettr":
            receipt = ETTRJointScheduleReceipt(
                general_positions=current.general_positions,
                ettr_positions=current.ettr_positions + positions,
                general_updates=current.general_updates,
                ettr_updates=current.ettr_updates + 1,
            )
        else:
            raise TheoryReactorError("joint-stream name differs")
        self._validate_receipt(receipt)
        self._receipt = receipt
        self._pending = None
        return receipt

    def state_dict(self) -> dict[str, object]:
        if self._pending is not None:
            raise TheoryReactorError(
                "joint-stream state cannot be saved during an update"
            )
        return {
            "schema": "shohin-ettr-joint-position-scheduler-v1",
            "config": asdict(self.config),
            "receipt": asdict(self._receipt),
        }

    def _candidate_error(
        self,
        *,
        stream: JointStreamName,
        positions: int,
    ) -> Fraction:
        general = self._receipt.general_positions
        ettr = self._receipt.ettr_positions
        if stream == "general":
            general += positions
        elif stream == "ettr":
            ettr += positions
        else:
            raise TheoryReactorError("joint-stream name differs")
        total = general + ettr
        target = self.config.ettr_fraction
        return abs(Fraction(ettr, total) - target)

    @staticmethod
    def _validate_charge(value: int) -> None:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise TheoryReactorError(
                "joint-stream charged positions must be a positive integer"
            )

    @classmethod
    def _validate_receipt(
        cls,
        receipt: ETTRJointScheduleReceipt,
    ) -> None:
        if not isinstance(receipt, ETTRJointScheduleReceipt):
            raise TheoryReactorError("joint-stream receipt type differs")
        values = (
            receipt.general_positions,
            receipt.ettr_positions,
            receipt.general_updates,
            receipt.ettr_updates,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in values
        ):
            raise TheoryReactorError("joint-stream receipt differs")
        if (
            (receipt.general_positions == 0)
            != (receipt.general_updates == 0)
            or (receipt.ettr_positions == 0)
            != (receipt.ettr_updates == 0)
        ):
            raise TheoryReactorError(
                "joint-stream positions and update counts disagree"
            )


@dataclass(frozen=True, slots=True)
class ETTRTriScheduleConfig:
    """Exact three-stream target mix in charged supervised positions."""

    general_position_weight: int
    instruction_position_weight: int
    ettr_position_weight: int

    def validate(self) -> None:
        values = (
            self.general_position_weight,
            self.instruction_position_weight,
            self.ettr_position_weight,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in values
        ):
            raise TheoryReactorError(
                "tri-stream position weights must be positive integers"
            )

    @property
    def fractions(self) -> tuple[Fraction, Fraction, Fraction]:
        self.validate()
        total = (
            self.general_position_weight
            + self.instruction_position_weight
            + self.ettr_position_weight
        )
        return (
            Fraction(self.general_position_weight, total),
            Fraction(self.instruction_position_weight, total),
            Fraction(self.ettr_position_weight, total),
        )


@dataclass(frozen=True, slots=True)
class ETTRTriScheduleReceipt:
    general_positions: int
    instruction_positions: int
    ettr_positions: int
    general_updates: int
    instruction_updates: int
    ettr_updates: int

    @property
    def total_positions(self) -> int:
        return (
            self.general_positions
            + self.instruction_positions
            + self.ettr_positions
        )


class ETTRTriPositionScheduler:
    """Choose among language, instruction, and ETTR by target-share error."""

    _STREAMS: tuple[TriStreamName, ...] = (
        "general",
        "instruction",
        "ettr",
    )

    def __init__(
        self,
        config: ETTRTriScheduleConfig,
        *,
        receipt: ETTRTriScheduleReceipt | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._receipt = (
            ETTRTriScheduleReceipt(0, 0, 0, 0, 0, 0)
            if receipt is None
            else receipt
        )
        self._validate_receipt(self._receipt)
        self._pending: tuple[TriStreamName, int] | None = None

    @property
    def receipt(self) -> ETTRTriScheduleReceipt:
        return self._receipt

    def select(
        self,
        *,
        general_positions: int,
        instruction_positions: int,
        ettr_positions: int,
    ) -> TriStreamName:
        if self._pending is not None:
            raise TheoryReactorError(
                "tri-stream choice must be recorded before selecting again"
            )
        charges = {
            "general": general_positions,
            "instruction": instruction_positions,
            "ettr": ettr_positions,
        }
        for value in charges.values():
            self._validate_charge(value)
        errors = {
            stream: self._candidate_error(stream, charges[stream])
            for stream in self._STREAMS
        }
        minimum = min(errors.values())
        tied = [
            stream for stream in self._STREAMS if errors[stream] == minimum
        ]
        if len(tied) == 1:
            selected = tied[0]
        else:
            deficits = self._current_deficits()
            selected = max(
                tied,
                key=lambda stream: (
                    deficits[stream],
                    -self._STREAMS.index(stream),
                ),
            )
        self._pending = (selected, charges[selected])
        return selected

    def record(
        self,
        *,
        stream: TriStreamName,
        positions: int,
    ) -> ETTRTriScheduleReceipt:
        self._validate_charge(positions)
        if self._pending != (stream, positions):
            raise TheoryReactorError(
                "tri-stream charge differs from the selected update"
            )
        values = asdict(self._receipt)
        values[f"{stream}_positions"] += positions
        values[f"{stream}_updates"] += 1
        receipt = ETTRTriScheduleReceipt(**values)
        self._validate_receipt(receipt)
        self._receipt = receipt
        self._pending = None
        return receipt

    def state_dict(self) -> dict[str, object]:
        if self._pending is not None:
            raise TheoryReactorError(
                "tri-stream state cannot be saved during an update"
            )
        return {
            "schema": "shohin-ettr-tri-position-scheduler-v1",
            "config": asdict(self.config),
            "receipt": asdict(self._receipt),
        }

    def _candidate_error(
        self,
        stream: TriStreamName,
        positions: int,
    ) -> Fraction:
        current = {
            name: getattr(self._receipt, f"{name}_positions")
            for name in self._STREAMS
        }
        current[stream] += positions
        total = sum(current.values())
        targets = dict(zip(self._STREAMS, self.config.fractions, strict=True))
        return sum(
            (
                abs(Fraction(current[name], total) - targets[name])
                for name in self._STREAMS
            ),
            start=Fraction(0, 1),
        )

    def _current_deficits(self) -> dict[TriStreamName, Fraction]:
        total = self._receipt.total_positions
        targets = dict(zip(self._STREAMS, self.config.fractions, strict=True))
        if total == 0:
            return targets
        return {
            name: targets[name]
            - Fraction(
                getattr(self._receipt, f"{name}_positions"),
                total,
            )
            for name in self._STREAMS
        }

    @staticmethod
    def _validate_charge(value: int) -> None:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise TheoryReactorError(
                "tri-stream charged positions must be a positive integer"
            )

    @classmethod
    def _validate_receipt(
        cls,
        receipt: ETTRTriScheduleReceipt,
    ) -> None:
        if not isinstance(receipt, ETTRTriScheduleReceipt):
            raise TheoryReactorError("tri-stream receipt type differs")
        for stream in cls._STREAMS:
            positions = getattr(receipt, f"{stream}_positions")
            updates = getattr(receipt, f"{stream}_updates")
            if (
                not isinstance(positions, int)
                or isinstance(positions, bool)
                or positions < 0
                or not isinstance(updates, int)
                or isinstance(updates, bool)
                or updates < 0
                or (positions == 0) != (updates == 0)
            ):
                raise TheoryReactorError("tri-stream receipt differs")


@dataclass(frozen=True, slots=True)
class GeneralLanguageStepConfig:
    gradient_accumulation_steps: int = 1
    gradient_clip: float = 1.0
    autocast_dtype: torch.dtype = torch.bfloat16

    def validate(self) -> None:
        if (
            not isinstance(self.gradient_accumulation_steps, int)
            or isinstance(self.gradient_accumulation_steps, bool)
            or self.gradient_accumulation_steps <= 0
            or not 0 < self.gradient_clip <= 100
            or self.autocast_dtype not in (torch.bfloat16, torch.float16)
        ):
            raise TheoryReactorError(
                "general-language train-step configuration differs"
            )


@dataclass(frozen=True, slots=True)
class GeneralLanguageUpdateReceipt:
    optimizer_step: int
    learning_rate_scale: float
    loss: torch.Tensor
    gradient_norm: torch.Tensor
    supervised_token_count: torch.Tensor


class GeneralLanguageUpdateStep(nn.Module):
    """One base-language update using the shared ETTR optimizer."""

    def __init__(
        self,
        model: EndogenousTypedTheoryReactorGPT,
        optimizer: ETTROptimizerBundle,
        *,
        step_config: GeneralLanguageStepConfig | None = None,
        gradient_synchronizer: (
            Callable[[Sequence[nn.Parameter]], None] | None
        ) = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.step_config = (
            GeneralLanguageStepConfig()
            if step_config is None
            else step_config
        )
        self.step_config.validate()
        if not optimizer.config.train_base:
            raise TheoryReactorError(
                "general-language updates require trainable base parameters"
            )
        if gradient_synchronizer is not None and not callable(
            gradient_synchronizer
        ):
            raise TheoryReactorError(
                "general-language gradient synchronizer is not callable"
            )
        self.gradient_synchronizer = gradient_synchronizer
        self._poisoned = False
        optimizer.assert_bound_to(model)
        optimizer.assert_healthy()

    def update(
        self,
        batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> GeneralLanguageUpdateReceipt:
        if self._poisoned:
            raise TheoryReactorError(
                "general-language train step is fail-stop; restart from "
                "the last verified checkpoint"
            )
        self.optimizer.assert_healthy()
        if len(batches) != self.step_config.gradient_accumulation_steps:
            raise TheoryReactorError(
                "general-language accumulation window differs"
            )
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        token_counts: list[torch.Tensor] = []
        try:
            for inputs, targets in batches:
                if (
                    not isinstance(inputs, torch.Tensor)
                    or not isinstance(targets, torch.Tensor)
                    or inputs.ndim != 2
                    or targets.shape != inputs.shape
                    or inputs.dtype != torch.long
                    or targets.dtype != torch.long
                    or inputs.device != targets.device
                    or not bool(targets.ne(-1).any())
                ):
                    raise TheoryReactorError(
                        "general-language batch differs"
                    )
                device_type = inputs.device.type
                with torch.autocast(
                    device_type=device_type,
                    dtype=self.step_config.autocast_dtype,
                    enabled=device_type in {"cuda", "cpu"},
                ):
                    _, loss = self.model.base(inputs, targets)
                    if loss is None:
                        raise TheoryReactorError(
                            "general-language loss is missing"
                        )
                    scaled = (
                        loss
                        / self.step_config.gradient_accumulation_steps
                    )
                if not bool(torch.isfinite(loss.detach())):
                    raise TheoryReactorError(
                        "general-language loss is nonfinite"
                    )
                scaled.backward()
                losses.append(loss.detach())
                token_counts.append(targets.ne(-1).sum().detach())
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        try:
            trainable = tuple(
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
            if self.gradient_synchronizer is not None:
                self.gradient_synchronizer(trainable)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                self.step_config.gradient_clip,
            )
            if not bool(torch.isfinite(gradient_norm.detach())):
                raise TheoryReactorError(
                    "general-language gradient norm is nonfinite"
                )
            scale = self.optimizer.apply_schedule()
            self.optimizer.step()
        except BaseException as error:
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.mark_failed_update()
            self._poisoned = True
            raise TheoryReactorError(
                "general-language optimizer update failed after backward; "
                "restart from the last verified checkpoint"
            ) from error
        return GeneralLanguageUpdateReceipt(
            optimizer_step=self.optimizer.next_update,
            learning_rate_scale=scale,
            loss=torch.stack(losses).mean(),
            gradient_norm=gradient_norm.detach(),
            supervised_token_count=torch.stack(token_counts).sum(),
        )


__all__ = [
    "ETTRJointPositionScheduler",
    "ETTRJointScheduleConfig",
    "ETTRJointScheduleReceipt",
    "ETTRTriPositionScheduler",
    "ETTRTriScheduleConfig",
    "ETTRTriScheduleReceipt",
    "GeneralLanguageStepConfig",
    "GeneralLanguageUpdateReceipt",
    "GeneralLanguageUpdateStep",
    "JointStreamName",
    "TriStreamName",
]
