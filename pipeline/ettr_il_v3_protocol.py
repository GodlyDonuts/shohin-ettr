"""Frozen constants and quota mechanics for the ETTR-IL-v3 initializer.

This module is intentionally pure. It does not enumerate data, open files,
load models, submit jobs, or access a network.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from typing import Mapping, Sequence


PROTOCOL = "R12-ETTR-IL-v3-initializer"
SCHEMA = "r12-ettr-il-v3-protocol-v1"
MASTER_SEED = hashlib.sha256(
    b"R12-ETTR-IL-v3-initializer|frozen-before-generation|2026-07-27"
).digest()
MASTER_SEED_SHA256 = hashlib.sha256(MASTER_SEED).hexdigest()

WORLD_WIDTH = 192
COMMAND_WIDTH = 96
QUERY_WIDTH = 48
CHARGED_POSITIONS_PER_ROW = 528
VIEWS_PER_CORE = 4
ROWS_PER_VIEW = 16
ROWS_PER_CORE = VIEWS_PER_CORE * ROWS_PER_VIEW
POSITIONS_PER_CORE = ROWS_PER_CORE * CHARGED_POSITIONS_PER_ROW

FAMILIES = ("horn", "local_rewrite", "resource")
SPLITS = (
    "train",
    "development",
    "confirmation",
    "train_reserve",
    "development_reserve",
    "confirmation_reserve",
)
CURRICULUM_STAGES = (
    "compiler_grounding",
    "atomic_transactions",
    "dependent_composition",
    "query_counterfactual_grounding",
    "closed_loop_invariance",
)
TRAIN_STAGE_CORES = {
    "compiler_grounding": 8_000,
    "atomic_transactions": 8_000,
    "dependent_composition": 12_000,
    "query_counterfactual_grounding": 6_000,
    "closed_loop_invariance": 6_000,
}
SPLIT_CORES = {
    "train": 40_000,
    "development": 5_000,
    "confirmation": 5_000,
    "train_reserve": 10_000,
    "development_reserve": 1_250,
    "confirmation_reserve": 1_250,
}
TRAINABLE_ETTR_PARAMETERS = 67_697_771
COMPLETE_SYSTEM_PARAMETERS = 192_779_435


class ProtocolError(ValueError):
    """The requested corpus shape differs from the frozen protocol."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def surplus(quota: int) -> int:
    if type(quota) is not int or quota < 0:
        raise ProtocolError("quota must be a nonnegative exact integer")
    return max(16, math.ceil(quota / 4))


def candidate_floor(quota: int) -> int:
    """Return the minimum semantically valid candidates before selection."""

    return 3 * (quota + surplus(quota))


def cyclic_balanced_allocation(
    total: int,
    labels: Sequence[str],
    *,
    context: Mapping[str, object],
) -> dict[str, int]:
    """Allocate ``total`` by quotient and one hash-rotated remainder."""

    if type(total) is not int or total < 0:
        raise ProtocolError("allocation total differs")
    ordered = tuple(labels)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ProtocolError("allocation labels differ")
    if any(type(label) is not str or not label for label in ordered):
        raise ProtocolError("allocation label differs")
    quotient, remainder = divmod(total, len(ordered))
    digest = hashlib.sha256(
        MASTER_SEED
        + b"|allocation|"
        + canonical_json_bytes(
            {
                "context": dict(context),
                "labels": list(ordered),
                "total": total,
            }
        )
    ).digest()
    offset = int.from_bytes(digest[:8], "big") % len(ordered)
    result = {label: quotient for label in ordered}
    for index in range(remainder):
        result[ordered[(offset + index) % len(ordered)]] += 1
    if sum(result.values()) != total:
        raise AssertionError("balanced allocation lost cardinality")
    if max(result.values()) - min(result.values()) > 1:
        raise AssertionError("balanced allocation exceeds one-core spread")
    return result


def split_family_allocation(split: str) -> dict[str, int]:
    if split not in SPLIT_CORES:
        raise ProtocolError("split differs")
    return cyclic_balanced_allocation(
        SPLIT_CORES[split],
        FAMILIES,
        context={"axis": "family", "split": split},
    )


def split_stage_allocation(split: str) -> dict[str, int]:
    """Allocate one split over the frozen train-stage proportions."""

    if split not in SPLIT_CORES:
        raise ProtocolError("split differs")
    if split == "train":
        return dict(TRAIN_STAGE_CORES)
    total = SPLIT_CORES[split]
    train_total = SPLIT_CORES["train"]
    result = {
        stage: total * TRAIN_STAGE_CORES[stage] // train_total
        for stage in CURRICULUM_STAGES
    }
    remainder = total - sum(result.values())
    ranked = sorted(
        CURRICULUM_STAGES,
        key=lambda stage: (
            hashlib.sha256(
                MASTER_SEED
                + b"|stage-remainder|"
                + canonical_json_bytes(
                    {
                        "split": split,
                        "stage": stage,
                        "total": total,
                    }
                )
            ).digest(),
            stage,
        ),
    )
    for stage in ranked[:remainder]:
        result[stage] += 1
    if sum(result.values()) != total:
        raise AssertionError("stage allocation lost cardinality")
    return result


def split_stage_family_allocation(split: str) -> dict[str, dict[str, int]]:
    """Return one exact matrix satisfying both stage and family marginals."""

    stage_totals = split_stage_allocation(split)
    family_totals = split_family_allocation(split)
    matrix = {
        stage: {
            family: stage_totals[stage] // len(FAMILIES)
            for family in FAMILIES
        }
        for stage in CURRICULUM_STAGES
    }
    row_needs = {
        stage: stage_totals[stage] % len(FAMILIES)
        for stage in CURRICULUM_STAGES
    }
    column_needs = {
        family: family_totals[family]
        - sum(matrix[stage][family] for stage in CURRICULUM_STAGES)
        for family in FAMILIES
    }
    if any(value < 0 for value in column_needs.values()):
        raise AssertionError("family marginal falls below matrix floor")

    def assignment_rank(stage: str, families: tuple[str, ...]) -> bytes:
        return hashlib.sha256(
            MASTER_SEED
            + b"|stage-family-matrix|"
            + canonical_json_bytes(
                {
                    "families": list(families),
                    "split": split,
                    "stage": stage,
                }
            )
        ).digest()

    def solve(row_index: int) -> bool:
        if row_index == len(CURRICULUM_STAGES):
            return not any(column_needs.values())
        stage = CURRICULUM_STAGES[row_index]
        need = row_needs[stage]
        choices = tuple(
            combination
            for combination in combinations(
                FAMILIES,
                need,
            )
            if all(column_needs[family] > 0 for family in combination)
        )
        for combination in sorted(
            choices,
            key=lambda value: (assignment_rank(stage, value), value),
        ):
            for family in combination:
                matrix[stage][family] += 1
                column_needs[family] -= 1
            if solve(row_index + 1):
                return True
            for family in combination:
                matrix[stage][family] -= 1
                column_needs[family] += 1
        return False

    if not solve(0):
        raise AssertionError("stage-family marginals are infeasible")
    for stage in CURRICULUM_STAGES:
        if sum(matrix[stage].values()) != stage_totals[stage]:
            raise AssertionError("stage-family row marginal differs")
        if max(matrix[stage].values()) - min(matrix[stage].values()) > 1:
            raise AssertionError("stage-family row is not balanced")
    for family in FAMILIES:
        if (
            sum(matrix[stage][family] for stage in CURRICULUM_STAGES)
            != family_totals[family]
        ):
            raise AssertionError("stage-family column marginal differs")
    return matrix


def train_stage_family_allocation(
    stage: str,
) -> dict[str, int]:
    if stage not in TRAIN_STAGE_CORES:
        raise ProtocolError("curriculum stage differs")
    return split_stage_family_allocation("train")[stage]


def orbit_owner(canonical_orbit: object) -> str:
    """Own one semantic orbit before rendering or target inspection."""

    digest = hashlib.sha256(
        MASTER_SEED + b"|orbit-owner|" + canonical_json_bytes(canonical_orbit)
    ).digest()
    return ("train", "development", "confirmation")[
        int.from_bytes(digest[:8], "big") % 3
    ]


@dataclass(frozen=True, slots=True)
class CorpusBudget:
    split: str
    cores: int
    views: int
    rows: int
    charged_positions: int

    def validate(self) -> None:
        if self.split not in SPLIT_CORES:
            raise ProtocolError("budget split differs")
        expected_cores = SPLIT_CORES[self.split]
        if (
            self.cores != expected_cores
            or self.views != expected_cores * VIEWS_PER_CORE
            or self.rows != expected_cores * ROWS_PER_CORE
            or self.charged_positions
            != expected_cores * POSITIONS_PER_CORE
        ):
            raise ProtocolError("budget arithmetic differs")


def corpus_budget(split: str) -> CorpusBudget:
    if split not in SPLIT_CORES:
        raise ProtocolError("split differs")
    cores = SPLIT_CORES[split]
    result = CorpusBudget(
        split=split,
        cores=cores,
        views=cores * VIEWS_PER_CORE,
        rows=cores * ROWS_PER_CORE,
        charged_positions=cores * POSITIONS_PER_CORE,
    )
    result.validate()
    return result


def protocol_receipt() -> dict[str, object]:
    budgets = {split: corpus_budget(split) for split in SPLITS}
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "master_seed_sha256": MASTER_SEED_SHA256,
        "geometry": {
            "world_width": WORLD_WIDTH,
            "command_width": COMMAND_WIDTH,
            "query_width": QUERY_WIDTH,
            "charged_positions_per_row": CHARGED_POSITIONS_PER_ROW,
            "views_per_core": VIEWS_PER_CORE,
            "rows_per_view": ROWS_PER_VIEW,
            "rows_per_core": ROWS_PER_CORE,
            "positions_per_core": POSITIONS_PER_CORE,
        },
        "parameters": {
            "trainable_ettr": TRAINABLE_ETTR_PARAMETERS,
            "complete_system": COMPLETE_SYSTEM_PARAMETERS,
        },
        "train_stage_cores": dict(TRAIN_STAGE_CORES),
        "split_cores": dict(SPLIT_CORES),
        "split_budgets": {
            split: {
                "cores": budget.cores,
                "views": budget.views,
                "rows": budget.rows,
                "charged_positions": budget.charged_positions,
            }
            for split, budget in budgets.items()
        },
        "split_family_allocations": {
            split: split_family_allocation(split)
            for split in SPLITS
        },
        "split_stage_allocations": {
            split: split_stage_allocation(split)
            for split in SPLITS
        },
        "split_stage_family_allocations": {
            split: split_stage_family_allocation(split)
            for split in SPLITS
        },
        "train_stage_family_allocations": {
            stage: train_stage_family_allocation(stage)
            for stage in CURRICULUM_STAGES
        },
        "candidate_rule": "3 * (quota + max(16, ceil(quota / 4)))",
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


__all__ = [
    "CHARGED_POSITIONS_PER_ROW",
    "COMMAND_WIDTH",
    "COMPLETE_SYSTEM_PARAMETERS",
    "CURRICULUM_STAGES",
    "CorpusBudget",
    "FAMILIES",
    "MASTER_SEED",
    "MASTER_SEED_SHA256",
    "POSITIONS_PER_CORE",
    "PROTOCOL",
    "ProtocolError",
    "QUERY_WIDTH",
    "ROWS_PER_CORE",
    "ROWS_PER_VIEW",
    "SCHEMA",
    "SPLITS",
    "SPLIT_CORES",
    "TRAINABLE_ETTR_PARAMETERS",
    "TRAIN_STAGE_CORES",
    "VIEWS_PER_CORE",
    "WORLD_WIDTH",
    "candidate_floor",
    "canonical_json_bytes",
    "corpus_budget",
    "cyclic_balanced_allocation",
    "orbit_owner",
    "protocol_receipt",
    "split_family_allocation",
    "split_stage_allocation",
    "split_stage_family_allocation",
    "surplus",
    "train_stage_family_allocation",
]
