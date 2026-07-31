#!/usr/bin/env python3
"""Inspect ETTR causal-query behavior from easiest to deepest episodes.

This is a read-only diagnostic. It reconstructs the raw and trained models
under the same immutable release/checkpoint contract as ``eval_ettr_v3.py``,
then retains bounded human-readable WORLD/COMMAND/QUERY examples alongside
the exact difference-in-differences margin used by the learning gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from tokenizers import Tokenizer

from ettr_checkpoint import load_ettr_checkpoint
from ettr_data_contract import ETTRContinuationBatch
from ettr_episode import CausalETTREpisodeRunner, ETTREpisodeSegment
from endogenous_typed_theory_reactor import ReactorTrace, TypedTheoryState
from ettr_objectives import (
    ETTRCausalQueryPair,
    ETTRObjectiveConfig,
    ETTRPacketTargets,
    ETTRTransactionPredictions,
    ETTRTransactionTargets,
    _operand_masks,
)
from ettr_optimization import ETTROptimizerBundle
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_v3 import (
    ETTRV3EvaluationError,
    _HEX40,
    _HEX64,
    _build_model,
    _parameter_sha256,
    _read_hash_bound_json,
    _sha256_file,
    _validate_checkpoint_cursor,
    _validate_run_contract,
    _write_no_replace,
)


REPORT_SCHEMA = "shohin-ettr-il-v3-causal-query-probe-v2"
_MARGIN_THRESHOLDS = (0.0, 0.1, 0.25, 0.5, 1.0)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--run-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--retain-per-depth", type=int, default=1)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if (
        _HEX64.fullmatch(args.release_sha256) is None
        or _HEX64.fullmatch(args.checkpoint_sha256) is None
        or _HEX64.fullmatch(args.run_contract_sha256) is None
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.architecture_seed < 2**63
        or not 0 <= args.data_seed < 2**63
        or args.max_batches < 2
        or not 1 <= args.retain_per_depth <= 8
    ):
        raise ETTRV3EvaluationError("ETTR causal-query probe arguments differ")


def _objective_pairs(
    model: torch.nn.Module,
    batch: ETTRContinuationBatch,
) -> tuple[
    Mapping[str, ETTRCausalQueryPair],
    Mapping[str, list[dict[str, object]]],
]:
    pairs, states, _ = _objective_pairs_and_traces(model, batch)
    return pairs, states


def _objective_pairs_and_traces(
    model: torch.nn.Module,
    batch: ETTRContinuationBatch,
) -> tuple[
    Mapping[str, ETTRCausalQueryPair],
    Mapping[str, list[dict[str, object]]],
    Mapping[str, list[dict[str, object]]],
]:
    pairs, states, traces, _, _ = _objective_geometry(model, batch)
    return pairs, states, traces


def _objective_geometry(
    model: torch.nn.Module,
    batch: ETTRContinuationBatch,
) -> tuple[
    Mapping[str, ETTRCausalQueryPair],
    Mapping[str, list[dict[str, object]]],
    Mapping[str, list[dict[str, object]]],
    Mapping[str, dict[str, object]],
    Mapping[str, dict[str, object]],
]:
    runner = CausalETTREpisodeRunner(model)
    steps = batch.transaction_targets.opcode.shape[1]
    output = runner(
        batch.episodes,
        reactor_steps=steps,
        hard=True,
        validate_batch=False,
        compute_losses=False,
    )
    (
        world_packet,
        world_command,
        world_target,
        command_packet,
        command_command,
        command_target,
    ) = batch.causal_rectangles.intervention_indices()
    interventions = runner.intervene(
        batch.episodes,
        output.initial_state,
        reactor_steps=steps,
        world_packet_index=world_packet,
        world_command_index=world_command,
        world_query_index=world_target,
        command_packet_index=command_packet,
        command_command_index=command_command,
        command_query_index=command_target,
        hard=True,
    )
    objective = batch.objective_batch(output, interventions)
    return (
        {
            "command": objective.command_query_binding,
            "world": objective.world_query_binding,
        },
        {
            "command": _state_pair_rows(
                interventions.command_terminal_state,
                output.terminal_state,
                command_packet,
            ),
            "world": _state_pair_rows(
                interventions.world_terminal_state,
                output.terminal_state,
                world_command,
            ),
        },
        {
            "command": _trace_pair_rows(
                interventions.command_trace,
                output.trace,
                command_packet,
            ),
            "world": _trace_pair_rows(
                interventions.world_trace,
                output.trace,
                world_command,
            ),
        },
        {
            "factual": _transaction_geometry_row(
                objective.transactions,
                objective.transaction_targets,
            ),
            "world_intervention": _transaction_geometry_row(
                objective.world_intervention_transactions,
                objective.world_intervention_transaction_targets,
            ),
            "command_intervention": _transaction_geometry_row(
                objective.command_intervention_transactions,
                objective.command_intervention_transaction_targets,
            ),
        },
        {
            "initial": _packet_geometry_row(
                objective.packet_prediction,
                objective.packet_targets,
            ),
            "factual_terminal": _packet_geometry_row(
                objective.terminal_packet_prediction,
                objective.terminal_packet_targets,
            ),
            "world_intervention_terminal": _packet_geometry_row(
                objective.world_intervention_prediction,
                objective.world_intervention_targets,
            ),
            "command_intervention_terminal": _packet_geometry_row(
                objective.command_intervention_prediction,
                objective.command_intervention_targets,
            ),
        },
    )


def _packet_geometry_row(
    prediction: TypedTheoryState,
    targets: ETTRPacketTargets,
) -> dict[str, object]:
    categorical_mask = targets.slot_mask & targets.active
    field_masks = {
        "active": targets.slot_mask,
        "root": targets.slot_mask,
        "value_code": categorical_mask,
        "type_index": categorical_mask,
        "relations": targets.relation_mask,
        "committed": torch.ones_like(targets.committed),
        "halted": torch.ones_like(targets.halted),
    }
    probabilities = {
        "active": prediction.active.detach().float(),
        "root": prediction.root.detach().float(),
        "value_code": prediction.value_probabilities.detach().float(),
        "type_index": prediction.type_probabilities.detach().float(),
        "relations": prediction.relations.detach().float(),
        "committed": prediction.committed.detach().float(),
        "halted": prediction.halted.detach().float(),
    }
    exact = torch.ones(
        targets.active.shape[0],
        dtype=torch.bool,
        device=targets.active.device,
    )
    fields: dict[str, dict[str, float | int | None]] = {}
    for name in ("value_code", "type_index"):
        mask = field_masks[name]
        labels = getattr(targets, name)
        values = probabilities[name]
        correct = values.argmax(-1).eq(labels)
        target_probability = values.gather(
            -1,
            labels.unsqueeze(-1),
        ).squeeze(-1)
        exact &= (correct | ~mask).reshape(correct.shape[0], -1).all(-1)
        fields[name] = {
            "correct": int((correct & mask).sum().cpu()),
            "negative_correct": None,
            "negative_support": None,
            "positive_correct": None,
            "positive_support": None,
            "support": int(mask.sum().cpu()),
            "target_probability_sum": float(
                target_probability.masked_select(mask).sum().cpu()
            ),
        }
    for name in (
        "active",
        "root",
        "relations",
        "committed",
        "halted",
    ):
        mask = field_masks[name]
        labels = getattr(targets, name)
        values = probabilities[name]
        choices = values.ge(0.5)
        correct = choices.eq(labels)
        positive = mask & labels
        negative = mask & ~labels
        target_probability = torch.where(labels, values, 1.0 - values)
        exact &= (correct | ~mask).reshape(correct.shape[0], -1).all(-1)
        fields[name] = {
            "correct": int((correct & mask).sum().cpu()),
            "negative_correct": int((correct & negative).sum().cpu()),
            "negative_support": int(negative.sum().cpu()),
            "positive_correct": int((correct & positive).sum().cpu()),
            "positive_support": int(positive.sum().cpu()),
            "support": int(mask.sum().cpu()),
            "target_probability_sum": float(
                target_probability.masked_select(mask).sum().cpu()
            ),
        }
    return {
        "complete_correct": int(exact.sum().cpu()),
        "complete_support": exact.numel(),
        "fields": fields,
    }


def _packet_geometry_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ETTRV3EvaluationError(
            "packet geometry population differs"
        )
    field_summary = {}
    for name in rows[0]["fields"]:
        support = sum(int(row["fields"][name]["support"]) for row in rows)
        correct = sum(int(row["fields"][name]["correct"]) for row in rows)
        probability_sum = sum(
            float(row["fields"][name]["target_probability_sum"])
            for row in rows
        )
        positive_values = [
            row["fields"][name]["positive_support"]
            for row in rows
        ]
        negative_values = [
            row["fields"][name]["negative_support"]
            for row in rows
        ]
        positive_support = (
            None
            if positive_values[0] is None
            else sum(int(value) for value in positive_values)
        )
        negative_support = (
            None
            if negative_values[0] is None
            else sum(int(value) for value in negative_values)
        )
        positive_correct = (
            None
            if positive_support is None
            else sum(
                int(row["fields"][name]["positive_correct"])
                for row in rows
            )
        )
        negative_correct = (
            None
            if negative_support is None
            else sum(
                int(row["fields"][name]["negative_correct"])
                for row in rows
            )
        )
        field_summary[name] = {
            "mean_target_probability": (
                probability_sum / support if support else None
            ),
            "negative_accuracy": (
                None
                if negative_support in {None, 0}
                else negative_correct / negative_support
            ),
            "negative_support": negative_support,
            "positive_accuracy": (
                None
                if positive_support in {None, 0}
                else positive_correct / positive_support
            ),
            "positive_support": positive_support,
            "support": support,
            "top1_accuracy": correct / support if support else None,
        }
    complete_correct = sum(int(row["complete_correct"]) for row in rows)
    complete_support = sum(int(row["complete_support"]) for row in rows)
    return {
        "complete_packet_accuracy": complete_correct / complete_support,
        "complete_packet_support": complete_support,
        "fields": field_summary,
    }


def _transaction_geometry_row(
    prediction: ETTRTransactionPredictions,
    targets: ETTRTransactionTargets,
) -> dict[str, object]:
    names = (
        "opcode",
        "source",
        "target",
        "relation",
        "type_index",
        "value_code",
    )
    masks = _operand_masks(targets)
    exact = targets.step_mask.clone()
    fields: dict[str, dict[str, float | int]] = {}
    for name, mask in zip(names, masks, strict=True):
        probabilities = getattr(prediction, name).detach().float()
        labels = getattr(targets, name)
        choices = probabilities.argmax(-1)
        correct = choices.eq(labels)
        exact &= ~mask | correct
        fields[name] = {
            "correct": int((correct & mask).sum().cpu()),
            "support": int(mask.sum().cpu()),
            "target_probability_sum": float(
                probabilities.gather(-1, labels.unsqueeze(-1))
                .squeeze(-1)
                .masked_select(mask)
                .sum()
                .cpu()
            ),
        }
    for name in ("committed", "halted"):
        mask = targets.step_mask
        probabilities = getattr(prediction, name).detach().float()
        labels = getattr(targets, name)
        choices = probabilities.ge(0.5)
        correct = choices.eq(labels)
        exact &= correct
        fields[name] = {
            "correct": int((correct & mask).sum().cpu()),
            "support": int(mask.sum().cpu()),
            "target_probability_sum": float(
                torch.where(labels, probabilities, 1.0 - probabilities)
                .masked_select(mask)
                .sum()
                .cpu()
            ),
        }
    opcode_choices = prediction.opcode.detach().float().argmax(-1)
    valid = targets.step_mask
    nonterminal_target = valid & targets.opcode.lt(6)
    return {
        "complete_correct": int((exact & valid).sum().cpu()),
        "complete_support": int(valid.sum().cpu()),
        "fields": fields,
        "predicted_opcode_counts": [
            int(((opcode_choices == index) & valid).sum().cpu())
            for index in range(prediction.opcode.shape[-1])
        ],
        "premature_terminal": int(
            (opcode_choices.ge(6) & nonterminal_target).sum().cpu()
        ),
        "premature_terminal_support": int(nonterminal_target.sum().cpu()),
        "target_opcode_counts": [
            int(((targets.opcode == index) & valid).sum().cpu())
            for index in range(prediction.opcode.shape[-1])
        ],
    }


def _transaction_geometry_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ETTRV3EvaluationError(
            "transaction geometry population differs"
        )
    field_names = tuple(rows[0]["fields"])
    field_summary = {}
    for name in field_names:
        correct = sum(int(row["fields"][name]["correct"]) for row in rows)
        support = sum(int(row["fields"][name]["support"]) for row in rows)
        probability_sum = sum(
            float(row["fields"][name]["target_probability_sum"])
            for row in rows
        )
        field_summary[name] = {
            "mean_target_probability": (
                probability_sum / support if support else None
            ),
            "support": support,
            "top1_accuracy": correct / support if support else None,
        }
    complete_correct = sum(int(row["complete_correct"]) for row in rows)
    complete_support = sum(int(row["complete_support"]) for row in rows)
    premature = sum(int(row["premature_terminal"]) for row in rows)
    premature_support = sum(
        int(row["premature_terminal_support"]) for row in rows
    )
    opcode_count = len(rows[0]["predicted_opcode_counts"])
    return {
        "complete_transaction_accuracy": complete_correct / complete_support,
        "complete_transaction_support": complete_support,
        "fields": field_summary,
        "predicted_opcode_counts": [
            sum(int(row["predicted_opcode_counts"][index]) for row in rows)
            for index in range(opcode_count)
        ],
        "premature_terminal_rate": (
            premature / premature_support
            if premature_support
            else None
        ),
        "premature_terminal_support": premature_support,
        "target_opcode_counts": [
            sum(int(row["target_opcode_counts"][index]) for row in rows)
            for index in range(opcode_count)
        ],
    }


def _state_pair_rows(
    correct: TypedTheoryState,
    foil_population: TypedTheoryState,
    foil_index: torch.Tensor,
) -> list[dict[str, object]]:
    fields = (
        "value_probabilities",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
        "halted",
    )
    correct_values = {
        name: getattr(correct, name).detach().float().cpu()
        for name in fields
    }
    foil_values = {
        name: (
            getattr(foil_population, name)
            .index_select(0, foil_index)
            .detach()
            .float()
            .cpu()
        )
        for name in fields
    }
    rows = []
    for index in range(foil_index.shape[0]):
        field_l1 = {
            name: float(
                (
                    correct_values[name][index]
                    - foil_values[name][index]
                )
                .abs()
                .sum()
            )
            for name in fields
        }
        correct_committed = float(correct_values["committed"][index])
        correct_halted = float(correct_values["halted"][index])
        foil_committed = float(foil_values["committed"][index])
        foil_halted = float(foil_values["halted"][index])
        rows.append(
            {
                "correct_answer_disposition": (
                    correct_committed >= 0.5 and correct_halted < 0.5
                ),
                "correct_committed": correct_committed,
                "correct_halted": correct_halted,
                "exact_state_equal": all(
                    value == 0.0 for value in field_l1.values()
                ),
                "field_l1": field_l1,
                "foil_answer_disposition": (
                    foil_committed >= 0.5 and foil_halted < 0.5
                ),
                "foil_committed": foil_committed,
                "foil_halted": foil_halted,
                "structural_state_equal": all(
                    field_l1[name] == 0.0
                    for name in fields
                    if name not in {"committed", "halted"}
                ),
            }
        )
    return rows


def _trace_pair_rows(
    correct: ReactorTrace,
    foil_population: ReactorTrace,
    foil_index: torch.Tensor,
) -> list[dict[str, object]]:
    probability_fields = (
        "opcode",
        "source",
        "target",
        "relation",
        "type_index",
        "value_code",
    )
    applied_fields = (
        "applied_opcode",
        "applied_source",
        "applied_target",
        "applied_relation",
        "applied_type_index",
        "applied_value_code",
        "active",
        "committed",
        "halted",
    )
    fields = probability_fields + applied_fields
    correct_values = {
        name: getattr(correct, name).detach().float().cpu()
        for name in fields
    }
    foil_values = {
        name: (
            getattr(foil_population, name)
            .index_select(0, foil_index)
            .detach()
            .float()
            .cpu()
        )
        for name in fields
    }
    rows = []
    for index in range(foil_index.shape[0]):
        field_l1 = {
            name: float(
                (
                    correct_values[name][index]
                    - foil_values[name][index]
                )
                .abs()
                .sum()
            )
            for name in fields
        }
        rows.append(
            {
                "applied_trace_equal": all(
                    field_l1[name] == 0.0 for name in applied_fields
                ),
                "field_l1": field_l1,
                "policy_trace_equal": all(
                    field_l1[name] == 0.0 for name in probability_fields
                ),
            }
        )
    return rows


def _pair_rows(pair: ETTRCausalQueryPair) -> list[dict[str, object]]:
    correct_logits = pair.correct_logits.detach().float().cpu()
    foil_logits = pair.foil_logits.detach().float().cpu()
    correct_target = pair.correct_target.detach().cpu()
    foil_target = pair.foil_target.detach().cpu()
    row = torch.arange(correct_target.shape[0])
    correct_for_correct = correct_logits[row, correct_target]
    correct_for_foil = correct_logits[row, foil_target]
    foil_for_correct = foil_logits[row, correct_target]
    foil_for_foil = foil_logits[row, foil_target]
    correct_delta = correct_for_correct - correct_for_foil
    foil_delta = foil_for_foil - foil_for_correct
    did = correct_delta + foil_delta
    correct_prediction = correct_logits.argmax(dim=-1)
    foil_prediction = foil_logits.argmax(dim=-1)
    result = []
    for index in range(correct_target.shape[0]):
        result.append(
            {
                "contrast": bool(
                    correct_target[index] != foil_target[index]
                ),
                "correct_delta": float(correct_delta[index]),
                "correct_prediction": int(correct_prediction[index]),
                "correct_target": int(correct_target[index]),
                "correct_top1": bool(
                    correct_prediction[index] == correct_target[index]
                ),
                "difference_in_differences": float(did[index]),
                "foil_delta": float(foil_delta[index]),
                "foil_prediction": int(foil_prediction[index]),
                "foil_target": int(foil_target[index]),
                "foil_top1": bool(
                    foil_prediction[index] == foil_target[index]
                ),
                "paired_order_correct": bool(correct_delta[index] > 0),
                "paired_order_foil": bool(foil_delta[index] > 0),
            }
        )
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ETTRV3EvaluationError("causal-query quantile population differs")
    ordered = sorted(values)
    location = fraction * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rate(values: Sequence[bool]) -> float:
    if not values:
        raise ETTRV3EvaluationError("causal-query rate population differs")
    return sum(values) / len(values)


def _depth_bucket(depth: int) -> str:
    if depth <= 0:
        raise ETTRV3EvaluationError("causal-query transaction depth differs")
    for upper in (1, 2, 4, 8, 16, 32):
        if depth <= upper:
            lower = 1 if upper == 1 else upper // 2 + 1
            return str(upper) if lower == upper else f"{lower}-{upper}"
    return "33-64"


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    contrast = [row for row in rows if row["contrast"] is True]
    if not contrast:
        raise ETTRV3EvaluationError("causal-query probe has no contrasts")

    def summarize(population: Sequence[Mapping[str, object]]) -> dict[str, object]:
        values = [
            float(row["difference_in_differences"])
            for row in population
        ]
        return {
            "correct_top1_rate": _rate(
                [bool(row["correct_top1"]) for row in population]
            ),
            "count": len(population),
            "difference_in_differences": {
                "maximum": max(values),
                "mean": sum(values) / len(values),
                "minimum": min(values),
                "p05": _quantile(values, 0.05),
                "p25": _quantile(values, 0.25),
                "p50": _quantile(values, 0.50),
                "p75": _quantile(values, 0.75),
                "p95": _quantile(values, 0.95),
            },
            "foil_top1_rate": _rate(
                [bool(row["foil_top1"]) for row in population]
            ),
            "joint_top1_rate": _rate(
                [
                    bool(row["correct_top1"]) and bool(row["foil_top1"])
                    for row in population
                ]
            ),
            "margin_rates": {
                f"{threshold:g}": _rate(
                    [
                        float(row["difference_in_differences"])
                        >= threshold
                        for row in population
                    ]
                )
                for threshold in _MARGIN_THRESHOLDS
            },
            "paired_order_joint_rate": _rate(
                [
                    bool(row["paired_order_correct"])
                    and bool(row["paired_order_foil"])
                    for row in population
                ]
            ),
        }

    by_depth: dict[str, list[Mapping[str, object]]] = {}
    for row in contrast:
        by_depth.setdefault(str(row["depth_bucket"]), []).append(row)
    return {
        **summarize(contrast),
        "all_pair_count": len(rows),
        "by_depth": {
            name: summarize(population)
            for name, population in sorted(by_depth.items())
        },
        "invariance_count": len(rows) - len(contrast),
        "strict_margin": 1.0,
    }


def _state_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ETTRV3EvaluationError("causal-query state population differs")
    fields = tuple(rows[0]["field_l1"])
    return {
        "correct_answer_disposition_rate": _rate(
            [bool(row["correct_answer_disposition"]) for row in rows]
        ),
        "count": len(rows),
        "exact_state_equal_rate": _rate(
            [bool(row["exact_state_equal"]) for row in rows]
        ),
        "field_l1_means": {
            name: sum(float(row["field_l1"][name]) for row in rows)
            / len(rows)
            for name in fields
        },
        "foil_answer_disposition_rate": _rate(
            [bool(row["foil_answer_disposition"]) for row in rows]
        ),
        "structural_state_equal_rate": _rate(
            [bool(row["structural_state_equal"]) for row in rows]
        ),
    }


def _trace_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ETTRV3EvaluationError("causal-query trace population differs")
    fields = tuple(rows[0]["field_l1"])
    return {
        "applied_trace_equal_rate": _rate(
            [bool(row["applied_trace_equal"]) for row in rows]
        ),
        "count": len(rows),
        "field_l1_means": {
            name: sum(float(row["field_l1"][name]) for row in rows)
            / len(rows)
            for name in fields
        },
        "policy_trace_equal_rate": _rate(
            [bool(row["policy_trace_equal"]) for row in rows]
        ),
    }


def _decode_segment(
    tokenizer: Tokenizer,
    segment: ETTREpisodeSegment,
    row: int,
) -> str:
    mask = segment.attention_mask[row].bool()
    tokens = segment.tokens[row][mask].tolist()
    return tokenizer.decode(tokens, skip_special_tokens=False)


def _decode_query_prompt(
    tokenizer: Tokenizer,
    batch: ETTRContinuationBatch,
    row: int,
) -> str:
    read = int(batch.episodes.query_read_index[row])
    tokens = batch.episodes.query.tokens[row, : read + 1].tolist()
    return tokenizer.decode(tokens, skip_special_tokens=False)


def _decode_token(tokenizer: Tokenizer, token: int) -> str:
    return tokenizer.decode([token], skip_special_tokens=False)


def _batch_metadata(
    tokenizer: Tokenizer,
    batch: ETTRContinuationBatch,
    kind: str,
) -> list[dict[str, object]]:
    (
        world_packet,
        world_command,
        world_target,
        command_packet,
        command_command,
        command_target,
    ) = batch.causal_rectangles.intervention_indices()
    if kind == "world":
        packet_index = world_packet
        command_index = world_command
        query_index = world_target
        foil_query_index = world_command
    elif kind == "command":
        packet_index = command_packet
        command_index = command_command
        query_index = command_target
        foil_query_index = command_packet
    else:
        raise ETTRV3EvaluationError("causal-query kind differs")
    depths = batch.transaction_targets.step_mask.sum(dim=1)
    records = []
    for pair_index in range(packet_index.shape[0]):
        packet_row = int(packet_index[pair_index])
        command_row = int(command_index[pair_index])
        query_row = int(query_index[pair_index])
        foil_query_row = int(foil_query_index[pair_index])
        depth = int(depths[query_row])
        records.append(
            {
                "command_episode_id": batch.episodes.episode_ids[command_row],
                "command_surface": _decode_segment(
                    tokenizer,
                    batch.episodes.command,
                    command_row,
                ),
                "depth": depth,
                "depth_bucket": _depth_bucket(depth),
                "foil_query_episode_id": (
                    batch.episodes.episode_ids[foil_query_row]
                ),
                "foil_query_prompt": _decode_query_prompt(
                    tokenizer,
                    batch,
                    foil_query_row,
                ),
                "packet_episode_id": batch.episodes.episode_ids[packet_row],
                "pair_index": pair_index,
                "query_episode_id": batch.episodes.episode_ids[query_row],
                "query_prompt": _decode_query_prompt(
                    tokenizer,
                    batch,
                    query_row,
                ),
                "world_surface": _decode_segment(
                    tokenizer,
                    batch.episodes.world,
                    packet_row,
                ),
            }
        )
    return records


def _retain_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    retain_per_depth: int,
) -> list[Mapping[str, object]]:
    selected: list[Mapping[str, object]] = []
    by_depth: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if row["checkpoint"]["contrast"] is True:
            by_depth.setdefault(str(row["depth_bucket"]), []).append(row)
    for population in by_depth.values():
        ordered = sorted(
            population,
            key=lambda row: (
                str(row["query_episode_id"]),
                str(row["packet_episode_id"]),
            ),
        )
        selected.extend(ordered[:retain_per_depth])
    extremes = sorted(
        (
            row
            for row in rows
            if row["checkpoint"]["contrast"] is True
        ),
        key=lambda row: float(
            row["checkpoint"]["difference_in_differences"]
        ),
    )
    selected.extend(extremes[:2])
    selected.extend(extremes[-2:])
    unique: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in selected:
        key = (
            str(row["packet_episode_id"]),
            str(row["command_episode_id"]),
            str(row["query_episode_id"]),
        )
        unique.setdefault(key, row)
    return list(unique.values())


def _attach_decoded_predictions(
    tokenizer: Tokenizer,
    row: dict[str, object],
) -> None:
    for arm in ("raw", "checkpoint"):
        values = row[arm]
        for name in (
            "correct_prediction",
            "correct_target",
            "foil_prediction",
            "foil_target",
        ):
            values[f"{name}_text"] = _decode_token(
                tokenizer,
                int(values[name]),
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise ETTRV3EvaluationError("ETTR causal-query probe requires CUDA")
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRV3EvaluationError("ETTR causal-query probe requires H100")

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    run_contract = _read_hash_bound_json(
        args.run_contract,
        expected_sha256=args.run_contract_sha256,
        label="ETTR run contract",
    )
    model_config, optimizer_config = _validate_run_contract(
        run_contract,
        release_sha256=args.release_sha256,
        release_source_commit=stream.release["source_commit"],
        architecture_seed=args.architecture_seed,
    )
    raw_model, raw_provenance = _build_model(
        args.protected_checkpoint,
        architecture_seed=args.architecture_seed,
        model_config=model_config,
        device=device,
    )
    if (
        raw_provenance.checkpoint_sha256
        != stream.manifest.protected_checkpoint_sha256
        or run_contract["parameter_receipt"]
        != asdict(raw_model.parameter_receipt())
    ):
        raise ETTRV3EvaluationError("ETTR causal-query provenance differs")
    checkpoint_model, checkpoint_provenance = _build_model(
        args.protected_checkpoint,
        architecture_seed=args.architecture_seed,
        model_config=model_config,
        device=device,
    )
    optimizer = ETTROptimizerBundle(checkpoint_model, optimizer_config)
    resumed = load_ettr_checkpoint(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        model=checkpoint_model,
        protected_base=checkpoint_provenance,
        optimizer=optimizer,
        scheduler=None,
    )
    _validate_checkpoint_cursor(
        resumed.progress,
        resumed.data_stream,
        run_contract=run_contract,
        stream=stream,
        release_sha256=args.release_sha256,
        protected_step=checkpoint_provenance.step,
    )
    del optimizer
    raw_model.eval()
    checkpoint_model.eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    rows: dict[str, list[dict[str, object]]] = {
        "command": [],
        "world": [],
    }
    packet_index = ETTRDiskPacketSufficiencyIndex(
        stream.packet_index_root
    )
    batches = 0
    try:
        iterator = stream.iter_positioned_batches(
            "development",
            rank=0,
            world_size=1,
            epoch=0,
            seed=args.data_seed,
        )
        for position, cpu_batch in iterator:
            if batches >= args.max_batches:
                break
            packet_index.verify_validation((cpu_batch,))
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(
                raw_model.config,
                ETTRObjectiveConfig(vocab_size=raw_model.base.cfg.vocab_size),
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                raw_pairs, raw_states = _objective_pairs(raw_model, batch)
                checkpoint_pairs, checkpoint_states = _objective_pairs(
                    checkpoint_model,
                    batch,
                )
            for kind in ("command", "world"):
                metadata = _batch_metadata(tokenizer, cpu_batch, kind)
                raw_values = _pair_rows(raw_pairs[kind])
                checkpoint_values = _pair_rows(checkpoint_pairs[kind])
                if not (
                    len(metadata)
                    == len(raw_values)
                    == len(checkpoint_values)
                    == len(raw_states[kind])
                    == len(checkpoint_states[kind])
                ):
                    raise ETTRV3EvaluationError(
                        "causal-query probe row population differs"
                    )
                for values, raw, checkpoint, raw_state, checkpoint_state in zip(
                    metadata,
                    raw_values,
                    checkpoint_values,
                    raw_states[kind],
                    checkpoint_states[kind],
                    strict=True,
                ):
                    values["development_position"] = position
                    values["raw"] = raw
                    values["checkpoint"] = checkpoint
                    values["raw_state"] = raw_state
                    values["checkpoint_state"] = checkpoint_state
                    rows[kind].append(values)
            batches += 1
            del (
                batch,
                raw_pairs,
                raw_states,
                checkpoint_pairs,
                checkpoint_states,
            )
    finally:
        packet_index.close()
    if batches != args.max_batches:
        raise ETTRV3EvaluationError(
            "causal-query development split is too short"
        )

    retained = {
        kind: _retain_examples(
            values,
            retain_per_depth=args.retain_per_depth,
        )
        for kind, values in rows.items()
    }
    for values in retained.values():
        for row in values:
            _attach_decoded_predictions(tokenizer, row)
    report = {
        "architecture_seed": args.architecture_seed,
        "arms": {
            arm: {
                kind: {
                    "query": _summary(
                        [
                            row[arm]
                            | {"depth_bucket": row["depth_bucket"]}
                            for row in values
                        ]
                    ),
                    "state": _state_summary(
                        [row[f"{arm}_state"] for row in values]
                    ),
                }
                for kind, values in rows.items()
            }
            for arm in ("raw", "checkpoint")
        },
        "batches": batches,
        "checkpoint": {
            "parameter_sha256": _parameter_sha256(checkpoint_model),
            "progress": asdict(resumed.progress),
            "run_contract_sha256": args.run_contract_sha256,
            "sha256": args.checkpoint_sha256,
        },
        "data_seed": args.data_seed,
        "device": {
            "bf16": torch.cuda.is_bf16_supported(),
            "name": torch.cuda.get_device_name(device),
        },
        "protected_checkpoint_sha256": (
            raw_provenance.checkpoint_sha256
        ),
        "raw_parameter_sha256": _parameter_sha256(raw_model),
        "release_file_sha256": args.release_sha256,
        "release_manifest_sha256": stream.manifest.sha256(),
        "retained_examples": retained,
        "schema": REPORT_SCHEMA,
        "source_commit": args.source_commit,
        "source_verification": source_verification,
        "split": "development",
        "tokenizer_sha256": _sha256_file(args.tokenizer),
    }
    payload = (
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    digest = _write_no_replace(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
