"""Exact held-out diagnostics for unordered operation effect sets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import torch

from parallel_terminal_state_compiler import AtomicTypedEdits
from operation_state_transition_compiler import (
    EFFECT_ALLOCATE,
    EFFECT_CLEAR,
    EFFECT_COMMIT,
    EFFECT_HALT,
    EFFECT_KIND_COUNT,
    EFFECT_LINK,
    EFFECT_NOOP,
    EFFECT_REJECT,
    EFFECT_REPLACE,
    EFFECT_ROOT_CLEAR,
    EFFECT_ROOT_SET,
    EFFECT_UNLINK,
    EFFECT_WRITE,
)
from train_parallel_terminal_state_pilot import _operation_effect_targets


class OperationEffectDiagnosticError(RuntimeError):
    """An effect-set diagnostic input or receipt differs."""


_ENTITY_KINDS = frozenset((EFFECT_ALLOCATE, EFFECT_WRITE, EFFECT_CLEAR, EFFECT_REPLACE))
_ROOT_KINDS = frozenset((EFFECT_ROOT_CLEAR, EFFECT_ROOT_SET))
_DISPOSITION_KINDS = frozenset((EFFECT_COMMIT, EFFECT_HALT, EFFECT_REJECT))


def _signature(
    kind: int,
    node: int,
    relation: int,
    root: int,
    value: int,
    type_index: int,
) -> tuple[int, int, int, int, int, int]:
    if kind == EFFECT_NOOP:
        return (kind, 0, 0, 0, 0, 0)
    if kind == EFFECT_ALLOCATE:
        return (kind, node, 0, 0, value, type_index)
    if kind == EFFECT_WRITE:
        return (kind, node, 0, 0, value, 0)
    if kind == EFFECT_CLEAR:
        return (kind, node, 0, 0, 0, 0)
    if kind == EFFECT_REPLACE:
        return (kind, node, 0, 0, value, type_index)
    if kind in (EFFECT_LINK, EFFECT_UNLINK):
        return (kind, 0, relation, 0, 0, 0)
    if kind == EFFECT_ROOT_SET:
        return (kind, 0, 0, root, 0, 0)
    if kind in (EFFECT_ROOT_CLEAR, EFFECT_COMMIT, EFFECT_HALT, EFFECT_REJECT):
        return (kind, 0, 0, 0, 0, 0)
    raise OperationEffectDiagnosticError("operation effect kind differs")


def _required_effect_fields(edits: AtomicTypedEdits) -> tuple[torch.Tensor, ...]:
    values = (
        edits.effect_kind,
        edits.effect_node_pointer,
        edits.effect_value_code,
        edits.effect_type_index,
        edits.effect_relation_link,
        edits.effect_relation_unlink,
        edits.effect_root_pointer,
    )
    if any(value is None for value in values):
        raise OperationEffectDiagnosticError(
            "operation effect predictions are incomplete"
        )
    return tuple(value for value in values if value is not None)


def _masked_row_exact(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape != target.shape or mask.shape != target.shape:
        raise OperationEffectDiagnosticError("dense effect geometry differs")
    return (predicted.eq(target) | ~mask).flatten(1).all(-1)


def effect_set_batch_counts(
    edits: AtomicTypedEdits,
    target: Mapping[str, torch.Tensor],
    *,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> dict[str, object]:
    """Return exact unordered and dense-action counts for one selected batch."""

    (
        effect_kind,
        effect_node_pointer,
        effect_value_code,
        effect_type_index,
        effect_relation_link,
        effect_relation_unlink,
        effect_root_pointer,
    ) = _required_effect_fields(edits)
    if (
        effect_kind.ndim != 3
        or effect_kind.shape[-1] != EFFECT_KIND_COUNT
        or slot_mask.shape != target["node_action"].shape
        or relation_mask.shape != target["relation_action"].shape
    ):
        raise OperationEffectDiagnosticError("operation effect geometry differs")
    batch, effects, _ = effect_kind.shape
    labels = _operation_effect_targets(
        dict(target),
        maximum_effects=effects,
        slot_mask=slot_mask,
        relation_mask=relation_mask,
    )

    predicted_kind = effect_kind.argmax(-1)
    pointer_channel = predicted_kind.ne(EFFECT_ALLOCATE).to(torch.long)
    predicted_node = (
        effect_node_pointer.gather(
            2,
            pointer_channel[:, :, None, None].expand(
                -1,
                -1,
                1,
                effect_node_pointer.shape[-1],
            ),
        )
        .squeeze(2)
        .argmax(-1)
    )
    predicted_link = effect_relation_link.flatten(2).argmax(-1)
    predicted_unlink = effect_relation_unlink.flatten(2).argmax(-1)
    predicted_relation = torch.where(
        predicted_kind.eq(EFFECT_UNLINK),
        predicted_unlink,
        predicted_link,
    )
    predicted_root = effect_root_pointer.argmax(-1)
    predicted_value = effect_value_code.argmax(-1)
    predicted_type = effect_type_index.argmax(-1)

    cpu_fields = {
        "predicted_kind": predicted_kind.detach().cpu().tolist(),
        "predicted_node": predicted_node.detach().cpu().tolist(),
        "predicted_relation": predicted_relation.detach().cpu().tolist(),
        "predicted_root": predicted_root.detach().cpu().tolist(),
        "predicted_value": predicted_value.detach().cpu().tolist(),
        "predicted_type": predicted_type.detach().cpu().tolist(),
        "target_kind": labels["kind"].detach().cpu().tolist(),
        "target_node": labels["node"].detach().cpu().tolist(),
        "target_relation": labels["relation"].detach().cpu().tolist(),
        "target_root": labels["root"].detach().cpu().tolist(),
        "target_value": labels["value"].detach().cpu().tolist(),
        "target_type": labels["type"].detach().cpu().tolist(),
    }
    counts: Counter[str] = Counter(instances=batch)
    target_histogram: Counter[str] = Counter()
    predicted_histogram: Counter[str] = Counter()
    true_positive_histogram: Counter[str] = Counter()

    categories = {
        "entity": _ENTITY_KINDS,
        "relation_link": frozenset((EFFECT_LINK,)),
        "relation_unlink": frozenset((EFFECT_UNLINK,)),
        "root": _ROOT_KINDS,
        "disposition": _DISPOSITION_KINDS,
    }
    for row in range(batch):
        predicted = []
        expected = []
        for rank in range(effects):
            predicted_kind_value = cpu_fields["predicted_kind"][row][rank]
            target_kind_value = cpu_fields["target_kind"][row][rank]
            predicted_histogram[str(predicted_kind_value)] += 1
            target_histogram[str(target_kind_value)] += 1
            if predicted_kind_value != EFFECT_NOOP:
                predicted.append(
                    _signature(
                        predicted_kind_value,
                        cpu_fields["predicted_node"][row][rank],
                        cpu_fields["predicted_relation"][row][rank],
                        cpu_fields["predicted_root"][row][rank],
                        cpu_fields["predicted_value"][row][rank],
                        cpu_fields["predicted_type"][row][rank],
                    )
                )
            if target_kind_value != EFFECT_NOOP:
                expected.append(
                    _signature(
                        target_kind_value,
                        cpu_fields["target_node"][row][rank],
                        cpu_fields["target_relation"][row][rank],
                        cpu_fields["target_root"][row][rank],
                        cpu_fields["target_value"][row][rank],
                        cpu_fields["target_type"][row][rank],
                    )
                )

        predicted_counter = Counter(predicted)
        expected_counter = Counter(expected)
        predicted_kinds = Counter(value[0] for value in predicted)
        expected_kinds = Counter(value[0] for value in expected)
        if all(
            kind in (EFFECT_WRITE, EFFECT_LINK)
            for kind in expected_kinds
            if kind != EFFECT_NOOP
        ):
            predicted_write = predicted_kinds[EFFECT_WRITE] > 0
            predicted_link = predicted_kinds[EFFECT_LINK] > 0
            expected_write = expected_kinds[EFFECT_WRITE] > 0
            expected_link = expected_kinds[EFFECT_LINK] > 0
            if expected_write and expected_link:
                raise OperationEffectDiagnosticError(
                    "operation effect target family is not exclusive"
                )
            predicted_family = (
                3
                if predicted_write and predicted_link
                else 1
                if predicted_write
                else 2
                if predicted_link
                else 0
            )
            expected_family = 1 if expected_write else 2 if expected_link else 0
            counts["operation_family_exact"] += predicted_family == expected_family
            counts["predicted_operation_family_conflict"] += (
                predicted_write and predicted_link
            )
            counts["operation_family_instances"] += 1
        for kind in range(EFFECT_KIND_COUNT):
            true_positive_histogram[str(kind)] += min(
                predicted_kinds[kind], expected_kinds[kind]
            )
        counts["effect_count_exact"] += len(predicted) == len(expected)
        counts["kind_multiset_exact"] += predicted_kinds == expected_kinds
        counts["complete_effect_set_exact"] += predicted_counter == expected_counter
        for name, kinds in categories.items():
            predicted_subset = Counter(
                value for value in predicted if value[0] in kinds
            )
            expected_subset = Counter(value for value in expected if value[0] in kinds)
            counts[f"{name}_set_exact"] += predicted_subset == expected_subset
            if expected_subset:
                counts[f"{name}_positive_instances"] += 1
                counts[f"{name}_positive_exact"] += predicted_subset == expected_subset

    node_action = edits.node_action.argmax(-1)
    relation_action = edits.relation_action.argmax(-1)
    root_action = edits.root_action.argmax(-1)
    disposition_action = edits.disposition_action.argmax(-1)
    value_code = edits.value_code.argmax(-1)
    type_index = edits.type_index.argmax(-1)
    target_node = target["node_action"]
    value_mask = slot_mask & (target_node.eq(1) | target_node.eq(2) | target_node.eq(4))
    type_mask = slot_mask & (target_node.eq(1) | target_node.eq(4))
    dense_exact = {
        "node_action_exact": _masked_row_exact(node_action, target_node, slot_mask),
        "relation_action_exact": _masked_row_exact(
            relation_action, target["relation_action"], relation_mask
        ),
        "root_action_exact": root_action.eq(target["root_action"]),
        "disposition_action_exact": disposition_action.eq(target["disposition_action"]),
        "value_payload_exact": _masked_row_exact(
            value_code, target["value_code"], value_mask
        ),
        "type_payload_exact": _masked_row_exact(
            type_index, target["type_index"], type_mask
        ),
    }
    complete = torch.ones(batch, dtype=torch.bool, device=effect_kind.device)
    for name, value in dense_exact.items():
        counts[name] += int(value.sum().detach().cpu())
        complete &= value
    counts["complete_dense_edit_exact"] += int(complete.sum().detach().cpu())
    return {
        "counts": dict(counts),
        "predicted_kind_histogram": dict(predicted_histogram),
        "target_kind_histogram": dict(target_histogram),
        "true_positive_kind_histogram": dict(true_positive_histogram),
    }


def merge_effect_diagnostics(
    destination: dict[str, object],
    source: Mapping[str, object],
) -> None:
    """Merge disjoint batch diagnostics without discarding zero classes."""

    for section in (
        "counts",
        "predicted_kind_histogram",
        "target_kind_histogram",
        "true_positive_kind_histogram",
    ):
        target_values = destination.setdefault(section, {})
        source_values = source.get(section)
        if not isinstance(target_values, dict) or not isinstance(
            source_values, Mapping
        ):
            raise OperationEffectDiagnosticError(
                "operation effect diagnostic receipt differs"
            )
        for name, value in source_values.items():
            if not isinstance(name, str) or not isinstance(value, int):
                raise OperationEffectDiagnosticError(
                    "operation effect diagnostic count differs"
                )
            target_values[name] = int(target_values.get(name, 0)) + value


def summarize_effect_diagnostics(values: Mapping[str, object]) -> dict[str, object]:
    """Add rates and per-kind precision/recall to raw exact counts."""

    counts = values.get("counts")
    predicted = values.get("predicted_kind_histogram")
    target = values.get("target_kind_histogram")
    true_positive = values.get("true_positive_kind_histogram")
    if not all(
        isinstance(value, Mapping)
        for value in (counts, predicted, target, true_positive)
    ):
        raise OperationEffectDiagnosticError(
            "operation effect diagnostic summary differs"
        )
    assert isinstance(counts, Mapping)
    assert isinstance(predicted, Mapping)
    assert isinstance(target, Mapping)
    assert isinstance(true_positive, Mapping)
    instances = int(counts.get("instances", 0))
    if instances <= 0:
        raise OperationEffectDiagnosticError("operation effect diagnostic is empty")
    rates = {
        name: int(value) / instances
        for name, value in counts.items()
        if name.endswith("_exact")
        and not name.endswith("_positive_exact")
        and name not in {"operation_state_exact", "terminal_state_exact"}
    }
    family_instances = int(counts.get("operation_family_instances", 0))
    diagnostic_rates = {
        "predicted_operation_family_conflict": (
            None
            if family_instances == 0
            else int(counts.get("predicted_operation_family_conflict", 0))
            / family_instances
        ),
    }
    if family_instances:
        rates["operation_family_exact"] = (
            int(counts.get("operation_family_exact", 0)) / family_instances
        )
    positive_rates = {}
    for name, value in counts.items():
        if not name.endswith("_positive_exact"):
            continue
        prefix = name[: -len("_positive_exact")]
        denominator = int(counts.get(f"{prefix}_positive_instances", 0))
        positive_rates[prefix] = None if denominator == 0 else int(value) / denominator
    per_kind = {}
    for kind in range(EFFECT_KIND_COUNT):
        key = str(kind)
        tp = int(true_positive.get(key, 0))
        predicted_count = int(predicted.get(key, 0))
        target_count = int(target.get(key, 0))
        per_kind[key] = {
            "precision": None if predicted_count == 0 else tp / predicted_count,
            "predicted": predicted_count,
            "recall": None if target_count == 0 else tp / target_count,
            "target": target_count,
            "true_positive": tp,
        }
    return {
        "counts": dict(counts),
        "exact_rates": rates,
        "diagnostic_rates": diagnostic_rates,
        "positive_exact_rates": positive_rates,
        "predicted_kind_histogram": dict(predicted),
        "target_kind_histogram": dict(target),
        "per_kind": per_kind,
    }


__all__ = [
    "OperationEffectDiagnosticError",
    "effect_set_batch_counts",
    "merge_effect_diagnostics",
    "summarize_effect_diagnostics",
]
