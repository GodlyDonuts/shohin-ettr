from __future__ import annotations

import torch
import torch.nn.functional as F

from operation_effect_set_diagnostics import (
    effect_set_batch_counts,
    summarize_effect_diagnostics,
)
from operation_state_transition_compiler import (
    EFFECT_ALLOCATE,
    EFFECT_COMMIT,
    EFFECT_KIND_COUNT,
    EFFECT_LINK,
    EFFECT_NOOP,
    EFFECT_ROOT_SET,
)
from parallel_terminal_state_compiler import AtomicTypedEdits


def _exact_edits(*, collapsed: bool = False) -> tuple[
    AtomicTypedEdits,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    batch = 1
    effects = 6
    slots = 3
    relations = 1
    values = 6
    types = 4
    node_action = torch.tensor([[0, 1, 0]])
    value_code = torch.tensor([[0, 4, 0]])
    type_index = torch.tensor([[0, 2, 0]])
    relation_action = torch.zeros(batch, relations, slots, slots, dtype=torch.long)
    relation_action[0, 0, 1, 2] = 1
    root_action = torch.tensor([3])
    disposition_action = torch.tensor([1])
    target = {
        "node_action": node_action,
        "value_code": value_code,
        "type_index": type_index,
        "relation_action": relation_action,
        "root_action": root_action,
        "disposition_action": disposition_action,
    }

    kinds = [
        EFFECT_COMMIT,
        EFFECT_LINK,
        EFFECT_ALLOCATE,
        EFFECT_ROOT_SET,
        EFFECT_NOOP,
        EFFECT_NOOP,
    ]
    if collapsed:
        kinds = [EFFECT_NOOP] * effects
    effect_kind = F.one_hot(torch.tensor([kinds]), EFFECT_KIND_COUNT).float()
    node_pointer = torch.zeros(batch, effects, 2, slots)
    node_pointer[:, :, :, 0] = 1.0
    node_pointer[0, 2, 0] = F.one_hot(torch.tensor(1), slots).float()
    effect_value = F.one_hot(torch.zeros(batch, effects, dtype=torch.long), values).float()
    effect_value[0, 2] = F.one_hot(torch.tensor(4), values).float()
    effect_type = F.one_hot(torch.zeros(batch, effects, dtype=torch.long), types).float()
    effect_type[0, 2] = F.one_hot(torch.tensor(2), types).float()
    relation_link = torch.zeros(batch, effects, relations, slots, slots)
    relation_link[:, :, 0, 0, 0] = 1.0
    relation_link[0, 1].zero_()
    relation_link[0, 1, 0, 1, 2] = 1.0
    relation_unlink = relation_link.clone()
    root_pointer = F.one_hot(
        torch.zeros(batch, effects, dtype=torch.long), slots
    ).float()
    root_pointer[0, 3] = F.one_hot(torch.tensor(1), slots).float()

    edits = AtomicTypedEdits(
        node_action=F.one_hot(node_action, 5).float(),
        value_code=F.one_hot(value_code, values).float(),
        type_index=F.one_hot(type_index, types).float(),
        relation_action=F.one_hot(relation_action, 3).float(),
        root_action=F.one_hot(root_action, slots + 2).float(),
        disposition_action=F.one_hot(disposition_action, 4).float(),
        effect_kind=effect_kind,
        effect_node_pointer=node_pointer,
        effect_value_code=effect_value,
        effect_type_index=effect_type,
        effect_relation_link=relation_link,
        effect_relation_unlink=relation_unlink,
        effect_root_pointer=root_pointer,
    )
    slot_mask = torch.ones(batch, slots, dtype=torch.bool)
    relation_mask = torch.ones(
        batch, relations, slots, slots, dtype=torch.bool
    )
    return edits, target, slot_mask, relation_mask


def test_effect_set_diagnostic_accepts_permuted_exact_set() -> None:
    edits, target, slot_mask, relation_mask = _exact_edits()
    report = summarize_effect_diagnostics(
        effect_set_batch_counts(
            edits,
            target,
            slot_mask=slot_mask,
            relation_mask=relation_mask,
        )
    )
    assert report["exact_rates"]["complete_effect_set_exact"] == 1.0
    assert report["exact_rates"]["complete_dense_edit_exact"] == 1.0
    assert report["positive_exact_rates"]["relation_link"] == 1.0


def test_effect_set_diagnostic_exposes_noop_collapse() -> None:
    edits, target, slot_mask, relation_mask = _exact_edits(collapsed=True)
    report = summarize_effect_diagnostics(
        effect_set_batch_counts(
            edits,
            target,
            slot_mask=slot_mask,
            relation_mask=relation_mask,
        )
    )
    assert report["exact_rates"]["kind_multiset_exact"] == 0.0
    assert report["positive_exact_rates"]["relation_link"] == 0.0
    assert report["per_kind"][str(EFFECT_NOOP)]["predicted"] == 6
