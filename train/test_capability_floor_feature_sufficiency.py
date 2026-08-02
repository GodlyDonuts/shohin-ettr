from types import SimpleNamespace

import torch

from capability_floor_feature_sufficiency import (
    _balanced_orbit_indices,
    _label_opposed_derangement,
    _runtime_state_payload,
)
from capability_floor_trajectory import UnifiedTrajectoryConfig


def _config() -> UnifiedTrajectoryConfig:
    return UnifiedTrajectoryConfig(
        input_width=12,
        state_width=12,
        num_slots=64,
        num_types=4,
        num_relations=3,
        num_value_codes=8,
        num_heads=3,
        core_layers=1,
        reader_layers=1,
        ff_multiplier=2,
        max_world_steps=2,
        max_command_steps=2,
        max_edges=16,
    )


def test_runtime_state_payload_preserves_exact_typed_fields() -> None:
    active = [False] * 64
    active[2] = True
    active[7] = True
    types = [0] * 64
    types[2] = 1
    types[7] = 3
    values = [0] * 64
    values[2] = 6
    values[7] = 4
    root = [False] * 64
    root[7] = True
    state = SimpleNamespace(
        active=active,
        type_index=types,
        value_code=values,
        root=root,
        relations=((2, 2, 7),),
        committed=True,
        halted=False,
    )
    payload = _runtime_state_payload(state, _config())
    assert payload["active"].nonzero().flatten().tolist() == [2, 7]
    assert payload["type_probabilities"][2, 1] == 1
    assert payload["type_probabilities"][7, 3] == 1
    assert payload["value_probabilities"][2, 6] == 1
    assert payload["value_probabilities"][7, 4] == 1
    assert payload["relations"][2, 2, 7] == 1
    assert payload["root"][7] == 1
    assert payload["committed"] == 1


def test_balanced_orbit_selection_keeps_whole_renderer_orbits() -> None:
    labels = []
    orbit_ids = []
    for label, count in ((0, 3), (1, 2), (2, 4)):
        for group in range(count):
            orbit = f"label={label}:group={group}"
            labels.extend([label] * 4)
            orbit_ids.extend([orbit] * 4)
    selected = _balanced_orbit_indices(
        torch.tensor(labels),
        orbit_ids,
        seed=11,
    )
    selected_labels = torch.tensor(labels).index_select(0, selected)
    assert selected.numel() == 3 * 2 * 4
    assert torch.bincount(selected_labels, minlength=3).tolist() == [8, 8, 8]
    counts = {}
    for index in selected.tolist():
        counts[orbit_ids[index]] = counts.get(orbit_ids[index], 0) + 1
    assert set(counts.values()) == {4}


def test_label_opposed_derangement_is_bijective_and_cross_family() -> None:
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    permutation = _label_opposed_derangement(labels)
    assert permutation.unique().numel() == labels.numel()
    assert not labels.index_select(0, permutation).eq(labels).any()
