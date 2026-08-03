import pytest
import torch

from capability_floor_layer_taps import (
    CapabilityFloorLayerTapError,
    TAP_NAMES,
    pool_task_taps,
    source_matched_world_swap_indices,
    virtual_feature_bundle,
)


def test_pool_task_taps_preserves_tap_and_public_role_geometry() -> None:
    taps = {
        name: torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        + offset
        for offset, name in enumerate(TAP_NAMES)
    }
    masks = (
        (
            (True, False, False, False, False),
            (False, True, True, False, False),
            (False, False, False, False, False),
            (False, False, False, True, False),
        ),
    )
    pooled, present = pool_task_taps(
        taps,
        row=1,
        source_length=5,
        role_masks=masks,
    )
    assert pooled.shape == (1, len(TAP_NAMES), 4, 3)
    assert present.tolist() == [[True, True, False, True]]
    assert pooled[0, 0, 0].float().tolist() == taps[TAP_NAMES[0]][1, 0].tolist()
    assert torch.equal(pooled[0, :, 2], torch.zeros_like(pooled[0, :, 2]))


def _bundle() -> dict[str, object]:
    examples = 3
    state = {
        "value_probabilities": torch.zeros(examples, 4, 7, dtype=torch.bool),
        "type_probabilities": torch.zeros(examples, 4, 3, dtype=torch.bool),
        "relations": torch.zeros(examples, 2, 4, 4, dtype=torch.bool),
        "active": torch.zeros(examples, 4, dtype=torch.bool),
        "root": torch.zeros(examples, 4, dtype=torch.bool),
        "committed": torch.zeros(examples, dtype=torch.bool),
    }
    split = {
        "identity": {
            "orbit_ids": [f"orbit-{index}" for index in range(examples)],
            "sample_ids": [f"sample-{index}" for index in range(examples)],
            "source_sha256": ["a" * 64] * examples,
        },
        "split_sha256": "b" * 64,
        "tensors": {
            "role_features": torch.randn(examples, len(TAP_NAMES), 4, 16),
            "role_present": torch.tensor(
                [[True, True, False, False]] * examples,
                dtype=torch.bool,
            ),
            "labels": torch.tensor([0, 1, 2]),
            "state": state,
        },
    }
    return {
        "candidate": "candidate",
        "config": {
            "input_width": 16,
            "state_width": 16,
            "num_slots": 4,
            "num_types": 3,
            "num_relations": 2,
            "num_value_codes": 7,
            "num_heads": 4,
            "core_layers": 1,
            "reader_layers": 1,
            "ff_multiplier": 2,
            "max_world_steps": 3,
            "max_command_steps": 3,
            "min_world_steps": 1,
            "min_command_steps": 1,
            "max_edges": 8,
        },
        "splits": {"train": split, "development": split},
    }


def test_virtual_bundle_exposes_only_selected_tap_as_role_tokens() -> None:
    bundle = virtual_feature_bundle(
        _bundle(),
        tap_name="block-09",
        bundle_sha256="c" * 64,
    )
    tensors = bundle["splits"]["train"]["tensors"]
    assert tensors["source_features"].shape == (3, 4, 16)
    assert tensors["source_mask"].tolist() == [[True, True, False, False]] * 3
    assert tensors["role_masks"][0].tolist() == [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]


def test_pool_rejects_role_geometry_drift() -> None:
    taps = {name: torch.zeros(1, 3, 4) for name in TAP_NAMES}
    with pytest.raises(CapabilityFloorLayerTapError, match="role geometry"):
        pool_task_taps(
            taps,
            row=0,
            source_length=3,
            role_masks=(((True, False, False),),),
        )


def test_source_matched_world_swap_keeps_only_label_changing_worlds() -> None:
    split = _bundle()["splits"]["development"]
    split["identity"] = {
        "orbit_ids": [
            "core:0:0",
            "core:2:0",
            "core:1:0",
            "core:3:0",
        ],
        "sample_ids": [
            "core:0:0:view=0",
            "core:2:0:view=0",
            "core:1:0:view=0",
            "core:3:0:view=0",
        ],
        "source_sha256": ["a" * 64, "a" * 64, "b" * 64, "b" * 64],
    }
    split["tensors"]["labels"] = torch.tensor([0, 2, 1, 1])
    source, swapped = source_matched_world_swap_indices(split)
    assert source.tolist() == [0, 1]
    assert swapped.tolist() == [1, 0]


def test_source_matched_world_swap_rejects_different_command_bytes() -> None:
    split = _bundle()["splits"]["development"]
    split["identity"] = {
        "orbit_ids": ["core:0:0", "core:2:0"],
        "sample_ids": ["core:0:0:view=0", "core:2:0:view=0"],
        "source_sha256": ["a" * 64, "b" * 64],
    }
    split["tensors"]["labels"] = torch.tensor([0, 2])
    with pytest.raises(CapabilityFloorLayerTapError, match="COMMAND source"):
        source_matched_world_swap_indices(split)
