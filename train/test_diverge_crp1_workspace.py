import pytest
import torch

from diverge_crp1_workspace import (
    CRP1WorkspaceError,
    CausalRevisionConfig,
    CausalRevisionPacket,
)


def _inputs():
    memory = torch.randn(2, 18, 24)
    active = torch.ones(2, 18, dtype=torch.bool)
    problem = torch.zeros_like(active)
    problem[:, :3] = True
    steps = torch.zeros(2, 6, 18, dtype=torch.bool)
    for batch, depth in enumerate((4, 6)):
        for index in range(depth):
            steps[batch, index, 3 + 2 * index : 5 + 2 * index] = True
    final = torch.zeros_like(active)
    final[0, 11:13] = True
    final[1, 15:17] = True
    return memory, active, problem, steps, final


def test_packet_selects_whole_candidate_and_masks_inactive() -> None:
    model = CausalRevisionPacket(
        CausalRevisionConfig(
            backbone_width=24,
            workspace_width=16,
            workspace_slots=3,
            recurrent_steps=2,
            attention_heads=4,
            max_trace_steps=6,
        )
    )
    inputs = _inputs()
    targets = torch.tensor([2, 5])
    output = model(*inputs, selection_targets=targets)
    assert output.prefix_states.shape == (2, 3, 24)
    assert output.candidate_logits.shape == (2, 7)
    assert output.selected_candidates.tolist() == [2, 5]
    assert output.candidate_active[0].tolist() == [True] * 5 + [False] * 2
    assert torch.all(output.candidate_logits[0, 5:] < -9999)
    expected = output.all_candidate_prefixes[torch.arange(2), targets]
    assert torch.equal(output.prefix_states, expected)


def test_reset_shift_and_packet_swap_are_causal() -> None:
    model = CausalRevisionPacket(
        CausalRevisionConfig(
            backbone_width=24,
            workspace_width=16,
            workspace_slots=3,
            recurrent_steps=1,
            attention_heads=4,
            max_trace_steps=6,
        )
    ).eval()
    inputs = _inputs()
    normal = model(*inputs)
    reset = model(*inputs, ablation="reset")
    shifted = model(*inputs, ablation="shift")
    swapped = model(*inputs, ablation="packet_swap")
    assert torch.count_nonzero(reset.prefix_states) == 0
    assert torch.all(shifted.selected_candidates >= 1)
    assert torch.equal(swapped.prefix_states[0], normal.prefix_states[1])
    assert torch.equal(swapped.prefix_states[1], normal.prefix_states[0])


def test_unguarded_arm_has_same_parameter_contract() -> None:
    model = CausalRevisionPacket(
        CausalRevisionConfig(
            backbone_width=24,
            workspace_width=16,
            workspace_slots=3,
            recurrent_steps=1,
            attention_heads=4,
            max_trace_steps=6,
        )
    )
    guarded = model(*_inputs())
    unguarded = model(*_inputs(), unguarded=True)
    assert guarded.candidate_logits.shape == unguarded.candidate_logits.shape
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_sparse_step_masks_fail_closed() -> None:
    model = CausalRevisionPacket(
        CausalRevisionConfig(
            backbone_width=24,
            workspace_width=16,
            workspace_slots=3,
            recurrent_steps=1,
            attention_heads=4,
            max_trace_steps=6,
        )
    )
    inputs = list(_inputs())
    inputs[3][0, 1] = False
    with pytest.raises(CRP1WorkspaceError):
        model(*inputs)
