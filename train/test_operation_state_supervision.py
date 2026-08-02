import torch
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    GenericTransactionReactor,
    TheoryReactorConfig,
    TypedTheoryState,
)
from ettr_objectives import ETTRTransactionTargets
from operation_state_supervision import (
    index_atomic_edits,
    operation_boundary_indices,
    oracle_operation_boundary_states,
)
from parallel_terminal_state_compiler import AtomicTypedEdits


def _targets() -> ETTRTransactionTargets:
    opcode = torch.tensor(
        [
            [1, 1, 1, 1, 1, 6, 0, 0],
            [1, 1, 1, 6, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    source = torch.tensor(
        [
            [48, 54, 49, 54, 55, 0, 0, 0],
            [48, 54, 55, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    step_mask = torch.tensor(
        [
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, False, False, False, False],
        ]
    )
    zeros = torch.zeros_like(opcode)
    return ETTRTransactionTargets(
        opcode=opcode,
        source=source,
        target=zeros,
        relation=zeros,
        type_index=zeros,
        value_code=torch.tensor(
            [
                [3, 1, 4, 2, 5, 0, 0, 0],
                [6, 1, 5, 0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
        committed=opcode.eq(6),
        halted=torch.zeros_like(step_mask),
        step_mask=step_mask,
    )


def _initial(config: TheoryReactorConfig) -> TypedTheoryState:
    active = torch.ones(2, config.num_slots)
    values = torch.zeros(2, config.num_slots, dtype=torch.long)
    types = torch.zeros_like(values)
    return TypedTheoryState(
        value_probabilities=F.one_hot(
            values, config.num_value_codes
        ).float(),
        type_probabilities=F.one_hot(types, config.num_types).float(),
        relations=torch.zeros(
            2,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ),
        active=active,
        root=F.one_hot(
            torch.zeros(2, dtype=torch.long), config.num_slots
        ).float(),
        committed=torch.zeros(2),
        halted=torch.zeros(2),
        step=0,
    )


def test_operation_boundaries_exclude_terminal_suffix() -> None:
    boundaries, mask = operation_boundary_indices(_targets())
    assert boundaries[0].tolist() == [1, 3, 0, 0, 0, 0]
    assert boundaries[1].tolist() == [1, 0, 0, 0, 0, 0]
    assert mask.sum(-1).tolist() == [2, 1]


def test_oracle_operation_states_replay_cumulative_writes() -> None:
    config = TheoryReactorConfig(
        d_model=16,
        state_width=16,
        num_slots=64,
        num_types=3,
        num_relations=2,
        num_value_codes=8,
        max_edges=8,
        num_heads=2,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        ff_multiplier=2,
        max_steps=16,
        stage_after_block=0,
    )
    result = oracle_operation_boundary_states(
        GenericTransactionReactor(config),
        _initial(config),
        _targets(),
    )
    first_values = result.states[0].value_probabilities.argmax(-1)
    second_values = result.states[1].value_probabilities.argmax(-1)
    assert first_values[:, 48].tolist() == [3, 6]
    assert first_values[:, 54].tolist() == [1, 1]
    assert second_values[0, 49].item() == 4
    assert second_values[0, 54].item() == 2
    assert result.last_state.value_probabilities.argmax(-1)[0, 49].item() == 4


def test_index_atomic_edits_preserves_operation_family() -> None:
    batch = 3
    edits = AtomicTypedEdits(
        node_action=torch.zeros(batch, 2, 5),
        value_code=torch.zeros(batch, 2, 7),
        type_index=torch.zeros(batch, 2, 3),
        relation_action=torch.zeros(batch, 2, 2, 2, 3),
        root_action=torch.zeros(batch, 4),
        disposition_action=torch.zeros(batch, 4),
        effect_family=F.one_hot(torch.tensor([0, 1, 2]), 3).float(),
    )
    selected = index_atomic_edits(edits, torch.tensor([2, 0]))
    assert selected.effect_family is not None
    assert selected.effect_family.argmax(-1).tolist() == [2, 0]
