from types import SimpleNamespace

import torch

from endogenous_typed_theory_reactor import TheoryReactorConfig, TypedTheoryState
from ettr_query_supervision import ETTRQuerySpecBatch, _query_specs
from train_typed_query_state_reader_pilot import _compiler_loss
from typed_query_state_reader import TypedQueryStateReader


def _state(config: TheoryReactorConfig) -> TypedTheoryState:
    batch = 2
    values = torch.zeros(batch, config.num_slots, config.num_value_codes)
    types = torch.zeros(batch, config.num_slots, config.num_types)
    values[:, :, 0] = 1.0
    types[:, :, 0] = 1.0
    active = torch.ones(batch, config.num_slots)
    root = torch.zeros_like(active)
    root[:, 0] = 1.0
    return TypedTheoryState(
        value_probabilities=values,
        type_probabilities=types,
        relations=torch.zeros(
            batch,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ),
        active=active,
        root=root,
        committed=torch.ones(batch),
        halted=torch.zeros(batch),
        step=1,
    )


def test_query_factor_parser_accepts_both_frozen_schemas() -> None:
    record = SimpleNamespace(
        assessor_only=SimpleNamespace(
            semantic_factors=SimpleNamespace(
                queries=(
                    {"args": [4], "op": "resource_cursor_ge"},
                    {
                        "arguments": [2, 1],
                        "op": "same_type_slots_equal",
                    },
                )
            )
        )
    )
    assert _query_specs(record) == ((3, (4,)), (9, (2, 1)))


def test_typed_reader_hides_post_read_tokens_and_scatter_is_exact() -> None:
    torch.manual_seed(7)
    config = TheoryReactorConfig(
        d_model=64,
        state_width=64,
        num_slots=4,
        num_types=3,
        num_relations=2,
        num_value_codes=8,
        max_edges=8,
        num_heads=4,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        max_steps=4,
        stage_after_block=1,
    )
    reader = TypedQueryStateReader(
        config,
        source_vocab_size=32,
        target_vocab_size=41,
        answer_token_ids=(3, 5, 7, 11),
        width=64,
        query_layers=1,
        state_layers=1,
        num_heads=4,
        max_query_tokens=6,
    ).eval()
    tokens = torch.tensor([[2, 4, 6, 8, 10, 12], [1, 3, 5, 7, 9, 11]])
    changed = tokens.clone()
    changed[:, 4:] = torch.tensor([[13, 14], [15, 16]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    read_index = torch.tensor([3, 3])
    state = _state(config)
    with torch.inference_mode():
        left = reader(tokens, mask, read_index, state, state)
        right = reader(changed, mask, read_index, state, state)
    torch.testing.assert_close(left.vocab_logits, right.vocab_logits)
    assert left.vocab_logits.shape == (2, 41)
    legal = torch.tensor([3, 5, 7, 11])
    illegal = torch.ones(41, dtype=torch.bool)
    illegal[legal] = False
    assert bool(
        left.vocab_logits[:, illegal]
        .eq(left.vocab_logits[:, illegal][:, :1])
        .all()
    )


def test_teacher_program_changes_only_program_selection_path() -> None:
    torch.manual_seed(11)
    config = TheoryReactorConfig(
        d_model=64,
        state_width=64,
        num_slots=4,
        num_types=3,
        num_relations=2,
        num_value_codes=8,
        max_edges=8,
        num_heads=4,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        max_steps=4,
        stage_after_block=1,
    )
    reader = TypedQueryStateReader(
        config,
        source_vocab_size=32,
        target_vocab_size=41,
        answer_token_ids=(3, 5, 7, 11),
        width=64,
        query_layers=1,
        state_layers=1,
        num_heads=4,
        max_query_tokens=6,
    ).eval()
    tokens = torch.tensor([[2, 4, 6, 8, 10, 12], [1, 3, 5, 7, 9, 11]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    read_index = torch.tensor([3, 3])
    state = _state(config)
    program = ETTRQuerySpecBatch(
        operation=torch.tensor([0, 10]),
        arguments=torch.tensor([[1, 2, 3], [2, 0, 0]]),
        argument_mask=torch.tensor(
            [[True, True, True], [True, False, False]]
        ),
    )
    with torch.inference_mode():
        autonomous = reader(tokens, mask, read_index, state, state)
        teacher = reader(
            tokens,
            mask,
            read_index,
            state,
            state,
            teacher_program=program,
        )
    torch.testing.assert_close(
        autonomous.operation_logits,
        teacher.operation_logits,
    )
    assert not torch.equal(autonomous.class_logits, teacher.class_logits)


def test_compiler_loss_accepts_an_all_zero_arity_batch() -> None:
    output = SimpleNamespace(
        operation_logits=torch.randn(2, 11, requires_grad=True),
        argument_logits=torch.randn(2, 3, 28, requires_grad=True),
        argument_present_logits=torch.randn(2, 3, 2, requires_grad=True),
    )
    specs = ETTRQuerySpecBatch(
        operation=torch.tensor([4, 4]),
        arguments=torch.zeros(2, 3, dtype=torch.long),
        argument_mask=torch.zeros(2, 3, dtype=torch.bool),
    )
    loss, parts = _compiler_loss(output, specs)
    assert float(parts["argument"].detach()) == 0.0
    loss.backward()
    assert output.argument_logits.grad is not None
