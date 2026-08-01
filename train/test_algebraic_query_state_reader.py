import torch

from algebraic_query_state_reader import AlgebraicQueryStateReader
from endogenous_typed_theory_reactor import TheoryReactorConfig, TypedTheoryState
from ettr_query_supervision import (
    ETTRQuerySpecBatch,
    QUERY_OPERATION_TO_INDEX,
)


def _config() -> TheoryReactorConfig:
    return TheoryReactorConfig(
        d_model=64,
        state_width=64,
        num_slots=64,
        num_types=8,
        num_relations=16,
        num_value_codes=256,
        max_edges=256,
        num_heads=4,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        max_steps=64,
        stage_after_block=1,
    )


def _state(config: TheoryReactorConfig) -> TypedTheoryState:
    values = torch.zeros(1, config.num_slots, config.num_value_codes)
    values[:, :, 0] = 1.0
    types = torch.zeros(1, config.num_slots, config.num_types)
    types[:, :, 0] = 1.0
    active = torch.ones(1, config.num_slots)
    return TypedTheoryState(
        value_probabilities=values,
        type_probabilities=types,
        relations=torch.zeros(
            1,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ),
        active=active,
        root=torch.zeros_like(active),
        committed=torch.ones(1),
        halted=torch.zeros(1),
        step=1,
    )


def _write_value(state: TypedTheoryState, slot: int, value: int) -> None:
    state.value_probabilities[:, slot] = 0.0
    state.value_probabilities[:, slot, value] = 1.0


def _spec(operation: str, arguments: tuple[int, ...]) -> ETTRQuerySpecBatch:
    padded = (*arguments, *(0 for _ in range(3 - len(arguments))))
    return ETTRQuerySpecBatch(
        operation=torch.tensor([QUERY_OPERATION_TO_INDEX[operation]]),
        arguments=torch.tensor([padded]),
        argument_mask=torch.tensor(
            [[index < len(arguments) for index in range(3)]]
        ),
    )


def _truth(
    reader: AlgebraicQueryStateReader,
    initial: TypedTheoryState,
    terminal: TypedTheoryState,
    operation: str,
    arguments: tuple[int, ...],
) -> bool:
    output = reader(
        torch.tensor([[2, 3, 4, 5]]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([3]),
        initial,
        terminal,
        teacher_program=_spec(operation, arguments),
    )
    return int(output.class_logits[0, :2].argmax()) == 1


def test_algebraic_oracle_program_executes_all_query_families() -> None:
    torch.manual_seed(3)
    config = _config()
    reader = AlgebraicQueryStateReader(
        config,
        source_vocab_size=32,
        target_vocab_size=41,
        answer_token_ids=(3, 5, 7, 11),
        width=64,
        query_layers=1,
        num_heads=4,
        max_query_tokens=6,
    ).eval()

    initial = _state(config)
    terminal = _state(config)
    terminal.relations[:, 11, 32, 35] = 1.0
    assert _truth(reader, initial, terminal, "horn_has", (3, 0, 3))
    assert _truth(reader, initial, terminal, "horn_count_ge", (1,))
    assert not _truth(reader, initial, terminal, "horn_count_ge", (2,))

    initial = _state(config)
    terminal = _state(config)
    _write_value(terminal, 32, 67)
    _write_value(terminal, 54, 69)
    _write_value(terminal, 55, 148)
    assert _truth(reader, initial, terminal, "resource_place_ge", (0, 2))
    assert not _truth(reader, initial, terminal, "resource_place_ge", (0, 3))
    assert _truth(reader, initial, terminal, "resource_cursor_ge", (4,))
    assert _truth(reader, initial, terminal, "resource_halt", ())

    initial = _state(config)
    terminal = _state(config)
    for slot, symbol in enumerate((2, 1, 2, 3, 2, 0)):
        _write_value(initial, 32 + slot, 33 + symbol)
        _write_value(terminal, 32 + slot, 33 + symbol)
    assert _truth(reader, initial, terminal, "slot_is", (2, 2))
    assert _truth(reader, initial, terminal, "type_count_ge", (0, 2, 3))
    assert _truth(reader, initial, terminal, "adjacent_is", (0, 2, 1))
    assert _truth(reader, initial, terminal, "pattern_exists", (1, 2))
    assert _truth(reader, initial, terminal, "same_type_slots_equal", (0, 2))
    assert not _truth(reader, initial, terminal, "slot_changed", (4,))
    _write_value(terminal, 36, 33)
    assert _truth(reader, initial, terminal, "slot_changed", (4,))


def test_algebraic_reader_hides_post_boundary_query_tokens() -> None:
    torch.manual_seed(5)
    config = _config()
    reader = AlgebraicQueryStateReader(
        config,
        source_vocab_size=32,
        target_vocab_size=41,
        answer_token_ids=(3, 5, 7, 11),
        width=64,
        query_layers=1,
        num_heads=4,
        max_query_tokens=6,
    ).eval()
    initial = _state(config)
    terminal = _state(config)
    left = torch.tensor([[2, 4, 6, 8, 10, 12]])
    right = left.clone()
    right[:, 4:] = torch.tensor([[13, 14]])
    mask = torch.ones_like(left, dtype=torch.bool)
    with torch.inference_mode():
        first = reader(left, mask, torch.tensor([3]), initial, terminal)
        second = reader(right, mask, torch.tensor([3]), initial, terminal)
    torch.testing.assert_close(first.vocab_logits, second.vocab_logits)


def test_algebraic_truth_supplies_state_gradients() -> None:
    torch.manual_seed(7)
    config = _config()
    reader = AlgebraicQueryStateReader(
        config,
        source_vocab_size=32,
        target_vocab_size=41,
        answer_token_ids=(3, 5, 7, 11),
        width=64,
        query_layers=1,
        num_heads=4,
        max_query_tokens=6,
    ).eval()
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    initial = _state(config)
    terminal = _state(config)
    terminal.value_probabilities.requires_grad_(True)
    output = reader(
        torch.tensor([[2, 3, 4, 5]]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([3]),
        initial,
        terminal,
        teacher_program=_spec("slot_is", (0, 1)),
    )
    (-output.class_logits[0, 1]).backward()
    gradient = terminal.value_probabilities.grad
    assert gradient is not None
    assert float(gradient[0, 32, 34].abs()) > 0.0


def test_semantic_basis_contains_only_query_consumed_coordinates() -> None:
    config = _config()
    reader = AlgebraicQueryStateReader(
        config,
        source_vocab_size=32,
        target_vocab_size=41,
        answer_token_ids=(3, 5, 7, 11),
        width=64,
        query_layers=1,
        num_heads=4,
        max_query_tokens=6,
    ).eval()
    initial = _state(config)
    terminal = _state(config)
    _write_value(initial, 32, 34)
    _write_value(terminal, 32, 35)
    _write_value(terminal, 54, 69)
    _write_value(terminal, 55, 148)
    terminal.relations[:, 11, 32, 35] = 1.0

    initial_basis, terminal_basis = reader.semantic_basis(initial, terminal)
    assert initial_basis.shape == (1, 24)
    assert terminal_basis.shape == (1, 284)
    assert initial_basis[0, 1] == 1.0
    assert terminal_basis.sum() > 0.0

    terminal.value_probabilities[:, 0, 200] = 1.0
    unchanged = reader.semantic_basis(initial, terminal)[1]
    torch.testing.assert_close(terminal_basis, unchanged)
