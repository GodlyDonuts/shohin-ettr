"""Focused deterministic checks for the RSM1 replay supervisor."""

from __future__ import annotations

import unittest

from diverge_rsm1_data import (
    MAX_STATE_SLOTS,
    RSM1DataError,
    build_replay_supervision,
    decode_state,
    encode_state,
    operation_surfaces,
    replay_targets,
    supervisor_states,
    tokenize_replay_example,
)
from build_diverge_crp1_board import generate_episode


class RSM1DataTest(unittest.TestCase):
    def test_all_families_reconstruct_terminal_state(self) -> None:
        for index, family in enumerate(("scalar", "register", "symbolic")):
            row = generate_episode(4100 + index, family, 7, True)
            states = supervisor_states(row)
            self.assertEqual(len(states), 8)
            self.assertEqual(states[-1], row["answer"])
            self.assertTrue(all(len(encode_state(value)) == MAX_STATE_SLOTS for value in states))

    def test_state_code_is_canonical(self) -> None:
        for value in ("-12345", "17,-204", "abcdefgh"):
            self.assertEqual(decode_state(encode_state(value)), value)
        with self.assertRaises(RSM1DataError):
            encode_state("UPPER")

    def test_operation_surfaces_exclude_successors(self) -> None:
        for index, family in enumerate(("scalar", "register", "symbolic")):
            row = generate_episode(5100 + index, family, 8, True)
            surfaces = operation_surfaces(row)
            self.assertEqual(len(surfaces), row["depth"])
            self.assertTrue(all("->" not in value and "=" not in value for value in surfaces))

    def test_replay_targets_start_before_selected_step(self) -> None:
        row = generate_episode(6201, "register", 8, True)
        states = supervisor_states(row)
        selected = int(row["error_index"])
        initial, successors = replay_targets(row, selected)
        self.assertEqual(decode_state(initial), states[selected - 1])
        self.assertEqual(
            tuple(decode_state(value) for value in successors),
            states[selected:],
        )
        terminal, no_steps = replay_targets(row, 0)
        self.assertEqual(decode_state(terminal), states[-1])
        self.assertEqual(no_steps, ())

    def test_tokenized_executor_sees_only_operation_surface(self) -> None:
        class CharacterTokenizer:
            chat_template = None

            def __call__(self, text, **_kwargs):
                return {
                    "input_ids": [ord(value) for value in text],
                    "offset_mapping": [(index, index + 1) for index in range(len(text))],
                }

        tokenizer = CharacterTokenizer()
        for index, family in enumerate(("scalar", "register", "symbolic")):
            row = generate_episode(7100 + index, family, 7, True)
            tokens = tokenize_replay_example(
                tokenizer,
                row,
                row["wrong_steps"],
                f"Final answer: \\boxed{{{row['wrong_answer']}}}",
                max_sequence_length=8192,
                packet_slots=6,
            )
            self.assertIsNotNone(tokens)
            assert tokens is not None
            surfaces = operation_surfaces(row)
            for surface, mask in zip(surfaces, tokens.operation_masks, strict=True):
                observed = "".join(
                    chr(token)
                    for token, selected in zip(tokens.prompt_ids, mask, strict=True)
                    if selected
                )
                self.assertEqual(observed, surface)

    def test_fixed_supervision_has_no_boundary_shift(self) -> None:
        row = generate_episode(8201, "symbolic", 9, True)
        states = supervisor_states(row)
        selected = int(row["error_index"])
        supervision = build_replay_supervision(row, selected)
        self.assertEqual(
            decode_state(supervision.initial), states[selected - 1]
        )
        self.assertEqual(sum(supervision.free_active), 9 - selected + 1)
        for index in range(selected - 1, 9):
            self.assertEqual(
                decode_state(supervision.free_targets[index]), states[index + 1]
            )
        for index in range(9):
            self.assertEqual(
                decode_state(supervision.oracle_predecessors[index]), states[index]
            )
            self.assertEqual(
                decode_state(supervision.oracle_targets[index]), states[index + 1]
            )


if __name__ == "__main__":
    unittest.main()
