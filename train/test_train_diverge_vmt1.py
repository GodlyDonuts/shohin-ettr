"""CPU contract tests for the DIVERGE-VMT1 trainer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hf_product_reasoning_train import ProductReasoningTrainError
from train_diverge_vmt1 import (
    VMT1_BOARD_SCHEMA,
    load_exact_board,
    reduce_fit_gate,
    tokenize_exact_board,
)


def _rows() -> list[dict]:
    rows = []
    for group in ("math", "science"):
        for correct_index in (0, 1):
            for index in range(4):
                responses = [f"response {index} left", f"response {index} right"]
                rows.append(
                    {
                        "schema": VMT1_BOARD_SCHEMA,
                        "identity_sha256": f"{group}-{correct_index}-{index}",
                        "question": f"question {group} {correct_index} {index}",
                        "responses": responses,
                        "correct": [correct_index == 0, correct_index == 1],
                        "correct_index": correct_index,
                        "training_group": group,
                        "token_accounting": {
                            "prompt_tokens": 0,
                            "response_tokens": [0, 0],
                            "workspace_slots": 2,
                            "maximum_total_tokens": 0,
                        },
                    }
                )
    return rows


class _Tokenizer:
    eos_token_id = 47
    chat_template = None

    @staticmethod
    def encode(text: str, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [len(piece) % 43 + 1 for piece in text.split()]


class TrainVMT1Tests(unittest.TestCase):
    def test_exact_board_and_token_accounting(self) -> None:
        rows = _rows()
        tokenizer = _Tokenizer()
        for row in rows:
            prompt = tokenizer.encode(
                "System: You are a careful reasoning assistant. Give concise, "
                "verifiable reasoning and a clearly marked final answer.\n\n"
                f"User: {row['question']}\n\nAssistant:",
                False,
            )
            responses = [
                tokenizer.encode(response, False) for response in row["responses"]
            ]
            row["token_accounting"] = {
                "prompt_tokens": len(prompt),
                "response_tokens": [len(response) for response in responses],
                "workspace_slots": 2,
                "maximum_total_tokens": max(
                    len(prompt) + len(response) + 3 for response in responses
                ),
            }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            board = root / "board.jsonl"
            report = root / "report.json"
            board.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            digest = hashlib.sha256(board.read_bytes()).hexdigest()
            report.write_text(
                json.dumps(
                    {
                        "schema": VMT1_BOARD_SCHEMA,
                        "status": "complete",
                        "output_sha256": digest,
                        "rows": 16,
                    }
                ),
                encoding="utf-8",
            )
            loaded, _, _ = load_exact_board(board, report, expected_rows=16)
            tokenized = tokenize_exact_board(
                tokenizer,
                loaded,
                max_sequence_length=128,
                workspace_slots=2,
            )
            self.assertEqual(len(tokenized), 16)
            self.assertTrue(
                all(
                    response[-1] == 47
                    for row in tokenized
                    for response in row["response_tokens"]
                )
            )

    def test_board_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            board = root / "board.jsonl"
            report = root / "report.json"
            board.write_text("{}\n", encoding="utf-8")
            report.write_text(
                json.dumps(
                    {
                        "schema": VMT1_BOARD_SCHEMA,
                        "status": "complete",
                        "output_sha256": "wrong",
                        "rows": 16,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProductReasoningTrainError, "hash"):
                load_exact_board(board, report, expected_rows=16)

    def test_fit_gate_accepts_only_full_conjunction(self) -> None:
        records = [{"correct_index": 0 if index < 8 else 1} for index in range(16)]
        before = {
            "selected_correct_response_nll_rows": [1.0] * 16,
            "finite": True,
        }
        after = {
            "selected_correct_response_nll_rows": [0.5] * 16,
            "selector_correct_rows": [True] * 16,
            "selector_correct": 16,
            "selector_accuracy": 1.0,
            "swapped_selector_accuracy": 0.0,
            "mean_matched_trace_cosine": 0.9,
            "mean_crossed_trace_cosine": 0.7,
            "mean_internal_trajectory_cosine": 0.8,
            "finite": True,
        }
        gate = reduce_fit_gate(
            before,
            after,
            records,
            frozen_parameters_unchanged=True,
            training_finite=True,
        )
        self.assertTrue(gate["qualified"])
        after["mean_internal_trajectory_cosine"] = 0.99
        self.assertFalse(
            reduce_fit_gate(
                before,
                after,
                records,
                frozen_parameters_unchanged=True,
                training_finite=True,
            )["qualified"]
        )


if __name__ == "__main__":
    unittest.main()
