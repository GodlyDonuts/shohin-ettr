from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from build_diverge_vmt1_board import SCHEMA, VMTBoardError, build_board


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return list(range(max(len(text.split()), 1)))


def _candidate(identity: str, group: str, sample: int, correct: bool) -> dict:
    return {
        "schema": "shohin-product-rollout-candidate-v1",
        "identity_sha256": identity,
        "question": f"question {identity}",
        "completion": f"candidate {sample} gives a complete distinct derivation for {identity}",
        "correct": correct,
        "sample_index": sample,
        "training_group": group,
        "task": "math500" if group == "math" else "bbh_logic",
        "prediction": str(int(correct)),
        "row_seed": sample,
        "max_token_exhausted": False,
    }


class VMTBoardTest(unittest.TestCase):
    def test_balanced_exact_board(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidates.jsonl"
            rows = []
            for group in ("math", "science"):
                for correct_index in (0, 1):
                    for index in range(3):
                        identity = f"{group}-{correct_index}-{index}"
                        rows.extend(
                            _candidate(identity, group, sample, sample == correct_index)
                            for sample in (0, 1)
                        )
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "board.jsonl"
            report_path = root / "report.json"
            report = build_board(
                source,
                output,
                report_path,
                tokenizer=FakeTokenizer(),
                render_prompt=lambda tokenizer, question: f"prompt {question}",
                per_cell=2,
                max_sequence_length=128,
                workspace_slots=8,
                seed=7,
            )
            selected = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["rows"], 8)
            self.assertEqual(len(selected), 8)
            self.assertEqual({row["schema"] for row in selected}, {SCHEMA})
            self.assertEqual(
                {(row["training_group"], row["correct_index"]) for row in selected},
                {("math", 0), ("math", 1), ("science", 0), ("science", 1)},
            )
            self.assertTrue(all(sum(row["correct"]) == 1 for row in selected))

    def test_exhausted_and_same_outcome_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidates.jsonl"
            rows = []
            for group in ("math", "science"):
                for correct_index in (0, 1):
                    identity = f"good-{group}-{correct_index}"
                    rows.extend(
                        _candidate(identity, group, sample, sample == correct_index)
                        for sample in (0, 1)
                    )
            bad = [_candidate("bad", "math", sample, True) for sample in (0, 1)]
            exhausted = [
                _candidate("exhausted", "science", sample, sample == 0)
                for sample in (0, 1)
            ]
            exhausted[1]["max_token_exhausted"] = True
            rows.extend(bad + exhausted)
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = build_board(
                source,
                root / "board.jsonl",
                root / "report.json",
                tokenizer=FakeTokenizer(),
                render_prompt=lambda tokenizer, question: question,
                per_cell=1,
                max_sequence_length=128,
                workspace_slots=8,
                seed=9,
            )
            self.assertEqual(report["counters"]["same_outcome"], 1)
            self.assertEqual(report["counters"]["exhausted_completion"], 1)

    def test_missing_balanced_cell_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidates.jsonl"
            rows = [
                _candidate("only", "math", sample, sample == 0) for sample in (0, 1)
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaises(VMTBoardError):
                build_board(
                    source,
                    root / "board.jsonl",
                    root / "report.json",
                    tokenizer=FakeTokenizer(),
                    render_prompt=lambda tokenizer, question: question,
                    per_cell=1,
                    max_sequence_length=128,
                    workspace_slots=8,
                    seed=1,
                )


if __name__ == "__main__":
    unittest.main()
