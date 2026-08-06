"""Frozen-threshold checks for the RSM1 component scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from score_diverge_rsm1_component import score


def _training() -> dict:
    return {
        "schema": "shohin-diverge-rsm1-training-report-v1",
        "status": "complete",
        "architecture": "diverge-rsm1",
        "packet_arm": "guarded",
        "updates": 1600,
        "identities_per_update": 8,
        "frozen_crp_unchanged": True,
        "replay_changed": True,
        "crp_checkpoint_sha256": "a" * 64,
        "data_sha256": "b" * 64,
    }


def _evaluation(*, symbolic_terminal: int = 136) -> dict:
    families = {
        "scalar": {
            "rows": 160,
            "terminal_correct": 150,
            "full_trajectory_correct": 140,
        },
        "register": {
            "rows": 160,
            "terminal_correct": 150,
            "full_trajectory_correct": 140,
        },
        "symbolic": {
            "rows": 160,
            "terminal_correct": symbolic_terminal,
            "full_trajectory_correct": 128,
        },
    }
    return {
        "schema": "shohin-diverge-rsm1-evaluation-v1",
        "status": "complete",
        "packet_arm": "guarded",
        "trace_kind": "wrong",
        "selection_mode": "forced",
        "ablation": "normal",
        "checkpoint_update": 1600,
        "model_unchanged": True,
        "crp_checkpoint_sha256": "a" * 64,
        "data_sha256": "c" * 64,
        "runtime_semantic_calls": 0,
        "overall": {
            "rows": 480,
            "terminal_correct": 436,
            "full_trajectory_correct": 408,
            "packet_correct": 480,
            "invalid_terminal": 0,
        },
        "families": families,
        "results": [{} for _ in range(480)],
    }


class RSM1GateTest(unittest.TestCase):
    def _run(self, evaluation: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.json"
            eval_path = root / "eval.json"
            output = root / "gate.json"
            train.write_text(json.dumps(_training()))
            eval_path.write_text(json.dumps(evaluation))
            return score(
                argparse.Namespace(
                    train_report=train,
                    evaluation=eval_path,
                    output=output,
                )
            )

    def test_exact_boundary_passes(self) -> None:
        self.assertTrue(self._run(_evaluation())["gate_pass"])

    def test_family_terminal_shortfall_fails(self) -> None:
        self.assertFalse(
            self._run(_evaluation(symbolic_terminal=135))["gate_pass"]
        )


if __name__ == "__main__":
    unittest.main()
