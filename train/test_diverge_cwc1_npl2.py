#!/usr/bin/env python3
"""Unit tests for the bounded CWC1-to-EWC1-to-NPL2 composition."""

from __future__ import annotations

import hashlib
import unittest

from diverge_cwc1_data import counterfactual_source
from diverge_cwc1_npl2_data import (
    _wrapper_record,
    audit_wrapper_records,
    validate_wrapper_record,
)
from eval_diverge_cwc1_ewc1_npl2 import _structure_score


def _fixture(serial: int) -> dict[str, object]:
    token = chr(ord("a") + serial % 26) * (1 + serial // 26)
    aliases = [f"alias{chr(ord('a') + index)}{token}q" for index in range(8)]
    registers = [f"regleft{token}q", f"regright{token}q"]
    symbols = (0, 3, 7, 1)
    source = (
        f"Begin with {registers[0]} = 31 and {registers[1]} = 72. "
        "Execute aliases in order: "
        + " | ".join(aliases[index] for index in symbols)
        + "."
    )
    program = {
        "depth": len(symbols),
        "program_id": hashlib.sha256(f"program:{serial}".encode()).hexdigest()[:24],
        "source_sha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
        "source_text": source,
    }
    episode = {
        "episode_id": hashlib.sha256(f"episode:{serial}".encode()).hexdigest()[:24],
        "aliases": aliases,
        "register_names": registers,
    }
    return _wrapper_record(
        split="development",
        episode=episode,
        phase="acquisition",
        phase_index=0,
        program=program,
        serial=serial,
    )


class CWC1NPL2Tests(unittest.TestCase):
    def test_wrapper_preserves_one_true_complete_world(self) -> None:
        row = _fixture(7)
        validate_wrapper_record(row)
        target = int(row["target_position"])
        self.assertTrue(row["candidate_programs"][target]["is_true"])
        self.assertFalse(row["candidate_programs"][1 - target]["is_true"])
        self.assertNotEqual(row["source_text"], counterfactual_source(row))

    def test_renderer_targets_are_balanced(self) -> None:
        report = audit_wrapper_records([_fixture(index) for index in range(128)])
        self.assertTrue(report["all_conditions_passed"])
        self.assertEqual(report["target_counts"], [64, 64])
        self.assertEqual(report["renderers"], 64)
        self.assertEqual(report["renderer_max_target_imbalance"], 0)

    def test_structure_controls_compare_whole_candidates(self) -> None:
        rows = [_fixture(index) for index in range(8)]
        targets = [int(row["target_position"]) for row in rows]
        opposites = [1 - target for target in targets]
        target_predictions = [
            (
                tuple(row["candidate_programs"][target]["initial_state"]),
                tuple(row["candidate_programs"][target]["symbols"]),
            )
            for row, target in zip(rows, targets, strict=True)
        ]
        opposite_predictions = [
            (
                tuple(row["candidate_programs"][opposite]["initial_state"]),
                tuple(row["candidate_programs"][opposite]["symbols"]),
            )
            for row, opposite in zip(rows, opposites, strict=True)
        ]
        selected = _structure_score(
            rows, targets, target_predictions, compare_to_true=True
        )
        forced = _structure_score(
            rows, opposites, opposite_predictions, compare_to_true=True
        )
        decoy_parse = _structure_score(
            rows, opposites, opposite_predictions, compare_to_true=False
        )
        self.assertEqual(selected["joint_rate"], 1.0)
        self.assertEqual(forced["joint_rate"], 0.0)
        self.assertEqual(decoy_parse["joint_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
