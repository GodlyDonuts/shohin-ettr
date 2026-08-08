#!/usr/bin/env python3
"""AQC1 product summary tests."""

from __future__ import annotations

import hashlib

from hf_aqc1_product_apply import arm_summary


def test_product_summary_keeps_whole_selected_trajectories() -> None:
    tasks = ("gsm8k", "math500", "gpqa", "bbh_logic", "humaneval", "mbpp", "aime")
    rows, selections = [], {}
    for index, task in enumerate(tasks):
        identity = hashlib.sha256(task.encode()).hexdigest()
        rows.append(
            {
                "identity_sha256": identity,
                "task": task,
                "candidates": [{"correct": False}, {"correct": True}],
            }
        )
        selections[identity] = 1
    summary = arm_summary(rows, selections, 2)
    assert summary["solved"] == 6
    assert summary["aime"]["correct"] == 1
    assert summary["domains"]["code"]["correct"] == 2
