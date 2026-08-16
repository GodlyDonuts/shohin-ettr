from __future__ import annotations

import hashlib

import build_q36_mtr_synthesis_training as module


def _candidate(lineage: str, correct: bool, tokens: int) -> dict:
    return {
        "lineage": lineage,
        "completion": f"Attempt from {lineage}",
        "correct": correct,
        "generated_tokens": tokens,
        "max_token_exhausted": False,
    }


def _row(index: int, pattern: str) -> dict:
    return {
        "identity_sha256": hashlib.sha256(f"row-{index}".encode()).hexdigest(),
        "task": "math500",
        "question": f"Compute {index}.",
        "candidates": [
            _candidate(lineage, bit == "1", 10 + owner)
            for owner, (lineage, bit) in enumerate(
                zip(module.sparse.LINEAGES, pattern, strict=True)
            )
        ],
    }


def test_target_prefers_shortest_verified_candidate() -> None:
    target, lineage, count = module._target(_row(0, "011"))
    assert lineage == "owner_71"
    assert target == "Attempt from owner_71"
    assert count == 2


def test_presentations_skip_all_wrong_and_reach_exact_geometry() -> None:
    patterns = ("000", "001", "010", "011", "100", "101", "110", "111")
    rows = [_row(index, patterns[index % len(patterns)]) for index in range(160)]
    presentations, report = module.build_presentations(rows, total=965)
    assert len(presentations) == 965
    assert report["eligible_identities"] == 140
    assert presentations[0]["presentation_index"] == 0
    assert presentations[-1]["presentation_index"] == 964
    assert all("Internal draft:\n" in row["question"] for row in presentations)
    assert all(row["development_labels_read"] == 0 for row in presentations)
