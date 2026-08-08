from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.analyze_product_lineage_union import (
    LineageUnionError,
    TASKS,
    analyze_union,
)


TOTALS = {
    "gsm8k": 4,
    "math500": 4,
    "humaneval": 2,
    "mbpp": 2,
    "gpqa": 4,
    "bbh_logic": 4,
    "aime": 3,
}


def _write(root: Path, arm: str, offset: int) -> Path:
    prefix = root / arm
    for task in TASKS:
        total = TOTALS[task]
        results = [
            {
                "correct": (index + offset) % 3 == 0,
                "identity_sha256": f"{task}-{index}",
            }
            for index in range(total)
        ]
        payload = {
            "correct": sum(row["correct"] for row in results),
            "data_sha256": f"data-{task}",
            "effective_enable_thinking": False,
            "generation_mode": "greedy",
            "generation_seed": 31,
            "generation_stop_token_ids": [1, 2],
            "max_new_tokens": 768,
            "results": results,
            "selection_sha256": f"selection-{task}",
            "status": "complete",
            "subset_seed": 20260802,
            "task": task,
            "total": total,
        }
        Path(f"{prefix}_{task}.json").write_text(json.dumps(payload), encoding="utf-8")
    return prefix


def test_union_counts_complementary_lineages(tmp_path: Path) -> None:
    report = analyze_union(
        left_name="base",
        left_prefix=_write(tmp_path, "base", 0),
        right_name="expert",
        right_prefix=_write(tmp_path, "expert", 1),
    )
    assert report["tasks"]["gsm8k"]["left_only"] > 0
    assert report["tasks"]["gsm8k"]["right_only"] > 0
    assert report["union"]["solved"] > 0


def test_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    left = _write(tmp_path, "base", 0)
    right = _write(tmp_path, "expert", 1)
    path = Path(f"{right}_gsm8k.json")
    payload = json.loads(path.read_text())
    payload["results"][0]["identity_sha256"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LineageUnionError, match="identities"):
        analyze_union(
            left_name="base",
            left_prefix=left,
            right_name="expert",
            right_prefix=right,
        )
