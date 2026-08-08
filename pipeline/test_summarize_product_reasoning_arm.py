from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.summarize_product_reasoning_arm import (
    ProductArmSummaryError,
    TASKS,
    summarize_arm,
)


TOTALS = {
    "gsm8k": 100,
    "math500": 100,
    "humaneval": 20,
    "mbpp": 20,
    "gpqa": 198,
    "bbh_logic": 100,
    "aime": 30,
}


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    prefix = root / "eval"
    for task in TASKS:
        Path(f"{prefix}_{task}.json").write_text(
            json.dumps(
                {
                    "correct": min(10, TOTALS[task]),
                    "status": "complete",
                    "task": task,
                    "total": TOTALS[task],
                }
            ),
            encoding="utf-8",
        )
    training = root / "training.json"
    training.write_text(json.dumps({"status": "complete", "updates": 256}))
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    return prefix, training, checkpoint


def test_summarizes_all_domains_and_hashes_inputs(tmp_path: Path) -> None:
    prefix, training, checkpoint = _fixture(tmp_path)
    report = summarize_arm(
        name="B1-9B",
        eval_prefix=prefix,
        training_report=training,
        checkpoint=checkpoint,
    )
    assert report["scores"]["solved"] == 60
    assert report["scores"]["total"] == 538
    assert report["scores"]["domains"]["code"]["correct"] == 20
    assert len(report["checkpoint"]["sha256"]) == 64


def test_missing_task_fails_closed(tmp_path: Path) -> None:
    prefix, training, checkpoint = _fixture(tmp_path)
    Path(f"{prefix}_gpqa.json").unlink()
    with pytest.raises(ProductArmSummaryError, match="missing artifact"):
        summarize_arm(
            name="B1-9B",
            eval_prefix=prefix,
            training_report=training,
            checkpoint=checkpoint,
        )
