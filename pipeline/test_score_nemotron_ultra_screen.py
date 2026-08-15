"""Contract tests for the 550B-A55B paired score reducer."""

from __future__ import annotations

import json
from pathlib import Path

import score_nemotron_super_screen as base
import score_nemotron_ultra_screen as ultra


def test_ultra_candidate_schema_is_distinct_and_exact(tmp_path: Path) -> None:
    identity = "a" * 64
    paths = []
    for index in range(base.SHARDS):
        path = tmp_path / f"{index}.jsonl"
        rows = []
        if index == 0:
            rows.append(
                {
                    "schema": ultra.CANDIDATE_SCHEMA,
                    "arm": "revision",
                    "identity_sha256": identity,
                    "task": "math500",
                    "completion": "answer",
                    "generated_tokens": 1,
                    "max_token_exhausted": False,
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        paths.append(path)
    loaded = base.load_candidates(
        "revision",
        paths,
        {identity},
        candidate_schema=ultra.CANDIDATE_SCHEMA,
    )
    assert set(loaded) == {identity}
    assert ultra.HOST == "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
    assert ultra.TOTAL_PARAMETERS == 550_000_000_000
    assert ultra.ACTIVE_PARAMETERS == 55_000_000_000


def test_ultra_score_wrapper_is_single_cpu_write_once_and_sandboxed() -> None:
    source = (
        Path(__file__)
        .with_name("jobs")
        .joinpath("nemotron_ultra_score.sbatch")
        .read_text(encoding="utf-8")
    )
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --gres" not in source
    assert "score_nemotron_ultra_screen.py" in source
    assert "q36_init_local_tmp" in source
    assert '[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]' in source
    assert 'chmod a-w "$OUTPUT" "$SANDBOX_RECEIPT"' in source
