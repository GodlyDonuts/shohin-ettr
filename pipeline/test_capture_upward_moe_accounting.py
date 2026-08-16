from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from capture_upward_moe_accounting import UpwardMoEAccountingError, capture


def _runner(text: str):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=text, stderr="")

    return run


def _row(job: str, elapsed: int, gpu: int = 1, raw_job: str | None = None) -> str:
    tres = "billing=16,cpu=16,mem=192G,node=1"
    if gpu:
        tres += f",gres/gpu={gpu},gres/gpu:nvidia_h100_pcie={gpu}"
    return (
        f"{job}|{raw_job or job}|COMPLETED|0:0|{elapsed}|{tres}|normal|evc35|0|"
        "2026-08-17T20:00:00|2026-08-17T20:01:00"
    )


def test_captures_exact_multistage_h100_seconds(tmp_path: Path) -> None:
    rows = [_row(str(job), 60) for job in range(101, 113)] + [_row("113", 7, 0)]
    allocations = [
        f"{stage},{job},1"
        for stage, jobs in (
            ("mechanics", range(101, 105)),
            ("training", range(105, 109)),
            ("evaluation", range(109, 113)),
        )
        for job in jobs
    ] + ["score,113,0"]
    output = tmp_path / "accounting.json"
    result = capture(
        host="Mixtral-8x22B",
        source_commit="a" * 40,
        allocations=allocations,
        output=output,
        runner=_runner("\n".join(rows)),
    )
    assert result["charged_gpu_seconds"] == 12 * 60
    assert result["charged_h100_hours"] == 0.2
    assert result["allocation_count"] == 13
    assert len(result["stages"]["mechanics"]) == 4
    assert json.loads(output.read_text()) == result
    assert not output.stat().st_mode & 0o222


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("FAILED|1:0", "complete exactly once"),
        ("COMPLETED|0:0|60|billing=16,cpu=16,mem=192G,node=1", "GPU count"),
    ],
)
def test_rejects_failed_or_non_gpu_allocation(
    tmp_path: Path, replacement: str, message: str
) -> None:
    row = _row("101", 60)
    if replacement.startswith("FAILED"):
        row = row.replace("COMPLETED|0:0", replacement)
    else:
        row = row.replace(
            "COMPLETED|0:0|60|billing=16,cpu=16,mem=192G,node=1,gres/gpu=1,gres/gpu:nvidia_h100_pcie=1",
            replacement,
        )
    with pytest.raises(UpwardMoEAccountingError, match=message):
        capture(
            host="Mixtral-8x22B",
            source_commit="a" * 40,
            allocations=["mechanics,101,1"],
            output=tmp_path / "accounting.json",
            runner=_runner(row),
        )


def test_rejects_missing_or_duplicate_job_coverage(tmp_path: Path) -> None:
    with pytest.raises(UpwardMoEAccountingError, match="coverage"):
        capture(
            host="Mixtral-8x22B",
            source_commit="a" * 40,
            allocations=["mechanics,101,1", "mechanics,102,1"],
            output=tmp_path / "accounting.json",
            runner=_runner(_row("101", 60)),
        )
    with pytest.raises(UpwardMoEAccountingError, match="identities"):
        capture(
            host="Mixtral-8x22B",
            source_commit="a" * 40,
            allocations=["mechanics,101,1", "training,101,1"],
            output=tmp_path / "accounting.json",
            runner=_runner(_row("101", 60)),
        )


def test_array_task_identity_uses_stable_job_id_not_internal_raw_id(
    tmp_path: Path,
) -> None:
    result = capture(
        host="Nemotron-Super",
        source_commit="b" * 40,
        allocations=["evaluation,760385_0,2"],
        output=tmp_path / "accounting.json",
        runner=_runner(_row("760385_0", 90, 2, raw_job="760900")),
    )
    record = result["stages"]["evaluation"][0]
    assert record["job_id"] == "760385_0"
    assert record["job_id_raw"] == "760900"


def test_job_is_cpu_only_runtime_bound_and_nonrequeueing() -> None:
    source = (
        Path(__file__).with_name("jobs") / "capture_upward_moe_accounting.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --no-requeue" in source
    assert "q36_verify_runtime" in source
    assert "RUNTIME_MANIFEST_SHA256" in source
    assert '[[ "$OUTPUT" == /* && ! -e "$OUTPUT"' in source
