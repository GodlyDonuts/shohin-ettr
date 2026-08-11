"""Static custody checks for VTE1 CPU admission."""

from pathlib import Path


SCRIPT = Path("pipeline/jobs/build_vte1_equivalence_data.sbatch").read_text()


def test_vte1_job_is_cpu_only_and_hash_bound() -> None:
    assert "--gres=" not in SCRIPT
    assert "RUNTIME_MANIFEST_SHA256" in SCRIPT
    assert "SOURCE_SHA256" in SCRIPT
    assert "SOURCE_REPORT_SHA256" in SCRIPT
    assert "MODEL_MANIFEST_SHA256" in SCRIPT
    assert "test ! -e \"$OUTPUT\"" in SCRIPT


def test_vte1_job_keeps_all_draft_shards_and_no_holdout() -> None:
    assert "seq 0 15" in SCRIPT
    assert "--max-sequence-length 4096" in SCRIPT
    assert "build_vte1_equivalence_data.py" in SCRIPT
    assert "holdout" not in SCRIPT.lower()
