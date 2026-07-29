"""Static custody gates for the Essential-Web private review job."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "pipeline" / "jobs" / "essential_web_reasoning_review.sbatch"


def test_review_job_binds_every_local_module_and_selection_code():
    script = JOB.read_text()
    for name in (
        "build_general_source_review_packet.py",
        "profile_general_source.py",
        "tokenize_shards.py",
        "verify_tokenized_shards.py",
    ):
        assert name in script
    for variable in (
        "BUILDER_SHA256",
        "PROFILE_SHA256",
        "TOKENIZE_SHA256",
        "VERIFY_SHARDS_SHA256",
    ):
        assert f"${variable}" in script
    assert "--selection-code \"$TOKENIZE\"" in script


def test_review_job_keeps_text_private_and_does_not_admit_data():
    script = JOB.read_text()
    assert "$BASE/scratchpad/private_reviews/" in script
    assert "chmod 0700" in script
    assert "== 600" in script
    assert "corpus remains quarantined" in script
    assert "train.py" not in script
