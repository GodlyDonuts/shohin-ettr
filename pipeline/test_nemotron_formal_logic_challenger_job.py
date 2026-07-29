"""Static gates for the pinned Nemotron Formal Logic challenger."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = (
    ROOT
    / "pipeline"
    / "jobs"
    / "nemotron_formal_logic_challenger.sbatch"
)


def test_job_pins_source_revision_bytes_and_hashes():
    script = JOB.read_text()
    for value in (
        "13fa979be2e7f7e62913eee0ec5e97c8fd6e24af",
        "73074239",
        "b8b641309dec27e836db6a2003d3045c6ed060c8d6a23e3a93100e1c8a3a450b",
        "7402",
        "93594fcc5d7e0a85bb8cbc6561b13d9d4c5c9cf4d21e7f81388829e86542e1bf",
    ):
        assert value in script
    assert "stat -c %a" in script
    assert "sha256sum -c" in script


def test_job_keeps_formal_logic_separate_and_hard_filtered():
    script = JOB.read_text()
    assert "CONFIG=Nemotron-Pretraining-Formal-Logic" in script
    assert "--input-files \"$SOURCE\"" in script
    assert "--required-value license=cc-by-4.0" in script
    assert '--required-value metadata.category="$CONFIG"' in script
    assert "--exact-dedup" in script
    assert "--decontam-grams" in script
    assert "--max-line-repeat-fraction 0.18" in script
    assert "Unconditional-Algorithmic" not in script


def test_job_is_candidate_only_and_cannot_train():
    script = JOB.read_text()
    assert "MAX_TOKENS=${MAX_TOKENS:-130000000}" in script
    assert "--require-external-inputs" in script
    assert "not training-admitted" in script
    assert "train.py" not in script
