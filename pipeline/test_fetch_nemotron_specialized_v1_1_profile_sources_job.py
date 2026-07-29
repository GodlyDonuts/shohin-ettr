"""Static gates for the pinned Nemotron Specialized v1.1 profile fetch."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = (
    ROOT
    / "pipeline"
    / "jobs"
    / "fetch_nemotron_specialized_v1_1_profile_sources.sbatch"
)


def test_fetch_job_pins_revision_bytes_and_hashes():
    script = JOB.read_text()
    assert "13fa979be2e7f7e62913eee0ec5e97c8fd6e24af" in script
    for size in ("7402", "73074239", "208343534"):
        assert size in script
    for digest in (
        "93594fcc5d7e0a85bb8cbc6561b13d9d4c5c9cf4d21e7f81388829e86542e1bf",
        "b8b641309dec27e836db6a2003d3045c6ed060c8d6a23e3a93100e1c8a3a450b",
        "706bab9781f6541cb9f40b8af61242e45c6e3dabc0d2f9b3171f08571da37ccb",
    ):
        assert digest in script


def test_fetch_job_is_resumable_hash_verified_and_nontraining():
    script = JOB.read_text()
    assert "--continue-at -" in script
    assert "sha256sum -c" in script
    assert "chmod 0444" in script
    assert "no training admission" in script
    assert "train.py" not in script
