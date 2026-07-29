from pathlib import Path


JOB = (
    Path(__file__).resolve().parent
    / "jobs"
    / "pes2o_tokenizer_batch_canary.sbatch"
).read_text(encoding="ascii")


def test_pes2o_batch_canary_uses_production_quality_contract() -> None:
    for value in (
        "common-pile/peS2o",
        "metadata.oa_license",
        "CCBY CCBYSA pd",
        "--min-chars 4000",
        "--max-line-repeat-fraction 0.12",
        "metadata.oa_url",
        "--max-tokens-per-domain 1000000000",
        "--require-external-inputs",
    ):
        assert value in JOB


def test_pes2o_batch_canary_proves_exact_artifact_parity() -> None:
    assert "run_one 1" in JOB
    assert "run_one 256" in JOB
    assert "batch canary artifact differs" in JOB
    assert "batch canary normalized manifests differ" in JOB
    assert '"artifacts_exact": True' in JOB
