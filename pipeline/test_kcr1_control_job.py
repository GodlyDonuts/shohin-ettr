from pathlib import Path


def test_kcr1_control_builder_is_cpu_only_and_hash_bound() -> None:
    script = Path("pipeline/jobs/build_kcr1_control_data.sbatch").read_text()
    assert "--gres" not in script
    assert "RUNTIME_MANIFEST_SHA256" in script
    assert "SOURCE_REPORT_SHA256" in script
    assert "MODEL_MANIFEST_SHA256" in script
    assert "--max-sequence-length 4096" in script
    assert "test ! -e \"$OUTPUT\"" in script
    assert "chmod -R a-w \"$OUTPUT\"" in script
