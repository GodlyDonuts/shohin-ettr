from pathlib import Path

SCRIPT = Path(__file__).parent / "jobs" / "build_nemotron_super_overlay_glibc228.sbatch"


def test_rebuilds_mamba_on_newton_glibc_without_science() -> None:
    source = SCRIPT.read_text()
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --gres" not in source
    assert "a14b1dff0454a3bc27d9eb31355dc01e4b2490ec" in source
    assert "MAMBA_FORCE_BUILD=TRUE" in source
    assert "CUDA_HOME=/apps/cuda/cuda-12.4.0" in source
    assert "max(versions) > (2, 28)" in source
    assert '"scientific_rows_read": 0' in source
    assert '"gpu_requested": False' in source
    assert "sha256sum -c SHA256SUMS" in source


def test_never_mutates_parent_overlay() -> None:
    source = SCRIPT.read_text()
    assert 'cp -a "$PARENT_OVERLAY/." "$partial/"' in source
    assert "rm -rf --one-file-system" in source
    assert '"$partial/mamba_ssm"' in source
    assert 'mv -- "$partial" "$OUTPUT_ROOT"' in source
