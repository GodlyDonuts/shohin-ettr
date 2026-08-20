"""Checks for the score-free Nemotron FP8 extension admission probe."""

import hashlib
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).with_name("nemotron_super_fp8_extension_probe.sbatch")
COMMON = Path(__file__).with_name("q36_mtr_common.sh")


def test_probe_is_one_h100_score_free_and_nonrequeueing() -> None:
    source = SCRIPT.read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
    assert "#SBATCH --time=00:10:00" in source
    assert "#SBATCH --no-requeue" in source
    assert "q36_export_nemotron_cuda_toolchain" in source
    assert "get_cuda_ext_fp8(raise_if_failed=True)" in source
    assert "extension.fake_e4m3fy(inputs, amax)" in source
    assert '"scientific_rows_read": 0' in source
    assert '"assessor_accessed": False' in source
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL" in source


def test_probe_binds_exact_toolchain_and_extension_receipts() -> None:
    source = SCRIPT.read_text()
    for field in (
        '"cuda_home"',
        '"nvcc_sha256"',
        '"cc_sha256"',
        '"cxx_sha256"',
        '"torch_cuda_arch_list"',
        '"extension_sha256"',
        '"functional_values_sha256"',
    ):
        assert field in source


def test_common_helper_pins_newton_cuda_and_compiler_bytes() -> None:
    source = COMMON.read_text()
    for fragment in (
        "Q36_NEMOTRON_CUDA_HOME=/apps/cuda/cuda-12.4.0",
        "Q36_NEMOTRON_GCC_ROOT=/apps/gcc/gcc-12.2.0",
        "e701519f13153518f0143cc0c18c66f0226eabf73ddd6a7eca0d36b26ebc976b",
        "b617db0d6e6fade76990baa29f1372255575d3178ee2e8f60ba19980db37100f",
        "6264680f3e8ee209ed3b2c22c4040282e9b63fb0d7ec17df71e81765e53db34d",
        "696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67",
        "Q36_NEMOTRON_NINJA_VERSION=1.13.0.git.kitware.jobserver-pipe-1",
        '"$OVERLAY_ROOT/bin/ninja" "$Q36_NEMOTRON_NINJA_SHA256"',
        '"$OVERLAY_ROOT/bin/ninja" --version',
        'export PATH="$OVERLAY_ROOT/bin:$CUDA_HOME/bin:',
        "export TORCH_CUDA_ARCH_LIST=9.0",
        'export TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"',
        'mkdir -m 700 "$TORCH_EXTENSIONS_DIR"',
    ):
        assert fragment in source


def test_overlay_verifier_accepts_authenticated_authoring_order(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "z.txt").write_bytes(b"z\n")
    (overlay / "a.txt").write_bytes(b"a\n")
    manifest = overlay / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256((overlay / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ("z.txt", "a.txt")
        ),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    command = 'PYTHON="$1"; source "$2"; ' 'q36_verify_overlay "$3" "$4"'
    subprocess.run(
        [
            "bash",
            "-c",
            command,
            "q36-overlay-test",
            sys.executable,
            str(COMMON.resolve()),
            str(overlay),
            manifest_sha256,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_overlay_verifier_rejects_member_tamper(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    member = overlay / "member.txt"
    member.write_bytes(b"qualified\n")
    manifest = overlay / "SHA256SUMS"
    manifest.write_text(
        f"{hashlib.sha256(member.read_bytes()).hexdigest()}  member.txt\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    member.write_bytes(b"tampered\n")
    command = 'PYTHON="$1"; source "$2"; ' 'q36_verify_overlay "$3" "$4"'
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "q36-overlay-test",
            sys.executable,
            str(COMMON.resolve()),
            str(overlay),
            manifest_sha256,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "overlay member hash differs" in result.stderr
