from pathlib import Path


def test_newton_verifier_limits_blas_threads_before_python_import() -> None:
    script = (
        Path(__file__).with_name("transfer_ettr_il_v3_to_newton.sh")
    ).read_text(encoding="ascii")
    command = (
        "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "\\\n    '$DEST_PYTHON'"
    )
    assert command in script
    assert script.index(command) < script.index(
        "from ettr_v3_streaming import ETTRV3StreamingRelease"
    )
