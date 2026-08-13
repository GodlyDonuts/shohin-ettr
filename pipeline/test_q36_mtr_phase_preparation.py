from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import re
import subprocess


def test_q36_phase_preparation_is_read_only_with_respect_to_scheduler() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "pipeline/jobs/q36_mtr_prepare_phase.sh"
    source = script.read_text(encoding="utf-8")
    assert not re.search(r"(^|[;&|]\s*)(srun|sbatch|scancel)(\s|$)", source, re.M)
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_q36_phase_preparation_pins_numerical_libraries_before_python() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "pipeline/jobs/q36_mtr_prepare_phase.sh").read_text(
        encoding="utf-8"
    )
    pin = (
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
    )
    assert source.count(pin) == 1
    assert source.index(pin) < source.index(
        'source "$RUNTIME/train/jobs/q36_mtr_common.sh"'
    )
    assert source.index(pin) < source.index('"$PYTHON"')


def test_q36_phase_preparation_qualifies_in_private_local_tmp() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "pipeline/jobs/q36_mtr_prepare_phase.sh").read_text(
        encoding="utf-8"
    )
    create = "sandbox_tmp=$(mktemp -d /tmp/q36-mtr-admission.XXXXXX)"
    export = 'export SLURM_TMPDIR="$sandbox_tmp" TMPDIR="$sandbox_tmp"'
    qualify = '"$RUNTIME/train/pcf1_code_sandbox.py" qualify'
    assert source.count(create) == 1
    assert source.count(export) == 1
    assert source.count('rmdir "$sandbox_tmp"') == 1
    assert source.index(create) < source.index(export) < source.index(qualify)
    assert "rm -rf" not in source


def test_q36_live_preflight_is_the_first_cpu_entrypoint() -> None:
    from compile_q36_mtr_plan import CPU_ENTRYPOINTS

    assert CPU_ENTRYPOINTS["preflight_cpu"] == "q36_mtr_live_preflight"


def _authorization(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": "shohin-q36-mtr-phase-authorization-v1",
                "status": "authorized",
                "scientific_submit_authorized": True,
                "source_commit": "1" * 40,
                "model_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
                "gate": "one_source_disjoint_development_gate",
                "automatic_retry": False,
                "automatic_successor": False,
                "automatic_confirmation": False,
                "stop_after_gate": True,
                "run_id": "q36-run",
                "run_root": str(run_root),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return authorization, run_root


def _check_authorization(
    authorization: Path, output: Path, run_id: str = "q36-run"
) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": "python3",
            "SOURCE_COMMIT": "1" * 40,
            "PHASE_AUTHORIZATION": str(authorization),
            "PHASE_AUTHORIZATION_SHA256": hashlib.sha256(
                authorization.read_bytes()
            ).hexdigest(),
            "RUN_ID": run_id,
            "OUTPUT": str(output),
        }
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; q36_require_authorization',
            "q36-test",
            str(root / "train/jobs/q36_mtr_common.sh"),
        ],
        env=environment,
        text=True,
        capture_output=True,
    )


def test_q36_common_binds_run_identity_and_output_root(tmp_path: Path) -> None:
    authorization, run_root = _authorization(tmp_path)
    assert (
        _check_authorization(authorization, run_root / "owner/report.json").returncode
        == 0
    )
    assert _check_authorization(authorization, tmp_path / "escape.json").returncode != 0
    assert (
        _check_authorization(
            authorization, run_root / "owner/report.json", "other-run"
        ).returncode
        != 0
    )
