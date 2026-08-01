from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).parent / "jobs" / "submit_algebraic_state_factorial.sh"
RECOVERY_SCRIPT = (
    Path(__file__).parent / "jobs" / "submit_algebraic_state_basis_recovery.sh"
)
SOFT_COMMIT = "1" * 40
BASIS_COMMIT = "2" * 40
RELEASE_SHA = "3" * 64


def _runtime(root: Path, name: str, commit: str) -> tuple[Path, str]:
    runtime = root / name
    job_dir = runtime / "train" / "jobs"
    job_dir.mkdir(parents=True)
    (runtime / "SOURCE_COMMIT").write_text(f"{commit}\n", encoding="ascii")
    manifest = b""
    (runtime / "SHA256SUMS").write_bytes(manifest)
    (job_dir / "algebraic_state_semantic_pilot.sbatch").write_text(
        "#!/bin/bash\n",
        encoding="ascii",
    )
    return runtime, hashlib.sha256(manifest).hexdigest()


def _environment(tmp_path: Path, *, fail_at: int | None = None) -> dict[str, str]:
    soft, soft_sha = _runtime(tmp_path, "soft", SOFT_COMMIT)
    basis, basis_sha = _runtime(tmp_path, "basis", BASIS_COMMIT)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "counter"
    calls = tmp_path / "sbatch-calls"
    cancels = tmp_path / "scancel-calls"
    fail = "" if fail_at is None else str(fail_at)
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'count=$(cat "$FAKE_COUNTER" 2>/dev/null || printf 0)\n'
        "count=$((count + 1))\n"
        'printf "%s\\n" "$count" > "$FAKE_COUNTER"\n'
        'printf "%s\\t%s\\n" "$count" "$*" >> "$FAKE_CALLS"\n'
        'if [[ -n "$FAKE_FAIL_AT" && "$count" = "$FAKE_FAIL_AT" ]]; then\n'
        "  exit 1\n"
        "fi\n"
        'printf "%s\\n" "$((720000 + count))"\n',
        encoding="ascii",
    )
    sbatch.chmod(0o700)
    scancel = fake_bin / "scancel"
    scancel.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$1" >> "$FAKE_CANCELS"\n',
        encoding="ascii",
    )
    scancel.chmod(0o700)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    return os.environ | {
        "BASIS_CODE_ROOT": str(basis),
        "BASIS_RUNTIME_SHA256": basis_sha,
        "BASIS_SOURCE_COMMIT": BASIS_COMMIT,
        "COMPILER_RUN_DIR": str(input_root / "compiler"),
        "ETTR_DATA_ROOT": str(input_root / "data"),
        "FAKE_CALLS": str(calls),
        "FAKE_CANCELS": str(cancels),
        "FAKE_COUNTER": str(counter),
        "FAKE_FAIL_AT": fail,
        "JOINT_RUN_DIR": str(input_root / "joint"),
        "OUTPUT_ROOT": str(output_root),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PROTECTED_CHECKPOINT": str(input_root / "checkpoint.pt"),
        "PYTHON_ROOT": str(input_root / "python"),
        "RELEASE_ROOT": str(input_root / "release"),
        "RELEASE_SHA256": RELEASE_SHA,
        "RUNTIME_TAG": "bb65958",
        "SOFT_CODE_ROOT": str(soft),
        "SOFT_RUNTIME_SHA256": soft_sha,
        "SOFT_SOURCE_COMMIT": SOFT_COMMIT,
        "SUBMISSION_RECEIPT": str(tmp_path / "submission.tsv"),
        "TOKENIZER": str(input_root / "tokenizer.json"),
    }


def test_dispatcher_submits_canaries_and_eight_dependency_held_arms(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    result = subprocess.run(
        ("bash", str(SCRIPT)),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = Path(environment["SUBMISSION_RECEIPT"])
    assert receipt.read_text(encoding="ascii") == result.stdout
    rows = receipt.read_text(encoding="ascii").splitlines()
    assert len(rows) == 14
    assert rows[4].split("\t")[:2] == ["soft-canary", "720001"]
    assert rows[5].split("\t")[:2] == ["basis-canary", "720002"]
    calls = Path(environment["FAKE_CALLS"]).read_text(encoding="ascii").splitlines()
    assert len(calls) == 10
    assert sum("--dependency=afterok:720001" in call for call in calls) == 4
    assert sum("--dependency=afterok:720002" in call for call in calls) == 4
    assert not Path(environment["FAKE_CANCELS"]).exists()


def test_dispatcher_cancels_partial_matrix_on_submission_failure(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, fail_at=4)
    result = subprocess.run(
        ("bash", str(SCRIPT)),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not Path(environment["SUBMISSION_RECEIPT"]).exists()
    canceled = Path(environment["FAKE_CANCELS"]).read_text(
        encoding="ascii"
    ).splitlines()
    assert canceled == ["720001", "720002", "720003"]


def test_basis_recovery_submits_canary_and_four_held_arms(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    result = subprocess.run(
        ("bash", str(RECOVERY_SCRIPT)),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = Path(environment["SUBMISSION_RECEIPT"])
    assert receipt.read_text(encoding="ascii") == result.stdout
    rows = receipt.read_text(encoding="ascii").splitlines()
    assert len(rows) == 13
    assert rows[8].split("\t")[:2] == ["basis1r2-canary", "720001"]
    calls = Path(environment["FAKE_CALLS"]).read_text(encoding="ascii").splitlines()
    assert len(calls) == 5
    assert sum("--dependency=afterok:720001" in call for call in calls) == 4
    assert not Path(environment["FAKE_CANCELS"]).exists()


def test_basis_recovery_cancels_partial_matrix_on_submission_failure(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, fail_at=3)
    result = subprocess.run(
        ("bash", str(RECOVERY_SCRIPT)),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not Path(environment["SUBMISSION_RECEIPT"]).exists()
    canceled = Path(environment["FAKE_CANCELS"]).read_text(
        encoding="ascii"
    ).splitlines()
    assert canceled == ["720001", "720002"]


def test_basis_recovery_admits_bounded_quotient_objective(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path) | {
        "OBJECTIVE_TAG": "qbrier",
        "SEMANTIC_ANSWER_WEIGHT": "0.0",
        "SEMANTIC_BASIS_SCORING": "brier",
    }
    result = subprocess.run(
        ("bash", str(RECOVERY_SCRIPT)),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = result.stdout.splitlines()
    assert rows[2] == "objective_tag\tqbrier"
    assert rows[3] == "semantic_answer_weight\t0.0"
    assert rows[4] == "semantic_basis_scoring\tbrier"
    assert rows[8].split("\t")[:2] == ["qbrier-canary", "720001"]
    calls = Path(environment["FAKE_CALLS"]).read_text(encoding="ascii").splitlines()
    assert all("SEMANTIC_ANSWER_WEIGHT=0.0" in call for call in calls)
    assert all("SEMANTIC_BASIS_SCORING=brier" in call for call in calls)
