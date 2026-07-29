from pathlib import Path


JOB = (
    Path(__file__).parent
    / "jobs"
    / "materialize_domain_balanced_residual.sbatch"
)


def test_domain_balance_job_is_hash_bound_and_candidate_only() -> None:
    text = JOB.read_text()
    for required in (
        "set -euo pipefail",
        "CODE_ROOT=${CODE_ROOT:?",
        "SOURCE_DIR=${SOURCE_DIR:?",
        "POLICY=${POLICY:?",
        "OUTPUT_DIR=${OUTPUT_DIR:?",
        "SOURCE_SELECTION_CODE_SHA256",
        "SELECTION_CODE_SHA256",
        "VERIFY_SHARDS_SHA256",
        "POLICY_SHA256",
        "RUNTIME_SHA256SUMS_SHA256",
        'sha256sum -c "$RUNTIME_SHA256SUMS"',
        "--source-selection-code",
        "--policy",
        "--require-external-inputs",
        "no training admission",
    ):
        assert required in text
    for forbidden in ("curl ", "wget ", "train.py", "sbatch "):
        assert forbidden not in text
