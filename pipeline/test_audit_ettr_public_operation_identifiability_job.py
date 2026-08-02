from pathlib import Path


def test_public_operation_identifiability_job_is_hash_bound_and_no_replace():
    payload = Path(
        "pipeline/jobs/audit_ettr_public_operation_identifiability_stokes.sbatch"
    ).read_text(encoding="ascii")
    for required in (
        "set -euo pipefail",
        "SOURCE_COMMIT",
        "RUNTIME_SHA256SUMS_SHA256",
        "sha256sum -c SHA256SUMS",
        '[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]',
        "audit_ettr_public_operation_identifiability.py",
    ):
        assert required in payload
