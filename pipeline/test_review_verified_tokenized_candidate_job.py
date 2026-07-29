from pathlib import Path


JOB = (
    Path(__file__).parent
    / "jobs"
    / "review_verified_tokenized_candidate.sbatch"
)


def test_review_job_binds_exact_code_and_candidate() -> None:
    text = JOB.read_text()
    for required in (
        "set -euo pipefail",
        "CODE_ROOT=${CODE_ROOT:?",
        "BUILDER_SHA256",
        "PROFILE_SHA256",
        "SELECTION_CODE_SHA256",
        "VERIFY_SHARDS_SHA256",
        "sha256sum -c -",
        "SHARD_DIR=${SHARD_DIR:?",
        '[[ -f "$SHARD_DIR/manifest.json"',
        "--selection-code",
        "corpus remains quarantined",
    ):
        assert required in text


def test_review_text_stays_private_and_outputs_are_fresh() -> None:
    text = JOB.read_text()
    assert "PRIVATE_OUT=${PRIVATE_OUT:?" in text
    assert "RECEIPT_OUT=${RECEIPT_OUT:?" in text
    assert '[[ "$path" == /* && ! -e "$path" && ! -L "$path" ]]' in text
    assert 'chmod 0700 "$(dirname "$PRIVATE_OUT")"' in text
    assert 'stat -c %a "$PRIVATE_OUT"' in text
    assert "git " not in text
