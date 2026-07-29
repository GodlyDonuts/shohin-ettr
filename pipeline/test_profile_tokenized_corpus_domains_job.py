from pathlib import Path


JOB = (
    Path(__file__).parent
    / "jobs"
    / "profile_tokenized_corpus_domains.sbatch"
)


def test_domain_profile_job_is_hash_bound_and_text_free() -> None:
    text = JOB.read_text()
    for required in (
        "set -euo pipefail",
        "CODE_ROOT=${CODE_ROOT:?",
        "SOURCE_DIR=${SOURCE_DIR:?",
        "OUTPUT=${OUTPUT:?",
        "PROFILE_SHA256",
        "SOURCE_SELECTION_CODE_SHA256",
        "RUNTIME_SHA256SUMS_SHA256",
        'sha256sum -c "$RUNTIME_SHA256SUMS"',
        "--source-selection-code",
        "text-free profile complete",
        "no training admission",
    ):
        assert required in text
    for forbidden in ("curl ", "wget ", "tokenize_shards.py \\", "train.py"):
        assert forbidden not in text
