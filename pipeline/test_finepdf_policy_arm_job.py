from pathlib import Path


JOB = Path(__file__).with_name("jobs") / "finepdfs_edu_policy_arm.sbatch"


def test_policy_arm_is_hash_bound_no_replace_and_quarantined():
    source = JOB.read_text()
    for required in (
        "TOKENIZE_SHA256=${TOKENIZE_SHA256:?",
        "VERIFY_SHARDS_SHA256=${VERIFY_SHARDS_SHA256:?",
        "POLICY_SHA256=${POLICY_SHA256:?",
        "BASE_CANDIDATE_PAYLOAD_SHA256=${BASE_CANDIDATE_PAYLOAD_SHA256:?",
        "--require-external-inputs",
        "--document-policy finepdf_core_v1",
        '--document-policy-allowed-tier "$ARM_TIER"',
        '[[ "$ARM_TIER" == core || "$ARM_TIER" == residual ]]',
        '[[ ! -e "$OUT"',
        "not training-admitted",
    ):
        assert required in source
    for forbidden in (
        "rm -rf",
        "--document-policy-allowed-tier reject",
        "git checkout",
        "git pull",
    ):
        assert forbidden not in source


def test_policy_arm_reuses_exact_base_source_and_eval_ledgers():
    source = JOB.read_text()
    assert '["source_files"]' in source
    assert '["decontamination"]["eval_files"]' in source
    assert '["decontamination"]["pickle_path"]' in source
    assert "HuggingFaceFW/finepdfs-edu" in source
    assert "9cfabe2127faca99b3d5c4dc6d1fcb397399ebde" in source
