"""Static checks for the matched 141B-A39B screen fan-out."""

from pathlib import Path

SCRIPT = Path(__file__).with_name("mixtral_8x22b_evaluate.sbatch")


def test_mixtral_evaluation_is_two_h100_sharded_and_nonrequeueing() -> None:
    source = SCRIPT.read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:2" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --time=04:00:00" in source
    assert '[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]]' in source
    assert "hf_mixtral_8x22b_evaluate.py" in source
    assert "--mechanics-report" in source
    assert "--shard-index" in source
    assert "q36_init_local_tmp" in source
    assert "trap q36_cleanup_local_tmp EXIT" in source
    assert "EXPECTED_MODEL_MANIFEST_SHA256" in source
    worker = SCRIPT.parents[1].joinpath("hf_mixtral_8x22b_evaluate.py").read_text()
    assert 'bnb_4bit_quant_type="nf4"' in worker
    assert "bnb_4bit_use_double_quant=True" in worker
    assert "trust_remote_code=False" in worker
    assert "set(device_map.values()) != {0, 1}" in worker
    assert "MODEL_MANIFEST_SHA256" not in worker


def test_mixtral_evaluation_keeps_controls_and_revision_separate() -> None:
    source = SCRIPT.read_text()
    assert (
        '[[ "$ARM" == unchanged || "$ARM" == self_refinement || "$ARM" == revision ]]'
        in source
    )
    assert '[[ "$DRAFT_CANDIDATES" == none ]]' in source
    assert "revision checkpoint supplied to control arm" in source
    assert 'chmod a-w "$candidates" "$report" "$output_dir"' in source
