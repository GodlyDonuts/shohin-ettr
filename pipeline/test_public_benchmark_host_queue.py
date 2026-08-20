from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "pipeline/jobs/run_one_h100_public_benchmark_host_queue.sh"


def test_host_queue_is_single_claimed_and_host_parameterized() -> None:
    source = SCRIPT.read_text()
    assert 'CLAIM="$ARTIFACT_ROOT/generation_controller.json"' in source
    assert "os.O_EXCL" in source
    assert '--host "$HOST"' in source
    assert '--model-loader "$MODEL_LOADER"' in source
    assert 'manifest="$BOARD_ROOT/manifests/$benchmark.json"' in source
    assert '--jobid="$ALLOCATION_JOB_ID"' in source


def test_host_queue_preserves_matched_frozen_inputs() -> None:
    source = SCRIPT.read_text()
    required = (
        '--draft-checkpoint "$DRAFT_CHECKPOINT"',
        '--draft-checkpoint-sha256 "$DRAFT_CHECKPOINT_SHA256"',
        '--revision-checkpoint "$REVISION_CHECKPOINT"',
        '--revision-checkpoint-sha256 "$REVISION_CHECKPOINT_SHA256"',
        '--model-revision "$MODEL_REVISION"',
        '--model-config-sha256 "$MODEL_CONFIG_SHA256"',
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
    )
    for fragment in required:
        assert fragment in source


def test_host_queue_covers_exact_public_benchmark_order() -> None:
    source = SCRIPT.read_text()
    assert (
        'ORDER=${ORDER:-"humaneval_plus mbpp_plus ifeval musr correctbench '
        'livebench livecodebench ruler longbench_pro mmlu_pro"}'
    ) in source
    assert '"benchmarks": 10' in source
