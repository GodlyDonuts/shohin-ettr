from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "pipeline/jobs/run_one_h100_public_benchmark_host_queue.sh"
SCORE_SCRIPT = ROOT / "pipeline/jobs/score_one_h100_public_benchmark_host_queue.sh"
SCORER = ROOT / "pipeline/jobs/score_one_h100_public_benchmark_queue.sh"


def test_host_queue_is_single_claimed_and_host_parameterized() -> None:
    source = SCRIPT.read_text()
    assert 'CLAIM="$ARTIFACT_ROOT/generation_controller.json"' in source
    assert "os.O_EXCL" in source
    assert '--host "$HOST"' in source
    assert '--model-loader "$MODEL_LOADER"' in source
    assert 'manifest="$BOARD_ROOT/manifests/$benchmark.json"' in source
    assert '--jobid="$ALLOCATION_JOB_ID"' in source
    assert "git -c core.preloadIndex=false" in source
    assert "host-queue runtime status check failed" in source
    for field in (
        '"allocation_job_id"',
        '"artifact_root"',
        '"board_root"',
        '"host"',
        '"model_revision"',
        '"draft_checkpoint_sha256"',
        '"revision_checkpoint_sha256"',
        '"benchmark_order"',
    ):
        assert field in source


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


def test_host_score_queue_is_single_claimed_and_waits_for_allocation() -> None:
    source = SCORE_SCRIPT.read_text()
    assert 'CLAIM="$ARTIFACT_ROOT/official_score_controller.json"' in source
    assert "os.O_EXCL" in source
    assert 'squeue -h -j "$ALLOCATION_JOB_ID" -t RUNNING' in source
    assert 'exec env BOARD_ROOT="$BOARD_ROOT" "$SCORER"' in source
    assert "git -c core.preloadIndex=false" in source
    assert "score-controller runtime status check failed" in source
    for field in (
        '"allocation_job_id"',
        '"artifact_root"',
        '"board_root"',
        '"source_commit"',
        '"duplicate_scoring"',
    ):
        assert field in source


def test_official_scorer_reads_shared_board_and_writes_host_artifacts() -> None:
    source = SCORER.read_text()
    assert 'BOARD_ROOT=${BOARD_ROOT:-"$ARTIFACT_ROOT"}' in source
    assert '--generation-root "$ARTIFACT_ROOT/full_generation/${benchmark}"' in source
    assert '--manifest "$BOARD_ROOT/manifests/${benchmark}.json"' in source
    assert '--ro-bind "$BOARD_ROOT/site_data_core" /assessors' in source
    assert '--ro-bind "$BOARD_ROOT/site_sources/livebench-src" /scorer' in source
    assert '--score-root "$SCORE_ROOT"' in source


def test_official_scorer_projects_only_the_pinned_base_python_root() -> None:
    source = SCORER.read_text()
    assert (
        'BASE_PYTHON_ROOT=$(dirname "$(dirname "$(realpath '
        '"$BASE_ENV_ROOT/bin/python3.13")")")'
    ) in source
    assert '--ro-bind "$BASE_PYTHON_ROOT" "$BASE_PYTHON_ROOT"' in source
    assert source.count('"${BWRAP_BASE_PYTHON_PROJECTION[@]}"') == 3
    assert '--ro-bind "$BOARD_ROOT"' not in source
