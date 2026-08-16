from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = (ROOT / "train/hf_mixtral_8x22b_multinode_tp_mechanics.py").read_text()
WORKER = (ROOT / "train/jobs/mixtral_8x22b_multinode_tp_mechanics.sbatch").read_text()
SUBMIT = (
    ROOT / "train/jobs/submit_mixtral_8x22b_multinode_tp_mechanics.sh"
).read_text()


def test_multinode_mechanics_uses_native_tensor_parallelism() -> None:
    assert "DistributedConfig(tp_size=world)" in PYTHON
    assert 'dist.init_process_group(backend="nccl")' in PYTHON
    assert "EXPECTED_WORLD_SIZE = 4" in PYTHON
    assert 'getattr(backbone, "hf_device_map", None) is not None' in PYTHON
    assert '"parallelism": "native-transformers-tensor-parallel"' in PYTHON


def test_multinode_mechanics_preserves_matched_scientific_surface() -> None:
    assert '"weight_dtype": "bfloat16"' in PYTHON
    assert '"quantization": "none"' in PYTHON
    assert "BitsAndBytesConfig" not in PYTHON
    assert "MixtralRevisionModel(backbone)" in PYTHON
    assert "_synchronize_gradients(model, world)" in PYTHON
    assert '"score_rows_read": 0' in PYTHON
    assert '"benchmark_rows_read": 0' in PYTHON
    assert "native_router_unchanged" in PYTHON


def test_worker_is_one_gpu_per_independent_slurm_request() -> None:
    assert "#SBATCH --nodes=1" in WORKER
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in WORKER
    assert "#SBATCH --no-requeue" in WORKER
    assert '"$PYTHON" -P -s -B -m torch.distributed.run' in WORKER
    assert "--nnodes=4" in WORKER
    assert "--nproc_per_node=1" in WORKER
    assert '--node_rank="$WORLD_RANK"' in WORKER
    assert "NCCL_IB_DISABLE=0" in WORKER


def test_submitter_requests_each_gpu_separately_and_avoids_orphans() -> None:
    assert "for rank in 0 1 2 3" in SUBMIT
    assert 'sbatch --parsable --export="$exports,WORLD_RANK=$rank"' in SUBMIT
    assert "cleanup_partial_submission" in SUBMIT
    assert 'scancel "${jobs[@]}"' in SUBMIT
    assert '[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]' in SUBMIT


def test_runtime_allowlist_contains_multinode_path() -> None:
    allowlist = (ROOT / "pipeline/upward_moe_runtime_allowlist.txt").read_text()
    for member in (
        "train/hf_mixtral_8x22b_multinode_tp_mechanics.py",
        "train/jobs/mixtral_8x22b_multinode_tp_mechanics.sbatch",
        "train/jobs/submit_mixtral_8x22b_multinode_tp_mechanics.sh",
    ):
        assert f"{member}\n" in allowlist
