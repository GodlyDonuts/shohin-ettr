from pathlib import Path


SCRIPT = (
    Path(__file__).parent
    / "jobs"
    / "run_reserved_ettr_v3_pilot.sh"
).read_text()


def test_reserved_ettr_pilot_uses_all_selected_h100s() -> None:
    assert '--jobid="$ALLOCATION_JOB_ID"' in SCRIPT
    assert '--nnodes="$NODES"' in SCRIPT
    assert '--nproc_per_node="$GPUS_PER_NODE"' in SCRIPT
    assert '--node_rank="$SLURM_NODEID"' in SCRIPT
    assert '--gpus-per-node="$GPUS_PER_NODE"' in SCRIPT
    assert "NCCL_IB_DISABLE=0" in SCRIPT


def test_reserved_ettr_pilot_binds_release_and_exact_resume() -> None:
    for argument in (
        "--release-root",
        "--release-sha256",
        "--protected-checkpoint",
        "--source-commit",
        "--resume-checkpoint",
        "--resume-sha256",
        "--checkpoint-every",
    ):
        assert argument in SCRIPT
    assert "SOURCE_COMMIT" in SCRIPT
    assert "SHA256SUMS" in SCRIPT


def test_reserved_ettr_pilot_supports_stable_soft_eager_training() -> None:
    assert "HARD_TRANSACTIONS=${HARD_TRANSACTIONS:-1}" in SCRIPT
    assert '"$COMPILE_MODE" != eager' in SCRIPT
    assert "compile_args=()" in SCRIPT
    assert "transaction_args=()" in SCRIPT
    assert "transaction_args+=(--soft-transactions)" in SCRIPT
    assert '"${compile_args[@]}"' in SCRIPT
    assert '"${transaction_args[@]}"' in SCRIPT
    assert "NLL_GRADIENT_CAP=${NLL_GRADIENT_CAP:-}" in SCRIPT
    assert 'transaction_args+=(--nll-gradient-cap "$NLL_GRADIENT_CAP")' in SCRIPT
    assert "QUERY_BINDING_WEIGHT=${QUERY_BINDING_WEIGHT:-1}" in SCRIPT
    assert '--query-binding-weight "$QUERY_BINDING_WEIGHT"' in SCRIPT


def test_reserved_ettr_pilot_runs_paired_development_gate() -> None:
    assert "eval_ettr_v3.py" in SCRIPT
    assert SCRIPT.count('--gpus-per-node="$GPUS_PER_NODE"') == 2
    assert "--gpus-per-node=1" not in SCRIPT
    assert "--checkpoint-sha256" in SCRIPT
    assert "--run-contract-sha256" in SCRIPT
    assert "--max-batches" in SCRIPT
    assert "development-evaluation.json" in SCRIPT
