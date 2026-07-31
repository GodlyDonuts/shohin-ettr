from pathlib import Path


SCRIPT = (
    Path(__file__).parent
    / "jobs"
    / "run_federated_ettr_progressive_coupling.sh"
).read_text(encoding="ascii")


def test_federated_coupling_launches_one_rank_per_visible_gpu() -> None:
    assert "IFS='@' read -r job node gpus extra" in SCRIPT
    assert 'export RANK="$rank"' in SCRIPT
    assert 'export LOCAL_RANK="$local_rank"' in SCRIPT
    assert 'export WORLD_SIZE="$world_size"' in SCRIPT
    assert '--gpus-per-node="$gpus"' in SCRIPT
    assert "NCCL_IB_DISABLE=0" in SCRIPT


def test_federated_coupling_binds_all_training_inputs() -> None:
    required = (
        "SOURCE_COMMIT",
        "RELEASE_SHA256",
        "CHECKPOINT_SHA256",
        "RUN_CONTRACT_SHA256",
        "INITIAL_COMPILER_SHA256",
        "INITIAL_REACTOR_SHA256",
        "INITIAL_READER_SHA256",
        "ARCHITECTURE_SEED",
        "DATA_SEED",
        "COUPLING_SEED",
        "COUNTERFACTUAL_DELTA_WEIGHT",
        "EXACT_ANCHOR_STEPS",
        "CREDIT_HORIZON",
        "READER_CAUSAL_BALANCE_MODE",
        "FREEZE_READER",
    )
    for name in required:
        assert f"${{{name}" in SCRIPT or f'"${name}"' in SCRIPT
    assert "sha256sum -c SHA256SUMS" in SCRIPT
    assert "refusing existing coupling output or launcher root" in SCRIPT
    assert '--reader-causal-balance-mode "$READER_CAUSAL_BALANCE_MODE"' in SCRIPT
    assert "reader_args+=(--freeze-reader)" in SCRIPT
