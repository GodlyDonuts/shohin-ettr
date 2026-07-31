from pathlib import Path


JOBS = Path(__file__).parent / "jobs"
RESERVATION = (
    JOBS / "reserve_ettr_h100_dispatch.sbatch"
).read_text(encoding="ascii")
DISPATCHER = (
    JOBS / "dispatch_federated_ettr_progressive_coupling.sh"
).read_text(encoding="ascii")


def test_dispatch_reservation_pins_and_hashes_one_command() -> None:
    assert 'CONTROL_ROOT="$ROOT/control/ettr-h100-dispatch-v1"' in RESERVATION
    assert 'exec {command_fd}<"$command"' in RESERVATION
    assert 'sha256sum "/proc/self/fd/$command_fd"' in RESERVATION
    assert 'bash "/proc/self/fd/$command_fd"' in RESERVATION
    assert 'mv -n "$status_tmp" "$status"' in RESERVATION


def test_dispatcher_avoids_login_side_srun_fanout() -> None:
    assert "\nsrun " not in DISPATCHER
    assert 'emit_export WORLD_SIZE "$world_size"' in DISPATCHER
    assert 'emit_export RANK "$rank"' in DISPATCHER
    assert "NCCL_IB_DISABLE" in DISPATCHER
    assert "train_ettr_progressive_coupling.py" in DISPATCHER


def test_dispatcher_binds_training_inputs_and_fresh_outputs() -> None:
    required = (
        "SOURCE_COMMIT",
        "RELEASE_SHA256",
        "CHECKPOINT_SHA256",
        "RUN_CONTRACT_SHA256",
        "INITIAL_COMPILER_SHA256",
        "INITIAL_REACTOR_SHA256",
        "INITIAL_READER_SHA256",
        "COUNTERFACTUAL_DELTA_WEIGHT",
        "EXACT_ANCHOR_STEPS",
        "CREDIT_HORIZON",
    )
    for value in required:
        assert value in DISPATCHER
    assert "sha256sum -c SHA256SUMS" in DISPATCHER
    assert "dispatch output or launcher root already exists" in DISPATCHER
