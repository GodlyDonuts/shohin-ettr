from pathlib import Path


ROOT = Path(__file__).resolve().parent
MEASURE = (ROOT / "jobs" / "dispatch_parallel_terminal_measurement.sbatch").read_text()
SUCCESSOR = (ROOT / "jobs" / "dispatch_operation_effect_successor.sbatch").read_text()


def test_measurement_chain_is_serial_and_receipt_bound() -> None:
    assert 'sha256sum "$TERMINAL_RUN_DIR/SHA256SUMS"' in MEASURE
    assert '--dependency="afterok:$eval_job"' in MEASURE
    assert '--dependency="afterok:$route_job"' in MEASURE
    assert "SUCCESSOR_SCRIPT_SHA256" in MEASURE
    assert "PLAN_OUTPUT=$PLAN_OUTPUT" in MEASURE


def test_successor_dispatch_uses_finite_planner_and_exact_warm_start() -> None:
    assert "plan_operation_effect_successor.py" in SUCCESSOR
    assert "TERMINAL_WARM_START_DIR=$warm_dir" in SUCCESSOR
    assert "TERMINAL_WARM_START_SHA256SUMS_SHA256=$warm_sha" in SUCCESSOR
    assert 'if [[ "$action" == stop ]]' in SUCCESSOR
    assert '--dependency="afterok:$fit_job"' in SUCCESSOR
    assert "UPDATES=$updates" in SUCCESSOR
    assert "CAUSAL_DELTA_WEIGHT=1" in SUCCESSOR


def test_successor_dispatch_forbids_parallel_scientific_branches() -> None:
    assert "for route" not in SUCCESSOR
    assert "srun" not in SUCCESSOR
    assert SUCCESSOR.count("fit_job=$(sbatch") == 1
