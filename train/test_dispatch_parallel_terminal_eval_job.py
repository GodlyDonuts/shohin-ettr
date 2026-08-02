from pathlib import Path


ROOT = Path(__file__).resolve().parent
DISPATCH = (ROOT / "jobs" / "dispatch_parallel_terminal_eval.sbatch").read_text()
ROUTE = (ROOT / "jobs" / "route_parallel_terminal_eval.sbatch").read_text()


def test_dispatch_binds_terminal_receipt_and_independent_seed() -> None:
    assert "TERMINAL_RUN_SHA256SUMS_SHA256=$terminal_sha" in DISPATCH
    assert "DATA_SEED=$DATA_SEED" in DISPATCH
    assert "--dependency=\"afterok:$eval_job\"" in DISPATCH
    assert "--kill-on-invalid-dep=yes" in DISPATCH
    assert "SELF_SHA256" in DISPATCH
    assert "ROUTE_SCRIPT_SHA256" in DISPATCH
    assert 'sha256sum "$ROUTE_SCRIPT"' in DISPATCH


def test_router_replays_evaluation_and_terminal_custody() -> None:
    assert 'cd "$TERMINAL_RUN_DIR"' in ROUTE
    assert 'cd "$EVAL_OUTPUT"' in ROUTE
    assert "sha256sum -c SHA256SUMS" in ROUTE
    assert "route_operation_effect_set_result.py" in ROUTE
    assert "--terminal-contract" in ROUTE


def test_dispatch_and_router_do_not_allocate_a_scientific_successor() -> None:
    for source in (DISPATCH, ROUTE):
        assert "parallel_terminal_state_pilot.sbatch" not in source
        assert "operation-family-island" not in source
