from pathlib import Path


SCRIPT = (
    Path(__file__).parent / "jobs" / "run_promoted_ettr_coupling.sh"
).read_text(encoding="ascii")


def test_promoted_coupling_requires_source_deleted_causal_gate() -> None:
    assert 'evaluation["gates"]["strict_learning_signal"]' in SCRIPT
    assert "source-deleted causal promotion gate is false" in SCRIPT
    assert "evaluated component identity differs" in SCRIPT
    assert 'hashlib.sha256(path.read_bytes()).hexdigest()' in SCRIPT


def test_promoted_coupling_preserves_winner_hyperparameters() -> None:
    required = (
        'learning_rates["compiler"]',
        'learning_rates["reactor"]',
        'learning_rates["reader"]',
        'loss_weights["compiler_delta"]',
        'coupling["exact_anchor_steps_per_update"]',
        'coupling["credit_horizon"]',
    )
    for field in required:
        assert field in SCRIPT


def test_promoted_coupling_continues_after_prior_data_cursor() -> None:
    assert "admitted[14] + admitted[15] * admitted[16]" in SCRIPT
    assert "prior_end + world_size - 1" in SCRIPT
    assert "run_federated_ettr_progressive_coupling.sh" in SCRIPT
