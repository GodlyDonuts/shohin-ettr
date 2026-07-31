from pathlib import Path


SCRIPT = (
    Path(__file__).parent / "jobs" / "run_promoted_ettr_coupling.sh"
).read_text(encoding="ascii")


def test_promoted_coupling_requires_source_deleted_causal_gate() -> None:
    assert 'evaluation["gates"]["strict_learning_signal"]' in SCRIPT
    assert "source-deleted causal promotion gate is false" in SCRIPT
    assert 'replicate_evaluation["gates"]["strict_learning_signal"]' in SCRIPT
    assert "replicate source-deleted causal promotion gate is false" in SCRIPT
    assert "evaluated component identity differs" in SCRIPT
    assert "replicate evaluated component identity differs" in SCRIPT
    assert 'hashlib.sha256(path.read_bytes()).hexdigest()' in SCRIPT


def test_promoted_coupling_requires_a_matched_opposite_seed() -> None:
    assert "replicate_seed = 2 if expected_seed == 1 else 1" in SCRIPT
    assert "replicate seed identity differs" in SCRIPT
    assert "replicated training recipe differs" in SCRIPT
    assert "replicated optimization recipe differs" in SCRIPT
    assert "replicated coupling recipe differs" in SCRIPT
    assert "replicated evaluated source identity differs" in SCRIPT


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
    assert '"reader_causal_balance_mode"' in SCRIPT
    assert '"reader_is_frozen_semantic_anchor"' in SCRIPT
    assert 'export READER_CAUSAL_BALANCE_MODE="${admitted[17]}"' in SCRIPT
    assert 'export FREEZE_READER="${admitted[18]}"' in SCRIPT


def test_promoted_coupling_continues_after_prior_data_cursor() -> None:
    assert "admitted[14] + admitted[15] * admitted[16]" in SCRIPT
    assert "prior_end + world_size - 1" in SCRIPT
    assert "run_federated_ettr_progressive_coupling.sh" in SCRIPT
