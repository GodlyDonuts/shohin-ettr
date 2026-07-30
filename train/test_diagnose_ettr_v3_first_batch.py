from pathlib import Path


SOURCE = Path(__file__).with_name(
    "diagnose_ettr_v3_first_batch.py"
).read_text()


def test_diagnostic_uses_the_exact_release_and_protected_checkpoint() -> None:
    assert "ETTRV3StreamingRelease(" in SOURCE
    assert "load_protected_base_model(" in SOURCE
    assert "stream.manifest.protected_checkpoint_sha256" in SOURCE
    assert "ETTRDiskPacketSufficiencyIndex(" in SOURCE


def test_diagnostic_compares_hard_and_soft_transaction_paths() -> None:
    assert 'choices=("hard", "soft")' in SOURCE
    assert 'hard_transactions=args.transaction_mode == "hard"' in SOURCE
    assert "--nll-gradient-cap" in SOURCE
    assert "nll_gradient_cap=args.nll_gradient_cap" in SOURCE
    assert '"nonfinite_elements"' in SOURCE
    assert '"finite_gradient_norm"' in SOURCE


def test_diagnostic_is_bounded_to_one_update_and_no_replace_output() -> None:
    assert "receipt = step.update((batch,))" in SOURCE
    assert "O_EXCL" in SOURCE
    assert '"world_size": 1' in SOURCE
    assert "save_ettr_checkpoint" not in SOURCE
