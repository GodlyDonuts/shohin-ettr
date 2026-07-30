from pathlib import Path


SCRIPT = (
    Path(__file__).parent
    / "jobs"
    / "watch_federated_ettr_v3_ladder.sh"
).read_text()


def test_ladder_waits_for_verified_direct_transfer() -> None:
    assert "shohin-ettr-il-v3-direct-transfer-receipt-v1" in SCRIPT
    assert 'receipt.get("release_file_sha256") != release_sha256' in SCRIPT
    assert 'receipt.get("source_commit") != source_commit' in SCRIPT
    assert 'release_sha256=$(verify_transfer' in SCRIPT
    assert '"$RELEASE_SOURCE_COMMIT" <<\'PY\'' in SCRIPT


def test_ladder_is_bounded_by_causal_development_gates() -> None:
    assert "for target in 100 500 2000" in SCRIPT
    assert 'get("strict_learning_signal") is not True' in SCRIPT
    assert "federated_ladder_stopped" in SCRIPT
    assert 'RESUME_CHECKPOINT="$resume_checkpoint"' in SCRIPT
    assert 'RESUME_SHA256="$resume_sha256"' in SCRIPT


def test_ladder_preserves_fixed_allocations_and_fresh_outputs() -> None:
    assert 'state=$(squeue -h -j "$job" -o "%T")' in SCRIPT
    assert '"$state" != RUNNING' in SCRIPT
    assert '"$name" != shohin-1h100-*' in SCRIPT
    assert "federated ladder output already exists" in SCRIPT
    assert 'ALLOCATION_JOB_IDS="$ALLOCATION_JOB_IDS"' in SCRIPT


def test_ladder_normalizes_schedule_without_training_the_base() -> None:
    assert 'TOTAL_UPDATES="$TOTAL_UPDATES"' in SCRIPT
    assert 'WARMUP_UPDATES="$WARMUP_UPDATES"' in SCRIPT
    assert "FREEZE_BASE=1" in SCRIPT
    assert "ACCUMULATION=1" in SCRIPT


def test_ladder_forwards_explicit_compile_strategy() -> None:
    assert "COMPILE_MODE=${COMPILE_MODE:-default}" in SCRIPT
    assert '"$COMPILE_MODE" != eager' in SCRIPT
    assert 'COMPILE_MODE="$COMPILE_MODE"' in SCRIPT


def test_ladder_forwards_explicit_transaction_strategy() -> None:
    assert "HARD_TRANSACTIONS=${HARD_TRANSACTIONS:-1}" in SCRIPT
    assert 'HARD_TRANSACTIONS="$HARD_TRANSACTIONS"' in SCRIPT
