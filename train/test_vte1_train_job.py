"""Static custody checks for the VTE1 GPU train wrapper."""

from pathlib import Path


def test_vte1_train_job_binds_objective_and_frozen_geometry() -> None:
    script = (
        Path(__file__).parent / "jobs" / "hf_vte1_train_reviser.sbatch"
    ).read_text(encoding="utf-8")
    assert "--loss-mode vte1_equivalence" in script
    assert '--updates "$UPDATES"' in script
    assert "--batch-size 1 --gradient-accumulation 8" in script
    assert "--max-sequence-length 4096 --learning-rate 2e-5" in script
    assert "--seed 2026081021" in script
    assert "--data-seed 2026081020" in script
    assert '--warm-start-checkpoint "$WARM_START"' in script
    assert "MASK_INTERNAL_DRAFT" in script
    assert 'test ! -e "$OUTPUT"' in script
