from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN = (ROOT / "jobs" / "hf_sctr1_train.sbatch").read_text(encoding="utf-8")
EVAL = (ROOT / "jobs" / "hf_sctr1_evaluate.sbatch").read_text(encoding="utf-8")
DRAFTS = (ROOT / "jobs" / "submit_sctr1_base_drafts.sh").read_text(encoding="utf-8")
CONTROL = (ROOT / "jobs" / "ttr1_control_evaluate.sbatch").read_text(encoding="utf-8")


def test_temporal_role_fit_supports_only_predeclared_arms() -> None:
    assert 'TRAIN_KIND=${TRAIN_KIND:-selective_commit}' in TRAIN
    assert '"selective_commit"' in TRAIN
    assert '"always_revise"' in TRAIN
    assert '"independent_commitment"' in TRAIN
    assert "--mask-internal-draft" in TRAIN


def test_selective_evaluator_is_hash_bound_and_fail_closed() -> None:
    assert "hf_sctr1_evaluate.py" in EVAL
    assert "RUNTIME_MANIFEST_SHA256" in EVAL
    assert "ADAPTER_CHECKPOINT_SHA256" in EVAL
    assert "test ! -e \"$REPORT\"" in EVAL


def test_base_drafts_are_parallel_and_adapter_free() -> None:
    assert "jobs=17" in DRAFTS
    assert "ADAPTER_CHECKPOINT" not in DRAFTS
    assert "MODEL_MANIFEST_SHA256" in DRAFTS


def test_standard_controls_allow_base_owner_but_bind_independent_adapter() -> None:
    assert 'if [[ "$CONTROL" == "independent_commitment" ]]' in CONTROL
    assert 'ADAPTER_ARGS+=(--adapter-checkpoint "$ADAPTER_CHECKPOINT")' in CONTROL
