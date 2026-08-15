from __future__ import annotations

from pathlib import Path

import pytest

from package_q36_mtr_runtime import Q36MTRRuntimeError, load_allowlist


def test_production_q36_allowlist_is_sorted_closed_and_exactly_one_dispatcher() -> None:
    root = Path(__file__).resolve().parents[1]
    entries = load_allowlist(root / "pipeline/q36_mtr_runtime_allowlist.txt")
    assert entries == sorted(entries)
    assert all((root / entry).is_file() for entry in entries)
    assert [entry for entry in entries if "dispatch" in entry.casefold()] == [
        "pipeline/dispatch_q36_mtr.py"
    ]
    assert not any("q35" in entry.casefold() for entry in entries)
    assert "train/hf_q36_mtr_train_temporal_gate.py" in entries
    assert "pipeline/analyze_q36_mtr_token_gates.py" in entries
    assert "train/temporal_residual_gate.py" in entries
    assert "train/jobs/q36_mtr_train_temporal_gate.sbatch" in entries
    assert "train/hf_q36_mtr_evaluate_temporal_gate.py" in entries
    assert "train/jobs/q36_mtr_evaluate_temporal_gate.sbatch" in entries
    assert "pipeline/score_q36_mtr_temporal_gate.py" in entries
    assert "pipeline/jobs/q36_mtr_score_temporal_gate.sbatch" in entries
    assert "train/hf_nemotron_super_mechanics.py" in entries
    assert "train/hf_nemotron_super_evaluate.py" in entries
    assert "pipeline/score_nemotron_super_screen.py" in entries
    assert "train/hf_nemotron_super_train_revision.py" in entries
    assert "train/jobs/nemotron_super_mechanics.sbatch" in entries
    assert "train/jobs/nemotron_super_evaluate.sbatch" in entries
    assert "pipeline/jobs/nemotron_super_score.sbatch" in entries
    assert "train/jobs/nemotron_super_train_revision.sbatch" in entries
    assert "train/nemotron_super_post_mixer_revision.py" in entries
    assert "train/q36_upward_moe_host.py" in entries
    assert "pipeline/build_nemotron_scale_transfer_basis.py" in entries
    assert "pipeline/jobs/build_nemotron_scale_transfer_basis.sbatch" in entries


@pytest.mark.parametrize(
    "entry",
    (
        "../train/hf_q36_mtr_train_role.py",
        "/tmp/q36.py",
        "train/ndr1_retry.py",
        "train/q35_edit_selector.py",
        "train/jobs/dispatch_pcf1.sh",
    ),
)
def test_q36_allowlist_rejects_escape_retry_and_dispatch(
    tmp_path: Path, entry: str
) -> None:
    path = tmp_path / "allowlist.txt"
    path.write_text(entry + "\n", encoding="utf-8")
    with pytest.raises(Q36MTRRuntimeError):
        load_allowlist(path)


def test_calibration_wrappers_expose_complete_runtime_import_closure() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "q36_mtr_calibration_correctness.sbatch",
        "q36_mtr_calibration_stack.sbatch",
        "q36_mtr_calibration_direct_commit.sbatch",
    ):
        wrapper = (root / "pipeline" / "jobs" / name).read_text(encoding="utf-8")
        assert "PYTHON RUNTIME" in wrapper
        assert 'PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"' in wrapper


def test_preview_scorer_is_packaged_with_preview_job() -> None:
    root = Path(__file__).resolve().parents[1]
    entries = load_allowlist(root / "pipeline/q36_mtr_runtime_allowlist.txt")
    assert "pipeline/jobs/q36_mtr_score_draft_preview.sbatch" in entries
    assert "pipeline/score_q36_mtr_draft_preview.py" in entries
