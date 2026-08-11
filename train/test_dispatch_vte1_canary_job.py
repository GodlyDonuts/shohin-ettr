"""Static custody checks for the VTE1 source-disjoint dispatcher."""

from pathlib import Path


def test_vte1_dispatch_is_dependency_safe_and_hash_bound() -> None:
    script = (Path(__file__).parent / "jobs" / "dispatch_vte1_canary.sbatch").read_text(
        encoding="utf-8"
    )
    assert "checkpoint_0000256.pt" in script
    assert 'r.get("updates")!=256' in script
    assert 'r.get("loss_mode")!="vte1_equivalence"' in script
    assert "cc312363f880e9048622b57cb0cb609acaf92a1ab9ed0552ec8383ea20da1c33" in script
    assert "07e08abe2480782afc77e35031d23bea71a737d019f307066af2bde786dd2ebd" in script
    assert "--dependency=afterok:" in script
    assert '--exclude="$EXCLUDE"' in script
    assert '"holdout_opened":False' in script
