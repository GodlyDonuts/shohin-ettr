from pathlib import Path


JOB = (
    Path(__file__).resolve().parent
    / "jobs"
    / "run_reserved_multinode_ddp_canary.sh"
).read_text(encoding="ascii")


def test_canary_requires_running_named_reservation_and_exact_source() -> None:
    assert 'state" != "RUNNING"' in JOB
    assert 'name" != shohin-*h100-*' in JOB
    assert "git rev-parse HEAD" in JOB
    assert "git status --porcelain --untracked-files=all" in JOB


def test_canary_uses_multinode_torchrun_and_infiniband_gate() -> None:
    assert "--nnodes=" in JOB
    assert "--nproc_per_node=" in JOB
    assert "--node_rank=" in JOB
    assert "--rdzv_endpoint=" in JOB
    assert "/sys/class/infiniband" in JOB
    assert '--gpus-per-node="$GPUS_PER_NODE"' in JOB


def test_canary_is_bounded_and_writes_only_isolated_output() -> None:
    assert "--steps \"$STEPS\"" in JOB
    assert "--ckpt-every 0" in JOB
    assert '[[ -e "$OUTDIR" || -L "$OUTDIR" ]]' in JOB
    assert '--out "$OUTDIR"' in JOB
    assert "flagship_out" not in JOB
    assert "--resume" not in JOB


def test_canary_uses_only_historical_admitted_shards() -> None:
    for name in ("finemath4", "openwebmath", "code_python", "finemath3"):
        assert f'"$SHARD_ROOT/{name}"' in JOB
    assert "dclm_baseline_25b" not in JOB
    assert "openmath_pt" not in JOB
