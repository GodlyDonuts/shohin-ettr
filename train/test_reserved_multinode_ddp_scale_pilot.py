from pathlib import Path


JOB = (
    Path(__file__).resolve().parent
    / "jobs"
    / "run_reserved_multinode_ddp_scale_pilot.sh"
).read_text(encoding="ascii")


def test_scale_pilot_requires_clean_exact_code_and_fresh_output() -> None:
    assert '"$(tr -d \'\\r\\n\' < SOURCE_COMMIT)" != "$SOURCE_COMMIT"' in JOB
    assert "sha256sum -c SHA256SUMS" in JOB
    assert '[[ -e "$OUTDIR" || -L "$OUTDIR" ]]' in JOB
    assert "--out \"$OUTDIR\"" in JOB
    assert "flagship_out" not in JOB
    assert "--resume" not in JOB


def test_scale_pilot_selects_only_healthy_reserved_nodes() -> None:
    assert "NODELIST=${NODELIST:?" in JOB
    assert "--nodelist=\"$NODELIST\"" in JOB
    assert "comm -23" in JOB
    assert "torch.cuda.is_available()" in JOB
    assert "/sys/class/infiniband" in JOB


def test_scale_pilot_uses_multinode_ddp_and_historical_admitted_shards() -> None:
    for name in ("finemath4", "openwebmath", "code_python", "finemath3"):
        assert f'"$SHARD_ROOT/{name}"' in JOB
    assert "dclm_baseline_25b" not in JOB
    assert "fineweb_edu_score4_core_10b_r2" not in JOB
    assert "--nnodes=\"$NODES\"" in JOB
    assert "--nproc_per_node=1" in JOB
    assert "--steps \"$STEPS\"" in JOB
    assert "--compile" in JOB
