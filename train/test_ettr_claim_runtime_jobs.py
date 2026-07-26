from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "pipeline/jobs/build_ettr_claim_runtime.sbatch"
SMOKE = ROOT / "pipeline/jobs/smoke_ettr_claim_runtime.sbatch"


def test_claim_runtime_build_is_cpu_only_commit_pinned_and_complete() -> None:
    source = BUILD.read_text(encoding="ascii")
    assert "#SBATCH --gres" not in source
    assert "SOURCE_COMMIT=${SOURCE_COMMIT:?" in source
    assert 'git -C "$SOURCE_ROOT" show "$SOURCE_COMMIT:train/$filename"' in source
    assert '"$ENV_ROOT/bin/python" -I -B' in source
    assert "cryptography" in source
    assert "safetensors" in source
    assert "torch" in source
    assert "ettr_claim_runtime.py" in source
    assert "ettr_deployment_contract.py" in source
    assert "landlock_stage_exec.py" in source
    assert "run_ettr_verified_stage.py" in source
    assert "runtime/app/candidate/world" in source
    assert "runtime/app/candidate/command" in source
    assert "runtime/app/candidate/query" in source
    assert "ettr-runtime-bundle-$stage.json" in source
    assert '--stage "$stage"' in source
    assert "$OUT.runtime-bundle-world.json" in source
    assert "$OUT.runtime-bundle-command.json" in source
    assert "$OUT.runtime-bundle-query.json" in source
    assert "*.pth" in source
    assert "$CKPT" not in source
    assert "SHARDS=" not in source


def test_claim_runtime_smoke_requires_pins_h100_cuda_and_bwrap_netns() -> None:
    source = SMOKE.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:h100:1" in source
    assert "EXPECTED_SHA256=${EXPECTED_SHA256:?" in source
    assert "EXPECTED_INVENTORY_SHA256=${EXPECTED_INVENTORY_SHA256:?" in source
    assert "EXPECTED_BWRAP_SHA256=${EXPECTED_BWRAP_SHA256:?" in source
    assert 'BWRAP_SHA256" = "$EXPECTED_BWRAP_SHA256' in source
    assert "--unshare-net" in source
    assert "--unshare-pid" in source
    assert "--unshare-ipc" in source
    assert "--unshare-uts" in source
    assert "torch.cuda.get_device_name(0)" in source
    assert "confined runtime imports/CUDA pass" in source
    assert "--dev-bind" in source
    assert "/lib64" in source
    assert "--clearenv" in source
    assert "safetensors" in source
    assert "verify-tree" in source
    assert "socket.create_connection" in source
    assert "$CKPT" not in source
    assert "SHARDS=" not in source
