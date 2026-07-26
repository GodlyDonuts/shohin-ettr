from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "pipeline/jobs/build_ettr_claim_runtime.sbatch"
SMOKE = ROOT / "pipeline/jobs/smoke_ettr_claim_runtime.sbatch"


def test_claim_runtime_build_is_cpu_only_commit_pinned_and_complete() -> None:
    source = BUILD.read_text(encoding="ascii")
    assert "#SBATCH --gres" not in source
    assert "SOURCE_COMMIT=${SOURCE_COMMIT:?" in source
    assert "SAFETENSORS_WHEEL=${SAFETENSORS_WHEEL:?" in source
    assert "EXPECTED_SAFETENSORS_WHEEL_SHA256=" in source
    assert "#SBATCH --export=NONE" in source
    assert "EXPECTED_HOST_PYTHON_SHA256=" in source
    assert "zipfile.ZipFile" in source
    assert "clearing special bits" in source
    assert 'getattr(os, "O_NOFOLLOW", 0)' in source
    assert "os.fchmod(descriptor, mode)" in source
    assert "normalized directory mode differs" in source
    assert "torch.version.cuda is None" in source
    assert "export PATH=/usr/bin:/bin" in source
    assert "unset CDPATH ENV BASH_ENV" in source
    assert "WHEEL_FD_PATH=/proc/self/fd/9" in source
    assert "/usr/bin/cmp -s" in source
    assert "publish_once" in source
    assert "$OUT.source-bundle.sha256" in source
    assert 'git -C "$SOURCE_ROOT" show "$SOURCE_COMMIT:train/$filename"' in source
    assert '"$ENV_ROOT/bin/python" -I - <<' not in source
    assert 'RUNTIME_PY="$STAGING/runtime/miniforge3/bin/python"' in source
    assert '"$RUNTIME_PY" -I -B' in source
    assert '"$RUNTIME_PY" -I -S -B' in source
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
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
    assert "#SBATCH --gres=gpu:h100:1" not in source
    assert "EXPECTED_SHA256=${EXPECTED_SHA256:?" in source
    assert "EXPECTED_INVENTORY_SHA256=${EXPECTED_INVENTORY_SHA256:?" in source
    assert "EXPECTED_SOURCE_BUNDLE_SHA256=" in source
    assert "EXPECTED_BWRAP_SHA256=${EXPECTED_BWRAP_SHA256:?" in source
    assert "#SBATCH --export=NONE" in source
    assert "TRUSTED_PYTHON=/usr/bin/python3.11" in source
    assert "EXPECTED_TRUSTED_PYTHON_SHA256=" in source
    assert "TRUSTED_VERIFIER=${TRUSTED_VERIFIER:?" in source
    assert "EXPECTED_TRUSTED_VERIFIER_SHA256=" in source
    assert "export PATH=/usr/bin:/bin" in source
    assert "unset CDPATH ENV BASH_ENV" in source
    assert "os.memfd_create" in source
    assert "fcntl.F_ADD_SEALS" in source
    assert "root-owned Python closure pass" in source
    assert "/usr/bin/env -i" in source
    assert 'BWRAP_SHA256" = "$EXPECTED_BWRAP_SHA256' in source
    assert "--unshare-net" in source
    assert "--unshare-pid" in source
    assert "--unshare-ipc" in source
    assert "--unshare-uts" in source
    assert "torch.cuda.get_device_name(0)" in source
    assert "confined runtime imports/CUDA pass" in source
    assert "--dev-bind" in source
    assert "/lib64" in source
    assert "--clearenv" not in source
    assert "safetensors" in source
    assert "run_trusted_verifier extract-exec" in source
    assert '"{ETTR_RUNTIME_ROOT}"' in source
    assert "--expected-archive-sha256" in source
    assert "--expected-source-bundle-sha256" in source
    assert '--setenv CUDA_VISIBLE_DEVICES "$ALLOCATED_GPU_INDEX"' in source
    assert (
        '"/dev/nvidia$ALLOCATED_GPU_INDEX" \\\n'
        '    "/dev/nvidia$ALLOCATED_GPU_INDEX"'
    ) in source
    assert "ETTR_ALLOCATED_GPU_INDEX" in source
    assert "torch.cuda.device_count() != 1" in source
    assert "socket.create_connection" in source
    assert "$CKPT" not in source
    assert "SHARDS=" not in source
