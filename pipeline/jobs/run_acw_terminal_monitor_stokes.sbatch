#!/bin/bash
#SBATCH -J shohin-acw-dev-monitor
#SBATCH -p normal
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH --no-requeue
#SBATCH -o /lustre/fs1/home/sa305415/shohin_acw/logs/acw_development_monitor_%j.out
#SBATCH --open-mode=append
#SBATCH --export=NONE
# Canonical post-job monitor. This fifth held CPU job runs only after all four
# scientific roles complete. It reopens their evidence under the pinned Stokes
# runtime, computes the retained baseline without printing metrics, freezes the
# trees, and writes an externally anchorable terminal receipt.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin_acw
REVIEWED_LOCAL_REF=refs/heads/codex/acw-g2
REVIEWED_TRACKING_REF=refs/remotes/origin/codex/acw-g2
REVIEWED_REMOTE_REF=refs/heads/codex/acw-g2
PY=/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13
PYZIP=/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python313.zip
STDLIB=/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13
DYNLIB=$STDLIB/lib-dynload
SITE=$STDLIB/site-packages
ROOT=$BASE/artifacts/r12/acw_development_g2
BUILDER=$BASE/pipeline/build_acw_development_manifest.py
PYCACHE=/tmp/shohin-acw-monitor-pycache-$SLURM_JOB_ID

[[ $# = 0 ]]
: "${SLURM_CPUS_PER_TASK:?}"
: "${SLURM_JOB_ID:?}"
: "${SLURM_JOB_NAME:?}"
: "${SLURM_JOB_NODELIST:?}"
: "${SLURM_NODELIST:?}"
: "${SLURM_SUBMIT_DIR:?}"
[[ "$SLURM_CPUS_PER_TASK" = 4 ]]
[[ "$SLURM_JOB_NODELIST" = "$SLURM_NODELIST" ]]
[[ "$SLURM_SUBMIT_DIR" = "$BASE" ]]
[[ ! -e "$PYZIP" ]]
test "$({ /usr/bin/awk -F ': ' '/model name/{print $2; exit}' /proc/cpuinfo; })" = \
  "Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz"

BATCH_TASK="/slurm/uid_$(/usr/bin/id -u)/job_${SLURM_JOB_ID}/step_batch/task_0"
/usr/bin/grep -Fq "$BATCH_TASK" /proc/self/cgroup

validate_python_origins() {
  /usr/bin/env -i \
    LANG=C \
    LC_ALL=C \
    PATH=/usr/bin:/bin \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    "$PY" -I -S -c '
# BEGIN ACW_PYTHON_ORIGIN_GUARD
import importlib.util
import os
import sys
from pathlib import Path

base, stdlib, dynlib, site, pyzip, executable = (
    Path(value) for value in sys.argv[1:]
)
for label, path in (
    ("base", base),
    ("stdlib", stdlib),
    ("dynlib", dynlib),
    ("site", site),
    ("executable", executable),
):
    if path.resolve(strict=True) != path:
        raise SystemExit(f"non-canonical {label} path: {path}")
if os.path.lexists(pyzip):
    raise SystemExit(f"forbidden Python ZIP path exists: {pyzip}")
if not sys.flags.no_site or not sys.flags.safe_path:
    raise SystemExit("origin guard requires isolated no-site safe-path Python")
if Path(sys.executable).resolve(strict=True) != executable:
    raise SystemExit("origin guard executable differs from the pinned Python")

collisions = []
stdlib_names = frozenset(sys.stdlib_module_names)
for root in (base, site):
    for entry in os.scandir(root):
        top_level = entry.name.split(".", 1)[0]
        if top_level in stdlib_names:
            collisions.append(f"{root}/{entry.name}")
if collisions:
    raise SystemExit("mutable stdlib collision: " + ", ".join(sorted(collisions)))

# Reproduce the committed -S -P search path exactly, then prove that mutable
# roots cannot win any intended standard-library import.
sys.path[:] = [str(base), str(site), str(pyzip), str(stdlib), str(dynlib)]
critical_stdlib = (
    "argparse",
    "copy",
    "ctypes",
    "dataclasses",
    "errno",
    "hashlib",
    "io",
    "itertools",
    "json",
    "math",
    "os",
    "pathlib",
    "platform",
    "random",
    "re",
    "resource",
    "secrets",
    "shutil",
    "stat",
    "statistics",
    "subprocess",
    "sys",
    "tempfile",
    "time",
    "typing",
)
for name in critical_stdlib:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"stdlib module is unresolved: {name}")
    if spec.origin in {"built-in", "frozen"}:
        continue
    origin = Path(spec.origin).resolve(strict=True)
    if origin.is_relative_to(base) or origin.is_relative_to(site):
        raise SystemExit(f"stdlib module resolved through a mutable root: {name}")
    if not origin.is_relative_to(stdlib):
        raise SystemExit(f"stdlib module escaped the pinned stdlib: {name}")

pipeline_spec = importlib.util.find_spec("pipeline")
locations = (
    []
    if pipeline_spec is None or pipeline_spec.submodule_search_locations is None
    else [Path(value).resolve(strict=True) for value in pipeline_spec.submodule_search_locations]
)
if pipeline_spec is None or pipeline_spec.origin is not None or locations != [base / "pipeline"]:
    raise SystemExit("pipeline namespace did not resolve only from the pinned checkout")
# END ACW_PYTHON_ORIGIN_GUARD
' "$BASE" "$STDLIB" "$DYNLIB" "$SITE" "$PYZIP" "$PY"
}

run_python() {
  validate_python_origins
  /usr/bin/env -i \
    ATEN_CPU_CAPABILITY=avx2 \
    CUDA_VISIBLE_DEVICES= \
    DNNL_MAX_CPU_ISA=AVX2 \
    HOME=/home/sa305415 \
    KMP_DETERMINISTIC_REDUCTION=TRUE \
    LANG=C \
    LC_ALL=C \
    LOGNAME=sa305415 \
    MKL_CBWR=AVX2 \
    MKL_DYNAMIC=FALSE \
    MKL_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    OMP_NUM_THREADS=1 \
    ONEDNN_MAX_CPU_ISA=AVX2 \
    OPENBLAS_CORETYPE=Haswell \
    OPENBLAS_NUM_THREADS=1 \
    PATH=/apps/slurm/current/bin:/usr/bin:/bin \
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="$PYCACHE" \
    PYTHONPATH=$BASE:/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages \
    PYTHONUTF8=1 \
    SHELL=/bin/bash \
    SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
    SLURM_JOB_ID="$SLURM_JOB_ID" \
    SLURM_JOB_NAME="$SLURM_JOB_NAME" \
    SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST" \
    SLURM_NODELIST="$SLURM_NODELIST" \
    SLURM_SUBMIT_DIR="$SLURM_SUBMIT_DIR" \
    TZ=UTC \
    USER=sa305415 \
    "$PY" -S -P "$@"
}

cleanup() {
  local status=$?
  /usr/bin/rm -rf "$PYCACHE"
  exit "$status"
}
trap cleanup EXIT

/usr/bin/rm -rf "$PYCACHE"
/usr/bin/install -d -m 700 "$PYCACHE"
cd "$BASE"
test "$({ /usr/bin/git --no-replace-objects symbolic-ref -q HEAD; })" = \
  "$REVIEWED_LOCAL_REF"
test "$({ /usr/bin/git --no-replace-objects rev-parse HEAD; })" = \
  "$({ /usr/bin/git --no-replace-objects rev-parse --verify "$REVIEWED_TRACKING_REF"; })"
test "$({ /usr/bin/git --no-replace-objects rev-parse HEAD; })" = \
  "$({ /usr/bin/git --no-replace-objects ls-remote origin "$REVIEWED_REMOTE_REF" | /usr/bin/awk '{print $1}'; })"
run_python "$BUILDER" validate-plan --require-ready
if ! run_python "$BUILDER" terminal-accounting --root "$ROOT"; then
  echo "[acw-development-monitor] retrying resumable terminal accounting" >&2
  run_python "$BUILDER" terminal-accounting --root "$ROOT"
fi
if ! run_python "$BUILDER" verify-terminal-accounting --root "$ROOT"; then
  echo "[acw-development-monitor] retrying terminal verification" >&2
  run_python "$BUILDER" terminal-accounting --root "$ROOT"
  run_python "$BUILDER" verify-terminal-accounting --root "$ROOT"
fi

trap - EXIT
/usr/bin/rm -rf "$PYCACHE"
echo "[acw-development-monitor] complete root=$ROOT performance_claim_ready=0"
