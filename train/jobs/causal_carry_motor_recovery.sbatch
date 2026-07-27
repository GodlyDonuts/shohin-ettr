#!/bin/bash -p
#SBATCH --job-name=carry-recovery-nulls-v3
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=evc22,evc26,evc31,evc32,evc36,evc37,evc40,evc43,evc44
#SBATCH --no-requeue
#SBATCH --open-mode=truncate
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/carry_recovery_nulls_%j_%r.out
#SBATCH -e /lustre/fs1/home/sa305415/shohin/logs/carry_recovery_nulls_%j_%r.err

if [[ $- != *p* ]] || [[ $(builtin type -t compgen) != builtin ]]; then
  printf '%s\n' "recovery wrapper requires a privileged clean Bash boundary" >&2
  exit 2
fi
STARTUP_FUNCTIONS=$(builtin compgen -A function || :)
if [[ -n "$STARTUP_FUNCTIONS" ]]; then
  printf '%s\n' "exported Bash functions are forbidden before recovery startup" >&2
  exit 2
fi
unset STARTUP_FUNCTIONS
if [[ -n ${BASH_ENV+x} ]]; then
  printf '%s\n' "BASH_ENV is forbidden before recovery startup" >&2
  exit 2
fi
while IFS= read -r declaration; do
  case "$declaration" in
    *' BASH_ENV='*|*' SHELLOPTS='*)
      printf '%s\n' "exported Bash startup controls are forbidden" >&2
      exit 2
      ;;
  esac
done < <(builtin export -p)

set -euo pipefail
umask 077
if [[ "$(umask)" != 0077 ]]; then
  echo "recovery wrapper could not enforce umask 0077" >&2
  exit 2
fi
export PATH=/usr/local/bin:/usr/bin:/bin
export LANG=C
export LC_ALL=C

FORBIDDEN_ENVIRONMENT=(
  BASH_ENV
  BLIS_ARCH_TYPE
  CUBLAS_FORCE_TF32
  CUBLASLT_WORKSPACE_SIZE
  CUDA_AUTO_BOOST
  CUDA_CACHE_DISABLE
  CUDA_DEVICE_MAX_CONNECTIONS
  CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT
  CUDA_DEVICE_WAITS_ON_EXCEPTION
  CUDA_FORCE_PTX_JIT
  CUDA_LAUNCH_BLOCKING
  CUDA_MANAGED_FORCE_DEVICE_ALLOC
  CUDA_MODULE_LOADING
  CUDNN_BENCHMARK
  CUDNN_DETERMINISTIC
  KMP_AFFINITY
  KMP_BLOCKTIME
  LD_AUDIT
  LD_DEBUG
  LD_LIBRARY_PATH
  LD_PRELOAD
  LD_PROFILE
  MKL_CBWR
  MKL_DEBUG_CPU_TYPE
  MKL_ENABLE_INSTRUCTIONS
  MKL_SERVICE_FORCE_INTEL
  NVIDIA_TF32_OVERRIDE
  NPY_BLAS_ORDER
  NPY_LAPACK_ORDER
  OPENBLAS_CORETYPE
  PYTHONHOME
  PYTHONINSPECT
  PYTHONPATH
  PYTHONSAFEPATH
  PYTHONSTARTUP
  PYTHONUSERBASE
  PYTHONWARNINGS
  PYTORCH_CUDA_ALLOC_CONF
  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
  TORCH_BLAS_PREFER_CUBLASLT
  TORCH_CUDNN_V8_API_DEBUG
  TORCH_CUDNN_V8_API_ENABLED
  TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
  TORCH_FORCE_WEIGHTS_ONLY_LOAD
  TORCH_LINALG_PREFER_CUSOLVER
  VECLIB_MAXIMUM_THREADS
)
for name in "${FORBIDDEN_ENVIRONMENT[@]}"; do
  if [[ -n ${!name+x} ]]; then
    echo "ambient recovery override is forbidden: $name" >&2
    exit 2
  fi
done
while IFS= read -r name; do
  case "$name" in
    CUBLAS_WORKSPACE_CONFIG|GIT_ATTR_NOSYSTEM|GIT_CONFIG_GLOBAL|GIT_CONFIG_NOSYSTEM|GIT_CONFIG_SYSTEM|GIT_NO_REPLACE_OBJECTS|MKL_NUM_THREADS|NUMEXPR_NUM_THREADS|OMP_NUM_THREADS|OPENBLAS_NUM_THREADS)
      ;;
    BASH_FUNC_*|BLIS_*|CUBLAS_*|CUBLASLT_*|CUDNN_*|DYLD_*|GIT_*|KMP_*|MKL_*|NCCL_*|NPY_BLAS_*|NPY_LAPACK_*|NUMEXPR_*|OMP_*|OPENBLAS_*|PYTORCH_*|SBATCH_*|TORCH_*|TORCHDYNAMO_*|TORCHINDUCTOR_*|VECLIB_*)
      echo "ambient recovery control variable is forbidden: $name" >&2
      exit 2
      ;;
  esac
done < <(builtin compgen -e)
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_SYSTEM=/dev/null
export GIT_ATTR_NOSYSTEM=1
export GIT_NO_REPLACE_OBJECTS=1

if (( $# != 6 )); then
  echo "recovery wrapper requires six reviewed positional controls" >&2
  exit 2
fi
readonly SOURCE_ROOT=$1
readonly RECOVERY_SOURCE_COMMIT=$2
readonly RECOVERY_SOURCE_MANIFEST_SHA256=$3
readonly HOSTILE_REVIEW=$4
readonly HOSTILE_REVIEW_SHA256=$5
readonly RECOVERY_PLAN_SHA256=$6
readonly CONSTANT_BIAS_CONTROL=zero_sum_delta_v1
readonly NUISANCE_ONLY_CONTROL=saturated_op_width_position_v1
readonly NULL_OPTIMIZATION_CONTROL=carry_motor_full_board_convex_selection_v2
readonly SELECTION_WIDTHS=4,6
readonly EXPECTED_EXCLUDED_NODES=evc22,evc26,evc31,evc32,evc36,evc37,evc40,evc43,evc44
readonly EXPECTED_EXCLUDED_EXPANDED=$'evc22\nevc26\nevc31\nevc32\nevc36\nevc37\nevc40\nevc43\nevc44'

readonly DATA_ROOT=/lustre/fs1/home/sa305415/shohin
readonly PY=$DATA_ROOT/miniforge3/bin/python
readonly SCONTROL=/usr/bin/scontrol
readonly SACCT=/usr/bin/sacct
readonly NVIDIA_SMI=/usr/bin/nvidia-smi
readonly UPSTREAM_COMMIT=a0c258e6709766c643cf127a429a7d6ef4a4211b
readonly UPSTREAM_PLAN_SHA256=1b845d47f6875df571169efb5adb0716dfbc5d266a2499e4a92451351a262b6d
readonly UPSTREAM_ROOT=$DATA_ROOT/artifacts/carry_motor/canonical_$UPSTREAM_COMMIT
readonly RECOVERY_PARENT=$DATA_ROOT/artifacts/carry_motor/recoveries
readonly RECOVERY_PARENT_RECEIPT=$RECOVERY_PARENT/parent_receipt.json
readonly RECOVERY_ROOT=$RECOVERY_PARENT/upstream_${UPSTREAM_PLAN_SHA256}_executor_${RECOVERY_SOURCE_COMMIT}
readonly RECOVERY_PLAN=$RECOVERY_ROOT/recovery_plan.json
readonly EXPECTED_REVIEW=$DATA_ROOT/artifacts/carry_motor/recovery_reviews/review_${RECOVERY_SOURCE_COMMIT}/hostile_review.json

for value in "$SOURCE_ROOT" "$PY" "$HOSTILE_REVIEW" "$RECOVERY_PARENT" \
  "$RECOVERY_PARENT_RECEIPT" "$RECOVERY_ROOT" "$RECOVERY_PLAN"; do
  if [[ "$value" != /* ]]; then
    echo "recovery paths must be absolute: $value" >&2
    exit 2
  fi
done
if [[ ! -x "$PY" || ! -x "$SCONTROL" || -L "$SCONTROL" || \
      ! -x "$SACCT" || -L "$SACCT" || \
      ! -x "$NVIDIA_SMI" || -L "$NVIDIA_SMI" ]]; then
  echo "pinned recovery interpreter, Git, Slurm, or GPU executable is unavailable" >&2
  exit 2
fi
if [[ "$(realpath -e -- "$SOURCE_ROOT")" != "$SOURCE_ROOT" ]]; then
  echo "recovery source root must be an exact physical path" >&2
  exit 2
fi
if [[ ! "$RECOVERY_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ || \
      "$RECOVERY_SOURCE_COMMIT" == "$UPSTREAM_COMMIT" ]]; then
  echo "recovery source commit is invalid or aliases upstream" >&2
  exit 2
fi
if [[ ! "$RECOVERY_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ || \
      ! "$HOSTILE_REVIEW_SHA256" =~ ^[0-9a-f]{64}$ || \
      ! "$RECOVERY_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "recovery receipts must be lowercase SHA-256" >&2
  exit 2
fi
if [[ "$HOSTILE_REVIEW" != "$EXPECTED_REVIEW" ]]; then
  echo "hostile-review path differs from commit-derived custody" >&2
  exit 2
fi
for name in CONFIRMATION_SECRET_FILE WIDTH8_SELECTION_FILE PUBLIC_WIDTH8_FILE; do
  if [[ -n ${!name+x} ]]; then
    echo "width8 and confirmation inputs are forbidden in recovery fit: $name" >&2
    exit 2
  fi
done

if [[ ! -f "$0" || -L "$0" ]] || ! cmp --silent -- "$0" \
  "$SOURCE_ROOT/train/jobs/causal_carry_motor_recovery.sbatch"; then
  echo "spooled recovery wrapper differs from reviewed checkout" >&2
  exit 2
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for pair in \
  "$HOSTILE_REVIEW:$HOSTILE_REVIEW_SHA256:444" \
  "$RECOVERY_PLAN:$RECOVERY_PLAN_SHA256:444"; do
  path=${pair%%:*}
  remainder=${pair#*:}
  receipt=${remainder%%:*}
  mode=${remainder##*:}
  if [[ ! -f "$path" || -L "$path" || \
        "$(stat -c '%F' "$path")" != "regular file" || \
        "$(stat -c '%a' "$path")" != "$mode" || \
        "$(stat -c '%h' "$path")" != 1 || \
        "$(sha256sum "$path" | awk '{print $1}')" != "$receipt" ]]; then
    echo "sealed recovery input failed: $path" >&2
    exit 2
  fi
  if [[ "$(stat -c '%a' "${path%/*}")" != 555 ]]; then
    echo "sealed recovery input parent mode failed: $path" >&2
    exit 2
  fi
done

if [[ "$(realpath -e -- "$RECOVERY_PARENT")" != "$RECOVERY_PARENT" || \
      ! -d "$RECOVERY_PARENT" || -L "$RECOVERY_PARENT" || \
      "$(stat -c '%a' "$RECOVERY_PARENT")" != 700 || \
      ! -f "$RECOVERY_PARENT_RECEIPT" || -L "$RECOVERY_PARENT_RECEIPT" || \
      "$(stat -c '%a' "$RECOVERY_PARENT_RECEIPT")" != 444 || \
      "$(stat -c '%h' "$RECOVERY_PARENT_RECEIPT")" != 1 ]]; then
  echo "recovery parent or durable receipt identity failed" >&2
  exit 2
fi

if [[ ! -d "$UPSTREAM_ROOT" || -L "$UPSTREAM_ROOT" || \
      "$(stat -c '%a' "$UPSTREAM_ROOT")" != 555 ]]; then
  echo "upstream canonical root is not immutable" >&2
  exit 2
fi
if [[ ! -d "$RECOVERY_ROOT" || -L "$RECOVERY_ROOT" || \
      "$(stat -c '%a' "$RECOVERY_ROOT")" != 555 ]]; then
  echo "recovery root is not immutable" >&2
  exit 2
fi
FIT_MODE=$(stat -c '%a' "$RECOVERY_ROOT/fit" 2>/dev/null || true)
FIT_CHILDREN=$(find "$RECOVERY_ROOT/fit" -mindepth 1 -maxdepth 1 -printf '%f\n' 2>/dev/null | sort || true)
if [[ ! -d "$RECOVERY_ROOT/fit" || -L "$RECOVERY_ROOT/fit" ]]; then
  echo "recovery fit output is not empty, recoverable, or sealed" >&2
  exit 2
fi
case "$FIT_MODE:$FIT_CHILDREN" in
  700:|700:motor.pt|700:.motor.pt.owner|\
  700:$'.motor.pt.owner\n.motor.pt.stage'|\
  700:$'.motor.pt.owner\nmotor.pt'|\
  700:$'.motor.pt.owner\n.motor.pt.stage\nmotor.pt'|555:motor.pt)
    ;;
  *)
    echo "recovery fit output is not empty, recoverable, or sealed" >&2
    exit 2
    ;;
esac
EXPECTED_FIT_LINKS=1
if [[ "$FIT_CHILDREN" == motor.pt || "$FIT_CHILDREN" == *$'\nmotor.pt' ]]; then
  FIT_ARTIFACT_MODE=$(stat -c '%a' "$RECOVERY_ROOT/fit/motor.pt" 2>/dev/null || true)
  FIT_ARTIFACT_LINKS=$(stat -c '%h' "$RECOVERY_ROOT/fit/motor.pt" 2>/dev/null || true)
  if [[ "$FIT_CHILDREN" == *$'.motor.pt.stage\nmotor.pt' ]]; then
    EXPECTED_FIT_LINKS=2
  fi
  if [[ ! -f "$RECOVERY_ROOT/fit/motor.pt" || -L "$RECOVERY_ROOT/fit/motor.pt" || \
        "$FIT_ARTIFACT_LINKS" != "$EXPECTED_FIT_LINKS" || \
        ( "$FIT_MODE" == 700 && "$FIT_ARTIFACT_MODE" != 444 ) || \
        ( "$FIT_MODE" == 555 && "$FIT_ARTIFACT_MODE" != 444 ) ]]; then
    echo "existing recovery fit artifact is not interrupted, recoverable, or sealed" >&2
    exit 2
  fi
fi
if [[ "$FIT_CHILDREN" == *'.motor.pt.owner'* ]] && \
   [[ ! -f "$RECOVERY_ROOT/fit/.motor.pt.owner" || \
      -L "$RECOVERY_ROOT/fit/.motor.pt.owner" || \
      "$(stat -c '%a:%h' "$RECOVERY_ROOT/fit/.motor.pt.owner")" != 600:1 ]]; then
  echo "recovery fit publisher ownership is malformed" >&2
  exit 2
fi
if [[ "$FIT_CHILDREN" == *'.motor.pt.stage'* ]]; then
  FIT_STAGE_MODE=$(stat -c '%a' "$RECOVERY_ROOT/fit/.motor.pt.stage" 2>/dev/null || true)
  FIT_STAGE_LINKS=$(stat -c '%h' "$RECOVERY_ROOT/fit/.motor.pt.stage" 2>/dev/null || true)
  if [[ ! -f "$RECOVERY_ROOT/fit/.motor.pt.stage" || \
        -L "$RECOVERY_ROOT/fit/.motor.pt.stage" || \
        ( "$FIT_STAGE_MODE" != 600 && "$FIT_STAGE_MODE" != 444 ) || \
        "$FIT_STAGE_LINKS" != "$EXPECTED_FIT_LINKS" ]]; then
    echo "recovery fit publisher stage is malformed" >&2
    exit 2
  fi
  if [[ "$EXPECTED_FIT_LINKS" == 2 && \
        "$(stat -c '%d:%i' "$RECOVERY_ROOT/fit/.motor.pt.stage")" != \
        "$(stat -c '%d:%i' "$RECOVERY_ROOT/fit/motor.pt")" ]]; then
    echo "recovery fit stage and committed target differ" >&2
    exit 2
  fi
fi

if [[ ! ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ || \
      ! ${CUDA_VISIBLE_DEVICES:-} =~ ^[^,[:space:]]+$ ]]; then
  echo "recovery fit requires one exact Slurm job and CUDA-visible device" >&2
  exit 2
fi
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
if [[ "$RESTART_COUNT" != 0 ]]; then
  echo "refusing restarted recovery fit" >&2
  exit 2
fi
JOB_RECORD=$("$SCONTROL" show job -o "$SLURM_JOB_ID")
if [[ "$JOB_RECORD" != *" Account=skattel "* || \
      "$JOB_RECORD" != *" Partition=normal "* || \
      "$JOB_RECORD" != *" Requeue=0 "* ]]; then
  echo "recovery fit requires live skattel/normal/Requeue=0 allocation" >&2
  exit 2
fi
NODE_LIST=$(sed -n 's/.* NodeList=\([^ ]*\).*/\1/p' <<<"$JOB_RECORD")
EXCLUDED_NODE_LIST=$(sed -n 's/.* ExcNodeList=\([^ ]*\).*/\1/p' <<<"$JOB_RECORD")
ALLOCATED_NODES=$("$SCONTROL" show hostnames "$NODE_LIST")
EXCLUDED_NODES=$("$SCONTROL" show hostnames "$EXCLUDED_NODE_LIST")
if [[ -z "$NODE_LIST" || -z "$EXCLUDED_NODE_LIST" || \
      "$ALLOCATED_NODES" != "$NODE_LIST" || \
      "$EXCLUDED_NODES" != "$EXPECTED_EXCLUDED_EXPANDED" ]]; then
  echo "recovery Slurm NodeList/ExcNodeList differs from sealed exclusions" >&2
  exit 2
fi
case ",$EXPECTED_EXCLUDED_NODES," in
  *",$NODE_LIST,"*)
    echo "recovery allocation landed on an excluded node: $NODE_LIST" >&2
    exit 2
    ;;
esac
for spec in "StdOut=carry_recovery_nulls_${SLURM_JOB_ID}_${RESTART_COUNT}.out" \
            "StdErr=carry_recovery_nulls_${SLURM_JOB_ID}_${RESTART_COUNT}.err"; do
  key=${spec%%=*}
  suffix=${spec#*=}
  observed=$(sed -n "s/.* ${key}=\([^ ]*\).*/\1/p" <<<"$JOB_RECORD")
  if [[ "$observed" != *"$suffix" ]]; then
    echo "recovery $key is not restart-distinguishing: $observed" >&2
    exit 2
  fi
done

if [[ "$("$NVIDIA_SMI" --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')" != 1 || \
      "$("$NVIDIA_SMI" --query-gpu=name --format=csv,noheader | tr -d '\r')" != \
      "NVIDIA H100 PCIe" ]]; then
  echo "recovery fit requires exactly one NVIDIA H100 PCIe" >&2
  exit 75
fi

"$PY" -I -S -B "$SOURCE_ROOT/train/causal_carry_motor_recovery.py" fit \
  --recovery-source-commit "$RECOVERY_SOURCE_COMMIT" \
  --recovery-source-manifest-sha256 "$RECOVERY_SOURCE_MANIFEST_SHA256" \
  --hostile-review "$HOSTILE_REVIEW" \
  --hostile-review-sha256 "$HOSTILE_REVIEW_SHA256" \
  --recovery-root "$RECOVERY_ROOT" \
  --recovery-plan "$RECOVERY_PLAN" \
  --recovery-plan-sha256 "$RECOVERY_PLAN_SHA256" \
  --constant-bias-control "$CONSTANT_BIAS_CONTROL" \
  --nuisance-only-control "$NUISANCE_ONLY_CONTROL" \
  --null-optimization-control "$NULL_OPTIMIZATION_CONTROL" \
  --selection-widths "$SELECTION_WIDTHS"
