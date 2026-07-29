#!/bin/bash
# Assemble a fresh immutable main tree from primary and hash-bound recovery
# worker outputs. This job is CPU-only and never mutates either source tree.
#SBATCH -J ettr-v3-assemble
#SBATCH -p normal
#SBATCH --account=skattel
#SBATCH --qos=normal
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/ettr-il-v3/logs/materialize-assembly-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
RUNTIME_ROOT=${RUNTIME_ROOT:?set the exact immutable assembly runtime}
SOURCE_ROOT=${SOURCE_ROOT:?set the exact frozen materializer source root}
TASK_MANIFEST=${TASK_MANIFEST:?set the immutable main task manifest}
PRIMARY_OUTPUT_ROOT=${PRIMARY_OUTPUT_ROOT:?set the original output root}
PRIMARY_REPORTS_ROOT=${PRIMARY_REPORTS_ROOT:?set the original reports root}
RECOVERY_BASE=${RECOVERY_BASE:?set the completed recovery root}
RECOVERY_INDICES=${RECOVERY_INDICES:?set comma-separated recovery indices}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a fresh assembled output root}
REPORTS_ROOT=${REPORTS_ROOT:?set a fresh assembled reports root}
RECEIPT=${RECEIPT:?set a fresh assembly receipt}
EXECUTION_SOURCE_COMMIT=${EXECUTION_SOURCE_COMMIT:?set the assembly source commit}
ASSEMBLY_SHA256=${ASSEMBLY_SHA256:?set the assembly source SHA-256}
ADAPTER_SOURCE_COMMIT=${ADAPTER_SOURCE_COMMIT:?set the adapter source commit}
ADAPTER_SHA256=${ADAPTER_SHA256:?set the adapter source SHA-256}
ASSEMBLER=$RUNTIME_ROOT/assemble_ettr_il_v3_materialization_recovery.py

for path in \
  "$RUNTIME_ROOT" \
  "$SOURCE_ROOT" \
  "$TASK_MANIFEST" \
  "$PRIMARY_OUTPUT_ROOT" \
  "$PRIMARY_REPORTS_ROOT" \
  "$RECOVERY_BASE" \
  "$OUTPUT_ROOT" \
  "$REPORTS_ROOT" \
  "$RECEIPT" \
  "$ASSEMBLER" \
  "$PY"; do
  [[ "$path" == /* ]] || {
    echo "[ettr-assembly] all paths must be absolute" >&2
    exit 2
  }
done
if [[ ! -d "$RUNTIME_ROOT" || -L "$RUNTIME_ROOT" \
  || ! -s "$RUNTIME_ROOT/SHA256SUMS" \
  || ! -d "$SOURCE_ROOT" || -L "$SOURCE_ROOT" \
  || ! -f "$TASK_MANIFEST" || -L "$TASK_MANIFEST" \
  || ! -d "$PRIMARY_OUTPUT_ROOT" || -L "$PRIMARY_OUTPUT_ROOT" \
  || ! -d "$PRIMARY_REPORTS_ROOT" || -L "$PRIMARY_REPORTS_ROOT" \
  || ! -d "$RECOVERY_BASE" || -L "$RECOVERY_BASE" \
  || -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT" \
  || -e "$REPORTS_ROOT" || -L "$REPORTS_ROOT" \
  || -e "$RECEIPT" || -L "$RECEIPT" \
  || ! -f "$ASSEMBLER" || -L "$ASSEMBLER" \
  || ! -x "$PY" ]]; then
  echo "[ettr-assembly] preflight differs" >&2
  exit 2
fi
if [[ ! "$EXECUTION_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
  || ! "$ASSEMBLY_SHA256" =~ ^[0-9a-f]{64}$ \
  || ! "$ADAPTER_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
  || ! "$ADAPTER_SHA256" =~ ^[0-9a-f]{64}$ \
  || ! "$RECOVERY_INDICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[ettr-assembly] scalar contract differs" >&2
  exit 2
fi
(cd "$RUNTIME_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
printf '%s  %s\n' "$ASSEMBLY_SHA256" "$ASSEMBLER" | sha256sum -c -

arguments=()
IFS=',' read -r -a recovery_indices <<<"$RECOVERY_INDICES"
for index in "${recovery_indices[@]}"; do
  task_root="$RECOVERY_BASE/task-$(printf '%05d' "$index")"
  [[ -d "$task_root/output" && -d "$task_root/reports" ]] || {
    echo "[ettr-assembly] recovery task is incomplete: $index" >&2
    exit 3
  }
  arguments+=(--recovery "$index|$task_root/output|$task_root/reports")
done

export PYTHONPATH="$RUNTIME_ROOT:$SOURCE_ROOT/pipeline"
"$PY" -u "$ASSEMBLER" \
  --task-manifest "$TASK_MANIFEST" \
  --primary-output-root "$PRIMARY_OUTPUT_ROOT" \
  --primary-reports-root "$PRIMARY_REPORTS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --reports-root "$REPORTS_ROOT" \
  --receipt "$RECEIPT" \
  "${arguments[@]}" \
  --execution-source-commit "$EXECUTION_SOURCE_COMMIT" \
  --execution-code-sha256 "$ASSEMBLY_SHA256" \
  --expected-adapter-source-commit "$ADAPTER_SOURCE_COMMIT" \
  --expected-adapter-sha256 "$ADAPTER_SHA256"
