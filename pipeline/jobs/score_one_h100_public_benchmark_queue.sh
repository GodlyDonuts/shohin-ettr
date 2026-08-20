#!/bin/bash
set -euo pipefail

: "${ALLOCATION_JOB_ID:?}" "${RUNTIME_ROOT:?}" "${ARTIFACT_ROOT:?}" "${PYTHON:?}"
: "${SCORING_DEPS:?}" "${BASE_ENV_ROOT:?}" "${NVIDIA_RULER_COMMIT:?}"
: "${EVALPLUS_COMMIT:?}" "${LIVECODEBENCH_COMMIT:?}" "${LIVEBENCH_COMMIT:?}"
: "${LONGBENCH_COMMIT:?}" "${EMBEDDING_MODEL:?}"

POLL_SECONDS=${POLL_SECONDS:-60}
FINAL_GENERATION_REPORT="$ARTIFACT_ROOT/full_generation/mmlu_pro/report.json"
SCORE_ROOT="$ARTIFACT_ROOT/official_scores"
REPORT_ROOT="$ARTIFACT_ROOT/official_score_reports"
LOG_ROOT="$ARTIFACT_ROOT/logs"
mkdir -p "$SCORE_ROOT" "$REPORT_ROOT" "$LOG_ROOT"

while [ ! -s "$FINAL_GENERATION_REPORT" ]; do
  if ! squeue -h -j "$ALLOCATION_JOB_ID" -t RUNNING | grep -q .; then
    echo "allocation ended before full generation completed" >&2
    exit 2
  fi
  sleep "$POLL_SECONDS"
done

mkdir -p "$ARTIFACT_ROOT/sandbox_manifests"
for benchmark in livecodebench livebench; do
  sandbox_manifest="$ARTIFACT_ROOT/sandbox_manifests/${benchmark}.json"
  if [ ! -s "$sandbox_manifest" ]; then
    "$PYTHON" - "$ARTIFACT_ROOT/manifests/${benchmark}.json" "$sandbox_manifest" "$benchmark" <<'PY'
import json
import os
from pathlib import Path
import sys

source, destination, benchmark = map(Path, sys.argv[1:])
payload = json.loads(source.read_text())
assert len(payload["benchmarks"]) == 1
assert payload["benchmarks"][0]["name"] == str(benchmark)
payload["benchmarks"][0]["questions"] = (
    f"/assessors/{benchmark}/full.questions.jsonl"
)
temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY
  fi
done

capture_final_json() {
  local log=$1 report=$2
  local temporary="${report}.tmp.$$"
  tail -n 1 "$log" > "$temporary"
  "$PYTHON" -m json.tool "$temporary" >/dev/null
  mv "$temporary" "$report"
}

assessor_root() {
  case "$1" in
    ifeval|musr|mmlu_pro) echo "$ARTIFACT_ROOT/data" ;;
    ruler) echo "$ARTIFACT_ROOT/site_data_ruler" ;;
    *) echo "$ARTIFACT_ROOT/site_data_core" ;;
  esac
}

for benchmark in ifeval musr correctbench mmlu_pro; do
  report="$REPORT_ROOT/${benchmark}.json"
  [ -s "$report" ] && continue
  "$PYTHON" "$RUNTIME_ROOT/pipeline/score_dense_public_campaign.py" \
    --manifest "$ARTIFACT_ROOT/manifests/${benchmark}.json" \
    --generation-root "$ARTIFACT_ROOT/full_generation/${benchmark}" \
    --assessor-root "$(assessor_root "$benchmark")" \
    --assessor-name full.assessors.jsonl \
    --ifeval-root "$ARTIFACT_ROOT/sources/google-research/instruction_following_eval" \
    --official-score-root "$SCORE_ROOT" \
    --output "$report"
done

if [ ! -s "$REPORT_ROOT/ruler.json" ]; then
  log="$LOG_ROOT/score_ruler.out"
  "$PYTHON" "$RUNTIME_ROOT/pipeline/score_dense_public_ruler.py" \
    --manifest "$ARTIFACT_ROOT/manifests/ruler.json" \
    --generation-root "$ARTIFACT_ROOT/full_generation/ruler" \
    --assessor-root "$ARTIFACT_ROOT/site_data_ruler" \
    --output-root "$SCORE_ROOT" \
    --ruler-commit "$NVIDIA_RULER_COMMIT" >"$log"
  capture_final_json "$log" "$REPORT_ROOT/ruler.json"
fi

run_evalplus() {
  local benchmark=$1 dataset=$2
  local report="$REPORT_ROOT/${benchmark}.json"
  [ -s "$report" ] && return
  local work="$ARTIFACT_ROOT/evalplus_work"
  "$PYTHON" "$RUNTIME_ROOT/pipeline/eval_dense_public_evalplus.py" export \
    --benchmark "$benchmark" \
    --manifest "$ARTIFACT_ROOT/manifests/${benchmark}.json" \
    --generation-root "$ARTIFACT_ROOT/full_generation/${benchmark}" \
    --assessor-root "$ARTIFACT_ROOT/site_data_core" \
    --work-root "$work" \
    --evalplus-commit "$EVALPLUS_COMMIT" \
    >"$LOG_ROOT/score_${benchmark}_export.out"
  for stage in direct_base unchanged_continuation trained_revision; do
    bwrap \
      --ro-bind /usr /usr --ro-bind /lib64 /lib64 --ro-bind /lib /lib --ro-bind /etc /etc \
      --ro-bind "$BASE_ENV_ROOT" /env \
      --ro-bind "$SCORING_DEPS" /deps \
      --ro-bind "$ARTIFACT_ROOT/site_sources/evalplus-full" /scorer \
      --ro-bind "$ARTIFACT_ROOT/evalplus_data" /data \
      --bind "$work" /work \
      --proc /proc --dev /dev --tmpfs /tmp \
      --unshare-net --unshare-pid --unshare-ipc --unshare-uts --die-with-parent \
      env HOME=/tmp TMPDIR=/tmp PYTHONPATH=/deps:/scorer \
        HUMANEVAL_OVERRIDE_PATH=/data/HumanEvalPlus-v0.1.10.jsonl \
        MBPP_OVERRIDE_PATH=/data/MbppPlus-v0.2.0.jsonl \
      /env/bin/python3.13 /scorer/evalplus/evaluate.py \
        --dataset "$dataset" --samples "/work/${benchmark}/${stage}.jsonl" --parallel 8 \
        >"$LOG_ROOT/score_${benchmark}_${stage}.out" \
        2>"$LOG_ROOT/score_${benchmark}_${stage}.err"
  done
  local log="$LOG_ROOT/score_${benchmark}_collect.out"
  "$PYTHON" "$RUNTIME_ROOT/pipeline/eval_dense_public_evalplus.py" collect \
    --benchmark "$benchmark" \
    --manifest "$ARTIFACT_ROOT/manifests/${benchmark}.json" \
    --generation-root "$ARTIFACT_ROOT/full_generation/${benchmark}" \
    --assessor-root "$ARTIFACT_ROOT/site_data_core" \
    --work-root "$work" --output-root "$SCORE_ROOT" \
    --evalplus-commit "$EVALPLUS_COMMIT" >"$log"
  capture_final_json "$log" "$report"
}
run_evalplus humaneval_plus humaneval
run_evalplus mbpp_plus mbpp

if [ ! -s "$REPORT_ROOT/livecodebench.json" ]; then
  log="$LOG_ROOT/score_livecodebench.out"
  mkdir -p "$SCORE_ROOT"
  bwrap \
    --ro-bind /usr /usr --ro-bind /lib64 /lib64 --ro-bind /lib /lib --ro-bind /etc /etc \
    --ro-bind "$BASE_ENV_ROOT" /env --ro-bind "$SCORING_DEPS" /deps \
    --ro-bind "$RUNTIME_ROOT" /runtime \
    --ro-bind "$ARTIFACT_ROOT/site_sources/livecodebench-full" /scorer \
    --ro-bind "$ARTIFACT_ROOT/sandbox_manifests/livecodebench.json" /manifest.json \
    --ro-bind "$ARTIFACT_ROOT/full_generation/livecodebench" /generation \
    --ro-bind "$ARTIFACT_ROOT/site_data_core" /assessors \
    --bind "$SCORE_ROOT" /scores \
    --proc /proc --dev /dev --tmpfs /tmp \
    --unshare-net --unshare-pid --unshare-ipc --unshare-uts --die-with-parent \
    env HOME=/tmp TMPDIR=/tmp SHOHIN_CODE_SANDBOX=1 PYTHONPATH=/deps:/runtime/pipeline:/scorer \
    /env/bin/python3.13 /runtime/pipeline/eval_dense_public_livecodebench.py \
      --manifest /manifest.json --generation-root /generation \
      --assessor-root /assessors --output-root /scores \
      --livecodebench-root /scorer --livecodebench-commit "$LIVECODEBENCH_COMMIT" \
      --workers 8 >"$log" 2>"$LOG_ROOT/score_livecodebench.err"
  capture_final_json "$log" "$REPORT_ROOT/livecodebench.json"
fi

if [ ! -s "$REPORT_ROOT/livebench.json" ]; then
  log="$LOG_ROOT/score_livebench.out"
  mkdir -p "$ARTIFACT_ROOT/livebench_work"
  bwrap \
    --ro-bind /usr /usr --ro-bind /lib64 /lib64 --ro-bind /lib /lib --ro-bind /etc /etc \
    --ro-bind "$BASE_ENV_ROOT" /env --ro-bind "$SCORING_DEPS" /deps \
    --ro-bind "$RUNTIME_ROOT" /runtime \
    --ro-bind "$ARTIFACT_ROOT/site_sources/livebench-src" /scorer \
    --ro-bind "$ARTIFACT_ROOT/sandbox_manifests/livebench.json" /manifest.json \
    --ro-bind "$ARTIFACT_ROOT/full_generation/livebench" /generation \
    --ro-bind "$ARTIFACT_ROOT/site_data_core" /assessors \
    --ro-bind "$ARTIFACT_ROOT/nltk_data" /nltk \
    --bind "$ARTIFACT_ROOT/livebench_work" /work --bind "$SCORE_ROOT" /scores \
    --proc /proc --dev /dev --tmpfs /tmp \
    --unshare-net --unshare-pid --unshare-ipc --unshare-uts --die-with-parent \
    env HOME=/tmp TMPDIR=/tmp NLTK_DATA=/nltk SHOHIN_CODE_SANDBOX=1 \
      PYTHONPATH=/deps:/runtime/pipeline:/scorer \
    /env/bin/python3.13 /runtime/pipeline/eval_dense_public_livebench.py \
      --manifest /manifest.json --generation-root /generation \
      --assessor-root /assessors --output-root /scores --work-root /work \
      --livebench-root /scorer --livebench-commit "$LIVEBENCH_COMMIT" \
      --release 2024-11-25 >"$log" 2>"$LOG_ROOT/score_livebench.err"
  capture_final_json "$log" "$REPORT_ROOT/livebench.json"
fi

if [ ! -s "$REPORT_ROOT/longbench_pro.json" ]; then
  log="$LOG_ROOT/score_longbench_pro.out"
  srun --jobid="$ALLOCATION_JOB_ID" --overlap --ntasks=1 --cpus-per-task=16 --mem=120G \
    env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
      PYTHONPATH="$RUNTIME_ROOT/pipeline:$ARTIFACT_ROOT/site_sources/longcontext-full/LongBench-Pro" \
    "$PYTHON" "$RUNTIME_ROOT/pipeline/eval_dense_public_longbench_pro.py" \
      --manifest "$ARTIFACT_ROOT/manifests/longbench_pro.json" \
      --generation-root "$ARTIFACT_ROOT/full_generation/longbench_pro" \
      --assessor-root "$ARTIFACT_ROOT/site_data_core" --output-root "$SCORE_ROOT" \
      --longbench-root "$ARTIFACT_ROOT/site_sources/longcontext-full/LongBench-Pro" \
      --longbench-commit "$LONGBENCH_COMMIT" --embedding-model "$EMBEDDING_MODEL" \
      >"$log" 2>"$LOG_ROOT/score_longbench_pro.err"
  capture_final_json "$log" "$REPORT_ROOT/longbench_pro.json"
fi

"$PYTHON" "$RUNTIME_ROOT/pipeline/aggregate_dense_public_official_scores.py" \
  --manifest "$ARTIFACT_ROOT/manifests/all_official.json" \
  --score-root "$SCORE_ROOT" \
  --output "$ARTIFACT_ROOT/official_aggregate.json"
