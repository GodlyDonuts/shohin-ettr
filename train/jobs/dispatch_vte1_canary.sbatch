#!/bin/bash
#SBATCH --job-name=vte1-eval-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/vte1_eval_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/vte1_eval_dispatch_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 TRAIN_ROOT DATA_ROOT OUTPUT_ROOT \
  DATA_SHA256 DATA_REPORT_SHA256 RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
SHARD_COUNT=${SHARD_COUNT:-4}
EXCLUDE_NODES=${EXCLUDE_NODES:-evc29}
EXCLUDE=${EXCLUDE_NODES//:/,}
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test "$SHARD_COUNT" -ge 2
test ! -e "$OUTPUT_ROOT"
test ! -e "$RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
checkpoint=$TRAIN_ROOT/checkpoint_0000256.pt
fit_report=$TRAIN_ROOT/report.json
data=$DATA_ROOT/development.jsonl
data_report=$DATA_ROOT/report.json
for required_path in "$checkpoint" "$fit_report" "$data" "$data_report"; do
  test -f "$required_path"
done
test "$(sha256sum "$data" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$data_report" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" - "$fit_report" "$data" "$data_report" <<'PY'
import hashlib,json,sys
from pathlib import Path
fit,data,data_report=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3])
r=json.loads(fit.read_text()); d=json.loads(data_report.read_text())
if r.get("schema")!="shohin-hf-product-reasoning-training-v1" or r.get("status")!="complete": raise SystemExit("VTE1 fit differs")
if r.get("updates")!=256 or r.get("loss_mode")!="vte1_equivalence" or r.get("vte1_temperature")!=0.1 or r.get("trainable_parameters")!=2704896: raise SystemExit("VTE1 fit geometry differs")
if r.get("data_sha256")!="cc312363f880e9048622b57cb0cb609acaf92a1ab9ed0552ec8383ea20da1c33": raise SystemExit("VTE1 fit data differs")
if r.get("warm_start_sha256")!="07e08abe2480782afc77e35031d23bea71a737d019f307066af2bde786dd2ebd": raise SystemExit("VTE1 warm start differs")
if d.get("schema")!="shohin-kcr1-development-canary-report-v1" or d.get("status")!="complete" or d.get("holdout_used") is not False: raise SystemExit("VTE1 canary report differs")
if d.get("admitted_sources")!=522 or d.get("presentations")!=1566 or d.get("zero_truncation") is not True: raise SystemExit("VTE1 canary geometry differs")
if d.get("output",{}).get("sha256")!=hashlib.sha256(data.read_bytes()).hexdigest(): raise SystemExit("VTE1 canary hash differs")
PY

checkpoint_sha=$(sha256sum "$checkpoint" | cut -d' ' -f1)
data_sha=$(sha256sum "$data" | cut -d' ' -f1)
data_report_sha=$(sha256sum "$data_report" | cut -d' ' -f1)
mkdir -p "$OUTPUT_ROOT" "$(dirname "$RECEIPT")"
jobs=()
reports=()
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_root=$OUTPUT_ROOT/shards/shard_$shard
  exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
  exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
  exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
  exports+=",MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
  exports+=",ADAPTER_CHECKPOINT=$checkpoint,ADAPTER_CHECKPOINT_SHA256=$checkpoint_sha"
  exports+=",DATA=$data,DATA_SHA256=$data_sha"
  exports+=",DATA_REPORT=$data_report,DATA_REPORT_SHA256=$data_report_sha"
  exports+=",SHARD_INDEX=$shard,SHARD_COUNT=$SHARD_COUNT"
  exports+=",CANDIDATES_OUTPUT=$shard_root/candidates.jsonl,REPORT=$shard_root/report.json"
  job=$(sbatch --parsable --exclude="$EXCLUDE" --export="$exports" \
    "$RUNTIME/train/jobs/hf_kcr1_evaluate.sbatch")
  jobs+=("$job")
  reports+=("$shard_root/report.json")
done
dependency=$(IFS=:; echo "${jobs[*]}")
merge_args=()
for shard_report in "${reports[@]}"; do
  merge_args+=(--shard-report "$shard_report")
done
merge_job=$(sbatch --parsable --dependency=afterok:"$dependency" \
  --partition=normal --time=00:20:00 --cpus-per-task=1 --mem=8G \
  --wrap="set -euo pipefail; export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline; $PYTHON $RUNTIME/pipeline/merge_kcr1_evaluation_shards.py --data $data ${merge_args[*]} --candidates-output $OUTPUT_ROOT/candidates.jsonl --report $OUTPUT_ROOT/report.json; chmod a-w $OUTPUT_ROOT/candidates.jsonl $OUTPUT_ROOT/report.json")
temporary=$RECEIPT.partial.$$
"$PYTHON" - "$temporary" "$checkpoint_sha" "$merge_job" "${jobs[*]}" <<'PY'
import json,sys
from pathlib import Path
path,checkpoint_sha,merge_job,jobs=sys.argv[1:]
Path(path).write_text(json.dumps({"checkpoint_sha256":checkpoint_sha,"evaluation_jobs":jobs.split(),"holdout_opened":False,"merge_job":merge_job,"status":"submitted"},sort_keys=True)+"\n")
PY
mv "$temporary" "$RECEIPT"
chmod a-w "$RECEIPT"
cat "$RECEIPT"
