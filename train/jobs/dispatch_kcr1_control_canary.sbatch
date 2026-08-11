#!/bin/bash
#SBATCH --job-name=kcr1-ctl-eval-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/kcr1_ctl_eval_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/kcr1_ctl_eval_dispatch_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 TRAIN_ROOT DATA_ROOT OUTPUT_ROOT \
  DATA_SHA256 DATA_REPORT_SHA256 RECEIPT CONTROL; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
case "$CONTROL" in
  action_permuted)
    EXPECTED_TRAIN_SHA=6fb14827dd3b7b8dfcfeda04e2bd2ae82c03138d313dcc184e21b150a79a5fee
    EXPECTED_MASK=false
    ;;
  draft_hidden)
    EXPECTED_TRAIN_SHA=3e63c4e248a32c9f4008a1c245ea6e18bcb2f8497058b6c0c237ca5a719863b9
    EXPECTED_MASK=true
    ;;
  *) echo "unsupported KCR1 canary control" >&2; exit 4 ;;
esac
SHARD_COUNT=${SHARD_COUNT:-4}
EXCLUDE_NODES=${EXCLUDE_NODES:-evc26:evc29:evc31:evc33:evc38:evc46}
EXCLUDE=${EXCLUDE_NODES//:/,}
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test "$SHARD_COUNT" -ge 2
test ! -e "$OUTPUT_ROOT"
test ! -e "$RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
checkpoint=$TRAIN_ROOT/checkpoint_0000512.pt
fit_report=$TRAIN_ROOT/report.json
data=$DATA_ROOT/development.jsonl
data_report=$DATA_ROOT/report.json
for required_path in "$checkpoint" "$fit_report" "$data" "$data_report"; do
  test -f "$required_path"
done
test "$(sha256sum "$data" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$data_report" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" - "$fit_report" "$data" "$data_report" \
  "$EXPECTED_TRAIN_SHA" "$EXPECTED_MASK" <<'PY'
import hashlib,json,sys
from pathlib import Path
fit,data,data_report,expected_train_sha,expected_mask=sys.argv[1:]
r=json.loads(Path(fit).read_text()); d=json.loads(Path(data_report).read_text())
if r.get("schema")!="shohin-hf-product-reasoning-training-v1" or r.get("status")!="complete": raise SystemExit("KCR1 control fit differs")
if r.get("updates")!=512 or r.get("loss_mode")!="kcr1_action_payload" or r.get("trainable_parameters")!=2704896: raise SystemExit("KCR1 control geometry differs")
if r.get("data_sha256")!=expected_train_sha or r.get("mask_internal_draft")!=(expected_mask=="true"): raise SystemExit("KCR1 control intervention differs")
if r.get("warm_start_sha256")!="854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971": raise SystemExit("KCR1 warm start differs")
if d.get("schema")!="shohin-kcr1-development-canary-report-v1" or d.get("status")!="complete" or d.get("holdout_used") is not False: raise SystemExit("KCR1 canary report differs")
if d.get("admitted_sources")!=522 or d.get("presentations")!=1566 or d.get("zero_truncation") is not True: raise SystemExit("KCR1 canary geometry differs")
if d.get("output",{}).get("sha256")!=hashlib.sha256(Path(data).read_bytes()).hexdigest(): raise SystemExit("KCR1 canary hash differs")
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
"$PYTHON" - "$temporary" "$checkpoint_sha" "$merge_job" "$CONTROL" \
  "${jobs[*]}" <<'PY'
import json,sys
from pathlib import Path
path,checkpoint_sha,merge_job,control,jobs=sys.argv[1:]
Path(path).write_text(json.dumps({"checkpoint_sha256":checkpoint_sha,"control":control,"evaluation_jobs":jobs.split(),"holdout_opened":False,"merge_job":merge_job,"status":"submitted"},sort_keys=True)+"\n")
PY
mv "$temporary" "$RECEIPT"
chmod a-w "$RECEIPT"
cat "$RECEIPT"
