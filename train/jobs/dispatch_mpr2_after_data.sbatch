#!/bin/bash
#SBATCH --job-name=mpr2-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mpr2_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mpr2_dispatch_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA_ROOT OUTPUT_ROOT OWNER_CANDIDATES OWNER_REPORT RECEIPT; do test -n "${!variable:-}" || exit 2; done
EXCLUDE=${EXCLUDE:-evc29,evc33,evc38,evc46}; SHARDS=8
COMPARISON=${COMPARISON:-mpr2}
case "$COMPARISON" in
  mpr2) COMPARE_JOB=$RUNTIME/pipeline/jobs/compare_mpr2_development.sbatch ;;
  mpr3) COMPARE_JOB=$RUNTIME/pipeline/jobs/compare_mpr3_development.sbatch ;;
  *) echo "unknown MPR comparison: $COMPARISON" >&2; exit 3 ;;
esac
test ! -e "$OUTPUT_ROOT" && test ! -e "$RECEIPT"; test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"; (cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
mkdir -p "$OUTPUT_ROOT" "$(dirname "$RECEIPT")"
DATA_REPORT=$DATA_ROOT/report.json; DATA_REPORT_SHA=$(sha256sum "$DATA_REPORT"|cut -d' ' -f1)
ALIGNED=$DATA_ROOT/train_aligned.jsonl; SHUFFLED=$DATA_ROOT/train_shuffled.jsonl
ALIGNED_SHA=$(sha256sum "$ALIGNED"|cut -d' ' -f1); SHUFFLED_SHA=$(sha256sum "$SHUFFLED"|cut -d' ' -f1)
submit(){ local raw job; raw=$(sbatch --parsable "$@"); job=${raw%%;*}; [[ $job =~ ^[0-9]+$ ]] || exit 5; printf '%s\n' "$job"; }
common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA"
mechanics=$(submit --exclude="$EXCLUDE" --export="$common,DATA=$ALIGNED,DATA_SHA256=$ALIGNED_SHA,OUTPUT=$OUTPUT_ROOT/mechanics_u1,ARM=aligned,DRAFT_CONTROL=normal,UPDATES=1,CHECKPOINT_INTERVAL=1" "$RUNTIME/train/jobs/train_mpr2_shared_residual.sbatch")
aligned_fit=$(submit --exclude="$EXCLUDE" --dependency=afterok:"$mechanics" --export="$common,DATA=$ALIGNED,DATA_SHA256=$ALIGNED_SHA,OUTPUT=$OUTPUT_ROOT/aligned_fit,ARM=aligned,DRAFT_CONTROL=normal" "$RUNTIME/train/jobs/train_mpr2_shared_residual.sbatch")
shuffled_fit=$(submit --exclude="$EXCLUDE" --dependency=afterok:"$mechanics" --export="$common,DATA=$SHUFFLED,DATA_SHA256=$SHUFFLED_SHA,OUTPUT=$OUTPUT_ROOT/shuffled_fit,ARM=shuffled,DRAFT_CONTROL=normal" "$RUNTIME/train/jobs/train_mpr2_shared_residual.sbatch")
hidden_fit=$(submit --exclude="$EXCLUDE" --dependency=afterok:"$mechanics" --export="$common,DATA=$ALIGNED,DATA_SHA256=$ALIGNED_SHA,OUTPUT=$OUTPUT_ROOT/hidden_fit,ARM=hidden,DRAFT_CONTROL=draft_unavailable" "$RUNTIME/train/jobs/train_mpr2_shared_residual.sbatch")
declare -A FIT_JOBS=([aligned]="$aligned_fit" [shuffled]="$shuffled_fit" [hidden]="$hidden_fit")
merges=(); shard_jobs=()
for arm in aligned shuffled hidden; do
  data_arm=$arm; [[ $arm == hidden ]] && data_arm=aligned
  data=$DATA_ROOT/development_${data_arm}.jsonl; data_report=$DATA_ROOT/development_${data_arm}_report.json
  data_sha=$(sha256sum "$data"|cut -d' ' -f1); report_sha=$(sha256sum "$data_report"|cut -d' ' -f1)
  root=$OUTPUT_ROOT/development/$arm; mkdir -p "$root/shards"; jobs=(); reports=()
  for index in $(seq 0 7); do suffix=$(printf '%02d' "$index"); checkpoint=$OUTPUT_ROOT/${arm}_fit/checkpoint_0000256.pt; report=$root/shards/report_$suffix.json
    job=$(submit --exclude="$EXCLUDE" --dependency=afterok:"${FIT_JOBS[$arm]}" --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_LOADER=causal,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256,DATA=$data,DATA_SHA256=$data_sha,DATA_REPORT=$data_report,DATA_REPORT_SHA256=$report_sha,SPLIT=development,STAGE_MODEL=false,ADAPTER_CHECKPOINT=$checkpoint,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS,CANDIDATES_OUTPUT=$root/shards/candidates_$suffix.jsonl,REPORT=$report" "$RUNTIME/train/jobs/hf_mpr1_evaluate_reviser.sbatch")
    jobs+=("$job"); shard_jobs+=("$job"); reports+=("$report")
  done
  dependency=$(IFS=:; echo "${jobs[*]}"); report_csv=$(IFS=:; echo "${reports[*]}")
  merge=$(submit --dependency=afterok:"$dependency" --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,EVALUATOR=idr1,DATA=$data,DATA_SHA256=$data_sha,SPLIT=development,SHARD_REPORTS=$report_csv,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json" "$RUNTIME/pipeline/jobs/merge_temporal_evaluation_shards.sbatch")
  merges+=("$merge")
done
semantic=$OUTPUT_ROOT/development/semantic_attribution.json
semantic_job=$(submit --dependency=afterok:"${merges[0]}" --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,UNCHANGED=$OWNER_CANDIDATES,TREATMENT=$OUTPUT_ROOT/development/aligned/candidates.jsonl,DEVELOPMENT_DATA=$DATA_ROOT/development_aligned.jsonl,OUTPUT=$semantic" "$RUNTIME/pipeline/jobs/analyze_ecr1_semantic_repairs.sbatch")
merge_dependency=$(IFS=:; echo "${merges[*]}"); comparison=$OUTPUT_ROOT/development/comparison.json
compare_job=$(submit --dependency=afterok:"$merge_dependency":"$semantic_job" --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,ALIGNED_REPORT=$OUTPUT_ROOT/development/aligned/report.json,SHUFFLED_REPORT=$OUTPUT_ROOT/development/shuffled/report.json,HIDDEN_REPORT=$OUTPUT_ROOT/development/hidden/report.json,OWNER_REPORT=$OWNER_REPORT,ALIGNED_FIT=$OUTPUT_ROOT/aligned_fit/report.json,SHUFFLED_FIT=$OUTPUT_ROOT/shuffled_fit/report.json,HIDDEN_FIT=$OUTPUT_ROOT/hidden_fit/report.json,DATA_REPORT=$DATA_REPORT,SEMANTIC_ATTRIBUTION=$semantic,OUTPUT=$comparison" "$COMPARE_JOB")
python3 - "$RECEIPT" "$mechanics" "$aligned_fit" "$shuffled_fit" "$hidden_fit" "$semantic_job" "$compare_job" "${shard_jobs[*]}" "${merges[*]}" <<'PY'
import json,os,sys
path,*v=sys.argv[1:]; p={"status":"submitted","mechanics_job":v[0],"aligned_fit_job":v[1],"shuffled_fit_job":v[2],"hidden_fit_job":v[3],"semantic_job":v[4],"comparison_job":v[5],"shard_jobs":v[6].split(),"merge_jobs":v[7].split(),"comparison":os.environ.get("COMPARISON","mpr2")}; t=f"{path}.partial.{os.getpid()}"; open(t,"x").write(json.dumps(p,sort_keys=True)+"\n"); os.replace(t,path)
PY
chmod a-w "$RECEIPT"; cat "$RECEIPT"
