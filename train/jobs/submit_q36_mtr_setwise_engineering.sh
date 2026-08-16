#!/bin/bash
# Submit the contextual three-owner Q36 selector behind existing exact score jobs.
# This engineering fan-out does not modify or duplicate any running trajectory
# generation or matched-control evaluation.

set -euo pipefail

SBATCH=/apps/slurm/current/bin/sbatch
RUN=/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_d9ff7f7_r1
RUNTIME=/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_runtime_5019d84_r1
RUNTIME_MANIFEST_SHA256=18ee66ea9924fbb8711dbdf403e885de592c986730d3cc5b733062d47cde88fb
SOURCE_COMMIT=d9ff7f7d79b953179bf90510731c0bcd2f02e722
PYTHON=/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python
MODEL_ROOT=/lustre/fs1/home/sa305415/shohin/artifacts/external/qwen3.6-35b-a3b-995ad96e
MODEL_MANIFEST=$MODEL_ROOT/SHA256SUMS
MODEL_MANIFEST_SHA256=06c9d8d8419244f2d001cb351e164f356718d9d77138e898b13afee35856f56e
MODEL_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0
MODEL_CONFIG_SHA256=93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99
PHASE_AUTHORIZATION=/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_prepare_d9ff7f7_r1/phase_authorization.json
PHASE_AUTHORIZATION_SHA256=8dfde3a9ff346a7f10c687bcfd95b95de11e25c79f653f53d0eaee3f736e485f
ENVIRONMENT_RECEIPT=$RUN/engineering_preview/runtime_receipts/5019d84.json
ENVIRONMENT_RECEIPT_SHA256=030e01c8d31e871615c6a95e53b4f8f1658cfede997f3dfeea979a63b8bb3c75
ENVIRONMENT_TREE_SHA256=6c3311032bc4efb065222378e053e1cc15266b37bd868aee2bc05aa94f8ebf9c
TRAIN_SOURCE=/lustre/fs1/home/sa305415/shohin/artifacts/pcf17_ministral_037f122_r1/prepared/sources/train_sources.jsonl
DEVELOPMENT_SOURCE=/lustre/fs1/home/sa305415/shohin/artifacts/pcf17_ministral_037f122_r1/prepared/sources/development_sources.jsonl
ASSESSOR_BOARD=/lustre/fs1/home/sa305415/shohin/artifacts/pcf17_ministral_037f122_r1/custodian/confirmation_assessors.jsonl
ALIGNED_CHECKPOINT=$RUN/engineering_preview/owner_71ac37c1_salvage/roles/aligned/checkpoint_0000256.pt
ROOT=$RUN/engineering_preview/setwise_semantic_commit

[[ -x "$SBATCH" && -d "$RUN" && -d "$RUNTIME" ]] || { printf 'setwise roots differ\n' >&2; exit 2; }
[[ "$(sha256sum "$RUNTIME/SHA256SUMS" | awk '{print $1}')" == "$RUNTIME_MANIFEST_SHA256" ]] || { printf 'setwise runtime differs\n' >&2; exit 2; }
for path in "$ROOT/data" "$ROOT/model" "$ROOT/selected"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { printf 'setwise output exists: %s\n' "$path" >&2; exit 2; }
done

current_candidates=()
owner71_candidates=()
owner8_candidates=()
current_train_scores=()
owner71_train_scores=()
owner8_train_scores=()
current_development_scores=()
owner71_development_scores=()
owner8_development_scores=()
for index in {00..15}; do
  current_candidates+=("$RUN/drafts/shard_$index/candidates.jsonl")
  owner71_candidates+=("/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_71ac37c1_r1/drafts/shard_$index/candidates.jsonl")
  if [[ "$index" == 00 ]]; then
    owner8_candidates+=("$RUN/engineering_preview/owner_8cb345e4_full_shard00/shard_00/candidates.jsonl")
  else
    owner8_candidates+=("/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_8cb345e4_r1/drafts/shard_$index/candidates.jsonl")
  fi
  current_train_scores+=("$RUN/engineering_preview/owner_semantic_commit/scores/current_train/shard_$index.json")
  owner71_train_scores+=("$RUN/engineering_preview/owner_semantic_commit/scores/owner71_train/shard_$index.json")
  owner8_train_scores+=("$RUN/engineering_preview/owner_semantic_commit/scores/owner8_train/shard_$index.json")
  current_development_scores+=("$RUN/engineering_preview/shard_${index}_development_score.json")
  owner71_development_scores+=("$RUN/engineering_preview/owner_71ac37c1_all_shards/shard_${index}_development_score.json")
  if [[ "$index" == 00 ]]; then
    owner8_development_scores+=("$RUN/engineering_preview/owner_semantic_commit/scores/owner8_development/shard_00.json")
  else
    owner8_development_scores+=("$RUN/engineering_preview/owner_8cb345e4_shards01_15/shard_${index}_development_score.json")
  fi
done
join_colon() { local IFS=:; printf '%s' "$*"; }

builder=$RUNTIME/pipeline/build_q36_mtr_setwise_commit_rows.py
builder_job=$(
  "$SBATCH" --parsable --dependency=afterok:758481:758482:758508:758509:758511:758512 \
    --chdir="$RUN" --job-name=q36-setwise-data \
    --export=ALL,PYTHON="$PYTHON",RUNTIME="$RUNTIME",BUILDER_SCRIPT="$builder",BUILDER_SCRIPT_SHA256="$(sha256sum "$builder" | awk '{print $1}')",TRAIN_SOURCE="$TRAIN_SOURCE",DEVELOPMENT_SOURCE="$DEVELOPMENT_SOURCE",CURRENT_CANDIDATES="$(join_colon "${current_candidates[@]}")",OWNER71_CANDIDATES="$(join_colon "${owner71_candidates[@]}")",OWNER8_CANDIDATES="$(join_colon "${owner8_candidates[@]}")",CURRENT_TRAIN_SCORES="$(join_colon "${current_train_scores[@]}")",OWNER71_TRAIN_SCORES="$(join_colon "${owner71_train_scores[@]}")",OWNER8_TRAIN_SCORES="$(join_colon "${owner8_train_scores[@]}")",CURRENT_DEVELOPMENT_SCORES="$(join_colon "${current_development_scores[@]}")",OWNER71_DEVELOPMENT_SCORES="$(join_colon "${owner71_development_scores[@]}")",OWNER8_DEVELOPMENT_SCORES="$(join_colon "${owner8_development_scores[@]}")",TRAINING_OUTPUT="$ROOT/data/training_rows.jsonl",TRAINING_REPORT="$ROOT/data/training_report.json",DEVELOPMENT_OUTPUT="$ROOT/data/development_rows.jsonl",DEVELOPMENT_REPORT="$ROOT/data/development_report.json" \
    "$RUNTIME/pipeline/jobs/q36_mtr_build_multi_owner_commit_pairs.sbatch"
)

trainer=$RUNTIME/train/hf_q36_mtr_train_setwise_commit.py
trainer_job=$(
  "$SBATCH" --parsable --dependency=afterok:"$builder_job" --chdir="$RUN" \
    --job-name=q36-setwise-train \
    --export=ALL,RUNTIME="$RUNTIME",RUNTIME_MANIFEST_SHA256="$RUNTIME_MANIFEST_SHA256",SOURCE_COMMIT="$SOURCE_COMMIT",PYTHON="$PYTHON",RUN_ID=q36-mtr-d9ff7f7-r1,MODEL_ROOT="$MODEL_ROOT",MODEL_MANIFEST="$MODEL_MANIFEST",MODEL_MANIFEST_SHA256="$MODEL_MANIFEST_SHA256",MODEL_REVISION="$MODEL_REVISION",MODEL_CONFIG_SHA256="$MODEL_CONFIG_SHA256",PHASE_AUTHORIZATION="$PHASE_AUTHORIZATION",PHASE_AUTHORIZATION_SHA256="$PHASE_AUTHORIZATION_SHA256",ENVIRONMENT_RECEIPT="$ENVIRONMENT_RECEIPT",ENVIRONMENT_RECEIPT_SHA256="$ENVIRONMENT_RECEIPT_SHA256",ENVIRONMENT_TREE_SHA256="$ENVIRONMENT_TREE_SHA256",TRAIN_SCRIPT="$trainer",TRAIN_SCRIPT_SHA256="$(sha256sum "$trainer" | awk '{print $1}')",ALIGNED_CHECKPOINT="$ALIGNED_CHECKPOINT",ROWS="$ROOT/data/training_rows.jsonl",ROWS_REPORT="$ROOT/data/training_report.json",OUTPUT="$ROOT/model" \
    "$RUNTIME/train/jobs/q36_mtr_train_setwise_commit.sbatch"
)

apply_script=$RUNTIME/train/hf_q36_mtr_apply_multi_owner_commit.py
apply_job=$(
  "$SBATCH" --parsable --dependency=afterok:"$trainer_job" --chdir="$RUN" \
    --job-name=q36-setwise-apply \
    --export=ALL,RUNTIME="$RUNTIME",RUNTIME_MANIFEST_SHA256="$RUNTIME_MANIFEST_SHA256",SOURCE_COMMIT="$SOURCE_COMMIT",PYTHON="$PYTHON",RUN_ID=q36-mtr-d9ff7f7-r1,MODEL_ROOT="$MODEL_ROOT",MODEL_MANIFEST="$MODEL_MANIFEST",MODEL_MANIFEST_SHA256="$MODEL_MANIFEST_SHA256",MODEL_REVISION="$MODEL_REVISION",MODEL_CONFIG_SHA256="$MODEL_CONFIG_SHA256",PHASE_AUTHORIZATION="$PHASE_AUTHORIZATION",PHASE_AUTHORIZATION_SHA256="$PHASE_AUTHORIZATION_SHA256",ENVIRONMENT_RECEIPT="$ENVIRONMENT_RECEIPT",ENVIRONMENT_RECEIPT_SHA256="$ENVIRONMENT_RECEIPT_SHA256",ENVIRONMENT_TREE_SHA256="$ENVIRONMENT_TREE_SHA256",APPLY_SCRIPT="$apply_script",APPLY_SCRIPT_SHA256="$(sha256sum "$apply_script" | awk '{print $1}')",ALIGNED_CHECKPOINT="$ALIGNED_CHECKPOINT",COMMIT_CHECKPOINT="$ROOT/model/setwise_commit.pt",HEAD_TYPE=setwise,DEVELOPMENT_SOURCE="$DEVELOPMENT_SOURCE",CURRENT_CANDIDATES="$(join_colon "${current_candidates[@]}")",OWNER71_CANDIDATES="$(join_colon "${owner71_candidates[@]}")",OWNER8_CANDIDATES="$(join_colon "${owner8_candidates[@]}")",OUTPUT="$ROOT/selected/candidates.jsonl",SELECTIONS="$ROOT/selected/selections.jsonl",REPORT="$ROOT/selected/application_report.json" \
    "$RUNTIME/train/jobs/q36_mtr_apply_multi_owner_commit.sbatch"
)

score_job=$(
  "$SBATCH" --parsable --dependency=afterok:"$apply_job" --chdir="$RUN" \
    --job-name=q36-setwise-score \
    --export=PYTHON="$PYTHON",RUNTIME="$RUNTIME",PREVIEW_SCRIPT="$RUN/engineering_preview/score_q36_mtr_draft_preview.py",PREVIEW_SCRIPT_SHA256=e1d97827ce1bb481b2e3dbc4c1cbbcc0adf0c43103958508736bbf2234e4fcbc,CANDIDATES="$ROOT/selected/candidates.jsonl",ASSESSOR_BOARD="$ASSESSOR_BOARD",OUTPUT="$ROOT/selected/development_score.json" \
    "$RUN/engineering_preview/q36_mtr_score_draft_preview.sbatch"
)

printf 'builder=%s\ntrainer=%s\napply=%s\nscore=%s\n' "$builder_job" "$trainer_job" "$apply_job" "$score_job"
