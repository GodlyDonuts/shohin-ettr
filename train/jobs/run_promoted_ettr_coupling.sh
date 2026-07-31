#!/bin/bash
# Continue a causally qualified ETTR arm across one or more live H100
# reservations. This wrapper derives every trainable input from the sealed
# training and source-deleted evaluation reports before invoking the generic
# federated launcher.

set -euo pipefail

SEED=${SEED:?set the promoted seed (1 or 2)}
INITIAL_TRAIN_DIR=${INITIAL_TRAIN_DIR:?set the completed coupling output}
EVALUATION_REPORT=${EVALUATION_REPORT:?set its source-deleted evaluation}
ALLOCATION_GROUPS=${ALLOCATION_GROUPS:?set JOB@NODE@GPUS groups}
CODE_ROOT=${CODE_ROOT:?set the immutable distributed source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set that source root commit}
OUTDIR=${OUTDIR:?set a fresh absolute output directory}
UPDATES=${UPDATES:-100}
START_POSITION=${START_POSITION:-}
WARMUP_UPDATES=${WARMUP_UPDATES:-0}
RAMP_UPDATES=${RAMP_UPDATES:-1}
LOG_EVERY=${LOG_EVERY:-10}
CPUS_PER_GPU=${CPUS_PER_GPU:-2}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
ROOT=/lustre/fs1/home/sa305415/shohin
RELEASE="$ROOT/artifacts/ettr_il_v3/training-e5f3705-packet-v2"

if [[ "$SEED" != 1 && "$SEED" != 2 ]]; then
  echo "promoted seed differs" >&2
  exit 2
fi
for path in \
  "$INITIAL_TRAIN_DIR" \
  "$EVALUATION_REPORT" \
  "$CODE_ROOT" \
  "$OUTDIR" \
  "$PYTHON_ROOT"; do
  if [[ "$path" != /* ]]; then
    echo "promotion paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "promotion source commit differs" >&2
  exit 2
fi
test -f "$INITIAL_TRAIN_DIR/report.json"
test -f "$INITIAL_TRAIN_DIR/coupling-contract.json"
test -f "$INITIAL_TRAIN_DIR/compiler-final.safetensors"
test -f "$INITIAL_TRAIN_DIR/reactor-final.safetensors"
test -f "$INITIAL_TRAIN_DIR/reader-final.safetensors"
test -f "$EVALUATION_REPORT"
test -x "$CODE_ROOT/train/jobs/run_federated_ettr_progressive_coupling.sh"

readarray -t admitted < <(
  "$PYTHON_ROOT/bin/python" - \
    "$INITIAL_TRAIN_DIR" "$EVALUATION_REPORT" "$SEED" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

train_dir = Path(sys.argv[1])
evaluation_path = Path(sys.argv[2])
expected_seed = int(sys.argv[3])

report = json.loads((train_dir / "report.json").read_text(encoding="ascii"))
contract = json.loads(
    (train_dir / "coupling-contract.json").read_text(encoding="ascii")
)
evaluation = json.loads(evaluation_path.read_text(encoding="ascii"))
if report["architecture_seed"] != contract["architecture_seed"]:
    raise SystemExit("training architecture seed differs")
if report["data_seed"] != contract["data_seed"]:
    raise SystemExit("training data seed differs")
if not evaluation["gates"]["strict_learning_signal"]:
    raise SystemExit("source-deleted causal promotion gate is false")

components = report["final_component_sha256"]
evaluated = evaluation["arms"]["component_assembly"]["component_sha256"]
if components != evaluated:
    raise SystemExit("evaluated component identity differs")
for name in ("compiler", "reactor", "reader"):
    path = train_dir / f"{name}-final.safetensors"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != components[name]:
        raise SystemExit(f"{name} component hash differs")

seed_contracts = {
    1: (2026072801, 2026072802),
    2: (2026072811, 2026072812),
}
if (report["architecture_seed"], report["data_seed"]) != seed_contracts[
    expected_seed
]:
    raise SystemExit("promoted seed identity differs")

learning_rates = contract["component_learning_rates"]
loss_weights = contract["high_level_loss_weights"]
if loss_weights["compiler_delta"] != loss_weights["reactor_delta"]:
    raise SystemExit("counterfactual delta weights differ")
coupling = contract["coupling"]
values = (
    components["compiler"],
    components["reactor"],
    components["reader"],
    str(report["architecture_seed"]),
    str(report["data_seed"]),
    str(coupling["seed"] + 1),
    str(learning_rates["compiler"]),
    str(learning_rates["reactor"]),
    str(learning_rates["reader"]),
    str(loss_weights["compiler_delta"]),
    str(coupling["exact_anchor_steps_per_update"]),
    str(coupling["credit_horizon"]),
    str(contract["weight_decay"]),
    str(contract["gradient_clip"]),
    str(report["start_position"]),
    str(report["updates"]),
    str(report.get("world_size", 1)),
)
print("\n".join(values))
PY
)
if (( ${#admitted[@]} != 17 )); then
  echo "promotion report admission differs" >&2
  exit 2
fi

IFS=',' read -r -a group_specs <<< "$ALLOCATION_GROUPS"
world_size=0
for spec in "${group_specs[@]}"; do
  IFS='@' read -r _ _ gpus extra <<< "$spec"
  if [[ -n "${extra:-}" || ! "$gpus" =~ ^[0-9]+$ || "$gpus" == 0 ]]; then
    echo "promotion allocation group differs: $spec" >&2
    exit 2
  fi
  world_size=$((world_size + gpus))
done
if (( world_size < 2 || world_size > 20 )); then
  echo "promotion world size differs" >&2
  exit 2
fi

if [[ -z "$START_POSITION" ]]; then
  prior_end=$((
    admitted[14] + admitted[15] * admitted[16]
  ))
  START_POSITION=$((
    (prior_end + world_size - 1) / world_size * world_size
  ))
fi

if [[ "$SEED" == 1 ]]; then
  CHECKPOINT="$ROOT/train/ettr_v3_packet_v2_world2_seed1_hard_adjoint1_cap4_warm333_u2000/train/checkpoint-update-0002000.pt"
  CHECKPOINT_SHA256=f9f5d94493dff1815da7d4d1f9abf44617adbac0da2871fe361a53f3d350297f
  RUN_CONTRACT="$ROOT/train/ettr_v3_packet_v2_world2_seed1_hard_adjoint1_cap4_warm333_u2000/train/run-contract.json"
  RUN_CONTRACT_SHA256=fdfdb2eafd261e014d6d0e5ff5ca11e273ca9f889bd9661ff04ecdb6c2424464
else
  CHECKPOINT="$ROOT/train/ettr_v3_packet_v2_world2_seed2_hard_adjoint1_capuncapped_warm333_u2000/train/checkpoint-update-0002000.pt"
  CHECKPOINT_SHA256=f14c041568c31d10d5ab427f22c1a07ac52c3d0a12a20fbf4f05719607687d4f
  RUN_CONTRACT="$ROOT/train/ettr_v3_packet_v2_world2_seed2_hard_adjoint1_capuncapped_warm333_u2000/train/run-contract.json"
  RUN_CONTRACT_SHA256=1191c2f97ec6a66b9547f4937eadd074017c592dcb0e35a04ea07af5789dfb16
fi

export ALLOCATION_GROUPS CODE_ROOT SOURCE_COMMIT
export RELEASE_ROOT="$RELEASE/release"
export RELEASE_SHA256=8c6d7d80603e29e92f14027929ae4ef7e848094a44a154ef37b2bcbf726d4462
export DATA_ROOT="$RELEASE/data"
export TOKENIZER="$RELEASE/tokenizer.json"
export PROTECTED_CHECKPOINT="$ROOT/train/flagship_out/ckpt_0300000.pt"
export CHECKPOINT CHECKPOINT_SHA256 RUN_CONTRACT RUN_CONTRACT_SHA256
export INITIAL_COMPILER="$INITIAL_TRAIN_DIR/compiler-final.safetensors"
export INITIAL_COMPILER_SHA256="${admitted[0]}"
export INITIAL_REACTOR="$INITIAL_TRAIN_DIR/reactor-final.safetensors"
export INITIAL_REACTOR_SHA256="${admitted[1]}"
export INITIAL_READER="$INITIAL_TRAIN_DIR/reader-final.safetensors"
export INITIAL_READER_SHA256="${admitted[2]}"
export OUTDIR
export ARCHITECTURE_SEED="${admitted[3]}"
export DATA_SEED="${admitted[4]}"
export COUPLING_SEED="${admitted[5]}"
export COMPILER_LEARNING_RATE="${admitted[6]}"
export REACTOR_LEARNING_RATE="${admitted[7]}"
export READER_LEARNING_RATE="${admitted[8]}"
export COUNTERFACTUAL_DELTA_WEIGHT="${admitted[9]}"
export EXACT_ANCHOR_STEPS="${admitted[10]}"
export CREDIT_HORIZON="${admitted[11]}"
export WEIGHT_DECAY="${admitted[12]}"
export GRADIENT_CLIP="${admitted[13]}"
export UPDATES START_POSITION WARMUP_UPDATES RAMP_UPDATES LOG_EVERY
export CPUS_PER_GPU PYTHON_ROOT

exec "$CODE_ROOT/train/jobs/run_federated_ettr_progressive_coupling.sh"
