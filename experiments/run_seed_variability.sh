#!/usr/bin/env bash
# Section 4.2 - training-seed variability (Section 4.2 of the paper).
#
# Trains, evaluates and then DELETES each checkpoint before starting the next, so
# peak disk stays at roughly one checkpoint per concurrent lane rather than the
# ~360 GB the full set would need. Full fine-tuning of an 8B model with AdamW does
# not fit on one 48 GB card: the optimiser state is about four times the parameter
# bytes and is allocated on the card holding each parameter, so each lane is given
# a comma-separated list of three devices.
#
# Usage: run_seed_variability.sh <gpu-list> <open|closed> <backbone:dataset:seed> ...
#   e.g. run_seed_variability.sh 0,1,2 open qwen3:medquad:1 qwen3:medquad:2
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
PIPE=../scripts/pipeline
OUT=${MEDLM_OUTPUT_DIR:-../outputs}
GPUS=$1; KIND=$2; shift 2

declare -A BASE=(
  [llama31]="NousResearch/Meta-Llama-3.1-8B-Instruct"
  [qwen3]="Qwen/Qwen3-8B"
  [deepseek]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

for spec in "$@"; do
  IFS=: read -r BB DS SEED <<< "$spec"
  B=${BASE[$BB]:-}
  [ -z "$B" ] && { echo "!! unknown backbone $BB, skipping $spec"; continue; }
  if [ "$KIND" = "closed" ]; then
    CK="$OUT/runs/ckptcf_${DS}_${BB}_seed${SEED}"
    RES="$OUT/runs/closed_${DS}_${BB}ft_s${SEED}/metrics.json"
    if [ ! -f "$RES" ]; then
      echo "=== TRAIN $spec $(date +%H:%M) ==="
      CUDA_VISIBLE_DEVICES=$GPUS $PY $PIPE/closedform_phase_b_adaptation.py \
        --dataset "$DS" --base "$B" --label "$BB" --seed "$SEED"
      echo "=== EVAL  $spec $(date +%H:%M) ==="
      CUDA_VISIBLE_DEVICES=$GPUS $PY $PIPE/closedform_phase_c_inference_and_selection.py \
        --dataset "$DS" --generator "$CK" --generator_label "${BB}ft_s${SEED}" \
        --conditions adapted,adapted_verifier --n_test 500 --seed 42
    fi
  else
    CK="$OUT/phase_b/${DS}_${BB}_seed${SEED}"
    RES="$OUT/phase_c/${DS}_${BB}T${SEED}/metrics.json"
    if [ ! -f "$RES" ]; then
      echo "=== TRAIN $spec $(date +%H:%M) ==="
      CUDA_VISIBLE_DEVICES=$GPUS $PY $PIPE/phase_b_prompt_and_adaptation.py \
        --dataset "$DS" --base "$B" --label "$BB" --seed "$SEED" \
        --n_train 500 --n_val 200 --epochs 3 --pool 2000
      echo "=== EVAL  $spec $(date +%H:%M) ==="
      CUDA_VISIBLE_DEVICES=$GPUS $PY $PIPE/phase_c_inference_and_selection.py \
        --dataset "$DS" --generator "$CK" --generator_label "${BB}T${SEED}" \
        --verifier "${MEDLM_VERIFIER:?set MEDLM_VERIFIER}" \
        --n_test 300 --n_train 2000 --prompt_sets norag,rag --n_cand 5 \
        --batch_size 8 --seed 42 --truncation_side left
    fi
  fi
  echo "=== DELETE checkpoint $CK ==="   # freed as soon as its evaluation is on disk
  rm -rf "$CK"
done
echo "LANE DONE $KIND $(date +%H:%M)"
