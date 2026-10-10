#!/usr/bin/env bash
# Section 4.3 - train on the source/topic-disjoint repartition at three seeds and
# evaluate retrieval against no retrieval on the disjoint test split. The
# checkpoint is deleted after each evaluation.
#
# --max_memory_map takes a per-visible-device cap in GiB. A uniform cap is wrong
# on a shared node: cards differ in how much they actually have free, and a cap
# above a card's real headroom only relocates the failure. Measure free memory and
# use roughly free/5.5 per card.
#
# Usage: run_group_disjoint_training.sh <gpu-list> <cap-map> <seed> ...
#   e.g. run_group_disjoint_training.sh 4,5,6 8,8,8 42 1 2
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
PIPE=../scripts/pipeline
OUT=${MEDLM_OUTPUT_DIR:-../outputs}
GPUS=$1; CAP=$2; shift 2
B=${MEDLM_LLAMA:-NousResearch/Meta-Llama-3.1-8B-Instruct}

for SEED in "$@"; do
  CK="$OUT/phase_b/medquad_group_disjoint_llama31_seed${SEED}"
  RES="$OUT/phase_c/medquad_group_disjoint_llama31T${SEED}/metrics.json"
  if [ ! -f "$RES" ]; then
    echo "=== TRAIN group-disjoint seed $SEED $(date +%H:%M) ==="
    CUDA_VISIBLE_DEVICES=$GPUS $PY $PIPE/phase_b_prompt_and_adaptation.py \
      --dataset medquad_group_disjoint --base "$B" --label llama31 --seed "$SEED" \
      --n_train 500 --n_val 200 --epochs 3 --pool 2000 --max_memory_map "$CAP"
    echo "=== EVAL  group-disjoint seed $SEED $(date +%H:%M) ==="
    CUDA_VISIBLE_DEVICES=$GPUS $PY $PIPE/phase_c_inference_and_selection.py \
      --dataset medquad_group_disjoint --generator "$CK" \
      --generator_label "llama31T${SEED}" --verifier "${MEDLM_VERIFIER:?set MEDLM_VERIFIER}" \
      --n_test 300 --n_train 2000 --prompt_sets norag,rag --n_cand 5 \
      --batch_size 8 --seed 42 --truncation_side left
  fi
  echo "=== DELETE $CK ==="; rm -rf "$CK"
done
echo "EXP3 LANE DONE $(date +%H:%M)"
