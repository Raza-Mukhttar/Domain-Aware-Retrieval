#!/usr/bin/env bash
# End-to-end run of the proposed pipeline: Phase A, Phase B, Phase C in order.
#
# Usage: run_pipeline.sh <dataset> <backbone-hub-id-or-path> <label> [seed]
#   dataset   medquad | icliniq | medquad_group_disjoint
#   label     short name used in the output paths
#
# The ClinicalBERT verifier is not stored in this repository. Train it once with
# train_verifier.py and export MEDLM_VERIFIER, or pass --verifier to Phase C.
set -euo pipefail
cd "$(dirname "$0")"

DATASET=${1:?dataset}
BASE=${2:?backbone hub id or local path}
LABEL=${3:?label}
SEED=${4:-42}
PY=${PYTHON:-python}
: "${MEDLM_VERIFIER:?set MEDLM_VERIFIER to the verifier directory or hub id}"

OUT=${MEDLM_OUTPUT_DIR:-../../outputs}
CKPT="$OUT/phase_b/${DATASET}_${LABEL}_seed${SEED}"

echo "=== Phase A: offline preparation and retrieval ==="
$PY phase_a_offline_preparation.py --dataset "$DATASET" --seed "$SEED" --n_train 2000 --n_test 300

echo "=== Phase B: prompt construction and generator adaptation ==="
$PY phase_b_prompt_and_adaptation.py --dataset "$DATASET" --base "$BASE" --label "$LABEL" \
    --seed "$SEED" --n_train 500 --n_val 200 --epochs 3 --pool 2000 --k 3

echo "=== Phase C: online inference and selection ==="
$PY phase_c_inference_and_selection.py --dataset "$DATASET" --generator "$CKPT" \
    --generator_label "$LABEL" --verifier "$MEDLM_VERIFIER" \
    --prompt_sets norag,rag,rag_qtype,rag_fullans --n_cand 5 --also_n1 \
    --n_test 300 --n_train 2000 --batch_size 8 --seed 42 --truncation_side left

echo "=== Checkpoint no longer needed; removing to keep peak disk at one checkpoint ==="
rm -rf "$CKPT"
echo "pipeline complete: results under $OUT/phase_c/${DATASET}_${LABEL}"
