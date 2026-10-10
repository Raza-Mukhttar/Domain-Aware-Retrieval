#!/usr/bin/env bash
# Closed-form track of the same three phases: adaptation then inference with
# verifier fusion, on MedQA / MMLU-Med / MedMCQA / PubMedQA.
#
# Usage: run_pipeline_closedform.sh <benchmark> <backbone-hub-id-or-path> <label> [seed]
set -euo pipefail
cd "$(dirname "$0")"

DATASET=${1:?benchmark}
BASE=${2:?backbone hub id or local path}
LABEL=${3:?label}
SEED=${4:-42}
PY=${PYTHON:-python}
OUT=${MEDLM_OUTPUT_DIR:-../../outputs}
CKPT="$OUT/runs/ckptcf_${DATASET}_${LABEL}_seed${SEED}"

echo "=== Phase B: task adaptation on $DATASET ==="
$PY closedform_phase_b_adaptation.py --dataset "$DATASET" --base "$BASE" --label "$LABEL" --seed "$SEED"

echo "=== Phase C: inference, retrieval and verifier fusion ==="
$PY closedform_phase_c_inference_and_selection.py --dataset "$DATASET" --generator "$CKPT" \
    --generator_label "${LABEL}ft" --conditions adapted,adapted_verifier,adapted_rag,adapted_rag_verifier \
    --n_test 500 --seed 42

echo "=== Removing checkpoint ==="
rm -rf "$CKPT"
