#!/bin/sh

# ── GPU ────────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1,2,3

# ── Shared Parameters ──────────────────────────────────────────────────────────
SYSTEM="Si"
DEVICE="cuda:0"
NUM_SPECIES=1
N_RUNS=2

DIR_DATA="Si_1.5_2.5_3.5"
DATA_ROOT="../data"
MODEL_ROOT="../models"
INIT_ROOT="../reference"

SCORE_DATA_PATH="${DATA_ROOT}/{system}_{score_model}"
SCORE_MODEL_PATH="${MODEL_ROOT}/pre_trained/{system}_{score_model}/model.ckpt"
INIT_PATH="${INIT_ROOT}/start_{system}_216/{system}_2.0_*.xyz"

REF_PATH="${INIT_ROOT}/amorph_{system}_216"

EXAFS_MODEL="${MODEL_ROOT}/Si_exafs.ckpt"
XANES_MODEL="${MODEL_ROOT}/Si_xanes.ckpt"

REF_MASTER_JSON="./results/a-Si_ref_stats.json"
DENOISE_FOLDER="unconditional"

glass cond_denoise \
        --score-model "1.5_2.5_3.5" \
        --system $SYSTEM --device $DEVICE --n-runs $N_RUNS \
        --score-data-path "$SCORE_DATA_PATH" \
        --score-model-path "$SCORE_MODEL_PATH" \
        --init-path "$INIT_PATH" \
        --ref-path "$REF_PATH" \
        --guidance-type adf --rho 1 --adf-cutoff 3.5 --angle-bins 100 \
        --no-save-traj
