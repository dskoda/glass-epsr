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

# ── train_score ────────────────────────────────────────────────────────────────
glass train_score $DIR_DATA --num-species $NUM_SPECIES --data-root $DATA_ROOT

# Resume from checkpoint
glass train_score $DIR_DATA --num-species $NUM_SPECIES --data-root $DATA_ROOT --resume

# ── train_spec (EXAFS) ─────────────────────────────────────────────────────────
glass train_spec xas_train \
	--spec-type exafs --num-species $NUM_SPECIES --out-dim 400 \
	--data-root $DATA_ROOT

Resume EXAFS
glass train_spec xas_train \
	--spec-type exafs --num-species $NUM_SPECIES --out-dim 400 \
	--data-root $DATA_ROOT --resume

# ── train_spec (XANES) ─────────────────────────────────────────────────────────
glass train_spec xas_train \
	--spec-type xanes --num-species $NUM_SPECIES --out-dim 100 \
	--data-root $DATA_ROOT

# Resume XANES
glass train_spec xas_train \
	--spec-type xanes --num-species $NUM_SPECIES --out-dim 100 \
	--data-root $DATA_ROOT --resume

# ── plot_loss ──────────────────────────────────────────────────────────────────
glass plot_loss /home/jwguo/03_denoiser/demo_Si/models/pre_trained --output score_loss.pdf

# ── uncond_denoise ─────────────────────────────────────────────────────────────
# Single model, full trajectory
glass uncond_denoise \
	--score-model "1.5_2.5_3.5" \
	--system $SYSTEM --device $DEVICE --n-runs $N_RUNS \
	--score-data-path "$SCORE_DATA_PATH" \
	--score-model-path "$SCORE_MODEL_PATH" \
	--init-path "$INIT_PATH"

# Multiple models, final frame only
glass uncond_denoise \
	--score-model "1.5" --score-model "2.5" --score-model "3.5" \
	--system $SYSTEM --device $DEVICE --n-runs $N_RUNS \
	--score-data-path "$SCORE_DATA_PATH" \
	--score-model-path "$SCORE_MODEL_PATH" \
	--init-path "$INIT_PATH" \
	--no-save-traj

# ── cond_denoise (computational ref) ───────────────────────────────────────────
glass cond_denoise \
	--score-model "1.5_2.5_3.5" \
     	--system $SYSTEM --device $DEVICE --n-runs $N_RUNS \
	--score-data-path "$SCORE_DATA_PATH" \
	--score-model-path "$SCORE_MODEL_PATH" \
	--init-path "$INIT_PATH" \
	--ref-path "$REF_PATH" \
	--guidance-type pdf --rho 1000 --tmax 1.0 --tstep 256 \
	--no-save-traj

# ── cond_denoise: ADF guidance ────────────────────────────────────────────────
glass cond_denoise \
	--score-model "1.5_2.5_3.5" \
	--system $SYSTEM --device $DEVICE --n-runs $N_RUNS \
	--score-data-path "$SCORE_DATA_PATH" \
	--score-model-path "$SCORE_MODEL_PATH" \
	--init-path "$INIT_PATH" \
	--ref-path "$REF_PATH" \
	--guidance-type adf --rho 1 --adf-cutoff 3.5 --angle-bins 100 \
	--no-save-traj

# ── cond_denoise: XRD guidance ────────────────────────────────────────────────
glass cond_denoise \
	--score-model "1.5_2.5_3.5" \
	--system $SYSTEM --device $DEVICE --n-runs $N_RUNS \
	--score-data-path "$SCORE_DATA_PATH" \
	--score-model-path "$SCORE_MODEL_PATH" \
	--init-path "$INIT_PATH" \
	--ref-path "$REF_PATH" \
	--guidance-type xrd --element-names Si --rho 5 \
	--qmin 1.0 --qmax 20.0 --qstep 0.1 \
	--no-save-traj

# ── cond_denoise: ND guidance ─────────────────────────────────────────────────
glass cond_denoise \
	--score-model "1.5_2.5_3.5" \
	--system $SYSTEM --device $DEVICE --n-runs $N_RUNS \
	--score-data-path "$SCORE_DATA_PATH" \
	--score-model-path "$SCORE_MODEL_PATH" \
	--init-path "$INIT_PATH" \
	--ref-path "$REF_PATH" \
	--guidance-type nd --element-names Si --rho 1 \
	--qmin 1.0 --qmax 20.0 --qstep 0.1 \
	--no-save-traj

# ── cond_denoise: EXAFS guidance ─────────────────────────────────────────────
glass cond_denoise \
	--score-model "1.5_2.5_3.5" \
	--system $SYSTEM --device $DEVICE --n-runs $N_RUNS \
	--score-data-path "$SCORE_DATA_PATH" \
	--score-model-path "$SCORE_MODEL_PATH" \
	--init-path "$INIT_PATH" \
	--ref-path "$REF_PATH" \
	--guidance-type exafs --spec-model-path $EXAFS_MODEL --rho 1e8 \
	--no-save-traj

# ── cond_denoise: XANES guidance ─────────────────────────────────────────────
glass cond_denoise \
	--score-model "1.5_2.5_3.5" \
	--system $SYSTEM --device $DEVICE --n-runs $N_RUNS \
	--score-data-path "$SCORE_DATA_PATH" \
	--score-model-path "$SCORE_MODEL_PATH" \
	--init-path "$INIT_PATH" \
	--ref-path "$REF_PATH" \
	--guidance-type xanes --spec-model-path $XANES_MODEL --rho 1e6 \
	--no-save-traj

# ── cond_denoise (experimental data) ────────────────────────────────────────────
glass cond_denoise \
	--score-model "1.5_2.5_3.5" \
	--system $SYSTEM --device $DEVICE --n-runs $N_RUNS \
	--score-data-path "$SCORE_DATA_PATH" \
	--score-model-path "$SCORE_MODEL_PATH" \
	--init-path "$INIT_PATH" \
	--exp-data ./data/exp_gr_si.json \
	--bin-size 100 \
	--guidance-type pdf --rho 1000 --tmax 1.0 --tstep 256 \
	--no-save-traj

# ── write spectrum features (denoise) ───────────────────────────────────────────────────
glass write_spec_feature \
	--system $SYSTEM \
	--denoise-tag "$DENOISE_FOLDER/Si-1.5_2.5_3.5" \
     	--exafs-model $EXAFS_MODEL --xanes-model $XANES_MODEL \
     	--outdir results

# Reference mode
glass write_spec_feature --mode reference \
	--system $SYSTEM \
	--atoms-path /home/jwguo/03_denoiser/reference/amorph_Si_216 \
	--exafs-model $EXAFS_MODEL --xanes-model $XANES_MODEL \
	--outdir results

# ── build_ref_stats ────────────────────────────────────────────────────────────
glass build_ref_stats \
	--input results/reference_Si_spectra.json \
	--system $SYSTEM \
	--atoms-path /home/jwguo/03_denoiser/reference/amorph_Si_216 \
	--outdir results

# ── calc_metrics ───────────────────────────────────────────────────────────────
glass calc_metrics \
	--denoise-json results/denoise_Si_unconditional_Si-1.5_2.5_3.5_spectra.json \
	--ref-master-json $REF_MASTER_JSON \
	--system $SYSTEM \
	--denoise-label "1.5_2.5_3.5" \
	--ref-label 1.5 --ref-label 2.0 \
	--outdir results
