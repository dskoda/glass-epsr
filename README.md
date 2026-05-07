# GLASS — Generative Learning for Atomic Structure Simulation

GLASS is a CLI tool for training and running score-based generative models for atomic glass structures, with support for unconditional and guided (conditional) denoising using structural and spectral observables.

---

## Installation

```bash
git clone git@github.com:digital-synthesis-lab/GLASS.git
cd GLASS
pip install -e .
```

After installation, the `glass` command is available in your terminal.

**Dependencies** (must be installed separately):
- `torch`, `lightning`
- `ase`
- `scikit-learn`
- `graphite` (provides `graphite.nn.periodic_radius_graph`)
- `debyecalculator` (for XRD/ND in `write_spec_feature`)
- `tensorboard`, `matplotlib`, `seaborn` (for `plot_loss`)

---

## Expected Data Layout

```
data/
  <sample_tag>/
    structures/
      train/
        *.xyz
    exafs/
      train/
        *.txt        # one row per atom
    xanes/
      train/
        *.txt        # one row per atom

models/
  pre_trained/
    <system>_<score_model>/
      model.ckpt     # best checkpoint

reference/
  amorph_<system>_216/
    <system>_*.xyz   # reference amorphous structures

  start_<system>_216/
    <system>_*.xyz   # initial (noisy) structures for denoising
```

---

## Commands

### `train_score` — Train a score model

Trains a score-based generative model on atomic structures.

```bash
glass train_score Si_1.5_2.5_3.5 \
    --num-species 1 \
    --data-root ./data
```

Resume from the latest checkpoint:

```bash
glass train_score Si_1.5_2.5_3.5 \
    --num-species 1 \
    --data-root ./data \
    --resume
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--num-species` | required | Number of atomic species |
| `--cutoff` | 5.0 | Graph cutoff radius (Å) |
| `--k` | 0.8 | Maximum noise level |
| `--max-epochs` | 12000 | Training epochs |
| `--data-root` | `./data` | Root data directory |
| `--save-dir` | `./models/` | Checkpoint output directory |
| `--resume` | — | Resume from latest checkpoint |

Checkpoints are saved to `./models/<sample_tag>/version_*/checkpoints/`.

---

### `train_spec` — Train EXAFS or XANES surrogate model

Trains a per-atom spectral surrogate model used for spectral guidance.

```bash
# Train EXAFS
glass train_spec xas_train \
    --spec-type exafs \
    --num-species 1 \
    --out-dim 400 \
    --data-root ./data

# Train XANES
glass train_spec xas_train \
    --spec-type xanes \
    --num-species 1 \
    --out-dim 100 \
    --data-root ./data

# Resume
glass train_spec xas_train \
    --spec-type exafs --num-species 1 \
    --data-root ./data --resume
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--spec-type` | required | `exafs` or `xanes` |
| `--num-species` | required | Number of atomic species |
| `--out-dim` | 400 (exafs) / 100 (xanes) | Output dimension |
| `--max-epochs` | 8000 (exafs) / 3000 (xanes) | Training epochs |
| `--data-root` | `./data` | Root data directory |
| `--save-dir` | `./models/` | Checkpoint output directory |
| `--resume` | — | Resume from latest checkpoint |

Checkpoints are saved to `./models/<sample_tag>_<spec_type>/version_*/checkpoints/`.

---

### `plot_loss` — Plot training loss curves

```bash
glass plot_loss ./models/pre_trained --output score_loss.pdf
```

Options:

| Option | Default | Description |
|---|---|---|
| `--output` | `score_LC_all.pdf` | Output PDF filename |
| `--ylim` | `0.2 3.2` | Y-axis range |
| `--model` | all | Filter specific model(s) |

---

### `uncond_denoise` — Unconditional denoising

Runs denoising trajectories without any experimental guidance.

```bash
glass uncond_denoise \
    --score-model "1.5_2.5_3.5" \
    --system Si \
    --score-data-path "./data/{system}_{score_model}" \
    --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \
    --init-path "./reference/start_Si_216/Si_2.0_*.xyz" \
    --device cuda:0 \
    --n-runs 10
```

`--init-path` accepts a single `.xyz` file, a glob pattern, or a directory.

Key options:

| Option | Default | Description |
|---|---|---|
| `--score-model` | required | Score model tag(s), repeatable |
| `--system` | `Si` | System name |
| `--device` | `cuda:0` | Torch device |
| `--n-runs` | 10 | Independent runs per structure |
| `--tmax` | 1.0 | Reverse SDE end time |
| `--tstep` | 256 | Number of SDE steps |
| `--save-traj/--no-save-traj` | save | Save full trajectory or final frame only |

Output is written to `denoise_logs/unconditional/<system>-<model>/<struct_id>/`.

---

### `cond_denoise` — Conditional (guided) denoising

Runs denoising with experimental or computational guidance. Supports 6 guidance types:

| Type | Observable | Notes |
|---|---|---|
| `pdf` | Pair distribution function | CPU-based |
| `adf` | Angular distribution function | CPU-based |
| `xrd` | X-ray diffraction I(q) | GPU-accelerated |
| `nd` | Neutron diffraction I(q) | GPU-accelerated |
| `exafs` | EXAFS spectrum | Requires `--spec-model-path` |
| `xanes` | XANES spectrum | Requires `--spec-model-path` |

**PDF guidance (computational reference):**

```bash
glass cond_denoise \
    --score-model "1.5_2.5_3.5" --system Si \
    --score-data-path "./data/{system}_{score_model}" \
    --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \
    --init-path "./reference/start_Si_216/Si_2.0_*.xyz" \
    --ref-path ./reference/amorph_Si_216 \
    --guidance-type pdf --rho 1000 \
    --device cuda:0 --n-runs 10 --no-save-traj
```

**PDF guidance (experimental data):**

```bash
glass cond_denoise \
    --score-model "1.5_2.5_3.5" --system Si \
    --score-data-path "./data/{system}_{score_model}" \
    --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \
    --init-path "./reference/start_Si_216/Si_2.0_*.xyz" \
    --exp-data ./data/exp_gr_si.json \
    --guidance-type pdf --rho 1000 \
    --device cuda:0 --n-runs 10 --no-save-traj
```

**XRD guidance:**

```bash
glass cond_denoise \
    --score-model "1.5_2.5_3.5" --system Si \
    --score-data-path "./data/{system}_{score_model}" \
    --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \
    --init-path "./reference/start_Si_216/Si_2.0_*.xyz" \
    --ref-path ./reference/amorph_Si_216 \
    --guidance-type xrd --element-names Si --rho 5 \
    --device cuda:0 --n-runs 10 --no-save-traj
```

**EXAFS/XANES guidance:**

```bash
glass cond_denoise \
    --score-model "1.5_2.5_3.5" --system Si \
    --score-data-path "./data/{system}_{score_model}" \
    --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \
    --init-path "./reference/start_Si_216/Si_2.0_*.xyz" \
    --ref-path ./reference/amorph_Si_216 \
    --guidance-type exafs --spec-model-path ./models/Si_exafs.ckpt --rho 1e8 \
    --device cuda:0 --n-runs 10 --no-save-traj
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--guidance-type` | `pdf` | One of: `pdf`, `adf`, `xrd`, `nd`, `exafs`, `xanes` |
| `--rho` | 1000 | Guidance strength (repeatable for sweep) |
| `--ref-path` | — | Directory of reference `.xyz` files |
| `--exp-data` | — | JSON file with experimental `{x, y}` data |
| `--element-names` | — | Required for `xrd`/`nd` |
| `--spec-model-path` | — | Required for `exafs`/`xanes` |
| `--bin-size` | 100 | PDF bins |
| `--adf-cutoff` | 3.5 | ADF triplet cutoff (Å) |
| `--qmin/--qmax/--qstep` | 1.0/20.0/0.1 | q-range for XRD/ND |

Output is written to `denoise_logs/guided/<system>-<model>/<struct_id>/<guidance_type>_rho<rho>_tmax<tmax>_nsteps<tstep>/`.

---

### `write_spec_feature` — Compute spectral features

Computes PDF, ADF, XRD, ND, EXAFS, and XANES for denoised or reference structures and writes them to a JSON file.

```bash
# Denoised structures
glass write_spec_feature \
    --system Si \
    --denoise-tag "unconditional/Si-1.5_2.5_3.5" \
    --exafs-model ./models/Si_exafs.ckpt \
    --xanes-model ./models/Si_xanes.ckpt \
    --outdir results

# Reference structures
glass write_spec_feature --mode reference \
    --system Si \
    --atoms-path ./reference/amorph_Si_216 \
    --exafs-model ./models/Si_exafs.ckpt \
    --xanes-model ./models/Si_xanes.ckpt \
    --outdir results
```

---

### `build_ref_stats` — Build reference statistics

Computes mean spectra, diversity, and normalization factors from the reference JSON.

```bash
glass build_ref_stats \
    --input results/reference_Si_spectra.json \
    --system Si \
    --atoms-path ./reference/amorph_Si_216 \
    --outdir final_data_dir
```

Output: `final_data_dir/a-<system>_ref_stats.json`

---

### `calc_metrics` — Compute error and diversity metrics

Compares denoised spectra against the reference master stats.

```bash
glass calc_metrics \
    --denoise-json results/denoise_Si_unconditional_Si-1.5_2.5_3.5_spectra.json \
    --ref-master-json final_data_dir/a-Si_ref_stats.json \
    --system Si \
    --denoise-label "1.5_2.5_3.5" \
    --ref-label 1.5 --ref-label 2.0 --ref-label 2.5 --ref-label 3.0 --ref-label 3.5 \
    --outdir final_data_dir
```

---

## Typical Workflow

```
1. train_score          → train the denoising score model
2. train_spec           → train EXAFS and XANES surrogate models
3. uncond_denoise       → run unconditional denoising
   cond_denoise         → run guided denoising (any guidance type)
4. write_spec_feature   → compute spectra for denoised + reference structures
5. build_ref_stats      → compute reference statistics
6. calc_metrics         → evaluate error and diversity
```

See `submit.sh` for a complete set of example commands for the Si system.

---

## Multi-GPU Training

Use `CUDA_VISIBLE_DEVICES` to select GPUs:

```bash
export CUDA_VISIBLE_DEVICES=1,2,3
glass train_score Si_1.5_2.5_3.5 --num-species 1 --device cuda:0
```

The default strategy (`ddp_find_unused_parameters_true`) supports multi-GPU via PyTorch Lightning.
