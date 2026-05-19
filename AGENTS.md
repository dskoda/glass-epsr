# AGENTS.md

This file provides guidance to AI assistants when working with code in this repository.

## What this repo is

A single Python package `glass` (defined by `./pyproject.toml`) for generative modeling of amorphous materials using score-based diffusion models.

### Components

1. **`glass.lit`** — Score-based generative model for reconstructing amorphous structures
   - Training, unconditional denoising, and guided denoising
   - Spectral/structural observables: PDF, ADF, XRD, ND, EXAFS, XANES
   - Requires `lightning` (not installed by default in this env)

2. **`glass.diffusion`** — SDE sampling machinery decoupled from Lightning
   - `sampling.denoise_by_sde` — unified reverse-SDE driver (uncond + cond)
   - `schedules.power_law_ts` — non-linear t trajectory
   - `annealing.simulated_anneal` / `make_anneal_fn` — Tersoff-based post-relaxation

3. **`glass.potentials.tersoff`** — PyTorch reimplementation of LAMMPS-style Tersoff potential
   - Single-species Si, autograd-based forces
   - `TorchTersoff` (raw energy+forces), `TorchTersoffCalculator` (ASE Calculator)
   - Scales to 10 000+ atoms via cell-list neighbour enumeration

4. **`glass.metrics`** — Structural analysis (non-differentiable)
   - PDF, ADF, coordination numbers, dihedrals, S(q), Voronoi, ring statistics
   - Error metrics: RMSE, cosine similarity, EMD, R-chi²

5. **`glass.utils.packing`** — Cell-list Poisson-disk + WCA Monte-Carlo fallback
   - Powers `glass initialize`; handles up to ~55% packing fraction

## Quick Start

```bash
pip install -e .
pip install -e ".[plot,diffraction,hpo]"
KMP_DUPLICATE_LIB_OK=TRUE pytest -v
```

## Architecture

### Package Layout

```
./
├── pyproject.toml
├── glass/
│   ├── cli/                            # CLI (package, not module)
│   │   ├── __main__.py                 # `python -m glass` → `glass` entrypoint
│   │   ├── generate.py                 # `glass generate`
│   │   ├── train.py                    # `glass train`
│   │   ├── initialize.py               # `glass initialize`
│   │   ├── metrics.py                  # `glass metrics` / `compare` / `pdf` / ...
│   │   └── analysis.py
│   ├── diffusion/
│   │   ├── sampling.py                 # denoise_by_sde (unified loop)
│   │   ├── schedules.py                # power_law_ts
│   │   └── annealing.py                # SA tail on Tersoff PES
│   ├── experiment.py                   # ExperimentConfig dataclass + paths
│   ├── lit/                            # Lightning training + denoising
│   │   ├── datamodules/
│   │   ├── functions/get_atoms.py      # ASE → tensor conversion
│   │   └── modules/
│   │       ├── prior.py                # LitScoreNet
│   │       ├── forward.py              # LitSpecNet
│   │       ├── likelihood.py           # LikelihoodScore (conditional term)
│   │       ├── tersoff_guidance.py     # TersoffEnergyGuidance + TersoffSchedule
│   │       ├── guidance.py             # create_guidance_model, target loaders
│   │       ├── differentiable_rdf.py
│   │       ├── differentiable_adf.py
│   │       ├── differentiable_xrd.py
│   │       └── differentiable_nd.py
│   ├── metrics/
│   │   ├── core.py                     # Dataclasses
│   │   ├── structural.py               # PDF, ADF
│   │   ├── geometric.py                # Coordination, dihedrals
│   │   ├── advanced.py                 # S(q), Voronoi
│   │   ├── rings.py                    # Ring statistics (Franzblau algorithm)
│   │   ├── errors.py                   # compute_all_errors
│   │   └── utils.py
│   ├── potentials/tersoff/             # Tersoff
│   │   ├── params.py
│   │   ├── neighbors.py                # dense + cell-list path
│   │   ├── potential.py                # padded triple-sum
│   │   ├── ase_calc.py
│   │   └── cli.py                      # `glass energy` / `glass md`
│   └── utils/
│       ├── atoms_utils.py              # ASE ↔ tensor, prior/target helpers
│       └── packing.py                  # Poisson-disk + MC fallback
├── scripts/
│   ├── hpo_unified.py                  # Joint uncond+cond+SA HPO (Optuna TPE)
│   └── hpo_generate.py                 # Legacy uncond-only HPO
└── tests/                              # 164+ tests
    ├── test_tersoff.py
    ├── test_tersoff_guidance.py
    ├── test_sampling_corrector.py      # uncond + cond + corrector + SA
    ├── test_schedules.py
    ├── test_annealing.py
    ├── test_initialize_packing.py
    ├── test_metrics.py
    └── data/Si_2.5_00.xyz
```

## Common Commands

### Testing

```bash
# Full test suite (~60s)
KMP_DUPLICATE_LIB_OK=TRUE pytest -v

# Single test
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_tersoff.py::test_snapshot_energy -v
```

### Training

```bash
glass train ./my_experiment --model-type score --num-species 1
glass train ./my_experiment --model-type spec --spec-type exafs --num-species 1
glass train ./my_experiment --resume
glass train ./my_experiment --max-epochs 5000 --lr 0.0005 --dim 256
```

### Initialisation

```bash
# Fast cell-list Poisson-disk packer (MC fallback at high density)
glass initialize --output init.xyz --density 2.5 --species Si \
    --num-atoms 216 --min-distance 2.0 --seed 0

# Exact cell, derive num-atoms from density
glass initialize --output init.xyz --cell-a 15.91 --density 2.5 --species Si \
    --min-distance 2.0
```

### Generation

```bash
# Unconditional — picks up defaults from experiment config.yaml
glass generate ./my_experiment --inits ./inits/

# PDF-guided (reuses the same Tersoff/corrector/SA pipeline)
glass generate ./my_experiment --inits ./inits/ \
    --guidance-type pdf --ref-path ./reference/

# XRD-guided
glass generate ./my_experiment --inits ./inits/ \
    --guidance-type xrd --ref-path ./reference/ \
    --element-names Si --rho 5
```

### Metrics

```bash
glass metrics structure.xyz --output metrics.json
glass metrics ./structures/*.xyz --output metrics.json
glass metrics structure.xyz --include-rings --rings-maxlength 12  # include ring stats
glass compare ref.xyz target.xyz
glass compare ref.json target.json --from-json
glass pdf structure.xyz --output pdf.json
glass coordination structure.xyz --output coord.json
glass rings structure.xyz --cutoff 3.0 --maxlength 10  # standalone ring stats
```

### Tersoff Potential

```bash
KMP_DUPLICATE_LIB_OK=TRUE glass energy ./tests/data/Si_2.5_00.xyz
KMP_DUPLICATE_LIB_OK=TRUE glass md \
    --input ./tests/data/Si_2.5_00.xyz \
    --ensemble nve --steps 100 --timestep 1.0
```

## Implementation Details

### Experiment Structure

```
my_experiment/
├── config.yaml          # ExperimentConfig serialisation
├── data/                # Training data (*.xyz)
├── checkpoints/
├── inits/
├── outputs/
└── logs/
```

### Reverse-SDE Denoising Flow (unconditional + conditional share one function)

`glass.diffusion.sampling.denoise_by_sde` is the single entry point for both
modes. Inside each predictor step:

```
p_score = prior(pos, t)                                      # score net
if tersoff_guidance:
    p_score += λ(t) · (−∇E_Tersoff / N)                      # empirical potential
if likelihood_fn:                                            # conditional only
    l_score, norm = likelihood_fn(species, pos, cell, t, cutoff)
    total = p_score + l_score
else:
    total = p_score

disp = (f(t)·pos − g²(t)·total)·dt + g(t)·noise
pos  = pos + disp
```

After the predictor, an optional Langevin corrector runs `n_corr` inner
steps (gated off when `t > corr_t_gate · t_max`), and an optional SA tail
runs on the Tersoff PES after the loop exits.

**Time schedule:** `power_law_ts(tmin, tmax, tstep, rho)` with
`rho > 1` concentrating steps at low noise (MD-like), `rho < 1` at high
noise (prior-dominated), `rho = 1` uniform.

**Guidance strength:** `LikelihoodScore` returns
`-(rho / norm.sum()) · ∇_pos ||target - pred(pos̄_0)||²`. The normalisation
makes `rho` roughly scale-invariant across features, but it still has to
balance against `tersoff_λ` (see Default Parameters below).

### HPO and Default Parameters

The inference defaults in `glass.experiment.ExperimentConfig` come from a
**joint unconditional + PDF-conditional HPO study** (`glass_unified_v1`)
run by `scripts/hpo_unified.py`. Each trial evaluates the same parameter
vector in both modes; the trial objective is
`0.5 · obj_uncond + 0.5 · obj_cond`, where each
`obj_mode = 1.0 · pdf_rmse + 1.0 · coord_emd + 0.25 · adf_rmse`.

Winning trial replay (10 inits × 5 seeds):

| | obj | pdf_rmse | coord_emd | adf_rmse |
|---|---|---|---|---|
| Unconditional | 0.039 | 0.018 | 0.012 | 0.039 |
| PDF-conditional | 0.046 | 0.013 | 0.023 | 0.042 |

Defaults chosen as the **top-5 consensus** (median across best trials),
not the literal winner, to avoid single-sample overfit:

| Param | Value | Notes |
|---|---|---|
| `tmin` | 1e-4 | lower floor than v1 |
| `tmax` | 0.54 | was 0.593 |
| `tstep` | 128 | 4× cheaper than v1's 512 |
| `t_schedule_rho` | **1.4** | **inverted** from v1's 0.55 — concentrates at low noise |
| `tersoff_guidance` | true | — |
| `tersoff_lambda` | 0.22 | — |
| `tersoff_schedule` | linear | was constant |
| `tersoff_t_gate` | 0.75 | ramp Tersoff in late |
| `tersoff_clamp` | 10.0 | unchanged |
| `n_corr` | 1 | — |
| `corr_step_size` | 0.13 | ~½ of v1 |
| `corr_t_gate` | 0.37 | tighter than v1's 0.6 |
| `rho` (guidance) | **35.0** | **~25× lower** than prior 1000 — crucial |
| `sa_n_steps` | 0 | SA hurts under the unified objective; disabled |

**Why these changes matter:**

- At `rho=1000` the PDF-likelihood gradient swamped `tersoff_λ=0.263`, so
  the conditional path lost the coordination-number wins from Tersoff.
  At `rho ≈ 35` all three score terms (prior, Tersoff, likelihood)
  contribute in the same order of magnitude.
- The inverted t-schedule (`t_rho: 0.55 → 1.4`) and `tstep: 512 → 128`
  together give a faster, more MD-like trajectory that spends more
  effective compute at low noise where the corrector is effective.
- Every top-5 trial had `N_anneal=0`. The Langevin corrector inside the
  main loop already captures what the SA tail was doing; running SA on
  the Tersoff PES after conditional denoising undoes the likelihood fit.

### Running / Extending the HPO

```bash
# Full run (4 GPUs, ~2 h, 200 trials)
python scripts/hpo_unified.py research/test/ \
    --ref-dir research/test/data/ \
    --init-dir /path/to/inits \
    --init-glob "Si_2.5_*.xyz" \
    --n-trials 200 --n-seeds 2 --n-inits 2 \
    --n-jobs 4 --devices cuda:0,cuda:1,cuda:2,cuda:3 \
    --study-name glass_unified_v2 \
    --storage research/hpo/glass_unified_v2.db

# Resume: re-run the exact same command (SQLite + load_if_exists=True)

# Replay best at higher seed count for robust verification
python scripts/hpo_unified.py research/test/ \
    --ref-dir ... --init-dir ... --init-glob "Si_2.5_*.xyz" \
    --study-name glass_unified_v2 \
    --storage research/hpo/glass_unified_v2.db \
    --replay-best --n-seeds 5 --n-inits 10
```

When promoting a new best to defaults:
1. Replay at 5 seeds × 10 inits to confirm the improvement is real.
2. Update `glass/experiment.py::ExperimentConfig` AND
   `research/test/config.yaml` (both sets of defaults must agree).
3. `pytest tests/test_experiment.py tests/test_hpo_objective.py` to confirm
   the round-trip still works.

### Metrics Module

**PDF Normalization**: `g(r) = (V/N²) × hist / (4πr²Δr)`
- Ensures g(r) → 1 as r → ∞.
- Gaussian smoothing optional (default σ=0.15 Å).

**Coordination Cutoff**: Auto-detected from PDF first minimum; fallback
`1.3 × first_peak`.

**Error Metrics**:
- PDF/ADF: RMSE, MAE, area between curves, cosine similarity, R-chi².
- Coordination: EMD (Wasserstein), histogram RMSE, mean/std error.
- Peak: position error, height error.
- Rings: RMSE, MAE, cosine similarity, EMD, total count error.

### Tersoff Potential Notes

- **Single-species only.** Homogeneous `(A, A, A)` keys.
- **ASE-compatible energy.** Matches `ase.calculators.tersoff` 3.25.0.
- **Forces via torch autograd** by default; an analytical path exists for
  testing (`energy_and_forces_analytical`).
- **Scales to large cells.** `build_neighbors` auto-switches to a cell-list
  at N > 256 and `_energy_from_pairs` uses a padded `(N, n_max, n_max)`
  triple-sum, so a 10 000-atom supercell fits in <1 GB GPU memory.

### Environment Variables

- `KMP_DUPLICATE_LIB_OK=TRUE` — Required on macOS/some Linux when mixing
  PyTorch and SciPy-backed ASE. Safe to always set.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — set automatically
  by `glass generate` to reduce fragmentation.

## Dependencies

**Core**: `torch`, `ase`, `numpy`, `click`, `scipy`

**Training**: `lightning`, `torch-geometric`, `scikit-learn`

**Optional**:
- `[test]`: pytest
- `[plot]`: matplotlib, seaborn, tensorboard
- `[diffraction]`: DebyeCalculator (pins Python <3.12)
- `[hpo]`: optuna (for `scripts/hpo_unified.py`)

## Adding New Features

### New Guidance Type

1. Create `DifferentiableXXX` in `glass/lit/modules/`.
2. Expose `forward(pos, species, cell) → predicted_feature`.
3. Register it in `glass/lit/modules/guidance.py::create_guidance_model`.
4. Add a branch in `glass/lit/modules/likelihood.py::LikelihoodScore.forward`.
5. Add a branch in `glass/utils/atoms_utils.py::compute_target_from_reference`.
6. Add the `guidance_type` literal to the `click.Choice` in
   `glass/cli/generate.py`.

### New Error Metric

1. Add function to `glass/metrics/errors.py`.
2. Signature: `metric_name(ref: Metrics, target: Metrics) → float`.
3. Add to `compute_all_errors()`.
4. If you want the HPO to optimise it, add a weight constant at the top
   of `scripts/hpo_unified.py` and include it in `_mode_obj`.

### New Ring Statistics Feature

1. The ring statistics module is at `glass/metrics/rings.py`.
2. Uses the Franzblau shortest-path algorithm (Python implementation).
3. CLI commands:
   - `glass rings structure.xyz` — standalone ring analysis
   - `glass metrics --include-rings` — include in comprehensive metrics
4. Error metrics for rings are in `glass/metrics/errors.py`:
   - `rings_rmse`, `rings_mae`, `rings_cosine_similarity`, `rings_emd`, `rings_total_error`

### Tuning Defaults

**Don't hand-tune defaults in a PR.** Run the unified HPO (even a short
50-trial study on one GPU gives a usable read), replay the best point at
5 seeds × 10 inits, and only then promote to `ExperimentConfig`.

## Key Files

- `glass/cli/generate.py` — CLI entry for denoising; plumbs all HPO flags.
- `glass/diffusion/sampling.py` — the single reverse-SDE loop.
- `glass/lit/modules/likelihood.py` — conditional guidance term.
- `glass/lit/modules/tersoff_guidance.py` — empirical-potential term.
- `glass/experiment.py` — `ExperimentConfig` (defaults live here).
- `scripts/hpo_unified.py` — joint uncond+cond Optuna study.
- `tests/test_sampling_corrector.py` — composition tests (prior + Tersoff
  + stub likelihood + corrector + SA tail).
- `tests/test_tersoff.py`, `tests/test_metrics.py` — reference physics checks.
- `tests/test_rings.py` — ring statistics tests (Franzblau algorithm).
