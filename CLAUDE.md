# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
   - PDF, ADF, coordination numbers, dihedrals, S(q), Voronoi
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
glass compare ref.xyz target.xyz
glass compare ref.json target.json --from-json
glass pdf structure.xyz --output pdf.json
glass coordination structure.xyz --output coord.json
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

The inference defaults in `glass.experiment.ExperimentConfig` come from
**`glass_unified_v3_ood`** (2026-05-18, `scripts/hpo_unified.py`), run
**after the Phase E Tersoff implementation bug was fixed** (see
"Tersoff implementation history" below). Each trial evaluates the same
parameter vector in unconditional and PDF-conditional modes at three
densities (ρ ∈ {1.5, 2.5, 3.5}); the trial objective is
`mean over densities of (0.5 · obj_uncond + 0.5 · obj_cond)`, where each
`obj_mode = 1.0 · pdf_rmse + 2.0 · coord_emd + 0.25 · adf_rmse`. The
`W_COORD=2.0` weighting (vs v1's 1.0) makes coord_emd parity with PDF
the primary objective.

Winning trial replay (5 inits × 5 seeds × 3 densities, n=75 per mode):

| Mode | pdf_rmse | coord_emd | adf_rmse |
|---|---|---|---|
| Unconditional (mean) | 0.429 | 0.872 | 0.071 |
| PDF-conditional (mean) | **0.028** | 0.173 | 0.060 |

Per-density breakdown of the best trial (cond mode, n=1×1):

| Density | pdf_rmse | coord_emd | adf_rmse |
|---|---|---|---|
| ρ=1.5 (OOD) | 0.056 | 0.380 | 0.073 |
| ρ=2.5 (in-distribution) | 0.012 | 0.019 | 0.045 |
| ρ=3.5 (OOD) | 0.020 | 0.009 | 0.033 |

v3_ood vs v2_ood (cond mode, replay 5×5×3): cond pdf_rmse improves
0.042 → 0.028 (33%) with coord_emd essentially unchanged (0.171 →
0.173). The big wins come from the bug-fix: at the per-trial level,
ρ=2.5 cond pdf 0.019 → 0.012 and ρ=3.5 cond pdf 0.032 → 0.020.

Defaults chosen as the **top-5 consensus** (median for floats, mode
for categoricals across the 5 best trials), not the literal winner:

| Param | v3_ood (current) | v2_ood (pre-fix) | v1 (in-dist only) | Notes |
|---|---|---|---|---|
| `tmin` | 4e-4 | 7e-4 | 1e-4 | unchanged-ish |
| `tmax` | 0.834 | 0.876 | 0.54 | unchanged |
| `tstep` | **512** | 256 | 128 | mode 5/5 in top-5; doubled again |
| `t_schedule_rho` | **0.98** | 1.35 | 1.4 | back to ~uniform — fixed Tersoff makes high-t informative |
| `tersoff_guidance` | true | true | true | — |
| `tersoff_lambda` | 0.23 | 0.20 | 0.22 | slight increase |
| `tersoff_schedule` | **sigmoid** | linear | linear | mode 2/5 (linear competitive); sigmoid + low t_gate = mid-trajectory active |
| `tersoff_t_gate` | **0.15** | 0.45 | 0.75 | Tersoff now activates *early* (high t) rather than only near the end |
| `tersoff_clamp` | 10.0 | 10.0 | 10.0 | unchanged |
| `n_corr` | **1** | 2 | 1 | top-5 unanimous on 1; the corrector polish role gone with bug fix |
| `corr_step_size` | **0.30** | 0.12 | 0.13 | larger steps, fewer of them |
| `corr_t_gate` | **0.79** | 0.58 | 0.37 | corrector now active over much wider noise range |
| `rho` (guidance) | **737.0** | 240.0 | 35.0 | **~3× higher than v2** — fixed Tersoff lets cond run hotter |
| `sa_n_steps` | 0 | 0 | 0 | SA still off; v3 confirms |

**Why these changes matter (v2_ood → v3_ood):**

- The Phase E Tersoff fix removed up to 180 meV/atom and 0.78 eV/Å
  RMS force errors at OOD densities. Once the angular term is computed
  correctly, Tersoff behaves like an actual energy term (not a noise
  term), so the optimizer can run the rest of the sampler more
  aggressively.
- `rho: 240 → 737` — the conditional gradient can now safely dominate
  the score at low t, because the score's bias on the local angular
  geometry is being absorbed by the (now-correct) Tersoff term.
- `n_corr: 2 → 1` — the corrector's role of polishing structure between
  predictor steps was a workaround for the buggy Tersoff. With the fix,
  one predictor step is enough.
- `tersoff_t_gate: 0.45 → 0.15` and `schedule: linear → sigmoid` mean
  Tersoff is now active during structural-decision-time (mid noise)
  rather than only at the very end. Consistent with Tersoff actually
  doing something useful now.
- `t_schedule_rho: 1.35 → 0.98` — the "concentrate at low noise"
  v2_ood/v1 trick reflected that high-noise steps were noisy under the
  buggy Tersoff. With the fix, near-uniform t spacing wins.

### Tersoff implementation history (Phase E)

In May 2026, a cross-check against LAMMPS (`research/density_extrapolation/experiments/phase_e_lammps_xcheck/`)
revealed an operator-precedence bug in
`glass/potentials/tersoff/potential.py`. The angular-term exponent
argument was written as `λ₃ · Δr^m` instead of `(λ₃ · Δr)^m`. The two
expressions agree on diamond Si (Δr = 0 for every triple) but
diverge in amorphous structures, with magnitude growing with density
(more neighbour pairs in the cutoff fringe):

| density | pre-fix \|E_glass − E_lammps\|/atom | RMS force diff | post-fix |
|---|---|---|---|
| 1.5 | 7.6 × 10⁻⁵ eV | 0.022 eV/Å | 1.9 × 10⁻¹⁵ eV |
| 2.5 (training) | 1.7 × 10⁻² eV (17 meV) | 0.24 eV/Å | 4.2 × 10⁻¹⁵ eV |
| 3.5 | 1.8 × 10⁻¹ eV (180 meV) | 0.78 eV/Å | 4.4 × 10⁻¹⁵ eV |

After the fix, glass torch Tersoff is bit-equivalent to LAMMPS at
machine precision (~1e-15 eV/atom) on all 30 reference structures.
ASE 3.25's reference Tersoff also agrees. `tests/test_tersoff.py`
now cross-checks against ASE on the amorphous Si_2.5_00.xyz snapshot
to prevent regression.

### How v3_ood was tuned (Phase A → Phase F)

The v1 → v2_ood → v3_ood progression came from a five-month
investigation in May 2026:

1. **Phase A** (`research/density_extrapolation/experiments/phase_a/`):
   rho × Tersoff × density matrix at OOD densities. Found that rho
   is the dominant lever and ρ=1.5 vs ρ=3.5 disagree on the joint
   optimum, motivating multi-density HPO.
2. **Phase B** (provisional rho=300 promotion).
3. **Phase C** (multi-density HPO `glass_unified_v2_ood`).
4. **Phase D** (post-hoc Tersoff treatments — SA, FIRE, iterated):
   all rejected, but with the buggy Tersoff.
5. **Phase E** (LAMMPS cross-check): found the operator-precedence
   bug in `potential.py`; fixed it.
6. **Phase F** (Phase D rerun + new HPO): all post-hoc treatments
   *still* fail at ρ=1.5, but `glass_unified_v3_ood` finds a
   significantly better default set. Promoted.

Diagnostic plots (each lives under its experiment's local results dir):
- `research/density_extrapolation/experiments/phase_a/results/phase_a_pareto.png` —
  PDF vs coord pareto by Tersoff config and density.
- `research/density_extrapolation/experiments/h1_diagnostic/results/h1_diagnostic.png` —
  first-peak position and coord vs density showing the density-blind prior.
- `research/density_extrapolation/experiments/rho_sweep/results/rho_sweep.png` —
  PDF / coord vs rho at ρ=1.5.

### Running / Extending the HPO

The script supports both single-density (legacy) and multi-density modes.
Multi-density uses `--init-globs` + `--ref-dirs` (parallel comma-separated
lists, one entry per density bucket).

```bash
# Single-density (legacy, ρ=2.5 only) — 4 GPUs, ~2 h, 200 trials
python scripts/hpo_unified.py research/test/ \
    --ref-dir research/test/data/ \
    --init-dir /path/to/inits \
    --init-glob "Si_2.5_*.xyz" \
    --n-trials 200 --n-seeds 2 --n-inits 2 \
    --n-jobs 4 --devices cuda:0,cuda:1,cuda:2,cuda:3 \
    --study-name glass_unified_v2 \
    --storage research/hpo/glass_unified_v2.db

# Multi-density (v2_ood pattern) — 4 GPUs, ~3 h, 200 trials
python scripts/hpo_unified.py research/density_extrapolation/experiment/ \
    --ref-dirs "research/density_extrapolation/results/generated/cond/density_1.5/reference,research/density_extrapolation/results/generated/cond/density_2.5/reference,research/density_extrapolation/results/generated/cond/density_3.5/reference" \
    --init-dir research/density_extrapolation/experiment/inits \
    --init-globs "init_Si_1.5_*.xyz,init_Si_2.5_*.xyz,init_Si_3.5_*.xyz" \
    --n-trials 200 --n-seeds 1 --n-inits 1 \
    --n-jobs 4 --devices cuda:0,cuda:1,cuda:2,cuda:3 \
    --study-name glass_unified_v2_ood \
    --storage research/hpo/glass_unified_v2_ood.db

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
- Rings: ring size distributions, RMSE, and other distribution metrics

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
