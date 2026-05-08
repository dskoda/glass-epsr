# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A single Python package `glass` (defined by `./pyproject.toml`) for generative modeling of amorphous materials using score-based diffusion models.

### Components

1. **`glass.lit`** — Score-based generative model for reconstructing amorphous structures
   - Training, unconditional denoising, and guided denoising
   - Spectral/structural observables: PDF, ADF, XRD, ND, EXAFS, XANES
   - Requires `lightning` (not installed by default in this env)

2. **`glass.potentials.torch_tersoff`** — PyTorch reimplementation of LAMMPS-style Tersoff potential
   - Single-species Si, autograd-based forces
   - Exports: `TorchTersoff`, `TorchTersoffCalculator` (ASE Calculator subclass), `silicon_calculator()`
   - Intended for energy-based guidance and post-denoising relaxation

3. **`glass.metrics`** — Structural analysis metrics (non-differentiable)
   - PDF (Pair Distribution Function), ADF (Angular Distribution Function)
   - Coordination numbers, dihedral angles, structure factor S(q), Voronoi analysis
   - Error metrics for comparing structures (RMSE, cosine similarity, EMD, etc.)

## Quick Start

```bash
# Install package
pip install -e .

# Install optional dependencies for full functionality
pip install -e ".[plot,diffraction]"

# Run tests
KMP_DUPLICATE_LIB_OK=TRUE pytest -v
```

## Architecture

### Package Layout

```
./
├── pyproject.toml                      # Package definition
├── glass/
│   ├── cli.py                          # Main CLI (train, generate, metrics, etc.)
│   ├── lit/                            # Lightning training + denoising
│   │   ├── datamodules/                # PyG Data pipeline
│   │   ├── functions/get_atoms.py      # ASE → tensor conversion
│   │   └── modules/
│   │       ├── prior.py                # LitScoreNet (score-based SDE)
│   │       ├── forward.py              # LitSpecNet (spectral surrogate)
│   │       ├── differentiable_rdf.py   # Differentiable PDF guidance
│   │       ├── differentiable_adf.py   # Differentiable ADF guidance
│   │       ├── differentiable_xrd.py   # Differentiable XRD guidance
│   │       └── differentiable_nd.py    # Differentiable ND guidance
│   ├── metrics/                        # Structural analysis (non-differentiable)
│   │   ├── core.py                     # Dataclasses (PDFMetrics, etc.)
│   │   ├── structural.py               # PDF, ADF computation
│   │   ├── geometric.py                # Coordination, dihedrals
│   │   ├── advanced.py                 # S(q), Voronoi
│   │   ├── errors.py                   # Error metrics for comparison
│   │   └── utils.py                    # JSON loading helpers
│   └── potentials/
│       └── torch_tersoff/              # Tersoff potential
│           ├── params.py               # Parameters
│           ├── neighbors.py            # Neighbor list
│           ├── potential.py            # Energy + forces
│           ├── ase_calc.py             # ASE Calculator
│           └── cli.py                  # MD, energy commands
└── tests/                              # Test suite
    ├── test_tersoff.py
    ├── test_metrics.py
    └── data/Si_2.5_00.xyz
```

## Common Commands

### Testing

```bash
# Full test suite (~3s, does not need lightning)
KMP_DUPLICATE_LIB_OK=TRUE pytest -v

# Single test
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_tersoff.py::test_snapshot_energy -v

# Metrics tests
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_metrics.py -v
```

### Training

```bash
# Create experiment and train score model
glass train ./my_experiment --model-type score --num-species 1

# Train spectral surrogate (EXAFS)
glass train ./my_experiment --model-type spec --spec-type exafs --num-species 1

# Resume training
glass train ./my_experiment --resume

# Override parameters
glass train ./my_experiment --max-epochs 5000 --lr 0.0005 --dim 256
```

### Generation

```bash
# Unconditional generation
glass generate ./my_experiment --inits ./inits/

# PDF-guided generation
glass generate ./my_experiment --inits ./inits/ \
    --guidance-type pdf --ref-path ./reference/

# XRD-guided generation
glass generate ./my_experiment --inits ./inits/ \
    --guidance-type xrd --ref-path ./reference/ \
    --element-names Si --rho 5
```

### Metrics

```bash
# Compute all metrics for a structure
glass metrics structure.xyz --output metrics.json

# Batch process
glass metrics ./structures/*.xyz --output metrics.json

# Compare two structures
glass compare ref.xyz target.xyz

# Compare using pre-computed metrics
glass compare ref.json target.json --from-json

# Individual metrics
glass pdf structure.xyz --output pdf.json
glass coordination structure.xyz --output coord.json
```

### Tersoff Potential

```bash
# Compute energy
KMP_DUPLICATE_LIB_OK=TRUE glass energy ./tests/data/Si_2.5_00.xyz

# Run MD
KMP_DUPLICATE_LIB_OK=TRUE glass md \
    --input ./tests/data/Si_2.5_00.xyz \
    --ensemble nve --steps 100 --timestep 1.0
```

## Implementation Details

### Experiment Structure

```
my_experiment/
├── config.yaml          # All parameters
├── data/                # Training data (*.xyz)
├── checkpoints/         # Model checkpoints
│   ├── best.ckpt
│   ├── last.ckpt
│   └── epoch_*.ckpt
├── inits/               # Initial structures for generation
├── outputs/             # Generated structures
└── logs/                # TensorBoard logs
```

### Denoising Flow

1. Pretrained `LitScoreNet` gives **prior score** (∇ log p)
2. `guidance_model` + reference defines **likelihood gradient**
3. Reverse SDE: `pos ← pos + (f(t)·pos − g²(t)·(prior + likelihood))·dt + g(t)·noise`
4. ASE ↔ tensor: `glass.lit.functions.get_atoms.initialize_atoms(atoms)`

### Metrics Module

**PDF Normalization**: `g(r) = (V/N²) × hist / (4πr²Δr)`
- Ensures g(r) → 1 as r → ∞ (bulk limit)
- Gaussian smoothing optional (default σ=0.15 Å)

**Coordination Cutoff**: Auto-detected from PDF first minimum
- Fallback: 1.3× first peak position

**Error Metrics**:
- PDF/ADF: RMSE, MAE, area between curves, cosine similarity, R-chi²
- Coordination: EMD (Wasserstein), histogram RMSE
- Peak: position error, height error

### Tersoff Potential Notes

- **Single-species constraint**: Only homogeneous `(A, A, A)` keys supported
- **ASE compatibility**: Energy matches ASE 3.25.0 formula
- **Forces**: Torch autograd (correct), not ASE analytical forces

### Environment Variables

- `KMP_DUPLICATE_LIB_OK=TRUE` — Required on macOS when mixing PyTorch and SciPy-backed ASE

## Dependencies

**Core**: `torch`, `ase`, `numpy`, `click`, `scipy`

**Training**: `lightning`, `torch-geometric`, `scikit-learn`

**Optional**:
- `[test]`: pytest
- `[plot]`: matplotlib, seaborn, tensorboard
- `[diffraction]`: DebyeCalculator (pins Python <3.12)

## Adding New Features

### New Guidance Type

1. Create `DifferentiableXXX` in `glass.lit/modules/`
2. Expose `forward(pos, species, cell) → predicted_feature`
3. Wire into `LikelihoodScore` in `cli.py`

### New Error Metric

1. Add function to `glass/metrics/errors.py`
2. Function signature: `metric_name(ref: Metrics, target: Metrics) → float`
3. Add to `compute_all_errors()`
4. Update CLI if needed

## Key Files

- `glass/cli.py` — Main CLI entry points
- `glass/experiment.py` — Experiment configuration management
- `glass/lit/modules/prior.py` — ScoreNet model
- `glass/metrics/` — All structural analysis
- `tests/test_metrics.py` — Metrics tests
- `tests/test_tersoff.py` — Tersoff tests
