# GLASS — Generative Learning for Atomic Structure Simulation

GLASS is a Python package and CLI tool for training and running score-based generative models for atomic glass structures. It supports unconditional and guided (conditional) denoising using structural and spectral observables.

---

## Installation

```bash
git clone git@github.com:digital-synthesis-lab/GLASS.git
cd GLASS
pip install -e .
```

After installation, the `glass` command is available in your terminal.

**Optional dependencies** for full functionality:

```bash
# For plotting and visualization
pip install -e ".[plot]"

# For diffraction calculations (requires Python <3.12)
pip install -e ".[diffraction]"
```

---

## Quick Start

### 1. Train a Score Model

```bash
# Create an experiment and train
glass train ./my_experiment --model-type score --num-species 1

# Resume training
glass train ./my_experiment --resume
```

### 2. Generate Structures

```bash
# Prepare initial structures
mkdir -p ./my_experiment/inits/
cp initial_structure.xyz ./my_experiment/inits/

# Generate unconditionally
glass generate ./my_experiment --inits ./my_experiment/inits/
```

### 3. Guided Generation

```bash
# Generate with PDF guidance
glass generate ./my_experiment --inits ./my_experiment/inits/ \
    --guidance-type pdf --ref-path ./reference/

# Generate with XRD guidance
glass generate ./my_experiment --inits ./my_experiment/inits/ \
    --guidance-type xrd --ref-path ./reference/ --element-names Si
```

---

## Main Commands

### Training

| Command | Description |
|---------|-------------|
| `glass train <experiment> --model-type score` | Train score-based generative model |
| `glass train <experiment> --model-type spec --spec-type exafs` | Train EXAFS surrogate model |
| `glass train <experiment> --resume` | Resume from last checkpoint |

### Generation

| Command | Description |
|---------|-------------|
| `glass generate <experiment> --inits <path>` | Unconditional denoising |
| `glass generate <experiment> --inits <path> --guidance-type pdf --ref-path <path>` | PDF-guided generation |
| `glass generate <experiment> --inits <path> --guidance-type xrd --element-names <names>` | XRD-guided generation |

**Guidance types**: `pdf`, `adf`, `xrd`, `nd`, `exafs`, `xanes`

### Structural Analysis

| Command | Description |
|---------|-------------|
| `glass metrics <structures...>` | Compute structural metrics (PDF, ADF, coordination, etc.) |
| `glass compare <ref> <target>` | Compare two structures and compute error metrics |
| `glass pdf <structure>` | Compute PDF only |
| `glass coordination <structure>` | Compute coordination numbers only |

### Tersoff Potential

| Command | Description |
|---------|-------------|
| `glass energy <structure.xyz>` | Compute Tersoff energy |
| `glass md --input <structure.xyz> --ensemble nve --steps 100` | Run MD simulation |

---

## Experiment Structure

Experiments are organized in self-contained folders:

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

---

## Features

### Generative Modeling
- **Score-based diffusion models** for atomic structures
- **Unconditional denoising** for structure generation
- **Guided denoising** with experimental observables:
  - Pair Distribution Function (PDF)
  - Angular Distribution Function (ADF)
  - X-ray/Neutron Diffraction (XRD/ND)
  - EXAFS/XANES spectroscopy

### Structural Analysis
- **PDF** with automatic peak detection and coordination cutoff
- **ADF** with dominant angle identification
- **Coordination numbers** with automatic cutoff detection
- **Dihedral angles**, **structure factor S(q)**, **Voronoi analysis**
- **Error metrics** for comparing structures (RMSE, cosine similarity, EMD)

### Molecular Dynamics
- **Tersoff potential** implementation with autograd-based forces
- **MD simulations** (NVE, NVT, NPT ensembles)
- **Energy minimization** for structure relaxation

---

## Environment Setup

On macOS, set this environment variable when using PyTorch with ASE:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

## Documentation

- **AGENTS.md** — Detailed technical documentation for developers
- **CLAUDE.md** — Same as AGENTS.md (for Claude Code)

---

## Citation

If you use GLASS in your research, please cite:

```bibtex
@software{glass2024,
  title={GLASS: Generative Learning for Atomic Structure Simulation},
  author={[Authors]},
  year={2024},
  url={https://github.com/digital-synthesis-lab/GLASS}
}
```

---

## License

[License information here]
