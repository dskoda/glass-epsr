# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A single Python package `glass` (defined by `./pyproject.toml`) that combines:

1. **`glass.lit`** — score-based generative model for reconstructing amorphous structures. Training, unconditional denoising, and guided denoising via spectral/structural observables (pdf, adf, xrd, nd, exafs, xanes). Requires `lightning` (not installed by default in this env).
2. **`glass.potentials.torch_tersoff`** — PyTorch reimplementation of the LAMMPS-style Tersoff interatomic potential. Single-species Si, autograd-based forces. Exports `TorchTersoff`, `TorchTersoffCalculator` (ASE `Calculator` subclass), and `silicon_calculator()`.

The Tersoff potential is intended to be wired into the denoising workflow as additional guidance (energy minimization) and as an optional post-denoising relaxation step.

## Common commands

```bash
pip install -e .

# Full test suite (fast — ~3 s, does not need lightning)
KMP_DUPLICATE_LIB_OK=TRUE pytest -v

# Single test
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_tersoff.py::test_snapshot_energy -v

# Tersoff CLI (works without lightning)
KMP_DUPLICATE_LIB_OK=TRUE python -m glass.potentials.torch_tersoff.cli energy ./tests/data/Si_2.5_00.xyz
KMP_DUPLICATE_LIB_OK=TRUE python -m glass.potentials.torch_tersoff.cli md \
    --input ./tests/data/Si_2.5_00.xyz --ensemble nve --steps 100 --timestep 1.0

# Full glass CLI (requires lightning installed separately)
KMP_DUPLICATE_LIB_OK=TRUE glass --help
# Tersoff md/energy are registered as subcommands of the glass group:
KMP_DUPLICATE_LIB_OK=TRUE glass energy ./tests/data/Si_2.5_00.xyz
KMP_DUPLICATE_LIB_OK=TRUE glass md --input ... --ensemble nve ...
```

`KMP_DUPLICATE_LIB_OK=TRUE` is needed on macOS when mixing PyTorch and SciPy-backed ASE in the same process. The CLI and the test file set this automatically; it's only needed explicitly for ad-hoc Python scripts.

## Architecture

### Package layout

```
./
├── pyproject.toml                      # single package definition: `glass`
├── glass/
│   ├── cli.py                          # click group `glass`: train/denoise + md/energy
│   ├── lit/                            # Lightning training + denoising code
│   │   ├── datamodules/                # StructureSpecDataModule (PyG Data pipeline)
│   │   ├── functions/get_atoms.py      # initialize_atoms(): ASE -> (species one-hot, pos, cell) tensors
│   │   └── modules/
│   │       ├── prior.py                # LitScoreNet (score-based SDE denoiser)
│   │       ├── forward.py              # LitSpecNet (per-atom spectral surrogate)
│   │       ├── differentiable_rdf.py   # DifferentiableRDF guidance
│   │       ├── differentiable_adf.py   # DifferentiableADF guidance
│   │       ├── differentiable_xrd.py   # DifferentiableXRD guidance
│   │       └── differentiable_nd.py    # DifferentiableND guidance
│   └── potentials/
│       └── torch_tersoff/              # PyTorch Tersoff potential
│           ├── params.py               # TersoffParameters dataclass (14 floats, LAMMPS order)
│           ├── neighbors.py            # build_neighbors: torch-native, orthorhombic
│           ├── potential.py            # TorchTersoff.energy + autograd/analytical forces
│           ├── ase_calc.py             # TorchTersoffCalculator (ASE Calculator subclass)
│           └── cli.py                  # Click commands `md`, `energy` (also attached to glass group)
└── tests/
    ├── test_tersoff.py                 # 6 tests validating the Tersoff implementation
    └── data/Si_2.5_00.xyz              # 216-atom disordered Si snapshot
```

### Denoising flow (glass.cli.cond_denoise, cli.py:1170–1549)

- A pretrained `LitScoreNet` gives the **prior score** (∇ log p of atomic structures).
- A `guidance_model` + reference target (from `--ref-path` or `--exp-data`) defines a **likelihood gradient**: `-(ρ/‖·‖) · ∂‖target − pred‖²/∂pos`.
- `_denoise_by_sde` runs a reverse SDE step: `pos ← pos + (f(t)·pos − g²(t)·(prior + likelihood))·dt + g(t)·noise`.
- Positions: `float32 (N, 3)` in Cartesian coords; cell: `float32 (3, 3)`; pbc always True.
- ASE ↔ tensor conversion: `glass.lit.functions.get_atoms.initialize_atoms(atoms) → (Z_list, one_hot_species, pos, cell)`.

### Adding a new guidance (planned work)

All existing guidance modules expose `forward(pos, species, cell)` returning a predicted feature. A Tersoff-based guidance (`DifferentiableTersoff`) would wrap `TorchTersoff.energy` and return a scalar energy. It plugs into `LikelihoodScore` (cli.py:1313–1358). The plan file at `/Users/dskoda/.claude/plans/the-current-repository-has-stateful-karp.md` describes the intended refactor to support combinable (list-of-specs) guidance and an optional post-denoising ASE relaxation.

### Single-species Tersoff constraint

`TorchTersoff.__init__` explicitly rejects parameter dicts with more than one key or with non-homogeneous `(A, A, A)` keys. Any multi-species extension has to change that check plus the energy code (currently `self.params` is a single object looked up once, not per-pair).

### ASE-compatibility quirk (non-obvious)

The installed `ase.calculators.tersoff` (3.25.0) computes the bond-order exponential as `arg = lambda3 * (r_ij - r_ik)**m` in `calc_bond_order` (line ~416). Our energy matches the **installed** ASE formula. For Si diamond `r_ij == r_ik` so the distinction is invisible; for the disordered 216-atom snapshot it accounts for a ~3.6 eV energy difference.

ASE's own **analytical** forces are inconsistent with its energy (the derivative code was not updated to match the energy change), so ASE's `get_forces()` disagrees with ASE's own `calculate_numerical_forces()` by up to ~6.8 eV/Å on the snapshot. Torch autograd is the correct gradient. Tests validate autograd against a finite-difference of our own torch energy on a random subset of atoms — do **not** replace this with a comparison against ASE's analytical forces.

### Test fixtures

`./tests/data/` holds `Si_2.5_00.xyz` (216-atom disordered Si snapshot, orthorhombic cell, PBC on). The test file imports ASE **before** torch to avoid an OpenMP init abort on macOS.

### Packaging

Single distribution: `./pyproject.toml` defines the `glass` package with console script `glass = "glass.cli:glass"`. Runtime deps: `torch`, `ase`, `numpy`, `click`, `lightning`, `torch-geometric`, `scikit-learn`, `scipy`. Two small pieces of functionality (`glass.nn`, `glass.diffusion`) were ported from the LLNL `graphite` package so glass has no external graphite dependency. Optional extras: `[test]` for pytest, `[plot]` for `matplotlib`/`seaborn`/`tensorboard`, `[diffraction]` for `DebyeCalculator` (kept optional because it pins Python `<3.12` and would block install on newer Python versions).
