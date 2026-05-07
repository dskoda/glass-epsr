# torch-tersoff

A PyTorch reimplementation of the LAMMPS-style Tersoff interatomic potential,
with an ASE `Calculator` wrapper and a small CLI for running molecular
dynamics.

Energies are computed in PyTorch (float64 by default); forces are obtained via
`torch.autograd.grad` on the positions, so they are the exact analytical
gradient of the energy expression.

## Install

```bash
pip install -e .
```

Runtime dependencies: `torch`, `ase`, `numpy`, `click`.

On macOS you may need `KMP_DUPLICATE_LIB_OK=TRUE` when mixing PyTorch and
SciPy-backed ASE in the same process. The CLI and the test file set this
automatically; you only need to set it explicitly for ad-hoc Python scripts.

## Python API

```python
from ase.build import bulk
from torch_tersoff import TorchTersoffCalculator, TersoffParameters

si_params = {
    ("Si", "Si", "Si"): TersoffParameters(
        A=3264.7, B=95.373,
        lambda1=3.2394, lambda2=1.3258, lambda3=1.3258,
        beta=0.33675, gamma=1.0, m=3.0, n=22.956,
        c=4.8381, d=2.0417, h=0.0,
        R=3.0, D=0.2,
    )
}

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = TorchTersoffCalculator(si_params)

print(atoms.get_potential_energy())
print(atoms.get_forces())
```

A pre-parameterized Si calculator is also available:

```python
from torch_tersoff import silicon_calculator
atoms.calc = silicon_calculator()
```

## Command line

After installing, the `torch-tersoff` command is available:

```bash
# Single-point energy and max |force|
torch-tersoff energy tests/data/Si_2.5_00.xyz

# NVE MD, 100 steps of 1 fs, 300 K initial temperature
torch-tersoff md --input tests/data/Si_2.5_00.xyz \
    --ensemble nve --timestep 1.0 --steps 100 \
    --temperature 300 --output md.traj

# NVT (Langevin) MD
torch-tersoff md --input tests/data/Si_2.5_00.xyz \
    --ensemble nvt --timestep 1.0 --steps 1000 \
    --temperature 1000 --friction 0.01 --output md.traj
```

The CLI uses `ase.md.verlet.VelocityVerlet` (NVE) or `ase.md.langevin.Langevin`
(NVT) under the hood, with the PyTorch Tersoff calculator attached to the
`Atoms` object.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The suite (6 tests, runs in a few seconds) checks:

- Energy agreement with `ase.calculators.tersoff.Tersoff` on a Si diamond
  cell and on a 216-atom disordered Si snapshot.
- Autograd forces against a central finite-difference of our own torch
  energy on a random subset of atoms. See the note below on why we do not
  compare against ASE's analytical forces.
- Internal consistency between our autograd and analytical force paths.
- Translation invariance of the energy.
- Neighbor-list sanity (four nearest neighbors in diamond Si).

## Notes

- Only single-species, homogeneous `(A, A, A)` parameter keys are supported
  in the current version. The API mirrors ASE's Tersoff so multi-species can
  be added later.
- The neighbor list is torch-native and assumes orthorhombic cells.
- The installed `ase.calculators.tersoff` (3.25.0) computes the bond-order
  exponential as `arg = lambda3 * (r_ij - r_ik)**m`; this package matches
  that energy. ASE's own **analytical** forces are inconsistent with that
  energy (the derivative code was not updated), so `Atoms.get_forces()` from
  ASE disagrees with ASE's numerical derivative of its own energy by up to
  ~7 eV/Å on the disordered snapshot. Torch autograd here is the correct
  gradient.
