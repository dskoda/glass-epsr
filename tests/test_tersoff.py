import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

# Import ASE before torch to avoid OpenMP init conflict on macOS.
from ase.build import bulk
from ase.calculators.tersoff import Tersoff as AseTersoff
from ase.calculators.tersoff import TersoffParameters as AseTersoffParameters
from ase.io import read

import numpy as np
import torch

from glass.potentials.tersoff import TersoffParameters, TorchTersoff
from glass.potentials.tersoff.neighbors import build_neighbors

DATA_DIR = Path(__file__).resolve().parent / "data"

SI_PARAMS_KW = dict(
    A=3264.7,
    B=95.373,
    lambda1=3.2394,
    lambda2=1.3258,
    lambda3=1.3258,
    beta=0.33675,
    gamma=1.00,
    m=3.00,
    n=22.956,
    c=4.8381,
    d=2.0417,
    h=0.0000,
    R=3.00,
    D=0.20,
)


def ase_calc():
    params = {("Si", "Si", "Si"): AseTersoffParameters(**SI_PARAMS_KW)}
    return AseTersoff(params)


def torch_calc():
    params = {("Si", "Si", "Si"): TersoffParameters(**SI_PARAMS_KW)}
    return TorchTersoff(params)


def _atoms_to_tensors(atoms, dtype=torch.float64):
    pos = torch.tensor(atoms.get_positions(), dtype=dtype)
    cell = torch.tensor(np.array(atoms.cell), dtype=dtype)
    pbc = tuple(bool(p) for p in atoms.pbc)
    return pos, cell, pbc


def test_diamond_energy():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = ase_calc()
    E_ase = atoms.get_potential_energy()

    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()
    E_torch = tc.energy(pos, cell, pbc).item()

    assert abs(E_torch - E_ase) < 1e-8, (E_torch, E_ase)


def test_snapshot_energy():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()

    E = tc.energy(pos, cell, pbc).item()

    assert abs(E + 936.3422548925636) < 1e-8, E


def test_snapshot_forces_autograd_vs_finite_diff():
    # Validate the autograd gradient against a central finite difference of
    # our own torch energy, on a small random subset of atoms. ASE's builtin
    # calculate_numerical_forces would also work but runs ~1300 ASE energy
    # evaluations (~2 minutes); doing the finite diff through our torch
    # energy is the same check and finishes in under a second.
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()

    _, F_torch = tc.energy_and_forces_autograd(pos, cell, pbc)

    rng = np.random.default_rng(0)
    sample_atoms = rng.choice(pos.shape[0], size=5, replace=False)
    h = 1e-4
    for a in sample_atoms:
        for d in range(3):
            pp = pos.clone()
            pp[a, d] += h
            E_plus = tc.energy(pp, cell, pbc).item()
            pp[a, d] -= 2 * h
            E_minus = tc.energy(pp, cell, pbc).item()
            fd = -(E_plus - E_minus) / (2 * h)
            assert abs(float(F_torch[a, d]) - fd) < 1e-5, (a, d, fd, F_torch[a, d])


def test_snapshot_forces_analytical_vs_autograd():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()

    _, F_auto = tc.energy_and_forces_autograd(pos, cell, pbc)
    _, F_ana = tc.energy_and_forces_analytical(pos, cell, pbc)

    max_diff = (F_auto - F_ana).abs().max().item()
    assert max_diff < 1e-6, max_diff


def test_translation_invariance():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()
    E1 = tc.energy(pos, cell, pbc).item()

    shift = torch.tensor([0.37, -1.21, 2.83], dtype=torch.float64)
    E2 = tc.energy(pos + shift, cell, pbc).item()

    assert abs(E1 - E2) < 1e-8


def test_neighbor_count_diamond():
    atoms = bulk("Si", "diamond", a=5.43)
    pos, cell, pbc = _atoms_to_tensors(atoms)
    R, D = SI_PARAMS_KW["R"], SI_PARAMS_KW["D"]
    i_idx, j_idx, _ = build_neighbors(pos, cell, pbc, R - D)
    counts = torch.bincount(i_idx, minlength=pos.shape[0])
    assert int(counts.min()) == 4
    assert int(counts.max()) == 4
