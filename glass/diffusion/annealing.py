"""Simulated-annealing post-relaxation using the Tersoff potential.

Runs a short MD-like Cartesian trajectory that mixes the Tersoff gradient
(already normalised by ``N_atoms`` and clamped inside
``TersoffEnergyGuidance``) with a decaying thermal noise. The temperature
schedule is geometric:

    T_k = T0 * (T_end / T0) ** (k / (n_steps - 1)),   k = 0..n_steps-1

Only the Tersoff potential is used — the score net is not involved. This is
the "anneal tail" that runs AFTER the reverse SDE has produced a near-physical
configuration.

``tersoff_relax`` provides a zero-temperature geometry optimisation via the
ASE FIRE optimizer backed by ``TorchTersoffCalculator``. It is intended as an
inter-restart relaxation that drives the structure toward a local minimum of
the Tersoff PES before the next denoising pass begins.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor


def _per_atom_norm_clamp(vec: Tensor, max_norm: float) -> Tensor:
    if max_norm is None or max_norm <= 0:
        return vec
    norms = vec.norm(dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norms + 1e-12), max=1.0)
    return vec * scale


def _wrap_pbc(pos: Tensor, cell: Tensor) -> Tensor:
    """Wrap Cartesian positions back into the primitive cell.

    Assumes ``cell`` rows are the lattice vectors.
    """
    cell_inv = torch.linalg.inv(cell.to(pos.dtype))
    frac = pos @ cell_inv.T
    frac = frac - torch.floor(frac)
    return frac @ cell.to(pos.dtype).T


def _declash_atoms(atoms, d_min: float = 1.5, max_iter: int = 50) -> None:
    """Push apart near-coincident atom pairs in-place (minimum-image).

    Denoised configurations occasionally collapse two atoms to sub-Å
    separation, where the Tersoff energy overflows to inf/NaN. This iteratively
    separates every pair closer than ``d_min`` by displacing both atoms along
    their minimum-image vector until the minimum pair distance clears ``d_min``
    or ``max_iter`` is reached. Pure geometry — no potential is evaluated, so it
    is robust even when forces are already non-finite.
    """
    import numpy as _np

    cell = _np.array(atoms.cell)
    cell_inv = _np.linalg.inv(cell)
    for _ in range(max_iter):
        pos = atoms.get_positions()
        # Minimum-image pairwise displacement vectors.
        rij = pos[:, None, :] - pos[None, :, :]
        frac = rij @ cell_inv.T
        frac -= _np.round(frac)
        rij = frac @ cell.T
        dist = _np.linalg.norm(rij, axis=-1)
        N = len(atoms)
        _np.fill_diagonal(dist, _np.inf)
        iu, ju = _np.where(_np.triu(dist < d_min, k=1))
        if iu.size == 0:
            return
        disp = _np.zeros_like(pos)
        for i, j in zip(iu.tolist(), ju.tolist()):
            d = dist[i, j]
            # Direction from j to i; fall back to a fixed axis if coincident.
            vec = rij[i, j]
            if d < 1e-6:
                vec = _np.array([1.0, 0.0, 0.0])
                d = 1e-6
            push = 0.5 * (d_min - d) * vec / d
            disp[i] += push
            disp[j] -= push
        atoms.set_positions(pos + disp)
        atoms.wrap()


def simulated_anneal(
    pos: Tensor,
    cell: Tensor,
    species: Tensor,
    tersoff_guidance,
    n_steps: int = 100,
    T0: float = 1e-2,
    T_end: float = 1e-5,
    lr: float = 1e-3,
    lr_clamp: float = 0.2,
    wrap: bool = True,
) -> Tensor:
    """Run ``n_steps`` of simulated annealing on ``pos``.

    Args:
        pos: (N, 3) Cartesian positions (Å). Detached from any graph.
        cell: (3, 3) cell tensor (Å).
        species: (N,) or (N, S) species tensor.
        tersoff_guidance: ``TersoffEnergyGuidance`` instance (returns
            ``-grad E / N_atoms``, already clamped).
        n_steps: Number of annealing steps.
        T0: Initial temperature (variance of the noise term).
        T_end: Final temperature. Must be > 0 for the geometric decay;
            a very small value (e.g. 1e-8) approximates T=0.
        lr: Step size on the force term.
        lr_clamp: Hard cap on per-atom displacement per step (Å).
        wrap: If True, wrap positions back into the cell each step.

    Returns:
        (N, 3) final positions tensor (same device/dtype as ``pos``).
    """
    if n_steps <= 0:
        return pos.detach()
    if T0 <= 0 or T_end <= 0:
        raise ValueError("T0 and T_end must be positive.")

    pos = pos.detach().clone()

    log_ratio = math.log(T_end / T0)
    denom = max(n_steps - 1, 1)
    for k in range(n_steps):
        T_k = T0 * math.exp(log_ratio * k / denom)
        force = tersoff_guidance(pos, cell, species)
        noise_scale = math.sqrt(2.0 * T_k)
        delta = lr * force + noise_scale * torch.randn_like(pos)
        delta = _per_atom_norm_clamp(delta, lr_clamp)
        pos = pos + delta
        if wrap:
            pos = _wrap_pbc(pos, cell)

    return pos.detach()


def make_anneal_fn(
    tersoff_guidance,
    n_steps: int = 100,
    T0: float = 1e-2,
    T_end: float = 1e-5,
    lr: float = 1e-3,
    lr_clamp: float = 0.2,
    wrap: bool = True,
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """Return an ``anneal_fn(pos, cell, species) -> pos`` closure."""

    def _fn(pos: Tensor, cell: Tensor, species: Tensor) -> Tensor:
        return simulated_anneal(
            pos=pos,
            cell=cell,
            species=species,
            tersoff_guidance=tersoff_guidance,
            n_steps=n_steps,
            T0=T0,
            T_end=T_end,
            lr=lr,
            lr_clamp=lr_clamp,
            wrap=wrap,
        )

    return _fn


# ---------------------------------------------------------------------------
# Geometry optimisation with FIRE via ASE + TorchTersoffCalculator
# ---------------------------------------------------------------------------

def tersoff_relax(
    pos: Tensor,
    cell: Tensor,
    numbers: np.ndarray,
    fmax: float = 0.1,
    max_steps: int = 200,
    logfile: Optional[str] = None,
) -> Tensor:
    """Geometry-optimise ``pos`` under the Tersoff potential using ASE FIRE.

    Wraps positions into an ASE ``Atoms`` object, attaches a
    ``TorchTersoffCalculator``, and runs the FIRE minimiser until the
    maximum per-atom force drops below ``fmax`` eV/Å or ``max_steps`` is
    reached.  Only the Si single-species parameterisation is supported.

    Args:
        pos: (N, 3) Cartesian positions in Å, on any device.
        cell: (3, 3) cell tensor in Å.
        numbers: (N,) integer array of atomic numbers (e.g. ``atoms.numbers``).
        fmax: Force convergence threshold in eV/Å.
        max_steps: Maximum number of FIRE steps.
        logfile: Path for ASE optimiser log.  ``None`` suppresses output.

    Returns:
        (N, 3) relaxed positions tensor on CPU, same dtype as input.
    """
    import ase
    from ase.optimize import FIRE as ASE_FIRE

    from glass.potentials.tersoff.ase_calc import silicon_calculator

    pos_np = pos.detach().cpu().to(torch.float64).numpy()
    cell_np = cell.detach().cpu().to(torch.float64).numpy()

    atoms = ase.Atoms(
        numbers=numbers,
        positions=pos_np,
        cell=cell_np,
        pbc=[True, True, True],
    )
    atoms.calc = silicon_calculator(dtype=torch.float64, device="cpu")

    opt = ASE_FIRE(atoms, logfile=logfile)
    opt.run(fmax=fmax, steps=max_steps)

    relaxed_np = atoms.get_positions()
    return torch.tensor(relaxed_np, dtype=pos.dtype)


def make_relax_fn(
    numbers: np.ndarray,
    fmax: float = 0.5,
    max_steps: int = 200,
    logfile: Optional[str] = None,
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """Return a ``relax_fn(pos, cell, species) -> pos`` closure for ``tersoff_relax``.

    Args:
        numbers: (N,) integer array of atomic numbers, passed through to ASE.
            Obtained from ``init_atoms.numbers`` in the generation loop.
    """

    def _fn(pos: Tensor, cell: Tensor, species: Tensor) -> Tensor:
        return tersoff_relax(
            pos=pos,
            cell=cell,
            numbers=numbers,
            fmax=fmax,
            max_steps=max_steps,
            logfile=logfile,
        )

    return _fn


# ---------------------------------------------------------------------------
# Finite-temperature NVT molecular dynamics via ASE Langevin + Tersoff
# ---------------------------------------------------------------------------

def nvt_md(
    pos: Tensor,
    cell: Tensor,
    numbers: np.ndarray,
    temperature: float = 600.0,
    n_steps: int = 1000,
    timestep: float = 1.0,
    friction: float = 0.01,
    pre_relax_steps: int = 10,
    declash_d_min: float = 1.5,
    device: str = "cpu",
    seed: Optional[int] = None,
    progress_fn: Optional[Callable] = None,
    progress_interval: int = 1,
    logfile: Optional[str] = None,
) -> Tensor:
    """Run NVT molecular dynamics on ``pos`` under the Tersoff potential.

    Wraps positions into an ASE ``Atoms`` object, attaches a
    ``TorchTersoffCalculator``, seeds Maxwell-Boltzmann velocities at
    ``temperature``, and integrates ``n_steps`` of Langevin dynamics. Intended
    as a finite-temperature inter-restart relaxation that thermally equilibrates
    the structure before the next denoising pass begins. Only the Si
    single-species parameterisation is supported.

    Denoised/initial structures often contain a few close contacts (sub-1.3 Å
    pairs) that carry huge Tersoff forces (~200 eV/Å). Integrating MD straight
    from those would spike the kinetic energy to thousands of K and eject atoms
    several cell-lengths, scrambling the geometry. A short FIRE pre-relax
    (``pre_relax_steps``) drains those forces first so the subsequent MD stays
    well-behaved.

    Args:
        pos: (N, 3) Cartesian positions in Å, on any device.
        cell: (3, 3) cell tensor in Å.
        numbers: (N,) integer array of atomic numbers (e.g. ``atoms.numbers``).
        temperature: Target / initial temperature in Kelvin.
        n_steps: Number of MD steps (1000 steps × 1 fs = 1 ps).
        timestep: Integration timestep in fs.
        friction: Langevin friction coefficient in 1/fs.
        pre_relax_steps: Number of FIRE geometry-optimisation steps run before
            MD to remove close contacts. 0 disables both the declash and the
            pre-relax.
        declash_d_min: Minimum allowed interatomic distance (Å). Pairs closer
            than this are pushed apart (minimum-image) before the FIRE pre-relax
            so the Tersoff forces are finite. Only active when
            ``pre_relax_steps > 0``.
        device: torch device for the Tersoff calculator (e.g. ``cpu``, ``cuda``).
        seed: RNG seed for velocity initialisation and the Langevin thermostat.
            ``None`` lets numpy pick a fresh seed each call.
        progress_fn: Optional callback ``progress_fn(step, T)`` invoked every
            ``progress_interval`` steps with the step count and instantaneous
            temperature (K). Used to drive a progress bar.
        progress_interval: Stride (in MD steps) between ``progress_fn`` calls.
        logfile: Unused placeholder for API symmetry with ``tersoff_relax``.

    Returns:
        (N, 3) final positions tensor on CPU, same dtype as input.
    """
    if n_steps <= 0:
        return pos.detach()

    import ase
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.optimize import FIRE as ASE_FIRE

    from glass.potentials.tersoff.ase_calc import silicon_calculator

    pos_np = pos.detach().cpu().to(torch.float64).numpy()
    cell_np = cell.detach().cpu().to(torch.float64).numpy()

    atoms = ase.Atoms(
        numbers=numbers,
        positions=pos_np,
        cell=cell_np,
        pbc=[True, True, True],
    )
    atoms.calc = silicon_calculator(dtype=torch.float64, device=device)

    # Short pre-relax to drain close-contact forces before MD so the integrator
    # does not explode on the first few steps. Denoised structures can contain
    # near-coincident atom pairs (sub-Å), at which the Tersoff energy overflows
    # to inf/NaN forces — FIRE alone cannot recover from non-finite forces, so a
    # cheap geometric declash separates such pairs first to make forces finite.
    if pre_relax_steps and pre_relax_steps > 0:
        _declash_atoms(atoms, d_min=declash_d_min)
        ASE_FIRE(atoms, logfile=logfile).run(fmax=0.0, steps=int(pre_relax_steps))

    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)

    dyn = Langevin(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature,
        friction=friction / units.fs,
        rng=rng,
    )

    if progress_fn is not None:
        n_atoms = len(atoms)

        def _report(_dyn=dyn, _atoms=atoms, _n=n_atoms):
            T = _atoms.get_kinetic_energy() / (1.5 * units.kB * _n)
            progress_fn(step=_dyn.nsteps, T=T)

        dyn.attach(_report, interval=max(1, int(progress_interval)))

    dyn.run(n_steps)

    atoms.wrap()
    return torch.tensor(atoms.get_positions(), dtype=pos.dtype)


def make_nvt_md_fn(
    numbers: np.ndarray,
    temperature: float = 600.0,
    n_steps: int = 1000,
    timestep: float = 1.0,
    friction: float = 0.01,
    pre_relax_steps: int = 10,
    declash_d_min: float = 1.5,
    device: str = "cpu",
    seed: Optional[int] = None,
    logfile: Optional[str] = None,
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """Return an ``md_fn(pos, cell, species) -> pos`` closure for ``nvt_md``.

    Args:
        numbers: (N,) integer array of atomic numbers, passed through to ASE.
            Obtained from ``init_atoms.numbers`` in the generation loop.
    """

    def _fn(
        pos: Tensor,
        cell: Tensor,
        species: Tensor,
        progress_fn: Optional[Callable] = None,
        progress_interval: int = 1,
    ) -> Tensor:
        return nvt_md(
            pos=pos,
            cell=cell,
            numbers=numbers,
            temperature=temperature,
            n_steps=n_steps,
            timestep=timestep,
            friction=friction,
            pre_relax_steps=pre_relax_steps,
            declash_d_min=declash_d_min,
            device=device,
            seed=seed,
            progress_fn=progress_fn,
            progress_interval=progress_interval,
            logfile=logfile,
        )

    return _fn
