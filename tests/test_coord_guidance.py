import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read

from glass.diffusion import VarianceExplodingDiffuser
from glass.diffusion.sampling import denoise_by_sde
from glass.lit.modules.coord_guidance import (
    DifferentiableCoordinationNumber,
    CoordinationLoss,
    CoordinationGuidance,
    CoordinationSchedule,
)
from glass.metrics.geometric import compute_coordination

DATA_DIR = Path(__file__).resolve().parent / "data"


def _atoms_to_tensors(atoms, dtype=torch.float64):
    pos = torch.tensor(atoms.get_positions(), dtype=dtype)
    cell = torch.tensor(np.array(atoms.cell), dtype=dtype)
    species = torch.zeros(pos.shape[0], dtype=torch.long)
    return pos, cell, species


# ----------------------------------------------------------------------------
# A. DifferentiableCoordinationNumber
# ----------------------------------------------------------------------------


def test_coord_matches_neighbor_list():
    atoms = read(DATA_DIR / "CRN.xyz")
    pos, cell, species = _atoms_to_tensors(atoms)

    r_cut = 2.85
    coord_fn = DifferentiableCoordinationNumber(r_cut=r_cut, smear=0.01)
    soft = coord_fn(pos, cell, species).detach().numpy()

    integer = compute_coordination(atoms, cutoff=r_cut).coordination_numbers

    diff = np.abs(soft - integer)
    assert diff.max() < 0.05, (diff.max(), diff.mean())
    assert abs(soft.mean() - 4.0) < 0.1, soft.mean()


def test_coord_shape_finite():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, species = _atoms_to_tensors(atoms)

    coord_fn = DifferentiableCoordinationNumber(r_cut=2.85, smear=0.3)
    coord = coord_fn(pos, cell, species)

    assert coord.shape == (pos.shape[0],)
    assert torch.isfinite(coord).all()


def test_coord_gradcheck():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")[:16]
    pos, cell, species = _atoms_to_tensors(atoms, dtype=torch.float64)

    coord_fn = DifferentiableCoordinationNumber(r_cut=2.85, smear=0.4)

    def fn(p):
        return coord_fn(p, cell, species).sum()

    pos.requires_grad_(True)
    assert torch.autograd.gradcheck(fn, (pos,), eps=1e-4, atol=1e-3, rtol=1e-3)


def test_coord_pbc_invariance():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, species = _atoms_to_tensors(atoms)

    coord_fn = DifferentiableCoordinationNumber(r_cut=2.85, smear=0.3)
    c0 = coord_fn(pos, cell, species)

    shift = 0.5 * cell.sum(dim=0)
    c1 = coord_fn(pos + shift, cell, species)

    assert torch.allclose(c0, c1, atol=1e-5)


# ----------------------------------------------------------------------------
# B. CoordinationLoss
# ----------------------------------------------------------------------------


def test_loss_low_penalty_monotone_and_bounded_grad():
    loss = CoordinationLoss(
        n_low=4.0, w_low=1.0, k_low=4.0,
        w_target=0.0, w_high=0.0,
    )

    a = torch.tensor([5.0, 4.0, 3.0, 2.0])
    b = torch.tensor([4.0, 4.0, 4.0, 4.0])
    assert loss(a) > loss(b)

    # Gradient bounds at four characteristic points: deep violation,
    # boundary, inside violation, well above threshold.
    coords = torch.tensor([0.0, 4.0, 3.0, 10.0], requires_grad=True)
    # Use sum (not mean) to get per-atom gradients directly comparable
    # to the analytic value.
    per_atom = loss._low(coords) * 1.0  # weight already 1
    per_atom.sum().backward()
    g = coords.grad

    # Bounded in [-1, 0] everywhere.
    assert (g <= 1e-12).all()
    assert (g >= -1.0 - 1e-6).all()
    # At c=0: gradient saturated near -1.
    assert g[0].item() == pytest.approx(-1.0, abs=1e-3)
    # At c=n_low (boundary): gradient = -0.5 (non-vanishing).
    assert g[1].item() == pytest.approx(-0.5, abs=1e-3)


def test_loss_target_zero_at_match_and_bounded_grad():
    sigma = 0.5
    loss = CoordinationLoss(
        n_target=4.0, sigma_target=sigma, w_target=1.0,
        w_low=0.0, w_high=0.0,
    )

    on_target = torch.full((4,), 4.0)
    assert loss(on_target).item() == pytest.approx(0.0, abs=1e-7)

    # Symmetric around target.
    above = loss(torch.tensor([4.5]))
    below = loss(torch.tensor([3.5]))
    assert above.item() == pytest.approx(below.item(), rel=1e-6)

    # Gradient bounded by sigma far from target.
    far = torch.tensor([4.0 + 100.0, 4.0 - 100.0], requires_grad=True)
    loss._target(far).sum().backward()
    g = far.grad
    assert g[0].item() == pytest.approx(sigma, abs=5e-3)
    assert g[1].item() == pytest.approx(-sigma, abs=5e-3)


def test_loss_high_penalty_symmetric():
    loss = CoordinationLoss(
        n_high=7.0, w_high=1.0, k_high=4.0,
        w_low=0.0, w_target=0.0,
    )

    a = torch.tensor([6.0, 7.0, 8.0, 9.0])
    b = torch.tensor([7.0, 7.0, 7.0, 7.0])
    assert loss(a) > loss(b)

    coords = torch.tensor([100.0, 7.0, 0.0], requires_grad=True)
    loss._high(coords).sum().backward()
    g = coords.grad

    assert (g >= -1e-12).all()
    assert (g <= 1.0 + 1e-6).all()
    assert g[0].item() == pytest.approx(1.0, abs=1e-3)
    assert g[1].item() == pytest.approx(0.5, abs=1e-3)


def test_loss_combined_additive():
    coords = torch.tensor([2.0, 4.0, 6.0, 9.0])

    full = CoordinationLoss(
        n_target=4.0, sigma_target=0.5, w_target=2.0,
        n_low=4.0, w_low=3.0, k_low=4.0,
        n_high=7.0, w_high=5.0, k_high=4.0,
    )
    only_target = CoordinationLoss(
        n_target=4.0, sigma_target=0.5, w_target=2.0,
        w_low=0.0, w_high=0.0,
    )
    only_low = CoordinationLoss(
        n_low=4.0, w_low=3.0, k_low=4.0,
        w_target=0.0, w_high=0.0,
    )
    only_high = CoordinationLoss(
        n_high=7.0, w_high=5.0, k_high=4.0,
        w_low=0.0, w_target=0.0,
    )

    expected = only_target(coords) + only_low(coords) + only_high(coords)
    assert full(coords).item() == pytest.approx(expected.item(), rel=1e-6)


# ----------------------------------------------------------------------------
# C. CoordinationGuidance
# ----------------------------------------------------------------------------


def _make_guidance(**loss_kwargs):
    coord_fn = DifferentiableCoordinationNumber(r_cut=2.85, smear=0.3)
    loss_fn = CoordinationLoss(**loss_kwargs)
    return CoordinationGuidance(coord_fn, loss_fn, clamp_norm=1e9)


def test_guidance_shape_finite():
    atoms = read(DATA_DIR / "CRN.xyz")
    pos, cell, species = _atoms_to_tensors(atoms)

    guide = _make_guidance(
        n_target=4.0, sigma_target=0.5, w_target=1.0,
        w_low=0.0, w_high=0.0,
    )
    out = guide(pos, cell, species)

    assert out.shape == pos.shape
    assert torch.isfinite(out).all()


def test_guidance_clamp():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, species = _atoms_to_tensors(atoms)

    coord_fn = DifferentiableCoordinationNumber(r_cut=2.85, smear=0.3)
    loss_fn = CoordinationLoss(
        n_low=4.0, w_low=10.0, k_low=4.0,
        w_target=0.0, w_high=0.0,
    )
    clamp = 0.5
    guide = CoordinationGuidance(coord_fn, loss_fn, clamp_norm=clamp)
    out = guide(pos, cell, species)

    assert float(out.norm(dim=-1).max()) <= clamp + 1e-6


def test_guidance_descends_target_loss():
    atoms = read(DATA_DIR / "CRN.xyz")
    pos, cell, species = _atoms_to_tensors(atoms)

    coord_fn = DifferentiableCoordinationNumber(r_cut=2.85, smear=0.3)
    loss_fn = CoordinationLoss(
        n_target=4.0, sigma_target=0.5, w_target=1.0,
        w_low=0.0, w_high=0.0,
    )
    guide = CoordinationGuidance(coord_fn, loss_fn, clamp_norm=1e9)

    L0 = loss_fn(coord_fn(pos, cell, species)).item()
    direction = guide(pos, cell, species).to(pos.dtype)
    pos_new = pos + 1e-2 * direction
    L1 = loss_fn(coord_fn(pos_new, cell, species)).item()

    assert L1 < L0, (L0, L1)


def test_guidance_low_pulls_neighbour_in():
    # Two atoms in a large cell. B is just outside r_cut so A's coord is ~0.
    cell = 50.0 * torch.eye(3, dtype=torch.float64)
    r_cut = 2.85
    delta = 0.5  # B sits inside the cosine ramp but past the inflection.
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [r_cut + delta, 0.0, 0.0]],
        dtype=torch.float64,
    )
    species = torch.zeros(2, dtype=torch.long)

    coord_fn = DifferentiableCoordinationNumber(r_cut=r_cut, smear=0.6)
    loss_fn = CoordinationLoss(
        n_low=4.0, w_low=10.0, k_low=4.0,
        w_target=0.0, w_high=0.0,
    )
    guide = CoordinationGuidance(coord_fn, loss_fn, clamp_norm=1e9)
    g = guide(pos, cell, species)

    direction_b_from_a = pos[1] - pos[0]
    # A's guidance should point toward B (positive x component).
    dot = float((g[0] * direction_b_from_a).sum())
    assert dot > 0.0, (g[0], direction_b_from_a)


def test_guidance_high_pushes_neighbours_out():
    # 9 atoms inside r_cut around a central atom -> high-coord violation.
    r_cut = 2.85
    cell = 50.0 * torch.eye(3, dtype=torch.float64)
    np.random.seed(0)
    central = np.array([0.0, 0.0, 0.0])
    # Place 9 atoms uniformly in a sphere of radius r_cut - 0.3.
    rs = (r_cut - 0.3) * np.random.rand(9) ** (1.0 / 3.0)
    thetas = np.arccos(2.0 * np.random.rand(9) - 1.0)
    phis = 2.0 * np.pi * np.random.rand(9)
    nb = np.stack(
        [
            rs * np.sin(thetas) * np.cos(phis),
            rs * np.sin(thetas) * np.sin(phis),
            rs * np.cos(thetas),
        ],
        axis=1,
    )
    pos = torch.tensor(np.vstack([central, nb]), dtype=torch.float64)
    species = torch.zeros(pos.shape[0], dtype=torch.long)

    coord_fn = DifferentiableCoordinationNumber(r_cut=r_cut, smear=0.3)
    loss_fn = CoordinationLoss(
        n_high=7.0, w_high=10.0, k_high=4.0,
        w_low=0.0, w_target=0.0,
    )
    guide = CoordinationGuidance(coord_fn, loss_fn, clamp_norm=1e9)
    g = guide(pos, cell, species)

    # Neighbours should be pushed away from the central atom.
    rel = pos[1:] - pos[0:1]
    dots = (g[1:] * rel).sum(dim=-1)
    assert float(dots.sum()) > 0.0, dots


# ----------------------------------------------------------------------------
# D. CoordinationSchedule
# ----------------------------------------------------------------------------


def test_schedule_constant_linear_sigmoid():
    lam0, tmax, t_gate, k = 0.2, 1.0, 0.3, 50.0

    const = CoordinationSchedule(schedule="constant", lambda_0=lam0, tmax=tmax)
    assert const(0.0) == pytest.approx(lam0)
    assert const(tmax) == pytest.approx(lam0)

    lin = CoordinationSchedule(schedule="linear", lambda_0=lam0, tmax=tmax)
    assert lin(0.0) == pytest.approx(lam0)
    assert lin(tmax / 2) == pytest.approx(lam0 * 0.5)
    assert lin(tmax) == pytest.approx(0.0)

    sig = CoordinationSchedule(
        schedule="sigmoid", lambda_0=lam0, tmax=tmax, t_gate=t_gate, k=k
    )
    assert sig(0.0) == pytest.approx(lam0, rel=1e-3)
    assert sig(t_gate) == pytest.approx(lam0 / 2.0, rel=1e-3)
    assert sig(tmax) == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------------
# E. Integration with denoise_by_sde
# ----------------------------------------------------------------------------


def _zero_score(species, pos, cell, t, cutoff):
    return torch.zeros_like(pos)


def test_sampling_with_coord_guidance_runs():
    diffuser = VarianceExplodingDiffuser(k=0.8)
    torch.manual_seed(0)
    n = 8
    L = 6.0
    species = torch.zeros(n, dtype=torch.long)
    pos = L * torch.rand(n, 3)
    cell = L * torch.eye(3)
    ts = torch.linspace(0.5, 1e-3, 5)

    coord_fn = DifferentiableCoordinationNumber(r_cut=2.85, smear=0.3)
    loss_fn = CoordinationLoss(
        n_target=4.0, sigma_target=0.5, w_target=1.0,
        w_low=0.0, w_high=0.0,
    )
    guide = CoordinationGuidance(coord_fn, loss_fn, clamp_norm=10.0)
    sched = CoordinationSchedule(schedule="constant", lambda_0=0.05, tmax=0.5)

    _, final = denoise_by_sde(
        species, pos.clone(), cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None, ts=ts, diffuser=diffuser,
        coord_guidance=guide, coord_schedule=sched,
    )
    assert final.shape == pos.shape
    assert torch.isfinite(final).all()


def test_sampling_coord_requires_schedule():
    diffuser = VarianceExplodingDiffuser(k=0.8)
    n = 4
    L = 5.0
    species = torch.zeros(n, dtype=torch.long)
    pos = L * torch.rand(n, 3)
    cell = L * torch.eye(3)
    ts = torch.linspace(0.5, 1e-3, 4)

    coord_fn = DifferentiableCoordinationNumber(r_cut=2.85, smear=0.3)
    loss_fn = CoordinationLoss(w_target=1.0)
    guide = CoordinationGuidance(coord_fn, loss_fn)

    with pytest.raises(ValueError):
        denoise_by_sde(
            species, pos, cell, cutoff=5.0,
            score_fn=_zero_score, likelihood_fn=None, ts=ts, diffuser=diffuser,
            coord_guidance=guide, coord_schedule=None,
        )
