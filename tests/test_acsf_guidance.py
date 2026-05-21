"""Tests for ACSF descriptors and structural-entropy guidance."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math

import pytest
import torch

from glass.descriptors import EntropyGuidance, EntropySchedule, TorchACSF
from glass.diffusion import VarianceExplodingDiffuser
from glass.diffusion.sampling import denoise_by_sde


DTYPE = torch.float64


def _diamond_si(a: float = 5.43) -> tuple[torch.Tensor, torch.Tensor]:
    """8-atom diamond-Si conventional cell (no species needed)."""
    cell = a * torch.eye(3, dtype=DTYPE)
    base = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
        dtype=DTYPE,
    )
    pos = base @ cell
    return pos, cell


def test_torch_acsf_for_silicon_shape():
    pos, cell = _diamond_si()
    acsf = TorchACSF.for_silicon().to(DTYPE)
    D = acsf(pos, cell)
    # G1 (1) + G2 (4) + G4 (2) = 7
    assert D.shape == (8, 7)
    assert torch.isfinite(D).all()


def test_translational_invariance():
    pos, cell = _diamond_si()
    acsf = TorchACSF.for_silicon().to(DTYPE)
    D0 = acsf(pos, cell)
    shift = torch.tensor([1.7, -0.3, 2.1], dtype=DTYPE)
    D1 = acsf(pos + shift, cell)
    assert torch.allclose(D0, D1, atol=1e-8)


def test_rotational_invariance():
    pos, cell = _diamond_si()
    acsf = TorchACSF.for_silicon().to(DTYPE)
    D0 = acsf(pos, cell)
    theta = 0.7
    R = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
    )
    D1 = acsf(pos @ R.T, cell @ R.T)
    assert torch.allclose(D0, D1, atol=1e-7)


def test_acsf_gradcheck():
    # Use a tiny perturbed cell so the cutoff sphere is nontrivial.
    torch.manual_seed(0)
    pos, cell = _diamond_si(a=5.43)
    pos = pos + 0.1 * torch.randn_like(pos)
    pos = pos.detach().requires_grad_(True)
    acsf = TorchACSF(
        r_cut=4.0,
        g2_params=((1.0, 1.0), (1.0, 2.0)),
        g4_params=((0.005, 1.0, 1.0),),
        include_g1=True,
    ).to(DTYPE)

    def f(p):
        return acsf(p, cell).sum()

    assert torch.autograd.gradcheck(f, (pos,), eps=1e-4, atol=1e-4)


def test_entropy_guidance_shape_finite():
    torch.manual_seed(1)
    pos, cell = _diamond_si()
    pos = pos + 0.05 * torch.randn_like(pos)
    acsf = TorchACSF.for_silicon().to(DTYPE)
    guidance = EntropyGuidance(acsf, clamp_norm=None).to(DTYPE)
    g = guidance(pos, cell)
    assert g.shape == pos.shape
    assert torch.isfinite(g).all()


def test_entropy_descent_decreases_variance():
    torch.manual_seed(2)
    pos, cell = _diamond_si()
    pos = pos + 0.20 * torch.randn_like(pos)  # heterogeneous
    acsf = TorchACSF.for_silicon().to(DTYPE)
    guidance = EntropyGuidance(acsf, clamp_norm=None).to(DTYPE)

    def loss_of(p):
        return acsf(p, cell).var(dim=0).mean().item()

    L0 = loss_of(pos)
    eta = 0.05
    p = pos.clone()
    for _ in range(50):
        # guidance returns -grad; descent direction is +guidance.
        p = p + eta * guidance(p, cell)
    L1 = loss_of(p)
    assert L1 < L0, f"variance did not decrease: {L0} -> {L1}"


def test_denoise_by_sde_with_entropy_smoke():
    diffuser = VarianceExplodingDiffuser(k=0.8)
    pos, cell = _diamond_si()
    pos = pos.to(torch.float32)
    cell = cell.to(torch.float32)
    species = torch.zeros(pos.shape[0], dtype=torch.long)
    ts = torch.linspace(0.3, 1e-3, 10)

    acsf = TorchACSF.for_silicon()
    entropy_fn = EntropyGuidance(acsf, clamp_norm=10.0)
    schedule = EntropySchedule(
        schedule="constant", lambda_0=0.1, tmax=0.3, t_gate=1.0
    )

    def zero_score(sp, p, c, t, co):
        return torch.zeros_like(p)

    torch.manual_seed(3)
    _, final = denoise_by_sde(
        species,
        pos.clone(),
        cell,
        cutoff=5.0,
        score_fn=zero_score,
        likelihood_fn=None,
        ts=ts,
        diffuser=diffuser,
        entropy_guidance=entropy_fn,
        entropy_schedule=schedule,
    )
    assert final.shape == pos.shape
    assert torch.isfinite(final).all()


def test_entropy_schedule_t_gate():
    sch = EntropySchedule(
        schedule="constant", lambda_0=2.0, tmax=1.0, t_gate=0.3
    )
    assert sch(0.1) == 2.0
    assert sch(0.5) == 0.0  # gated off
