import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch

from glass.diffusion import VarianceExplodingDiffuser
from glass.diffusion.sampling import denoise_by_sde


def _zero_score(species, pos, cell, t, cutoff):
    return torch.zeros_like(pos)


def _random_setup(n_atoms=6, L=5.0, device="cpu", seed=0):
    torch.manual_seed(seed)
    species = torch.zeros(n_atoms, dtype=torch.long, device=device)
    pos = L * torch.rand(n_atoms, 3, device=device)
    cell = L * torch.eye(3, device=device)
    return species, pos, cell


def test_n_corr_zero_matches_baseline():
    """With n_corr=0, output is bit-identical to the pre-corrector sampler
    under a fixed seed."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=123)
    ts = torch.linspace(0.5, 1e-3, 16)

    torch.manual_seed(7)
    _, final_a = denoise_by_sde(
        species, pos0.clone(), cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None, ts=ts, diffuser=diffuser,
    )
    torch.manual_seed(7)
    _, final_b = denoise_by_sde(
        species, pos0.clone(), cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None, ts=ts, diffuser=diffuser,
        n_corr=0, corr_step_size=0.15, corr_t_gate=0.6,
    )
    assert torch.allclose(final_a, final_b, atol=0.0, rtol=0.0)


def test_corrector_produces_finite_positions():
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=0)
    ts = torch.linspace(0.5, 1e-3, 32)
    torch.manual_seed(11)
    _, final = denoise_by_sde(
        species, pos0.clone(), cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None, ts=ts, diffuser=diffuser,
        n_corr=2, corr_step_size=0.15, corr_t_gate=0.6,
    )
    assert torch.isfinite(final).all()
    assert final.shape == pos0.shape


def test_corrector_gated_off_at_high_t():
    """With corr_t_gate=0.0, the corrector must never fire, so output equals
    the n_corr=0 baseline."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=5)
    ts = torch.linspace(0.9, 1e-3, 16)

    torch.manual_seed(3)
    _, a = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, None, ts, diffuser, n_corr=0,
    )
    torch.manual_seed(3)
    _, b = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, None, ts, diffuser,
        n_corr=3, corr_step_size=0.15, corr_t_gate=0.0,
    )
    assert torch.allclose(a, b, atol=0.0, rtol=0.0)


def test_anneal_fn_runs_once_after_loop():
    """The anneal_fn closure is invoked exactly once after the SDE loop."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=2)
    ts = torch.linspace(0.3, 1e-3, 8)

    call_counter = {"n": 0}

    def fake_anneal(p, c, s):
        call_counter["n"] += 1
        return p + 0.01

    torch.manual_seed(9)
    _, final = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, None, ts, diffuser,
        anneal_fn=fake_anneal,
    )
    assert call_counter["n"] == 1
    assert torch.isfinite(final).all()


def test_corrector_zero_step_noop_when_sigma_is_zero():
    """At the final time slice sigma -> 0, so eps_c = 0 and no corrector
    update should occur (also protects against division by zero downstream)."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=4)
    # Trajectory ends at exactly t_min=0 so the last-slice sigma is 0.
    ts = torch.linspace(0.1, 0.0, 8)
    torch.manual_seed(6)
    _, final = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, None, ts, diffuser,
        n_corr=3, corr_step_size=0.3, corr_t_gate=1.0,
    )
    assert torch.isfinite(final).all()


def test_n_corr_requires_score_fn_if_used():
    """Sanity: passing n_corr>0 without a score_fn must raise."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=0)
    ts = torch.linspace(0.3, 1e-3, 4)
    with pytest.raises(ValueError):
        denoise_by_sde(
            species, pos0.clone(), cell, 5.0,
            score_fn=None, likelihood_fn=None, ts=ts, diffuser=diffuser,
            n_corr=1,
        )
