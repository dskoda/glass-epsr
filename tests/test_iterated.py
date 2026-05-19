import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch

from glass.diffusion import VarianceExplodingDiffuser
from glass.diffusion.iterated import iterated_refine


def _zero_score(species, pos, cell, t, cutoff):
    return torch.zeros_like(pos)


def _setup(n_atoms=6, L=5.0, seed=0):
    torch.manual_seed(seed)
    species = torch.zeros(n_atoms, dtype=torch.long)
    pos = L * torch.rand(n_atoms, 3)
    cell = L * torch.eye(3)
    return species, pos, cell


def test_zero_cycles_returns_input():
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos, cell = _setup(seed=1)
    ts = torch.linspace(0.5, 1e-3, 16)
    final, log = iterated_refine(
        species=species, pos=pos.clone(), cell=cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None,
        ts_full=ts, diffuser=diffuser,
        n_cycles=0,
    )
    assert torch.equal(final, pos)
    assert log == []


def test_iterated_runs_to_completion():
    """Smoke: with a zero score and no SA, n_cycles cycles execute and
    return a finite tensor of the right shape."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos, cell = _setup(seed=2)
    ts = torch.linspace(1e-3, 0.5, 16)  # ascending, like power_law_ts
    torch.manual_seed(0)
    final, log = iterated_refine(
        species=species, pos=pos.clone(), cell=cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None,
        ts_full=ts, diffuser=diffuser,
        t_star_frac=0.4, n_cycles=3, rmsd_tol=0.0,
    )
    assert torch.isfinite(final).all()
    assert final.shape == pos.shape
    # rmsd_tol=0 forces all 3 cycles to run.
    assert len(log) == 3
    for rec in log:
        assert "rmsd" in rec
        assert "t_star" in rec
        assert rec["n_steps"] >= 1


def test_iterated_early_stops_on_convergence():
    """A huge rmsd_tol terminates after a single cycle."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos, cell = _setup(seed=3)
    ts = torch.linspace(1e-3, 0.5, 16)
    torch.manual_seed(0)
    final, log = iterated_refine(
        species=species, pos=pos.clone(), cell=cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None,
        ts_full=ts, diffuser=diffuser,
        t_star_frac=0.4, n_cycles=10, rmsd_tol=1e9,
    )
    # tol is huge → first cycle's rmsd is below it, converges immediately.
    assert len(log) == 1
    assert log[0].get("converged") is True
    assert torch.isfinite(final).all()


def test_iterated_calls_anneal_each_cycle():
    """The anneal_fn closure (Tersoff SA tail) runs once per cycle."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos, cell = _setup(seed=4)
    ts = torch.linspace(1e-3, 0.5, 16)
    n_calls = {"n": 0}

    def fake_anneal(p, c, s):
        n_calls["n"] += 1
        return p

    torch.manual_seed(0)
    final, log = iterated_refine(
        species=species, pos=pos.clone(), cell=cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None,
        ts_full=ts, diffuser=diffuser,
        t_star_frac=0.4, n_cycles=4, rmsd_tol=0.0,
        anneal_fn=fake_anneal,
    )
    assert n_calls["n"] == 4
    assert torch.isfinite(final).all()


def test_iterated_t_star_too_small_skips_with_log():
    """If t_star_frac is so small that ts has < 2 entries, return input
    with a skipped marker."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos, cell = _setup(seed=5)
    ts = torch.linspace(0.5, 0.99, 4)  # tiny grid, all > t_star
    final, log = iterated_refine(
        species=species, pos=pos.clone(), cell=cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None,
        ts_full=ts, diffuser=diffuser,
        t_star_frac=0.0, n_cycles=3,
    )
    assert torch.equal(final, pos)
    assert len(log) == 1
    assert log[0].get("skipped") is True
