"""Tests for GuidanceProfiler and its integration with denoise_by_sde."""

import json
import math
import tempfile

import pytest
import torch

from glass.diffusion.profiler import GuidanceProfiler, StepRecord
from glass.diffusion.sampling import denoise_by_sde


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _FakeDiffuser:
    """Minimal diffuser: VPSDE-like with f=0, g=1, sigma=sqrt(t)."""

    def f(self, t):
        return torch.zeros(1)

    def g(self, t):
        return torch.ones(1)

    def g2(self, t):
        return torch.ones(1)

    def sigma(self, t):
        return t.sqrt()


def _const_score(sp, pos, cell, t, cutoff):
    return torch.ones_like(pos) * 0.1


def _const_guidance(pos, cell, species):
    return torch.ones_like(pos) * 0.5


N = 4
_species = torch.zeros(N, dtype=torch.long)
_pos = torch.rand(N, 3)
_cell = torch.eye(3) * 5.0
_diffuser = _FakeDiffuser()
_ts = torch.linspace(0.5, 0.1, 6)  # 5 steps


# ---------------------------------------------------------------------------
# GuidanceProfiler unit tests
# ---------------------------------------------------------------------------

def test_profiler_record_and_rms():
    profiler = GuidanceProfiler()
    prior = torch.ones(N, 3) * 2.0
    disp = torch.ones(N, 3)
    profiler.record(step=0, t=0.5, prior_score=prior, total_disp=disp)
    profiler.record(step=1, t=0.4, prior_score=prior * 0.5, total_disp=disp)

    rms = profiler.score_rms()
    assert len(rms["prior"]) == 2
    assert rms["tersoff"] == [None, None]
    assert rms["likelihood"] == [None, None]
    # prior RMS at step 0: all entries = 2.0, so RMS = 2.0
    assert abs(rms["prior"][0] - 2.0) < 1e-5


def test_profiler_per_atom_disp_norms():
    profiler = GuidanceProfiler()
    prior = torch.ones(N, 3)
    disp = torch.ones(N, 3) * 3.0
    tersoff = torch.ones(N, 3) * 0.2
    profiler.record(step=0, t=0.5, prior_score=prior, total_disp=disp, tersoff_score=tersoff)
    profiler.record(step=1, t=0.4, prior_score=prior, total_disp=disp, tersoff_score=tersoff)

    norms = profiler.per_atom_disp_norms()
    # total_disp per atom norm = sqrt(3*9)=sqrt(27); two steps → 2*sqrt(27)
    expected_disp = 2.0 * math.sqrt(27)
    assert abs(float(norms["total_disp"][0]) - expected_disp) < 1e-4
    # tersoff norm = sqrt(3*0.04)=sqrt(0.12); two steps
    expected_trs = 2.0 * math.sqrt(3 * 0.04)
    assert abs(float(norms["tersoff"][0]) - expected_trs) < 1e-4


def test_profiler_to_dict_serialisable():
    profiler = GuidanceProfiler()
    prior = torch.ones(N, 3)
    disp = torch.ones(N, 3)
    profiler.record(step=0, t=0.5, prior_score=prior, total_disp=disp)
    d = profiler.to_dict()
    # Must be JSON serialisable
    json.dumps(d)
    assert d["n_steps"] == 1
    assert "ts" in d
    assert "score_rms" in d
    assert "per_atom_disp_norms" in d


def test_profiler_save_json():
    profiler = GuidanceProfiler()
    prior = torch.ones(N, 3)
    disp = torch.ones(N, 3)
    profiler.record(step=0, t=0.5, prior_score=prior, total_disp=disp)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    profiler.save_json(path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["n_steps"] == 1


def test_profiler_summary_table():
    profiler = GuidanceProfiler()
    prior = torch.ones(N, 3) * 0.1
    disp = torch.ones(N, 3) * 0.01
    profiler.record(step=0, t=0.5, prior_score=prior, total_disp=disp)
    table = profiler.summary_table()
    assert "prior" in table
    assert "tersoff" in table
    assert "0.50000" in table


# ---------------------------------------------------------------------------
# Integration: denoise_by_sde with profiler
# ---------------------------------------------------------------------------

def test_denoise_produces_records_with_profiler():
    profiler = GuidanceProfiler()
    _, final = denoise_by_sde(
        species=_species,
        pos=_pos.clone(),
        cell=_cell,
        cutoff=5.0,
        score_fn=_const_score,
        likelihood_fn=None,
        ts=_ts,
        diffuser=_diffuser,
        profiler=profiler,
    )
    assert len(profiler.records) == len(_ts) - 1
    assert profiler.records[0].prior_score.shape == (N, 3)
    assert profiler.records[0].total_disp.shape == (N, 3)
    assert profiler.records[0].tersoff_score is None
    assert profiler.records[0].likelihood_score is None


def test_denoise_profiler_captures_tersoff():
    profiler = GuidanceProfiler()

    def _sched(t):
        return 1.0

    _, _ = denoise_by_sde(
        species=_species,
        pos=_pos.clone(),
        cell=_cell,
        cutoff=5.0,
        score_fn=_const_score,
        likelihood_fn=None,
        ts=_ts,
        diffuser=_diffuser,
        tersoff_guidance=_const_guidance,
        tersoff_schedule=_sched,
        profiler=profiler,
    )
    # All steps should have a tersoff_score
    assert all(r.tersoff_score is not None for r in profiler.records)
    assert profiler.records[0].tersoff_score.shape == (N, 3)


def test_denoise_profiler_captures_likelihood():
    profiler = GuidanceProfiler()

    def _likelihood(sp, pos, cell, t, cutoff):
        return torch.ones_like(pos) * 0.3, torch.ones(1)

    _, _ = denoise_by_sde(
        species=_species,
        pos=_pos.clone(),
        cell=_cell,
        cutoff=5.0,
        score_fn=_const_score,
        likelihood_fn=_likelihood,
        ts=_ts,
        diffuser=_diffuser,
        profiler=profiler,
    )
    assert all(r.likelihood_score is not None for r in profiler.records)
    assert profiler.records[0].likelihood_score.shape == (N, 3)


def test_denoise_none_profiler_identical_output():
    """With profiler=None the output should be bit-identical to no-profiler call."""
    torch.manual_seed(42)
    pos_a = _pos.clone()
    _, out_a = denoise_by_sde(
        species=_species, pos=pos_a, cell=_cell, cutoff=5.0,
        score_fn=_const_score, likelihood_fn=None, ts=_ts, diffuser=_diffuser,
        profiler=None,
    )
    torch.manual_seed(42)
    pos_b = _pos.clone()
    _, out_b = denoise_by_sde(
        species=_species, pos=pos_b, cell=_cell, cutoff=5.0,
        score_fn=_const_score, likelihood_fn=None, ts=_ts, diffuser=_diffuser,
        profiler=GuidanceProfiler(),
    )
    assert torch.allclose(out_a, out_b)
