"""Iterated SDEdit-style refinement on the Tersoff PES.

Given a near-physical structure ``x_0`` produced by the reverse SDE,
re-noise it to an intermediate level, run a partial reverse SDE back
to ``t=0``, then optionally run a Tersoff SA tail. Repeat until the
positions stop changing (or for a fixed number of cycles).

Each cycle is two operations:

    1. Forward q(x_{t*} | x_0):  x_{t*} = x_0 + sigma(t*) · eps
    2. Partial reverse SDE from t=t* back to t=t_min, on the original
       schedule (subset of ts). Optionally followed by SA on Tersoff.

The cycle does NOT change the cell (NVT-like). The same score / Tersoff
/ likelihood plumbing is used as in ``denoise_by_sde``.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor

from glass.diffusion.sampling import denoise_by_sde


def iterated_refine(
    species: Tensor,
    pos: Tensor,
    cell: Tensor,
    cutoff: float,
    score_fn: Callable,
    likelihood_fn: Optional[Callable],
    ts_full: Tensor,
    diffuser,
    *,
    t_star_frac: float = 0.2,
    n_cycles: int = 5,
    rmsd_tol: float = 0.05,
    tersoff_guidance: Optional[Callable] = None,
    tersoff_schedule: Optional[Callable] = None,
    n_corr: int = 0,
    corr_step_size: float = 0.15,
    corr_use_tersoff: bool = True,
    corr_t_gate: float = 0.6,
    anneal_fn: Optional[Callable] = None,
    progress_fn: Optional[Callable] = None,
) -> Tuple[Tensor, List[dict]]:
    """Run iterated SDEdit + Tersoff polishing on `pos`.

    Args:
        species, pos, cell, cutoff, score_fn, likelihood_fn,
            tersoff_guidance, tersoff_schedule, n_corr, corr_step_size,
            corr_use_tersoff, corr_t_gate, anneal_fn:
                Same as ``denoise_by_sde``. ``anneal_fn`` (Tersoff SA)
                runs at the end of every cycle.
        ts_full: The full forward-noise time grid the diffusion was trained
            on (so the partial reverse path matches the training distribution
            of t).
        t_star_frac: Fraction of ``ts_full.max()`` to use as the re-noise
            level each cycle. Smaller = lighter editing, larger = closer
            to a full re-generation.
        n_cycles: Maximum number of cycles. The loop also stops early
            when the per-atom RMSD between consecutive cycles drops
            below ``rmsd_tol``.
        rmsd_tol: Convergence threshold (Å, per-atom RMSD).
        progress_fn: Optional callback invoked with each cycle's record.

    Returns:
        (final_pos, cycle_log) where cycle_log is a list of dicts with
        per-cycle ``rmsd``, ``t_star``, and ``n_steps`` info.
    """
    if n_cycles <= 0:
        return pos.detach(), []

    ts_full = ts_full.to(pos.device).view(-1)
    t_max = float(ts_full.max().item())
    t_star = float(t_star_frac) * t_max

    # Subset of ts: include only steps with t <= t_star; we need the
    # smallest t (the floor) and the largest t below t_star, in
    # ascending order so the reverse loop walks from high t to low t.
    # ``denoise_by_sde`` iterates ts[1:], so the first entry must equal
    # the desired starting t (which the loop then steps DOWN from on
    # subsequent iterations) — see sampling.py:80-82 for the dt sign.
    # The original ts comes from power_law_ts as ASCENDING, so we
    # match that convention.
    mask = ts_full <= t_star + 1e-9
    ts_sub = ts_full[mask]
    if ts_sub.numel() < 2:
        # Not enough steps to do meaningful refinement; keep pos as-is.
        return pos.detach(), [{
            "cycle": 0, "skipped": True,
            "reason": f"t_star={t_star} too small for ts_full grid",
        }]

    cycle_log: List[dict] = []
    pos_prev = pos.detach()
    pos = pos.detach().clone()

    for cycle in range(1, n_cycles + 1):
        # 1. Forward noise to t_star
        t_star_tensor = ts_sub[-1].clone()
        sigma = float(diffuser.sigma(t_star_tensor))
        noise = torch.randn_like(pos)
        pos_noisy = pos + sigma * noise

        # 2. Partial reverse SDE from t_star -> t_min, including SA tail
        _, pos_new = denoise_by_sde(
            species=species,
            pos=pos_noisy,
            cell=cell,
            cutoff=cutoff,
            score_fn=score_fn,
            likelihood_fn=likelihood_fn,
            ts=ts_sub,
            diffuser=diffuser,
            save_traj=False,
            tersoff_guidance=tersoff_guidance,
            tersoff_schedule=tersoff_schedule,
            n_corr=n_corr,
            corr_step_size=corr_step_size,
            corr_use_tersoff=corr_use_tersoff,
            corr_t_gate=corr_t_gate,
            anneal_fn=anneal_fn,
        )

        # Per-atom RMSD vs previous cycle
        diff = pos_new.detach() - pos_prev.detach()
        rmsd = float(diff.pow(2).sum(dim=-1).mean().sqrt())

        rec = {
            "cycle": cycle,
            "t_star": float(t_star),
            "n_steps": int(ts_sub.numel() - 1),
            "rmsd": rmsd,
        }
        cycle_log.append(rec)
        if progress_fn is not None:
            progress_fn(rec)

        pos_prev = pos.detach()
        pos = pos_new.detach()

        if rmsd < rmsd_tol:
            cycle_log[-1]["converged"] = True
            break

    return pos, cycle_log
