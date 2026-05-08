"""SDE sampling utilities for structure generation.

This module provides functions for running the reverse SDE to denoise structures.
"""

import torch
from torch import Tensor
from typing import Callable, Optional, List, Tuple


def denoise_by_sde(
    species: Tensor,
    pos: Tensor,
    cell: Tensor,
    cutoff: float,
    score_fn: Callable,
    likelihood_fn: Optional[Callable],
    ts: Tensor,
    diffuser,
    save_traj: bool = False,
    progress_fn: Optional[Callable] = None,
    tersoff_guidance: Optional[Callable] = None,
    tersoff_schedule: Optional[Callable] = None,
) -> Tuple[Optional[List[Tensor]], Tensor]:
    """Run reverse SDE to denoise atomic positions.

    Args:
        species: Atomic species tensor [N,]
        pos: Atomic positions tensor [N, 3]
        cell: Unit cell tensor [3, 3]
        cutoff: Graph cutoff radius
        score_fn: Score function (species, pos, cell, t, cutoff) -> score
        likelihood_fn: Likelihood function for conditional generation, or None
        ts: Time steps tensor [n_steps]
        diffuser: Diffuser object with f, g, g2 functions
        save_traj: If True, return full trajectory
        progress_fn: Optional callback for progress updates (step, t, metrics)
        tersoff_guidance: Optional callable (pos, cell, species) -> (N, 3)
            returning a per-atom "force-like" vector that points toward lower
            Tersoff energy. Added to the score at each step with weight
            ``tersoff_schedule(t)``.
        tersoff_schedule: Optional callable t -> float producing the
            time-dependent weight lambda(t). Required when
            ``tersoff_guidance`` is provided.

    Returns:
        (traj, final_pos) where traj is None if save_traj is False,
        otherwise a list of position tensors
    """
    ts = ts.to(pos.device).view(-1, 1)
    f, g, g2 = diffuser.f, diffuser.g, diffuser.g2

    traj = [pos.detach().cpu().clone()] if save_traj else None
    pos = pos.detach()

    if tersoff_guidance is not None and tersoff_schedule is None:
        raise ValueError(
            "tersoff_schedule must be provided when tersoff_guidance is set."
        )

    for i, t in enumerate(ts[1:]):
        dt = ts[i + 1] - ts[i]
        eps = dt.abs().sqrt() * torch.randn_like(pos)

        with torch.no_grad():
            p_score = score_fn(species, pos, cell, t, cutoff)

        t_score = None
        if tersoff_guidance is not None:
            lam = float(tersoff_schedule(t))
            if lam != 0.0:
                guidance_vec = tersoff_guidance(pos.detach(), cell, species)
                t_score = lam * guidance_vec
                p_score = p_score + t_score

        if likelihood_fn is not None:
            l_score, norm = likelihood_fn(species, pos, cell, t, cutoff)
            disp = (f(t) * pos - g2(t) * (p_score + l_score)) * dt + g(t) * eps
            if progress_fn is not None:
                progress_fn(
                    step=i,
                    t=t.item(),
                    p_norm=p_score.norm().item(),
                    l_norm=l_score.norm().item(),
                    target_norm=norm.sum().item(),
                )
        else:
            disp = (f(t) * pos - g2(t) * p_score) * dt + g(t) * eps
            if progress_fn is not None:
                progress_fn(
                    step=i,
                    t=t.item(),
                    p_norm=p_score.norm().item(),
                    t_norm=(t_score.norm().item() if t_score is not None else None),
                )

        pos = (pos + disp).detach()
        if save_traj:
            traj.append(pos.cpu().clone())

    return traj, pos
