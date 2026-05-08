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
    
    Returns:
        (traj, final_pos) where traj is None if save_traj is False,
        otherwise a list of position tensors
    """
    ts = ts.to(pos.device).view(-1, 1)
    f, g, g2 = diffuser.f, diffuser.g, diffuser.g2
    
    traj = [pos.detach().cpu().clone()] if save_traj else None
    pos = pos.detach()
    
    for i, t in enumerate(ts[1:]):
        dt = ts[i + 1] - ts[i]
        eps = dt.abs().sqrt() * torch.randn_like(pos)
        
        with torch.no_grad():
            p_score = score_fn(species, pos, cell, t, cutoff)
        
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
                )
        
        pos = (pos + disp).detach()
        if save_traj:
            traj.append(pos.cpu().clone())
    
    return traj, pos
