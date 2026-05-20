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
    n_corr: int = 0,
    corr_step_size: float = 0.15,
    corr_use_tersoff: bool = True,
    corr_t_gate: float = 0.6,
    anneal_fn: Optional[Callable] = None,
    tersoff_tweedie: bool = True,
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
        n_corr: Number of Langevin corrector steps per predictor step. Zero
            disables the corrector (default).
        corr_step_size: Langevin step size. Effective step is
            ``corr_step_size * sigma(t)**2``.
        corr_use_tersoff: If True, the Tersoff guidance term is added to the
            score inside each corrector step.
        corr_t_gate: Skip the corrector whenever ``t > corr_t_gate * t_max``
            of the trajectory, to avoid blow-ups at high noise.
        anneal_fn: Optional callable (pos, cell, species) -> pos, run once
            after the main reverse-SDE loop for post-relaxation.
        tersoff_tweedie: If True (default), evaluate Tersoff on the Tweedie
            denoised estimate x̂₀ = x_t + σ²·score rather than on the noisy
            x_t directly. The Tweedie estimate is a cleaner proxy for the
            clean structure, so the potential energy surface is less distorted
            by diffusion noise. Set to False to recover the legacy behaviour
            (evaluate Tersoff on noisy positions).

    Returns:
        (traj, final_pos) where traj is None if save_traj is False,
        otherwise a list of position tensors
    """
    ts = ts.to(pos.device).view(-1, 1)
    f, g, g2 = diffuser.f, diffuser.g, diffuser.g2
    t_max = float(ts.max().item())

    traj = [pos.detach().cpu().clone()] if save_traj else None
    pos = pos.detach()

    if tersoff_guidance is not None and tersoff_schedule is None:
        raise ValueError(
            "tersoff_schedule must be provided when tersoff_guidance is set."
        )
    if n_corr > 0 and score_fn is None:
        raise ValueError("score_fn is required when n_corr > 0.")

    last_idx = len(ts) - 2  # index of the last executed step inside the loop
    for i, t in enumerate(ts[1:]):
        dt = ts[i + 1] - ts[i]
        eps = dt.abs().sqrt() * torch.randn_like(pos)

        with torch.no_grad():
            p_score = score_fn(species, pos, cell, t, cutoff)

        t_score = None
        if tersoff_guidance is not None:
            lam = float(tersoff_schedule(t))
            if lam != 0.0:
                if tersoff_tweedie:
                    sigma_t = diffuser.sigma(t)
                    tersoff_pos = (pos + sigma_t ** 2 * p_score).detach()
                else:
                    tersoff_pos = pos
                guidance_vec = tersoff_guidance(tersoff_pos, cell, species)
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

        # Langevin corrector: tighten local structure at the current noise
        # level. Gated off at high t (unstable) and on the final step
        # (sigma -> 0 makes the step meaningless).
        if (
            n_corr > 0
            and i < last_idx
            and float(t.item()) <= corr_t_gate * t_max
        ):
            sigma_t = float(diffuser.sigma(t))
            eps_c = corr_step_size * (sigma_t ** 2)
            if eps_c > 0.0:
                noise_coef = (2.0 * eps_c) ** 0.5
                for _ in range(n_corr):
                    with torch.no_grad():
                        c_score = score_fn(species, pos, cell, t, cutoff)
                    if (
                        corr_use_tersoff
                        and tersoff_guidance is not None
                    ):
                        lam_c = float(tersoff_schedule(t))
                        if lam_c != 0.0:
                            if tersoff_tweedie:
                                sigma_t_c = diffuser.sigma(t)
                                tersoff_pos_c = (pos + sigma_t_c ** 2 * c_score).detach()
                            else:
                                tersoff_pos_c = pos
                            c_score = c_score + lam_c * tersoff_guidance(
                                tersoff_pos_c, cell, species
                            )
                    pos = (
                        pos
                        + eps_c * c_score
                        + noise_coef * torch.randn_like(pos)
                    ).detach()

        if save_traj:
            traj.append(pos.cpu().clone())

    if anneal_fn is not None:
        pos = anneal_fn(pos, cell, species).detach()
        if save_traj:
            traj.append(pos.cpu().clone())

    return traj, pos
