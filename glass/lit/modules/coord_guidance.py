"""Differentiable coordination-number guidance for the reverse SDE.

Provides three composable building blocks plus a schedule:

- ``DifferentiableCoordinationNumber`` — soft per-atom coordination via a
  cosine switching function. Replaces the integer count from ASE neighbour
  lists with a fully autograd-friendly value.
- ``CoordinationLoss`` — three penalty modes that share a single coord
  tensor: a softplus hinge on low-coord violations, a pseudo-Huber pull
  toward a target value, and a softplus hinge on high-coord violations.
  The hinge gradient is bounded in ``[-1, 0]`` for low and ``[0, 1]`` for
  high (so it neither vanishes near the boundary nor explodes at extreme
  violations); the target gradient is bounded by ``±sigma_target``.
- ``CoordinationGuidance`` — produces a per-atom force-like score
  ``-grad(loss) / N_atoms`` matching the ``TersoffEnergyGuidance``
  contract, ready to add to the score in ``denoise_by_sde``.
- ``CoordinationSchedule`` — time-dependent weight ``lambda(t)``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _cosine_switch(dists: Tensor, r_cut: float, w: float) -> Tensor:
    """Cosine switching function used to count neighbours smoothly.

    f(r) = 1                                if r < r_cut - w
         = ½(1 + cos(π · (r - (r_cut - w)) / (2w)))   if r_cut - w ≤ r < r_cut + w
         = 0                                if r ≥ r_cut + w

    Bounded support keeps the per-atom coord finite and lets us mask out
    far pairs without breaking autograd. ``w`` controls how soft the
    cutoff is — small ``w`` recovers a hard step, large ``w`` makes the
    counting more permissive.
    """
    r0 = r_cut - w
    r1 = r_cut + w
    inner = (dists - r0) / (2.0 * w)
    inner = inner.clamp(min=0.0, max=1.0)
    f = 0.5 * (1.0 + torch.cos(math.pi * inner))
    f = torch.where(dists < r0, torch.ones_like(f), f)
    f = torch.where(dists >= r1, torch.zeros_like(f), f)
    return f


class DifferentiableCoordinationNumber(nn.Module):
    """Soft coordination number per atom, fully differentiable in positions.

    For each atom ``i``, returns ``c_i = Σ_{j≠i} f(r_ij)`` where
    ``r_ij`` is the minimum-image distance and ``f`` is the cosine
    switch. Self-pairs are masked by a small distance threshold.

    Args:
        r_cut: Cutoff radius (Å). Use the first PDF minimum of the
            target system (e.g. 2.85 for amorphous Si).
        smear: Half-width of the cosine switch (Å). Larger values make
            the count smoother; smaller values approach a hard step.
    """

    def __init__(self, r_cut: float = 2.85, smear: float = 0.30) -> None:
        super().__init__()
        if smear <= 0.0:
            raise ValueError(f"smear must be positive, got {smear}")
        self.r_cut = float(r_cut)
        self.smear = float(smear)

    def forward(
        self, pos: Tensor, cell: Tensor, species: Tensor | None = None
    ) -> Tensor:
        # Minimum-image pairwise distances.
        rij = pos[:, None, :] - pos[None, :, :]
        cell_inv = torch.inverse(cell.to(rij.dtype))
        frac = rij @ cell_inv.T
        frac = frac - frac.round()
        rij_pbc = frac @ cell.to(rij.dtype).T
        dists = torch.norm(rij_pbc, dim=-1)

        # Mask self-pairs and pairs outside the cosine support.
        self_mask = dists > 1e-5
        f = _cosine_switch(dists, self.r_cut, self.smear)
        f = f * self_mask
        return f.sum(dim=-1)


class CoordinationLoss(nn.Module):
    """Per-atom coordination loss with three composable penalty modes.

    All three modes are evaluated on the same coord tensor and added with
    user weights. The mean over atoms is returned as a scalar loss.

    **Low-coord hinge** (modes 1 / 3):
        ``L_low(c)  = softplus(k_low  · (n_low  - c)) / k_low``
        ``L_high(c) = softplus(k_high · (c - n_high)) / k_high``

    The gradient w.r.t. ``c`` is a sigmoid bounded in ``[-1, 0]`` for
    ``L_low`` and ``[0, 1]`` for ``L_high`` — non-vanishing at the
    threshold (≈ ±0.5) and saturating at the extreme (±1), so residual
    violations keep getting pushed without ever exploding.

    **Target pull** (mode 2):
        ``L_target(c) = σ² · (sqrt(1 + ((c - n_target)/σ)²) - 1)``

    Pseudo-Huber: quadratic near the target (≈ ½(c - n_target)²) and
    linear far from it (gradient asymptote ±σ). A heavy outlier cannot
    overpower a low/high hinge.

    Args:
        n_target, sigma_target, w_target: target-mode parameters.
        n_low, w_low, k_low: low-coord hinge parameters.
        n_high, w_high, k_high: high-coord hinge parameters.
    """

    def __init__(
        self,
        n_target: float = 4.0,
        sigma_target: float = 0.5,
        w_target: float = 1.0,
        n_low: float = 4.0,
        w_low: float = 0.0,
        k_low: float = 4.0,
        n_high: float = 7.0,
        w_high: float = 0.0,
        k_high: float = 4.0,
    ) -> None:
        super().__init__()
        if sigma_target <= 0:
            raise ValueError("sigma_target must be > 0")
        if k_low <= 0 or k_high <= 0:
            raise ValueError("k_low and k_high must be > 0")
        self.n_target = float(n_target)
        self.sigma_target = float(sigma_target)
        self.w_target = float(w_target)
        self.n_low = float(n_low)
        self.w_low = float(w_low)
        self.k_low = float(k_low)
        self.n_high = float(n_high)
        self.w_high = float(w_high)
        self.k_high = float(k_high)

    @staticmethod
    def _softplus_hinge(x: Tensor, k: float) -> Tensor:
        # softplus(k * x) / k, evaluated stably for large positive x.
        return nn.functional.softplus(k * x) / k

    def _low(self, coord: Tensor) -> Tensor:
        return self._softplus_hinge(self.n_low - coord, self.k_low)

    def _high(self, coord: Tensor) -> Tensor:
        return self._softplus_hinge(coord - self.n_high, self.k_high)

    def _target(self, coord: Tensor) -> Tensor:
        s = self.sigma_target
        z = (coord - self.n_target) / s
        return (s * s) * (torch.sqrt(1.0 + z * z) - 1.0)

    def forward(self, coord: Tensor) -> Tensor:
        per_atom = torch.zeros_like(coord)
        if self.w_low != 0.0:
            per_atom = per_atom + self.w_low * self._low(coord)
        if self.w_target != 0.0:
            per_atom = per_atom + self.w_target * self._target(coord)
        if self.w_high != 0.0:
            per_atom = per_atom + self.w_high * self._high(coord)
        return per_atom.mean()


class CoordinationGuidance(nn.Module):
    """Per-atom force-like guidance from ``-grad(coord_loss) / N_atoms``.

    Mirrors the ``TersoffEnergyGuidance`` contract so the sampling loop
    can compose it with the existing schedule. The output points
    "downhill" in the coord loss and is safe to ADD to the score.

    Args:
        coord_fn: a ``DifferentiableCoordinationNumber``.
        loss_fn: a ``CoordinationLoss``.
        clamp_norm: per-atom magnitude cap (Å units). 0 / None disables.
        dtype: working dtype for the autograd evaluation.
    """

    def __init__(
        self,
        coord_fn: DifferentiableCoordinationNumber,
        loss_fn: CoordinationLoss,
        clamp_norm: float = 10.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.coord_fn = coord_fn
        self.loss_fn = loss_fn
        self.clamp_norm = clamp_norm
        self.dtype = dtype

    def _clamp_per_atom_norm(self, vec: Tensor) -> Tensor:
        if self.clamp_norm is None or self.clamp_norm <= 0:
            return vec
        norms = vec.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.clamp_norm / (norms + 1e-12), max=1.0)
        return vec * scale

    def forward(
        self, pos: Tensor, cell: Tensor, species: Tensor | None = None
    ) -> Tensor:
        device = pos.device
        leaf = pos.detach().to(self.dtype).clone().requires_grad_(True)
        cell_t = cell.detach().to(self.dtype)

        with torch.enable_grad():
            coord = self.coord_fn(leaf, cell_t, species)
            loss = self.loss_fn(coord)
            (grad,) = torch.autograd.grad(loss, leaf)

        n_atoms = pos.shape[0]
        guidance = (-grad / max(n_atoms, 1)).to(device=device, dtype=pos.dtype)
        guidance = torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)
        guidance = self._clamp_per_atom_norm(guidance)
        return guidance.detach()


class CoordinationSchedule:
    """Time-dependent weight ``lambda(t)`` for coord guidance.

    Same shapes as ``TersoffSchedule``: ``constant``, ``linear``,
    ``sigmoid``. See that class for semantics.
    """

    def __init__(
        self,
        schedule: str = "constant",
        lambda_0: float = 1.0,
        tmax: float = 1.0,
        t_gate: float = 0.3,
        k: float = 20.0,
    ) -> None:
        if schedule not in ("constant", "linear", "sigmoid"):
            raise ValueError(
                f"Unknown schedule '{schedule}'. "
                "Expected one of: constant, linear, sigmoid."
            )
        self.schedule = schedule
        self.lambda_0 = float(lambda_0)
        self.tmax = float(tmax)
        self.t_gate = float(t_gate)
        self.k = float(k)

    def __call__(self, t) -> float:
        if isinstance(t, Tensor):
            t_val = float(t.detach().flatten()[0].item())
        else:
            t_val = float(t)

        if self.schedule == "constant":
            return self.lambda_0
        if self.schedule == "linear":
            ramp = 1.0 - t_val / max(self.tmax, 1e-12)
            return self.lambda_0 * max(0.0, min(1.0, ramp))
        return self.lambda_0 * _sigmoid(-self.k * (t_val - self.t_gate))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
