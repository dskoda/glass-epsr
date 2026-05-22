"""Structural-entropy guidance term for the reverse SDE.

Computes the gradient of the cross-atom variance of ACSF descriptors and
returns it as a per-atom "force-like" score vector. Minimising this
variance pushes every atom toward the same local environment — the
defining property of a continuous random network (Cliffe et al., PRB 95,
224108, 2017).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from glass.descriptors.acsf import TorchACSF


class EntropyGuidance(nn.Module):
    """Gradient of ACSF descriptor variance, returned per atom.

    Args:
        acsf: A configured :class:`TorchACSF` instance.
        clamp_norm: Optional per-atom guidance-norm clamp (Å-units / N). Set
            to None or non-positive to disable.
    """

    def __init__(
        self,
        acsf: TorchACSF,
        clamp_norm: Optional[float] = 10.0,
    ) -> None:
        super().__init__()
        self.acsf = acsf
        self.clamp_norm = clamp_norm

    def _clamp(self, vec: Tensor) -> Tensor:
        if self.clamp_norm is None or self.clamp_norm <= 0:
            return vec
        norms = vec.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.clamp_norm / (norms + 1e-12), max=1.0)
        return vec * scale

    def forward(
        self,
        pos: Tensor,
        cell: Tensor,
        species: Optional[Tensor] = None,
    ) -> Tensor:
        """Return ``-grad_pos L(pos)``, where ``L = mean(var(D, dim=0))``.

        Args:
            pos: ``(N, 3)`` Tweedie-denoised positions (Å).
            cell: ``(3, 3)`` periodic cell (Å).
            species: Unused (single-species).

        Returns:
            ``(N, 3)`` guidance vector pointing toward lower descriptor
            variance.
        """
        leaf = pos.detach().clone().requires_grad_(True)
        cell_d = cell.detach()
        with torch.enable_grad():
            descriptors = self.acsf(leaf, cell_d, species)
            if descriptors.shape[0] < 2 or descriptors.shape[1] == 0:
                return torch.zeros_like(pos)
            loss = descriptors.var(dim=0).mean()
            (grad,) = torch.autograd.grad(loss, leaf)
        guidance = (-grad).to(dtype=pos.dtype, device=pos.device)
        guidance = torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)
        guidance = self._clamp(guidance)
        return guidance.detach()


class EntropySchedule:
    """Time-dependent weight ``lambda(t)`` for the entropy guidance term.

    Mirrors :class:`glass.lit.modules.tersoff_guidance.TersoffSchedule`.
    """

    def __init__(
        self,
        schedule: str = "constant",
        lambda_0: float = 1.0,
        tmax: float = 1.0,
        t_gate: float = 1.0,
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
            # Hard t_gate cutoff for the constant schedule.
            if t_val > self.t_gate * self.tmax:
                return 0.0
            return self.lambda_0
        if self.schedule == "linear":
            ramp = 1.0 - t_val / max(self.tmax, 1e-12)
            return self.lambda_0 * max(0.0, min(1.0, ramp))
        # sigmoid
        x = -self.k * (t_val - self.t_gate)
        if x >= 0:
            z = math.exp(-x)
            sig = 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            sig = z / (1.0 + z)
        return self.lambda_0 * sig
