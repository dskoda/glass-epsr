"""Reward functions for RLPF fine-tuning of the score-based diffusion model.

Rewards combine Tersoff energy and PDF similarity to guide the model toward
physically meaningful amorphous Si structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class RewardConfig:
    """Configuration for reward function weighting."""

    w_energy: float = 1.0
    w_pdf: float = 1.0


class TersoffPDFReward:
    """Reward function combining Tersoff energy and PDF similarity.

    reward = -(w_energy * E/N + w_pdf * pdf_rmse)

    A higher (less negative) reward indicates a better structure.

    Args:
        tersoff_calc: TorchTersoff instance for energy evaluation.
        target_g_r: Target PDF values (1-D numpy array).
        target_r: r-grid for target PDF (1-D numpy array).
        w_energy: Weight for the energy term.
        w_pdf: Weight for the PDF RMSE term.
        device: Device string (e.g. "cpu" or "cuda:0").
    """

    def __init__(
        self,
        tersoff_calc,
        target_g_r: np.ndarray,
        target_r: np.ndarray,
        w_energy: float = 1.0,
        w_pdf: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.tersoff_calc = tersoff_calc
        self.target_g_r = np.asarray(target_g_r, dtype=np.float64)
        self.target_r = np.asarray(target_r, dtype=np.float64)
        self.w_energy = w_energy
        self.w_pdf = w_pdf
        self.device = device

    def __call__(
        self,
        pos: Tensor,
        cell: Tensor,
        species: Tensor,
    ) -> Tuple[float, dict]:
        """Compute reward for a structure.

        Args:
            pos: Atomic positions tensor [N, 3].
            cell: Unit cell tensor [3, 3].
            species: Atomic species tensor [N, num_species] (one-hot) or [N].

        Returns:
            (scalar_reward, {"energy": float, "pdf": float})
        """
        from glass.metrics.structural import compute_pdf

        n_atoms = pos.shape[0]

        # --- Tersoff energy ---
        pos_f64 = pos.detach().double().cpu()
        cell_f64 = cell.detach().double().cpu()
        with torch.no_grad():
            energy = float(self.tersoff_calc.energy(pos_f64, cell_f64).item())
        energy_per_atom = energy / n_atoms

        # --- PDF RMSE ---
        # Build ASE Atoms from tensors
        import ase
        import numpy as np

        pos_np = pos.detach().cpu().numpy().astype(np.float64)
        cell_np = cell.detach().cpu().numpy().astype(np.float64)
        # species is one-hot [N, num_sp] or integer [N]
        symbols = ["Si"] * n_atoms
        atoms = ase.Atoms(symbols=symbols, positions=pos_np, cell=cell_np, pbc=True)

        pdf_metrics = compute_pdf(atoms)
        pred_g_r = pdf_metrics.g_r
        pred_r = pdf_metrics.r

        # Align to target r-grid via interpolation
        aligned_pred = np.interp(self.target_r, pred_r, pred_g_r)
        pdf_rmse = float(np.sqrt(np.mean((aligned_pred - self.target_g_r) ** 2)))

        reward = -(self.w_energy * energy_per_atom + self.w_pdf * pdf_rmse)

        return reward, {"energy": energy_per_atom, "pdf": pdf_rmse}
