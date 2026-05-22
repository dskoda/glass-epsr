"""Differentiable Atom-Centred Symmetry Functions (Behler 2011).

Implements G1 (cosine cutoff sum), G2 (radial Gaussians) and G4 (angular)
descriptors using only :mod:`torch` primitives, so the resulting per-atom
descriptor matrix is autograd-differentiable with respect to atomic
positions. Used by :mod:`glass.descriptors.entropy` to compute a
"structural entropy" guidance term during reverse-SDE sampling.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from glass.nn.cluster import periodic_radius_graph


def _scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = torch.zeros(
        (dim_size,) + src.shape[1:], dtype=src.dtype, device=src.device
    )
    out.index_add_(0, index, src)
    return out


class TorchACSF(nn.Module):
    """Per-atom ACSF descriptor matrix (autograd-safe).

    Args:
        r_cut: Cutoff radius (Å) shared by G1, G2, G4.
        g2_params: Sequence of ``(eta, Rs)`` pairs for G2 radial channels.
        g4_params: Sequence of ``(eta, zeta, lambda)`` triples for G4 angular
            channels. May be empty.
        include_g1: If True, prepend a G1 (cosine cutoff sum) channel.
        max_triples: Soft ceiling on the number of (i,j,k) triples enumerated
            for G4. If exceeded, G4 is skipped (zero columns) and a warning
            is emitted, to avoid OOM on very dense cells.
    """

    def __init__(
        self,
        r_cut: float = 4.0,
        g2_params: Sequence[Tuple[float, float]] = (),
        g4_params: Sequence[Tuple[float, float, float]] = (),
        include_g1: bool = True,
        max_triples: int = 500_000,
    ) -> None:
        super().__init__()
        self.r_cut = float(r_cut)
        self.include_g1 = bool(include_g1)
        self.max_triples = int(max_triples)

        if len(g2_params) == 0:
            g2_t = torch.zeros(0, 2)
        else:
            g2_t = torch.as_tensor(g2_params, dtype=torch.get_default_dtype())
        self.register_buffer("g2_params", g2_t, persistent=False)

        if len(g4_params) == 0:
            g4_t = torch.zeros(0, 3)
        else:
            g4_t = torch.as_tensor(g4_params, dtype=torch.get_default_dtype())
        self.register_buffer("g4_params", g4_t, persistent=False)

    @classmethod
    def for_silicon(cls, r_cut: float = 4.0, **kwargs) -> "TorchACSF":
        """Default Si-CRN parameter set.

        G2 follows the JAX reference (``research/acsf/simple-acsf.py``):
        ``(eta, Rs) ∈ {(1,1),(1,2),(1,3),(1,4)}``. G4 is a minimal
        Behler-style angular set with ``eta=0.005, zeta=1, lambda ∈ {-1,+1}``.
        """
        g2 = ((1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (1.0, 4.0))
        g4 = ((0.005, 1.0, -1.0), (0.005, 1.0, 1.0))
        return cls(r_cut=r_cut, g2_params=g2, g4_params=g4, **kwargs)

    @property
    def n_channels(self) -> int:
        return (
            (1 if self.include_g1 else 0)
            + self.g2_params.shape[0]
            + self.g4_params.shape[0]
        )

    def _cutoff(self, r: Tensor) -> Tensor:
        fc = 0.5 * (torch.cos(math.pi * r / self.r_cut) + 1.0)
        return fc * (r < self.r_cut).to(fc.dtype)

    def forward(
        self, pos: Tensor, cell: Tensor, species: Optional[Tensor] = None
    ) -> Tensor:
        """Compute ``[N, M]`` descriptor matrix.

        ``species`` is accepted for API parity but unused (single-species).
        """
        del species
        n = pos.shape[0]
        device = pos.device
        dtype = pos.dtype

        edge_index, edge_vec = periodic_radius_graph(pos, self.r_cut, cell)
        src = edge_index[0]
        r = edge_vec.norm(dim=-1)
        fc = self._cutoff(r)

        cols: List[Tensor] = []

        if self.include_g1:
            cols.append(_scatter_add(fc, src, n).unsqueeze(-1))

        if self.g2_params.shape[0] > 0:
            etas = self.g2_params[:, 0].to(dtype=dtype, device=device)  # (Pg2,)
            rss = self.g2_params[:, 1].to(dtype=dtype, device=device)
            r_e = r.unsqueeze(-1)  # (E, 1)
            fc_e = fc.unsqueeze(-1)  # (E, 1)
            g2_pairs = torch.exp(-etas * (r_e - rss) ** 2) * fc_e  # (E, Pg2)
            g2_atoms = _scatter_add(g2_pairs, src, n)  # (N, Pg2)
            cols.append(g2_atoms)

        if self.g4_params.shape[0] > 0:
            g4 = self._g4(edge_index, edge_vec, r, fc, n, device, dtype)
            cols.append(g4)

        if not cols:
            return torch.zeros((n, 0), dtype=dtype, device=device)
        return torch.cat(cols, dim=-1)

    def _g4(
        self,
        edge_index: Tensor,
        edge_vec: Tensor,
        r: Tensor,
        fc: Tensor,
        n: int,
        device,
        dtype,
    ) -> Tensor:
        n_g4 = self.g4_params.shape[0]
        zeros = torch.zeros((n, n_g4), dtype=dtype, device=device)

        src = edge_index[0]
        sort_idx = torch.argsort(src, stable=True)
        src_s = src[sort_idx]
        vec_s = edge_vec[sort_idx]
        r_s = r[sort_idx]
        fc_s = fc[sort_idx]

        if src_s.numel() == 0:
            return zeros

        unique_i, counts = torch.unique_consecutive(src_s, return_counts=True)
        # Number of unordered pairs per group
        n_triples = int((counts * (counts - 1) // 2).sum().item())
        if n_triples == 0:
            return zeros
        if n_triples > self.max_triples:
            warnings.warn(
                f"TorchACSF: skipping G4 (n_triples={n_triples} > "
                f"max_triples={self.max_triples}).",
                stacklevel=2,
            )
            return zeros

        # Build pair indices group-by-group. Loop is over unique source atoms
        # (≤ N), inner work is vectorised.
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=counts.dtype, device=device),
                counts.cumsum(0),
            ]
        )
        ea_list: List[Tensor] = []
        eb_list: List[Tensor] = []
        ti_list: List[Tensor] = []
        for g in range(unique_i.shape[0]):
            c = int(counts[g].item())
            if c < 2:
                continue
            start = int(offsets[g].item())
            end = int(offsets[g + 1].item())
            idx = torch.arange(start, end, device=device)
            pairs = torch.combinations(idx, 2)
            ea_list.append(pairs[:, 0])
            eb_list.append(pairs[:, 1])
            ti_list.append(unique_i[g].expand(pairs.shape[0]))

        if not ea_list:
            return zeros

        ea = torch.cat(ea_list)
        eb = torch.cat(eb_list)
        ti = torch.cat(ti_list)

        v_ij = vec_s[ea]
        v_ik = vec_s[eb]
        r_ij = r_s[ea]
        r_ik = r_s[eb]
        fc_ij = fc_s[ea]
        fc_ik = fc_s[eb]
        v_jk = v_ik - v_ij
        r_jk = v_jk.norm(dim=-1)
        fc_jk = self._cutoff(r_jk)

        cos_theta = (v_ij * v_ik).sum(dim=-1) / (r_ij * r_ik + 1e-12)

        etas = self.g4_params[:, 0].to(dtype=dtype, device=device)  # (Pg4,)
        zetas = self.g4_params[:, 1].to(dtype=dtype, device=device)
        lams = self.g4_params[:, 2].to(dtype=dtype, device=device)

        ct_e = cos_theta.unsqueeze(-1)  # (T, 1)
        rij2 = (r_ij ** 2).unsqueeze(-1)
        rik2 = (r_ik ** 2).unsqueeze(-1)
        rjk2 = (r_jk ** 2).unsqueeze(-1)
        fc_prod = (fc_ij * fc_ik * fc_jk).unsqueeze(-1)

        ang = (1.0 + lams * ct_e).clamp(min=0.0) ** zetas
        rad = torch.exp(-etas * (rij2 + rik2 + rjk2))
        prefactor = 2.0 ** (1.0 - zetas)
        per_triple = prefactor * ang * rad * fc_prod  # (T, Pg4)

        g4_atoms = _scatter_add(per_triple, ti, n)
        return g4_atoms
