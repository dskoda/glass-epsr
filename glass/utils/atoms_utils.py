"""Utilities for ASE atoms manipulation and tensor conversion.

This module provides helper functions for converting between ASE Atoms objects
and PyTorch tensors used by the models.
"""

import torch
from torch import Tensor
from typing import Tuple, Union
import ase

from glass.lit.functions.get_atoms import initialize_atoms


def atoms_to_device(
    atoms: ase.Atoms,
    device: Union[str, torch.device],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert ASE Atoms to tensors and move to device.
    
    Args:
        atoms: ASE Atoms object
        device: Target device for tensors
    
    Returns:
        (z_list, species, pos, cell) where:
        - z_list: Atomic numbers list
        - species: One-hot encoded species tensor
        - pos: Positions tensor [N, 3]
        - cell: Cell tensor [3, 3]
    """
    device = torch.device(device)
    _, species, pos, cell = initialize_atoms(atoms)
    return (
        species.to(device),
        pos.to(device=device, dtype=torch.float32),
        cell.to(device=device, dtype=torch.float32),
    )


def compute_prior_score(
    species: Tensor,
    pos: Tensor,
    cell: Tensor,
    t: Tensor,
    cutoff: float,
    score_net,
    diffuser,
) -> Tensor:
    """Compute prior score using the score network.
    
    Args:
        species: Atomic species tensor
        pos: Atomic positions tensor
        cell: Unit cell tensor
        t: Time step tensor
        cutoff: Graph cutoff radius
        score_net: Score network (EMA model)
        diffuser: Diffuser for sigma computation
    
    Returns:
        Prior score tensor
    """
    from glass.nn import periodic_radius_graph
    
    edge_index, edge_vec = periodic_radius_graph(pos, cutoff, cell)
    edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
    return score_net.ema_model(species, edge_index, edge_attr, t, diffuser.sigma(t))


def compute_target_from_reference(
    ref_atoms: ase.Atoms,
    guidance_model,
    guidance_type: str,
    cutoff: float,
    device: Union[str, torch.device],
) -> Tensor:
    """Compute target feature from reference atoms.
    
    Args:
        ref_atoms: Reference ASE Atoms object
        guidance_model: Guidance model for computing features
        guidance_type: Type of guidance
        cutoff: Graph cutoff radius
        device: Target device for output tensor
    
    Returns:
        Target feature tensor on specified device
    """
    from glass.nn import periodic_radius_graph
    
    device = torch.device(device)
    _, ref_species, ref_pos, ref_cell = initialize_atoms(ref_atoms)
    
    if guidance_type in ("pdf", "adf"):
        return guidance_model(
            ref_pos.cpu(), ref_species.cpu(), ref_cell.cpu()
        )[1].to(device)
    
    elif guidance_type in ("xrd", "nd"):
        return guidance_model(
            ref_pos.to(device), ref_species.to(device)
        )
    
    elif guidance_type in ("exafs", "xanes"):
        ei_r, ev_r = periodic_radius_graph(
            ref_pos.to(device), cutoff, ref_cell.to(device)
        )
        ea_r = torch.hstack([ev_r, ev_r.norm(dim=-1, keepdim=True)])
        with torch.no_grad():
            return guidance_model(ref_species.to(device), ei_r, ea_r)
    
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")
