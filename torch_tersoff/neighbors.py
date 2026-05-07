import itertools

import torch


def build_neighbors(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc,
    cutoff: float,
):
    """Enumerate directed neighbor pairs (i, j, shift) with |r_ij| < cutoff.

    Parameters
    ----------
    positions : (N, 3) float tensor
    cell : (3, 3) float tensor (assumed orthorhombic / diagonal)
    pbc : iterable of 3 bools
    cutoff : float

    Returns
    -------
    i_idx, j_idx : (M,) long tensors
    shift_vec : (M, 3) float tensor of cartesian shifts (= S @ cell)
        so that r_j_image = positions[j] + shift_vec, and
        r_ij_vec = positions[j] + shift_vec - positions[i].
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]

    pbc = tuple(bool(p) for p in pbc)

    # Only enumerate image shifts along PBC directions.
    ranges = [(-1, 0, 1) if p else (0,) for p in pbc]
    shift_ints = torch.tensor(
        list(itertools.product(*ranges)), dtype=torch.long, device=device
    )  # (S, 3)

    cell_t = cell.to(dtype=dtype, device=device)
    shift_vecs = shift_ints.to(dtype=dtype) @ cell_t  # (S, 3)

    # Pairwise displacements: r_j + shift - r_i
    # Build (S, N, N, 3)
    ri = positions.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 3)
    rj = positions.unsqueeze(0).unsqueeze(1)  # (1, 1, N, 3)
    s = shift_vecs.unsqueeze(1).unsqueeze(1)  # (S, 1, 1, 3)
    disp = rj + s - ri  # (S, N, N, 3)
    r2 = (disp * disp).sum(dim=-1)  # (S, N, N)

    # Mask: within cutoff, exclude i==j at shift==0 (self).
    in_cut = r2 < cutoff * cutoff
    S = shift_ints.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # (1, N, N)
    zero_shift_mask = (shift_ints == 0).all(dim=1)  # (S,)
    self_mask = eye & zero_shift_mask.view(S, 1, 1)
    in_cut = in_cut & (~self_mask)
    # Also exclude zero-distance (shouldn't happen otherwise)
    in_cut = in_cut & (r2 > 1e-24)

    s_idx, i_idx, j_idx = torch.nonzero(in_cut, as_tuple=True)
    shift_out = shift_vecs[s_idx]
    return i_idx, j_idx, shift_out
