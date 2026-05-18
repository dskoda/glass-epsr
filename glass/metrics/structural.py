"""Structural metrics: PDF and ADF computations."""

import numpy as np
from typing import Optional
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.signal import find_peaks

from glass.metrics.core import PDFMetrics, ADFMetrics


def compute_pdf(
    atoms: Atoms,
    cutoff: float = 8.0,
    bin_size: int = 200,
    sigma: Optional[float] = 0.15,
    find_peaks_prominence: float = 0.1,
) -> PDFMetrics:
    """Compute Pair Distribution Function (PDF/RDF) for ASE Atoms.
    
    Memory-efficient implementation using ASE neighbor_list.
    
    Args:
        atoms: ASE Atoms object
        cutoff: Maximum distance for PDF computation
        bin_size: Number of bins
        sigma: Gaussian smoothing sigma. Default 0.15 Å.
        find_peaks_prominence: Minimum prominence for peak detection
    
    Returns:
        PDFMetrics object with PDF and extracted features
    """
    # Get neighbor pairs
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    
    # Remove self-interactions
    mask = i != j
    d = d[mask]
    
    if len(d) == 0:
        r = np.linspace(0, cutoff, bin_size)
        g_r = np.zeros(bin_size)
        return PDFMetrics(
            r=r, g_r=g_r,
            first_peak_position=None, first_peak_height=None,
            first_minima_position=None, first_minima_height=None,
            coord_cutoff=None,
        )
    
    # Create histogram
    bin_edges = np.linspace(0, cutoff, bin_size + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    delta_r = bin_edges[1] - bin_edges[0]
    
    hist, _ = np.histogram(d, bins=bin_edges)
    
    # Normalization: g(r) = (V / N^2) * hist / (4πr²Δr)
    n_atoms = len(atoms)
    volume = atoms.get_volume()
    prefactor = volume / (n_atoms ** 2)
    
    shell_volumes = np.zeros_like(bin_centers)
    mask = bin_centers > 0
    shell_volumes[mask] = 4 * np.pi * bin_centers[mask]**2 * delta_r
    
    g_r = prefactor * hist / (shell_volumes + 1e-10)
    
    # Apply smoothing
    if sigma is not None and sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        g_r = gaussian_filter1d(g_r, sigma=sigma/delta_r, mode='nearest')
    
    # Find peaks and minimum
    first_peak_pos = None
    first_peak_height = None
    first_min_pos = None
    first_min_height = None
    coord_cutoff = None
    
    peaks, _ = find_peaks(g_r, prominence=find_peaks_prominence, distance=5)
    
    if len(peaks) > 0:
        first_peak_idx = peaks[0]
        first_peak_pos = float(bin_centers[first_peak_idx])
        first_peak_height = float(g_r[first_peak_idx])
        
        # Find minimum after first peak
        search_start = first_peak_idx + 3
        search_end = len(g_r)
        if len(peaks) > 1:
            search_end = min(search_end, peaks[1] - 3)
        
        if search_start < search_end:
            min_idx = search_start + np.argmin(g_r[search_start:search_end])
            first_min_pos = float(bin_centers[min_idx])
            first_min_height = float(g_r[min_idx])
            
            if first_min_pos < 5.0 and first_min_pos > first_peak_pos + 0.1:
                coord_cutoff = first_min_pos
            else:
                coord_cutoff = min(first_peak_pos * 1.3, 3.2)
        
        if coord_cutoff is None:
            coord_cutoff = min(first_peak_pos * 1.35, 3.2)
    
    return PDFMetrics(
        r=bin_centers,
        g_r=g_r,
        first_peak_position=first_peak_pos,
        first_peak_height=first_peak_height,
        first_minima_position=first_min_pos,
        first_minima_height=first_min_height,
        coord_cutoff=coord_cutoff,
    )


def compute_adf(
    atoms: Atoms,
    cutoff: Optional[float] = None,
    bin_size: int = 100,
    sigma: float = 0.05,
    auto_cutoff: bool = True,
) -> ADFMetrics:
    """Compute Angular Distribution Function (ADF) for ASE Atoms.
    
    Args:
        atoms: ASE Atoms object
        cutoff: Cutoff for triplet search
        bin_size: Number of angle bins
        sigma: Gaussian smoothing sigma
        auto_cutoff: If True and cutoff is None, use PDF first minimum
    
    Returns:
        ADFMetrics object
    """
    from ase.neighborlist import NeighborList
    
    if cutoff is None and auto_cutoff:
        pdf_metrics = compute_pdf(atoms, cutoff=8.0, bin_size=100)
        cutoff = pdf_metrics.coord_cutoff or 3.5
    elif cutoff is None:
        cutoff = 3.5
    
    nl = NeighborList([cutoff / 2.0] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    angles = []
    
    for j in range(len(atoms)):
        neighbors_j, offsets_j = nl.get_neighbors(j)
        
        for idx_a, i in enumerate(neighbors_j):
            if i >= j:
                continue
            for idx_b, k in enumerate(neighbors_j):
                if k <= j or k == i:
                    continue
                
                p_i = positions[i] + offsets_j[idx_a] @ cell
                p_j = positions[j]
                p_k = positions[k]
                
                v1 = p_i - p_j
                v2 = p_k - p_j
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 > 1e-10 and norm2 > 1e-10:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angles.append(np.arccos(cos_angle))
    
    angles = np.array(angles)
    
    angle_bins = np.linspace(0, np.pi, bin_size + 1)
    bin_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    
    if len(angles) == 0:
        adf = np.zeros(bin_size)
    else:
        adf = np.zeros(bin_size)
        for angle in angles:
            adf += np.exp(-0.5 * ((bin_centers - angle) / sigma) ** 2)
        adf = adf / (len(angles) * sigma * np.sqrt(2 * np.pi))
    
    dominant_angle = None
    dominant_angle_deg = None
    
    peaks, _ = find_peaks(adf, prominence=0.01)
    if len(peaks) > 0:
        dominant_idx = peaks[0]
        dominant_angle = float(bin_centers[dominant_idx])
        dominant_angle_deg = float(np.degrees(dominant_angle))
    
    return ADFMetrics(
        angles=bin_centers,
        adf=adf,
        dominant_angle=dominant_angle,
        dominant_angle_degree=dominant_angle_deg,
    )
