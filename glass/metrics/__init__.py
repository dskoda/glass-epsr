"""Structural metrics computation for atomic structures.

This module provides non-differentiable metrics for analyzing atomic structures,
including PDF, ADF, coordination numbers, dihedral angles, and structure factors.
"""

# Core dataclasses
from glass.metrics.core import (
    PDFMetrics,
    ADFMetrics,
    CoordinationMetrics,
    DihedralMetrics,
    StructureFactorMetrics,
    VoronoiMetrics,
    StructuralMetrics,
)

# Structural metrics (PDF, ADF)
from glass.metrics.structural import (
    compute_pdf,
    compute_adf,
)

# Geometric metrics (coordination, dihedrals)
from glass.metrics.geometric import (
    compute_coordination,
    compute_dihedrals,
)

# Advanced metrics (structure factor, Voronoi)
from glass.metrics.advanced import (
    compute_structure_factor,
    compute_voronoi,
)

# Error metrics for comparison
from glass.metrics.errors import (
    # PDF error metrics
    pdf_rmse,
    pdf_mae,
    pdf_area_between,
    pdf_cosine_similarity,
    pdf_r_chi2,
    pdf_peak_position_error,
    pdf_peak_height_error,
    # ADF error metrics
    adf_rmse,
    adf_cosine_similarity,
    # Coordination error metrics
    coordination_emd,
    coordination_histogram_rmse,
    coordination_mean_error,
    coordination_std_error,
    # Combined metrics
    compute_all_errors,
    compute_weighted_error,
)

# Utility functions
from glass.metrics.utils import load_metrics_from_json


# Main entry point
from ase import Atoms
from typing import Optional


def compute_all_metrics(
    atoms: Atoms,
    pdf_cutoff: float = 8.0,
    adf_cutoff: Optional[float] = None,
    coord_cutoff: Optional[float] = None,
    auto_cutoff: bool = True,
    include_dihedrals: bool = True,
    include_sq: bool = True,
    include_voronoi: bool = True,
) -> StructuralMetrics:
    """Compute all structural metrics for ASE Atoms.
    
    This is the main entry point for computing comprehensive structural metrics.
    
    Args:
        atoms: ASE Atoms object
        pdf_cutoff: Maximum r for PDF computation
        adf_cutoff: Cutoff for ADF. If None and auto_cutoff=True, uses PDF minimum
        coord_cutoff: Cutoff for coordination. If None and auto_cutoff=True, uses PDF minimum
        auto_cutoff: If True, automatically determine ADF and coordination cutoffs from PDF
        include_dihedrals: Whether to compute dihedral angles
        include_sq: Whether to compute structure factor S(q)
        include_voronoi: Whether to compute Voronoi analysis
    
    Returns:
        StructuralMetrics object with all computed metrics
    """
    # Basic structure info
    n_atoms = len(atoms)
    composition = atoms.get_chemical_formula()
    cell = atoms.cell.cellpar().tolist()  # [a, b, c, alpha, beta, gamma]
    volume = atoms.get_volume()
    density = n_atoms / volume if volume > 0 else 0.0
    
    # Compute PDF (always needed, may be used for auto-cutoff)
    pdf_metrics = compute_pdf(atoms, cutoff=pdf_cutoff)
    
    # Determine cutoffs
    if auto_cutoff and pdf_metrics.coord_cutoff is not None:
        if adf_cutoff is None:
            adf_cutoff = pdf_metrics.coord_cutoff
        if coord_cutoff is None:
            coord_cutoff = pdf_metrics.coord_cutoff
    
    # Use defaults if still None
    if adf_cutoff is None:
        adf_cutoff = 3.5
    if coord_cutoff is None:
        coord_cutoff = 3.0
    
    # Compute other metrics
    adf_metrics = compute_adf(atoms, cutoff=adf_cutoff, auto_cutoff=False)
    coord_metrics = compute_coordination(atoms, cutoff=coord_cutoff, auto_cutoff=False)
    
    # Optional metrics
    dihedral_metrics = None
    sq_metrics = None
    voronoi_metrics = None
    
    if include_dihedrals:
        dihedral_metrics = compute_dihedrals(atoms)
    
    if include_sq:
        sq_metrics = compute_structure_factor(atoms)
    
    if include_voronoi:
        voronoi_metrics = compute_voronoi(atoms)
    
    return StructuralMetrics(
        n_atoms=n_atoms,
        composition=composition,
        cell=cell,
        density=density,
        pdf=pdf_metrics,
        adf=adf_metrics,
        coordination=coord_metrics,
        dihedrals=dihedral_metrics,
        structure_factor=sq_metrics,
        voronoi=voronoi_metrics,
    )


__all__ = [
    # Dataclasses
    'PDFMetrics',
    'ADFMetrics',
    'CoordinationMetrics',
    'DihedralMetrics',
    'StructureFactorMetrics',
    'VoronoiMetrics',
    'StructuralMetrics',
    # Computation functions
    'compute_pdf',
    'compute_adf',
    'compute_coordination',
    'compute_dihedrals',
    'compute_structure_factor',
    'compute_voronoi',
    'compute_all_metrics',
    # Error metrics
    'pdf_rmse',
    'pdf_mae',
    'pdf_area_between',
    'pdf_cosine_similarity',
    'pdf_r_chi2',
    'pdf_peak_position_error',
    'pdf_peak_height_error',
    'adf_rmse',
    'adf_cosine_similarity',
    'coordination_emd',
    'coordination_histogram_rmse',
    'coordination_mean_error',
    'coordination_std_error',
    'compute_all_errors',
    'compute_weighted_error',
    # Utilities
    'load_metrics_from_json',
]
