"""Utility functions for structural metrics."""

import json
from pathlib import Path
from typing import Union, Dict, Any

from glass.metrics.core import StructuralMetrics


def load_metrics_from_json(filepath: Union[str, Path]) -> StructuralMetrics:
    """Load StructuralMetrics from a JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        StructuralMetrics object
    """
    import numpy as np
    from glass.metrics.core import (
        PDFMetrics, ADFMetrics, CoordinationMetrics,
        DihedralMetrics, StructureFactorMetrics, VoronoiMetrics,
    )
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Build PDF metrics
    pdf_data = data['pdf']
    pdf = PDFMetrics(
        r=np.array(pdf_data['r']),
        g_r=np.array(pdf_data['g_r']),
        first_peak_position=pdf_data.get('first_peak_position'),
        first_peak_height=pdf_data.get('first_peak_height'),
        first_minima_position=pdf_data.get('first_minima_position'),
        first_minima_height=pdf_data.get('first_minima_height'),
        coord_cutoff=pdf_data.get('coord_cutoff'),
    )
    
    # Build ADF metrics
    adf_data = data['adf']
    adf = ADFMetrics(
        angles=np.array(adf_data['angles']),
        adf=np.array(adf_data['adf']),
        dominant_angle=adf_data.get('dominant_angle'),
        dominant_angle_degree=adf_data.get('dominant_angle_degree'),
    )
    
    # Build Coordination metrics
    coord_data = data['coordination']
    coordination = CoordinationMetrics(
        coordination_numbers=np.array(coord_data['coordination_numbers']),
        mean_coordination=coord_data['mean_coordination'],
        std_coordination=coord_data['std_coordination'],
        coordination_histogram=np.array(coord_data['coordination_histogram']),
        histogram_bins=np.array(coord_data['histogram_bins']),
    )
    
    # Build optional metrics
    dihedrals = None
    if 'dihedrals' in data and data['dihedrals']:
        dih_data = data['dihedrals']
        dihedrals = DihedralMetrics(
            dihedral_angles=np.array(dih_data['dihedral_angles']),
            dihedral_histogram=np.array(dih_data['dihedral_histogram']),
            histogram_bins=np.array(dih_data['histogram_bins']),
            mean_dihedral=dih_data['mean_dihedral'],
            std_dihedral=dih_data['std_dihedral'],
        )
    
    structure_factor = None
    if 'structure_factor' in data and data['structure_factor']:
        sq_data = data['structure_factor']
        structure_factor = StructureFactorMetrics(
            q=np.array(sq_data['q']),
            s_q=np.array(sq_data['s_q']),
            s_q_total=np.array(sq_data['s_q_total']) if 's_q_total' in sq_data else None,
        )
    
    voronoi = None
    if 'voronoi' in data and data['voronoi']:
        vor_data = data['voronoi']
        voronoi = VoronoiMetrics(
            voronoi_indices=[tuple(idx) for idx in vor_data['voronoi_indices']],
            index_histogram=vor_data['index_histogram'],
            index_labels=vor_data['index_labels'],
            mean_volume=vor_data['mean_volume'],
            volume_std=vor_data['volume_std'],
        )
    
    return StructuralMetrics(
        n_atoms=data['n_atoms'],
        composition=data['composition'],
        cell=data['cell'],
        density=data['density'],
        pdf=pdf,
        adf=adf,
        coordination=coordination,
        dihedrals=dihedrals,
        structure_factor=structure_factor,
        voronoi=voronoi,
    )
