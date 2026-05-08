"""Advanced metrics: structure factor and Voronoi analysis."""

import numpy as np
from typing import Optional
from ase import Atoms

from glass.metrics.core import StructureFactorMetrics, VoronoiMetrics


def compute_structure_factor(
    atoms: Atoms,
    q_min: float = 0.5,
    q_max: float = 20.0,
    q_step: float = 0.1,
    method: str = "debye",
) -> Optional[StructureFactorMetrics]:
    """Compute structure factor S(q) for ASE Atoms.
    
    Uses the Debye formula for X-ray scattering.
    
    Args:
        atoms: ASE Atoms object
        q_min: Minimum q value
        q_max: Maximum q value
        q_step: Q step size
        method: Computation method ("debye" or "from_pdf")
    
    Returns:
        StructureFactorMetrics object, or None if computation fails
    """
    try:
        from DebyeCalculator import DebyeCalculator
    except ImportError:
        return None
    
    try:
        # Initialize Debye calculator
        calculator = DebyeCalculator(
            qmin=q_min,
            qmax=q_max,
            qstep=q_step,
        )
        
        # Compute structure factor
        q, s_q = calculator.compute(atoms)
        
        return StructureFactorMetrics(
            q=q,
            s_q=s_q,
            s_q_total=s_q,  # For single species, total is the same
        )
    except Exception:
        return None


def compute_voronoi(
    atoms: Atoms,
    compute_indices: bool = True,
) -> Optional[VoronoiMetrics]:
    """Compute Voronoi analysis for ASE Atoms using ovito.
    
    Args:
        atoms: ASE Atoms object
        compute_indices: Whether to compute Voronoi indices
    
    Returns:
        VoronoiMetrics object, or None if ovito not available
    """
    try:
        from ovito.io import ase_to_ovito
        from ovito.modifiers import VoronoiAnalysisModifier
    except ImportError:
        return None
    
    try:
        # Convert to ovito
        pipeline = ase_to_ovito(atoms)
        
        # Apply Voronoi analysis
        modifier = VoronoiAnalysisModifier(
            compute_indices=compute_indices,
            edge_threshold=0.1,
        )
        pipeline.modifiers.append(modifier)
        
        # Evaluate
        data = pipeline.compute()
        
        # Extract results
        volumes = np.array(data.particles['Voronoi Volume'])
        
        voronoi_indices = []
        index_histogram = {}
        index_labels = []
        
        if compute_indices:
            indices = np.array(data.particles['Voronoi Index'])
            
            for idx in indices:
                idx_tuple = tuple(idx)
                voronoi_indices.append(idx_tuple)
                
                # Create label like <0,3,0,0>
                label = f"<{','.join(map(str, idx))}>"
                index_histogram[label] = index_histogram.get(label, 0) + 1
                
                if label not in index_labels:
                    index_labels.append(label)
        
        return VoronoiMetrics(
            voronoi_indices=voronoi_indices,
            index_histogram=index_histogram,
            index_labels=index_labels,
            mean_volume=float(np.mean(volumes)),
            volume_std=float(np.std(volumes)),
        )
    except Exception:
        return None
