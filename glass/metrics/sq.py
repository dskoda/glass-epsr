import numpy as np
from typing import Optional
from ase import Atoms

from glass.metrics.core import StructureFactorMetrics


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
