"""Advanced metrics: Voronoi analysis."""

from typing import Optional

import numpy as np
from ase import Atoms
from glass.metrics.core import VoronoiMetrics


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
        volumes = np.array(data.particles["Voronoi Volume"])

        voronoi_indices = []
        index_histogram = {}
        index_labels = []

        if compute_indices:
            indices = np.array(data.particles["Voronoi Index"])

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
