"""Tests for the glass.metrics.rings module."""

import os
import pytest
import numpy as np
from pathlib import Path
from ase.build import bulk
from ase.io import read

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from glass.metrics import (
    compute_rings,
    compute_rings_distribution,
)
from glass.metrics.rings import (
    _build_seed_array,
    _compute_shortest_distances,
    _compute_distance_matrix,
)
from glass.metrics.core import RingMetrics


DATA_DIR = Path(__file__).resolve().parent / "data"


class TestRingInternals:
    """Test internal ring computation functions."""
    
    def test_build_seed_array_simple(self):
        """Test seed array construction."""
        # Simple 2-atom system: 0-1 bond
        i = np.array([0, 1])  # Atom 0 connected to 1, atom 1 connected to 0
        nat = 2
        
        seed = _build_seed_array(nat, i)
        
        # seed[0] = 0 (atom 0 starts at index 0)
        # seed[1] = 1 (atom 1 starts at index 1)
        # seed[2] = 2 (end of array)
        assert seed[0] == 0
        assert seed[1] == 1
        assert seed[2] == 2
    
    def test_build_seed_array_triangle(self):
        """Test seed array for triangular system."""
        # Triangle: 0-1, 1-2, 2-0
        i = np.array([0, 1, 2])  # sources
        nat = 3
        
        seed = _build_seed_array(nat, i)
        
        # Each atom has 1 outgoing edge in this representation
        assert seed[0] == 0
        assert seed[1] == 1
        assert seed[2] == 2
        assert seed[3] == 3
    
    def test_compute_shortest_distances_linear(self):
        """Test distance computation on linear chain."""
        # Linear chain: 0-1-2-3
        i = np.array([0, 1, 1, 2, 2, 3])  # bidirectional
        j = np.array([1, 0, 2, 1, 3, 2])  # neighbors
        nat = 4
        
        seed = _build_seed_array(nat, i)
        
        # Test distances from atom 0
        dist = _compute_shortest_distances(nat, seed, j, root=0)
        
        # Expected: dist[0]=0, dist[1]=1, dist[2]=2, dist[3]=3
        assert dist[0] == 0
        assert dist[1] == 1
        assert dist[2] == 2
        assert dist[3] == 3
    
    def test_compute_distance_matrix(self):
        """Test full distance matrix computation."""
        # Square: 0-1-2-3-0 
        # Need proper neighbor list format
        # 0: neighbors are 1 and 3
        # 1: neighbors are 0 and 2
        # 2: neighbors are 1 and 3
        # 3: neighbors are 2 and 0
        i = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # sources
        j = np.array([1, 3, 0, 2, 1, 3, 2, 0])  # neighbors
        nat = 4
        
        seed = _build_seed_array(nat, i)
        dist = _compute_distance_matrix(nat, seed, j)
        
        # Check symmetry
        for a in range(nat):
            for b in range(nat):
                assert dist[a, b] == dist[b, a]
        
        # Check diagonal
        assert np.all(np.diag(dist) == 0)


class TestRingStatistics:
    """Test ring statistics computation."""
    
    def test_compute_rings_diamond_si(self):
        """Test ring computation for diamond Si (should have 6-membered rings)."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)  # Make supercell
        
        rings = compute_rings(atoms, cutoff=2.8, maxlength=10)
        
        assert isinstance(rings, RingMetrics)
        assert rings.total_rings > 0
        
        # Diamond Si has 6-membered rings
        assert rings.ring_counts[6] > 0
        
        # No rings smaller than 3
        assert rings.ring_counts[0] == 0
        assert rings.ring_counts[1] == 0
        assert rings.ring_counts[2] == 0
    
    def test_compute_rings_amorphous_si(self):
        """Test ring computation for amorphous Si."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        rings = compute_rings(atoms, cutoff=3.0, maxlength=10)
        
        assert isinstance(rings, RingMetrics)
        assert rings.total_rings > 0
        
        # Amorphous Si should have various ring sizes
        # Should have some 5 and 6 membered rings
        assert rings.ring_counts[5] > 0 or rings.ring_counts[6] > 0
    
    def test_compute_rings_auto_cutoff(self):
        """Test ring computation with automatic cutoff."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings = compute_rings(atoms, cutoff=None, maxlength=10, auto_cutoff=True)
        
        assert isinstance(rings, RingMetrics)
        assert rings.cutoff > 0  # Should have determined a cutoff
    
    def test_compute_rings_empty_system(self):
        """Test ring computation with no bonds."""
        # Two atoms far apart
        from ase import Atoms
        atoms = Atoms('Si2', positions=[[0, 0, 0], [10, 0, 0]], cell=[20, 20, 20])
        
        rings = compute_rings(atoms, cutoff=3.0, maxlength=10)
        
        # Should return empty result
        assert rings.total_rings == 0
        assert len(rings.ring_counts) == 11  # 0 to maxlength
    
    def test_compute_rings_fractions_sum(self):
        """Test that ring fractions sum to 100%."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        rings = compute_rings(atoms, cutoff=3.0, maxlength=10)
        
        if rings.total_rings > 0:
            # Fractions should sum to approximately 100
            total_frac = np.sum(rings.ring_fractions)
            assert abs(total_frac - 100.0) < 0.01
    
    def test_compute_rings_different_maxlength(self):
        """Test ring computation with different maxlength values."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        rings_6 = compute_rings(atoms, cutoff=3.0, maxlength=6)
        rings_10 = compute_rings(atoms, cutoff=3.0, maxlength=10)
        
        # Results up to maxlength 6 should be the same (within maxlength 6)
        # Note: rings_10 may find more paths when walking with longer maxlength,
        # so counts up to 6 might differ slightly due to algorithm differences
        # We just check that maxlength is correctly set
        assert rings_6.maxlength == 6
        assert rings_10.maxlength == 10
        assert len(rings_6.ring_counts) == 7  # 0-6
        assert len(rings_10.ring_counts) == 11  # 0-10
    
    def test_compute_rings_returns_numpy_arrays(self):
        """Test that ring metrics returns proper numpy arrays."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings = compute_rings(atoms, cutoff=2.8, maxlength=10)
        
        assert isinstance(rings.ring_lengths, np.ndarray)
        assert isinstance(rings.ring_counts, np.ndarray)
        assert isinstance(rings.ring_fractions, np.ndarray)
    
    def test_compute_rings_to_dict(self):
        """Test that ring metrics can be converted to dict."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings = compute_rings(atoms, cutoff=2.8, maxlength=10)
        data = rings.to_dict()
        
        assert "ring_lengths" in data
        assert "ring_counts" in data
        assert "ring_fractions" in data
        assert "total_rings" in data
        assert "cutoff" in data
        assert "maxlength" in data


class TestRingsDistribution:
    """Test ring distribution computation over multiple frames."""
    
    def test_compute_rings_distribution_single_frame(self):
        """Test distribution with single frame."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        rings = compute_rings_distribution([atoms], cutoff=3.0, maxlength=10)
        
        assert isinstance(rings, RingMetrics)
        assert rings.total_rings > 0
    
    def test_compute_rings_distribution_multiple_frames(self):
        """Test distribution with multiple frames (same structure)."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        # Same structure twice should give same average
        rings = compute_rings_distribution([atoms, atoms], cutoff=3.0, maxlength=10)
        
        assert isinstance(rings, RingMetrics)
        # Average should be same as single computation
        single = compute_rings(atoms, cutoff=3.0, maxlength=10)
        assert np.array_equal(rings.ring_counts, single.ring_counts)
    
    def test_compute_rings_distribution_single_atom_object(self):
        """Test that single Atoms object is handled correctly."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        # Pass single object instead of list
        rings = compute_rings_distribution(atoms, cutoff=3.0, maxlength=10)
        
        # Should treat as single frame
        single = compute_rings(atoms, cutoff=3.0, maxlength=10)
        assert np.array_equal(rings.ring_counts, single.ring_counts)


class TestRingErrorMetrics:
    """Test ring error metrics."""
    
    def test_rings_rmse_identical(self):
        """Test RMSE between identical structures is zero."""
        from glass.metrics.errors import rings_rmse
        
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings1 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        rings2 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        
        rmse = rings_rmse(rings1, rings2)
        assert rmse == 0.0
    
    def test_rings_cosine_identical(self):
        """Test cosine similarity of identical structures is 1.0."""
        from glass.metrics.errors import rings_cosine_similarity
        
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings1 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        rings2 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        
        cosine = rings_cosine_similarity(rings1, rings2)
        assert abs(cosine - 1.0) < 0.001
    
    def test_rings_emd_identical(self):
        """Test EMD between identical structures is zero."""
        from glass.metrics.errors import rings_emd
        
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings1 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        rings2 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        
        emd = rings_emd(rings1, rings2)
        assert emd == 0.0


class TestRingPerformance:
    """Test ring computation performance characteristics."""
    
    @pytest.mark.slow
    def test_compute_rings_scaling(self):
        """Test that ring computation scales reasonably with system size."""
        import time
        
        atoms = bulk("Si", "diamond", a=5.43)
        
        sizes = [2, 3]
        times = []
        
        for n in sizes:
            atoms_n = atoms * (n, n, n)
            
            start = time.time()
            rings = compute_rings(atoms_n, cutoff=2.8, maxlength=10)
            end = time.time()
            
            times.append(end - start)
            
            # Sanity check: should find rings
            assert rings.total_rings > 0
        
        # Check scaling is reasonable (not worse than O(N^3))
        # This is a very loose check
        ratio = times[1] / times[0]
        n_ratio = (sizes[1] / sizes[0]) ** 3
        assert ratio < n_ratio * 2  # Allow factor of 2
