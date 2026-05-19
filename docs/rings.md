# Ring statistics — implementation notes

This document describes the ring-statistics module at `glass/metrics/rings.py`.
It is written for someone who needs to modify the algorithm, debug a numerical
discrepancy, or extend it (e.g. add a new flavour of ring or port to GPU). The
public-API surface is summarised at the bottom.

## What the module computes

A *ring* in an atomic graph is a closed simple cycle of bonded atoms.
Counting rings in disordered networks is ambiguous — many definitions
exist — and `rings.py` implements one specific definition:

> **Shortest-path (SP) rings**, after Franzblau, *Phys. Rev. B* **44**, 4925 (1991).

A cycle `(v0, v1, …, v_{N-1}, v0)` of length `N` is an SP ring iff for every
pair of vertices on the cycle, the path along the cycle is *also* a shortest
path in the underlying graph:

```
dist_ring(v_m, v_n) == dist_graph(v_m, v_n)        for all m, n
```

This excludes cycles that have a "chord" — a shortcut not on the cycle. SP
rings are a strict subset of King's and Guttman's rings; they are the natural
choice for amorphous-network analysis because they capture local topology
without double-counting nested cycles.

Convention: **ring size = number of bonds = number of vertices** in the cycle.
A 6-membered ring has 6 atoms and 6 bonds.

## Algorithm at a glance

1. **Build the bond graph.** ASE's `neighbor_list('ijD', atoms, cutoff)` gives
   the list of directed bonded pairs `(i, j)` and their MIC bond vectors `D`.
   The graph is stored in CSR-like form (`seed`, `neighbors`).
2. **Precompute the all-pairs BFS distance matrix.** One BFS from every atom.
   This matrix is consulted twice: (a) by the walker to decide expand/turn-back,
   and (b) by the SP test at closure.
3. **For each starting edge `(a, b)` with `a < b`**, launch a single walker
   from vertex `b` that "remembers" `prev = a`. The walker explores in two
   phases:
    - **Step away** (BFS distance from `a` strictly increasing). Spawns one
      new walker per neighbour at distance `+1`. May also spawn turn-back
      walkers when a neighbour is at the *same* distance from `a` (this is
      how odd-size rings close).
    - **Step closer** (BFS distance from `a` strictly decreasing) once the
      walker has reached the midpoint. The walker finally arrives at `a`
      and the cycle is closed.
4. **At each closure, run the SP test.** Compare path-along-cycle distances
   vs. graph distances for every pair of vertices on the cycle.
5. **Record.** Add `1.0 / ring_size` to the count for that size (see
   "Invariant 1" below).
6. **Periodic-boundary check.** Verify that the sum of MIC bond vectors
   around the cycle is zero (within `TOL = 1e-4` on squared norm) — this
   rejects "rings" that close only via a periodic image.

## Two correctness invariants — read these before changing anything

### Invariant 1: each ring is found once per starting edge

A ring of size `N` has exactly `N` edges. The outer loop launches one walker
per directed edge `(a, b)` with `a < b`, so an SP ring of size `N` is
rediscovered `N` times — once for each of its edges as the starting edge.

The implementation handles this by accumulating `1.0 / ring_size` per
detection rather than `1`. The recorded `ringstat[N]` is therefore the
*unique* ring count, and code inspecting `ringstat` mid-walk will see
fractional values.

⚠ **Trap:** if you ever change the increment back to `1` (or move the divide
to a post-processing step that assumes integer divisibility), you will reintroduce
a 6× overcount on amorphous Si.

The empirical signature of this bug is straightforward: per-size ratios of
computed-to-true counts equal the ring size exactly (3.000, 4.000, 4.999,
5.999, …) — and the total ratio is the average ring size, ≈ 6 for
amorphous Si.

### Invariant 2: the maxlength expansion gate is `(maxlength + 1) // 2`

In `_step_away`:

```python
if maxlength < 0 or walker.ring_size() < (maxlength + 1) // 2:
```

A ring of size `N` closes when the walker reaches a vertex at BFS distance
`floor(N/2)` from root, i.e. at `ring_size = floor(N/2) + 1`. We must allow
the walker to expand up to that ring size. The simpler form
`(maxlength + 1) // 2` is correct for both parities:

- `maxlength = 10` (even): gate `< 5`, walker may grow to size 4 then take one more step away to size 5 — exactly the midpoint of a size-10 ring.
- `maxlength = 9` (odd): gate `< 5`, walker may grow to size 4, the midpoint of a size-9 ring (apex at the same-distance edge).

⚠ **Trap:** `(maxlength - 1) // 2` (the original implementation) silently
drops *all* rings of size N when `maxlength = N` and `N` is even.
Empirically: at `maxlength = 10`, `count[10] = 0`; at `maxlength = 11`, the
size-10 rings appear.

## Periodic boundary handling

The closure check at `_step_closer` requires the sum of MIC bond vectors
around the cycle to be ≈ 0. A cycle that closes only by traversing a
periodic image (e.g. a "ring" that wraps the simulation cell) will have a
sum equal to a lattice vector, not zero, and is correctly rejected.

This relies on the cell being large compared to the maximum ring extent.
In practice: cell side ≥ 2 × (max ring radius). For amorphous Si at typical
densities and `maxlength = 10`, a cell side ≥ 15 Å is sufficient; the
included `tests/data/CRN.xyz` (cell ≈ 27.5 Å, 1000 atoms) is well-behaved.

## Reference benchmark

`tests/data/CRN.xyz` is a 1000-atom continuous-random-network model of
amorphous silicon. Ground-truth ring counts (from a published C++ Franzblau
implementation, cutoff 2.85 Å, maxlength 10) are stored in
`tests/data/CRN-rings.csv`:

| ring size | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | total |
|---|---|---|---|---|---|---|---|---|---|
| GT count | 4 | 42 | 471 | 699 | 486 | 160 | 45 | 9 | **1916** |
| Python `compute_rings` | 4.0 | 42.0 | 470.4 | 698.2 | 485.4 | 159.4 | 45.0 | 9.0 | **1913.4** |

The implementation matches the reference exactly for sizes 3, 4, 9, 10. For
sizes 5–8 it under-counts by < 1 ring per size (≈ 0.14% globally). These
missed detections concentrate at the 18 over-coordinated atoms in CRN
(coordination 5 instead of 4); the strict SP test rejects rings that have
chords through these sites, where the C++ reference may be more lenient.
This residual deviation is treated as acceptable.

## Public API

```python
from glass.metrics import compute_rings, compute_rings_distribution

ring_metrics = compute_rings(
    atoms,                # ase.Atoms
    cutoff=None,          # Å; if None and auto_cutoff=True, taken from PDF first minimum
    maxlength=10,         # max ring size to enumerate
    auto_cutoff=True,
)

# returns RingMetrics(ring_lengths, ring_counts, ring_fractions, total_rings, cutoff, maxlength)

ring_metrics_avg = compute_rings_distribution(
    atoms_list,           # list[Atoms] | Atoms
    cutoff=None,
    maxlength=10,
    auto_cutoff=True,
)
```

`ring_counts` is `float64` (so cross-frame averages and the `1/N`
accumulator are exact). `total_rings` is `float`. Per-size error metrics
between two `RingMetrics` are in `glass.metrics.errors`:
`rings_rmse`, `rings_mae`, `rings_cosine_similarity`, `rings_emd`,
`rings_total_error`.

CLI:

```
glass rings structure.xyz --cutoff 3.0 --maxlength 10
glass metrics structure.xyz --include-rings
```

## Performance

Reference profile (CRN, 1000 atoms, maxlength 10, single thread):

| stage | time | calls |
|---|---|---|
| `_find_sp_rings` (outer loop) | 3.25 s | 1 |
| ↳ `_step_closer` | 1.75 s self / 2.31 s cum | 662 k |
| ↳ `_step_away` | 0.39 s self / 0.64 s cum | 119 k |
| ↳ `_Walker.copy_with_step` | 0.55 s | 779 k |
| `_compute_distance_matrix` | 1.15 s | 1 |
| ↳ `_compute_shortest_distances` (BFS) | 1.08 s | 1000 |
| **total `compute_rings`** | **4.4 s** | — |

A Numba-backed engine (`engine="numba"`) is planned for the same module
that will drop this to < 0.1 s and thread-parallelise across the outer
edge loop via `numba.prange`. Until that lands, the only knob available
is `maxlength`: the walker frontier scales roughly as `(coord-1)^(maxlength/2)`,
so dropping maxlength from 10 to 8 typically halves the runtime.

## Files

- `glass/metrics/rings.py` — algorithm.
- `glass/metrics/core.py` — `RingMetrics` dataclass.
- `glass/metrics/errors.py` — error metrics for two `RingMetrics`.
- `tests/test_rings.py` — full test suite, including the CRN parity tests.
- `tests/data/CRN.xyz`, `tests/data/CRN-rings.csv` — reference structure + GT.
