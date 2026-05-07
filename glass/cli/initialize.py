"""Utility to initialize random atomic structures for denoising.

This module provides a command to generate initial configurations with
specified density, composition, and minimum distance constraints.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import click
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols, atomic_numbers
from ase.io import write


def _calculate_cell_volume(n_atoms, density, species_list, counts_list):
    """Calculate the cell volume based on density and composition.

    Args:
        n_atoms: Total number of atoms
        density: Density in g/cm^3
        species_list: List of chemical symbols
        counts_list: List of atom counts per species

    Returns:
        Volume in Angstrom^3
    """
    from ase.data import atomic_masses

    # Calculate total mass in amu
    total_mass = sum(
        counts_list[i] * atomic_masses[atomic_numbers[species_list[i]]]
        for i in range(len(species_list))
    )

    # Convert density from g/cm^3 to amu/Angstrom^3
    # 1 g/cm^3 = 0.602214 amu/Angs^3
    density_amu = density * 0.602214

    # Volume = mass / density
    volume = total_mass / density_amu

    return volume


def _generate_random_positions(
    n_atoms, cell, min_distance, max_attempts=100000, rng=None
):
    """Generate random positions with minimum distance constraint.

    Uses rejection sampling with a distance grid for efficiency:
    1. Generate random position
    2. Check distance to all existing atoms (with periodic boundaries)
    3. Accept if all distances >= min_distance, otherwise reject and retry

    Args:
        n_atoms: Number of atoms to place
        cell: 3x3 cell matrix or [a, b, c] for orthorhombic
        min_distance: Minimum distance between any two atoms
        max_attempts: Maximum attempts per atom before giving up
        rng: NumPy random number generator

    Returns:
        Array of shape (n_atoms, 3) with positions

    Raises:
        RuntimeError: If unable to place atoms satisfying constraints
    """
    if rng is None:
        rng = np.random.default_rng()

    if np.ndim(cell) == 1:
        # Orthorhombic cell: [a, b, c]
        cell = np.diag(cell)

    cell_inv = np.linalg.inv(cell)
    positions = []
    min_dist_sq = min_distance ** 2

    def min_image_distance_sq(pos, existing_pos):
        """Calculate minimum image distance squared."""
        diff = pos - existing_pos
        # Apply PBC by converting to fractional, wrapping, back to Cartesian
        diff_frac = diff @ cell_inv
        diff_frac -= np.round(diff_frac)
        diff = diff_frac @ cell
        return np.dot(diff, diff)

    for i in range(n_atoms):
        placed = False
        for attempt in range(max_attempts):
            # Generate random position in fractional coordinates
            frac = rng.random(3)
            # Convert to Cartesian
            pos = frac @ cell

            # Check minimum distance to existing atoms
            too_close = False
            for existing_pos in positions:
                dist_sq = min_image_distance_sq(pos, existing_pos)
                if dist_sq < min_dist_sq:
                    too_close = True
                    break

            if too_close:
                continue  # Too close, try again

            positions.append(pos)
            placed = True
            break

        if not placed:
            raise RuntimeError(
                f"Unable to place atom {i+1}/{n_atoms} after {max_attempts} attempts. "
                f"Try reducing density, increasing cell size, or decreasing min_distance."
            )

    return np.array(positions)


def _generate_structure(
    species_list,
    counts_list,
    density,
    min_distance,
    box_shape="cube",
    rng=None,
):
    """Generate a single random structure.

    Args:
        species_list: List of chemical symbols (e.g., ['Si', 'Ge'])
        counts_list: List of atom counts per species
        density: Density in g/cm^3
        min_distance: Minimum distance between atoms in Angstroms
        box_shape: Shape of the simulation box ('cube' or 'orthorhombic')
        rng: NumPy random number generator

    Returns:
        ASE Atoms object
    """
    if rng is None:
        rng = np.random.default_rng()

    # Total number of atoms
    n_atoms = sum(counts_list)

    # Calculate cell volume
    volume = _calculate_cell_volume(n_atoms, density, species_list, counts_list)

    # Create cell based on shape
    if box_shape == "cube":
        a = volume ** (1 / 3)
        cell = np.eye(3) * a
    elif box_shape == "orthorhombic":
        # For orthorhombic, we use a cubic shape as default
        # User can specify custom cell if needed in future
        a = volume ** (1 / 3)
        cell = np.eye(3) * a
    else:
        raise ValueError(f"Unknown box_shape: {box_shape}")

    # Generate positions
    positions = _generate_random_positions(n_atoms, cell, min_distance, rng=rng)

    # Create symbols list
    symbols = []
    for species, count in zip(species_list, counts_list):
        symbols.extend([species] * count)

    # Shuffle positions to randomize species arrangement
    rng.shuffle(positions)

    # Create Atoms object
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    return atoms


@click.command(
    "initialize",
    help="""
Initialize random atomic structures for denoising.

Generates one or more random structures with specified density, composition,
and minimum distance constraints. The structures are saved to an XYZ file.

REQUIREMENTS:
  - Density (g/cm^3)
  - Number of atoms per species
  - Species symbols (e.g., Si, Ge, C)
  - Minimum distance between atoms (to avoid overlaps)

NOTES:
  - Density and min_distance must be compatible. Random placement with rejection
    sampling has practical limits (~30-40% packing fraction). For Si with 
    min_distance=2.0, use density <= 1.5 g/cm^3; for amorphous structures,
    use lower densities (0.5-1.0 g/cm^3).
  - The actual amorphous Si density is ~2.33 g/cm^3, but random initialization
    requires lower density to avoid atomic overlaps.

EXAMPLES:

  # Single-species (Si) structure - recommended for initialization
  glass initialize --output init_Si.xyz --density 1.0 --species Si --num-atoms 216 --min-distance 2.0

  # Multi-species structure (Si-Ge alloy)
  glass initialize --output init_SiGe.xyz --density 2.0 --species Si Ge --num-atoms 108 108 --min-distance 2.0

  # Multiple structures with different random seeds
  glass initialize --output init_Si.xyz --density 1.0 --species Si --num-atoms 216 --min-distance 2.0 --num-structures 10

  # Custom random seed for reproducibility
  glass initialize --output init_Si.xyz --density 1.0 --species Si --num-atoms 216 --min-distance 2.0 --seed 42

  # Orthorhombic cell (currently same as cube, placeholder for future features)
  glass initialize --output init_Si.xyz --density 1.0 --species Si --num-atoms 216 --min-distance 2.0 --box-shape cube
""",
)
@click.option(
    "--output",
    "-o",
    type=str,
    required=True,
    help="Output XYZ file path.",
)
@click.option(
    "--density",
    "-d",
    type=float,
    required=True,
    help="Density in g/cm^3.",
)
@click.option(
    "--species",
    "-s",
    multiple=True,
    required=True,
    help="Chemical species symbols (e.g., Si, Ge). Can be repeated for multiple species.",
)
@click.option(
    "--num-atoms",
    "-n",
    multiple=True,
    type=int,
    required=True,
    help="Number of atoms for each species. Order must match --species.",
)
@click.option(
    "--min-distance",
    "-m",
    type=float,
    default=1.0,
    show_default=True,
    help="Minimum distance between atoms in Angstroms.",
)
@click.option(
    "--num-structures",
    "-N",
    type=int,
    default=1,
    show_default=True,
    help="Number of structures to generate.",
)
@click.option(
    "--box-shape",
    type=click.Choice(["cube", "orthorhombic"]),
    default="cube",
    show_default=True,
    help="Shape of the simulation box.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility (default: random).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output.",
)
def initialize(
    output,
    density,
    species,
    num_atoms,
    min_distance,
    num_structures,
    box_shape,
    seed,
    verbose,
):
    """Initialize random atomic structures for denoising."""
    # Validate inputs
    if len(species) != len(num_atoms):
        raise click.ClickException(
            f"Number of species ({len(species)}) must match number of --num-atoms values ({len(num_atoms)}). "
            f"Species: {species}, num-atoms: {num_atoms}"
        )

    # Validate species symbols
    for s in species:
        if s not in chemical_symbols:
            raise click.ClickException(
                f"Unknown species symbol: '{s}'. Must be a valid chemical symbol."
            )

    # Validate counts
    for i, n in enumerate(num_atoms):
        if n <= 0:
            raise click.ClickException(
                f"Number of atoms must be positive. Got {n} for species {species[i]}."
            )

    # Validate density
    if density <= 0:
        raise click.ClickException(f"Density must be positive. Got {density}.")

    # Validate min_distance
    if min_distance <= 0:
        raise click.ClickException(f"Minimum distance must be positive. Got {min_distance}.")

    # Initialize random number generator
    if seed is not None:
        rng = np.random.default_rng(seed)
        click.echo(f"Using random seed: {seed}")
    else:
        rng = np.random.default_rng()

    # Convert to lists
    species_list = list(species)
    counts_list = [int(n) for n in num_atoms]
    total_atoms = sum(counts_list)

    click.echo(f"Generating {num_structures} structure(s)...")
    click.echo(f"  Species: {species_list}")
    click.echo(f"  Atoms per species: {counts_list}")
    click.echo(f"  Total atoms: {total_atoms}")
    click.echo(f"  Density: {density} g/cm^3")
    click.echo(f"  Min distance: {min_distance} Angstroms")
    click.echo(f"  Box shape: {box_shape}")
    click.echo(f"  Output: {output}")

    structures = []
    for i in range(num_structures):
        if verbose:
            click.echo(f"Generating structure {i+1}/{num_structures}...")

        # Create a separate RNG for each structure to avoid getting "stuck"
        # Use the base seed + structure index for reproducibility
        if seed is not None:
            structure_rng = np.random.default_rng(seed + i)
        else:
            structure_rng = np.random.default_rng()

        try:
            atoms = _generate_structure(
                species_list=species_list,
                counts_list=counts_list,
                density=density,
                min_distance=min_distance,
                box_shape=box_shape,
                rng=structure_rng,
            )
        except RuntimeError as e:
            raise click.ClickException(f"Failed to generate structure {i+1}: {e}")

        structures.append(atoms)

        if verbose:
            cell = atoms.get_cell()
            click.echo(f"  Cell: a={cell[0, 0]:.3f}, b={cell[1, 1]:.3f}, c={cell[2, 2]:.3f}")
            click.echo(f"  Volume: {atoms.get_volume():.2f} Angstrom^3")

    # Write output file
    if len(structures) == 1:
        write(output, structures[0])
    else:
        write(output, structures)

    click.echo(f"✓ Successfully wrote {len(structures)} structure(s) to {output}")


if __name__ == "__main__":
    initialize()
