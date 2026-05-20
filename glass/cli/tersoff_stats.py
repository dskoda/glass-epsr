"""glass tersoff-stats — batch Tersoff energy/force analysis command."""

import json
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
from ase.io import read

from glass.metrics.tersoff import (
    TersoffMetrics,
    compute_tersoff_metrics,
    tersoff_energy_error,
    tersoff_forces_emd,
    tersoff_forces_histogram_rmse,
    tersoff_forces_max_error,
    tersoff_forces_rms_error,
)


@click.command(
    "tersoff-stats",
    help="""
Compute Tersoff potential energy and force statistics for Si structures.

For each structure computes:
  - Energy per atom [eV/atom]
  - RMS, max, mean, std of per-atom force magnitudes [eV/Å]
  - Normalised histogram of per-atom force magnitudes

If --ref is given, also computes error metrics between each structure and
the reference (energy_error, forces_rms_error, forces_max_error,
forces_histogram_rmse, forces_emd).

EXAMPLES:

  # Single-structure stats
  glass tersoff-stats structure.xyz

  # Batch
  glass tersoff-stats ./generated/*.xyz --output tersoff.json

  # Compare against a reference structure
  glass tersoff-stats ./generated/*.xyz --ref reference.xyz --output tersoff.json
""",
)
@click.argument(
    "structures",
    nargs=-1,
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="tersoff_stats.json",
    show_default=True,
    help="Output JSON file path.",
)
@click.option(
    "--ref",
    "ref_path",
    type=click.Path(exists=True),
    default=None,
    help="Reference structure or JSON of reference metrics for error comparison.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Torch device ('cpu' or 'cuda').",
)
@click.option(
    "--indent",
    type=int,
    default=2,
    show_default=True,
    help="JSON indentation level.",
)
def tersoff_stats(
    structures: List[str],
    output: str,
    ref_path: Optional[str],
    device: str,
    indent: int,
):
    """Compute Tersoff energy and force statistics for Si structures."""
    # Load reference metrics if supplied
    ref_metrics: Optional[TersoffMetrics] = None
    if ref_path is not None:
        rp = Path(ref_path)
        if rp.suffix == ".json":
            with open(rp) as f:
                data = json.load(f)
            # Accept either a bare TersoffMetrics dict or a structures-keyed output
            if "energy_per_atom" in data:
                ref_metrics = TersoffMetrics.from_dict(data)
            else:
                # Take the first structure entry that has tersoff data
                for v in data.get("structures", {}).values():
                    if "tersoff" in v and "error" not in v:
                        ref_metrics = TersoffMetrics.from_dict(v["tersoff"])
                        break
            if ref_metrics is None:
                click.echo(
                    f"WARNING: could not find Tersoff metrics in {ref_path}", err=True
                )
        else:
            ref_atoms = read(str(rp))
            click.echo(f"Computing reference Tersoff metrics: {rp.name}")
            ref_metrics = compute_tersoff_metrics(ref_atoms, device=device)
            click.echo(
                f"  E/atom={ref_metrics.energy_per_atom:.4f} eV/atom  "
                f"F_rms={ref_metrics.forces_rms:.4f} eV/Å  "
                f"F_max={ref_metrics.forces_max:.4f} eV/Å"
            )
        click.echo()

    results = {}
    for structure_path in structures:
        path = Path(structure_path)
        click.echo(f"Processing: {path.name}")
        try:
            atoms = read(str(path))
            m = compute_tersoff_metrics(atoms, device=device)

            click.echo(
                f"  E/atom={m.energy_per_atom:.4f} eV/atom  "
                f"F_rms={m.forces_rms:.4f} eV/Å  "
                f"F_max={m.forces_max:.4f} eV/Å  "
                f"F_mean={m.forces_mean:.4f} eV/Å"
            )

            entry = m.to_dict()

            if ref_metrics is not None:
                errors = {
                    "energy_error": tersoff_energy_error(ref_metrics, m),
                    "forces_rms_error": tersoff_forces_rms_error(ref_metrics, m),
                    "forces_max_error": tersoff_forces_max_error(ref_metrics, m),
                    "forces_histogram_rmse": tersoff_forces_histogram_rmse(ref_metrics, m),
                    "forces_emd": tersoff_forces_emd(ref_metrics, m),
                }
                entry["errors_vs_ref"] = errors
                click.echo(
                    f"  vs ref: ΔE/atom={errors['energy_error']:.4f} eV  "
                    f"ΔF_rms={errors['forces_rms_error']:.4f} eV/Å  "
                    f"F_emd={errors['forces_emd']:.4f}"
                )

            results[path.name] = entry
            click.echo()

        except Exception as e:
            click.echo(f"  ERROR: {e}", err=True)
            results[path.name] = {"error": str(e)}

    # Aggregate summary across all successful results
    valid = {
        k: v for k, v in results.items() if "error" not in v
    }
    summary: dict = {}
    if valid:
        e_vals = [v["energy_per_atom"] for v in valid.values()]
        fr_vals = [v["forces_rms"] for v in valid.values()]
        fm_vals = [v["forces_max"] for v in valid.values()]
        summary = {
            "n_structures": len(valid),
            "energy_per_atom_mean": float(np.mean(e_vals)),
            "energy_per_atom_std": float(np.std(e_vals)),
            "forces_rms_mean": float(np.mean(fr_vals)),
            "forces_rms_std": float(np.std(fr_vals)),
            "forces_max_mean": float(np.mean(fm_vals)),
            "forces_max_std": float(np.std(fm_vals)),
        }
        if ref_metrics is not None:
            ee = [v["errors_vs_ref"]["energy_error"] for v in valid.values()]
            fe = [v["errors_vs_ref"]["forces_emd"] for v in valid.values()]
            summary["mean_energy_error"] = float(np.mean(ee))
            summary["mean_forces_emd"] = float(np.mean(fe))

        click.echo("Summary:")
        click.echo(
            f"  E/atom = {summary['energy_per_atom_mean']:.4f} ± "
            f"{summary['energy_per_atom_std']:.4f} eV/atom"
        )
        click.echo(
            f"  F_rms  = {summary['forces_rms_mean']:.4f} ± "
            f"{summary['forces_rms_std']:.4f} eV/Å"
        )
        if ref_metrics is not None:
            click.echo(
                f"  mean ΔE/atom vs ref = {summary['mean_energy_error']:.4f} eV"
            )
            click.echo(
                f"  mean F EMD vs ref   = {summary['mean_forces_emd']:.4f} eV/Å"
            )

    output_data = {
        "metadata": {
            "n_structures": len(structures),
            "device": device,
            "ref": ref_path,
        },
        "summary": summary,
        "structures": results,
    }
    if ref_metrics is not None:
        output_data["reference"] = ref_metrics.to_dict()

    out_path = Path(output)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=indent)
    click.echo(f"\nSaved to {out_path}")
