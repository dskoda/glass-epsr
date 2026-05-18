#!/usr/bin/env python3
"""
Script to generate structures from initialization file using trained model.

This script loads a trained score model checkpoint and runs the denoising process
on an initialization structure, producing a final generated structure.

Example:
    python scripts/generate_from_init.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
import torch
from ase.io import read, write
from ase import Atoms

from glass.lit.modules import LitScoreNet, DifferentiableRDF
from glass.lit.datamodules import StructureSpecDataModule
from glass.diffusion.sampling import denoise_by_sde
from glass.utils.atoms_utils import atoms_to_device, compute_prior_score


def compute_pdf_comparison(initial_atoms: Atoms, final_atoms: Atoms, cutoff: float = 5.0, bin_size: int = 100):
    """Compute and compare PDFs of initial and final structures."""
    from glass.lit.functions.get_atoms import initialize_atoms
    
    pdf_model = DifferentiableRDF(cutoff=cutoff, bin_size=bin_size, sigma=0.15)
    pdf_model.eval()
    
    # Compute initial PDF
    _, species, pos, cell = initialize_atoms(initial_atoms)
    with torch.no_grad():
        _, init_pdf, _ = pdf_model(pos.cpu(), species.cpu(), cell.cpu())
    
    # Compute final PDF
    _, species, pos, cell = initialize_atoms(final_atoms)
    with torch.no_grad():
        _, final_pdf, _ = pdf_model(pos.cpu(), species.cpu(), cell.cpu())
    
    init_pdf = init_pdf.cpu().numpy()
    final_pdf = final_pdf.cpu().numpy()
    
    return init_pdf, final_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Generate structures from initialization file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tests/data/silicon.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="tests/data/init_random_Si_1.5.xyz",
        help="Path to initialization structure",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated.xyz",
        help="Output file path",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.001,
        help="Minimum time for SDE",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=1.0,
        help="Maximum time for SDE",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Graph cutoff radius",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )
    parser.add_argument(
        "--compare-pdf",
        action="store_true",
        help="Compare PDFs before and after generation",
    )
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    score_net = LitScoreNet.load_from_checkpoint(
        args.checkpoint,
        map_location=args.device,
    )
    score_net.eval()
    score_net.ema_model.to(args.device)
    score_net.ema_model.eval()
    print("Model loaded successfully")
    
    print(f"Loading initialization from {args.init}...")
    init_atoms = read(args.init)
    print(f"Loaded {len(init_atoms)} atoms")
    
    # Prepare tensors
    species, pos, cell = atoms_to_device(init_atoms, args.device)
    
    # Setup datamodule to get diffuser
    data_dir = str(Path(args.checkpoint).parent) + "/"
    datamodule = StructureSpecDataModule(
        data_dir=data_dir,
        cutoff=args.cutoff,
        train_prior=True,
        k=0.8,
        train_size=0.9,
        scale_y=1.0,
        dup=1,
        batch_size=1,
        num_workers=0,
    )
    datamodule.setup()
    diffuser = datamodule.train_set.diffuser
    
    # Create time steps
    ts = torch.linspace(args.tmax, args.tmin, args.n_steps, device=args.device)
    print(f"\nRunning generation with {args.n_steps} steps...")
    print(f"  Device: {args.device}")
    print(f"  t range: [{args.tmin}, {args.tmax}]")
    
    def score_fn(sp, p, c, t, co):
        return compute_prior_score(sp, p, c, t, co, score_net, diffuser)
    
    # Run denoising
    import time
    start_time = time.time()
    
    _, final_pos = denoise_by_sde(
        species=species,
        pos=pos,
        cell=cell,
        cutoff=args.cutoff,
        score_fn=score_fn,
        likelihood_fn=None,
        ts=ts,
        diffuser=diffuser,
        save_traj=False,
        progress_fn=lambda step, t, **kwargs: (
            print(f"  Step {step}/{args.n_steps}: t={t:.4f}", end="\r") 
            if step % 20 == 0 else None
        ) if step < args.n_steps else print(),
    )
    
    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed:.1f}s")
    
    # Create final atoms
    final_atoms = Atoms(
        numbers=init_atoms.numbers,
        positions=final_pos.cpu().numpy(),
        cell=init_atoms.cell,
        pbc=init_atoms.pbc,
    )
    final_atoms.wrap()
    
    # Save output
    write(args.output, final_atoms)
    print(f"Saved generated structure to {args.output}")
    
    # Compare PDFs if requested
    if args.compare_pdf:
        print("\nComparing PDFs...")
        init_pdf, final_pdf = compute_pdf_comparison(init_atoms, final_atoms)
        
        init_first_peak = float(init_pdf.max())
        final_first_peak = float(final_pdf.max())
        
        print(f"  Initial PDF max: {init_first_peak:.2f}")
        print(f"  Final PDF max: {final_first_peak:.2f}")
        print(f"  Change: {((final_first_peak - init_first_peak) / init_first_peak * 100):+.1f}%")


if __name__ == "__main__":
    main()
