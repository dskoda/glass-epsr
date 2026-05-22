"""glass refine: iterated SDEdit + Tersoff polishing on existing structures.

Loads an already-generated structure (typically from `glass generate`),
re-noises it to an intermediate noise level, runs a partial reverse SDE
back to t=0, and optionally a Tersoff SA tail. Repeats for K cycles or
until per-atom RMSD between consecutive cycles drops below `--rmsd-tol`.

Designed for OOD coord_emd refinement: the score net's prior is
density-blind, but a few short re-noise/denoise cycles let the
likelihood + Tersoff terms iteratively pull coord toward the reference
without rebuilding the structure from scratch.
"""

import copy
import json
import os
from pathlib import Path

import ase.io
import click
import numpy as np
import torch
from ase.io import read

from glass.diffusion.iterated import iterated_refine
from glass.diffusion.schedules import power_law_ts
from glass.diffusion.annealing import make_anneal_fn
from glass.experiment import Experiment
from glass.lit.datamodules import StructureSpecDataModule
from glass.lit.modules import LitScoreNet
from glass.lit.modules.guidance import create_guidance_model
from glass.lit.modules.likelihood import LikelihoodScore
from glass.lit.modules.tersoff_guidance import (
    TersoffEnergyGuidance,
    TersoffSchedule,
)
from glass.utils.atoms import (
    atoms_to_device,
    compute_prior_score,
    compute_target_from_reference,
)


@click.command(
    "refine",
    help="""
Refine an already-generated structure via iterated SDEdit + Tersoff polish.

EXAMPLE:

  glass refine ./my_experiment \\
      --inputs ./generated/init_Si_1.5_0/.../00_final.xyz \\
      --ref-path ./reference \\
      --rho 240 \\
      --t-star-frac 0.2 --n-cycles 5
""",
)
@click.argument("experiment_path", type=click.Path(exists=True))
@click.option("--inputs", required=True, multiple=True, type=click.Path(exists=True),
              help="Paths to input xyz files (can be repeated).")
@click.option("--outdir", type=click.Path(), default=None,
              help="Output dir (default: alongside each input).")
@click.option("--checkpoint", type=str, default="best", show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--cutoff", type=float, default=None)
# Re-noise / iteration controls
@click.option("--t-star-frac", type=float, default=0.2, show_default=True,
              help="Fraction of tmax to use as the re-noise level each cycle.")
@click.option("--n-cycles", type=int, default=5, show_default=True,
              help="Maximum number of refinement cycles.")
@click.option("--rmsd-tol", type=float, default=0.05, show_default=True,
              help="Stop early when per-atom RMSD vs previous cycle < this (Å).")
# Inherited from generate
@click.option("--guidance-type",
              type=click.Choice(["pdf", "adf", "xrd", "nd", "exafs", "xanes"]),
              default=None)
@click.option("--ref-path", type=click.Path(), default=None,
              help="Directory of reference *.xyz (basename matches input).")
@click.option("--rho", type=float, default=None)
@click.option("--bin-size", type=int, default=None)
@click.option("--tmin", type=float, default=None)
@click.option("--tmax", type=float, default=None)
@click.option("--tstep", type=int, default=None)
@click.option("--t-schedule-rho", type=float, default=None)
@click.option("--n-corr", type=int, default=None)
@click.option("--corr-step-size", type=float, default=None)
@click.option("--corr-t-gate", type=float, default=None)
@click.option("--corr-use-tersoff/--no-corr-use-tersoff", default=None)
@click.option("--tersoff-guidance/--no-tersoff-guidance", default=None)
@click.option("--tersoff-lambda", type=float, default=None)
@click.option("--tersoff-schedule",
              type=click.Choice(["constant", "linear", "sigmoid"]), default=None)
@click.option("--tersoff-t-gate", type=float, default=None)
@click.option("--tersoff-clamp", type=float, default=None)
# SA tail (off by default, shared with generate)
@click.option("--sa-n-steps", type=int, default=0, show_default=True)
@click.option("--sa-t0", type=float, default=1e-2, show_default=True)
@click.option("--sa-t-end", type=float, default=1e-5, show_default=True)
@click.option("--sa-lr", type=float, default=1e-3, show_default=True)
@click.option("--sa-lr-clamp", type=float, default=0.2, show_default=True)
def refine(
    experiment_path,
    inputs,
    outdir,
    checkpoint,
    device,
    cutoff,
    t_star_frac,
    n_cycles,
    rmsd_tol,
    guidance_type,
    ref_path,
    rho,
    bin_size,
    tmin,
    tmax,
    tstep,
    t_schedule_rho,
    n_corr,
    corr_step_size,
    corr_t_gate,
    corr_use_tersoff,
    tersoff_guidance,
    tersoff_lambda,
    tersoff_schedule,
    tersoff_t_gate,
    tersoff_clamp,
    sa_n_steps,
    sa_t0,
    sa_t_end,
    sa_lr,
    sa_lr_clamp,
):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    experiment = Experiment(experiment_path)
    config = experiment.load_config()

    # Defaults from config
    device = device or config.device
    cutoff = cutoff or config.cutoff
    tmin = tmin or config.tmin
    tmax = tmax or config.tmax
    tstep = tstep or config.tstep
    t_schedule_rho = t_schedule_rho if t_schedule_rho is not None else config.t_schedule_rho
    n_corr = n_corr if n_corr is not None else config.n_corr
    corr_step_size = corr_step_size if corr_step_size is not None else config.corr_step_size
    corr_t_gate = corr_t_gate if corr_t_gate is not None else config.corr_t_gate
    corr_use_tersoff = corr_use_tersoff if corr_use_tersoff is not None else config.corr_use_tersoff
    tersoff_guidance_flag = tersoff_guidance if tersoff_guidance is not None else config.tersoff_guidance
    tersoff_lambda = tersoff_lambda if tersoff_lambda is not None else config.tersoff_lambda
    tersoff_schedule = tersoff_schedule or config.tersoff_schedule
    tersoff_t_gate = tersoff_t_gate if tersoff_t_gate is not None else config.tersoff_t_gate
    tersoff_clamp = tersoff_clamp if tersoff_clamp is not None else config.tersoff_clamp
    rho = rho if rho is not None else config.rho
    bin_size = bin_size if bin_size is not None else config.bin_size

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")

    ckpt_path = experiment.find_checkpoint(checkpoint)
    score_net = LitScoreNet.load_from_checkpoint(ckpt_path, map_location=device)
    score_net.eval()
    score_net.ema_model.to(device)
    score_net.ema_model.eval()

    datamodule = StructureSpecDataModule(
        data_dir=experiment.get_data_dir_for_datamodule(),
        cutoff=cutoff, train_prior=True, k=config.k,
        train_size=0.9, scale_y=1.0, dup=128,
        batch_size=32, num_workers=0,
    )
    datamodule.setup()
    diffuser = datamodule.train_set.diffuser

    tersoff_guidance_fn = None
    tersoff_schedule_fn = None
    if tersoff_guidance_flag:
        tersoff_guidance_fn = TersoffEnergyGuidance(clamp_norm=tersoff_clamp)
        tersoff_schedule_fn = TersoffSchedule(
            schedule=tersoff_schedule,
            lambda_0=tersoff_lambda,
            tmax=tmax, t_gate=tersoff_t_gate,
        )

    anneal_fn = None
    if sa_n_steps and sa_n_steps > 0:
        sa_guidance = tersoff_guidance_fn or TersoffEnergyGuidance(clamp_norm=tersoff_clamp)
        anneal_fn = make_anneal_fn(
            tersoff_guidance=sa_guidance,
            n_steps=sa_n_steps, T0=sa_t0, T_end=sa_t_end,
            lr=sa_lr, lr_clamp=sa_lr_clamp,
        )

    ts_full = power_law_ts(tmin, tmax, tstep, rho=t_schedule_rho, device=device)

    guidance_model = None
    if guidance_type:
        guidance_model = create_guidance_model(
            guidance_type=guidance_type, device=device, cutoff=cutoff,
            bin_size=bin_size,
        )

    def prior_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser):
        return compute_prior_score(sp, p, c, t, co, _sn, _df)

    for input_path in inputs:
        input_path = Path(input_path)

        # The "sub_id" is the init filename, not the xyz filename. Recover
        # it from a sibling params.json (preferred) or by walking up the
        # tree until we find a directory that looks like an init id.
        sub_id = input_path.stem
        pjson_path = input_path.parent / "params.json"
        if pjson_path.exists():
            try:
                rec = json.loads(pjson_path.read_text())
                if rec.get("sub_id"):
                    sub_id = rec["sub_id"]
            except Exception:
                pass
        else:
            # Heuristic: walk up looking for a "init_*" dir (matches the
            # research/density_extrapolation layout).
            for ancestor in input_path.parents:
                if ancestor.name.startswith("init_"):
                    sub_id = ancestor.name
                    break

        click.echo(f"\nRefining: {input_path}  (sub_id={sub_id})")

        atoms = read(str(input_path), "-1")
        species, pos, cell = atoms_to_device(copy.deepcopy(atoms), device)
        cell_np = cell.detach().cpu().numpy()

        likelihood_fn = None
        if guidance_type:
            ref_xyz = Path(ref_path) / f"{sub_id}.xyz"
            if not ref_xyz.exists():
                click.echo(f"  Warning: no ref for {sub_id} at {ref_xyz}; skipping cond term")
            else:
                ref_atoms = read(str(ref_xyz), "-1")
                target_y = compute_target_from_reference(
                    ref_atoms, guidance_model, guidance_type, cutoff, device,
                )
                likelihood_fn = LikelihoodScore(
                    score_net.ema_model, guidance_model, target_y,
                    rho, diffuser, guidance_type, cutoff,
                )

        cycle_log = []
        def _on_cycle(rec):
            cycle_log.append(rec)
            click.echo(
                f"  cycle {rec['cycle']:2d}  t*={rec['t_star']:.3f}  "
                f"n_steps={rec['n_steps']:3d}  rmsd={rec['rmsd']:.4f}"
            )

        final_pos, _ = iterated_refine(
            species=species, pos=pos, cell=cell, cutoff=cutoff,
            score_fn=prior_fn, likelihood_fn=likelihood_fn,
            ts_full=ts_full, diffuser=diffuser,
            t_star_frac=t_star_frac,
            n_cycles=n_cycles, rmsd_tol=rmsd_tol,
            tersoff_guidance=tersoff_guidance_fn,
            tersoff_schedule=tersoff_schedule_fn,
            n_corr=n_corr, corr_step_size=corr_step_size,
            corr_use_tersoff=corr_use_tersoff, corr_t_gate=corr_t_gate,
            anneal_fn=anneal_fn, progress_fn=_on_cycle,
        )

        # Write output xyz next to input if --outdir not set
        out_dir = Path(outdir) if outdir else input_path.parent / "refined"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_xyz = out_dir / input_path.name

        atoms_out = ase.Atoms(
            numbers=atoms.numbers,
            positions=final_pos.cpu().numpy(),
            cell=cell_np,
            pbc=[True, True, True],
        )
        atoms_out.wrap()
        ase.io.write(str(out_xyz), atoms_out)

        # Side-car params.json so analyze_runs.py picks up hparams
        params = {
            "treatment": "phase_d_iterated",
            "source_xyz": str(input_path),
            "sub_id": sub_id,
            "guidance_type": guidance_type,
            "rho": rho if guidance_type else None,
            "ref_path": ref_path,
            "tmin": tmin, "tmax": tmax, "tstep": tstep,
            "t_schedule_rho": t_schedule_rho,
            "tersoff_guidance": bool(tersoff_guidance_flag),
            "tersoff_lambda": tersoff_lambda if tersoff_guidance_flag else None,
            "tersoff_schedule": tersoff_schedule if tersoff_guidance_flag else None,
            "tersoff_t_gate": tersoff_t_gate if tersoff_guidance_flag else None,
            "tersoff_clamp": tersoff_clamp if tersoff_guidance_flag else None,
            "n_corr": n_corr, "corr_step_size": corr_step_size,
            "corr_t_gate": corr_t_gate, "corr_use_tersoff": corr_use_tersoff,
            "sa_n_steps": sa_n_steps,
            "sa_T0": sa_t0 if sa_n_steps else None,
            "sa_T_end": sa_t_end if sa_n_steps else None,
            "sa_lr": sa_lr if sa_n_steps else None,
            "sa_lr_clamp": sa_lr_clamp if sa_n_steps else None,
            "t_star_frac": t_star_frac,
            "n_cycles_max": n_cycles,
            "n_cycles_run": len(cycle_log),
            "rmsd_tol": rmsd_tol,
            "cycle_log": cycle_log,
        }
        with open(out_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2, default=str)
        click.echo(f"  wrote {out_xyz}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    click.echo("\nDone.")
