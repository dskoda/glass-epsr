import os
import glob
import math
import click
import torch
import numpy as np
from tqdm import tqdm

from glass.experiment import Experiment
from glass.lit.datamodules import StructureSpecDataModule
from glass.lit.modules import LitScoreNet, LitSpecNet


@click.command(
    "generate",
    help="""
Generate atomic structures using a trained score model.

Runs denoising from initial structures to generate new configurations.
Supports both unconditional and conditional (guided) generation.

EXPERIMENT STRUCTURE:
    ./my_experiment/
    ├── config.yaml          # Training configuration
    ├── checkpoints/         # Model checkpoints
    │   ├── best.ckpt
    │   ├── last.ckpt
    │   └── epoch_*.ckpt
    ├── inits/               # Initial structures (*.xyz)
    └── outputs/             # Generated structures

EXAMPLES:

  # Unconditional generation (uses best checkpoint by default)
  glass generate ./my_experiment/ --inits ./my_experiment/inits/

  # Use specific checkpoint
  glass generate ./my_experiment/ --inits ./inits/ --checkpoint last.ckpt

  # Conditional generation with PDF guidance
  glass generate ./my_experiment/ --inits ./inits/ \\
      --guidance-type pdf --ref-path ./reference/

  # Multiple runs per structure
  glass generate ./my_experiment/ --inits ./inits/ --n-runs 20

GUIDANCE TYPES:
  pdf    -- Pair Distribution Function (computational or experimental)
  adf    -- Angular Distribution Function
  xrd    -- X-ray Diffraction
  nd     -- Neutron Diffraction
  exafs  -- Extended X-ray Absorption Fine Structure
  xanes  -- X-ray Absorption Near Edge Structure
""",
)
@click.argument("experiment_path", type=click.Path(exists=True))
@click.option(
    "--inits",
    type=click.Path(),
    required=True,
    help="Directory containing initial structures (*.xyz files).",
)
@click.option(
    "--checkpoint",
    type=str,
    default="best",
    show_default=True,
    help="Checkpoint to use: 'best', 'last', or filename.",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory (default: EXPERIMENT/outputs/).",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Torch device (default: from config or cuda:0).",
)
@click.option(
    "--tmin",
    type=float,
    default=None,
    help="Start time for reverse SDE.",
)
@click.option(
    "--tmax",
    type=float,
    default=None,
    help="End time for reverse SDE.",
)
@click.option(
    "--tstep",
    type=int,
    default=None,
    help="Number of SDE time steps.",
)
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="Graph cutoff radius (Å).",
)
@click.option(
    "--n-runs",
    type=int,
    default=None,
    help="Number of independent runs per structure.",
)
@click.option(
    "--save-traj/--no-save-traj",
    default=None,
    help="Save full trajectory or only final frame.",
)
# Conditional generation options
@click.option(
    "--guidance-type",
    type=click.Choice(["pdf", "adf", "xrd", "nd", "exafs", "xanes"]),
    default=None,
    help="Enable conditional generation with guidance.",
)
@click.option(
    "--ref-path",
    type=click.Path(),
    default=None,
    help="Directory of reference structures for computational guidance.",
)
@click.option(
    "--exp-data",
    type=click.Path(),
    default=None,
    help="JSON file with experimental data for guidance.",
)
@click.option(
    "--rho",
    type=float,
    default=None,
    help="Guidance strength.",
)
@click.option(
    "--spec-model-path",
    type=click.Path(),
    default=None,
    help="[exafs/xanes] Path to spectral model checkpoint.",
)
# PDF options
@click.option(
    "--bin-size",
    type=int,
    default=None,
    help="[pdf] Number of RDF bins.",
)
# ADF options
@click.option(
    "--angle-bins",
    type=int,
    default=None,
    help="[adf] Number of angle bins.",
)
@click.option(
    "--adf-sigma",
    type=float,
    default=None,
    help="[adf] Gaussian kernel width.",
)
@click.option(
    "--adf-cutoff",
    type=float,
    default=None,
    help="[adf] Cutoff radius for triplet search.",
)
# XRD/ND options
@click.option(
    "--element-names",
    "element_names",
    multiple=True,
    default=None,
    help="[xrd/nd] Element names (can be repeated).",
)
@click.option(
    "--qmin",
    type=float,
    default=None,
    help="[xrd/nd] Minimum q value.",
)
@click.option(
    "--qmax",
    type=float,
    default=None,
    help="[xrd/nd] Maximum q value.",
)
@click.option(
    "--qstep",
    type=float,
    default=None,
    help="[xrd/nd] Q step size.",
)
@click.option(
    "--biso",
    type=float,
    default=None,
    help="[xrd/nd] Debye-Waller B factor.",
)
def generate(
    experiment_path,
    inits,
    checkpoint,
    outdir,
    device,
    tmin,
    tmax,
    tstep,
    cutoff,
    n_runs,
    save_traj,
    guidance_type,
    ref_path,
    exp_data,
    rho,
    spec_model_path,
    bin_size,
    angle_bins,
    adf_sigma,
    adf_cutoff,
    element_names,
    qmin,
    qmax,
    qstep,
    biso,
):
    """Generate structures using trained score model."""
    import copy
    import ase
    from ase.io import read
    from torch import nn
    
    from glass.nn import periodic_radius_graph
    from glass.lit.functions.get_atoms import initialize_atoms
    from glass.lit.modules import (
        DifferentiableRDF,
        DifferentiableADF,
        DifferentiableXRD,
        DifferentiableND,
    )
    
    # Set CUDA memory config
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Load experiment
    experiment = Experiment(experiment_path)
    config = experiment.load_config()
    
    # Verify this is a score model experiment
    if config.model_type != "score":
        raise click.ClickException(
            f"Experiment '{experiment_path}' is not a score model (type: {config.model_type}).\n"
            "Generation requires a score model experiment."
        )
    
    # Apply CLI overrides
    if device is None:
        device = config.device
    if tmin is None:
        tmin = config.tmin
    if tmax is None:
        tmax = config.tmax
    if tstep is None:
        tstep = config.tstep
    if cutoff is None:
        cutoff = config.cutoff
    if n_runs is None:
        n_runs = config.n_runs
    if save_traj is None:
        save_traj = config.save_traj
    if rho is None:
        rho = config.rho
    if bin_size is None:
        bin_size = config.bin_size
    if angle_bins is None:
        angle_bins = config.angle_bins
    if adf_sigma is None:
        adf_sigma = config.adf_sigma
    if adf_cutoff is None:
        adf_cutoff = config.adf_cutoff
    if element_names is None or len(element_names) == 0:
        element_names = config.element_names
    if qmin is None:
        qmin = config.qmin
    if qmax is None:
        qmax = config.qmax
    if qstep is None:
        qstep = config.qstep
    if biso is None:
        biso = config.biso
    
    # Resolve output directory
    if outdir is None:
        outdir = experiment.outputs_dir
    else:
        outdir = os.path.join(outdir)
    os.makedirs(outdir, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")
    if torch.cuda.is_available():
        click.echo(f"GPU: {torch.cuda.get_device_name(device.index or 0)}")
    
    # Load checkpoint
    ckpt_path = experiment.find_checkpoint(checkpoint)
    click.echo(f"Loading checkpoint: {ckpt_path}")
    
    # Load model
    score_net = LitScoreNet.load_from_checkpoint(ckpt_path, map_location=device)
    score_net.eval()
    score_net.ema_model.to(device)
    score_net.ema_model.eval()
    
    # Setup datamodule to get diffuser
    datamodule = StructureSpecDataModule(
        data_dir=experiment.get_data_dir_for_datamodule(),
        cutoff=cutoff,
        train_prior=True,
        k=config.k,
        train_size=0.9,
        scale_y=1.0,
        dup=128,
        batch_size=32,
        num_workers=8,
    )
    datamodule.setup()
    diffuser = datamodule.train_set.diffuser
    
    # Setup guidance model if conditional
    guidance_model = None
    if guidance_type:
        click.echo(f"Guidance type: {guidance_type}")
        
        if guidance_type == "pdf":
            guidance_model = DifferentiableRDF(
                cutoff=cutoff, bin_size=bin_size, sigma=0.15
            )
            guidance_model.eval()
        elif guidance_type == "adf":
            guidance_model = DifferentiableADF(
                cutoff=adf_cutoff,
                angle_bins=angle_bins,
                angle_range=[0, math.pi],
                sigma=adf_sigma,
                normalize=False,
            )
            guidance_model.eval()
        elif guidance_type == "xrd":
            if not element_names:
                raise click.ClickException(
                    "--element-names required for xrd guidance"
                )
            guidance_model = DifferentiableXRD(
                q_vals=[qmin, qmax, qstep],
                element_names=list(element_names),
                biso=biso,
            )
            guidance_model.to(device).eval()
        elif guidance_type == "nd":
            if not element_names:
                raise click.ClickException(
                    "--element-names required for nd guidance"
                )
            guidance_model = DifferentiableND(
                q_vals=[qmin, qmax, qstep],
                element_names=list(element_names),
                biso=biso,
            )
            guidance_model.to(device).eval()
        elif guidance_type in ("exafs", "xanes"):
            if not spec_model_path:
                raise click.ClickException(
                    f"--spec-model-path required for {guidance_type} guidance"
                )
            spec_net = LitSpecNet.load_from_checkpoint(spec_model_path, map_location=device)
            spec_net.eval()
            spec_net.ema_model.to(device).eval()
            guidance_model = spec_net.ema_model
        
        # Validate ref_path vs exp_data
        if not ref_path and not exp_data:
            raise click.ClickException(
                "Conditional generation requires --ref-path or --exp-data"
            )
        if ref_path and exp_data:
            raise click.ClickException("--ref-path and --exp-data are mutually exclusive")
    
    # Load experimental target if specified
    exp_target_y = None
    if exp_data:
        import json
        with open(exp_data) as f:
            d = json.load(f)
        x_exp = np.array(d.get("x", d.get("r")))
        y_exp = np.array(d.get("y", d.get("g")))
        
        if guidance_type == "pdf":
            bin_edges = np.linspace(0, cutoff, bin_size + 1)
            x_grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        elif guidance_type == "adf":
            x_grid = np.linspace(0, math.pi, angle_bins)
        elif guidance_type in ("xrd", "nd"):
            x_grid = np.arange(qmin, qmax, qstep)
        else:
            raise click.ClickException(
                f"--exp-data not yet supported for {guidance_type}"
            )
        
        y_interp = np.interp(x_grid, x_exp, y_exp)
        exp_target_y = torch.from_numpy(y_interp).float().unsqueeze(0).to(device)
        click.echo(f"Loaded experimental target from {exp_data}")
    
    # Helper functions
    def _to_device(atoms):
        _, species, pos, cell = initialize_atoms(atoms)
        return (
            species.to(device),
            pos.to(device=device, dtype=torch.float32),
            cell.to(device=device, dtype=torch.float32),
        )
    
    def _prior_score(species, pos, cell, t, cut, score_net, diffuser):
        edge_index, edge_vec = periodic_radius_graph(pos, cut, cell)
        edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
        return score_net.ema_model(species, edge_index, edge_attr, t, diffuser.sigma(t))
    
    # Likelihood score class for conditional generation
    class LikelihoodScore(nn.Module):
        def __init__(
            self, score_net, guidance_model, target_y, rho, diffuser, guidance_type, cut
        ):
            super().__init__()
            self.score_net = score_net
            self.guidance_model = guidance_model
            self.target_y = target_y
            self.rho = rho
            self.diffuser = diffuser
            self.guidance_type = guidance_type
            self.cut = cut
        
        def forward(self, species, pos, cell, t, cut):
            with torch.enable_grad():
                pos = pos.detach().clone().requires_grad_(True)
                edge_index, edge_vec = periodic_radius_graph(pos, cut, cell)
                edge_attr = torch.hstack(
                    [edge_vec, edge_vec.norm(dim=-1, keepdim=True)]
                )
                sigma = self.diffuser.sigma(t)
                with torch.no_grad():
                    score = self.score_net(species, edge_index, edge_attr, t, sigma)
                est_clean_pos = pos + sigma.pow(2) * score
                
                if self.guidance_type in ("pdf", "adf"):
                    pred_y = self.guidance_model(
                        est_clean_pos.cpu(), species.cpu(), cell.cpu()
                    )[1].to(pos.device)
                    norm = torch.linalg.norm(
                        self.target_y - pred_y, dim=1, keepdim=True
                    )
                elif self.guidance_type in ("xrd", "nd"):
                    pred_y = self.guidance_model(est_clean_pos, species)
                    norm = torch.linalg.norm(self.target_y - pred_y)
                elif self.guidance_type in ("exafs", "xanes"):
                    ei2, ev2 = periodic_radius_graph(est_clean_pos, self.cut, cell)
                    ea2 = torch.hstack([ev2, ev2.norm(dim=-1, keepdim=True)])
                    pred_y = self.guidance_model(species, ei2, ea2)
                    norm = torch.linalg.norm(
                        self.target_y - pred_y, dim=1, keepdim=True
                    )
                
                loss = norm.square().mean()
                grad = torch.autograd.grad(loss, est_clean_pos)[0]
            return -(self.rho / (norm.sum() + 1e-12)) * grad.detach(), norm.detach()
    
    # SDE denoising
    def _denoise_by_sde(
        species, pos, cell, cut, score_fn, likelihood_fn, ts, diffuser, save_traj
    ):
        ts = ts.to(pos.device).view(-1, 1)
        f, g, g2 = diffuser.f, diffuser.g, diffuser.g2
        traj = [pos.detach().cpu().clone()] if save_traj else None
        pos = pos.detach()
        
        for i, t in enumerate(ts[1:]):
            dt = ts[i + 1] - ts[i]
            eps = dt.abs().sqrt() * torch.randn_like(pos)
            with torch.no_grad():
                p_score = score_fn(species, pos, cell, t, cut)
            
            if likelihood_fn is not None:
                l_score, norm = likelihood_fn(species, pos, cell, t, cut)
                click.echo(
                    f"    p={p_score.norm().item():.3f}  "
                    f"l={l_score.norm().item():.3f}  "
                    f"tgt={norm.sum().item():.4f}",
                    err=True,
                )
                disp = (f(t) * pos - g2(t) * (p_score + l_score)) * dt + g(t) * eps
            else:
                disp = (f(t) * pos - g2(t) * p_score) * dt + g(t) * eps
            
            pos = (pos + disp).detach()
            if save_traj:
                traj.append(pos.cpu().clone())
        
        return traj if save_traj else pos
    
    # Get initial structures
    init_files = experiment.get_init_files(inits)
    if not init_files:
        raise click.ClickException(f"No .xyz files found in {inits}")
    click.echo(f"Found {len(init_files)} initial structures")
    
    # Time steps
    ts_torch = torch.linspace(tmax, tmin, tstep, device=device)
    
    # Process each initial structure
    for init_file in init_files:
        init_atoms = read(init_file, "-1")
        sub_id = os.path.basename(init_file).replace(".xyz", "")
        click.echo(f"\nProcessing: {sub_id}")
        
        # Get reference target if needed
        target_y = None
        if ref_path:
            ref_file = os.path.join(ref_path, f"{sub_id}.xyz")
            if not os.path.exists(ref_file):
                click.echo(f"  Warning: reference {ref_file} not found, skipping")
                continue
            ref_atoms = read(ref_file, "-1")
            if not (np.all(init_atoms.pbc) and np.all(ref_atoms.pbc)):
                raise click.ClickException(f"PBC must be True for both init and ref")
            if not np.allclose(init_atoms.get_cell(), ref_atoms.get_cell(), atol=1e-5):
                raise click.ClickException(f"Init and ref cells must match")
            
            _, ref_species, ref_pos, ref_cell = initialize_atoms(ref_atoms)
            
            if guidance_type in ("pdf", "adf"):
                target_y = guidance_model(
                    ref_pos.cpu(), ref_species.cpu(), ref_cell.cpu()
                )[1].to(device)
            elif guidance_type in ("xrd", "nd"):
                target_y = guidance_model(
                    ref_pos.to(device), ref_species.to(device)
                )
            elif guidance_type in ("exafs", "xanes"):
                ei_r, ev_r = periodic_radius_graph(
                    ref_pos.to(device), cutoff, ref_cell.to(device)
                )
                ea_r = torch.hstack([ev_r, ev_r.norm(dim=-1, keepdim=True)])
                with torch.no_grad():
                    target_y = guidance_model(ref_species.to(device), ei_r, ea_r)
        elif exp_data:
            target_y = exp_target_y
        
        # Setup likelihood function if conditional
        likelihood_fn = None
        if guidance_type:
            likelihood_fn = LikelihoodScore(
                score_net.ema_model,
                guidance_model,
                target_y,
                rho,
                diffuser,
                guidance_type,
                cutoff,
            )
        
        # Run generation
        run_outdir = os.path.join(outdir, sub_id)
        if guidance_type:
            rho_str = f"{int(rho)}" if rho == int(rho) else f"{rho:.2f}"
            run_tag = f"{guidance_type}_rho{rho_str}_tmax{tmax}_nsteps{tstep}"
            run_outdir = os.path.join(outdir, sub_id, run_tag)
        os.makedirs(run_outdir, exist_ok=True)
        
        def prior_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser):
            return _prior_score(sp, p, c, t, co, _sn, _df)
        
        for i in range(n_runs):
            species, pos, cell = _to_device(copy.deepcopy(init_atoms))
            cell_np = cell.detach().cpu().numpy()
            
            result = _denoise_by_sde(
                species,
                pos,
                cell,
                cutoff,
                prior_fn,
                likelihood_fn,
                ts_torch,
                diffuser,
                save_traj,
            )
            
            if save_traj:
                traj = []
                for p in result:
                    a = ase.Atoms(
                        numbers=init_atoms.numbers,
                        positions=p.numpy(),
                        cell=cell_np,
                        pbc=[True] * 3,
                    )
                    a.wrap()
                    traj.append(a)
                ase.io.write(f"{run_outdir}/{i:02}_traj.xyz", traj)
                ase.io.write(f"{run_outdir}/{i:02}_final.xyz", traj[-1])
            else:
                final = ase.Atoms(
                    numbers=init_atoms.numbers,
                    positions=result.cpu().numpy(),
                    cell=cell_np,
                    pbc=[True] * 3,
                )
                final.wrap()
                ase.io.write(f"{run_outdir}/{i:02}_final.xyz", final)
            
            click.echo(f"  Run {i+1}/{n_runs} complete")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    click.echo(f"\nGeneration complete! Results saved to: {outdir}")
