import os
import click
import torch
import numpy as np
from tqdm import tqdm

from glass.experiment import Experiment
from glass.lit.datamodules import StructureSpecDataModule
from glass.lit.modules import LitScoreNet, LitSpecNet
from glass.diffusion.sampling import denoise_by_sde
from glass.lit.modules.likelihood import LikelihoodScore
from glass.lit.modules.guidance import create_guidance_model, load_experimental_data
from glass.lit.modules.tersoff_guidance import (
    TersoffEnergyGuidance,
    TersoffSchedule,
)
from glass.descriptors import TorchACSF, EntropyGuidance, EntropySchedule
from glass.diffusion.schedules import power_law_ts
from glass.diffusion.annealing import make_anneal_fn
from glass.utils.atoms_utils import (
    atoms_to_device,
    compute_prior_score,
    compute_target_from_reference,
)


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
# Tersoff (empirical-potential) guidance
@click.option(
    "--tersoff-guidance/--no-tersoff-guidance",
    default=None,
    help="Add Tersoff-energy gradient as an auxiliary score term during the reverse SDE.",
)
@click.option(
    "--tersoff-lambda",
    type=float,
    default=None,
    help="[tersoff] Weight lambda_0 for the Tersoff guidance schedule.",
)
@click.option(
    "--tersoff-schedule",
    type=click.Choice(["constant", "linear", "sigmoid"]),
    default=None,
    help="[tersoff] Shape of lambda(t).",
)
@click.option(
    "--tersoff-t-gate",
    type=float,
    default=None,
    help="[tersoff] Gate time for the sigmoid schedule.",
)
@click.option(
    "--tersoff-clamp",
    type=float,
    default=None,
    help="[tersoff] Per-atom guidance-norm clamp (Å units of autograd/N).",
)
@click.option(
    "--tersoff-tweedie/--no-tersoff-tweedie",
    default=None,
    help="[tersoff] Evaluate Tersoff on the Tweedie denoised estimate x̂₀ = x_t + σ²·score "
         "rather than on the noisy x_t. Default: True.",
)
# Langevin predictor-corrector
@click.option(
    "--n-corr",
    type=int,
    default=None,
    help="Number of Langevin corrector steps per predictor step. 0 disables.",
)
@click.option(
    "--corr-step-size",
    type=float,
    default=None,
    help="Corrector step size (effective step = corr_step_size * sigma(t)^2).",
)
@click.option(
    "--corr-use-tersoff/--no-corr-use-tersoff",
    default=None,
    help="Include Tersoff gradient inside the corrector loop.",
)
@click.option(
    "--corr-t-gate",
    type=float,
    default=None,
    help="Skip corrector when t > corr_t_gate * t_max.",
)
# Non-linear t schedule
@click.option(
    "--t-schedule-rho",
    type=float,
    default=None,
    help="Power-law exponent for the t trajectory (1.0 = linspace, >1 concentrates near t=0).",
)
# Simulated-annealing post-relaxation
@click.option(
    "--sa-n-steps",
    type=int,
    default=None,
    help="Number of simulated-annealing steps after the reverse SDE. 0 disables.",
)
@click.option(
    "--sa-t0",
    type=float,
    default=None,
    help="[SA] Initial temperature.",
)
@click.option(
    "--sa-t-end",
    type=float,
    default=None,
    help="[SA] Final temperature.",
)
@click.option(
    "--sa-lr",
    type=float,
    default=None,
    help="[SA] Step size on the Tersoff force.",
)
@click.option(
    "--sa-lr-clamp",
    type=float,
    default=None,
    help="[SA] Per-atom per-step displacement cap (Å).",
)
# Structural-entropy (ACSF variance) guidance
@click.option(
    "--entropy-guidance/--no-entropy-guidance",
    default=None,
    help="Add ACSF descriptor-variance gradient as an auxiliary score term.",
)
@click.option(
    "--entropy-lambda",
    type=float,
    default=None,
    help="[entropy] Weight lambda_0 for the entropy guidance schedule.",
)
@click.option(
    "--entropy-schedule",
    type=click.Choice(["constant", "linear", "sigmoid"]),
    default=None,
    help="[entropy] Shape of lambda(t).",
)
@click.option(
    "--entropy-t-gate",
    type=float,
    default=None,
    help="[entropy] Gate time fraction (active when t/tmax <= t_gate).",
)
@click.option(
    "--entropy-r-cut",
    type=float,
    default=None,
    help="[entropy] ACSF cutoff radius (Å).",
)
@click.option(
    "--n-restart",
    type=int,
    default=None,
    show_default=True,
    help="Number of full denoising passes per structure. 1 = single pass (default). "
         "Each restart starts from the previous pass output (same cell/species/guidance).",
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
    tersoff_guidance,
    tersoff_lambda,
    tersoff_schedule,
    tersoff_t_gate,
    tersoff_clamp,
    tersoff_tweedie,
    n_corr,
    corr_step_size,
    corr_use_tersoff,
    corr_t_gate,
    t_schedule_rho,
    sa_n_steps,
    sa_t0,
    sa_t_end,
    sa_lr,
    sa_lr_clamp,
    entropy_guidance,
    entropy_lambda,
    entropy_schedule,
    entropy_t_gate,
    entropy_r_cut,
    n_restart,
):
    """Generate structures using trained score model."""
    import ase
    from ase.io import read
    
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
    device = device or config.device
    tmin = tmin or config.tmin
    tmax = tmax or config.tmax
    tstep = tstep or config.tstep
    cutoff = cutoff or config.cutoff
    n_runs = n_runs or config.n_runs
    save_traj = save_traj if save_traj is not None else config.save_traj
    rho = rho or config.rho
    bin_size = bin_size or config.bin_size
    angle_bins = angle_bins or config.angle_bins
    adf_sigma = adf_sigma or config.adf_sigma
    adf_cutoff = adf_cutoff or config.adf_cutoff
    element_names = list(element_names) if element_names else config.element_names
    qmin = qmin or config.qmin
    qmax = qmax or config.qmax
    qstep = qstep or config.qstep
    biso = biso or config.biso
    # Tersoff guidance
    tersoff_guidance = (
        tersoff_guidance if tersoff_guidance is not None else config.tersoff_guidance
    )
    tersoff_lambda = (
        tersoff_lambda if tersoff_lambda is not None else config.tersoff_lambda
    )
    tersoff_schedule = tersoff_schedule or config.tersoff_schedule
    tersoff_t_gate = (
        tersoff_t_gate if tersoff_t_gate is not None else config.tersoff_t_gate
    )
    tersoff_clamp = (
        tersoff_clamp if tersoff_clamp is not None else config.tersoff_clamp
    )
    tersoff_tweedie = (
        tersoff_tweedie if tersoff_tweedie is not None else getattr(config, "tersoff_tweedie", True)
    )
    # Sampler refinements
    n_corr = n_corr if n_corr is not None else config.n_corr
    corr_step_size = corr_step_size if corr_step_size is not None else config.corr_step_size
    corr_use_tersoff = (
        corr_use_tersoff if corr_use_tersoff is not None else config.corr_use_tersoff
    )
    corr_t_gate = corr_t_gate if corr_t_gate is not None else config.corr_t_gate
    t_schedule_rho = (
        t_schedule_rho if t_schedule_rho is not None else config.t_schedule_rho
    )
    sa_n_steps = sa_n_steps if sa_n_steps is not None else config.sa_n_steps
    sa_t0 = sa_t0 if sa_t0 is not None else config.sa_T0
    sa_t_end = sa_t_end if sa_t_end is not None else config.sa_T_end
    sa_lr = sa_lr if sa_lr is not None else config.sa_lr
    sa_lr_clamp = sa_lr_clamp if sa_lr_clamp is not None else config.sa_lr_clamp
    # Entropy guidance
    entropy_guidance = (
        entropy_guidance if entropy_guidance is not None
        else getattr(config, "entropy_guidance", False)
    )
    entropy_lambda = (
        entropy_lambda if entropy_lambda is not None
        else getattr(config, "entropy_lambda", 1.0)
    )
    entropy_schedule = entropy_schedule or getattr(config, "entropy_schedule", "constant")
    entropy_t_gate = (
        entropy_t_gate if entropy_t_gate is not None
        else getattr(config, "entropy_t_gate", 1.0)
    )
    entropy_r_cut = (
        entropy_r_cut if entropy_r_cut is not None
        else getattr(config, "entropy_r_cut", 4.0)
    )
    n_restart = n_restart if n_restart is not None else getattr(config, "n_restart", 1)
    n_restart = max(1, int(n_restart))
    
    # Resolve output directory
    if outdir is None:
        outdir = experiment.outputs_dir
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
    
    # Setup Tersoff-energy guidance if requested (independent of conditional guidance).
    tersoff_guidance_fn = None
    tersoff_schedule_fn = None
    if tersoff_guidance:
        click.echo(
            f"Tersoff guidance: schedule={tersoff_schedule} "
            f"lambda0={tersoff_lambda} t_gate={tersoff_t_gate} "
            f"clamp={tersoff_clamp}"
        )
        tersoff_guidance_fn = TersoffEnergyGuidance(
            clamp_norm=tersoff_clamp,
        )
        tersoff_schedule_fn = TersoffSchedule(
            schedule=tersoff_schedule,
            lambda_0=tersoff_lambda,
            tmax=tmax,
            t_gate=tersoff_t_gate,
        )

    entropy_guidance_fn = None
    entropy_schedule_fn = None
    if entropy_guidance:
        click.echo(
            f"Entropy guidance: schedule={entropy_schedule} "
            f"lambda0={entropy_lambda} t_gate={entropy_t_gate} "
            f"r_cut={entropy_r_cut}"
        )
        entropy_guidance_fn = EntropyGuidance(
            acsf=TorchACSF.for_silicon(r_cut=entropy_r_cut),
        )
        entropy_schedule_fn = EntropySchedule(
            schedule=entropy_schedule,
            lambda_0=entropy_lambda,
            tmax=tmax,
            t_gate=entropy_t_gate,
        )

    # Simulated-annealing post-relaxation: SA always runs on the Tersoff PES,
    # so auto-create a guidance instance if the user did not enable Tersoff.
    anneal_fn = None
    sa_guidance_fn = tersoff_guidance_fn
    if sa_n_steps and sa_n_steps > 0:
        if sa_guidance_fn is None:
            sa_guidance_fn = TersoffEnergyGuidance(clamp_norm=tersoff_clamp)
        click.echo(
            f"SA tail: n_steps={sa_n_steps} T0={sa_t0} T_end={sa_t_end} "
            f"lr={sa_lr} lr_clamp={sa_lr_clamp}"
        )
        anneal_fn = make_anneal_fn(
            tersoff_guidance=sa_guidance_fn,
            n_steps=sa_n_steps,
            T0=sa_t0,
            T_end=sa_t_end,
            lr=sa_lr,
            lr_clamp=sa_lr_clamp,
        )

    if n_corr and n_corr > 0:
        click.echo(
            f"Corrector: n_corr={n_corr} step_size={corr_step_size} "
            f"use_tersoff={corr_use_tersoff} t_gate={corr_t_gate}"
        )
    if t_schedule_rho != 1.0:
        click.echo(f"Non-linear t schedule: rho={t_schedule_rho}")

    # Setup guidance model if conditional
    guidance_model = None
    if guidance_type:
        click.echo(f"Guidance type: {guidance_type}")
        
        guidance_model = create_guidance_model(
            guidance_type=guidance_type,
            device=device,
            cutoff=cutoff,
            bin_size=bin_size,
            angle_bins=angle_bins,
            adf_sigma=adf_sigma,
            adf_cutoff=adf_cutoff,
            element_names=element_names,
            qmin=qmin,
            qmax=qmax,
            qstep=qstep,
            biso=biso,
            spec_model_path=spec_model_path,
        )
        
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
        exp_target_y = load_experimental_data(
            exp_data_path=exp_data,
            guidance_type=guidance_type,
            cutoff=cutoff,
            bin_size=bin_size,
            angle_bins=angle_bins,
            qmin=qmin,
            qmax=qmax,
            qstep=qstep,
            device=device,
        )
        click.echo(f"Loaded experimental target from {exp_data}")
    
    # Get initial structures
    init_files = experiment.get_init_files(inits)
    if not init_files:
        raise click.ClickException(f"No .xyz files found in {inits}")
    click.echo(f"Found {len(init_files)} initial structures")
    
    # Time steps (power-law; rho=1.0 == linspace)
    ts_torch = power_law_ts(tmin, tmax, tstep, rho=t_schedule_rho, device=device)
    
    # Progress callback for denoising
    def progress_callback(step, t, p_norm, l_norm=None, target_norm=None):
        if l_norm is not None:
            click.echo(
                f"    p={p_norm:.3f}  l={l_norm:.3f}  tgt={target_norm:.4f}",
                err=True,
            )
    
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
                raise click.ClickException("PBC must be True for both init and ref")
            if not np.allclose(init_atoms.get_cell(), ref_atoms.get_cell(), atol=1e-5):
                raise click.ClickException("Init and ref cells must match")
            
            target_y = compute_target_from_reference(
                ref_atoms, guidance_model, guidance_type, cutoff, device
            )
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
        tag_parts = []
        if guidance_type:
            rho_str = f"{int(rho)}" if rho == int(rho) else f"{rho:.2f}"
            tag_parts.append(f"{guidance_type}_rho{rho_str}")
        if tersoff_guidance:
            tag_parts.append(
                f"tersoff_{tersoff_schedule}_lam{tersoff_lambda:g}"
                f"_gate{tersoff_t_gate:g}"
            )
        tag_parts.append(f"tmax{tmax}_nsteps{tstep}")
        run_outdir = os.path.join(outdir, sub_id, "_".join(tag_parts)) \
            if tag_parts else os.path.join(outdir, sub_id)

        extra_tag = []
        if n_corr and n_corr > 0:
            extra_tag.append(f"corr{n_corr}s{corr_step_size:g}")
        if t_schedule_rho != 1.0:
            extra_tag.append(f"rho{t_schedule_rho:g}")
        if sa_n_steps and sa_n_steps > 0:
            extra_tag.append(f"sa{sa_n_steps}lr{sa_lr:g}")
        if entropy_guidance:
            extra_tag.append(f"ent_{entropy_schedule}_lam{entropy_lambda:g}")
        if extra_tag:
            run_outdir = os.path.join(run_outdir, "_".join(extra_tag))
        os.makedirs(run_outdir, exist_ok=True)

        # Persist the full generation-hparam vector next to the outputs so
        # downstream analysis can attribute each xyz to its exact config.
        # The path-tag encoding is lossy (omits flags at default, rounds
        # floats); params.json is the authoritative record.
        import json as _json
        params_record = {
            "mode": "cond" if guidance_type else "uncond",
            "guidance_type": guidance_type,
            "rho": rho if guidance_type else None,
            "ref_path": ref_path,
            "exp_data": exp_data,
            "tmin": tmin, "tmax": tmax, "tstep": tstep,
            "t_schedule_rho": t_schedule_rho,
            "cutoff": cutoff,
            "tersoff_guidance": bool(tersoff_guidance),
            "tersoff_lambda": tersoff_lambda if tersoff_guidance else None,
            "tersoff_schedule": tersoff_schedule if tersoff_guidance else None,
            "tersoff_t_gate": tersoff_t_gate if tersoff_guidance else None,
            "tersoff_clamp": tersoff_clamp if tersoff_guidance else None,
            "tersoff_tweedie": tersoff_tweedie if tersoff_guidance else None,
            "n_corr": n_corr,
            "corr_step_size": corr_step_size if n_corr else None,
            "corr_use_tersoff": corr_use_tersoff if n_corr else None,
            "corr_t_gate": corr_t_gate if n_corr else None,
            "sa_n_steps": sa_n_steps,
            "sa_T0": sa_t0 if sa_n_steps else None,
            "sa_T_end": sa_t_end if sa_n_steps else None,
            "sa_lr": sa_lr if sa_n_steps else None,
            "sa_lr_clamp": sa_lr_clamp if sa_n_steps else None,
            "checkpoint": str(ckpt_path),
            "n_runs": n_runs,
            "n_restart": n_restart,
            "entropy_guidance": bool(entropy_guidance),
            "entropy_lambda": entropy_lambda if entropy_guidance else None,
            "entropy_schedule": entropy_schedule if entropy_guidance else None,
            "entropy_t_gate": entropy_t_gate if entropy_guidance else None,
            "entropy_r_cut": entropy_r_cut if entropy_guidance else None,
            "init_file": init_file,
            "sub_id": sub_id,
        }
        with open(os.path.join(run_outdir, "params.json"), "w") as _pf:
            _json.dump(params_record, _pf, indent=2, default=str)

        # Prior function wrapper
        def prior_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser):
            return compute_prior_score(sp, p, c, t, co, _sn, _df)
        
        for i in range(n_runs):
            import copy
            species, pos, cell = atoms_to_device(copy.deepcopy(init_atoms), device)
            cell_np = cell.detach().cpu().numpy()
            
            pos_current = pos
            traj = None
            for _restart in range(n_restart):
                last_restart = (_restart == n_restart - 1)
                traj, pos_current = denoise_by_sde(
                    species=species,
                    pos=pos_current,
                    cell=cell,
                    cutoff=cutoff,
                    score_fn=prior_fn,
                    likelihood_fn=likelihood_fn,
                    ts=ts_torch,
                    diffuser=diffuser,
                    save_traj=save_traj and last_restart,
                    progress_fn=progress_callback if (guidance_type and last_restart) else None,
                    tersoff_guidance=tersoff_guidance_fn,
                    tersoff_schedule=tersoff_schedule_fn,
                    tersoff_tweedie=tersoff_tweedie,
                    n_corr=n_corr,
                    corr_step_size=corr_step_size,
                    corr_use_tersoff=corr_use_tersoff,
                    corr_t_gate=corr_t_gate,
                    anneal_fn=anneal_fn if last_restart else None,
                    entropy_guidance=entropy_guidance_fn,
                    entropy_schedule=entropy_schedule_fn,
                )
            final_pos = pos_current
            
            if save_traj:
                traj_list = []
                for p in traj:
                    a = ase.Atoms(
                        numbers=init_atoms.numbers,
                        positions=p.numpy(),
                        cell=cell_np,
                        pbc=[True] * 3,
                    )
                    a.wrap()
                    traj_list.append(a)
                ase.io.write(f"{run_outdir}/{i:02}_traj.xyz", traj_list)
                ase.io.write(f"{run_outdir}/{i:02}_final.xyz", traj_list[-1])
            else:
                final = ase.Atoms(
                    numbers=init_atoms.numbers,
                    positions=final_pos.cpu().numpy(),
                    cell=cell_np,
                    pbc=[True] * 3,
                )
                final.wrap()
                ase.io.write(f"{run_outdir}/{i:02}_final.xyz", final)
            
            click.echo(f"  Run {i+1}/{n_runs} complete")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    click.echo(f"\nGeneration complete! Results saved to: {outdir}")
