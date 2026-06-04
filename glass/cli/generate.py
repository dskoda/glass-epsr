import os
import click
import torch
import numpy as np
from tqdm import tqdm

from glass.experiment import Experiment
from glass.lit.datamodules import StructureSpecDataModule
from glass.lit.modules import LitScoreNet, LitSpecNet
from glass.diffusion.sampling import denoise_by_sde
from glass.diffusion.profiler import GuidanceProfiler
from glass.lit.modules.likelihood import LikelihoodScore
from glass.lit.modules.guidance import create_guidance_model, load_experimental_data
from glass.lit.modules.tersoff_guidance import (
    TersoffEnergyGuidance,
    TersoffSchedule,
)
from glass.descriptors import TorchACSF, EntropyGuidance, EntropySchedule
from glass.lit.modules.coord_guidance import (
    DifferentiableCoordinationNumber,
    CoordinationLoss,
    CoordinationGuidance,
    CoordinationSchedule,
)
from glass.diffusion.schedules import power_law_ts
from glass.diffusion.annealing import make_anneal_fn, make_nvt_md_fn, make_relax_fn
from glass.utils.atoms import (
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

  # Conditional generation with PDF guidance (averaged over all refs)
  glass generate ./my_experiment/ --inits ./inits/ \\
      --guidance-type pdf --ref-path ./reference/

  # Conditional generation from a single reference structure
  glass generate ./my_experiment/ --inits ./inits/ \\
      --guidance-type pdf --ref-structure ./reference/Si_2.5_00.xyz

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
    help="Directory of reference structures; all *.xyz files are averaged into one target.",
)
@click.option(
    "--ref-structure",
    type=click.Path(),
    default=None,
    help="Single reference structure file (.xyz) for computational guidance.",
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
# Tersoff geometry optimisation after each restart
@click.option(
    "--tersoff-relax/--no-tersoff-relax",
    default=False,
    help="Run FIRE geometry optimisation under the Tersoff potential at the "
         "end of every restart. Requires Tersoff guidance to be active. "
         "Drives the structure toward a local energy minimum before the next "
         "denoising pass. Only Si (single-species) is currently supported.",
)
@click.option(
    "--tersoff-relax-fmax",
    type=float,
    default=0.1,
    show_default=True,
    help="[relax] FIRE convergence threshold: maximum per-atom force (eV/Å).",
)
@click.option(
    "--tersoff-relax-steps",
    type=int,
    default=200,
    show_default=True,
    help="[relax] Maximum number of FIRE steps per restart.",
)
# NVT molecular-dynamics inter-restart relaxation
@click.option(
    "--nvt-md/--no-nvt-md",
    default=None,
    help="Run a short NVT Langevin MD on the Tersoff PES after the "
         "unconditional prepass and after each intermediate restart (never "
         "after the final restart). Thermalises the structure before the next "
         "denoising pass. Requires Tersoff guidance; Si single-species only.",
)
@click.option(
    "--nvt-md-temperature",
    type=float,
    default=None,
    help="[NVT-MD] Target temperature in Kelvin.",
)
@click.option(
    "--nvt-md-n-steps",
    type=int,
    default=None,
    help="[NVT-MD] Number of MD steps (1000 steps × 1 fs = 1 ps).",
)
@click.option(
    "--nvt-md-timestep",
    type=float,
    default=None,
    help="[NVT-MD] Integration timestep in fs.",
)
@click.option(
    "--nvt-md-friction",
    type=float,
    default=None,
    help="[NVT-MD] Langevin friction in 1/fs.",
)
@click.option(
    "--nvt-md-pre-relax-steps",
    type=int,
    default=None,
    help="[NVT-MD] FIRE geometry-optimisation steps run before MD to drain "
         "close-contact forces. 0 disables both declash and pre-relax.",
)
@click.option(
    "--nvt-md-declash-d-min",
    type=float,
    default=None,
    help="[NVT-MD] Minimum pair distance (Å) enforced before the pre-relax. "
         "Near-coincident denoised atoms overflow Tersoff to NaN; declashing "
         "separates them so forces are finite.",
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
# Differentiable coordination-number guidance
@click.option(
    "--coord-guidance/--no-coord-guidance",
    default=None,
    help="Add differentiable coordination-number gradient as an auxiliary score term.",
)
@click.option(
    "--coord-lambda",
    type=float,
    default=None,
    help="[coord] Weight lambda_0 for the coord guidance schedule.",
)
@click.option(
    "--coord-schedule",
    type=click.Choice(["constant", "linear", "sigmoid"]),
    default=None,
    help="[coord] Shape of lambda(t).",
)
@click.option(
    "--coord-t-gate",
    type=float,
    default=None,
    help="[coord] Gate time fraction for the sigmoid schedule.",
)
@click.option(
    "--coord-r-cut",
    type=float,
    default=None,
    help="[coord] Cutoff radius for soft neighbour count (Å). If omitted and --ref-path is given, auto-derived from the first minimum of the reference PDF.",
)
@click.option(
    "--coord-smear",
    type=float,
    default=None,
    help="[coord] Half-width of the cosine switching function (Å).",
)
@click.option(
    "--coord-clamp",
    type=float,
    default=None,
    help="[coord] Per-atom guidance-norm clamp.",
)
@click.option(
    "--coord-n-target",
    type=float,
    default=None,
    help="[coord] Target coordination number.",
)
@click.option(
    "--coord-sigma-target",
    type=float,
    default=None,
    help="[coord] Tolerance around target (pseudo-Huber width).",
)
@click.option(
    "--coord-w-target",
    type=float,
    default=None,
    help="[coord] Weight on the target-match penalty (0 disables).",
)
@click.option(
    "--coord-n-low",
    type=float,
    default=None,
    help="[coord] Low-coord threshold; atoms with c < n_low are penalised.",
)
@click.option(
    "--coord-w-low",
    type=float,
    default=None,
    help="[coord] Weight on the low-coord softplus hinge (0 disables).",
)
@click.option(
    "--coord-k-low",
    type=float,
    default=None,
    help="[coord] Sharpness of the low-coord hinge.",
)
@click.option(
    "--coord-n-high",
    type=float,
    default=None,
    help="[coord] High-coord threshold; atoms with c > n_high are penalised.",
)
@click.option(
    "--coord-w-high",
    type=float,
    default=None,
    help="[coord] Weight on the high-coord softplus hinge (0 disables).",
)
@click.option(
    "--coord-k-high",
    type=float,
    default=None,
    help="[coord] Sharpness of the high-coord hinge.",
)
@click.option(
    "--n-restart",
    type=int,
    default=None,
    show_default=True,
    help="Number of full denoising passes per structure. 1 = single pass (default). "
         "Each restart starts from the previous pass output (same cell/species/guidance).",
)
@click.option(
    "--uncond-prepass/--no-uncond-prepass",
    default=None,
    help="Run one unconditional denoising pass on the init structure before "
         "the conditional passes. Only applies when guidance_type is set. "
         "Default: enabled.",
)
@click.option(
    "--lambda-prior",
    type=float,
    default=None,
    help="Scale factor applied to the prior score network output at every SDE step. "
         "1.0 (default) = unmodified prior. 0.0 turns off the prior entirely, "
         "leaving only guidance terms. Values > 1 amplify the prior.",
)
@click.option(
    "--params",
    "params_file",
    type=click.Path(exists=True),
    default=None,
    help="YAML file with generation hyperparameters to override experiment config. "
         "Only keys present in the file are applied; explicit CLI flags take precedence.",
)
@click.option(
    "--profile-guidance/--no-profile-guidance",
    default=False,
    help="Record per-step guidance contribution norms and save to "
         "<run_outdir>/guidance_profile_run{N}_restart{R}.json.",
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
    ref_structure,
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
    coord_guidance,
    coord_lambda,
    coord_schedule,
    coord_t_gate,
    coord_r_cut,
    coord_smear,
    coord_clamp,
    coord_n_target,
    coord_sigma_target,
    coord_w_target,
    coord_n_low,
    coord_w_low,
    coord_k_low,
    coord_n_high,
    coord_w_high,
    coord_k_high,
    n_restart,
    uncond_prepass,
    lambda_prior,
    params_file,
    profile_guidance,
    tersoff_relax,
    tersoff_relax_fmax,
    tersoff_relax_steps,
    nvt_md,
    nvt_md_temperature,
    nvt_md_n_steps,
    nvt_md_timestep,
    nvt_md_friction,
    nvt_md_pre_relax_steps,
    nvt_md_declash_d_min,
):
    """Generate structures using trained score model."""
    import ase
    from ase.io import read

    # Set CUDA memory config
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load experiment.  If the experiment has no config.yaml, fall back to
    # the built-in ExperimentConfig defaults (mirrors glass/default_params.yaml)
    # so that `glass generate` works on any experiment dir that just contains
    # checkpoints/ and inits/.
    experiment = Experiment(experiment_path)
    if experiment.config_path.exists():
        config = experiment.load_config()
    else:
        from glass.experiment import ExperimentConfig
        click.echo(
            f"No config.yaml in {experiment.root}; using built-in defaults."
        )
        config = ExperimentConfig()

    # Apply params-file overrides (between experiment config and explicit CLI flags)
    if params_file:
        config.update_from_yaml(params_file)
        click.echo(f"Applied params overrides from {params_file}")

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
    # NVT-MD inter-restart relaxation
    nvt_md = nvt_md if nvt_md is not None else getattr(config, "nvt_md", False)
    nvt_md_temperature = (
        nvt_md_temperature if nvt_md_temperature is not None
        else getattr(config, "nvt_md_temperature", 600.0)
    )
    nvt_md_n_steps = (
        nvt_md_n_steps if nvt_md_n_steps is not None
        else getattr(config, "nvt_md_n_steps", 1000)
    )
    nvt_md_timestep = (
        nvt_md_timestep if nvt_md_timestep is not None
        else getattr(config, "nvt_md_timestep", 1.0)
    )
    nvt_md_friction = (
        nvt_md_friction if nvt_md_friction is not None
        else getattr(config, "nvt_md_friction", 0.01)
    )
    nvt_md_pre_relax_steps = (
        nvt_md_pre_relax_steps if nvt_md_pre_relax_steps is not None
        else getattr(config, "nvt_md_pre_relax_steps", 10)
    )
    nvt_md_declash_d_min = (
        nvt_md_declash_d_min if nvt_md_declash_d_min is not None
        else getattr(config, "nvt_md_declash_d_min", 1.5)
    )
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
    # Coordination-number guidance
    coord_guidance = (
        coord_guidance if coord_guidance is not None
        else getattr(config, "coord_guidance", False)
    )
    coord_lambda = (
        coord_lambda if coord_lambda is not None
        else getattr(config, "coord_lambda", 1.0)
    )
    coord_schedule = coord_schedule or getattr(config, "coord_schedule", "constant")
    coord_t_gate = (
        coord_t_gate if coord_t_gate is not None
        else getattr(config, "coord_t_gate", 1.0)
    )
    coord_r_cut = (
        coord_r_cut if coord_r_cut is not None
        else getattr(config, "coord_r_cut", None)
    )
    coord_smear = (
        coord_smear if coord_smear is not None
        else getattr(config, "coord_smear", 0.30)
    )
    coord_clamp = (
        coord_clamp if coord_clamp is not None
        else getattr(config, "coord_clamp", 10.0)
    )
    coord_n_target = (
        coord_n_target if coord_n_target is not None
        else getattr(config, "coord_n_target", 4.0)
    )
    coord_sigma_target = (
        coord_sigma_target if coord_sigma_target is not None
        else getattr(config, "coord_sigma_target", 0.5)
    )
    coord_w_target = (
        coord_w_target if coord_w_target is not None
        else getattr(config, "coord_w_target", 1.0)
    )
    coord_n_low = (
        coord_n_low if coord_n_low is not None
        else getattr(config, "coord_n_low", 4.0)
    )
    coord_w_low = (
        coord_w_low if coord_w_low is not None
        else getattr(config, "coord_w_low", 0.0)
    )
    coord_k_low = (
        coord_k_low if coord_k_low is not None
        else getattr(config, "coord_k_low", 4.0)
    )
    coord_n_high = (
        coord_n_high if coord_n_high is not None
        else getattr(config, "coord_n_high", 7.0)
    )
    coord_w_high = (
        coord_w_high if coord_w_high is not None
        else getattr(config, "coord_w_high", 0.0)
    )
    coord_k_high = (
        coord_k_high if coord_k_high is not None
        else getattr(config, "coord_k_high", 4.0)
    )
    n_restart = n_restart if n_restart is not None else getattr(config, "n_restart", 1)
    n_restart = max(1, int(n_restart))
    uncond_prepass = (
        uncond_prepass if uncond_prepass is not None
        else getattr(config, "uncond_prepass", True)
    )
    lambda_prior = (
        lambda_prior if lambda_prior is not None
        else getattr(config, "lambda_prior", 1.0)
    )

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

    coord_guidance_fn = None
    coord_schedule_fn = None
    if coord_guidance:
        if coord_r_cut is None:
            if not ref_path:
                raise click.ClickException(
                    "--coord-r-cut not specified and could not be auto-derived: "
                    "either pass --coord-r-cut, set coord_r_cut in config.yaml, "
                    "or provide --ref-path so the cutoff can be taken from the "
                    "reference PDF first minimum."
                )
            import glob as _glob
            from glass.metrics.structural import compute_pdf
            cut_files = sorted(_glob.glob(os.path.join(ref_path, "*.xyz")))
            if not cut_files:
                raise click.ClickException(
                    f"--coord-r-cut auto-derivation failed: no .xyz files in {ref_path}"
                )
            mins = []
            for rf in cut_files:
                pdf = compute_pdf(read(rf, "-1"))
                if pdf.first_minima_position is not None:
                    mins.append(pdf.first_minima_position)
            if not mins:
                raise click.ClickException(
                    "--coord-r-cut auto-derivation failed: no first PDF minimum "
                    f"detected in any reference structure under {ref_path}."
                )
            coord_r_cut = float(np.mean(mins))
            click.echo(
                f"Auto-derived coord_r_cut from {len(mins)} reference PDFs: "
                f"{coord_r_cut:.3f} Å"
            )
        click.echo(
            f"Coord guidance: schedule={coord_schedule} "
            f"lambda0={coord_lambda} t_gate={coord_t_gate} "
            f"r_cut={coord_r_cut} smear={coord_smear} "
            f"low(n={coord_n_low},w={coord_w_low},k={coord_k_low}) "
            f"target(n={coord_n_target},sigma={coord_sigma_target},w={coord_w_target}) "
            f"high(n={coord_n_high},w={coord_w_high},k={coord_k_high})"
        )
        coord_guidance_fn = CoordinationGuidance(
            coord_fn=DifferentiableCoordinationNumber(
                r_cut=coord_r_cut, smear=coord_smear,
            ),
            loss_fn=CoordinationLoss(
                n_target=coord_n_target,
                sigma_target=coord_sigma_target,
                w_target=coord_w_target,
                n_low=coord_n_low,
                w_low=coord_w_low,
                k_low=coord_k_low,
                n_high=coord_n_high,
                w_high=coord_w_high,
                k_high=coord_k_high,
            ),
            clamp_norm=coord_clamp,
        )
        coord_schedule_fn = CoordinationSchedule(
            schedule=coord_schedule,
            lambda_0=coord_lambda,
            tmax=tmax,
            t_gate=coord_t_gate,
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

    if tersoff_relax:
        if not tersoff_guidance:
            raise click.ClickException(
                "--tersoff-relax requires --tersoff-guidance to be active."
            )
        click.echo(
            f"Tersoff relax: fmax={tersoff_relax_fmax} eV/Å  max_steps={tersoff_relax_steps}"
        )

    if nvt_md:
        if not tersoff_guidance:
            raise click.ClickException(
                "--nvt-md requires --tersoff-guidance to be active."
            )
        click.echo(
            f"NVT-MD: T={nvt_md_temperature} K  n_steps={nvt_md_n_steps} "
            f"timestep={nvt_md_timestep} fs  friction={nvt_md_friction} 1/fs "
            f"pre_relax={nvt_md_pre_relax_steps} declash_d_min={nvt_md_declash_d_min} Å "
            f"(after prepass + intermediate restarts only)"
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
        
        # Validate reference source options
        n_ref_sources = sum(bool(x) for x in [ref_path, ref_structure, exp_data])
        if n_ref_sources == 0:
            raise click.ClickException(
                "Conditional generation requires --ref-path, --ref-structure, or --exp-data"
            )
        if n_ref_sources > 1:
            raise click.ClickException(
                "--ref-path, --ref-structure, and --exp-data are mutually exclusive"
            )

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

    # Pre-compute averaged target from all structures in --ref-path
    ref_path_target_y = None
    if ref_path and guidance_type:
        import glob as _glob
        ref_files = sorted(_glob.glob(os.path.join(ref_path, "*.xyz")))
        if not ref_files:
            raise click.ClickException(f"No .xyz files found in --ref-path {ref_path}")
        click.echo(f"Computing averaged target from {len(ref_files)} reference structures in {ref_path}")
        targets = []
        for rf in ref_files:
            ref_atoms = read(rf, "-1")
            targets.append(
                compute_target_from_reference(ref_atoms, guidance_model, guidance_type, cutoff, device)
            )
        ref_path_target_y = torch.stack(targets).mean(dim=0)

    # Pre-compute target from single --ref-structure file
    ref_structure_target_y = None
    if ref_structure and guidance_type:
        if not os.path.exists(ref_structure):
            raise click.ClickException(f"--ref-structure file not found: {ref_structure}")
        click.echo(f"Computing target from reference structure: {ref_structure}")
        ref_atoms = read(ref_structure, "-1")
        ref_structure_target_y = compute_target_from_reference(
            ref_atoms, guidance_model, guidance_type, cutoff, device
        )

    # Get initial structures
    init_files = experiment.get_init_files(inits)
    if not init_files:
        raise click.ClickException(f"No .xyz files found in {inits}")
    click.echo(f"Found {len(init_files)} initial structures")

    # Time steps (power-law; rho=1.0 == linspace)
    ts_torch = power_law_ts(tmin, tmax, tstep, rho=t_schedule_rho, device=device)

    # Process each initial structure
    for init_file in init_files:
        init_atoms = read(init_file, "-1")
        sub_id = os.path.basename(init_file).replace(".xyz", "")
        click.echo(f"\nProcessing: {sub_id}")

        relax_fn = (
            make_relax_fn(
                numbers=init_atoms.numbers,
                fmax=tersoff_relax_fmax,
                max_steps=tersoff_relax_steps,
            )
            if tersoff_relax
            else None
        )

        nvt_md_fn = (
            make_nvt_md_fn(
                numbers=init_atoms.numbers,
                temperature=nvt_md_temperature,
                n_steps=nvt_md_n_steps,
                timestep=nvt_md_timestep,
                friction=nvt_md_friction,
                pre_relax_steps=nvt_md_pre_relax_steps,
                declash_d_min=nvt_md_declash_d_min,
                device=str(device),
            )
            if nvt_md
            else None
        )

        # Select pre-computed target for this structure
        target_y = None
        if ref_path_target_y is not None:
            target_y = ref_path_target_y
        elif ref_structure_target_y is not None:
            target_y = ref_structure_target_y
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
        if coord_guidance:
            extra_tag.append(f"coord_{coord_schedule}_lam{coord_lambda:g}")
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
            "ref_structure": ref_structure,
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
            "nvt_md": bool(nvt_md),
            "nvt_md_temperature": nvt_md_temperature if nvt_md else None,
            "nvt_md_n_steps": nvt_md_n_steps if nvt_md else None,
            "nvt_md_timestep": nvt_md_timestep if nvt_md else None,
            "nvt_md_friction": nvt_md_friction if nvt_md else None,
            "checkpoint": str(ckpt_path),
            "n_runs": n_runs,
            "n_restart": n_restart,
            "uncond_prepass": bool(uncond_prepass) if guidance_type else None,
            "entropy_guidance": bool(entropy_guidance),
            "entropy_lambda": entropy_lambda if entropy_guidance else None,
            "entropy_schedule": entropy_schedule if entropy_guidance else None,
            "entropy_t_gate": entropy_t_gate if entropy_guidance else None,
            "entropy_r_cut": entropy_r_cut if entropy_guidance else None,
            "coord_guidance": bool(coord_guidance),
            "coord_lambda": coord_lambda if coord_guidance else None,
            "coord_schedule": coord_schedule if coord_guidance else None,
            "coord_t_gate": coord_t_gate if coord_guidance else None,
            "coord_r_cut": coord_r_cut if coord_guidance else None,
            "coord_smear": coord_smear if coord_guidance else None,
            "coord_clamp": coord_clamp if coord_guidance else None,
            "coord_n_target": coord_n_target if coord_guidance else None,
            "coord_sigma_target": coord_sigma_target if coord_guidance else None,
            "coord_w_target": coord_w_target if coord_guidance else None,
            "coord_n_low": coord_n_low if coord_guidance else None,
            "coord_w_low": coord_w_low if coord_guidance else None,
            "coord_k_low": coord_k_low if coord_guidance else None,
            "coord_n_high": coord_n_high if coord_guidance else None,
            "coord_w_high": coord_w_high if coord_guidance else None,
            "coord_k_high": coord_k_high if coord_guidance else None,
            "lambda_prior": lambda_prior,
            "init_file": init_file,
            "sub_id": sub_id,
        }
        with open(os.path.join(run_outdir, "params.json"), "w") as _pf:
            _json.dump(params_record, _pf, indent=2, default=str)

        # Prior function wrapper
        _lp = float(lambda_prior)
        def prior_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser, _lp=_lp):
            return _lp * compute_prior_score(sp, p, c, t, co, _sn, _df)
        
        for i in range(n_runs):
            import copy
            species, pos, cell = atoms_to_device(copy.deepcopy(init_atoms), device)
            cell_np = cell.detach().cpu().numpy()

            pos_current = pos
            traj = None

            def run_nvt_md(p, label):
                """Run the NVT-MD thermalisation with a tqdm bar like denoising."""
                md_pbar = tqdm(
                    total=nvt_md_n_steps,
                    desc=f"  run {i+1}/{n_runs} {label}",
                    unit="step",
                    leave=False,
                    dynamic_ncols=True,
                )

                def md_progress(step, T, _pbar=md_pbar):
                    _pbar.n = min(step, nvt_md_n_steps)
                    _pbar.set_postfix(T=f"{T:.0f}K")
                    _pbar.refresh()

                out = nvt_md_fn(p, cell, species, progress_fn=md_progress).to(p.device)
                md_pbar.n = nvt_md_n_steps
                md_pbar.close()
                return out

            # Optional unconditional prepass: a single full denoising pass with
            # no likelihood guidance to give the conditional restarts a more
            # physical starting point than the raw init.
            if guidance_type and uncond_prepass:
                pre_pbar = tqdm(
                    total=len(ts_torch),
                    desc=f"  run {i+1}/{n_runs} uncond-prepass ",
                    unit="step",
                    leave=False,
                    dynamic_ncols=True,
                )

                def pre_progress(step, t, p_norm, _pbar=pre_pbar, **_kw):
                    _pbar.update(1)
                    _pbar.set_postfix(p=f"{p_norm:.3f}")

                _, pos_current = denoise_by_sde(
                    species=species,
                    pos=pos_current,
                    cell=cell,
                    cutoff=cutoff,
                    score_fn=prior_fn,
                    likelihood_fn=None,
                    ts=ts_torch,
                    diffuser=diffuser,
                    save_traj=False,
                    progress_fn=pre_progress,
                    tersoff_guidance=tersoff_guidance_fn,
                    tersoff_schedule=tersoff_schedule_fn,
                    tersoff_tweedie=tersoff_tweedie,
                    n_corr=n_corr,
                    corr_step_size=corr_step_size,
                    corr_use_tersoff=corr_use_tersoff,
                    corr_t_gate=corr_t_gate,
                    anneal_fn=None,
                    entropy_guidance=entropy_guidance_fn,
                    entropy_schedule=entropy_schedule_fn,
                    coord_guidance=coord_guidance_fn,
                    coord_schedule=coord_schedule_fn,
                )
                pre_pbar.close()

                # Thermalise the prepass output before the conditional restarts.
                if nvt_md_fn is not None:
                    pos_current = run_nvt_md(pos_current, "nvt-md (post-prepass) ")

            for _restart in range(n_restart):
                last_restart = (_restart == n_restart - 1)
                total_steps = len(ts_torch)
                restart_label = f"restart {_restart+1}/{n_restart} " if n_restart > 1 else ""
                pbar = tqdm(
                    total=total_steps,
                    desc=f"  run {i+1}/{n_runs} {restart_label}",
                    unit="step",
                    leave=False,
                    dynamic_ncols=True,
                )

                def progress_callback(step, t, p_norm, l_norm=None, target_norm=None, _pbar=pbar, **_kw):
                    _pbar.update(1)
                    if l_norm is not None:
                        _pbar.set_postfix(p=f"{p_norm:.3f}", l=f"{l_norm:.3f}", tgt=f"{target_norm:.4f}")
                    else:
                        _pbar.set_postfix(p=f"{p_norm:.3f}")

                run_profiler = GuidanceProfiler() if profile_guidance else None

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
                    progress_fn=progress_callback,
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
                    coord_guidance=coord_guidance_fn,
                    coord_schedule=coord_schedule_fn,
                    profiler=run_profiler,
                )
                pbar.close()

                if relax_fn is not None:
                    pos_current = relax_fn(pos_current, cell, species).to(pos_current.device)

                # NVT MD thermalisation between passes — skip after the final
                # restart so the returned structure is the denoised result.
                if nvt_md_fn is not None and not last_restart:
                    pos_current = run_nvt_md(
                        pos_current, f"nvt-md (post-restart {_restart+1}/{n_restart}) "
                    )

                if run_profiler is not None:
                    profile_path = os.path.join(
                        run_outdir, f"{i:02}_guidance_profile_restart{_restart}.json"
                    )
                    run_profiler.save_json(profile_path)

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
