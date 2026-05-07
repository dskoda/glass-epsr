import glob
import os
import sys
import warnings

import click
import lightning as L
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from glass.lit.datamodules import StructureSpecDataModule
from glass.lit.modules import LitScoreNet


@click.group()
def glass():
    """CLI for training and running the glass uncond_denoiser workflows."""
    pass


@glass.command(
    "train_score",
    help="""
Train (or resume) a score-based generative model for atomic structures.

DATA:
  SAMPLE_TAG determines the dataset location:
    ./data/<sample_tag>/

CHECKPOINTS:
  Saved under:
    ./models/<sample_tag>/

EXAMPLES:

  # Train from scratch (recommended defaults)
  glass train_score Si_1.5_2.5_3.5 --num-species 2 --data-root /home/jwguo/03_denoiser/demo_Si/data

  # Resume from latest checkpoint
  glass train_score Si_1.5_2.5_3.5 --num-species 2 --data-root /home/jwguo/03_denoiser/demo_Si/data --resume

  # Full control example
  glass train_score Si_1.5_2.5_3.5 \\
      --num-species 2 \\
      --data-root /home/jwguo/03_denoiser/demo_Si/data \\
      --max-epochs 12000 \\
      --cutoff 5 \\
      --k 0.8 \\
      --dup 4 \\
      --n-conv 5 \\
      --dim 200 \\
      --ema-decay 0.9999 \\
      --lr 0.001 \\
      --resume

KEY PARAMETERS:

  --k              Noise level (higher = harder denoising task)
  --cutoff         Graph cutoff radius (Å)
  --n-conv, --dim  Model capacity
  --dup            Useful for small datasets (data repetition)
  --ema-decay      Stabilizes sampling (keep ~0.999–0.9999)

""",
)
@click.argument("sample_tag", type=str)
@click.option(
    "--num-species", type=int, required=True, help="Number of atomic species."
)
@click.option(
    "--new-model/--resume",
    default=True,
    help="Start a new model or resume from the latest checkpoint.",
)
@click.option(
    "--cutoff",
    type=float,
    default=5.0,
    show_default=True,
    help="Cutoff radius for atomic graph construction.",
)
@click.option(
    "--k",
    type=float,
    default=0.8,
    show_default=True,
    help="Maximum noise level applied in training.",
)
@click.option(
    "--dup",
    type=int,
    default=4,
    show_default=True,
    help="Dataset duplication factor for longer epochs.",
)
@click.option(
    "--train-size",
    type=float,
    default=0.9,
    show_default=True,
    help="Train/validation split fraction.",
)
@click.option(
    "--scale-y",
    type=float,
    default=1.0,
    show_default=True,
    help="Scale factor for spectroscopy curve.",
)
@click.option(
    "--batch-size", type=int, default=1, show_default=True, help="Batch size."
)
@click.option(
    "--num-workers",
    type=int,
    default=8,
    show_default=True,
    help="Number of dataloader workers.",
)
@click.option(
    "--n-conv",
    type=int,
    default=5,
    show_default=True,
    help="Number of convolution layers.",
)
@click.option(
    "--dim", type=int, default=200, show_default=True, help="Hidden dimension."
)
@click.option(
    "--ema-decay", type=float, default=0.9999, show_default=True, help="EMA decay."
)
@click.option(
    "--lr",
    "--learn-rate",
    type=float,
    default=1e-3,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--max-epochs",
    type=int,
    default=12000,
    show_default=True,
    help="Maximum number of training epochs.",
)
@click.option(
    "--accelerator",
    type=str,
    default="gpu",
    show_default=True,
    help="Lightning accelerator.",
)
@click.option(
    "--strategy",
    type=str,
    default="ddp_find_unused_parameters_true",
    show_default=True,
    help="Lightning distributed training strategy.",
)
@click.option(
    "--refresh-rate",
    type=int,
    default=10,
    show_default=True,
    help="Progress bar refresh rate.",
)
@click.option(
    "--save-dir",
    type=str,
    default="./models/",
    show_default=True,
    help="Directory for logs/checkpoints.",
)
@click.option(
    "--data-root",
    type=str,
    default="./data",
    show_default=True,
    help="Root directory containing sample data folders.",
)
@click.option(
    "--matmul-precision",
    type=click.Choice(["highest", "high", "medium"]),
    default="medium",
    show_default=True,
    help="Torch float32 matmul precision.",
)
def train_score(
    sample_tag,
    num_species,
    new_model,
    cutoff,
    k,
    dup,
    train_size,
    scale_y,
    batch_size,
    num_workers,
    n_conv,
    dim,
    ema_decay,
    lr,
    max_epochs,
    accelerator,
    strategy,
    refresh_rate,
    save_dir,
    data_root,
    matmul_precision,
):
    """Train a score model for a given sample tag."""

    torch.set_float32_matmul_precision(matmul_precision)

    click.echo(f"Sample tag: {sample_tag}")
    click.echo(f"Number of species: {num_species}")

    datamodule = StructureSpecDataModule(
        data_dir=f"{data_root}/{sample_tag}/",
        cutoff=cutoff,
        train_prior=True,
        k=k,
        train_size=train_size,
        scale_y=scale_y,
        dup=dup,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if new_model:
        score_net = LitScoreNet(
            num_species=num_species,
            num_convs=n_conv,
            dim=dim,
            ema_decay=ema_decay,
            learn_rate=lr,
        )
        checkpoint = None
        click.echo("Starting a new model instance")
    else:
        checkpoints = sorted(
            glob.glob(f"{save_dir}/{sample_tag}/version_*/checkpoints/*.ckpt")
        )
        if not checkpoints:
            raise click.ClickException(
                f"No checkpoints found under {save_dir}/{sample_tag}/version_*/checkpoints/*.ckpt"
            )
        checkpoint = checkpoints[-1]
        score_net = LitScoreNet.load_from_checkpoint(checkpoint)
        click.echo(f"Loaded model weights from {checkpoint}")

    trainer = L.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        logger=TensorBoardLogger(save_dir=save_dir, name=sample_tag),
        callbacks=[TQDMProgressBar(refresh_rate=refresh_rate)],
        strategy=strategy,
    )

    trainer.fit(score_net, datamodule, ckpt_path=checkpoint)


@glass.command(
    "train_spec",
    help="""
Train (or resume) a per-atom spectral surrogate model (EXAFS or XANES).

Data is expected at:
  DATA_ROOT/SAMPLE_TAG/structures/train/*.xyz
  DATA_ROOT/SAMPLE_TAG/{exafs,xanes}/train/*.txt  (one row per atom)

Checkpoints are saved under:
  SAVE_DIR/SAMPLE_TAG/

EXAMPLES:

  # Train EXAFS from scratch
  glass train_spec Si_exafs --spec-type exafs --num-species 1 --out-dim 400

  # Train XANES, resume from checkpoint
  glass train_spec Si_xanes --spec-type xanes --num-species 1 --out-dim 100 --resume

  # Multi-species EXAFS
  glass train_spec ZrNiAl_exafs --spec-type exafs --num-species 3 --out-dim 400 \\
      --data-root /home/jwguo/03_denoiser/demo_Si/data --max-epochs 8000
""",
)
@click.argument("sample_tag", type=str)
@click.option(
    "--spec-type",
    type=click.Choice(["exafs", "xanes"]),
    required=True,
    help="Spectral model type.",
)
@click.option(
    "--num-species", type=int, required=True, help="Number of atomic species."
)
@click.option(
    "--out-dim",
    type=int,
    default=None,
    show_default=True,
    help="Output dimension (default: 400 for exafs, 100 for xanes).",
)
@click.option(
    "--new-model/--resume",
    default=True,
    help="Start a new model or resume from the latest checkpoint.",
)
@click.option(
    "--cutoff",
    type=float,
    default=5.0,
    show_default=True,
    help="Graph cutoff radius (Å).",
)
@click.option(
    "--k", type=float, default=0.8, show_default=True, help="Maximum noise level."
)
@click.option(
    "--dup",
    type=int,
    default=128,
    show_default=True,
    help="Dataset duplication factor.",
)
@click.option(
    "--train-size",
    type=float,
    default=0.9,
    show_default=True,
    help="Train/validation split fraction.",
)
@click.option(
    "--batch-size", type=int, default=32, show_default=True, help="Batch size."
)
@click.option(
    "--num-workers",
    type=int,
    default=8,
    show_default=True,
    help="Number of dataloader workers.",
)
@click.option(
    "--n-conv",
    type=int,
    default=5,
    show_default=True,
    help="Number of convolution layers.",
)
@click.option(
    "--dim", type=int, default=200, show_default=True, help="Hidden dimension."
)
@click.option(
    "--ema-decay", type=float, default=0.9999, show_default=True, help="EMA decay."
)
@click.option(
    "--lr",
    "--learn-rate",
    type=float,
    default=1e-3,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--max-epochs",
    type=int,
    default=None,
    show_default=True,
    help="Max training epochs (default: 8000 for exafs, 3000 for xanes).",
)
@click.option(
    "--accelerator",
    type=str,
    default="gpu",
    show_default=True,
    help="Lightning accelerator.",
)
@click.option(
    "--strategy",
    type=str,
    default="ddp_find_unused_parameters_true",
    show_default=True,
    help="Lightning distributed strategy.",
)
@click.option(
    "--refresh-rate",
    type=int,
    default=10,
    show_default=True,
    help="Progress bar refresh rate.",
)
@click.option(
    "--save-dir",
    type=str,
    default="./models/",
    show_default=True,
    help="Directory for logs/checkpoints.",
)
@click.option(
    "--data-root",
    type=str,
    default="./data",
    show_default=True,
    help="Root directory containing sample data folders.",
)
@click.option(
    "--matmul-precision",
    type=click.Choice(["highest", "high", "medium"]),
    default="medium",
    show_default=True,
    help="Torch matmul precision.",
)
def train_spec(
    sample_tag,
    spec_type,
    num_species,
    out_dim,
    new_model,
    cutoff,
    k,
    dup,
    train_size,
    batch_size,
    num_workers,
    n_conv,
    dim,
    ema_decay,
    lr,
    max_epochs,
    accelerator,
    strategy,
    refresh_rate,
    save_dir,
    data_root,
    matmul_precision,
):
    """Train a per-atom EXAFS or XANES surrogate model."""
    from glass.lit.modules import LitSpecNet

    torch.set_float32_matmul_precision(matmul_precision)

    # apply type-specific defaults
    if out_dim is None:
        out_dim = 400 if spec_type == "exafs" else 100
    if max_epochs is None:
        max_epochs = 8000 if spec_type == "exafs" else 3000

    click.echo(f"Sample tag:  {sample_tag}")
    click.echo(
        f"Spec type:   {spec_type}  |  out_dim={out_dim}  |  max_epochs={max_epochs}"
    )
    click.echo(f"Num species: {num_species}")

    data_dir = os.path.join(data_root, sample_tag)
    xyz_found = (
        glob.glob(os.path.join(data_dir, "structures", "train", "*.xyz"))
        or glob.glob(os.path.join(data_dir, "structures", "*.xyz"))
        or glob.glob(os.path.join(data_dir, "*.xyz"))
    )
    if not xyz_found:
        raise click.ClickException(
            f"No .xyz files found under {data_dir}\n"
            f"Expected layout: {data_dir}/structures/train/*.xyz\n"
            f"                 {data_dir}/{spec_type}/train/*.txt  (one row per atom)"
        )
    click.echo(f"Data dir:    {data_dir}  ({len(xyz_found)} structures found)")
    data_dir = data_dir + "/"  # StructureSpecDataModule expects trailing slash

    datamodule = StructureSpecDataModule(
        data_dir=data_dir,
        cutoff=cutoff,
        train_prior=True,
        k=k,
        train_size=train_size,
        scale_y=1.0,
        dup=dup,
        batch_size=batch_size,
        num_workers=num_workers,
        guide_type=spec_type,
    )

    if new_model:
        spec_net = LitSpecNet(
            num_species=num_species,
            num_convs=n_conv,
            dim=dim,
            out_dim=out_dim,
            ema_decay=ema_decay,
            learn_rate=lr,
        )
        checkpoint = None
        click.echo("Starting a new model instance")
    else:
        checkpoints = sorted(
            glob.glob(
                f"{save_dir}/{sample_tag}_{spec_type}/version_*/checkpoints/*.ckpt"
            )
        )
        if not checkpoints:
            raise click.ClickException(
                f"No checkpoints found under {save_dir}/{sample_tag}_{spec_type}/version_*/checkpoints/*.ckpt"
            )
        checkpoint = checkpoints[-1]
        spec_net = LitSpecNet.load_from_checkpoint(checkpoint)
        click.echo(f"Loaded model weights from {checkpoint}")

    trainer = L.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        logger=TensorBoardLogger(save_dir=save_dir, name=f"{sample_tag}_{spec_type}"),
        callbacks=[TQDMProgressBar(refresh_rate=refresh_rate)],
        strategy=strategy,
    )

    trainer.fit(spec_net, datamodule, ckpt_path=checkpoint)


@glass.command(
    "plot_loss",
    help="""
Plot training loss curves from TensorBoard event files.

MODELS_DIR is the directory containing model subdirectories (each with version*/events* files).
Defaults to ./models/.

EXAMPLES:

  # Plot all models in ./models/
  glass plot_loss

  # Plot models in a specific directory
  glass plot_loss /home/jwguo/03_denoiser/demo_Si/models

  # Custom output file and y-axis range
  glass plot_loss --output my_plot.pdf --ylim 0.1 2.0

  # Only plot specific models
  glass plot_loss --model Si_1.5_2.5_3.5 --model Si_2.0_3.0

""",
)
@click.argument("models_dir", type=click.Path(exists=True), default="./models/")
@click.option(
    "--model",
    "model_filter",
    multiple=True,
    help="Plot only these model(s). Can be repeated. Default: all.",
)
@click.option(
    "--output",
    type=str,
    default="score_LC_all.pdf",
    show_default=True,
    help="Output PDF filename.",
)
@click.option(
    "--ylim",
    type=(float, float),
    default=(0.2, 3.2),
    show_default=True,
    help="Y-axis limits: MIN MAX.",
)
@click.option(
    "--step",
    type=int,
    default=20,
    show_default=True,
    help="Downsample factor for plotted points.",
)
@click.option(
    "--figsize",
    type=(float, float),
    default=(10.0, 6.0),
    show_default=True,
    help="Figure size: W H.",
)
def plot_loss(models_dir, model_filter, output, ylim, step, figsize):
    """Plot loss curves from TensorBoard logs in MODELS_DIR."""
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    import seaborn as sns
    from tensorboard.backend.event_processing import event_accumulator

    warnings.filterwarnings("ignore")

    params = {
        "legend.fontsize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
        "axes.linewidth": 1.5,
    }
    pylab.rcParams.update(params)
    sns.set_context("talk")

    def get_lc(model_path):
        events = sorted(glob.glob(os.path.join(model_path, "version*", "events*")))
        epoch, loss = [], []
        for event_file in events:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            try:
                epoch += [int(x.value) for x in ea.Scalars("epoch")]
                loss += [x.value for x in ea.Scalars("train_loss")]
            except KeyError:
                pass
        return {"epoch": epoch, "loss": loss}

    all_models = sorted(
        f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))
    )
    models = (
        [m for m in all_models if m in model_filter] if model_filter else all_models
    )

    if not models:
        raise click.ClickException(f"No model directories found in {models_dir}")

    click.echo(f"Found models: {models}")

    colors = sns.color_palette("deep")
    plt.figure(figsize=figsize)
    for i, model in enumerate(models):
        data = get_lc(os.path.join(models_dir, model))
        if not data["epoch"]:
            click.echo(f"  Warning: no loss data found for {model}, skipping.")
            continue
        plt.plot(
            data["epoch"][::step],
            data["loss"][::step],
            ".",
            label=model,
            color=colors[i % len(colors)],
        )

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    click.echo(f"Saved plot to {output}")


@glass.command(
    "uncond_denoise",
    help="""
Run unconditional denoising (GPU-accelerated) for one or more score models.

Loops over each SCORE_MODEL, finds matching .xyz files under ATOMS_PATH,
and runs N_RUNS independent denoising trajectories per structure.

Checkpoints are expected at: CKPT_DIR/{system}_score_{model}.ckpt
Data modules are loaded from:  SCORE_PATH/data/{system}_{model}/

EXAMPLES:

  # Single model, single file
  glass uncond_denoise --score-model 1.5_2.5_3.5 --system Si \\
      --score-data-path /home/jwguo/03_denoiser/00_score_models \\
      --score-model-path /home/jwguo/03_denoiser/demo_Si/Si_score_1.5_2.5_3.5.ckpt \\
      --init-path /home/jwguo/03_denoiser/reference/start_Si_216/Si_01.xyz

  # Directory input (loads all Si_*.xyz inside)
  glass uncond_denoise --score-model 1.5_2.5_3.5 --system Si \\
      --score-data-path /home/jwguo/03_denoiser/00_score_models \\
      --score-model-path /home/jwguo/03_denoiser/demo_Si/Si_score_1.5_2.5_3.5.ckpt \\
      --init-path /home/jwguo/03_denoiser/reference/start_Si_216

  # Glob pattern input
  glass uncond_denoise --score-model 1.5_2.5_3.5 --system Si \\
      --score-data-path /home/jwguo/03_denoiser/00_score_models \\
      --score-model-path /home/jwguo/03_denoiser/demo_Si/Si_score_1.5_2.5_3.5.ckpt \\
      --init-path "/home/jwguo/03_denoiser/reference/start_Si_216/Si_0[1-3].xyz"
""",
)
@click.option(
    "--score-model",
    "score_models",
    multiple=True,
    required=True,
    help="Score model tag(s). Can be repeated.",
)
@click.option(
    "--system",
    type=str,
    default="Si",
    show_default=True,
    help="System name (e.g. Si, MoS2).",
)
@click.option(
    "--score-data-path",
    type=str,
    required=True,
    help="Root dir containing data/{system}_{model}/ subfolders.",
)
@click.option(
    "--score-model-path",
    type=str,
    required=True,
    help="Full path to the .ckpt checkpoint file.",
)
@click.option(
    "--init-path",
    type=str,
    required=True,
    help="Initial structures: a .xyz file, a glob pattern, or a directory (all {system}_*.xyz inside).",
)
@click.option(
    "--outdir",
    type=str,
    default="denoise_logs/unconditional",
    show_default=True,
    help="Output root directory.",
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    show_default=True,
    help="Torch device (e.g. cuda:0, cuda:1, cpu).",
)
@click.option(
    "--tmin",
    type=float,
    default=0.001,
    show_default=True,
    help="Start time for reverse SDE.",
)
@click.option(
    "--tmax",
    type=float,
    default=1.0,
    show_default=True,
    help="End time for reverse SDE.",
)
@click.option(
    "--tstep",
    type=int,
    default=256,
    show_default=True,
    help="Number of SDE time steps.",
)
@click.option(
    "--cutoff",
    type=float,
    default=5.0,
    show_default=True,
    help="Graph cutoff radius (Å).",
)
@click.option(
    "--n-runs",
    type=int,
    default=10,
    show_default=True,
    help="Number of independent runs per structure.",
)
@click.option(
    "--save-traj/--no-save-traj",
    default=True,
    show_default=True,
    help="Save full trajectory or only final frame.",
)
def uncond_denoise(
    score_models,
    system,
    score_data_path,
    score_model_path,
    init_path,
    outdir,
    device,
    tmin,
    tmax,
    tstep,
    cutoff,
    n_runs,
    save_traj,
):
    """GPU unconditional denoising over one or more score models."""
    import copy
    import ase
    from ase.io import read

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")
    if torch.cuda.is_available():
        click.echo(f"GPU: {torch.cuda.get_device_name(device.index or 0)}")

    from graphite.nn import periodic_radius_graph
    from glass.lit.functions.get_atoms import initialize_atoms

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

    def _denoise_by_sde(species, pos, cell, cut, score_fn, ts, diffuser, save_traj):
        ts = ts.to(pos.device).view(-1, 1)
        f, g, g2 = diffuser.f, diffuser.g, diffuser.g2
        traj = [pos.detach().cpu().clone()] if save_traj else None
        pos = pos.detach()

        for i, t in enumerate(ts[1:]):
            dt = ts[i + 1] - ts[i]
            eps = dt.abs().sqrt() * torch.randn_like(pos)
            score = score_fn(species, pos, cell, t, cut)
            disp = (f(t) * pos - g2(t) * score) * dt + g(t) * eps
            pos = (pos + disp).detach()
            if save_traj:
                traj.append(pos.cpu().clone())

        return traj if save_traj else pos

    for score_model in score_models:
        main_id = f"{system}-{score_model}"
        click.echo(f"\n=== Score model: {main_id} ===")

        resolved_data_path = score_data_path.format(
            score_model=score_model, system=system
        )
        resolved_model_path = score_model_path.format(
            score_model=score_model, system=system
        )

        click.echo(f"  Data path:  {resolved_data_path}")
        click.echo(f"  Model path: {resolved_model_path}")

        datamodule = StructureSpecDataModule(
            data_dir=resolved_data_path,
            cutoff=cutoff,
            train_prior=True,
            k=0.8,
            train_size=0.9,
            scale_y=1.0,
            dup=128,
            batch_size=32,
            num_workers=8,
        )
        datamodule.setup()
        diffuser = datamodule.train_set.diffuser

        ckpt_path = resolved_model_path
        if not os.path.exists(ckpt_path):
            raise click.ClickException(f"Checkpoint not found: {ckpt_path}")
        score_net = LitScoreNet.load_from_checkpoint(ckpt_path, map_location=device)
        score_net.eval()
        score_net.ema_model.to(device)
        score_net.ema_model.eval()

        ts_torch = torch.linspace(tmax, tmin, tstep, device=device)

        resolved_init_path = init_path.format(system=system, score_model=score_model)
        if os.path.isfile(resolved_init_path):
            xyz_files = [resolved_init_path]
        elif os.path.isdir(resolved_init_path):
            xyz_files = sorted(
                glob.glob(os.path.join(resolved_init_path, f"{system}_*.xyz"))
            )
        else:
            xyz_files = sorted(glob.glob(resolved_init_path))  # treat as glob pattern
        if not xyz_files:
            click.echo(
                f"  Warning: no .xyz files found for {resolved_init_path}, skipping."
            )
            continue

        for sample_tag in xyz_files:
            init_atoms = read(sample_tag, "-1")
            sub_id = os.path.basename(sample_tag).replace(".xyz", "")
            click.echo(sub_id)

            run_outdir = os.path.join(outdir, main_id, sub_id)
            os.makedirs(run_outdir, exist_ok=True)

            for i in range(n_runs):
                species, pos, cell = _to_device(copy.deepcopy(init_atoms))
                cell_np = cell.detach().cpu().numpy()

                def score_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser):
                    return _prior_score(sp, p, c, t, co, _sn, _df)

                with torch.no_grad():
                    result = _denoise_by_sde(
                        species,
                        pos,
                        cell,
                        cutoff,
                        score_fn,
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

                click.echo(f"  {sub_id} run #{i} done")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


@glass.command(
    "cond_denoise",
    help="""
Run conditional (guided) denoising with spectral/structural guidance.

Supported guidance types:
  pdf    -- pair distribution function (DifferentiableRDF, CPU)
  adf    -- angular distribution function (DifferentiableADF, CPU)
  xrd    -- X-ray diffraction I(q) (DifferentiableXRD, GPU)
  nd     -- neutron diffraction I(q) (DifferentiableND, GPU)
  exafs  -- EXAFS spectrum via LitSpecNet (GPU)
  xanes  -- XANES spectrum via LitSpecNet (GPU)

EXAMPLES:

  # PDF guidance (computational ref)
  glass cond_denoise \\
      --score-model "1.5_2.5_3.5" --system Si \\
      --score-data-path "./data/{system}_{score_model}" \\
      --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \\
      --init-path "/home/jwguo/03_denoiser/reference/start_{system}_216/{system}_2.0_*.xyz" \\
      --ref-path /home/jwguo/03_denoiser/reference/amorph_Si_216 \\
      --guidance-type pdf --rho 1000 --device cuda:0 --n-runs 10

  # XRD guidance
  glass cond_denoise \\
      --score-model "1.5_2.5_3.5" --system Si \\
      --score-data-path "./data/{system}_{score_model}" \\
      --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \\
      --init-path "/home/jwguo/03_denoiser/reference/start_{system}_216/{system}_2.0_*.xyz" \\
      --ref-path /home/jwguo/03_denoiser/reference/amorph_Si_216 \\
      --guidance-type xrd --element-names Si --rho 5 --device cuda:0

  # EXAFS/XANES guidance
  glass cond_denoise \\
      --score-model "1.5_2.5_3.5" --system Si \\
      --score-data-path "./data/{system}_{score_model}" \\
      --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \\
      --init-path "/home/jwguo/03_denoiser/reference/start_{system}_216/{system}_2.0_*.xyz" \\
      --ref-path /home/jwguo/03_denoiser/reference/amorph_Si_216 \\
      --guidance-type exafs --spec-model-path ./models/Si_exafs.ckpt --rho 1e8 --device cuda:0

  # PDF guidance (experimental data)
  glass cond_denoise \\
      --score-model "1.5_2.5_3.5" --system Si \\
      --score-data-path "./data/{system}_{score_model}" \\
      --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \\
      --init-path "/home/jwguo/03_denoiser/reference/start_{system}_216/{system}_2.0_*.xyz" \\
      --exp-data ./data/exp_gr_si.json \\
      --guidance-type pdf --rho 1000 --device cuda:0
""",
)
@click.option(
    "--score-model",
    "score_models",
    multiple=True,
    required=True,
    help="Score model tag(s). Can be repeated.",
)
@click.option(
    "--system", type=str, default="Si", show_default=True, help="System name."
)
@click.option(
    "--score-data-path",
    type=str,
    required=True,
    help="Root dir containing data/{system}_{model}/ subfolders.",
)
@click.option(
    "--score-model-path",
    type=str,
    required=True,
    help="Full path to .ckpt checkpoint file (supports {system}, {score_model} placeholders).",
)
@click.option(
    "--init-path",
    type=str,
    required=True,
    help="Initial structures: a .xyz file, glob pattern, or directory.",
)
@click.option(
    "--ref-path",
    type=str,
    default=None,
    help="Directory of reference .xyz files (computational guidance). Matched by filename.",
)
@click.option(
    "--exp-data",
    type=str,
    default=None,
    help="JSON file with experimental data (keys: x, y or r, g). Used as global guidance target.",
)
@click.option(
    "--guidance-type",
    type=click.Choice(["pdf", "adf", "xrd", "nd", "exafs", "xanes"]),
    default="pdf",
    show_default=True,
    help="Guidance type.",
)
@click.option(
    "--rho",
    "rho_list",
    multiple=True,
    type=float,
    default=[1000.0],
    show_default=True,
    help="Guidance strength(s). Can be repeated.",
)
@click.option(
    "--outdir",
    type=str,
    default="denoise_logs/guided",
    show_default=True,
    help="Output root directory.",
)
@click.option(
    "--device", type=str, default="cuda:0", show_default=True, help="Torch device."
)
@click.option(
    "--tmin",
    type=float,
    default=0.001,
    show_default=True,
    help="Start time for reverse SDE.",
)
@click.option(
    "--tmax",
    type=float,
    default=1.0,
    show_default=True,
    help="End time for reverse SDE.",
)
@click.option(
    "--tstep",
    type=int,
    default=256,
    show_default=True,
    help="Number of SDE time steps.",
)
@click.option(
    "--cutoff",
    type=float,
    default=5.0,
    show_default=True,
    help="Graph cutoff radius (Å).",
)
@click.option(
    "--n-runs",
    type=int,
    default=10,
    show_default=True,
    help="Number of independent runs per structure.",
)
@click.option(
    "--save-traj/--no-save-traj",
    default=False,
    show_default=True,
    help="Save full trajectory or only final frame.",
)
# --- pdf options ---
@click.option(
    "--bin-size",
    type=int,
    default=100,
    show_default=True,
    help="[pdf] Number of RDF bins.",
)
# --- adf options ---
@click.option(
    "--angle-bins",
    type=int,
    default=100,
    show_default=True,
    help="[adf] Number of angle bins.",
)
@click.option(
    "--adf-sigma",
    type=float,
    default=0.1,
    show_default=True,
    help="[adf] Gaussian kernel width for ADF.",
)
@click.option(
    "--adf-cutoff",
    type=float,
    default=3.5,
    show_default=True,
    help="[adf] Cutoff radius for ADF triplet search (Å).",
)
# --- xrd/nd options ---
@click.option(
    "--element-names",
    "element_names",
    multiple=True,
    default=(),
    help="[xrd/nd] Element name(s) in order (e.g. --element-names Si). Can be repeated.",
)
@click.option(
    "--qmin",
    type=float,
    default=1.0,
    show_default=True,
    help="[xrd/nd] Minimum q value (Å⁻¹).",
)
@click.option(
    "--qmax",
    type=float,
    default=20.0,
    show_default=True,
    help="[xrd/nd] Maximum q value (Å⁻¹).",
)
@click.option(
    "--qstep",
    type=float,
    default=0.1,
    show_default=True,
    help="[xrd/nd] Q step size (Å⁻¹).",
)
@click.option(
    "--biso",
    type=float,
    default=1.5,
    show_default=True,
    help="[xrd/nd] Debye-Waller B factor.",
)
# --- exafs/xanes options ---
@click.option(
    "--spec-model-path",
    type=str,
    default=None,
    help="[exafs/xanes] Path to LitSpecNet checkpoint.",
)
def cond_denoise(
    score_models,
    system,
    score_data_path,
    score_model_path,
    init_path,
    ref_path,
    exp_data,
    guidance_type,
    rho_list,
    outdir,
    device,
    tmin,
    tmax,
    tstep,
    cutoff,
    n_runs,
    save_traj,
    bin_size,
    angle_bins,
    adf_sigma,
    adf_cutoff,
    element_names,
    qmin,
    qmax,
    qstep,
    biso,
    spec_model_path,
):
    """GPU conditional denoising with spectral/structural guidance."""
    import copy
    import math
    import numpy as np
    import ase
    from ase.io import read
    from torch import nn
    from graphite.nn import periodic_radius_graph
    from glass.lit.functions.get_atoms import initialize_atoms
    from glass.lit.modules import (
        DifferentiableRDF,
        DifferentiableADF,
        DifferentiableXRD,
        DifferentiableND,
        LitSpecNet,
    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")
    if torch.cuda.is_available():
        click.echo(f"GPU: {torch.cuda.get_device_name(device.index or 0)}")

    # --- validate inputs ---
    if not ref_path and not exp_data:
        raise click.ClickException(
            "Provide either --ref-path (computational) or --exp-data (experimental)."
        )
    if ref_path and exp_data:
        raise click.ClickException("--ref-path and --exp-data are mutually exclusive.")
    if guidance_type in ("xrd", "nd") and not element_names:
        raise click.ClickException(
            f"--element-names required for guidance-type '{guidance_type}'."
        )
    if guidance_type in ("exafs", "xanes") and not spec_model_path:
        raise click.ClickException(
            f"--spec-model-path required for guidance-type '{guidance_type}'."
        )

    # --- build guidance model ---
    if guidance_type == "pdf":
        guidance_model = DifferentiableRDF(cutoff=cutoff, bin_size=bin_size, sigma=0.15)
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
        guidance_model = DifferentiableXRD(
            q_vals=[qmin, qmax, qstep],
            element_names=list(element_names),
            biso=biso,
        )
        guidance_model.to(device).eval()
    elif guidance_type == "nd":
        guidance_model = DifferentiableND(
            q_vals=[qmin, qmax, qstep],
            element_names=list(element_names),
            biso=biso,
        )
        guidance_model.to(device).eval()
    elif guidance_type in ("exafs", "xanes"):
        spec_net_g = LitSpecNet.load_from_checkpoint(
            spec_model_path, map_location=device
        )
        spec_net_g.eval()
        spec_net_g.ema_model.to(device).eval()
        guidance_model = spec_net_g.ema_model
    click.echo(f"Guidance type: {guidance_type}")

    # --- load experimental target (global, interpolated once) ---
    exp_target_y = None
    if exp_data:
        import json as _json

        with open(exp_data) as _f:
            _d = _json.load(_f)
        x_exp = np.array(_d.get("x", _d.get("r")))
        y_exp = np.array(_d.get("y", _d.get("g")))

        if guidance_type == "pdf":
            bin_edges = np.linspace(0, cutoff, bin_size + 1)
            x_grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        elif guidance_type == "adf":
            x_grid = np.linspace(0, math.pi, angle_bins)
        elif guidance_type in ("xrd", "nd"):
            x_grid = np.arange(qmin, qmax, qstep)
        else:
            raise click.ClickException(
                f"--exp-data not yet supported for guidance-type '{guidance_type}'."
            )

        y_interp = np.interp(x_grid, x_exp, y_exp)
        exp_target_y = torch.from_numpy(y_interp).float().unsqueeze(0).to(device)
        click.echo(f"Loaded experimental target from {exp_data} (n_pts={len(x_grid)})")

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
            l_score, norm = likelihood_fn(species, pos, cell, t, cut)
            click.echo(
                f"    p={p_score.norm().item():.3f}  "
                f"l={l_score.norm().item():.3f}  "
                f"tgt={norm.sum().item():.4f}",
                err=True,
            )
            disp = (f(t) * pos - g2(t) * (p_score + l_score)) * dt + g(t) * eps
            pos = (pos + disp).detach()
            if save_traj:
                traj.append(pos.cpu().clone())

        return traj if save_traj else pos

    # --- collect init xyz files ---
    resolved_init_path = init_path.format(system=system)
    if os.path.isfile(resolved_init_path):
        xyz_files = [resolved_init_path]
    elif os.path.isdir(resolved_init_path):
        xyz_files = sorted(
            glob.glob(os.path.join(resolved_init_path, f"{system}_*.xyz"))
        )
    else:
        xyz_files = sorted(glob.glob(resolved_init_path))
    if not xyz_files:
        raise click.ClickException(f"No .xyz files found for {resolved_init_path}")

    for score_model in score_models:
        main_id = f"{system}-{score_model}"
        click.echo(f"\n=== Score model: {main_id} ===")

        resolved_data_path = score_data_path.format(
            score_model=score_model, system=system
        )
        resolved_model_path = score_model_path.format(
            score_model=score_model, system=system
        )

        datamodule = StructureSpecDataModule(
            data_dir=resolved_data_path,
            cutoff=cutoff,
            train_prior=True,
            k=0.8,
            train_size=0.9,
            scale_y=1.0,
            dup=128,
            batch_size=32,
            num_workers=8,
        )
        datamodule.setup()
        diffuser = datamodule.train_set.diffuser

        if not os.path.exists(resolved_model_path):
            raise click.ClickException(f"Checkpoint not found: {resolved_model_path}")
        score_net = LitScoreNet.load_from_checkpoint(
            resolved_model_path, map_location=device
        )
        score_net.eval()
        score_net.ema_model.to(device)
        score_net.ema_model.eval()

        ts_torch = torch.linspace(tmax, tmin, tstep, device=device)

        if ref_path:
            resolved_ref_path = ref_path.format(system=system)
            if os.path.isfile(resolved_ref_path):
                resolved_ref_path = os.path.dirname(resolved_ref_path)
            if not os.path.isdir(resolved_ref_path):
                raise click.ClickException(
                    f"--ref-path is not a valid directory: {resolved_ref_path}"
                )
            click.echo(f"  Ref dir: {resolved_ref_path}")

        for sample_tag in xyz_files:
            sub_id = os.path.basename(sample_tag).replace(".xyz", "")
            init_atoms = read(sample_tag, "-1")

            if exp_data:
                target_y = exp_target_y
            else:
                ref_file = os.path.join(resolved_ref_path, f"{sub_id}.xyz")
                if not os.path.exists(ref_file):
                    click.echo(f"  Warning: {ref_file} not found, skipping.")
                    continue
                ref_atoms = read(ref_file, "-1")
                if not (np.all(init_atoms.pbc) and np.all(ref_atoms.pbc)):
                    raise click.ClickException(
                        f"PBC must be True for both init and ref atoms ({sub_id})."
                    )
                if not np.allclose(
                    init_atoms.get_cell(), ref_atoms.get_cell(), atol=1e-5
                ):
                    raise click.ClickException(
                        f"Init and ref cells must match ({sub_id})."
                    )
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

            click.echo(f"  {sub_id}")

            for rho in rho_list:
                rho_str = f"{int(rho)}" if rho == int(rho) else f"{rho}"
                run_tag = f"{guidance_type}_rho{rho_str}_tmax{tmax}_nsteps{tstep}"
                run_outdir = os.path.join(outdir, main_id, sub_id, run_tag)
                os.makedirs(run_outdir, exist_ok=True)

                likelihood_fn = LikelihoodScore(
                    score_net.ema_model,
                    guidance_model,
                    target_y,
                    rho,
                    diffuser,
                    guidance_type,
                    cutoff,
                )

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

                    click.echo(f"    {sub_id} [{run_tag}] run #{i} done")

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


@glass.command(
    "write_spec_feature",
    help="""
Compute and write structural/spectral features for denoised or reference structures.

Features computed per structure: PDF, ADF, XRD, ND, EXAFS, XANES.
Each structure's features are saved as a single JSON file in OUTDIR.

MODES:

  denoise   -- reads *_final.xyz from denoise_logs/{denoise_tag}/{system}-*/init_*/
  reference -- reads {system}_*.xyz from a user-specified --atoms-path

EXAMPLES:

  # Denoise mode
  glass write_spec_feature --mode denoise --system Si \\
      --denoise-tag "unconditional/Si-1.5_2.5_3.5" \\
      --exafs-model ./models/Si_exafs.ckpt \\
      --xanes-model ./models/Si_xanes.ckpt

  # Reference mode
  glass write_spec_feature --mode reference --system Si \\
      --atoms-path /home/jwguo/03_denoiser/reference/amorph_Si_216 \\
      --exafs-model ./models/Si_exafs.ckpt \\
      --xanes-model ./models/Si_xanes.ckpt \\
      --outdir results/reference
""",
)
@click.option(
    "--mode",
    type=click.Choice(["denoise", "reference"]),
    default="denoise",
    show_default=True,
    help="Input source mode.",
)
@click.option(
    "--system", type=str, default="Si", show_default=True, help="System name."
)
@click.option(
    "--denoise-tag",
    type=str,
    default="*",
    show_default=True,
    help="Glob tag under denoise_logs/ for denoise mode.",
)
@click.option(
    "--denoise-root",
    type=str,
    default="denoise_logs",
    show_default=True,
    help="Root directory for denoised outputs.",
)
@click.option(
    "--atoms-path",
    type=str,
    default=None,
    help="Directory with {system}_*.xyz files (reference mode only).",
)
@click.option(
    "--outdir",
    type=str,
    default="results",
    show_default=True,
    help="Output directory for the combined JSON file.",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output JSON filename. Default: {mode}_{system}_spectra.json.",
)
@click.option(
    "--exafs-model",
    type=str,
    required=True,
    help="Path to EXAFS LitSpecNet checkpoint.",
)
@click.option(
    "--xanes-model",
    type=str,
    required=True,
    help="Path to XANES LitSpecNet checkpoint.",
)
@click.option(
    "--qmin",
    type=float,
    default=1.0,
    show_default=True,
    help="Minimum q value for XRD/ND.",
)
@click.option(
    "--qmax",
    type=float,
    default=20.0,
    show_default=True,
    help="Maximum q value for XRD/ND.",
)
@click.option(
    "--qstep",
    type=float,
    default=0.1,
    show_default=True,
    help="Q step size for XRD/ND.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Device for spectral model inference.",
)
def write_spec_feature(
    mode,
    system,
    denoise_tag,
    denoise_root,
    atoms_path,
    outdir,
    output,
    exafs_model,
    xanes_model,
    qmin,
    qmax,
    qstep,
    device,
):
    """Compute PDF, ADF, XRD, ND, EXAFS, XANES and write to JSON."""
    import json
    import numpy as np
    from ase.io import read
    from ase.data import chemical_symbols
    from collections import defaultdict
    from glass.lit.modules import DifferentiableRDF, DifferentiableADF, LitSpecNet
    from glass.lit.functions.get_atoms import initialize_atoms
    from graphite.nn import periodic_radius_graph
    from debyecalculator import DebyeCalculator

    q_vals = [qmin, qmax, qstep]

    def _compute_iq(pos, species, Z_list):
        import warnings

        pos_np = pos.detach().cpu().numpy()
        species_indices = species.argmax(dim=1).detach().cpu().numpy()
        elements = np.array(Z_list)[species_indices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            calc = DebyeCalculator(
                qmin=q_vals[0],
                qmax=q_vals[1],
                qstep=q_vals[2],
                qdamp=0.04,
                rmin=0,
                rmax=20,
                rstep=0.01,
                rthres=0.0,
                biso=1.5,
                device=device,
                rad_type="xray",
            )
            q, iq_xrd = calc.iq((elements, pos_np))
            calc.update_parameters(rad_type="neutron")
            _, iq_nd = calc.iq((elements, pos_np))
        return q, iq_xrd, iq_nd

    def _get_spec(spec_net, pos, species, cell, atomic_numbers, cutoff=5):
        edge_index, edge_vec = periodic_radius_graph(pos, cutoff, cell)
        edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
        ys = spec_net.ema_model(species, edge_index, edge_attr)
        species_indices = torch.argmax(species, dim=1).tolist()
        element_indices = defaultdict(list)
        for i, idx in enumerate(species_indices):
            element_indices[atomic_numbers.get(idx, f"elem_{idx}")].append(i)
        return {
            elem: ys[torch.tensor(idxs, device=ys.device)].mean(dim=0).tolist()
            for elem, idxs in element_indices.items()
        }

    def _write_into_dict(spec_type, x_values, y_types, y_values, atomic_numbers):
        spec_dict = {"bins": x_values.tolist()}
        if spec_type == "PDF":
            for idx, (a, b) in enumerate(y_types):
                spec_dict[f"{atomic_numbers[a]}-{atomic_numbers[b]}"] = y_values[
                    idx
                ].tolist()
        elif spec_type == "ADF":
            for idx, (a, b, c) in enumerate(y_types):
                spec_dict[
                    f"{atomic_numbers[a]}-{atomic_numbers[b]}-{atomic_numbers[c]}"
                ] = y_values[idx].tolist()
        return spec_dict

    def _process(atoms_file):
        atoms = read(atoms_file, "-1")
        Z_list, species, pos, cell = initialize_atoms(atoms)
        atomic_numbers = {i: chemical_symbols[Z] for i, Z in enumerate(Z_list)}

        rdf_model = DifferentiableRDF(cutoff=8.0, bin_size=100, sigma=0.15)
        rdf_bins, rdf_hist, pair_types = rdf_model(pos, species, cell)

        adf_model = DifferentiableADF(
            cutoff=3.0,
            angle_bins=100,
            angle_range=[0, np.pi],
            sigma=0.1,
            normalize=False,
        )
        adf_bins, adf_hist, triplet_types = adf_model(pos, species, cell)

        q, iq_xrd, iq_nd = _compute_iq(pos, species, Z_list)

        exafs_net = LitSpecNet.load_from_checkpoint(exafs_model)
        exafs_net.ema_model.to(device)
        exafs = _get_spec(exafs_net, pos, species, cell, atomic_numbers)

        xanes_net = LitSpecNet.load_from_checkpoint(xanes_model)
        xanes_net.ema_model.to(device)
        xanes = _get_spec(xanes_net, pos, species, cell, atomic_numbers)

        return {
            "PDF": _write_into_dict(
                "PDF", rdf_bins, pair_types, rdf_hist, atomic_numbers
            ),
            "ADF": _write_into_dict(
                "ADF", adf_bins, triplet_types, adf_hist, atomic_numbers
            ),
            "XRD": {"q": q.tolist(), "xrd": iq_xrd.tolist()},
            "ND": {"q": q.tolist(), "nd": iq_nd.tolist()},
            "EXAFS": exafs,
            "XANES": xanes,
        }

    # --- collect xyz files ---
    if mode == "denoise":
        search = os.path.join(denoise_root, denoise_tag, f"{system}_*", "*_final.xyz")
        click.echo(f"Searching: {search}")
        xyz_files = sorted(glob.glob(search))
    else:
        if not atoms_path:
            raise click.ClickException("--atoms-path is required for reference mode.")
        search = os.path.join(atoms_path, f"{system}_*.xyz")
        click.echo(f"Searching: {search}")
        xyz_files = sorted(glob.glob(search))

    if not xyz_files:
        raise click.ClickException(f"No .xyz files found under: {search}")

    click.echo(f"Found {len(xyz_files)} structure(s).")
    os.makedirs(outdir, exist_ok=True)

    if output:
        out_filename = output
    elif mode == "denoise":
        tag = denoise_tag.replace("/", "_")
        out_filename = f"denoise_{system}_{tag}_spectra.json"
    else:
        out_filename = f"reference_{system}_spectra.json"
    out_path = os.path.join(outdir, out_filename)

    combined = {}
    for xyz_file in xyz_files:
        if mode == "denoise":
            # e.g. denoise_logs/unconditional/Si-1.5_2.5_3.5/Si_2.0_01/00_final.xyz
            parts = xyz_file.replace("_final.xyz", "").split(os.sep)
            run_id = parts[-1]  # 00
            struct_id = parts[-2]  # Si_2.0_01
            model_id = parts[-3]  # Si-1.5_2.5_3.5
            key = f"{model_id}/{struct_id}/{run_id}"
        else:
            key = os.path.basename(xyz_file).replace(".xyz", "")

        combined[key] = _process(xyz_file)
        click.echo(f"  {key} done")

    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    click.echo(f"Saved: {out_path}")


@glass.command(
    "calc_metrics",
    help="""
Compute error and diversity metrics comparing denoised structures to reference.

Reads feature JSON files produced by `write_spec_feature`, compares denoised spectra
against a reference master JSON, and saves results to a single JSON per denoise folder.

Output structure per group (denoise_label x ref_label):
  error  -- mean normalized error vs reference
  score  -- error - ref_div (lower is better)

EXAMPLES:

  glass calc_metrics \\
      --ref-master-json final_data_dir/a-Si_ref_stats.json \\
      --system Si \\
      --denoise-folder unconditional \\
      --denoise-label 1.5_2.5_3.5 \\
      --ref-label 1.5 --ref-label 2.0 --ref-label 2.5 --ref-label 3.0 --ref-label 3.5 \\
      --outdir final_data_dir

  # Multiple denoise densities
  glass calc_metrics \\
      --ref-master-json final_data_dir/a-Si_ref_stats.json \\
      --system Si \\
      --denoise-folder unconditional \\
      --denoise-label 1.5 --denoise-label 2.5 --denoise-label 3.5 \\
      --ref-label 1.5 --ref-label 2.0 --ref-label 2.5 \\
      --outdir final_data_dir
""",
)
@click.option(
    "--denoise-json",
    type=str,
    required=True,
    help="Combined denoise spectra JSON from write_spec_feature.",
)
@click.option(
    "--ref-master-json",
    type=str,
    required=True,
    help="Reference master stats JSON from build_ref_stats.",
)
@click.option(
    "--system", type=str, default="Si", show_default=True, help="System name."
)
@click.option(
    "--denoise-label",
    "denoise_label_list",
    multiple=True,
    required=True,
    help="Label(s) identifying the denoised model/condition. Can be repeated.",
)
@click.option(
    "--ref-label",
    "ref_label_list",
    multiple=True,
    required=True,
    help="Label(s) identifying the reference condition. Can be repeated.",
)
@click.option(
    "--outdir",
    type=str,
    default="final_data_dir",
    show_default=True,
    help="Output directory for metrics JSON.",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output JSON filename. Default: a-{system}_metrics.json.",
)
@click.option(
    "--spectrum-types",
    "spectrum_types",
    multiple=True,
    default=("XRD", "ND", "XANES", "EXAFS", "PDF", "ADF"),
    show_default=True,
    help="Spectrum types to evaluate. Can be repeated.",
)
@click.option(
    "--exafs-slice",
    type=(int, int),
    default=(40, 280),
    show_default=True,
    help="EXAFS index slice: START END.",
)
@click.option(
    "--xrd-nd-npts",
    type=int,
    default=100,
    show_default=True,
    help="Number of points to use for XRD/ND spectra.",
)
def calc_metrics(
    denoise_json,
    ref_master_json,
    system,
    denoise_label_list,
    ref_label_list,
    outdir,
    output,
    spectrum_types,
    exafs_slice,
    xrd_nd_npts,
):
    """Compute error and diversity metrics for denoised vs reference structures."""
    import json
    import numpy as np

    def _is_axis_key(spec_type, key, idx):
        if spec_type in ["PDF", "ADF"] and key == "bins":
            return True
        if spec_type in ["XRD", "ND"] and key == "q":
            return True
        if spec_type not in ["EXAFS", "XANES"] and idx == 0:
            return True
        return False

    def _trim(spec_type, values):
        arr = np.asarray(values, dtype=float)
        if spec_type == "EXAFS":
            arr = arr[exafs_slice[0] : exafs_slice[1]]
        if spec_type in ["XRD", "ND"]:
            arr = arr[:xrd_nd_npts]
        return arr.tolist()

    def _extract_spectra(entry, spec_type):
        """Extract trimmed spectra vectors from a single JSON entry."""
        spectra_by_key = {}
        if spec_type not in entry:
            return spectra_by_key
        for idx, (key, values) in enumerate(entry[spec_type].items()):
            if _is_axis_key(spec_type, key, idx):
                continue
            if spec_type in ["EXAFS", "XRD", "ND"]:
                values = _trim(spec_type, values)
            spectra_by_key[key] = values
        return spectra_by_key

    def _mean_normalized_error(pred, ref, norm_factor):
        pred, ref = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
        err = np.abs(pred - ref).mean()
        return float(err / norm_factor) if norm_factor > 0 else float(err)

    with open(denoise_json) as f:
        all_denoise = json.load(f)
    with open(ref_master_json) as f:
        ref_master = json.load(f)

    os.makedirs(outdir, exist_ok=True)
    out_filename = output or f"a-{system}_metrics.json"
    out_path = os.path.join(outdir, out_filename)

    out = {
        "meta": {
            "system": system,
            "denoise_json": denoise_json,
            "ref_master_json": ref_master_json,
            "denoise_label_list": list(denoise_label_list),
            "ref_label_list": list(ref_label_list),
            "spectrum_types": list(spectrum_types),
            "exafs_slice": list(exafs_slice),
            "xrd_nd_npts": xrd_nd_npts,
        },
        "groups": {},
    }

    for denoise_label in denoise_label_list:
        click.echo(f"\n=== denoise_label: {denoise_label} ===")

        # keys: "{system}-{denoise_label}/{struct_id}/{run_id}"
        model_prefix = f"{system}-{denoise_label}/"
        model_entries = {
            k: v for k, v in all_denoise.items() if k.startswith(model_prefix)
        }
        click.echo(f"  Found {len(model_entries)} runs for {model_prefix}")

        for ref_label in ref_label_list:
            group_key = f"denoise_label={denoise_label} ref_label={ref_label}"
            ref_block = ref_master["groups"][ref_label]["stats"]

            # filter entries matching this ref_label in the struct part
            # key format: {system}-{denoise_label}/{system}_{ref_label}_{idx}/{run_id}
            struct_prefix = f"{system}_{ref_label}_"
            matching = {
                k: v
                for k, v in model_entries.items()
                if os.path.basename(os.path.dirname(k)).startswith(struct_prefix)
            }

            click.echo(f"  ref_label={ref_label}: {len(matching)} run(s) found")

            # group by struct_id -> list of run entries
            structs = {}
            for k, v in matching.items():
                parts = k.split("/")
                struct_id, run_id = parts[-2], parts[-1]
                structs.setdefault(struct_id, {})[run_id] = v

            grp = {
                "samples": {
                    struct_id: {"runs": runs} for struct_id, runs in structs.items()
                },
                "stats": {},
            }

            # aggregate stats per spectrum type
            for spec_type in spectrum_types:
                mean_ref_all = ref_block[spec_type]["mean_ref_spec"]
                ref_div = ref_block[spec_type]["ref_div"]
                norm_factor = ref_block[spec_type]["norm_factor"]

                per_key_sample_means = {}
                n_samples_with_runs, n_runs_total = 0, 0

                for struct_id, runs in structs.items():
                    run_vecs_by_key = {}
                    for run_id, entry in runs.items():
                        spec = _extract_spectra(entry, spec_type)
                        for key, vec in spec.items():
                            run_vecs_by_key.setdefault(key, []).append(vec)
                    if not run_vecs_by_key:
                        continue
                    n_samples_with_runs += 1
                    n_runs_total += len(runs)
                    for key, vecs in run_vecs_by_key.items():
                        per_sample_mean = np.mean(np.asarray(vecs, dtype=float), axis=0)
                        per_key_sample_means.setdefault(key, []).append(per_sample_mean)

                error_values, score_values = {}, {}
                for key, vecs in per_key_sample_means.items():
                    if key not in mean_ref_all:
                        continue
                    mean_denoise = np.mean(np.asarray(vecs, dtype=float), axis=0)
                    err = _mean_normalized_error(
                        mean_denoise, mean_ref_all[key], norm_factor
                    )
                    div = float(ref_div.get(key, 0.0))
                    error_values[key] = float(err)
                    score_values[key] = float(err - div)

                grp["stats"][spec_type] = {
                    "norm_factor": float(norm_factor),
                    "ref_div": {k: float(v) for k, v in ref_div.items()},
                    "n_samples_with_runs": int(n_samples_with_runs),
                    "n_runs": int(n_runs_total),
                    "n_keys": int(len(per_key_sample_means)),
                    "error": error_values,
                    "score": score_values,
                }
                avg_error = (
                    float(np.mean(list(error_values.values())))
                    if error_values
                    else float("nan")
                )
                avg_div = (
                    float(np.mean(list(ref_div.values()))) if ref_div else float("nan")
                )
                avg_score = (
                    float(np.mean(list(score_values.values())))
                    if score_values
                    else float("nan")
                )
                click.echo(
                    f"    {spec_type}: n_keys={len(per_key_sample_means)} "
                    f"n_samples={n_samples_with_runs} n_runs={n_runs_total} "
                    f"norm={norm_factor:.4g} | "
                    f"avg_error={avg_error:.4f}  avg_div={avg_div:.4f}  avg_score={avg_score:.4f}"
                )
            out["groups"][group_key] = grp

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    click.echo(f"\nSaved: {out_path}")


@glass.command(
    "build_ref_stats",
    help="""
Build the reference master stats JSON from a reference spectra JSON.

Reads the single JSON produced by `write_spec_feature --mode reference`,
groups structures by a regex-extracted value (e.g. density, temperature),
and computes per-group norm_factor, ref_div, and mean_ref_spec for each
spectrum type. Output is used as input to `calc_metrics`.

EXAMPLES:

  glass build_ref_stats \\
      --input results/reference_Si_spectra.json \\
      --system Si \\
      --atoms-path /home/jwguo/03_denoiser/reference/amorph_Si_216 \\
      --outdir final_data_dir

  # Custom grouping (e.g. by temperature instead of density)
  glass build_ref_stats \\
      --input results/reference_Si_spectra.json \\
      --group-var temperature \\
      --group-regex "_T(\\\\d+)_" \\
      --outdir final_data_dir
""",
)
@click.option(
    "--input",
    "input_json",
    type=str,
    required=True,
    help="Reference spectra JSON from write_spec_feature --mode reference.",
)
@click.option(
    "--system", type=str, default="Si", show_default=True, help="System name."
)
@click.option(
    "--atoms-path",
    type=str,
    default=None,
    help="Directory with reference .xyz files for storing atoms info (optional).",
)
@click.option(
    "--outdir",
    type=str,
    default="final_data_dir",
    show_default=True,
    help="Output directory.",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output JSON filename. Default: a-{system}_ref_stats.json.",
)
@click.option(
    "--group-var",
    type=str,
    default="density",
    show_default=True,
    help="Name of the grouping variable (used in output metadata).",
)
@click.option(
    "--group-regex",
    type=str,
    default=r"_(\d+(?:\.\d+)?)_",
    show_default=True,
    help="Regex to extract group value from structure key.",
)
@click.option(
    "--spectrum-types",
    "spectrum_types",
    multiple=True,
    default=("XRD", "ND", "XANES", "EXAFS", "PDF", "ADF"),
    show_default=True,
    help="Spectrum types to process. Can be repeated.",
)
@click.option(
    "--exafs-slice",
    type=(int, int),
    default=(40, 280),
    show_default=True,
    help="EXAFS index slice: START END.",
)
@click.option(
    "--xrd-nd-npts",
    type=int,
    default=100,
    show_default=True,
    help="Number of points to use for XRD/ND.",
)
def build_ref_stats(
    input_json,
    system,
    atoms_path,
    outdir,
    output,
    group_var,
    group_regex,
    spectrum_types,
    exafs_slice,
    xrd_nd_npts,
):
    """Build reference master stats JSON from a reference spectra JSON."""
    import json
    import re
    import numpy as np
    from ase.io import read

    def _is_axis_key(spec_type, key, idx):
        if spec_type in ["PDF", "ADF"] and key == "bins":
            return True
        if spec_type in ["XRD", "ND"] and key == "q":
            return True
        if spec_type not in ["EXAFS", "XANES"] and idx == 0:
            return True
        return False

    def _trim(spec_type, values):
        arr = np.asarray(values, dtype=float)
        if spec_type == "EXAFS":
            arr = arr[exafs_slice[0] : exafs_slice[1]]
        if spec_type in ["XRD", "ND"]:
            arr = arr[:xrd_nd_npts]
        return arr.tolist()

    def _load_spectra(group_data, spec_type):
        spectra_by_key = {}
        for skey, sdata in group_data.items():
            if spec_type not in sdata:
                continue
            for idx, (key, values) in enumerate(sdata[spec_type].items()):
                if _is_axis_key(spec_type, key, idx):
                    continue
                if spec_type in ["EXAFS", "XRD", "ND"]:
                    values = _trim(spec_type, values)
                spectra_by_key.setdefault(key, []).append(values)
        return spectra_by_key

    def _norm_factor(ref_dict):
        all_spectra = [v for vlist in ref_dict.values() for v in vlist]
        if not all_spectra:
            return 0.0
        return float(np.max(np.abs(np.asarray(all_spectra, dtype=float))))

    def _diversity(spectra_list, norm):
        arr = np.asarray(spectra_list, dtype=float)
        std = float(arr.std(axis=0).mean())
        return std / norm if norm > 0 else std

    with open(input_json) as f:
        all_data = json.load(f)

    # group keys by regex-extracted value
    groups = {}
    for skey in all_data:
        m = re.search(group_regex, skey)
        if not m:
            click.echo(f"  [skip] could not parse {group_var} from: {skey}")
            continue
        gv = m.group(1)
        groups.setdefault(gv, []).append(skey)

    if not groups:
        raise click.ClickException(
            f"No groups found. Check --group-regex against your structure keys."
        )

    os.makedirs(outdir, exist_ok=True)
    out_filename = output or f"a-{system}_ref_stats.json"
    out_path = os.path.join(outdir, out_filename)

    out = {
        "meta": {
            "system": system,
            "group_var": group_var,
            "group_regex": group_regex,
            "input_json": input_json,
            "spectrum_types": list(spectrum_types),
            "exafs_slice": list(exafs_slice),
            "xrd_nd_npts": xrd_nd_npts,
        },
        "groups": {},
    }

    for gv in sorted(
        groups, key=lambda x: float(x) if re.fullmatch(r"\d+(\.\d+)?", x) else x
    ):
        keys = groups[gv]
        click.echo(f"\n[{group_var}={gv}] n_structures={len(keys)}")

        group_data = {k: all_data[k] for k in keys}

        grp = {
            "meta": {group_var: gv, "n_structures": len(keys)},
            "samples": {},
            "stats": {},
        }

        # ingest samples
        for skey in keys:
            entry = {
                "spectra": {
                    st: all_data[skey][st]
                    for st in spectrum_types
                    if st in all_data[skey]
                }
            }
            if atoms_path:
                xyz_file = os.path.join(atoms_path, f"{skey}.xyz")
                if os.path.exists(xyz_file):
                    atoms = read(xyz_file, "-1")
                    entry["atoms"] = {
                        "numbers": atoms.get_atomic_numbers().tolist(),
                        "positions": atoms.get_positions().tolist(),
                        "cell": atoms.get_cell().tolist(),
                        "pbc": atoms.get_pbc().tolist(),
                    }
            grp["samples"][skey] = entry

        # compute stats per spectrum type
        for spec_type in spectrum_types:
            ref_spec = _load_spectra(group_data, spec_type)
            norm = _norm_factor(ref_spec)
            ref_div = {k: float(_diversity(v, norm)) for k, v in ref_spec.items()}
            mean_ref = {
                k: np.mean(np.asarray(v, dtype=float), axis=0).tolist()
                for k, v in ref_spec.items()
                if v
            }
            grp["stats"][spec_type] = {
                "mean_ref_spec": mean_ref,
                "norm_factor": float(norm),
                "ref_div": ref_div,
                "n_keys": len(ref_spec),
            }
            avg_div = (
                float(np.mean(list(ref_div.values()))) if ref_div else float("nan")
            )
            click.echo(
                f"  {spec_type}: n_keys={len(ref_spec)} norm={norm:.4g} | "
                f"avg_div={avg_div:.4f}"
            )

        out["groups"][gv] = grp

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    click.echo(f"\nSaved: {out_path}")


# Register Tersoff potential subcommands (md, energy) under the `glass` group.
from glass.potentials.torch_tersoff.cli import md as _tersoff_md
from glass.potentials.torch_tersoff.cli import energy as _tersoff_energy

glass.add_command(_tersoff_md, name="md")
glass.add_command(_tersoff_energy, name="energy")


if __name__ == "__main__":
    glass()
