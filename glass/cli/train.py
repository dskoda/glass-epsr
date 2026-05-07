import glob
import os

import click
import lightning as L
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from glass.lit.datamodules import StructureSpecDataModule
from glass.lit.modules import LitScoreNet


@click.command(
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


@click.command(
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